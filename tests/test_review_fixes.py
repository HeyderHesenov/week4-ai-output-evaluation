"""Code review tapıntılarının regressiya testləri.

Hər test MƏHZ bir tapıntıya bağlıdır və düzəlişdən ƏVVƏL uğursuz olurdu.
Adları uzundur, çünki testin adı tapıntının ifadəsidir: baq geri qayıtsa,
uğursuz testin adı nəyin sındığını izahsız deyir.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.artifacts import RunPaths, load_run
from eval.config import Settings
from eval.dataset import NumericClaim, load_dataset
from eval.errors import ConfigError, DatasetError
from eval.graders import extract_numbers, numeric_claim_satisfied
from eval.metrics import UNMEASURABLE
from eval.report import _outcomes
from eval.rootcause import RootCause
from eval.split import LedgerEntry, append_holdout_ledger, read_holdout_ledger
from eval.variants import load_variants
from tests.conftest import PROJECT_ROOT


# --- graders: ədəd ayrıştırma ------------------------------------------------


def test_SIYAHIDAKI_ededlerin_hamisi_tapilir() -> None:
    """«60, 600 və 6 000» — acgöz nişan üçünü də bir yerdə udub itirirdi."""
    values = [v for v, _, _ in extract_numbers("Limitlər: 60, 600 və 6 000 sorğu")]
    assert values == [60.0, 600.0, 6000.0]


def test_vergulle_ayrilmis_qisa_siyahi_ITMIR() -> None:
    assert [v for v, _, _ in extract_numbers("Zonalar: 1, 2, 3")] == [1.0, 2.0, 3.0]


def test_qrup_ve_onluq_ayiricilari_HELE_DE_isleyir() -> None:
    """Düzəliş köhnə davranışı pozmamalıdır."""
    assert [v for v, _, _ in extract_numbers("6 000 sorğu")] == [6000.0]
    assert [v for v, _, _ in extract_numbers("1.234,56 manat")] == [1234.56]


def test_SAAT_formati_ayri_eded_kimi_cixarilmir() -> None:
    """«01:00» vahidsiz `value=1`/`value=0` iddialarını yanlış təsdiqləyirdi."""
    assert extract_numbers("saat 01:00-da alınır") == ()
    assert not numeric_claim_satisfied(
        "Ehtiyat nüsxə saat 01:00-da alınır", NumericClaim(value=1.0, unit="")
    )


# --- graders: imtina səbəbi --------------------------------------------------


def _refusal_case(reason_in: tuple[str, ...] = ()):
    from eval.dataset import EvalCase, Expected

    return EvalCase(
        id="dev_ref",
        question="Gəlir nə qədərdir?",
        category="out_of_corpus",
        split="dev",
        gradable="deterministic",
        expected=Expected(kind="refusal", reason_in=reason_in),
    )


def _refused_observation(reason: str):
    from tests.conftest import make_observation

    obs = make_observation("dev_ref", answer_text="Sənədlərdə cavab tapılmadı.")
    return type(obs)(**{**vars(obs), "refused": True, "reason": reason})


def test_BOS_INDEKS_imtinasi_defolt_olaraq_KECMIR() -> None:
    """`reason_in` verilməyibsə, empty_index qəbul olunanlar arasında OLMAMALIDIR.

    Boş indeks səbəbindən imtina sistemin işlədiyini deyil, QURULMADIĞINI
    göstərir — `check_refusal`-ın öz docstring-i məhz bunu deyir, amma
    fallback bütün səbəbləri qəbul edib qurulma nasazlığını keçirirdi.
    """
    from eval.graders import check_refusal

    outcome = check_refusal(_refusal_case(), _refused_observation("empty_index"))
    assert outcome is not None and not outcome.ok


def test_ADI_imtina_sebebi_defolt_olaraq_KECIR() -> None:
    """Düzəliş qanuni imtinaları bloklamamalıdır."""
    from eval.graders import check_refusal

    outcome = check_refusal(_refusal_case(), _refused_observation("low_relevance"))
    assert outcome is not None and outcome.ok


# --- report: McNemar ölçülə bilməyənləri geriləmə saymır ----------------------


def _run_with_causes(categories: list[str]):
    class _FakeRun:
        causes = tuple(
            RootCause(case_id=f"c{i}", repeat=1, category=cat, detail="")
            for i, cat in enumerate(categories)
        )

    return _FakeRun()


def test_HAKIM_XETASI_McNemar_da_gerileme_sayilmir() -> None:
    """Bir 429 baş sətirdə «geriləmə» kimi görünürdü."""
    base = _outcomes(_run_with_causes(["ok", "ok"]))
    variant = _outcomes(_run_with_causes(["ok", "judge_error"]))

    from eval.metrics import mcnemar, stable_pass

    result = mcnemar(stable_pass(base), stable_pass(variant))
    assert result.regressed == 0
    assert "judge_error" in UNMEASURABLE


def test_olculebilen_HEQIQI_gerileme_HELE_DE_gorunur() -> None:
    base = _outcomes(_run_with_causes(["ok", "ok"]))
    variant = _outcomes(_run_with_causes(["ok", "generation"]))

    from eval.metrics import mcnemar, stable_pass

    assert mcnemar(stable_pass(base), stable_pass(variant)).regressed == 1


# --- variants: baseline əvəz edilə bilməz -------------------------------------


def test_BASELINE_id_si_fayl_ile_EVEZ_EDILE_BILMEZ(tmp_path) -> None:
    (tmp_path / "zz.yaml").write_text(
        "id: baseline\nlabel: saxta\nsystem_suffix: 'ƏLAVƏ QAYDA'\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="baseline"):
        load_variants(tmp_path)


# --- artifacts: qiymətlər təkrar üzrə ayrılır ---------------------------------


def test_qiymetler_TEKRAR_uzre_ayrilir() -> None:
    run = load_run(
        RunPaths.for_run(PROJECT_ROOT / "runs", "20260810T083111Z-holdout-v1_tam_cavab")
    )
    assert len(run.grade_map()) == len(run.grades), (
        "grade_map case_id üzrə yığsaydı, 21 qiymət 7-yə düşərdi və "
        "reclassify sonuncu təkrarın qiymətini hamısına şamil edərdi"
    )
    assert {g.repeat for g in run.grades} == {1, 2, 3}


# --- dataset: mühafizələr -----------------------------------------------------


@pytest.mark.parametrize("body", ["cases:\n", "cases: 5\n", "cases: 'x'\n"])
def test_bos_ve_ya_skalyar_cases_DATASETERROR_verir(tmp_path, body) -> None:
    """Xam TypeError `cli.main` tərəfindən tutulmur — istifadəçi traceback görür."""
    path = tmp_path / "t.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(DatasetError):
        load_dataset(path)


def test_rubrikadaki_SERBEST_reqem_yalanci_sizma_saymir(tmp_path) -> None:
    """«0-3 şkalası» ifadəsi `value: 3` olan case-i bloklamamalıdır."""
    path = tmp_path / "t.yaml"
    path.write_text(
        "cases:\n"
        "  - id: dev_x\n"
        "    question: 'Neçə zona?'\n"
        "    category: normal\n"
        "    split: dev\n"
        "    gradable: both\n"
        "    gold_sources: [atlas_infra_qeydleri.md]\n"
        "    expected:\n"
        "      kind: answer\n"
        "      numeric:\n"
        "        - value: 3\n"
        "          unit: 'zona'\n"
        "      rubric: 'Bal 0-3 şkalasındadır; cavab zona sayını deməlidir.'\n",
        encoding="utf-8",
    )
    # Bölgü örtüyü yoxlaması ayrı xəta verə bilər; bizi maraqlandıran
    # YALNIZ sızma mühafizəsinin işə düşməməsidir.
    try:
        load_dataset(path)
    except DatasetError as exc:
        assert "gözlənilən ədədi" not in str(exc), exc


def test_rubrika_HEQIQI_ededi_ehtiva_edirse_HELE_DE_bloklanir(tmp_path) -> None:
    path = tmp_path / "t.yaml"
    path.write_text(
        "cases:\n"
        "  - id: dev_x\n"
        "    question: 'Neçə zona?'\n"
        "    category: normal\n"
        "    split: dev\n"
        "    gradable: both\n"
        "    gold_sources: [atlas_infra_qeydleri.md]\n"
        "    expected:\n"
        "      kind: answer\n"
        "      numeric:\n"
        "        - value: 3\n"
        "          unit: 'zona'\n"
        "      rubric: 'Cavab 3 zona deməlidir.'\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="gözlənilən ədədi"):
        load_dataset(path)


# --- config: fallback modeli qabaqcadan yoxlanır ------------------------------


def test_QIYMETSIZ_fallback_modeli_run_dan_EVVEL_xeta_verir(monkeypatch) -> None:
    """Əks halda xəta yalnız hesabatda — pul xərcləndikdən sonra — çıxırdı."""
    monkeypatch.setenv("JUDGE_FALLBACK_MODEL", "model-yoxdur-9")
    monkeypatch.setenv("SUT_COMMIT", "0" * 40)
    with pytest.raises(ConfigError, match="qiymət cədvəlində yoxdur"):
        Settings.load()


# --- config: manifest maşından asılı deyil ------------------------------------


def test_manifest_yolu_REPOYA_NISBI_ve_hash_masindan_asili_deyil(monkeypatch) -> None:
    """Mütləq yol həm istifadəçi adını sızdırırdı, həm hash-i maşına bağlayırdı."""
    monkeypatch.setenv("SUT_COMMIT", "0" * 40)
    public = Settings.load().public_dict()

    assert public["sut_path"] == "vendor/week2-rag-document-qa"
    assert "/Users" not in json.dumps(public), "manifestə lokal yol düşməməlidir"


# --- split: registr atomik yazılır --------------------------------------------


def test_registr_ATOMIK_yazilir_ve_muveqqeti_fayl_qalmir(tmp_path) -> None:
    path = tmp_path / "holdout_ledger.json"
    entry = LedgerEntry(
        run_id="r1", at="2026-08-10T00:00:00Z", variant_id="baseline",
        harness_commit="abc", case_count=8, note="",
    )
    append_holdout_ledger(path, entry)
    append_holdout_ledger(path, LedgerEntry(**{**vars(entry), "run_id": "r2"}))

    assert len(read_holdout_ledger(path)) == 2
    assert not (tmp_path / "holdout_ledger.json.tmp").exists()
    assert json.loads(path.read_text(encoding="utf-8"))[0]["run_id"] == "r1"


# =============================================================================
# 2026-08-12 `/code-review xhigh 2d9308f` icmalı
# =============================================================================
#
# Aşağıdakı testlərin hamısı `2d9308f` üzərində UĞURSUZ olurdu. Bölmə tarixli
# saxlanılır, çünki tapıntının hansı icmaldan gəldiyi onun kontekstidir.


# --- report: indeks/retrieval kimliyi ----------------------------------------


def _manifest(**sut_retrieval) -> dict:
    """Yalnız `sut_retrieval` ilə fərqlənən manifest — qalan hər şey eynidir.

    `sha256` QƏSDƏN eynidir: tapıntı elə budur ki, uyğun barmaq izi qapı
    parametrlərindəki fərqi görünməz edirdi.
    """
    return {
        "config_hash": "a",
        "config": {},
        "sut_chunk_count": 29,
        "sut_index": {"sha256": "aaaa"},
        "sut_retrieval": {
            "top_k": 4,
            "relevance_threshold": 0.42,
            "lexical_threshold": 0.35,
            "chunk_size": 500,
            "embedding_model": "text-embedding-3-small",
            **sut_retrieval,
        },
    }


def test_eyni_barmaq_izi_QAPI_ferqini_gizletmir() -> None:
    """Barmaq izi chunk ID-lərindən törəyir, qapı parametrləri isə ID dəyişmir.

    `lexical_threshold` 0.35 → 0.17 heç bir chunk ID-ni dəyişmir, yəni iki run
    eyni barmaq izi verir. Əvvəlki `_index_diff_lines` məhz burada `return []`
    edirdi və addım 3-ə (`sut_retrieval` müqayisəsi) heç vaxt çatmırdı —
    funksiyanın mövcud olma səbəbi olan hal xəbərdarlıqsız keçirdi.
    """
    from eval.report import _config_diff_lines

    lines = _config_diff_lines(_manifest(), _manifest(lexical_threshold=0.17))

    assert len(lines) == 1, lines
    assert "lexical_threshold" in lines[0]
    assert "0.35" in lines[0] and "0.17" in lines[0]


def test_eyni_barmaq_izi_EMBEDDING_modeli_ferqini_gizletmir() -> None:
    """Barmaq izi vektorları GÖRMÜR — yalnız chunk mətnlərini.

    Sübut bu commit-in öz artefaktındadır (`logs/probes/20260812T191757Z-…`):
    baseline və `text-embedding-3-large` sətirləri eyni `sha=53974e67bd222968`
    daşıyır, halbuki indekslər tamamilə fərqli vektorlardan ibarətdir.
    `embedding_model` artefaktda ÜMUMİYYƏTLƏ yazılmırdı, ona görə iki run
    «eyni ölçülmüş sistem» kimi görünürdü.
    """
    from eval.report import _config_diff_lines

    lines = _config_diff_lines(
        _manifest(), _manifest(embedding_model="text-embedding-3-large")
    )

    assert len(lines) == 1, lines
    assert "embedding_model" in lines[0]
    assert "text-embedding-3-large" in lines[0]


def test_effective_retrieval_EMBEDDING_modelini_qeyd_edir() -> None:
    """Yuxarıdakı hesabat sətri yalnız sahə artefakta DÜŞÜRSƏ mümkündür."""
    from eval.sut import _effective_retrieval

    class SahteSettings:
        top_k = 4
        embedding_model = "text-embedding-3-large"

    class SahtePipeline:
        settings = SahteSettings()

    assert _effective_retrieval(SahtePipeline())["embedding_model"] == (
        "text-embedding-3-large"
    )


# --- artifacts: atılan chunk-ların balları -----------------------------------


def test_retrieval_ballari_GERI_OXUNUR(tmp_path) -> None:
    """`scores` yazılırdı, amma dekoder onu bərpa etmirdi — səssiz `()`.

    Nəticə: README-nin «astananı nə qədər endirmək lazımdır?» sualına
    saxlanmış artefaktdan cavab vermək iddiası karkasın ÖZ oxuyucusundan
    keçmirdi; dəyərlərə yalnız `observations.jsonl`-ı xam grep etməklə
    çatmaq olurdu. Bu, məhz artefakt oxumağı əvəz etmək üçün yazılmış
    sahənin faydasını sıfırlayırdı.
    """
    from eval.artifacts import RunWriter
    from eval.observation import RetrievalCall
    from tests.conftest import make_observation

    ballar = (0.77, 0.51, 0.335, 0.12)
    obs = make_observation(
        "c1",
        retrieval_calls=[
            RetrievalCall(
                mode="hybrid", k=4, latency_ms=120.0, returned=4,
                top_score=ballar[0], query_chars=30, scores=ballar,
            )
        ],
    )
    w = RunWriter(RunPaths.for_run(tmp_path, "run-001"))
    w.write_manifest({"run_id": "run-001"})
    w.append_observation(obs)

    oxunan = load_run(RunPaths.for_run(tmp_path, "run-001")).observations[0]
    assert oxunan.retrieval_calls[0].scores == ballar
    assert oxunan == obs, "gediş-gəliş tam olmalıdır, yalnız `scores` deyil"


# --- artifacts: yarımçıq kəsilmiş probe --------------------------------------
#
# Qovluq ölçmədən ƏVVƏL yaranırdı, manifest isə ən SONDA yazılırdı. Aradakı
# pəncərədə hər şey səhv gedirdi: Ctrl-C manifestsiz qovluq qoyurdu (bütün
# dəst hamı üçün qırmızı olurdu), və `ProbePaths.exists()` məhz həmin sonuncu
# fayla baxdığı üçün təkrar icra yarımçıq qovluğa DAVAM edirdi — bir
# artefaktda iki icranın sətirləri. Bu, elə `2d9308f`-in düzəltdiyi qüsurdur.


def _probe_kimliyi() -> dict:
    return {"probe_tool": "sweep", "argv": ["python", "x.py"], "started_at": "20260813T000000Z"}


def test_probe_manifesti_ILK_ANDA_yazilir(tmp_path) -> None:
    from eval.artifacts import ProbePaths, ProbeWriter

    paths = ProbePaths.for_probe(tmp_path, "20260813T000000Z-sweep-top_k")
    ProbeWriter(paths, kimlik=_probe_kimliyi())  # heç bir sətir yazılmadan

    manifest = json.loads(paths.file("manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "yarımçıq"
    assert manifest["argv"] == ["python", "x.py"], "kimlik ilk andan tam olmalıdır"


def test_movcud_probe_qovluguna_TEKRAR_yazilmir(tmp_path) -> None:
    """Guard manifestə yox, QOVLUĞA baxmalıdır.

    Manifest ən son yazılan fayl idi, ona görə yarımçıq qovluq «mövcud
    deyil» sayılırdı və `rows.jsonl`-a əlavə yazılırdı.
    """
    from eval.artifacts import ArtifactError, ProbePaths, ProbeWriter

    paths = ProbePaths.for_probe(tmp_path, "20260813T000000Z-sweep-top_k")
    paths.root.mkdir(parents=True)
    (paths.root / "rows.jsonl").write_text('{"əvvəlki": "icra"}\n', encoding="utf-8")

    with pytest.raises(ArtifactError, match="mövcuddur"):
        ProbeWriter(paths, kimlik=_probe_kimliyi())

    assert paths.file("rows.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_ugursuz_probe_manifestde_UGURSUZ_isarelenir(tmp_path) -> None:
    """Kəsilmiş ölçmə İZ QOYUR, amma dəsti qırmızı etmir.

    Yalan iki cürdür: yarımçıq nəticəni tam kimi göstərmək və yarımçıq
    qovluğu heç nə deməmək. Status sahəsi ikisini də aradan qaldırır.
    """
    from eval.artifacts import ProbePaths, ProbeWriter

    paths = ProbePaths.for_probe(tmp_path, "20260813T000000Z-eksperiment-chunking")
    writer = ProbeWriter(paths, kimlik=_probe_kimliyi())
    writer.mark_failed("ConfigError")

    manifest = json.loads(paths.file("manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "uğursuz"
    assert manifest["error"] == "ConfigError"
    assert manifest["argv"] == ["python", "x.py"], "kimlik itməməlidir"


def test_ugurlu_probe_manifesti_TAMAM_isarelenir(tmp_path) -> None:
    from eval.artifacts import ProbePaths, ProbeWriter

    paths = ProbePaths.for_probe(tmp_path, "20260813T000000Z-sweep-top_k")
    writer = ProbeWriter(paths, kimlik=_probe_kimliyi())
    writer.write_manifest({**_probe_kimliyi(), "namizəd_sayı": 3})

    manifest = json.loads(paths.file("manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "tamam"
    assert manifest["namizəd_sayı"] == 3


# --- probe_common: təkrar istehsal əmri işlək olmalıdır -----------------------


def test_summary_daki_emr_YAPISDIRILA_bilir() -> None:
    """`<müvəqqəti-qovluq>` shell-də yönləndirmədir, ad deyil.

    Artefaktın bəyan edilmiş məqsədi «bu cədvəl hansı əmrlə alınıb?» sualına
    cavab verməkdir; `<` və `>` sitatsız qalanda yapışdırılan sətir
    `--workdir`-i dəyərsiz qoyur və «no such file or directory» verir — yəni
    artefakt işləməyən əmr təqdim edir. Token DƏYİŞMİR (o, artefaktlarda
    artıq mövcuddur), yalnız sitatlanır.

    YOXLAMA ÜSULU: `shlex.split` DEFAULT halda `<` və `>`-ni adi simvol
    sayır, yəni baqı GÖRMÜR — bu test onunla yaşıl qalardı. `punctuation_chars`
    isə shell metasimvollarını modelləşdirir. Həqiqi `/bin/sh` ilə yoxlanılıb:
    sitatsız sətir `syntax error near unexpected token` verir.
    """
    import shlex

    from eval.artifacts import MUVEQQETI_YOL
    from tools.probe_common import summary_metni

    argv = ["python", "tools/retrieval_experiments.py", "--workdir", MUVEQQETI_YOL]
    metn = summary_metni(basliq="B", probe_id_="p", argv=argv, govde="g")
    emr = metn.split("```bash\n", 1)[1].split("\n```", 1)[0]

    lexer = shlex.shlex(emr, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    assert list(lexer) == argv, "shell əmri geri parse olunmalıdır"


# --- retrieval_experiments: matched büdcə invariantı --------------------------


def test_MATCHED_budce_baseline_i_asmir_cox_bolunen_sualda() -> None:
    """«cəm baseline-dan böyük ola bilmir» iddiası YALAN idi.

    `k = max(1, top_k // len(sorgular))`: `top_k=4` və 5 alt-sorğuda `k=1`
    verir, cəm isə 5 — yəni «matched» etiketli sətir `✗ BÜDCƏ ARTIQ` damğası
    alır və alət 3 kodu ilə çıxır. Cari dev korpusunda baş vermir (yalnız 2
    sual bölünür, hərəsi 3-ə), ona görə çap olunmuş heç bir rəqəm səhv deyil
    — amma rejimin adı doğru olmalıdır.
    """
    from tests.test_retrieval_experiments import _FakePipeline
    from tools.retrieval_experiments import evaluate, sub_queries

    sual = (
        "Kart limiti nə qədərdir və kvota həddi nə qədərdir və SLA müddəti nə "
        "qədərdir və cache ömrü nə qədərdir?"
    )
    assert len(sub_queries(sual)) == 5, "test məhz çox bölünən sual tələb edir"

    row = evaluate(_FakePipeline(), sual, set(), expand=True, budget_mode="matched")

    # 5 sorğu → 4-ə kəsilir (1 atılır) → hərəsinə k=1 → büdcə TAM 4,
    # yəni baseline ilə eyni: rejimin adı artıq davranışı düzgün deyir.
    assert row["retrieval_budget"] == 4, "matched rejim baseline büdcəsini aşmamalıdır"
    assert row["queries"] == 4
    assert row["dropped_subqueries"] == 1, "kəsmə SƏSSİZ olmamalıdır"


def test_MATCHED_rejimde_TAM_sual_her_zaman_qalir() -> None:
    """Kəsmə tam sualı ata bilməz — o, genişləndirmənin təhlükəsizlik payıdır."""
    from tools.retrieval_experiments import matched_sorgular

    sorgular = ["TAM SUAL", "a", "b", "c", "d", "e"]
    saxlanan, atilan = matched_sorgular(sorgular, top_k=2)

    assert saxlanan[0] == "TAM SUAL"
    assert len(saxlanan) == 2 and atilan == 4


# --- sut: indeks kimliyinin oxunuşu ------------------------------------------


class _StoreSuz:
    """`settings` var, `store` yoxdur — preflight-dən əvvəlki hal."""

    class _S:
        persist_dir = "/xeyali/yol/chroma_c500"

    settings = _S()


def test_store_oxunmasa_da_PERSIST_DIR_adi_qalir() -> None:
    """`persist_dir_name` store oxunuşundan asılı deyil — onunla birgə atılmamalıdır."""
    from eval.sut import _index_identity

    assert _index_identity(_StoreSuz())["persist_dir_name"] == "chroma_c500"


def test_store_oxunusundaki_GOZLENILMEYEN_xeta_udulmur() -> None:
    """Səssiz `{}` «artefakt köhnədir» kimi oxunur — halbuki sistem xarabdır.

    Bu, `tools/retrieval_experiments.py:ensure_index`-in eyni commit-də
    müdafiə etdiyi qaydadır: pin edilmiş commit dəyişəndə səhv XƏTA kimi
    görünməlidir, «səssiz sıfır» kimi yox. Sxem səhvləri (`AttributeError`
    və s.) udulur — onlar həqiqətən «sübut yoxdur» deməkdir; disk/chroma
    xətası isə udulmur.
    """
    from eval.sut import _index_identity

    class _Xarab(_StoreSuz):
        class _Store:
            @property
            def _store(self):
                raise OSError("disk oxunmur")

        store = _Store()

    with pytest.raises(OSError):
        _index_identity(_Xarab())

    class _SxemDeyisib(_StoreSuz):
        class _Store:
            pass  # `_store` yoxdur — SUT API-si dəyişib

        store = _Store()

    kimlik = _index_identity(_SxemDeyisib())
    assert "sha256" not in kimlik, "sübut yoxdursa, iddia da edilmir"
    assert kimlik["persist_dir_name"] == "chroma_c500"


# --- yüngül tapıntılar --------------------------------------------------------


def test_barmaq_izi_ALET_ve_KARKASDA_eyni_funksiyadir() -> None:
    """Birləşdirmə iddiasını şərh yox, kimlik qorumalıdır.

    Düstur iki yerdə KOPYALANMIŞDI: biri dəyişsə, «probe artefaktı ilə run
    manifesti birləşdirilə bilir» iddiası səssizcə yalan olardı.
    """
    from eval.sut import index_fingerprint as karkasda
    from tools.retrieval_experiments import index_fingerprint as aletde

    assert aletde is karkasda


def test_probe_adindaki_oxlar_EKSPERIMENT_destinden_toremelidir() -> None:
    """Sabit siyahı qovluq adına run-un etmədiyi ölçməni yazdıra bilirdi."""
    from tools.retrieval_experiments import EXPERIMENTS, Experiment, swept_axes

    assert swept_axes(EXPERIMENTS) == ["chunking", "embedding", "genişləndirmə"]
    assert swept_axes([Experiment("baseline")]) == []
    assert swept_axes([Experiment("yalnız chunking", chunk_size=500)]) == ["chunking"]


def test_config_hash_SEHV_deyerde_ConfigError_qaldirir() -> None:
    """Modulun bütün digər səhv-konfiqurasiya yolları `ConfigError` verir."""
    settings = Settings.load()
    pozuq = Settings(**{**vars(settings), "retrieval_threshold": float("nan")})

    with pytest.raises(ConfigError, match="NaN"):
        pozuq.config_hash


def test_probe_qovlugunun_adi_ile_started_at_EYNI_andir(monkeypatch, tmp_path) -> None:
    """İki ayrı `indi_utc()` çağırışı saniyə sərhədini kəsə bilirdi.

    Qovluq adı hər şeyin istinad etdiyi kimlikdir — sənəd ona görə sitat
    gətirir. Manifestdəki vaxtın ondan bir saniyə fərqlənməsi «hansı icra?»
    sualını sonradan çətinləşdirir.
    """
    import tools.probe_common as pc
    from tests.conftest import make_dataset

    damgalar = iter(["20260813T131459Z", "20260813T131500Z"])
    monkeypatch.setattr(pc, "indi_utc", lambda: next(damgalar))

    writer, kimlik = pc.probe_yarat(
        alet="sweep", oxlar=["top_k"], argv=["python", "x.py"],
        settings=Settings.load(), dataset=make_dataset([]), probes_dir=tmp_path,
    )

    assert writer.paths.probe_id.startswith(kimlik["started_at"])
