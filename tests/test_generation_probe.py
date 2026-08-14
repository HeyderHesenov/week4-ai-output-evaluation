"""Generation probe — saxlanmış kontekstin təkrar oynadılması.

`rag.*` BURADA İMPORT EDİLMİR: `rag/store.py` modul səviyyəsində
`langchain_chroma`-nı çəkir, CI isə onu quraşdırmır (yalnız anthropic,
python-dotenv, PyYAML). SUT qurucuları `SutQurucular` ilə inyeksiya olunur,
ona görə bütün dəst açarsız, şəbəkəsiz və chroma-sız işləyir — repo-nun CI
şərhinin iddia etdiyi kimi, «lokalda keçir, CI-də keçmir» sinfi yaranmır.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from eval.observation import ChunkView


@dataclass
class _SahteChunk:
    """SUT-un `RetrievedChunk`-ı ilə EYNİ sahə dəsti."""

    label: int
    text: str
    source: str
    page: int
    chunk_index: int
    score: float
    lexical_score: float
    chunk_id: str


def _cv(**over) -> ChunkView:
    base = dict(
        label=1, source="atlas_api_senedi.md", page=0, chunk_index=3,
        score=0.61, lexical_score=0.28, chunk_id="abc123", text="Limit 200-dür.",
    )
    base.update(over)
    return ChunkView(**base)


def _qurucular(**over):
    from tools.generation_probe import SutQurucular

    base = dict(
        system_instruction="SİSTEM QAYDALARI",
        build_user_message=lambda soru, chunks, nonce: (
            f"<<<KONTEKST:{nonce}>>>\n"
            + "\n".join(c.text for c in chunks)
            + f"\n<<<SUAL>>>{soru}"
        ),
        chunk_cls=_SahteChunk,
        is_refusal=lambda t: t.strip().lower().startswith(
            "sənədlərdə bu suala cavab tapılmadı"
        ),
        make_nonce=lambda: "NONCE1",
    )
    base.update(over)
    return SutQurucular(**base)


# --- Task 1: qurucu inyeksiyası və prompt bərpası ----------------------------


def test_sahe_desti_uygun_gelmirse_OLCME_DAYANIR() -> None:
    """Sahə uyğunsuzluğu bərpanı SƏSSİZCƏ sadiqsiz edərdi.

    `RetrievedChunk` SUT-a aiddir; pin edilmiş commit dəyişib sahə əlavə
    olunsa, adapter köhnə dəstlə obyekt qurar, prompt fərqlənər və cədvəl
    «eyni kontekstdə ölçüldü» iddiasını yalandan daşıyar. Qapı testdə deyil,
    İCRA VAXTINDA olmalıdır — CI `rag.*`-ı import edə bilmir, ona görə orada
    həqiqi sinif heç vaxt yoxlanmazdı.
    """
    from tools.generation_probe import ProbeError, yoxla_saheler

    yoxla_saheler(_SahteChunk)  # uyğun gəlir → susur

    @dataclass
    class _Deyisib(_SahteChunk):
        yeni_sahe: str = ""

    with pytest.raises(ProbeError, match="sahə dəsti"):
        yoxla_saheler(_Deyisib)


def test_adapter_butun_saheleri_KOCURUR() -> None:
    from tools.generation_probe import sut_chunk

    chunk = sut_chunk(_cv(), q=_qurucular())

    assert chunk.label == 1
    assert chunk.text == "Limit 200-dür."
    assert chunk.source == "atlas_api_senedi.md"
    assert chunk.chunk_id == "abc123"
    assert chunk.score == 0.61
    assert chunk.lexical_score == 0.28


def test_mesajlar_INYEKSIYA_olunmus_qurucunu_cagirir() -> None:
    """Prompt probe-da yenidən yazılmır — qurucu kənardan gəlir."""
    from tools.generation_probe import mesajlar

    msgs = mesajlar(
        "Limit nə qədərdir?", [_cv()], suffix="", nonce="NONCE1", q=_qurucular()
    )

    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "<<<KONTEKST:NONCE1>>>" in msgs[1]["content"]
    assert "Limit nə qədərdir?" in msgs[1]["content"]
    assert "Limit 200-dür." in msgs[1]["content"]


def test_suffiks_system_mesajinin_SONUNA_elave_olunur() -> None:
    """Variant mexanizmi ilə eyni forma: qaydalar əvəzlənmir, əlavə olunur."""
    from tools.generation_probe import mesajlar

    q = _qurucular()
    baseline = mesajlar("S?", [_cv()], suffix="", nonce="N", q=q)[0]["content"]
    genis = mesajlar("S?", [_cv()], suffix="15. Əlavə qayda.", nonce="N", q=q)[0]["content"]

    assert genis.startswith(baseline)
    assert genis.endswith("15. Əlavə qayda.")


# --- Task 2: determinist təsnifat --------------------------------------------
#
# Korpus faktları (`atlas_infra_qeydleri.md`, `atlas_api_senedi.md`):
#   `atlas-cache-edge` YALNIZ Enterprise planında ayrılır, limiti 128 GB;
#   korpusda bir neçə fərqli «limit» var — sürət, səhifələmə, yaddaş, gündəlik.


def test_imtina_INYEKSIYA_olunmus_askarlayici_ile_taninir() -> None:
    """SUT-un öz `_is_refusal`-ı işlədilir — ikinci imtina tərifi yaradılmır."""
    from tools.generation_probe import tesnif

    assert tesnif(
        "dev_ambiguous_limit", "Sənədlərdə bu suala cavab tapılmadı.", q=_qurucular()
    ) == "imtina"


def test_muqeddime_QEBUL_ve_DUZELDILDI_ayird_olunur() -> None:
    """`dev_false_premise_free_cache`: doğru davranış Enterprise-ı ADLANDIRMAQ.

    Möhürlənmiş case-in mənfi yoxlaması (`not_contains: 128`) bunu ayırd edə
    bilmir — imtina da onu keçir. Probe-un meyarı buna görə daha dardır.
    """
    from tools.generation_probe import tesnif

    q = _qurucular()
    cid = "dev_false_premise_free_cache"
    assert tesnif(cid, "Sənədlərdə bu suala cavab tapılmadı.", q=q) == "imtina"
    assert tesnif(cid, "Free planı üçün yaddaş limiti 128 GB-dır [1].", q=q) == "muqeddime_qebul"
    assert tesnif(
        cid,
        "atlas-cache-edge yalnız Enterprise planında ayrılır, Free planına aid deyil [1].",
        q=q,
    ) == "muqeddime_duzeldildi"


def test_TEK_oxunus_ve_ADLANDIRILMIS_oxunuslar_ayird_olunur() -> None:
    """`dev_ambiguous_limit`: baseline «Limit 200-dür [1]» — tək oxunuş."""
    from tools.generation_probe import tesnif

    q = _qurucular()
    cid = "dev_ambiguous_limit"
    assert tesnif(cid, "Limit 200-dür [1].", q=q) == "tek_oxunus"
    assert tesnif(
        cid, "Korpusda bir neçə limit var: sürət limiti [1] və səhifələmə limiti [2].", q=q
    ) == "oxunuslar_adlandi"
    assert tesnif(cid, "Hansı limiti nəzərdə tutursunuz?", q=q) == "oxunuslar_adlandi"


def test_out_of_corpus_ucun_IMTINA_yaxsi_davranisdir() -> None:
    from tools.generation_probe import tesnif

    assert tesnif(
        "dev_out_of_corpus_graphql", "Sənədlərdə bu suala cavab tapılmadı.", q=_qurucular()
    ) == "imtina"


def test_probe_case_leri_MODELE_CATAN_uclukdur() -> None:
    """`dev_out_of_corpus_ceo` qəsdən yoxdur: qapı onu modelə çatmamış kəsir.

    Saxlanmış artefaktda `reason=low_relevance` və LLM çağırışı SIFIRDIR, yəni
    heç bir prompt dəyişikliyi onu sındıra bilməz. Cədvələ salmaq ölçülməmiş
    şeyi ölçülmüş kimi göstərərdi.
    """
    from tools.generation_probe import PROBE_CASES

    assert [c.case_id for c in PROBE_CASES] == [
        "dev_false_premise_free_cache",
        "dev_ambiguous_limit",
        "dev_out_of_corpus_graphql",
    ]
    assert [c.nezaret_davranisi for c in PROBE_CASES] == ["imtina", "tek_oxunus", "imtina"]


# --- Task 3: qollar, ölçmə dövrü, nəzarət qapısı -----------------------------


class _SahteLlm:
    """`invoke(messages)` — SUT-un `ChatOpenAI`-dən işlətdiyi yeganə metod.

    `model_name` `ChatOpenAI`-də var və probe modelin adını oradan oxuyur —
    saxta obyekt də onu daşımalıdır, yoxsa model qapısı test edilə bilməz.
    """

    def __init__(self, cavablar: dict[str, str], *, model: str = "gpt-4o-mini") -> None:
        self.cavablar = cavablar
        self.model_name = model
        self.cagirislar: list[list[dict]] = []

    def invoke(self, messages):
        self.cagirislar.append(messages)
        system = messages[0]["content"]
        for acar, cavab in self.cavablar.items():
            if acar and acar in system:
                return type("R", (), {"content": cavab})()
        return type("R", (), {"content": self.cavablar[""]})()


def _obs(case_id: str, question: str) -> dict:
    return {
        "case_id": case_id, "question": question,
        "chunks": [_cv()], "system_sha256": "", "model": "",
    }


def test_her_qol_her_case_ucun_UC_defe_isledilir() -> None:
    from tools.generation_probe import QOLLAR, olc

    llm = _SahteLlm({"": "Sənədlərdə bu suala cavab tapılmadı."})
    setirler = olc(
        {"dev_out_of_corpus_graphql": _obs("dev_out_of_corpus_graphql", "GraphQL?")},
        llm=llm, q=_qurucular(), tekrar=3,
    )

    assert len(setirler) == len(QOLLAR) * 3
    assert len(llm.cagirislar) == len(QOLLAR) * 3
    assert sorted({s["tekrar"] for s in setirler}) == [1, 2, 3]
    assert all(s["sinif"] == "imtina" for s in setirler)


def test_qollar_system_mesajini_FERQLENDIRIR() -> None:
    """Dörd qol dörd fərqli system prompt-u deməkdir — yoxsa müqayisə boşdur."""
    from tools.generation_probe import QOLLAR, olc

    llm = _SahteLlm({"": "Sənədlərdə bu suala cavab tapılmadı."})
    setirler = olc(
        {"dev_out_of_corpus_graphql": _obs("dev_out_of_corpus_graphql", "GraphQL?")},
        llm=llm, q=_qurucular(), tekrar=1,
    )

    assert len({s["system_sha256"] for s in setirler}) == len(QOLLAR)


def test_nezaret_qolu_orijinal_davranisi_TEKRAR_ISTEHSAL_etmese_qapi_baglanir() -> None:
    """Sadiq olmayan təkrar oynatmada bütün cədvəl mənasızdır."""
    from tools.generation_probe import NEZARET_QOLU, nezaret_sadiqdir

    yaxsi = [{"case_id": "dev_ambiguous_limit", "qol": NEZARET_QOLU,
              "tekrar": t, "sinif": "tek_oxunus"} for t in (1, 2, 3)]
    assert nezaret_sadiqdir(yaxsi) == (True, [])

    pis = [{**r, "sinif": "oxunuslar_adlandi"} for r in yaxsi]
    ok, sebebler = nezaret_sadiqdir(pis)
    assert ok is False
    assert "dev_ambiguous_limit" in sebebler[0]


def test_nezaret_COXLUQ_uzre_qiymetlendirilir() -> None:
    """Model determinist deyil — 3 təkrarın 2-si uyğun gəlirsə sadiqdir."""
    from tools.generation_probe import NEZARET_QOLU, nezaret_sadiqdir

    setirler = [
        {"case_id": "dev_ambiguous_limit", "qol": NEZARET_QOLU, "tekrar": 1, "sinif": "tek_oxunus"},
        {"case_id": "dev_ambiguous_limit", "qol": NEZARET_QOLU, "tekrar": 2, "sinif": "tek_oxunus"},
        {"case_id": "dev_ambiguous_limit", "qol": NEZARET_QOLU, "tekrar": 3, "sinif": "imtina"},
    ]
    assert nezaret_sadiqdir(setirler)[0] is True


# --- Task 4: CLI, həqiqi SUT və artefakt -------------------------------------


def test_probe_hash_i_KARKASIN_yazdigi_hash_ile_EYNI_formadadir() -> None:
    """İki hash forması qapını YALANDAN bağlayırdı.

    Probe `hexdigest()[:16]` yazırdı, karkas isə tam 64 simvol
    (`eval/instrument.py`). `system_prompt_deyismeyib` onları birbaşa
    tutuşdurduğu üçün prompt heç dəyişməsə belə uyğunsuzluq görünərdi:
    alət 5 kodu qaytarar, PULLU run boşa gedər və cədvəl «etibarsız»
    möhürü ilə yazılardı. Ona görə probe karkasın ÖZ funksiyasını çağırır —
    ikinci tərif qalmır.
    """
    from eval.instrument import sha256_text
    from tools.generation_probe import NEZARET_QOLU, olc

    q = _qurucular()
    setirler = olc(
        {"dev_out_of_corpus_graphql": _obs("dev_out_of_corpus_graphql", "GraphQL?")},
        llm=_SahteLlm({"": "Sənədlərdə bu suala cavab tapılmadı."}), q=q, tekrar=1,
    )

    nezaret = next(s for s in setirler if s["qol"] == NEZARET_QOLU)
    assert nezaret["system_sha256"] == sha256_text(q.system_instruction)
    assert len(nezaret["system_sha256"]) == 64


def test_SYSTEM_PROMPT_run_dan_beri_deyisibse_qapi_baglanir() -> None:
    """Qol 0 SUT-un prompt-unu OLDUĞU KİMİ işlədir — hash uyğun gəlməlidir.

    Uyğun gəlmirsə, SUT-un system prompt-u saxlanmış run-dan bəri dəyişib
    (pin yenilənib və ya submodule sürüşüb). Bu, davranış yoxlamasından
    AYRIDIR və ondan güclüdür: davranış təsadüfən üst-üstə düşə bilər,
    hash düşməz.
    """
    from tools.generation_probe import NEZARET_QOLU, system_prompt_deyismeyib

    setirler = [{"case_id": "c1", "qol": NEZARET_QOLU, "system_sha256": "aaaa"}]
    assert system_prompt_deyismeyib(setirler, {"c1": {"system_sha256": "aaaa"}}) == (True, [])

    ok, sebebler = system_prompt_deyismeyib(setirler, {"c1": {"system_sha256": "bbbb"}})
    assert ok is False
    assert "aaaa" in sebebler[0] and "bbbb" in sebebler[0]


def test_system_prompt_yoxlamasi_yalniz_NEZARET_qoluna_aiddir() -> None:
    """Qol 1-3 prompt-u QƏSDƏN dəyişir — orada uyğunsuzluq gözləniləndir."""
    from tools.generation_probe import system_prompt_deyismeyib

    setirler = [{"case_id": "c1", "qol": "1-müqəddimə", "system_sha256": "ferqli"}]
    assert system_prompt_deyismeyib(setirler, {"c1": {"system_sha256": "aaaa"}}) == (True, [])


def test_MODEL_run_dakindan_ferqlidirse_qapi_baglanir() -> None:
    """Probe LLM-i env-dəki `LLM_MODEL`-dən qurulur — o dəyişə bilər.

    Başqa modellə ölçmək prompt qollarını müqayisə etməyi dayandırır: fərq
    qoldan da, modeldən də gələ bilər və ayırd edilə bilməz. Davranış qapısı
    bunu tutmaya bilər, çünki başqa model də «tek_oxunus» qaytara bilər.
    """
    from tools.generation_probe import NEZARET_QOLU, model_deyismeyib

    setirler = [{"case_id": "c1", "qol": NEZARET_QOLU, "model": "gpt-4o-mini"}]
    assert model_deyismeyib(setirler, {"c1": {"model": "gpt-4o-mini"}}) == (True, [])

    ok, sebebler = model_deyismeyib(setirler, {"c1": {"model": "gpt-4o"}})
    assert ok is False
    assert "gpt-4o-mini" in sebebler[0] and "gpt-4o" in sebebler[0]


def test_model_yoxlamasi_BOS_deyeri_ATLAYIR() -> None:
    """Çağırışı olmayan case-in modeli bilinmir — uydurulmuş ad yazılmır."""
    from tools.generation_probe import NEZARET_QOLU, model_deyismeyib

    setirler = [{"case_id": "c1", "qol": NEZARET_QOLU, "model": "gpt-4o-mini"}]
    assert model_deyismeyib(setirler, {"c1": {"model": ""}}) == (True, [])


def test_repo_dan_KENAR_artefakt_yolu_aleti_cokdurmur(tmp_path) -> None:
    """`relative_to` kənar yolda `ValueError` atırdı.

    Ölçmə BİTDİKDƏN sonra, artefakt artıq diskdə olduğu halda alət çökürdü və
    `except BaseException` onu «uğursuz» kimi möhürləyirdi — yəni uğurlu
    ölçmə yalandan uğursuz görünürdü. Üç probe alətinin hər üçü eyni sətri
    işlədirdi.
    """
    from probe_common import PROJECT_ROOT, artefakt_yolu

    assert artefakt_yolu(PROJECT_ROOT / "logs" / "probes" / "p1") == "logs/probes/p1"
    assert artefakt_yolu(tmp_path / "p1") == str(tmp_path / "p1")


def test_nezaret_sadiq_deyilse_alet_BES_kodu_qaytarir(monkeypatch, tmp_path, capsys) -> None:
    """Etibarsız cədvəl SƏSSİZ yazılmır — çıxış kodu onu deyir."""
    import tools.generation_probe as mod

    monkeypatch.setattr(mod, "musahideleri_oxu", lambda run_id: {
        "dev_ambiguous_limit": _obs("dev_ambiguous_limit", "Limit nə qədərdir?"),
    })
    monkeypatch.setattr(mod, "sut_qurucular", lambda settings: _qurucular())
    # Nəzarət qolunda GÖZLƏNİLMƏYƏN davranış: «tek_oxunus» əvəzinə imtina.
    monkeypatch.setattr(mod, "llm_qur", lambda settings: _SahteLlm(
        {"": "Sənədlərdə bu suala cavab tapılmadı."}
    ))

    kod = mod.main(["--run", "r1", "--probes-dir", str(tmp_path)])

    assert kod == 5
    assert "nəzarət" in capsys.readouterr().err.lower()


def test_ugurlu_probe_ARTEFAKT_yazir(monkeypatch, tmp_path) -> None:
    import json

    import tools.generation_probe as mod

    monkeypatch.setattr(mod, "musahideleri_oxu", lambda run_id: {
        "dev_ambiguous_limit": _obs("dev_ambiguous_limit", "Limit nə qədərdir?"),
    })
    monkeypatch.setattr(mod, "sut_qurucular", lambda settings: _qurucular())
    monkeypatch.setattr(mod, "llm_qur", lambda settings: _SahteLlm({"": "Limit 200-dür [1]."}))

    assert mod.main(["--run", "r1", "--probes-dir", str(tmp_path)]) == 0

    qovluq = next(p for p in tmp_path.iterdir() if p.is_dir())
    manifest = json.loads((qovluq / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "tamam"
    assert manifest["nezaret_sadiqdir"] is True
    # Hansı modelin ölçüldüyü artefaktdan oxunmalıdır — sonradan soruşmaq üçün
    # başqa mənbə yoxdur.
    assert manifest["model"] == "gpt-4o-mini"
    assert (qovluq / "summary.md").exists()
    assert (qovluq / "rows.jsonl").read_text(encoding="utf-8").count("\n") == 12
