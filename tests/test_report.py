"""Hesabat — hər rəqəmin yanında onun etibarlılığı olmalıdır."""

from __future__ import annotations

import pytest

from eval.artifacts import RunArtifacts
from eval.config import load_prices
from eval.judge import Verdict
from eval.report import build_report
from eval.rootcause import RootCause
from eval.split import LedgerEntry
from tests.conftest import PROJECT_ROOT, make_case, make_dataset, make_llm_call, make_observation

GOLD = "atlas_api_senedi.md"


@pytest.fixture(scope="module")
def prices():
    return load_prices(PROJECT_ROOT / "data" / "prices.yaml")


@pytest.fixture
def dataset():
    return make_dataset(
        [
            make_case("dev_a", split="dev", category="normal", numeric=[{"value": 1}]),
            make_case("hold_a", split="holdout", category="normal", numeric=[{"value": 1}]),
        ]
    )


def manifest(**overrides):
    base = {
        "run_id": "run-x",
        "split": "dev",
        "variant_id": "baseline",
        "variant_label": "Baseline",
        "started_at": "2026-08-10T12:00:00Z",
        "finished_at": "2026-08-10T12:05:00Z",
        "case_count": 2,
        "repeats": 1,
        "dataset_sha256": "a" * 64,
        "split_manifest_sha256": "b" * 64,
        "variant_sha256": "c" * 64,
        "judge_prompt_sha256": "d" * 64,
        "harness_commit": "e" * 40,
        "sut_commit": "f" * 40,
        "config_hash": "0123456789abcdef",
        "config": {"judge_model": "claude-opus-5"},
        "max_holdout_similarity": 0.21,
    }
    base.update(overrides)
    return base


def run(causes, *, observations=(), verdicts=(), **manifest_overrides) -> RunArtifacts:
    return RunArtifacts(
        run_id="run-x",
        manifest=manifest(**manifest_overrides),
        observations=tuple(observations),
        grades=(),
        verdicts=tuple(verdicts),
        causes=tuple(causes),
    )


def cause(case_id: str, category: str, repeat: int = 1) -> RootCause:
    return RootCause(case_id=case_id, repeat=repeat, category=category, detail="detal")


def verdict(case_id: str, score: int | None, **kwargs) -> Verdict:
    return Verdict(
        case_id=case_id,
        repeat=1,
        score=score,
        faithful=True,
        complete=True,
        reason="",
        flags=(),
        judge_model="claude-opus-5",
        served_model="claude-opus-5",
        judge_prompt_sha256="d" * 64,
        threshold=2,
        **kwargs,
    )


# --- xülasə -----------------------------------------------------------------


def test_pass_rate_INTERVALI_ile_capa_gedir(dataset, prices) -> None:
    text = build_report(run([cause("dev_a", "ok"), cause("hold_a", "ok")]), dataset, prices)
    assert "95% CI" in text
    assert "Sərt nisbət" in text


def test_olculebilmeyen_netice_XEBERDARLIQ_verir(dataset, prices) -> None:
    text = build_report(run([cause("dev_a", "judge_error")]), dataset, prices)
    assert "ölçülə bilmədi" in text
    assert "məlumat DAŞIMIR" in text


def test_kok_sebeb_QAT_uzre_verilir(dataset, prices) -> None:
    text = build_report(
        run([cause("dev_a", "retrieval_miss"), cause("hold_a", "generation_wrong")]),
        dataset,
        prices,
    )
    assert "| retrieval | 1 |" in text
    assert "| generasiya | 1 |" in text


def test_ugursuz_caseler_SADALANIR(dataset, prices) -> None:
    text = build_report(run([cause("dev_a", "retrieval_miss")]), dataset, prices)
    assert "## Uğursuz case-lər" in text
    assert "`dev_a`" in text


# --- müqayisə ---------------------------------------------------------------


def test_ehemiyyetsiz_yaxsilasma_GIZLEDILMIR(dataset, prices) -> None:
    """2 case-lik yaxşılaşma «prompt işlədi» demək üçün kifayət etmir."""
    baseline = run([cause("dev_a", "retrieval_miss"), cause("hold_a", "ok")])
    variant = run([cause("dev_a", "ok"), cause("hold_a", "ok")], variant_id="v1")

    text = build_report(variant, dataset, prices, baseline=baseline)

    assert "McNemar" in text
    assert "əhəmiyyətli DEYİL" in text
    assert "dəlil kifayət etmir" in text


def test_gerileme_ACIQ_qeyd_olunur(dataset, prices) -> None:
    baseline = run([cause("dev_a", "ok"), cause("hold_a", "ok")])
    variant = run([cause("dev_a", "ok"), cause("hold_a", "generation_wrong")], variant_id="v1")

    text = build_report(variant, dataset, prices, baseline=baseline)
    assert "GERİLƏYİB" in text


# --- xərc -------------------------------------------------------------------


def test_tesdiqlenmemis_qiymet_BANNERI(dataset, prices) -> None:
    obs = make_observation("dev_a", llm_calls=[make_llm_call(model="gpt-4o-mini")])
    text = build_report(run([cause("dev_a", "ok")], observations=[obs]), dataset, prices)

    assert "Təsdiqlənməmiş qiymətlər" in text
    assert "gpt-4o-mini" in text


def test_embedding_boslugu_HESABATDA_gorunur(dataset, prices) -> None:
    text = build_report(run([cause("dev_a", "ok")]), dataset, prices)
    assert "Embedding çağırışları ölçülmür" in text


def test_token_sayi_olmayan_cagirislar_XEBERDARLIGI(dataset, prices) -> None:
    from dataclasses import replace

    call = replace(make_llm_call(), usage_source="missing")
    obs = make_observation("dev_a", llm_calls=[call])
    text = build_report(run([cause("dev_a", "ok")], observations=[obs]), dataset, prices)

    assert "usage_source=missing" in text


# --- gecikmə ----------------------------------------------------------------


def test_gecikme_uc_sutunu_ve_QALIGI_gosterir(dataset, prices) -> None:
    obs = make_observation("dev_a", total_ms=1000.0, llm_calls=[make_llm_call(latency_ms=700.0)])
    text = build_report(run([cause("dev_a", "ok")], observations=[obs]), dataset, prices)

    assert "Bağlanmayan qalıq" in text
    assert "overhead" in text


# --- hakim ------------------------------------------------------------------


def test_insan_etiketi_yoxdursa_ACIQ_deyilir(dataset, prices) -> None:
    text = build_report(
        run([cause("dev_a", "ok")], observations=[make_observation("dev_a")],
            verdicts=[verdict("dev_a", 3)]),
        dataset,
        prices,
    )
    assert "kappa hesablanmadı" in text


def test_asagi_kappa_XEBERDARLIQ_verir(dataset, prices) -> None:
    verdicts = [verdict("dev_a", 3), verdict("hold_a", 3)]
    observations = [make_observation("dev_a"), make_observation("hold_a")]
    text = build_report(
        run([cause("dev_a", "ok")], observations=observations, verdicts=verdicts),
        dataset,
        prices,
        human_labels={"dev_a": 0, "hold_a": 3},
    )
    assert "kappa < 0.60" in text


def test_hakim_xetalari_SADALANIR_ve_sifir_bal_deyil(dataset, prices) -> None:
    text = build_report(
        run([cause("dev_a", "judge_error")], observations=[make_observation("dev_a")],
            verdicts=[verdict("dev_a", None, error="hakim imtina etdi")]),
        dataset,
        prices,
    )
    assert "heç biri 0 bala çevrilməyib" in text
    assert "hakim imtina etdi" in text


# --- bütövlük ---------------------------------------------------------------


def test_olculmus_oxsarliq_RAQEM_kimi_capa_gedir(dataset, prices) -> None:
    text = build_report(run([cause("dev_a", "ok")]), dataset, prices)
    assert "**0.21**" in text
    assert "iddia deyil, rəqəmdir" in text


def test_holdout_registri_TAM_cap_olunur(dataset, prices) -> None:
    ledger = [
        LedgerEntry(run_id=f"r{i}", at=f"2026-08-1{i}", variant_id="baseline",
                    harness_commit="x", case_count=8, note=f"qeyd {i}")
        for i in range(3)
    ]
    text = build_report(run([cause("dev_a", "ok")]), dataset, prices, ledger=ledger)

    assert "Holdout **3** dəfə işlədilib" in text
    assert "qeyd 2" in text
    assert "ikidən çox dəfə işlədilib" in text


def test_holdout_islenmeyibse_deyilir(dataset, prices) -> None:
    text = build_report(run([cause("dev_a", "ok")]), dataset, prices)
    assert "Holdout hələ işlədilməyib" in text


# --- sabitlik ---------------------------------------------------------------


def test_tek_tekrarda_SURUSKENLIK_olculmediyi_deyilir(dataset, prices) -> None:
    text = build_report(run([cause("dev_a", "ok")]), dataset, prices)
    assert "REPEATS=1" in text
    assert "ölçülməyib" in text


def test_surusen_case_ler_XEBERDARLIQ_kimi_cixir(dataset, prices) -> None:
    causes = [cause("dev_a", "ok", 1), cause("dev_a", "generation_wrong", 2)]
    text = build_report(run(causes, repeats=2), dataset, prices)

    assert "Sürüşən case-lər" in text
    assert "dev_a" in text


# --- konfiqurasiya fərqi müqayisədə görünür ---------------------------------


def _cfg_diff(base_cfg, var_cfg, base_sut=None, var_sut=None):
    """`sut_retrieval` default olaraq EYNİ verilir ki, test yalnız `config`
    fərqini ölçsün — əks halda hər testə indeks xəbərdarlığı qarışardı."""
    from eval.report import _config_diff_lines

    same = {"top_k": 4, "chunk_size": 800}
    fp = {"sha256": "eyni-barmaq-izi"}
    return _config_diff_lines(
        {"config_hash": "aaa", "config": base_cfg, "sut_retrieval": base_sut or same,
         "sut_chunk_count": 19, "sut_index": fp},
        {"config_hash": "bbb", "config": var_cfg, "sut_retrieval": var_sut or same,
         "sut_chunk_count": 19, "sut_index": fp},
    )


def test_eyni_hash_ve_eyni_indeks_ise_XEBERDARLIQ_yoxdur() -> None:
    from eval.report import _config_diff_lines

    same = {
        "config_hash": "aaa",
        "config": {"repeats": 1},
        "sut_retrieval": {"top_k": 4, "chunk_size": 800},
        "sut_chunk_count": 19,
        "sut_index": {"sha256": "eyni"},
    }
    assert _config_diff_lines(same, same) == []


def test_CHUNKING_ferqi_config_hash_eyni_olsa_da_tutulur() -> None:
    """Bu testin səbəbi real bir yanlış hesabatdır (2026-08-12).

    `CHUNK_SIZE=500` SUT-a env ilə gedir, framework `config`-una düşmür,
    ona görə `config_hash` EYNİ qalır. Əvvəlki versiya bu halda «ölçülən
    sistem eynidir» yazırdı — halbuki indeks tamam başqa idi.
    """
    from eval.report import _config_diff_lines

    base = {
        "config_hash": "eyni",
        "config": {"repeats": 1},
        "sut_chunk_count": 19,
        "sut_retrieval": {"chunk_size": 800, "persist_dir_name": "chroma"},
    }
    var = {
        "config_hash": "eyni",
        "config": {"repeats": 1},
        "sut_chunk_count": 19,
        "sut_retrieval": {"chunk_size": 500, "persist_dir_name": "chroma_c500"},
    }
    lines = _config_diff_lines(base, var)
    assert len(lines) == 1
    assert "SUT retrieval konfiqurasiyası fərqlidir" in lines[0]
    assert "800 → 500" in lines[0]


def test_sut_retrieval_qeydi_YOXDURSA_tesdiqlene_bilmir_deyilir() -> None:
    """Köhnə artefaktla müqayisə «eynidir» kimi göstərilməməlidir."""
    from eval.report import _config_diff_lines

    base = {"config_hash": "aaa", "config": {}}  # heç bir şahid yoxdur
    var = {"config_hash": "aaa", "config": {}, "sut_retrieval": {"chunk_size": 500}}
    lines = _config_diff_lines(base, var)
    assert len(lines) == 1
    assert "TƏSDİQLƏNƏ BİLMİR" in lines[0]


def test_retrieval_ferqi_MUQAYISEDE_acik_gosterilir() -> None:
    """Bu, retrieval dövrünün əsas mühafizəsidir.

    TOP_K dəyişmiş run baseline-dan qanuni olaraq fərqlidir; oxucu fərqin
    variantdan yox, konfiqurasiyadan gəldiyini görməlidir.
    """
    lines = _cfg_diff({"repeats": 1}, {"repeats": 1, "retrieval_top_k": 8})
    assert len(lines) == 1
    assert "Konfiqurasiya fərqlidir" in lines[0]
    assert "retrieval_top_k" in lines[0]
    assert "→ 8" in lines[0]


def test_YALNIZ_temsili_ferq_olcme_ferqi_kimi_gosterilmir() -> None:
    """`sut_path` mütləq→nisbi keçidi ölçməni dəyişmir; yalan həyəcan olmasın."""
    lines = _cfg_diff({"sut_path": "/abs/vendor/sut"}, {"sut_path": "vendor/sut"})
    assert len(lines) == 1
    assert "təmsili" in lines[0]
    assert "Konfiqurasiya fərqlidir" not in lines[0]


def test_esasli_ferq_varsa_TEMSILI_qeyd_susdurulur() -> None:
    """Əsaslı fərq varsa, `config_hash`-in niyə fərqləndiyi onsuz da izah
    olunub — təmsili qeydi də çap etmək oxucunu yayındırır."""
    lines = _cfg_diff(
        {"sut_path": "/abs/vendor/sut", "repeats": 1},
        {"sut_path": "vendor/sut", "repeats": 3},
    )
    assert len(lines) == 1
    assert "Konfiqurasiya fərqlidir" in lines[0] and "repeats" in lines[0]
    assert "təmsili" not in lines[0]


def test_hash_ferqli_amma_config_boshdursa_KOHNE_SXEM_deyilir() -> None:
    lines = _cfg_diff({}, {})
    assert len(lines) == 1
    assert "köhnə sxemlə" in lines[0]


# --- indeks kimliyi müqayisədə (2026-08-12) ---------------------------------


def test_BARMAQ_IZI_ferqi_en_guclu_ifadedir() -> None:
    """Barmaq izi indeksin MƏZMUNUNDAN törəyir — chunk sayı üst-üstə düşsə də
    fərqli mətnləri tutur."""
    from eval.report import _config_diff_lines

    base = {"config_hash": "a", "config": {}, "sut_chunk_count": 29,
            "sut_index": {"sha256": "aaaa"}}
    var = {"config_hash": "a", "config": {}, "sut_chunk_count": 29,
           "sut_index": {"sha256": "bbbb"}}
    lines = _config_diff_lines(base, var)
    assert len(lines) == 1
    assert "İndeks FƏRQLİDİR" in lines[0]
    assert "aaaa" in lines[0] and "bbbb" in lines[0]


def test_sut_chunk_count_ferqi_TUTULUR_kohne_artefaktla_da() -> None:
    """Real cüt: 2026-08-10 baseline-ında 19, 2026-08-12-də 29 chunk.

    Köhnə manifestdə nə `sut_index`, nə `sut_retrieval` var — amma
    `sut_chunk_count` HƏR manifestdə mövcuddur, ona görə yeganə şahiddir.
    """
    from eval.report import _config_diff_lines

    base = {"config_hash": "a", "config": {}, "sut_chunk_count": 19}
    var = {"config_hash": "a", "config": {}, "sut_chunk_count": 29,
           "sut_index": {"sha256": "bbbb"}}
    lines = _config_diff_lines(base, var)
    assert len(lines) == 1
    assert "chunk sayı fərqlidir" in lines[0]
    assert "19 → 29" in lines[0]


def test_eyni_barmaq_izi_XEBERDARLIQ_vermir() -> None:
    from eval.report import _config_diff_lines

    same = {"config_hash": "a", "config": {}, "sut_chunk_count": 29,
            "sut_index": {"sha256": "aaaa"}}
    assert _config_diff_lines(same, same) == []


def test_hec_bir_sahid_yoxdursa_TESDIQLENE_BILMIR_deyilir() -> None:
    from eval.report import _config_diff_lines

    base = {"config_hash": "a", "config": {}}
    var = {"config_hash": "a", "config": {}}
    lines = _config_diff_lines(base, var)
    assert len(lines) == 1
    assert "TƏSDİQLƏNƏ BİLMİR" in lines[0]
    assert "niyyətdir" in lines[0]
