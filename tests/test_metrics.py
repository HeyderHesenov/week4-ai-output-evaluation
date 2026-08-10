"""Metriklərin riyaziyyatı — əl ilə hesablanmış dəyərlərə qarşı."""

from __future__ import annotations

import pytest

from eval.config import load_prices
from eval.judge import Verdict
from eval.metrics import (
    CostReport,
    LabelSource,
    McNemarResult,
    PassRate,
    agreement,
    cohen_kappa,
    cost_report,
    judge_bias,
    latency_report,
    mcnemar,
    pass_rate,
    percentile,
    spearman,
    stable_pass,
    unstable_cases,
    wilson_interval,
)
from eval.rootcause import RootCause
from tests.conftest import PROJECT_ROOT, make_llm_call, make_observation


def cause(category: str, case_id: str = "c") -> RootCause:
    return RootCause(case_id=case_id, repeat=1, category=category, detail="")


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
        served_model=kwargs.pop("served_model", "claude-opus-5"),
        judge_prompt_sha256="0" * 64,
        threshold=2,
        **kwargs,
    )


# --- Wilson -----------------------------------------------------------------


def test_wilson_KAMIL_neticede_de_en_genis_deyil() -> None:
    """8/8-də Wald intervalı 0 eni verir — sistematik yalan. Wilson vermir."""
    low, high = wilson_interval(8, 8)
    assert high == pytest.approx(1.0)
    assert 0.60 < low < 0.70


def test_wilson_SIFIR_neticede_de_yuxari_hedd_verir() -> None:
    low, high = wilson_interval(0, 8)
    assert low == pytest.approx(0.0)
    assert 0.30 < high < 0.40


def test_wilson_kicik_n_de_GENIS_qalir() -> None:
    dar = wilson_interval(80, 100)
    genis = wilson_interval(8, 10)
    assert (genis[1] - genis[0]) > (dar[1] - dar[0])


def test_bos_deste_TAM_MELUMATSIZLIQ_gosterir() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)


# --- pass rate --------------------------------------------------------------


def test_olculebilmeyenler_MEXRECDEN_cixir() -> None:
    causes = [cause("ok"), cause("ok"), cause("retrieval_miss"), cause("judge_error")]
    rate = pass_rate(causes)

    assert rate.passed == 2
    assert rate.measured == 3
    assert rate.unmeasurable == 1
    assert rate.rate == pytest.approx(2 / 3)


def test_strict_rate_olculebilmeyeni_UGURSUZ_sayir() -> None:
    """İki rəqəm çap olunur, çünki bir rəqəm hər iki oxunuşa xidmət edə bilmir."""
    causes = [cause("ok"), cause("judge_error")]
    rate = pass_rate(causes)

    assert rate.rate == pytest.approx(1.0)
    assert rate.strict_rate == pytest.approx(0.5)


def test_describe_INTERVALI_de_gosterir() -> None:
    text = PassRate(passed=8, measured=8, unmeasurable=0).describe()
    assert "8/8" in text and "95% CI" in text


# --- sabitlik ---------------------------------------------------------------


def test_surusen_case_ler_SADALANIR() -> None:
    outcomes = {"a": [True, True, True], "b": [True, False, True], "c": [False, False]}
    assert unstable_cases(outcomes) == ("b",)


def test_sabit_kecid_BUTUN_tekrarlari_teleb_edir() -> None:
    outcomes = {"a": [True, True], "b": [True, False]}
    assert stable_pass(outcomes) == {"a": True, "b": False}


# --- McNemar ----------------------------------------------------------------


def test_mcnemar_YALNIZ_deyisen_caselere_baxir() -> None:
    baseline = {"a": True, "b": False, "c": False, "d": True}
    variant = {"a": True, "b": True, "c": True, "d": True}
    result = mcnemar(baseline, variant)

    assert result.improved == 2
    assert result.regressed == 0
    assert result.unchanged == 2


def test_mcnemar_kicik_yaxsilasma_EHEMIYYETLI_deyil() -> None:
    """2 case-lik yaxşılaşma 20-lik dəstdə təsadüf ola bilər — test bunu deyir."""
    baseline = {f"c{i}": False for i in range(2)} | {f"k{i}": True for i in range(18)}
    variant = {f"c{i}": True for i in range(2)} | {f"k{i}": True for i in range(18)}
    result = mcnemar(baseline, variant)

    assert result.improved == 2 and result.regressed == 0
    assert result.p_value == pytest.approx(0.5)
    assert result.significant is False


def test_mcnemar_boyuk_yaxsilasma_EHEMIYYETLIDIR() -> None:
    baseline = {f"c{i}": False for i in range(8)}
    variant = {f"c{i}": True for i in range(8)}
    result = mcnemar(baseline, variant)

    assert result.p_value < 0.01
    assert result.significant is True


def test_mcnemar_deyisiklik_yoxdursa_p_bir() -> None:
    same = {"a": True, "b": False}
    assert mcnemar(same, same).p_value == pytest.approx(1.0)


def test_mcnemar_describe_HER_IKI_isreni_gosterir() -> None:
    text = McNemarResult(improved=3, regressed=1, unchanged=16, p_value=0.6).describe()
    assert "+3" in text and "-1" in text


# --- kappa və qərəz ---------------------------------------------------------


def test_kappa_tam_raziliqda_bir() -> None:
    assert cohen_kappa([3, 2, 1, 0], [3, 2, 1, 0]) == pytest.approx(1.0)


def test_kappa_TESADUFI_raziliqi_cixir() -> None:
    # Bir tərəf HƏMİŞƏ '3' verirsə, xam razılıq 50%-dir, amma bu razılığın
    # hamısı təsadüfidir — kappa 0 göstərir və rəqəm düzgün oxunur.
    assert agreement([3, 3, 3, 3], [3, 2, 3, 2]) == pytest.approx(0.5)
    assert cohen_kappa([3, 3, 3, 3], [3, 2, 3, 2]) == pytest.approx(0.0)
    # Yalnız bir etiket işlənibsə kappa təyin olunmur; razılıq tamdırsa 1.
    assert cohen_kappa([3, 3, 3, 3], [3, 3, 3, 3]) == pytest.approx(1.0)


def test_kappa_uzunluqlar_ferqlidirse_XETA() -> None:
    with pytest.raises(ValueError):
        cohen_kappa([1, 2], [1])


def test_spearman_monoton_asililiqda_bir() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_BAGLARI_orta_sira_ile_isleyir() -> None:
    assert spearman([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)


RUN_A = "20260810T000000Z-dev-baseline"
RUN_B = "20260810T111111Z-dev-v1"


def keyed(run_id: str, verdicts) -> dict[tuple[str, str, int], Verdict]:
    return {(run_id, v.case_id, v.repeat): v for v in verdicts}


def sources(run_id: str, *case_ids: str, repeat: int = 1) -> dict[str, LabelSource]:
    return {cid: LabelSource(run_id, repeat) for cid in case_ids}


def test_verbosity_qerezi_olculur() -> None:
    """Hakim uzun cavaba yüksək bal verirsə, ρ müsbət və böyük olur."""
    verdicts = keyed(RUN_A, [verdict(f"c{i}", s) for i, s in enumerate([0, 1, 2, 3])])
    lengths = {(RUN_A, "c0", 1): 50, (RUN_A, "c1", 1): 200,
               (RUN_A, "c2", 1): 600, (RUN_A, "c3", 1): 1500}
    bias = judge_bias({}, {}, verdicts, lengths)

    assert bias.verbosity_rho == pytest.approx(1.0)
    assert bias.verbosity_sample == 4


def test_hakim_xetali_verdiktler_QERZ_hesabina_girmir() -> None:
    verdicts = keyed(RUN_A, [verdict("c0", 3), verdict("c1", None, error="imtina")])
    lengths = {(RUN_A, "c0", 1): 10, (RUN_A, "c1", 1): 10}
    bias = judge_bias({"c0": 3, "c1": 0}, sources(RUN_A, "c0", "c1"), verdicts, lengths)

    assert bias.sample_size == 1
    # İmtina 0 bala çevrilmir və səssizcə itmir — ayrıca sadalanır.
    assert bias.unmeasurable == ("c1",)
    assert bias.unmatched == ()


def test_TEKRARLAR_kappa_nin_n_ini_SISIRTMIR() -> None:
    """Bir insan qərarı bir cütdür — üç təkrar onu üç dəfə saymamalıdır.

    Bu, ölçmənin ən asan pozulan yeridir: holdout-da hər case üç dəfə
    işlədilir, `case_id` üzrə qoşalaşdırma isə n-i üç dəfə şişirdib güvən
    intervalını olduğundan dar göstərərdi.
    """
    verdicts = keyed(RUN_A, [
        Verdict(**{**vars(verdict("c0", 3)), "repeat": r}) for r in (1, 2, 3)
    ])
    lengths = {(RUN_A, "c0", r): 100 for r in (1, 2, 3)}

    bias = judge_bias({"c0": 3}, sources(RUN_A, "c0"), verdicts, lengths)

    assert bias.sample_size == 1
    # Verbosity ayrı sualdır: orada hər verdikt müstəqil müşahidədir.
    assert bias.verbosity_sample == 3


def test_etiketler_BIR_NECE_RUN_uzre_qosalasir() -> None:
    """9 etiket iki run-a bölünübsə, hər ikisi ölçülməlidir — biri yox."""
    verdicts = {**keyed(RUN_A, [verdict("dev_a", 3)]),
                **keyed(RUN_B, [verdict("hold_a", 1)])}
    lengths = {(RUN_A, "dev_a", 1): 10, (RUN_B, "hold_a", 1): 10}
    label_sources = {**sources(RUN_A, "dev_a"), **sources(RUN_B, "hold_a")}

    bias = judge_bias({"dev_a": 3, "hold_a": 1}, label_sources, verdicts, lengths)

    assert bias.sample_size == 2
    assert bias.unmatched == ()


def test_etiket_YANLIS_RUN_a_baglidirsa_sessizce_atilmir() -> None:
    """Mənbəsi tapılmayan etiket kappadan çıxır, AMMA görünən qalır."""
    verdicts = keyed(RUN_A, [verdict("dev_a", 3)])
    lengths = {(RUN_A, "dev_a", 1): 10}

    bias = judge_bias(
        {"dev_a": 3, "hold_a": 1},
        {**sources(RUN_A, "dev_a"), **sources(RUN_B, "hold_a")},
        verdicts,
        lengths,
    )

    assert bias.sample_size == 1
    assert bias.unmatched == ("hold_a",)


def test_etiketin_MENBESI_yoxdursa_unmatched() -> None:
    verdicts = keyed(RUN_A, [verdict("dev_a", 3)])
    bias = judge_bias({"dev_a": 3}, {}, verdicts, {(RUN_A, "dev_a", 1): 10})

    assert bias.sample_size == 0
    assert bias.unmatched == ("dev_a",)


def test_verbosity_HER_TEKRARIN_OZ_uzunlugunu_isledir() -> None:
    """Uzunluq `case_id` ilə açarlansaydı, son təkrar hamısını əvəz edərdi."""
    verdicts = keyed(RUN_A, [
        Verdict(**{**vars(verdict("c0", s)), "repeat": r})
        for r, s in ((1, 0), (2, 1), (3, 3))
    ])
    lengths = {(RUN_A, "c0", 1): 50, (RUN_A, "c0", 2): 200, (RUN_A, "c0", 3): 900}

    bias = judge_bias({}, {}, verdicts, lengths)

    assert bias.verbosity_sample == 3
    assert bias.verbosity_rho == pytest.approx(1.0)


# --- xərc -------------------------------------------------------------------


@pytest.fixture(scope="module")
def prices():
    return load_prices(PROJECT_ROOT / "data" / "prices.yaml")


def test_xerc_model_uzre_bolunur(prices) -> None:
    calls = [make_llm_call(model="gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)]
    verdicts = [verdict("c0", 3, input_tokens=1_000_000, output_tokens=0)]
    report = cost_report(calls, verdicts, prices)

    by_model = {m.model: m for m in report.models}
    assert by_model["gpt-4o-mini"].usd == pytest.approx(0.15)
    assert by_model["claude-opus-5"].usd == pytest.approx(5.00)
    assert report.total_usd == pytest.approx(5.15)


def test_kes_oxunusu_ONDA_BIR_qiymetle_hesablanir(prices) -> None:
    report = cost_report(
        [],
        [verdict("c0", 3, input_tokens=0, output_tokens=0, cached_read_tokens=1_000_000)],
        prices,
    )
    assert report.total_usd == pytest.approx(0.50)


def test_tesdiqlenmemis_qiymetler_SADALANIR(prices) -> None:
    report = cost_report([make_llm_call()], [], prices)
    assert "gpt-4o-mini" in report.unverified


def test_token_sayi_olmayan_cagirislar_SAYILIR(prices) -> None:
    from dataclasses import replace

    call = replace(make_llm_call(), usage_source="missing", input_tokens=0, output_tokens=0)
    report = cost_report([call], [], prices)
    assert report.missing_usage == 1


def test_embedding_boslugu_ACIQ_qeyd_olunur() -> None:
    assert "Embedding" in CostReport(models=(), unverified=()).note


def test_case_basina_xerc(prices) -> None:
    report = cost_report(
        [make_llm_call(model="gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)],
        [],
        prices,
    )
    assert report.per_case_usd(10) == pytest.approx(0.015)


# --- gecikmə ----------------------------------------------------------------


def test_gecikme_cemi_BAGLANIR() -> None:
    from eval.observation import RetrievalCall

    obs = make_observation(
        total_ms=1000.0,
        llm_calls=[make_llm_call(latency_ms=700.0)],
        retrieval_calls=[
            RetrievalCall(mode="hybrid", k=4, latency_ms=200.0, returned=4, top_score=0.8, query_chars=20)
        ],
    )
    report = latency_report([obs])

    assert report.llm_ms == pytest.approx(700.0)
    assert report.retrieval_ms == pytest.approx(200.0)
    assert report.overhead_ms == pytest.approx(100.0)
    assert report.unreconciled_ms == pytest.approx(0.0)
    assert report.share(report.llm_ms) == pytest.approx(0.7)


def test_persentil_interpolyasiya_edir() -> None:
    assert percentile([10, 20, 30, 40], 0.5) == pytest.approx(25.0)
    assert percentile([10], 0.95) == pytest.approx(10.0)
    assert percentile([], 0.5) == pytest.approx(0.0)
