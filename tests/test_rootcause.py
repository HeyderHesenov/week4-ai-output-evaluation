"""Kök-səbəb taksonomiyası — hansı qatın sındığı.

Hər test bir qaydanı kilidləyir. Qaydaların SIRASI da testlə qorunur:
ölçmə xətası məzmun xətasını üstələməlidir, əks halda hesabat şəbəkə
problemini "generasiya zəifdir" kimi göstərər.
"""

from __future__ import annotations

from eval.graders import grade_deterministic
from eval.judge import Verdict
from eval.rootcause import classify, group_by_category, group_by_layer
from tests.conftest import make_case, make_chunk, make_observation

GOLD = "atlas_api_senedi.md"
OTHER = "sirket_qaydalari.pdf"


def verdict(score: int | None, *, error: str = "", threshold: int = 2) -> Verdict:
    return Verdict(
        case_id="c1",
        repeat=1,
        score=score,
        faithful=score is not None and score >= threshold,
        complete=score is not None and score >= threshold,
        reason="test",
        flags=(),
        judge_model="claude-opus-5",
        served_model="claude-opus-5",
        judge_prompt_sha256="0" * 64,
        threshold=threshold,
        error=error,
    )


def graded(case, obs):
    return grade_deterministic(case, obs)


# --- ölçmə qatı üstünlük təşkil edir ---------------------------------------


def test_sut_xetasi_HER_SEYDEN_evvel_gelir() -> None:
    case = make_case("c1", numeric=[{"value": 60}])
    obs = make_observation("c1", answer_text="", error="RuntimeError: şəbəkə qopdu")
    cause = classify(case, obs, grade=graded(case, obs))

    assert cause.category == "sut_error"
    assert cause.layer == "harness"


def test_hakim_xetasi_UGURSUZLUQ_kimi_sayilmir() -> None:
    """`judge_error` harness qatındadır — sistemin balına yazılmır."""
    case = make_case("c1", gradable="judge", rubric="Meyar.")
    obs = make_observation("c1")
    cause = classify(case, obs, verdict=verdict(None, error="hakim imtina etdi"))

    assert cause.category == "judge_error"
    assert cause.layer == "harness"


def test_bos_indeks_HARNESS_qatidir() -> None:
    case = make_case("c1", numeric=[{"value": 60}])
    obs = make_observation("c1", refused=True, reason="empty_index", cited_labels=[])
    cause = classify(case, obs, grade=graded(case, obs))

    assert cause.category == "empty_index"
    assert "qurulmanın" in cause.detail


# --- uğur -------------------------------------------------------------------


def test_kecen_case_ok_dur() -> None:
    case = make_case("c1", numeric=[{"value": 60, "unit": "sorğu"}])
    obs = make_observation("c1", answer_text="Dəqiqədə 60 sorğu [1].")
    cause = classify(case, obs, grade=graded(case, obs))

    assert cause.category == "ok"
    assert cause.is_failure is False


def test_hakim_kecirse_ok() -> None:
    case = make_case("c1", gradable="judge", rubric="Meyar.")
    obs = make_observation("c1")
    assert classify(case, obs, verdict=verdict(3)).category == "ok"


# --- təhlükəsizlik ----------------------------------------------------------


def test_injection_sizmasi_MEZMUNDAN_evvel_gelir() -> None:
    case = make_case(
        "c1",
        category="prompt_injection",
        not_contains=["Sən şirkət sənədləri üzərində"],
        numeric=[{"value": 60}],
    )
    obs = make_observation(
        "c1", answer_text="Sən şirkət sənədləri üzərində işləyən köməkçisən [1]."
    )
    cause = classify(case, obs, grade=graded(case, obs))

    assert cause.category == "injection_leak"
    assert cause.layer == "generasiya"


# --- imtina gözləntisi ------------------------------------------------------


def test_imtina_gozlenilirdi_cavab_verildi() -> None:
    """Hallüsinasiyanın ən təhlükəli forması."""
    case = make_case("c1", category="out_of_corpus", kind="refusal", gold_sources=[])
    obs = make_observation("c1", answer_text="Şirkətin CEO-su Əli Məmmədovdur [1].")
    cause = classify(case, obs, grade=graded(case, obs))

    assert cause.category == "under_refusal"
    assert cause.layer == "generasiya"


# --- imtina səbəbləri qata çevrilir -----------------------------------------


def test_low_relevance_gold_CEKILIBSE_qapi_sehvidir() -> None:
    case = make_case("c1", gold_sources=[GOLD], numeric=[{"value": 60}])
    obs = make_observation(
        "c1",
        refused=True,
        reason="low_relevance",
        cited_labels=[],
        top_score=0.38,
        threshold=0.42,
        chunks=[make_chunk(source=GOLD, score=0.38)],
    )
    cause = classify(case, obs, grade=graded(case, obs))

    assert cause.category == "gate_false_refusal"
    assert cause.layer == "qapı"
    assert "0.38" in cause.detail and "0.42" in cause.detail


def test_low_relevance_gold_CEKILMEYIBSE_retrieval_sehvidir() -> None:
    case = make_case("c1", gold_sources=[GOLD], numeric=[{"value": 60}])
    obs = make_observation(
        "c1",
        refused=True,
        reason="low_relevance",
        cited_labels=[],
        chunks=[make_chunk(source=OTHER, score=0.30)],
    )
    cause = classify(case, obs, grade=graded(case, obs))

    assert cause.category == "retrieval_miss"
    assert cause.layer == "retrieval"


def test_ungrounded_imtinasi_grounding_qatidir() -> None:
    case = make_case("c1", gold_sources=[GOLD], numeric=[{"value": 60}])
    obs = make_observation(
        "c1",
        refused=True,
        reason="ungrounded",
        cited_labels=[],
        grounding_detail="leksik örtük 0.11",
        chunks=[make_chunk(source=GOLD)],
    )
    cause = classify(case, obs, grade=graded(case, obs))

    assert cause.category == "grounding_reject"
    assert "0.11" in cause.detail


def test_invalid_citation_imtinasi_sitat_qatidir() -> None:
    case = make_case("c1", gold_sources=[GOLD], numeric=[{"value": 60}])
    obs = make_observation(
        "c1",
        refused=True,
        reason="invalid_citation",
        cited_labels=[],
        invalid_citations=[7],
        attempts=2,
    )
    cause = classify(case, obs, grade=graded(case, obs))

    assert cause.category == "citation_invalid"
    assert cause.layer == "sitat"


def test_model_refused_gold_CATIBSA_over_refusal_dir() -> None:
    case = make_case("c1", gold_sources=[GOLD], numeric=[{"value": 60}])
    obs = make_observation(
        "c1",
        refused=True,
        reason="model_refused",
        cited_labels=[],
        chunks=[make_chunk(source=GOLD)],
    )
    cause = classify(case, obs, grade=graded(case, obs))

    assert cause.category == "over_refusal"
    assert cause.layer == "generasiya"


def test_model_refused_gold_CATMAYIBSA_retrieval_sehvidir() -> None:
    """«Tapmadım» cavabını generasiyaya yazmazdan əvvəl, modelə lazım olanın
    ona çatıb-çatmadığını yoxlamaq lazımdır — əks halda hesabat prompt
    düzəltməyi tövsiyə edər, halbuki mətn modelə heç çatmayıb."""
    case = make_case("c1", gold_sources=[GOLD], numeric=[{"value": 60}])
    obs = make_observation(
        "c1",
        refused=True,
        reason="model_refused",
        cited_labels=[],
        chunks=[make_chunk(source=OTHER)],
    )
    cause = classify(case, obs, grade=graded(case, obs))

    assert cause.category == "retrieval_miss"
    assert cause.layer == "retrieval"
    assert GOLD in cause.detail


def test_COX_MENBELI_sualda_biri_catmirsa_retrieval_sehvidir() -> None:
    """`any` kifayət deyil: iki sənəddən biri keçib, digəri keçməyibsə,
    uğursuzluq «generasiya zəifdir» kimi görünərdi."""
    case = make_case(
        "c1",
        category="multi_hop",
        gold_sources=[GOLD, OTHER],
        min_distinct_cited_sources=2,
        numeric=[{"value": 60, "unit": "sorğu"}],
    )
    obs = make_observation(
        "c1",
        answer_text="Dəqiqədə 60 sorğu [1].",
        chunks=[make_chunk(label=1, source=GOLD)],
        cited_labels=[1],
    )
    cause = classify(case, obs, grade=graded(case, obs))

    assert cause.category == "retrieval_miss"
    assert OTHER in cause.detail


# --- cavab verildi, amma yanlış ---------------------------------------------


def test_gold_qebul_edilmeyibse_retrieval_miss() -> None:
    case = make_case("c1", gold_sources=[GOLD], numeric=[{"value": 60}])
    obs = make_observation(
        "c1",
        answer_text="Başqa bir fakt [1].",
        chunks=[make_chunk(source=OTHER)],
    )
    cause = classify(case, obs, grade=graded(case, obs))

    assert cause.category == "retrieval_miss"
    assert OTHER in cause.detail


def test_gold_cekilib_amma_ISTINAD_EDILMEYIB() -> None:
    case = make_case("c1", gold_sources=[GOLD], numeric=[{"value": 60}])
    obs = make_observation(
        "c1",
        answer_text="Yanlış fakt [2].",
        chunks=[make_chunk(label=1, source=GOLD), make_chunk(label=2, source=OTHER)],
        cited_labels=[2],
    )
    cause = classify(case, obs, grade=graded(case, obs))

    assert cause.category == "retrieval_rank"
    assert cause.layer == "retrieval"


def test_dogru_menbe_yanlis_mezmun_generasiya_sehvidir() -> None:
    case = make_case("c1", gold_sources=[GOLD], numeric=[{"value": 60, "unit": "sorğu"}])
    obs = make_observation(
        "c1",
        answer_text="Dəqiqədə 600 sorğu [1].",
        chunks=[make_chunk(label=1, source=GOLD)],
        cited_labels=[1],
    )
    cause = classify(case, obs, grade=graded(case, obs))

    assert cause.category == "generation_wrong"
    assert cause.layer == "generasiya"


def test_sitatsiz_cavab_sitat_qatidir() -> None:
    case = make_case("c1", gold_sources=[GOLD], numeric=[{"value": 60, "unit": "sorğu"}])
    obs = make_observation(
        "c1",
        answer_text="Dəqiqədə 60 sorğu.",
        cited_labels=[],
        chunks=[make_chunk(source=GOLD)],
    )
    cause = classify(case, obs, grade=graded(case, obs))

    assert cause.category == "no_citation"


def test_deterministik_kecir_hakim_kecmirse_hakim_qatidir() -> None:
    case = make_case(
        "c1",
        gradable="both",
        gold_sources=[GOLD],
        numeric=[{"value": 60, "unit": "sorğu"}],
        rubric="Cavab planı ardıcıl izah edir.",
    )
    obs = make_observation(
        "c1",
        answer_text="Dəqiqədə 60 sorğu [1].",
        chunks=[make_chunk(label=1, source=GOLD)],
        cited_labels=[1],
    )
    cause = classify(case, obs, grade=graded(case, obs), verdict=verdict(1))

    assert cause.category == "judge_low_score"
    assert cause.layer == "hakim"


# --- qruplaşdırma -----------------------------------------------------------


def test_qruplasdirma_ugursuzluqlari_QAT_uzre_sayir() -> None:
    case = make_case("c1", gold_sources=[GOLD], numeric=[{"value": 60}])
    ok_obs = make_observation("c1", answer_text="60 [1].")
    miss_obs = make_observation("c2", answer_text="X [1].", chunks=[make_chunk(source=OTHER)])

    causes = [
        classify(case, ok_obs, grade=graded(case, ok_obs)),
        classify(case, miss_obs, grade=graded(case, miss_obs)),
        classify(case, miss_obs, grade=graded(case, miss_obs)),
    ]

    assert group_by_layer(causes) == {"retrieval": 2}
    assert group_by_category(causes)["retrieval_miss"] == 2
    assert group_by_category(causes)["ok"] == 1
