"""Uğursuzluğun KÖK SƏBƏBİ — hansı qat sındı.

NİYƏ «UĞURSUZ OLDU» KİFAYƏT DEYİL
----------------------------------
Pass-rate 70% olan iki sistem tamamilə fərqli düzəliş tələb edə bilər:
birində uğursuzluqların hamısı retrieval-dan (chunk ölçüsü, astana,
embedding), digərində generasiyadan (prompt) gəlir. Prompt-u düzəltmək
birinci sistemdə bir bal da qazandırmaz. Buna görə hər uğursuzluq QATA
görə təsnif olunur və hesabatdakı əsas cədvəl kateqoriya deyil, qat üzrədir.

TAKSONOMİYA XAM SÜBUTDAN TÖRƏYİR, ONUN İÇİNƏ YAZILMIR
------------------------------------------------------
`observation.py` docstring-indəki ayrım burada işə düşür: `classify()`
saxlanmış `SutObservation` üzərində yenidən işlədilə bilir. Taksonomiya
dəyişəndə `eval.cli reclassify` sistemi (və pulu) yenidən sərf etmədən
bütün run-ları yeni qaydalarla hesablayır.

BİR TƏLƏ: `obs.chunks`-un MƏNASI İMTİNA SƏBƏBİNDƏN ASILIDIR
------------------------------------------------------------
Week 2 pipeline-ı `low_relevance` imtinasında ÇƏKİLMİŞ bütün chunk-ları,
qalan hallarda isə yalnız QƏBUL EDİLMİŞ chunk-ları qaytarır
(vendor/.../rag/pipeline.py: `_refused(question, REASON_LOW_RELEVANCE,
retrieved, ...)` vs `refuse(...)` → `chunks`). Yəni:

  - `low_relevance` imtinasında gold mənbənin çəkilib-çəkilmədiyini
    GÖRÜRÜK → `retrieval_miss` ilə `gate_false_refusal` ayrıla bilir.
  - Cavab verilmiş halda isə yalnız qəbul edilmişləri görürük → qapının
    atdığı gold chunk `retrieval_miss` kimi görünür. Bu, dürüst
    məhdudiyyətdir və hesabatda qeyd olunur; ayırmaq üçün SUT-a əlavə
    ölçmə nöqtəsi lazım gələrdi, o isə «sistemi dəyişmədik» iddiasını pozar.
"""

from __future__ import annotations

from dataclasses import dataclass

from .dataset import EvalCase
from .graders import GradeResult
from .judge import Verdict
from .observation import SutObservation

# Qatlar — hesabatın əsas qruplaşdırması.
LAYER_NONE = "yoxdur"
LAYER_HARNESS = "harness"
LAYER_RETRIEVAL = "retrieval"
LAYER_GATE = "qapı"
LAYER_GENERATION = "generasiya"
LAYER_GROUNDING = "grounding"
LAYER_CITATION = "sitat"
LAYER_JUDGE = "hakim"

CAUSE_LAYERS: dict[str, str] = {
    "ok": LAYER_NONE,
    "sut_error": LAYER_HARNESS,
    "judge_error": LAYER_HARNESS,
    "empty_index": LAYER_HARNESS,
    "injection_leak": LAYER_GENERATION,
    "under_refusal": LAYER_GENERATION,
    "generation_wrong": LAYER_GENERATION,
    "over_refusal": LAYER_GENERATION,
    "retrieval_miss": LAYER_RETRIEVAL,
    "retrieval_rank": LAYER_RETRIEVAL,
    "gate_false_refusal": LAYER_GATE,
    "grounding_reject": LAYER_GROUNDING,
    "citation_invalid": LAYER_CITATION,
    "no_citation": LAYER_CITATION,
    "judge_low_score": LAYER_JUDGE,
}


@dataclass(frozen=True)
class RootCause:
    case_id: str
    repeat: int
    category: str
    detail: str

    @property
    def layer(self) -> str:
        return CAUSE_LAYERS.get(self.category, LAYER_NONE)

    @property
    def is_failure(self) -> bool:
        return self.category != "ok"


# ---------------------------------------------------------------------------
# Köməkçilər
# ---------------------------------------------------------------------------


def _failed(grade: GradeResult | None, name: str) -> bool:
    if grade is None:
        return False
    return any(c.name == name and not c.ok for c in grade.checks)


def _gold_in_chunks(case: EvalCase, obs: SutObservation) -> bool:
    if not case.gold_sources:
        return False
    gold = set(case.gold_sources)
    return any(c.source in gold for c in obs.chunks)


def _missing_gold(case: EvalCase, obs: SutObservation) -> set[str]:
    """Modelə ÇATMAYAN gold mənbələr.

    Çox mənbəli suallarda `any` kifayət deyil: iki sənəddən biri qapıdan
    keçib, digəri keçməyibsə, model sualın yarısına cavab verə bilir və
    uğursuzluq «generasiya zəifdir» kimi görünür. Halbuki səbəb yuxarı
    qatdadır — modelə lazım olan mətn heç çatmayıb.
    """
    if not case.gold_sources:
        return set()
    return set(case.gold_sources) - {c.source for c in obs.chunks}


def _gold_cited(case: EvalCase, obs: SutObservation) -> bool:
    if not case.gold_sources:
        return False
    return bool(set(case.gold_sources) & set(obs.cited_sources))


def _sources(obs: SutObservation) -> str:
    seen = sorted({c.source for c in obs.chunks})
    return ", ".join(seen) if seen else "(chunk yoxdur)"


# ---------------------------------------------------------------------------
# Təsnifat
# ---------------------------------------------------------------------------


def classify(
    case: EvalCase,
    obs: SutObservation,
    *,
    grade: GradeResult | None = None,
    verdict: Verdict | None = None,
) -> RootCause:
    """Sıralı qaydalar — ilk uyğunluq qalib gəlir.

    Sıra qəsdəndir: ölçmə xətaları (harness) əvvəl gəlir, çünki onlar
    sistemin davranışı haqqında HEÇ NƏ demir və digər qaydaların girişini
    etibarsız edir.
    """

    def cause(category: str, detail: str) -> RootCause:
        return RootCause(case_id=obs.case_id, repeat=obs.repeat, category=category, detail=detail)

    # 1. Ölçmə qatının xətaları -------------------------------------------
    if obs.error:
        return cause("sut_error", obs.error)

    if verdict is not None and not verdict.ok:
        return cause("judge_error", verdict.error or "hakim verdikt vermədi")

    if obs.reason == "empty_index":
        return cause(
            "empty_index",
            "vektor indeksi boşdur — bu, sistemin deyil, qurulmanın nəticəsidir",
        )

    # 2. Uğur ---------------------------------------------------------------
    deterministic_ok = grade is None or grade.passed
    judge_ok = verdict is None or verdict.passed
    if deterministic_ok and judge_ok:
        return cause("ok", grade.detail if grade is not None else "keçdi")

    # 3. Təhlükəsizlik: system prompt sızması ------------------------------
    # Ən yuxarıda dayanır, çünki sızma baş veribsə, cavabın məzmun
    # keyfiyyəti artıq əhəmiyyətsizdir.
    if _failed(grade, "not_contains"):
        return cause(
            "injection_leak",
            "cavabda olmamalı sətir var (system prompt sızması və ya qadağan olunmuş məzmun)",
        )

    # 4. İmtina gözlənilirdi, cavab verildi --------------------------------
    if case.expected.kind == "refusal" and not obs.refused:
        return cause(
            "under_refusal",
            f"imtina gözlənilirdi, sistem cavab verdi (top_score={obs.top_score:.2f})",
        )

    # 5. İmtina edildi — səbəb kodu qatı göstərir --------------------------
    if obs.refused:
        return _classify_refusal(case, obs, cause)

    # 6. Cavab verildi, amma yoxlamadan keçmədi ----------------------------
    if _failed(grade, "citation"):
        if obs.invalid_citations:
            return cause(
                "citation_invalid",
                f"etibarsız istinad nömrələri: {sorted(obs.invalid_citations)}",
            )
        return cause("no_citation", "cavabda mənbə istinadı yoxdur")

    missing = _missing_gold(case, obs)
    if missing:
        return cause(
            "retrieval_miss",
            f"gold mənbə(lər) modelə çatmayıb: {', '.join(sorted(missing))}; "
            f"qəbul edilən: {_sources(obs)}",
        )

    if case.gold_sources and not _gold_cited(case, obs):
        return cause(
            "retrieval_rank",
            f"gold mənbə çəkilib, lakin cavab ona istinad etməyib "
            f"(istinadlar: {sorted(obs.cited_sources) or 'yoxdur'})",
        )

    if grade is not None and not grade.passed:
        return cause("generation_wrong", grade.detail)

    return cause(
        "judge_low_score",
        f"hakim balı {verdict.score if verdict else '?'} < {verdict.threshold if verdict else '?'}"
        + (f" — {verdict.reason}" if verdict and verdict.reason else ""),
    )


def _classify_refusal(case: EvalCase, obs: SutObservation, cause) -> RootCause:
    """İmtina səbəb kodunu qata çevirir."""
    reason = obs.reason

    if reason == "low_relevance":
        # Bu YEGANƏ haldır ki, `obs.chunks` çəkilmiş HAMISI-dır (modul
        # docstring-inə bax) — yəni qapının səhvini retrieval səhvindən
        # burada həqiqətən ayıra bilirik.
        if _gold_in_chunks(case, obs):
            return cause(
                "gate_false_refusal",
                f"gold mənbə çəkilib, lakin qapı buraxmayıb "
                f"(top_score={obs.top_score:.2f}, astana={obs.threshold:.2f}, "
                f"top_lexical={obs.top_lexical:.2f})",
            )
        return cause(
            "retrieval_miss",
            f"gold mənbə çəkilməyib (çəkilən: {_sources(obs)}, "
            f"top_score={obs.top_score:.2f})",
        )

    if reason == "ungrounded":
        return cause(
            "grounding_reject",
            f"cavab istinad etdiyi bloklarla təsdiqlənmədi: {obs.grounding_detail or '(detal yoxdur)'}",
        )

    if reason == "invalid_citation":
        return cause(
            "citation_invalid",
            f"mövcud olmayan mənbəyə istinad: {sorted(obs.invalid_citations)}, "
            f"{obs.attempts} cəhddən sonra imtina",
        )

    if reason == "no_citation":
        return cause("no_citation", f"model sitatsız cavab verdi ({obs.attempts} cəhd)")

    if reason == "model_refused":
        # Model «tapmadım» deyibsə, ƏVVƏLCƏ ona lazım olanın çatıb-çatmadığını
        # yoxlamaq lazımdır. Gold mənbə qəbul edilmiş chunk-lar arasında
        # yoxdursa, bu, generasiya deyil, yuxarı qat uğursuzluğudur — və
        # prompt dəyişikliyi ilə DÜZƏLMİR.
        missing = _missing_gold(case, obs)
        if missing:
            return cause(
                "retrieval_miss",
                f"model imtina etdi, çünki gold mənbə(lər) ona çatmayıb: "
                f"{', '.join(sorted(missing))}; qəbul edilən: {_sources(obs)} "
                f"(top_score={obs.top_score:.2f}, astana={obs.threshold:.2f}). "
                "Qapı ilə retrieval bu yolda ayırd edilə bilmir — modul "
                "docstring-indəki məhdudiyyətə bax.",
            )
        return cause(
            "over_refusal",
            "gold mənbə modelə çatdığı halda model kontekstdə cavab tapmadı",
        )

    return cause("over_refusal", f"gözlənilməyən imtina (səbəb: {reason})")


# ---------------------------------------------------------------------------
# Qruplaşdırma
# ---------------------------------------------------------------------------


def group_by_category(causes: list[RootCause]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in causes:
        out[c.category] = out.get(c.category, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def group_by_layer(causes: list[RootCause]) -> dict[str, int]:
    """Uğursuzluqların qat üzrə paylanması — hesabatın əsas cədvəli."""
    out: dict[str, int] = {}
    for c in causes:
        if not c.is_failure:
            continue
        out[c.layer] = out.get(c.layer, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))
