"""Deterministik qraderlər — maşınla yoxlanan cavablar üçün.

NİYƏ NORMALİZASİYA SUT-DAN İMPORT EDİLMİR
------------------------------------------
Week 2-də hazır `rag.lexical` folding funksiyası var. Onu import etmək
cazibədardır, amma səhvdir: əgər SUT-un folding-ində baq varsa (məsələn
azərbaycan dilindəki `I`/`ı` cütünü səhv çevirirsə), eyni baqlı funksiya
həm cavabı üretirken, həm qiymətləndirərkən işləyər və baq ÖLÇÜLƏ
BİLMƏZ. Qiymətləndirici ölçdüyü sistemdən müstəqil olmalıdır.

NİYƏ `contains: ["21"]` DEYİL, `numeric`
----------------------------------------
"21" alt-sətri "2100" mətnində də var. Ədədi iddialar rəqəmi mətndən
parse edir, vahidi rəqəmin YAXINLIĞINDA tələb edir və dözümlülük qəbul
edir. Bu, eyni zamanda "99.9" ilə "99.95" arasındakı fərqi düzgün ayırır —
alt-sətir yoxlaması bunu prinsipcə edə bilməz.

AZƏRBAYCAN SAY SÖZLƏRİ
-----------------------
Korpus bəzi faktları rəqəmlə ("128 GB"), bəzilərini sözlə ("üç fərqli
coğrafi zonada") yazır. Model mənbəni əks etdirdiyi üçün hər iki forma
gözlənilir. Çıxarıcı 1-10 arası tək sözləri və onluqları tanıyır;
mürəkkəb saylar ("on beş") DƏSTƏKLƏNMİR və bu, məhdudiyyət kimi
sənədləşdirilib — korpusda belə hal yoxdur.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .dataset import EvalCase, NumericClaim
from .observation import SutObservation

# SUT-un maşınla oxunaqlı imtina səbəbləri (vendor/.../rag/pipeline.py).
# Burada TƏKRAR yazılır, import edilmir: SUT pin edilmiş commit-dədir və
# bu siyahının dəyişməsi bizim üçün sınıq test kimi görünməlidir.
REFUSAL_REASONS = frozenset(
    {
        "empty_index",
        "low_relevance",
        "model_refused",
        "no_citation",
        "invalid_citation",
        "ungrounded",
        "empty_question",
    }
)

# Vahidin rəqəmdən sonra axtarıldığı pəncərə (simvol). 24 seçilib: "10 iş
# gününə qədər" kimi ifadələrdə vahid rəqəmdən 1-2 söz sonra gəlir, amma
# pəncərə çox geniş olsa qonşu cümlədəki vahid yanlış uyğunluq verər.
UNIT_WINDOW = 24

_AZ_NUMBER_WORDS: dict[str, float] = {
    "sıfır": 0,
    "bir": 1,
    "iki": 2,
    "üç": 3,
    "dörd": 4,
    "beş": 5,
    "altı": 6,
    "yeddi": 7,
    "səkkiz": 8,
    "doqquz": 9,
    "on": 10,
    "iyirmi": 20,
    "otuz": 30,
    "qırx": 40,
    "əlli": 50,
    "altmış": 60,
    "yetmiş": 70,
    "səksən": 80,
    "doxsan": 90,
    "yüz": 100,
    "min": 1000,
}

# Rəqəm nişanı: qrup ayırıcıları (boşluq, nöqtə, vergül, NBSP) daxil.
# HƏR AYIRICI RƏQƏMLƏ İZLƏNMƏLİDİR. Əvvəlki acgöz ifadə ayırıcıların
# ARDICILLIĞINI da udurdu: «60, 600» BİR nişan kimi tutulur, parse edilə
# bilmir və HƏR İKİ ədəd itirdi («Zonalar: 1, 2, 3» → heç nə). Yəni siyahı
# şəklində verilmiş DÜZGÜN cavab qrader tərəfindən uğursuz sayılırdı.
_NUMBER_TOKEN = re.compile(r"\d+(?:[\u0020\u00a0.,]\d+)*")

# Saat formatı: «01:00». Belə nişanların içindəki rəqəmlər AYRI ədəd kimi
# çıxarılmamalıdır — «saat 01:00» mətni vahidsiz `value=1` və `value=0`
# iddialarını yanlış təsdiqləyirdi (vahid yoxdursa yaxınlıq yoxlanmır).
_CLOCK_TIME = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?")


# ---------------------------------------------------------------------------
# Normalizasiya
# ---------------------------------------------------------------------------


def normalize_az(text: str) -> str:
    """Azərbaycan dilinə uyğun kiçik hərfə salma + boşluq sıxılması.

    Python-un default `lower()`-i azərbaycanca üçün yanlışdır:
    'I'.lower() == 'i' (olmalıdır 'ı'), 'İ'.lower() == 'i̇' (i + birləşən
    nöqtə, U+0307). Hər ikisi alt-sətir uyğunluğunu səssizcə sındırır,
    ona görə əvvəlcə açıq xəritələmə tətbiq olunur.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("İ", "i").replace("I", "ı")
    return re.sub(r"\s+", " ", text.lower()).strip()


def _unit_alternatives(unit: str) -> tuple[str, ...]:
    """'AZN|manat' → ('azn', 'manat'). Boş vahid = tələb yoxdur."""
    if not unit.strip():
        return ()
    return tuple(normalize_az(part) for part in unit.split("|") if part.strip())


# ---------------------------------------------------------------------------
# Ədəd çıxarma
# ---------------------------------------------------------------------------


def _parse_number_token(token: str) -> float | None:
    """'1 500' → 1500.0, '99,9' → 99.9, '1.500' → 1500.0.

    Ayırıcı qərarı: son ayırıcıdan sonra DƏQİQ 3 rəqəm varsa qrup
    ayırıcısıdır, əks halda onluq ayırıcıdır. '1.500' → 1500 (min),
    '99.95' → 99.95 (onluq). Bu evristika azərbaycanca mətnlərdə hər
    iki üslubun qarışdığı halları düzgün həll edir.
    """
    cleaned = token.replace(" ", " ").strip()
    cleaned = re.sub(r"(?<=\d)\s+(?=\d)", "", cleaned)  # "1 500" → "1500"
    if not cleaned:
        return None

    separators = [(i, ch) for i, ch in enumerate(cleaned) if ch in ".,"]
    if not separators:
        try:
            return float(cleaned)
        except ValueError:
            return None

    last_index, _ = separators[-1]
    tail = cleaned[last_index + 1 :]
    if not tail.isdigit():
        return None

    if len(tail) == 3 and len(separators) >= 1:
        # Son ayırıcı qrup ayırıcısıdır → bütün ayırıcılar silinir.
        digits = re.sub(r"[.,]", "", cleaned)
        return float(digits) if digits.isdigit() else None

    head = re.sub(r"[.,]", "", cleaned[:last_index])
    if not head.isdigit():
        return None
    try:
        return float(f"{head}.{tail}")
    except ValueError:
        return None


def extract_numbers(text: str) -> tuple[tuple[float, int, int], ...]:
    """Mətndəki ədədləri (dəyər, başlanğıc, son) kimi qaytarır.

    Həm rəqəmləri, həm Azərbaycan say sözlərini tutur. Mövqelər
    NORMALLAŞDIRILMIŞ mətnə görədir — vahid yaxınlığı da orada yoxlanır.
    """
    normalized = normalize_az(text)
    found: list[tuple[float, int, int]] = []

    # Saat nişanlarının aralıqları: onların içindəki rəqəmlər ədəd deyil.
    clock_spans = [(m.start(), m.end()) for m in _CLOCK_TIME.finditer(normalized)]

    for match in _NUMBER_TOKEN.finditer(normalized):
        if any(s <= match.start() and match.end() <= e for s, e in clock_spans):
            continue
        value = _parse_number_token(match.group())
        if value is not None:
            found.append((value, match.start(), match.end()))

    for word, value in _AZ_NUMBER_WORDS.items():
        # Söz sərhədi məcburidir: "üç" nişanı "üçün" sözünün içindədir və
        # sərhədsiz axtarış onu yanlış tutardı.
        for match in re.finditer(rf"\b{re.escape(word)}\b", normalized):
            found.append((value, match.start(), match.end()))

    return tuple(sorted(found, key=lambda item: item[1]))


def numeric_claim_satisfied(text: str, claim: NumericClaim) -> bool:
    normalized = normalize_az(text)
    units = _unit_alternatives(claim.unit)

    for value, start, end in extract_numbers(text):
        if abs(value - claim.value) > claim.tolerance + 1e-9:
            continue
        if not units:
            return True
        after = normalized[end : end + UNIT_WINDOW]
        before = normalized[max(0, start - UNIT_WINDOW) : start]
        if any(u in after or u in before for u in units):
            return True
    return False


# ---------------------------------------------------------------------------
# Yoxlama nəticələri
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckOutcome:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class GradeResult:
    case_id: str
    passed: bool
    checks: tuple[CheckOutcome, ...]
    # Təkrar nömrəsi. Onsuz `reclassify` bütün təkrarları case_id üzrə
    # yığırdı və SONUNCU təkrarın qiymətini əvvəlkilərə də şamil edirdi —
    # məhz «flapping» case-lərdə, yəni `unstable_cases`-in aşkarlamalı
    # olduğu yerlərdə kök-səbəb yanlış çıxırdı.
    repeat: int = 1

    @property
    def failed_checks(self) -> tuple[CheckOutcome, ...]:
        return tuple(c for c in self.checks if not c.ok)

    @property
    def detail(self) -> str:
        if self.passed:
            return "bütün yoxlamalar keçdi"
        return "; ".join(f"{c.name}: {c.detail}" for c in self.failed_checks)


# ---------------------------------------------------------------------------
# Ayrı-ayrı yoxlamalar
# ---------------------------------------------------------------------------


def check_contains(case: EvalCase, obs: SutObservation) -> CheckOutcome | None:
    needles = case.expected.contains
    if not needles:
        return None
    text = normalize_az(obs.answer_text)
    missing = [n for n in needles if normalize_az(n) not in text]
    return CheckOutcome(
        name="contains",
        ok=not missing,
        detail="hamısı var" if not missing else f"çatışmır: {', '.join(missing)}",
    )


def check_not_contains(case: EvalCase, obs: SutObservation) -> CheckOutcome | None:
    needles = case.expected.not_contains
    if not needles:
        return None
    text = normalize_az(obs.answer_text)
    present = [n for n in needles if normalize_az(n) in text]
    return CheckOutcome(
        name="not_contains",
        ok=not present,
        detail="qadağan olunan sətir yoxdur"
        if not present
        else f"olmamalı idi: {', '.join(present)}",
    )


def check_numeric(case: EvalCase, obs: SutObservation) -> CheckOutcome | None:
    claims = case.expected.numeric
    if not claims:
        return None
    missing = [c.describe() for c in claims if not numeric_claim_satisfied(obs.answer_text, c)]
    return CheckOutcome(
        name="numeric",
        ok=not missing,
        detail="bütün ədədi iddialar təsdiqləndi"
        if not missing
        else f"tapılmadı: {', '.join(missing)}",
    )


def check_refusal(case: EvalCase, obs: SutObservation) -> CheckOutcome | None:
    """İmtina + DÜZGÜN SƏBƏB.

    Yalnız "imtina etdi" yoxlaması zəifdir: boş indeks səbəbindən imtina
    da onu keçərdi, halbuki bu, sistemin işlədiyini deyil, qurulmadığını
    göstərir. Səbəb kodunu da tələb etmək iddianı ciddiləşdirir.
    """
    if case.expected.kind != "refusal":
        return None
    # `reason_in` verilməyibsə, `empty_index` İSTİSNA olunur: yuxarıdakı
    # docstring məhz onu deyir — boş indeks səbəbindən imtina sistemin
    # işlədiyini deyil, QURULMADIĞINI göstərir. Əvvəlki fallback bütün
    # səbəbləri qəbul edirdi və qurulma nasazlığını «düzgün davranış»
    # kimi keçirirdi.
    allowed = set(case.expected.reason_in) or (set(REFUSAL_REASONS) - {"empty_index"})
    ok = obs.refused and obs.reason in allowed
    if obs.refused and obs.reason not in allowed:
        detail = f"imtina etdi, lakin səbəb '{obs.reason}' qəbul olunanlar arasında deyil"
    elif not obs.refused:
        detail = f"imtina gözlənilirdi, cavab verildi (səbəb: {obs.reason})"
    else:
        detail = f"imtina, səbəb: {obs.reason}"
    return CheckOutcome(name="refusal", ok=ok, detail=detail)


def check_answered(case: EvalCase, obs: SutObservation) -> CheckOutcome | None:
    """`kind: answer` halında imtina uğursuzluqdur."""
    if case.expected.kind != "answer":
        return None
    return CheckOutcome(
        name="answered",
        ok=not obs.refused,
        detail="cavab verildi"
        if not obs.refused
        else f"cavab gözlənilirdi, imtina edildi (səbəb: {obs.reason})",
    )


def check_citation(case: EvalCase, obs: SutObservation) -> CheckOutcome | None:
    if not case.expected.require_citation or obs.refused:
        return None
    ok = bool(obs.cited_labels) and not obs.invalid_citations
    if not obs.cited_labels:
        detail = "cavabda mənbə istinadı yoxdur"
    elif obs.invalid_citations:
        detail = f"etibarsız istinad: {sorted(obs.invalid_citations)}"
    else:
        detail = f"istinadlar: {sorted(obs.cited_labels)}"
    return CheckOutcome(name="citation", ok=ok, detail=detail)


def check_distinct_sources(case: EvalCase, obs: SutObservation) -> CheckOutcome | None:
    required = case.expected.min_distinct_cited_sources
    if required <= 0 or obs.refused:
        return None
    sources = obs.cited_sources
    return CheckOutcome(
        name="distinct_sources",
        ok=len(sources) >= required,
        detail=f"{len(sources)}/{required} fərqli mənbə"
        + (f" ({', '.join(sorted(sources))})" if sources else ""),
    )


_CHECKS = (
    check_answered,
    check_refusal,
    check_contains,
    check_not_contains,
    check_numeric,
    check_citation,
    check_distinct_sources,
)


def grade_deterministic(case: EvalCase, obs: SutObservation) -> GradeResult:
    """Bütün tətbiq olunan yoxlamaları işlədir. Hamısı keçməlidir."""
    checks = tuple(c for check in _CHECKS if (c := check(case, obs)) is not None)
    return GradeResult(
        case_id=case.id,
        passed=all(c.ok for c in checks) and not obs.error,
        checks=checks,
        repeat=obs.repeat,
    )
