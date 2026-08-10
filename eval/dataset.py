"""Test dəstinin yüklənməsi, validasiyası və hash-lənməsi.

NİYƏ `contains: ["21"]` KİFAYƏT DEYİL
-------------------------------------
Week 2-nin öz eval harness-i gözlənilən faktı alt-sətir kimi yoxlayırdı.
"21" alt-sətri "2100 manat" cavabında da var — yəni tamamilə yanlış cavab
testi keçir. Buna görə ədədi iddialar ayrıca `NumericClaim` tipi ilə
ifadə olunur: dəyər + vahid + dözümlülük, və qrader rəqəmi mətndən
parse edib vahidin yaxınlıqda olmasını tələb edir.

DATASET HASH-I NİYƏ XAM BAYT ÜZƏRİNDƏN DEYİL
--------------------------------------------
`sha256` kanonik JSON üzərindən hesablanır (`sort_keys=True`). Beləliklə
YAML-ı yenidən formatlamaq (şərh əlavə etmək, sıralamaq, sitat üslubunu
dəyişmək) möhürlənmiş bölgünü etibarsızlaşdırmır, amma İSTƏNİLƏN
semantik dəyişiklik — bir sözün redaktəsi belə — dərhal tutulur.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from .errors import DatasetError

# Korpus faylları — `gold_sources` yalnız bunlardan ibarət ola bilər.
# Yazı səhvi ilə uydurulmuş mənbə adı bütün kök-səbəb təsnifatını
# səssizcə sındırardı (heç bir chunk uyğun gəlməz → hər şey
# `retrieval_miss` görünər), buna görə ağ siyahı sərtdir.
CORPUS_FILES = frozenset(
    {
        "atlas_api_senedi.md",
        "atlas_infra_qeydleri.md",
        "sirket_qaydalari.pdf",
    }
)

CATEGORIES = frozenset(
    {
        "normal",
        "numeric_precision",
        "boundary",
        "multi_hop",
        "out_of_corpus",
        "false_premise",
        "ambiguous",
        "prompt_injection",
        "language_mixed",
        "open_ended",
    }
)

SPLITS = ("dev", "holdout")
GRADABLE = ("deterministic", "judge", "both")
EXPECTED_KINDS = ("answer", "refusal", "any")


@dataclass(frozen=True)
class NumericClaim:
    """Cavabda olmalı olan ədədi fakt."""

    value: float
    unit: str = ""
    tolerance: float = 0.0

    def describe(self) -> str:
        unit = f" {self.unit}" if self.unit else ""
        tol = f" (±{self.tolerance})" if self.tolerance else ""
        return f"{_fmt_number(self.value)}{unit}{tol}"


@dataclass(frozen=True)
class Expected:
    """Bir case-in gözləntisi."""

    kind: str = "answer"
    contains: tuple[str, ...] = ()
    not_contains: tuple[str, ...] = ()
    numeric: tuple[NumericClaim, ...] = ()
    reason_in: tuple[str, ...] = ()
    require_citation: bool = True
    min_distinct_cited_sources: int = 0
    # Hakimə göstərilən meyar. Gözlənilən CAVABI ehtiva ETMƏMƏLİDİR —
    # əks halda hakimə cavab vərəqəsi verilmiş olur.
    rubric: str = ""

    @property
    def has_assertion(self) -> bool:
        return bool(
            self.contains
            or self.not_contains
            or self.numeric
            or self.kind == "refusal"
            or self.min_distinct_cited_sources
        )


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    category: str
    split: str
    gradable: str
    expected: Expected
    gold_sources: tuple[str, ...] = ()
    gold_contains: tuple[str, ...] = ()
    note: str = ""

    @property
    def wants_judge(self) -> bool:
        return self.gradable in ("judge", "both")

    @property
    def wants_deterministic(self) -> bool:
        return self.gradable in ("deterministic", "both")


@dataclass(frozen=True)
class Dataset:
    cases: tuple[EvalCase, ...]
    sha256: str

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def subset(self, split: str) -> "Dataset":
        """Bölgü üzrə alt-dəst. Hash DƏYİŞMİR — o, bütün dəstin kimliyidir."""
        if split == "all":
            return self
        if split not in SPLITS:
            raise DatasetError(f"Naməlum bölgü: '{split}'. Gözlənilən: dev|holdout|all")
        return Dataset(
            cases=tuple(c for c in self.cases if c.split == split),
            sha256=self.sha256,
        )

    def by_id(self, case_id: str) -> EvalCase:
        for case in self.cases:
            if case.id == case_id:
                return case
        raise DatasetError(f"'{case_id}' id-li case tapılmadı.")

    def ids(self) -> frozenset[str]:
        return frozenset(c.id for c in self.cases)

    def ids_for(self, split: str) -> frozenset[str]:
        return frozenset(c.id for c in self.cases if c.split == split)

    def categories(self) -> dict[str, list[EvalCase]]:
        out: dict[str, list[EvalCase]] = {}
        for case in self.cases:
            out.setdefault(case.category, []).append(case)
        return out


# ---------------------------------------------------------------------------
# Yükləmə
# ---------------------------------------------------------------------------


def load_dataset(path: Path) -> Dataset:
    if not path.exists():
        raise DatasetError(f"Test dəsti tapılmadı: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or "cases" not in raw:
        raise DatasetError(f"{path}: kökdə 'cases' siyahısı gözlənilir.")
    # Açar var, amma dəyər siyahı deyil (boş `cases:` → None, yaxud skalyar):
    # tipi yoxlamasaq `enumerate` xam TypeError atır, `cli.main` isə yalnız
    # `EvalError` tutur — istifadəçi diaqnostika əvəzinə traceback görür.
    if not isinstance(raw["cases"], list):
        raise DatasetError(
            f"{path}: 'cases' siyahı olmalıdır, verilən: "
            f"{type(raw['cases']).__name__}."
        )

    cases = tuple(_parse_case(item, index=i) for i, item in enumerate(raw["cases"]))
    validate_dataset(cases)
    return Dataset(cases=cases, sha256=dataset_hash(cases))


def dataset_hash(cases: Sequence[EvalCase]) -> str:
    """Kanonik JSON üzərindən sha256 — YAML formatına həssas deyil."""
    payload = json.dumps(
        [_canonical(c) for c in sorted(cases, key=lambda c: c.id)],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical(case: EvalCase) -> dict[str, Any]:
    e = case.expected
    return {
        "id": case.id,
        "question": case.question,
        "category": case.category,
        "split": case.split,
        "gradable": case.gradable,
        "gold_sources": sorted(case.gold_sources),
        "gold_contains": sorted(case.gold_contains),
        "expected": {
            "kind": e.kind,
            "contains": sorted(e.contains),
            "not_contains": sorted(e.not_contains),
            "numeric": sorted(
                ({"value": n.value, "unit": n.unit, "tolerance": n.tolerance} for n in e.numeric),
                key=lambda d: (d["value"], d["unit"]),
            ),
            "reason_in": sorted(e.reason_in),
            "require_citation": e.require_citation,
            "min_distinct_cited_sources": e.min_distinct_cited_sources,
            "rubric": e.rubric,
        },
    }
    # QEYD: `note` qəsdən hash-ə DAXİL DEYİL — o, insan üçün şərhdir və
    # redaktəsi qiymətləndirmənin mənasını dəyişmir.


def _parse_case(item: Any, *, index: int) -> EvalCase:
    if not isinstance(item, dict):
        raise DatasetError(f"cases[{index}]: sözlük gözlənilir.")
    try:
        case_id = str(item["id"]).strip()
        question = str(item["question"]).strip()
        category = str(item["category"]).strip()
        split = str(item["split"]).strip()
        gradable = str(item["gradable"]).strip()
    except KeyError as exc:
        raise DatasetError(f"cases[{index}]: {exc} sahəsi yoxdur.") from None

    return EvalCase(
        id=case_id,
        question=question,
        category=category,
        split=split,
        gradable=gradable,
        expected=_parse_expected(item.get("expected") or {}, case_id=case_id),
        gold_sources=_as_tuple(item.get("gold_sources")),
        gold_contains=_as_tuple(item.get("gold_contains")),
        note=str(item.get("note", "")).strip(),
    )


def _parse_expected(raw: Any, *, case_id: str) -> Expected:
    if not isinstance(raw, dict):
        raise DatasetError(f"{case_id}: 'expected' sözlük olmalıdır.")
    numeric_raw = raw.get("numeric") or []
    if not isinstance(numeric_raw, list):
        raise DatasetError(f"{case_id}: 'expected.numeric' siyahı olmalıdır.")

    numeric: list[NumericClaim] = []
    for spec in numeric_raw:
        if not isinstance(spec, dict) or "value" not in spec:
            raise DatasetError(
                f"{case_id}: hər numeric qeydi 'value' sahəsi olan sözlük olmalıdır."
            )
        numeric.append(
            NumericClaim(
                value=float(spec["value"]),
                unit=str(spec.get("unit", "")).strip(),
                tolerance=float(spec.get("tolerance", 0.0)),
            )
        )

    return Expected(
        kind=str(raw.get("kind", "answer")).strip(),
        contains=_as_tuple(raw.get("contains")),
        not_contains=_as_tuple(raw.get("not_contains")),
        numeric=tuple(numeric),
        reason_in=_as_tuple(raw.get("reason_in")),
        require_citation=bool(raw.get("require_citation", True)),
        min_distinct_cited_sources=int(raw.get("min_distinct_cited_sources", 0)),
        rubric=str(raw.get("rubric", "")).strip(),
    )


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(v) for v in value)
    raise DatasetError(f"Siyahı və ya sətir gözlənilirdi, '{type(value).__name__}' gəldi.")


# ---------------------------------------------------------------------------
# Validasiya
# ---------------------------------------------------------------------------


def validate_dataset(cases: Sequence[EvalCase]) -> None:
    if not cases:
        raise DatasetError("Test dəsti boşdur.")

    # KİMLİK ƏVVƏL, MƏZMUN SONRA. İki keçid qəsdəndir: eyni id-li iki case-in
    # birincisi məzmun qaydasını da pozarsa, tək keçidli döngə məzmun xətasını
    # atar və TƏKRARLANAN ID heç vaxt bildirilməz. Struktur problemi məzmun
    # probleminin arxasında gizlənməməlidir — istifadəçi əvvəlcə "hansı case"
    # sualını həll edə bilməlidir.
    seen: set[str] = set()
    for case in cases:
        if not case.id:
            raise DatasetError("Boş case id-si var.")
        if case.id in seen:
            raise DatasetError(f"Təkrarlanan case id: '{case.id}'")
        seen.add(case.id)

    for case in cases:
        _validate_case(case)

    _validate_split_coverage(cases)


def _validate_case(case: EvalCase) -> None:
    cid = case.id
    if case.category not in CATEGORIES:
        raise DatasetError(
            f"{cid}: naməlum kateqoriya '{case.category}'. "
            f"İcazə verilənlər: {', '.join(sorted(CATEGORIES))}"
        )
    if case.split not in SPLITS:
        raise DatasetError(f"{cid}: 'split' dev və ya holdout olmalıdır.")
    if case.gradable not in GRADABLE:
        raise DatasetError(
            f"{cid}: 'gradable' {', '.join(GRADABLE)} dəyərlərindən biri olmalıdır."
        )
    if not case.question:
        raise DatasetError(f"{cid}: sual boş ola bilməz.")

    expected = case.expected
    if expected.kind not in EXPECTED_KINDS:
        raise DatasetError(
            f"{cid}: 'expected.kind' {', '.join(EXPECTED_KINDS)} olmalıdır."
        )

    if case.wants_judge and not expected.rubric:
        raise DatasetError(
            f"{cid}: hakim tərəfindən qiymətləndirilən case üçün 'rubric' məcburidir — "
            "hakimə nəyə görə bal verəcəyi deyilməlidir."
        )
    if case.wants_deterministic and not expected.has_assertion:
        raise DatasetError(
            f"{cid}: deterministik case-də heç bir iddia yoxdur "
            "(contains / not_contains / numeric / refusal / citation). "
            "İddiasız test həmişə keçir və heç nə ölçmür."
        )

    if expected.kind == "refusal" and (expected.contains or expected.numeric):
        raise DatasetError(
            f"{cid}: imtina gözlənilən case-də 'contains'/'numeric' ola bilməz — "
            "imtina mətnində məzmun faktı axtarmaq ziddiyyətlidir."
        )

    for source in case.gold_sources:
        if source not in CORPUS_FILES:
            raise DatasetError(
                f"{cid}: '{source}' korpusda yoxdur. "
                f"Mövcud fayllar: {', '.join(sorted(CORPUS_FILES))}"
            )

    if case.category == "out_of_corpus" and case.gold_sources:
        raise DatasetError(
            f"{cid}: korpusdan kənar sualın 'gold_sources'-u ola bilməz."
        )

    # SIZMA MÜHAFİZƏSİ: rubrikanın içində gözlənilən cavabın özü olmamalıdır.
    # Əks halda hakim cavab vərəqəsini oxuyur və "hakim düzgün tanıyır" nəticəsi
    # heç nə sübut etmir.
    lowered_rubric = expected.rubric.casefold()
    for needle in expected.contains:
        if needle and needle.casefold() in lowered_rubric:
            raise DatasetError(
                f"{cid}: rubrika gözlənilən cavabı ('{needle}') ehtiva edir. "
                "Hakim cavab vərəqəsini görməməlidir — rubrikanı MEYAR kimi yazın, "
                "cavab kimi yox."
            )
    for claim in expected.numeric:
        token = _fmt_number(claim.value)
        # SIZMA = «rubrika iddianı ÖDƏYİR», sadəcə rəqəmi ehtiva etmir.
        # Xam alt-sətir yoxlaması «3»-ü rubrikadakı «0-3 şkalası» və ya
        # «2026» içində də tapır və tamamilə qanuni case bütün dəstin
        # yüklənməsini dayandırırdı. Qraderin öz məntiqini (dəyər + vahid
        # yaxınlığı) təkrar istifadə etmək həm dəqiqdir, həm də tərifi
        # birmənalı edir: hakim rubrikadan cavabı OXUYA bilirsə, bu sızmadır.
        # Lokal import: `graders` bu modulu import edir, ona görə modul
        # səviyyəsində əks-import dövrə yaradardı.
        from .graders import numeric_claim_satisfied  # noqa: PLC0415

        # İKİ ƏLAMƏTİN BİRLƏŞMƏSİ:
        #   (a) ədəd rubrikada MÜSTƏQİL nişan kimi görünür — «Cavabda 128
        #       rəqəmi olmalıdır» açıq-aşkar cavab vərəqəsidir;
        #   (b) rubrika iddianı ödəyir (dəyər + vahid yaxınlığı) — «üç zona»
        #       kimi söz-rəqəm halları da tutulur.
        # (a)-da diapazon sərhədləri istisna olunur: «0-3 şkalası» rubrikanın
        # ÖZ qiymətləndirmə şkalasıdır, gözlənilən cavab deyil. Xam alt-sətir
        # yoxlaması onu sızma sayır və qanuni case-i bloklayırdı.
        standalone = bool(token) and re.search(
            rf"(?<![\d\-–]){re.escape(token)}(?![\d\-–])", lowered_rubric
        )
        if standalone or numeric_claim_satisfied(expected.rubric, claim):
            raise DatasetError(
                f"{cid}: rubrika gözlənilən ədədi ({token}) ehtiva edir. "
                "Hakim cavab vərəqəsini görməməlidir."
            )


def _validate_split_coverage(cases: Sequence[EvalCase]) -> None:
    """≥2 üzvü olan hər kateqoriya HƏR İKİ bölgüdə təmsil olunmalıdır.

    Səbəb: holdout yalnız asan kateqoriyalardan ibarət olsa, "yaxşılaşma
    holdout-da da təsdiqləndi" iddiası dəyərsizdir — çətin hallar dev-də
    qalıb və heç vaxt kor yoxlanmayıb.
    """
    by_category: dict[str, set[str]] = {}
    for case in cases:
        by_category.setdefault(case.category, set()).add(case.split)

    counts: dict[str, int] = {}
    for case in cases:
        counts[case.category] = counts.get(case.category, 0) + 1

    problems = [
        f"'{cat}' ({counts[cat]} case, yalnız {', '.join(sorted(splits))})"
        for cat, splits in sorted(by_category.items())
        if counts[cat] >= 2 and len(splits) < 2
    ]
    if problems:
        raise DatasetError(
            "Bölgü örtüyü qaydası pozulub — ən azı 2 üzvü olan hər kateqoriya "
            "həm dev, həm holdout bölgüsündə olmalıdır:\n  "
            + "\n  ".join(problems)
        )


def _fmt_number(value: float) -> str:
    """1500.0 → '1500', 99.9 → '99.9'."""
    if value == int(value):
        return str(int(value))
    return str(value)
