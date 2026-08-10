"""dev/holdout intizamı — test dəstinin çirklənməsinə qarşı mühafizə.

TƏLƏ
----
Prompt-u yaxşılaşdırmaq üçün istifadə olunan nümunələrlə sonra
"yaxşılaşmanın işlədiyini" sübut etmək dövri validasiyadır: model həmin
nümunələrə uyğunlaşdırılıb, ona görə orada yaxşı nəticə göstərməsi
məlumat daşımır. Buna görə case-lərin bir hissəsi (holdout) optimallaşdırma
zamanı HEÇ VAXT görünmür.

BEŞ MEXANİZM — HƏR BİRİ AYRI HÜCUM VEKTORUNU BAĞLAYIR
------------------------------------------------------
1. MÖHÜRLƏNMİŞ BÖLGÜ. `dataset_sha256` manifestə yazılır. Holdout sualını
   uğursuz olduğunu gördükdən sonra sakitcə redaktə etmək hash-i dəyişir
   və bütün əvvəlki run-ları etibarsızlaşdırır.
2. MƏNŞƏ YOXLAMASI. Hər few-shot nümunəsi hansı case-dən törədiyini elan
   edir; `source_case_ids ⊆ dev_ids` olmalıdır. Bu tələ üçün ən vacib
   sətir budur.
3. PARAFRAZ SIZMASI. Elan edilmiş mənşə kifayət deyil — nümunə holdout
   sualından *törədilə* bilər, amma onu göstərməyə bilər. Token-səviyyəli
   Jaccard oxşarlığı hər variant fraqmentini hər holdout sualı ilə
   müqayisə edir. Ölçülən maksimum dəyər hesabatda ÇAP OLUNUR — iddia yox,
   rəqəm.
4. STRUKTUR AYRILIQ. `eval.cli optimize` yalnız `--split dev` qəbul edir.
   Optimallaşdırma dövrü holdout nəticələrini fiziki olaraq oxuya bilmir.
5. YALNIZ-ƏLAVƏ REGİSTR. Hər holdout icrası `runs/holdout_ledger.json`-a
   yazılır və hesabatda tam çap olunur. Bu QƏSDƏN kilid deyil: sui-istifadəni
   GÖRÜNƏN və DAİMİ etmək onu qeyri-mümkün etməkdən daha dürüstdür, çünki
   yoxlayıcı bir yoxsa doqquz holdout run-ı olduğuna özü hökm verə bilir.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .dataset import Dataset
from .errors import ContaminationError
from .variants import PromptVariant

# Jaccard həddi. 0.60 empirik seçilib: fərqli mövzulu iki azərbaycanca sual
# adətən 0.15-0.30 aralığındadır (ortaq köməkçi sözlər), eyni sualın
# parafrazı isə 0.65-dən yuxarı qalxır.
JACCARD_THRESHOLD = 0.60

_TOKEN = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class SplitManifest:
    dataset_sha256: str
    dev_ids: frozenset[str]
    holdout_ids: frozenset[str]
    sealed_at: str
    sealed_by_commit: str
    coverage_exceptions: tuple[str, ...] = ()

    def to_json(self) -> dict[str, object]:
        return {
            "dataset_sha256": self.dataset_sha256,
            "dev_ids": sorted(self.dev_ids),
            "holdout_ids": sorted(self.holdout_ids),
            "sealed_at": self.sealed_at,
            "sealed_by_commit": self.sealed_by_commit,
            "coverage_exceptions": list(self.coverage_exceptions),
        }

    @property
    def sha256(self) -> str:
        import hashlib

        payload = json.dumps(self.to_json(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def seal_split(
    dataset: Dataset,
    *,
    sealed_at: str,
    sealed_by_commit: str,
    coverage_exceptions: Sequence[str] = (),
) -> SplitManifest:
    return SplitManifest(
        dataset_sha256=dataset.sha256,
        dev_ids=dataset.ids_for("dev"),
        holdout_ids=dataset.ids_for("holdout"),
        sealed_at=sealed_at,
        sealed_by_commit=sealed_by_commit,
        coverage_exceptions=tuple(coverage_exceptions),
    )


def write_split_manifest(manifest: SplitManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_json(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_split_manifest(path: Path) -> SplitManifest:
    if not path.exists():
        raise ContaminationError(
            f"Bölgü manifesti tapılmadı: {path}\n"
            "Möhürlənmiş bölgü olmadan dev/holdout intizamı yoxlanıla bilməz.\n"
            "Həlli: python -m eval.cli seal-split"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return SplitManifest(
        dataset_sha256=str(raw["dataset_sha256"]),
        dev_ids=frozenset(raw.get("dev_ids", [])),
        holdout_ids=frozenset(raw.get("holdout_ids", [])),
        sealed_at=str(raw.get("sealed_at", "")),
        sealed_by_commit=str(raw.get("sealed_by_commit", "")),
        coverage_exceptions=tuple(raw.get("coverage_exceptions", [])),
    )


# ---------------------------------------------------------------------------
# Mühafizə
# ---------------------------------------------------------------------------


def _tokens(text: str) -> frozenset[str]:
    from .graders import normalize_az

    return frozenset(_TOKEN.findall(normalize_az(text)))


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass(frozen=True)
class SimilarityHit:
    fragment: str
    case_id: str
    score: float


class ContaminationGuard:
    """dev/holdout intizamının icraçısı."""

    def __init__(self, manifest: SplitManifest, dataset: Dataset) -> None:
        self._manifest = manifest
        self._dataset = dataset

    @property
    def manifest(self) -> SplitManifest:
        return self._manifest

    def assert_dataset_unchanged(self) -> None:
        if self._dataset.sha256 != self._manifest.dataset_sha256:
            raise ContaminationError(
                "Test dəsti möhürləndikdən SONRA dəyişdirilib.\n"
                f"  möhürlənmiş: {self._manifest.dataset_sha256[:16]}\n"
                f"  cari:        {self._dataset.sha256[:16]}\n"
                "Bu, bütün əvvəlki run-ları etibarsızlaşdırır: holdout sualının "
                "uğursuzluğunu gördükdən sonra onu redaktə etmək qiymətləndirməni "
                "mənasız edir.\n"
                "Dəyişiklik qəsdəndirsə: bölgünü yenidən möhürləyin (seal-split "
                "--force) və bunu hesabatda AÇIQ qeyd edin."
            )

    def assert_split_allowed(self, split: str, *, action: str) -> None:
        """Optimallaşdırma holdout-a toxuna bilməz."""
        if action == "optimize" and split != "dev":
            raise ContaminationError(
                f"'{action}' yalnız --split dev ilə işləyə bilər (verilib: '{split}').\n"
                "Prompt-u holdout nəticələrinə baxaraq tənzimləmək dövri "
                "validasiyadır: sonrakı 'holdout təsdiqi' heç nə sübut etməz."
            )

    def assert_variant_clean(self, variant: PromptVariant) -> tuple[SimilarityHit, ...]:
        """Variantın holdout ilə əlaqəsini yoxlayır və ölçülən oxşarlığı qaytarır.

        İki ayrı yoxlama: elan edilmiş mənşə (sərt) və parafraz oxşarlığı
        (statistik). Birincisi yalanı, ikincisi unutqanlığı tutur.
        """
        if variant.is_identity:
            return ()

        illegal = variant.source_case_ids - self._manifest.dev_ids
        if illegal:
            unknown = illegal - self._dataset.ids()
            holdout = illegal & self._manifest.holdout_ids
            parts = []
            if holdout:
                parts.append(f"holdout case-ləri: {', '.join(sorted(holdout))}")
            if unknown:
                parts.append(f"naməlum id-lər: {', '.join(sorted(unknown))}")
            raise ContaminationError(
                f"'{variant.id}' variantının few-shot mənşəyi qaydanı pozur — "
                + "; ".join(parts)
                + ".\nFew-shot nümunələri YALNIZ dev case-lərinin uğursuzluğundan "
                "törəyə bilər."
            )

        hits = self.holdout_similarity(variant.text_fragments())
        breaches = [h for h in hits if h.score >= JACCARD_THRESHOLD]
        if breaches:
            worst = breaches[0]
            raise ContaminationError(
                f"'{variant.id}' variantının mətni holdout sualına həddindən artıq "
                f"oxşayır (Jaccard {worst.score:.2f} ≥ {JACCARD_THRESHOLD}).\n"
                f"  fraqment: {worst.fragment[:120]}\n"
                f"  holdout case: {worst.case_id}\n"
                "Elan edilmiş mənşə təmiz olsa belə, mətn holdout sualından "
                "törəmiş ola bilər. Fraqmenti yenidən yazın."
            )
        return hits

    def holdout_similarity(self, fragments: Sequence[str]) -> tuple[SimilarityHit, ...]:
        """Hər fraqment üçün ən oxşar holdout sualı, azalan sıra ilə."""
        holdout_cases = [c for c in self._dataset if c.id in self._manifest.holdout_ids]
        hits: list[SimilarityHit] = []
        for fragment in fragments:
            best: SimilarityHit | None = None
            for case in holdout_cases:
                score = jaccard(fragment, case.question)
                if best is None or score > best.score:
                    best = SimilarityHit(fragment=fragment, case_id=case.id, score=score)
            if best is not None:
                hits.append(best)
        return tuple(sorted(hits, key=lambda h: h.score, reverse=True))


# ---------------------------------------------------------------------------
# Holdout registri
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LedgerEntry:
    run_id: str
    at: str
    variant_id: str
    harness_commit: str
    case_count: int
    note: str = ""

    def to_json(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "at": self.at,
            "variant_id": self.variant_id,
            "harness_commit": self.harness_commit,
            "case_count": self.case_count,
            "note": self.note,
        }


def read_holdout_ledger(path: Path) -> tuple[LedgerEntry, ...]:
    if not path.exists():
        return ()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        LedgerEntry(
            run_id=str(item["run_id"]),
            at=str(item["at"]),
            variant_id=str(item["variant_id"]),
            harness_commit=str(item.get("harness_commit", "")),
            case_count=int(item.get("case_count", 0)),
            note=str(item.get("note", "")),
        )
        for item in raw
    )


def append_holdout_ledger(path: Path, entry: LedgerEntry) -> tuple[LedgerEntry, ...]:
    """Yalnız-əlavə. Mövcud girişlər heç vaxt silinmir və ya redaktə olunmur."""
    entries = read_holdout_ledger(path) + (entry,)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps([e.to_json() for e in entries], indent=2, ensure_ascii=False) + "\n"
    )
    # ATOMİK YAZI. Registr bütöv fayl kimi yenidən yazılır; birbaşa yazı
    # zamanı kəsilmə faylı KƏSİK qoyar və BÜTÜN əvvəlki girişlər itərdi —
    # yəni «yalnız-əlavə» zəmanətinin tam əksi. Müvəqqəti fayl + `os.replace`
    # eyni fayl sistemində atomikdir: ya köhnə, ya yeni məzmun görünür.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)
    return entries
