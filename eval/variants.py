"""Prompt variantları — SUT-a toxunmadan few-shot və prompt dəyişikliyi.

NİYƏ MESAJ TRANSFORMASİYASI, NİYƏ SUT-U REDAKTƏ ETMİRİK
--------------------------------------------------------
Week 2-nin `SYSTEM_INSTRUCTION`-u modul sabitidir. Onu dəyişmək üçün ya
SUT faylını redaktə etmək (pin edilmiş commit iddiasını pozur), ya da
monkey-patch (səssiz və izlənməz) lazım gələrdi. Əvəzinə `RagPipeline`-ın
onsuz da mövcud olan `llm=` inyeksiya nöqtəsindən istifadə olunur:
sarğı `.invoke(messages)` çağırışını tutur və mesaj siyahısını göndərməzdən
əvvəl çevirir. SUT-un bir sətri belə dəyişmir.

DÜRÜST QEYD: variant sintez prompt-una MESAJ SƏVİYYƏSİNDƏ tətbiq olunur.
Bunu istehsala köçürmək Week 2-də `SYSTEM_INSTRUCTION` sabitini redaktə
etmək deməkdir. `logs/before_after.md` tətbiq ediləcək dəqiq diff-i
göstərir ki, "qiymətləndirmədə işlədi, istehsalda başqa şey oldu"
boşluğu qalmasın.

SİTAT TƏLƏSİ (bu qaydanın səbəbi ölçmə ilə tapılıb)
----------------------------------------------------
Week 2 `[N]` sitatlarını HƏMİN sorğunun qəbul edilmiş label-larına qarşı
yoxlayır. Aralıqdan kənar nömrə `invalid_citation` → məcburi yenidən
generasiya → imtina zəncirini işə salır. Yəni modelə `[2]` yazmağı
öyrədən few-shot nümunəsi pass-rate-i cavabın KEYFİYYƏTİ ilə heç bir
əlaqəsi olmayan mexanizmlə aşağı sala bilər və bu düşmə səhvən prompt
məzmununa yazılardı. Buna görə: hər nümunə DƏQİQ BİR kontekst bloku
göstərir və yalnız `[1]`-ə istinad edir. `validate()` bunu məcbur edir.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .errors import ConfigError

_CITATION = re.compile(r"\[(\d{1,2})\]")


def variant_sentinel(variant_id: str) -> str:
    """İdempotentlik nişanı — ikiqat tətbiqin qarşısını alır."""
    return f"<<<W4-VARIANT:{variant_id}>>>"


@dataclass(frozen=True)
class FewShotExample:
    """Sintez prompt-una əlavə olunan bir nümunə cütlüyü."""

    user: str
    assistant: str
    # Mənşə: bu nümunə HANSI dev case-lərinin uğursuzluğuna baxaraq
    # yazılıb. ContaminationGuard bunun ⊆ dev_ids olduğunu yoxlayır.
    source_case_ids: tuple[str, ...] = ()

    def validate(self, *, variant_id: str) -> None:
        if not self.user.strip() or not self.assistant.strip():
            raise ConfigError(f"{variant_id}: few-shot nümunəsi boş ola bilməz.")
        labels = {int(m) for m in _CITATION.findall(self.assistant)}
        if labels - {1}:
            raise ConfigError(
                f"{variant_id}: few-shot nümunəsi yalnız [1] istinadını işlədə bilər, "
                f"tapıldı: {sorted(labels)}.\n"
                "Səbəb: SUT sitat nömrəsini HƏMİN sorğunun label-larına qarşı yoxlayır. "
                "Aralıqdan kənar nömrə invalid_citation → yenidən generasiya → imtina "
                "zəncirini işə salır və pass-rate düşməsi səhvən prompt məzmununa yazılar."
            )


@dataclass(frozen=True)
class PromptVariant:
    """Sintez prompt-una tətbiq olunan çevirmə."""

    id: str
    label: str
    description: str = ""
    system_suffix: str = ""
    few_shot: tuple[FewShotExample, ...] = ()

    # -- yoxlama ------------------------------------------------------------

    def validate(self) -> None:
        if not self.id.strip():
            raise ConfigError("Variantın 'id' sahəsi boş ola bilməz.")
        for example in self.few_shot:
            example.validate(variant_id=self.id)

    @property
    def is_identity(self) -> bool:
        return not self.system_suffix.strip() and not self.few_shot

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            {
                "id": self.id,
                "system_suffix": self.system_suffix,
                "few_shot": [
                    {"user": e.user, "assistant": e.assistant} for e in self.few_shot
                ],
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def source_case_ids(self) -> frozenset[str]:
        return frozenset(cid for e in self.few_shot for cid in e.source_case_ids)

    def text_fragments(self) -> tuple[str, ...]:
        """Çirklənmə yoxlaması üçün variantın bütün insan-yazılı mətni."""
        parts: list[str] = []
        if self.system_suffix.strip():
            parts.extend(p for p in re.split(r"(?<=[.!?])\s+|\n+", self.system_suffix) if p.strip())
        for example in self.few_shot:
            parts.append(example.user)
            parts.append(example.assistant)
        return tuple(p.strip() for p in parts if p.strip())

    # -- tətbiq -------------------------------------------------------------

    def apply(self, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """YENİ siyahı qaytarır — arqument HEÇ VAXT dəyişdirilmir.

        `RagPipeline` öz `messages` siyahısını korreksiya cəhdləri arasında
        YERİNDƏ genişləndirir. Arqumenti dəyişdirsək, 2-ci cəhd ikiqat
        çevrilmiş girişlə gedərdi. `tests/test_variants.py` bunu kilidləyir.
        """
        original = [dict(m) for m in messages]
        if self.is_identity or not original:
            return original

        head = original[0]
        if head.get("role") != "system":
            return original

        sentinel = variant_sentinel(self.id)
        system_text = str(head.get("content", ""))
        if sentinel in system_text:  # artıq tətbiq olunub
            return original

        if self.system_suffix.strip():
            head = {
                **head,
                "content": f"{system_text}\n\n{self.system_suffix.strip()}\n{sentinel}",
            }
        else:
            head = {**head, "content": f"{system_text}\n{sentinel}"}

        shots: list[dict[str, Any]] = []
        for example in self.few_shot:
            shots.append({"role": "user", "content": example.user})
            shots.append({"role": "assistant", "content": example.assistant})

        return [head, *shots, *original[1:]]


IDENTITY_VARIANT = PromptVariant(
    id="baseline",
    label="Baseline (dəyişiklik yoxdur)",
    description="SUT-un öz prompt-u, toxunulmamış. Bütün müqayisələrin sıfır nöqtəsi.",
)


# ---------------------------------------------------------------------------
# Yükləmə
# ---------------------------------------------------------------------------


def load_variant(path: Path) -> PromptVariant:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: kökdə sözlük gözlənilir.")

    examples_raw = raw.get("few_shot") or []
    if not isinstance(examples_raw, list):
        raise ConfigError(f"{path}: 'few_shot' siyahı olmalıdır.")

    examples: list[FewShotExample] = []
    for item in examples_raw:
        if not isinstance(item, dict):
            raise ConfigError(f"{path}: hər few_shot qeydi sözlük olmalıdır.")
        sources = item.get("source_case_ids") or []
        if isinstance(sources, str):
            sources = [sources]
        examples.append(
            FewShotExample(
                user=str(item.get("user", "")).strip(),
                assistant=str(item.get("assistant", "")).strip(),
                source_case_ids=tuple(str(s) for s in sources),
            )
        )

    variant = PromptVariant(
        id=str(raw.get("id", path.stem)).strip(),
        label=str(raw.get("label", path.stem)).strip(),
        description=str(raw.get("description", "")).strip(),
        system_suffix=str(raw.get("system_suffix", "")).strip(),
        few_shot=tuple(examples),
    )
    variant.validate()
    return variant


def load_variants(directory: Path) -> dict[str, PromptVariant]:
    variants: dict[str, PromptVariant] = {IDENTITY_VARIANT.id: IDENTITY_VARIANT}
    if not directory.exists():
        return variants
    for path in sorted(directory.glob("*.yaml")):
        variant = load_variant(path)
        # `baseline` MÜSTƏSNA DEYİL, ƏKSİNƏ: o, toxunulmaz sıfır nöqtəsidir.
        # Əvvəlki şərt məhz onu istisna edirdi, yəni `id: baseline` yazılmış
        # istənilən YAML identity variantı SƏSSİZCƏ əvəz edə bilirdi — sonra
        # `--variant baseline` dəyişdirilmiş prompt-la işləyər, manifest və
        # hesabat isə onu «dəyişiklik yoxdur» kimi etiketləyərdi.
        if variant.id in variants:
            raise ConfigError(
                f"Təkrarlanan variant id: '{variant.id}' ({path})."
                + (
                    " 'baseline' identity variantının qorunan adıdır və"
                    " fayl ilə əvəz edilə bilməz."
                    if variant.id == IDENTITY_VARIANT.id
                    else ""
                )
            )
        variants[variant.id] = variant
    return variants
