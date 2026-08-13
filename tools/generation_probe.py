"""Generation probe — kontekst düzgün olduqda prompt qaydası nə dəyişir.

NİYƏ AYRICA ALƏT
----------------
Sual generasiya qatı haqqındadır, ona görə retrieval SABİT saxlanmalıdır. Tam
run işlətmək onu hər dəfə yenidən icra edər və qapı davranışı nəticəyə
qarışar. Burada chunk-lar SAXLANMIŞ artefaktdan oxunur: retrieval
konstruksiyaya görə sabitdir, embedding çağırılmır, indeks açılmır.

Hakim də çağırılmır — 2026-08-12-də kappa 0.13 ölçüldü, yəni hakimdən keçən
hər rəqəm ehtiyat bayrağı daşıyır. Bu cədvəl tam determinist qalır.

NİYƏ SUT QURUCULARI İNYEKSİYA OLUNUR
------------------------------------
`rag.pipeline` modul səviyyəsində `rag.store`-u, o isə `langchain_chroma`-nı
import edir. CI bu paketləri quraşdırmır (yalnız anthropic/dotenv/PyYAML), ona
görə testlərdə həqiqi import «lokalda keçir, CI-də keçmir» sinfini yaradardı —
repo-nun CI şərhi məhz bu sinfin mövcud olmadığını iddia edir. Prompt
qurucularını burada TƏKRAR YAZMAQ isə daha pis olardı: onda ölçdüyümüz prompt
SUT-un prompt-u olmazdı.

Sadiqliyi qoruyan şey testlər deyil, İCRA VAXTI QAPISIDIR: `sut_qurucular()`
SUT-un `RetrievedChunk` sahə dəstini yoxlayır və uyğunsuzluqda ölçməni
dayandırır.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_common import (  # noqa: E402
    PROBES_DIRNAME,
    add_sut_to_path,
    cedvel,
    probe_yarat,
    summary_metni,
)

from eval.config import Settings  # noqa: E402
from eval.errors import EvalError  # noqa: E402
from eval.observation import ChunkView  # noqa: E402


class ProbeError(EvalError):
    """Probe ölçməni davam etdirə bilmir."""


# SUT-un `RetrievedChunk` dataclass-ının GÖZLƏNİLƏN sahə dəsti.
SUT_CHUNK_SAHELERI = frozenset(
    {"label", "text", "source", "page", "chunk_index", "score", "lexical_score", "chunk_id"}
)


@dataclass(frozen=True)
class SutQurucular:
    """SUT-dan götürülən hər şey — bir yerdə, inyeksiya oluna bilən formada."""

    system_instruction: str
    build_user_message: Callable[[str, list, str], str]
    chunk_cls: type
    is_refusal: Callable[[str], bool]
    make_nonce: Callable[[], str]


def yoxla_saheler(chunk_cls: type) -> None:
    """Sahə dəsti gözlənildiyi kimidirmi — ÖLÇMƏ QAPISI.

    Test deyil, icra vaxtı yoxlaması: CI `rag.*`-ı import edə bilmədiyi üçün
    həqiqi sinif orada heç vaxt yoxlanmazdı. Uyğunsuzluqda ölçmə dayanır,
    çünki səssiz davam etmək «eyni kontekstdə ölçüldü» iddiasını yalan edərdi.
    """
    var_olan = {f.name for f in fields(chunk_cls)}
    if var_olan != SUT_CHUNK_SAHELERI:
        artiq = sorted(var_olan - SUT_CHUNK_SAHELERI)
        eskik = sorted(SUT_CHUNK_SAHELERI - var_olan)
        raise ProbeError(
            f"{chunk_cls.__name__} sahə dəsti gözlənildiyi kimi deyil "
            f"(artıq: {artiq or '—'}, əskik: {eskik or '—'}).\n"
            "SUT-un pin edilmiş commit-i dəyişib. Adapter köhnə dəstlə obyekt "
            "qursa, prompt səssizcə fərqlənər və cədvəl «eyni kontekstdə "
            "ölçüldü» iddiasını yalandan daşıyar."
        )


def sut_chunk(cv: ChunkView, *, q: SutQurucular) -> Any:
    """`ChunkView` → SUT-un `RetrievedChunk`-ı."""
    return q.chunk_cls(
        label=cv.label,
        text=cv.text,
        source=cv.source,
        page=cv.page,
        chunk_index=cv.chunk_index,
        score=cv.score,
        lexical_score=cv.lexical_score,
        chunk_id=cv.chunk_id,
    )


def mesajlar(
    question: str,
    chunks: Sequence[ChunkView],
    *,
    suffix: str,
    nonce: str,
    q: SutQurucular,
) -> list[dict]:
    """SUT-un `_answer`-inin qurduğu mesajların EYNİSİ (+ suffiks)."""
    system = q.system_instruction + (f"\n\n{suffix}" if suffix.strip() else "")
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": q.build_user_message(
                question, [sut_chunk(c, q=q) for c in chunks], nonce
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Təsnifat
# ---------------------------------------------------------------------------
#
# MEYARLAR QABAQCADAN SABİTDİR. Cədvələ baxdıqdan sonra meyar seçmək nəticəni
# istənilən istiqamətə çevirə bilər, ona görə onlar burada yazılır və
# artefaktda çap olunur.

# Korpus faktı (`atlas_infra_qeydleri.md`): `atlas-cache-edge` YALNIZ Enterprise
# planında ayrılır, limiti 128 GB. Müqəddiməni düzəltmək = doğru planı
# adlandırmaq; müqəddiməni qəbul etmək = Free üçün rəqəm vermək.
_DUZGUN_PLAN = "enterprise"
_YANLIS_REQEM = "128"

# Korpusdakı fərqli «limit» oxunuşları (`atlas_api_senedi.md` + infra qeydləri).
_LIMIT_NOVLERI = ("sürət", "səhifələmə", "yaddaş", "gündəlik", "eyni vaxtda")
_AYDINLASDIRMA = ("hansı limit", "hansı limiti", "dəqiqləşdir")


def _fold(text: str) -> str:
    """Azərbaycan «I/ı» problemi üçün SUT ilə eyni qaydada kiçildir."""
    return text.replace("I", "ı").replace("İ", "i").casefold()


@dataclass(frozen=True)
class ProbeCase:
    case_id: str
    nezaret_davranisi: str   # qol 0-dan GÖZLƏNİLƏN sinif (etibarlılıq qapısı)
    izah: str


PROBE_CASES = (
    ProbeCase(
        "dev_false_premise_free_cache", "imtina",
        "mənfi yoxlama ilə keçir, amma imtina ilə — düzəliş bacarığı ölçülməyib",
    ),
    ProbeCase(
        "dev_ambiguous_limit", "tek_oxunus",
        "«Limit 200-dür [1]» — səssizcə bir oxunuş seçilib",
    ),
    ProbeCase(
        "dev_out_of_corpus_graphql", "imtina",
        "REQRESSİYA NƏZARƏTİ — qol 2 imtinanı sındırmamalıdır",
    ),
)


def tesnif(case_id: str, text: str, *, q: SutQurucular) -> str:
    """Cavab mətni → davranış sinfi. Tam determinist, hakimsiz."""
    if q.is_refusal(text):
        return "imtina"
    alt = _fold(text)

    if case_id == "dev_false_premise_free_cache":
        if _DUZGUN_PLAN in alt:
            return "muqeddime_duzeldildi"
        if _YANLIS_REQEM in alt:
            return "muqeddime_qebul"
        return "diger"

    if case_id == "dev_ambiguous_limit":
        if any(a in alt for a in _AYDINLASDIRMA):
            return "oxunuslar_adlandi"
        if sum(1 for n in _LIMIT_NOVLERI if n in alt) >= 2:
            return "oxunuslar_adlandi"
        return "tek_oxunus"

    return "diger"
