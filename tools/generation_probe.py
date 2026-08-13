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
