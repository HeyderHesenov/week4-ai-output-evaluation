"""Generation probe — saxlanmış kontekstin təkrar oynadılması.

`rag.*` BURADA İMPORT EDİLMİR: `rag/store.py` modul səviyyəsində
`langchain_chroma`-nı çəkir, CI isə onu quraşdırmır (yalnız anthropic,
python-dotenv, PyYAML). SUT qurucuları `SutQurucular` ilə inyeksiya olunur,
ona görə bütün dəst açarsız, şəbəkəsiz və chroma-sız işləyir — repo-nun CI
şərhinin iddia etdiyi kimi, «lokalda keçir, CI-də keçmir» sinfi yaranmır.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from eval.observation import ChunkView


@dataclass
class _SahteChunk:
    """SUT-un `RetrievedChunk`-ı ilə EYNİ sahə dəsti."""

    label: int
    text: str
    source: str
    page: int
    chunk_index: int
    score: float
    lexical_score: float
    chunk_id: str


def _cv(**over) -> ChunkView:
    base = dict(
        label=1, source="atlas_api_senedi.md", page=0, chunk_index=3,
        score=0.61, lexical_score=0.28, chunk_id="abc123", text="Limit 200-dür.",
    )
    base.update(over)
    return ChunkView(**base)


def _qurucular(**over):
    from tools.generation_probe import SutQurucular

    base = dict(
        system_instruction="SİSTEM QAYDALARI",
        build_user_message=lambda soru, chunks, nonce: (
            f"<<<KONTEKST:{nonce}>>>\n"
            + "\n".join(c.text for c in chunks)
            + f"\n<<<SUAL>>>{soru}"
        ),
        chunk_cls=_SahteChunk,
        is_refusal=lambda t: t.strip().lower().startswith(
            "sənədlərdə bu suala cavab tapılmadı"
        ),
        make_nonce=lambda: "NONCE1",
    )
    base.update(over)
    return SutQurucular(**base)


# --- Task 1: qurucu inyeksiyası və prompt bərpası ----------------------------


def test_sahe_desti_uygun_gelmirse_OLCME_DAYANIR() -> None:
    """Sahə uyğunsuzluğu bərpanı SƏSSİZCƏ sadiqsiz edərdi.

    `RetrievedChunk` SUT-a aiddir; pin edilmiş commit dəyişib sahə əlavə
    olunsa, adapter köhnə dəstlə obyekt qurar, prompt fərqlənər və cədvəl
    «eyni kontekstdə ölçüldü» iddiasını yalandan daşıyar. Qapı testdə deyil,
    İCRA VAXTINDA olmalıdır — CI `rag.*`-ı import edə bilmir, ona görə orada
    həqiqi sinif heç vaxt yoxlanmazdı.
    """
    from tools.generation_probe import ProbeError, yoxla_saheler

    yoxla_saheler(_SahteChunk)  # uyğun gəlir → susur

    @dataclass
    class _Deyisib(_SahteChunk):
        yeni_sahe: str = ""

    with pytest.raises(ProbeError, match="sahə dəsti"):
        yoxla_saheler(_Deyisib)


def test_adapter_butun_saheleri_KOCURUR() -> None:
    from tools.generation_probe import sut_chunk

    chunk = sut_chunk(_cv(), q=_qurucular())

    assert chunk.label == 1
    assert chunk.text == "Limit 200-dür."
    assert chunk.source == "atlas_api_senedi.md"
    assert chunk.chunk_id == "abc123"
    assert chunk.score == 0.61
    assert chunk.lexical_score == 0.28


def test_mesajlar_INYEKSIYA_olunmus_qurucunu_cagirir() -> None:
    """Prompt probe-da yenidən yazılmır — qurucu kənardan gəlir."""
    from tools.generation_probe import mesajlar

    msgs = mesajlar(
        "Limit nə qədərdir?", [_cv()], suffix="", nonce="NONCE1", q=_qurucular()
    )

    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "<<<KONTEKST:NONCE1>>>" in msgs[1]["content"]
    assert "Limit nə qədərdir?" in msgs[1]["content"]
    assert "Limit 200-dür." in msgs[1]["content"]


def test_suffiks_system_mesajinin_SONUNA_elave_olunur() -> None:
    """Variant mexanizmi ilə eyni forma: qaydalar əvəzlənmir, əlavə olunur."""
    from tools.generation_probe import mesajlar

    q = _qurucular()
    baseline = mesajlar("S?", [_cv()], suffix="", nonce="N", q=q)[0]["content"]
    genis = mesajlar("S?", [_cv()], suffix="15. Əlavə qayda.", nonce="N", q=q)[0]["content"]

    assert genis.startswith(baseline)
    assert genis.endswith("15. Əlavə qayda.")
