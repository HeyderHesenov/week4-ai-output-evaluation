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


# --- Task 2: determinist təsnifat --------------------------------------------
#
# Korpus faktları (`atlas_infra_qeydleri.md`, `atlas_api_senedi.md`):
#   `atlas-cache-edge` YALNIZ Enterprise planında ayrılır, limiti 128 GB;
#   korpusda bir neçə fərqli «limit» var — sürət, səhifələmə, yaddaş, gündəlik.


def test_imtina_INYEKSIYA_olunmus_askarlayici_ile_taninir() -> None:
    """SUT-un öz `_is_refusal`-ı işlədilir — ikinci imtina tərifi yaradılmır."""
    from tools.generation_probe import tesnif

    assert tesnif(
        "dev_ambiguous_limit", "Sənədlərdə bu suala cavab tapılmadı.", q=_qurucular()
    ) == "imtina"


def test_muqeddime_QEBUL_ve_DUZELDILDI_ayird_olunur() -> None:
    """`dev_false_premise_free_cache`: doğru davranış Enterprise-ı ADLANDIRMAQ.

    Möhürlənmiş case-in mənfi yoxlaması (`not_contains: 128`) bunu ayırd edə
    bilmir — imtina da onu keçir. Probe-un meyarı buna görə daha dardır.
    """
    from tools.generation_probe import tesnif

    q = _qurucular()
    cid = "dev_false_premise_free_cache"
    assert tesnif(cid, "Sənədlərdə bu suala cavab tapılmadı.", q=q) == "imtina"
    assert tesnif(cid, "Free planı üçün yaddaş limiti 128 GB-dır [1].", q=q) == "muqeddime_qebul"
    assert tesnif(
        cid,
        "atlas-cache-edge yalnız Enterprise planında ayrılır, Free planına aid deyil [1].",
        q=q,
    ) == "muqeddime_duzeldildi"


def test_TEK_oxunus_ve_ADLANDIRILMIS_oxunuslar_ayird_olunur() -> None:
    """`dev_ambiguous_limit`: baseline «Limit 200-dür [1]» — tək oxunuş."""
    from tools.generation_probe import tesnif

    q = _qurucular()
    cid = "dev_ambiguous_limit"
    assert tesnif(cid, "Limit 200-dür [1].", q=q) == "tek_oxunus"
    assert tesnif(
        cid, "Korpusda bir neçə limit var: sürət limiti [1] və səhifələmə limiti [2].", q=q
    ) == "oxunuslar_adlandi"
    assert tesnif(cid, "Hansı limiti nəzərdə tutursunuz?", q=q) == "oxunuslar_adlandi"


def test_out_of_corpus_ucun_IMTINA_yaxsi_davranisdir() -> None:
    from tools.generation_probe import tesnif

    assert tesnif(
        "dev_out_of_corpus_graphql", "Sənədlərdə bu suala cavab tapılmadı.", q=_qurucular()
    ) == "imtina"


def test_probe_case_leri_MODELE_CATAN_uclukdur() -> None:
    """`dev_out_of_corpus_ceo` qəsdən yoxdur: qapı onu modelə çatmamış kəsir.

    Saxlanmış artefaktda `reason=low_relevance` və LLM çağırışı SIFIRDIR, yəni
    heç bir prompt dəyişikliyi onu sındıra bilməz. Cədvələ salmaq ölçülməmiş
    şeyi ölçülmüş kimi göstərərdi.
    """
    from tools.generation_probe import PROBE_CASES

    assert [c.case_id for c in PROBE_CASES] == [
        "dev_false_premise_free_cache",
        "dev_ambiguous_limit",
        "dev_out_of_corpus_graphql",
    ]
    assert [c.nezaret_davranisi for c in PROBE_CASES] == ["imtina", "tek_oxunus", "imtina"]
