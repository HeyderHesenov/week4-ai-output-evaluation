"""Deterministik qraderlər — alt-sətir tələsi, azərbaycanca folding, ədədlər."""

from __future__ import annotations

import pytest

from eval.dataset import NumericClaim
from eval.graders import (
    extract_numbers,
    grade_deterministic,
    normalize_az,
    numeric_claim_satisfied,
)
from tests.conftest import make_case, make_chunk, make_observation


# --- azərbaycanca normalizasiya ---------------------------------------------


def test_python_lower_i_TELESI_bagladi() -> None:
    """`'I'.lower()` azərbaycanca 'ı' verməlidir, 'i' yox."""
    assert normalize_az("IŞIQ") == "ışıq"
    assert normalize_az("İSTİFADƏ") == "istifadə"


def test_birlesen_nokte_U0307_qalmir() -> None:
    """`'İ'.lower()` 'i' + U+0307 verir və alt-sətir uyğunluğunu sındırır."""
    assert "̇" not in normalize_az("İndeks")
    assert normalize_az("İndeks") == "indeks"


def test_bosluqlar_sixilir() -> None:
    assert normalize_az("  bir   iki \n üç ") == "bir iki üç"


# --- ədəd çıxarma -----------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1 500 manat", 1500.0),
        ("1.500 manat", 1500.0),
        ("99,9 faiz", 99.9),
        ("99.95 faiz", 99.95),
        ("128 GB", 128.0),
    ],
)
def test_ayirici_evristikasi(text, expected) -> None:
    values = [v for v, _, _ in extract_numbers(text)]
    assert expected in values


def test_azerbaycan_say_sozleri_taninir() -> None:
    values = [v for v, _, _ in extract_numbers("üç fərqli coğrafi zonada")]
    assert 3.0 in values


def test_soz_serhedi_UCUN_sozunu_tutmur() -> None:
    """`üç` nişanı `üçün` sözünün içindədir — sərhədsiz axtarış onu tutardı."""
    values = [v for v, _, _ in extract_numbers("bunun üçün lazımdır")]
    assert 3.0 not in values


# --- ədədi iddia ------------------------------------------------------------


def test_ALT_SETIR_telesi_bagladi() -> None:
    """«21» alt-sətri «2100» mətnində də var — Week 2-nin harness-i buna aldanırdı."""
    claim = NumericClaim(value=21, unit="iş gün")
    assert numeric_claim_satisfied("İllik məzuniyyət 21 iş günüdür.", claim) is True
    assert numeric_claim_satisfied("Ödəniş 2100 manatdır.", claim) is False


def test_99_9_ile_99_95_FERQLENIR() -> None:
    claim = NumericClaim(value=99.9, unit="%")
    assert numeric_claim_satisfied("SLA 99.9%", claim) is True
    assert numeric_claim_satisfied("SLA 99.95%", claim) is False


def test_vahid_RAQEMIN_yaxinliginda_axtarilir() -> None:
    claim = NumericClaim(value=10, unit="iş gün")
    assert numeric_claim_satisfied("10 iş gününə qədər ödənilir", claim) is True
    # Pəncərədən (24 simvol) kənardakı vahid uyğunluq saymır.
    assert (
        numeric_claim_satisfied(
            "10 nəfər burada çalışır və hər biri fərqlidir; iş günü 8 saatdır", claim
        )
        is False
    )


def test_PENCERE_evristikasinin_bilinen_zeifliyi() -> None:
    """Sənədləşdirilmiş məhdudiyyət: pəncərə daxilindəki qonşu vahid yanlış uyğunluq verir.

    `UNIT_WINDOW = 24` kompromisdir: dar pəncərə «10 iş gününə qədər» kimi
    doğru halları itirir, geniş pəncərə isə qonşu cümlədən vahid oğurlayır.
    Bu test səhvi GİZLƏTMİR — onun sərhədini kilidləyir ki, pəncərə
    gələcəkdə dəyişəndə nəticəsi görünsün.
    """
    claim = NumericClaim(value=10, unit="iş gün")
    assert numeric_claim_satisfied("10 nəfər işləyir; iş günü 8 saatdır", claim) is True


def test_vahid_alternativleri_boru_ile_verilir() -> None:
    claim = NumericClaim(value=1500, unit="AZN|manat")
    assert numeric_claim_satisfied("1500 manat", claim) is True
    assert numeric_claim_satisfied("1500 AZN", claim) is True


def test_dozumluluk_nezere_alinir() -> None:
    claim = NumericClaim(value=100, unit="", tolerance=5)
    assert numeric_claim_satisfied("təxminən 103 ədəd", claim) is True
    assert numeric_claim_satisfied("təxminən 110 ədəd", claim) is False


# --- yoxlamaların birləşməsi ------------------------------------------------


def test_butun_yoxlamalar_KECMELIDIR() -> None:
    case = make_case("c", contains=["free"], numeric=[{"value": 60, "unit": "sorğu"}])
    obs = make_observation(answer_text="Free planında dəqiqədə 60 sorğu [1].")
    assert grade_deterministic(case, obs).passed is True

    obs_missing = make_observation(answer_text="Dəqiqədə 60 sorğu [1].")
    result = grade_deterministic(case, obs_missing)
    assert result.passed is False
    assert "contains" in result.detail


def test_SUT_xetasi_qiymeti_ugursuz_edir() -> None:
    case = make_case("c", numeric=[{"value": 60}])
    obs = make_observation(answer_text="60 [1].", error="RuntimeError")
    assert grade_deterministic(case, obs).passed is False


def test_imtina_SEBEB_kodu_ile_yoxlanilir() -> None:
    """Yalnız «imtina etdi» zəifdir: boş indeks imtinası da onu keçərdi."""
    case = make_case("c", kind="refusal", reason_in=["low_relevance"], gold_sources=[])
    ok = make_observation(refused=True, reason="low_relevance", cited_labels=[])
    bad = make_observation(refused=True, reason="empty_index", cited_labels=[])

    assert grade_deterministic(case, ok).passed is True
    assert grade_deterministic(case, bad).passed is False


def test_sitat_teleb_olunur_ve_ETIBARSIZ_istinad_reddedilir() -> None:
    case = make_case("c", numeric=[{"value": 60}])
    obs = make_observation(answer_text="60 [9].", cited_labels=[], invalid_citations=[9])
    result = grade_deterministic(case, obs)

    assert result.passed is False
    assert "istinad" in result.detail


def test_ferqli_menbe_sayi_teleb_edile_bilir() -> None:
    case = make_case("c", min_distinct_cited_sources=2, numeric=[{"value": 60}])
    one = make_observation(
        answer_text="60 [1].",
        chunks=[make_chunk(label=1, source="atlas_api_senedi.md")],
        cited_labels=[1],
    )
    two = make_observation(
        answer_text="60 [1][2].",
        chunks=[
            make_chunk(label=1, source="atlas_api_senedi.md"),
            make_chunk(label=2, source="sirket_qaydalari.pdf"),
        ],
        cited_labels=[1, 2],
    )
    assert grade_deterministic(case, one).passed is False
    assert grade_deterministic(case, two).passed is True


def test_imtina_halinda_sitat_yoxlanmir() -> None:
    case = make_case("c", kind="refusal", gold_sources=[])
    obs = make_observation(refused=True, reason="low_relevance", cited_labels=[])
    names = {c.name for c in grade_deterministic(case, obs).checks}
    assert "citation" not in names
