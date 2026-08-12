"""Konfiqurasiya: açar sızmır, placeholder tutulur, qiymət cədvəli sərtdir."""

from __future__ import annotations

import json

import pytest

from eval.config import (
    ANTHROPIC_PLACEHOLDER,
    OPENAI_PLACEHOLDER,
    ModelPrice,
    Settings,
    load_prices,
)
from eval.errors import ConfigError
from tests.conftest import make_settings


def test_repr_ACARLARI_sizdirmir() -> None:
    text = repr(make_settings())
    assert "sk-test-fake-openai-key" not in text
    assert "sk-ant-test-fake-anthropic-key" not in text


def test_public_dict_ACARLARI_sizdirmir() -> None:
    payload = json.dumps(make_settings().public_dict(), ensure_ascii=False)
    assert "sk-" not in payload
    assert not [k for k in make_settings().public_dict() if "key" in k.lower()]


def test_public_dict_asdict_MINASINI_bagladi() -> None:
    """`asdict()` açarı qaytarır — buna görə serializasiya ağ siyahıdır."""
    import dataclasses

    leaked = dataclasses.asdict(make_settings())
    assert "sk-test-fake-openai-key-0000000000" in leaked["openai_api_key"]
    assert "openai_api_key" not in make_settings().public_dict()


def test_config_hash_ACAR_deyerinden_ASILI_DEYIL() -> None:
    a = make_settings(openai_api_key="sk-aaa", anthropic_api_key="sk-ant-aaa")
    b = make_settings(openai_api_key="sk-bbb", anthropic_api_key="sk-ant-bbb")
    assert a.config_hash == b.config_hash


def test_config_hash_MODEL_deyisende_deyisir() -> None:
    a = make_settings(judge_model="claude-opus-5")
    b = make_settings(judge_model="claude-sonnet-5")
    assert a.config_hash != b.config_hash


@pytest.mark.parametrize(
    "field_name, placeholder, method",
    [
        ("openai_api_key", OPENAI_PLACEHOLDER, "require_openai_key"),
        ("anthropic_api_key", ANTHROPIC_PLACEHOLDER, "require_anthropic_key"),
    ],
)
def test_placeholder_acar_UC_ADDIMLI_duzelis_verir(
    field_name: str, placeholder: str, method: str
) -> None:
    settings = make_settings(**{field_name: placeholder})
    with pytest.raises(ConfigError) as excinfo:
        getattr(settings, method)()
    message = str(excinfo.value)
    assert "1)" in message and "2)" in message and "3)" in message


def test_bos_acar_da_tutulur() -> None:
    with pytest.raises(ConfigError):
        make_settings(anthropic_api_key="").require_anthropic_key()


def test_live_secrets_placeholderi_ATIR() -> None:
    settings = make_settings(openai_api_key=OPENAI_PLACEHOLDER)
    assert settings.openai_api_key not in settings.live_secrets
    assert settings.anthropic_api_key in settings.live_secrets


def test_judge_max_tokens_DUSUNCE_ucun_asagi_hedd() -> None:
    """max_tokens düşüncəni də əhatə edir — kiçik dəyər boş cavab verir."""
    with pytest.raises(ConfigError, match="düşüncə"):
        make_settings(judge_max_tokens=300)


def test_namelum_effort_reddedilir() -> None:
    with pytest.raises(ConfigError, match="JUDGE_EFFORT"):
        make_settings(judge_effort="ultra")


def test_namelum_grounding_mode_reddedilir() -> None:
    with pytest.raises(ConfigError, match="GROUNDING_MODE"):
        make_settings(grounding_mode="sehrli")


def test_namelum_settings_sahesi_AYDIN_xeta_verir() -> None:
    with pytest.raises(ConfigError, match="Tanınmayan"):
        Settings.load(bele_bir_sahe_yoxdur=1)


# --- qiymət cədvəli -------------------------------------------------------


def test_qiymet_cedvelinde_OLMAYAN_model_SERT_xetadir(settings: Settings) -> None:
    table = load_prices(settings.prices_path)
    with pytest.raises(ConfigError, match="qiymət cədvəlində yoxdur"):
        table.price_for("uydurma-model-9")


def test_keslenmis_token_ONDA_BIR_qiymetle_hesablanir() -> None:
    price = ModelPrice(
        provider="anthropic",
        input_per_mtok=5.0,
        output_per_mtok=25.0,
        cache_read_multiplier=0.1,
    )
    tam = price.cost_usd(1_000_000, 0)
    kes = price.cost_usd(0, 0, cached_read_tokens=1_000_000)
    assert tam == pytest.approx(5.0)
    assert kes == pytest.approx(0.5)


def test_kes_yazisi_bir_qat_iyirmi_bes_faizdir() -> None:
    price = ModelPrice(
        provider="anthropic",
        input_per_mtok=5.0,
        output_per_mtok=25.0,
        cache_write_multiplier=1.25,
    )
    assert price.cost_usd(0, 0, cached_write_tokens=1_000_000) == pytest.approx(6.25)


def test_tesdiqlenmemis_modeller_SADALANIR(settings: Settings) -> None:
    table = load_prices(settings.prices_path)
    assert "gpt-4o-mini" in table.unverified
    assert "claude-opus-5" not in table.unverified


# --- retrieval override-ları (CP7) -----------------------------------------


def test_retrieval_override_TEYIN_OLUNMAYIB_deyilse_sozlukde_yoxdur() -> None:
    """Boş override = «SUT öz dəyərini işlətsin», uydurulmuş rəqəm deyil."""
    assert make_settings().retrieval_overrides() == {}


def test_retrieval_override_SUT_sahe_adlarina_cevrilir() -> None:
    """Açar adları `rag.config.Settings` sahələri ilə ÜST-ÜSTƏ düşməlidir.

    Səhv ad `replace()`-də TypeError verər — amma yalnız canlı run zamanı,
    yəni açar yoxlanıb pul xərclənməyə başlayandan sonra.
    """
    s = make_settings(
        retrieval_top_k=6,
        retrieval_threshold=0.35,
        retrieval_soft_floor_margin=0.15,
    )
    assert s.retrieval_overrides() == {
        "top_k": 6,
        "relevance_threshold": 0.35,
        "soft_floor_margin": 0.15,
    }


def test_retrieval_override_QISMEN_verile_biler() -> None:
    s = make_settings(retrieval_top_k=8)
    assert s.retrieval_overrides() == {"top_k": 8}


def test_retrieval_override_YOXDURSA_config_hash_DEYISMIR() -> None:
    """Sxem genişləndi, amma köhnə run-larla müqayisə pozulmamalıdır.

    Retrieval açarları sabit `None` kimi `public_dict()`-ə düşsəydi, hash
    BÜTÜN köhnə run-lar üçün dəyişər və hər müqayisə yalandan
    «konfiqurasiya fərqlidir» deyərdi.
    """
    payload = make_settings().public_dict()
    assert not [k for k in payload if k.startswith("retrieval_")]


def test_retrieval_override_VERILENDE_config_hash_deyisir() -> None:
    base = make_settings()
    tuned = make_settings(retrieval_threshold=0.30)
    assert tuned.config_hash != base.config_hash
    assert tuned.public_dict()["retrieval_relevance_threshold"] == 0.30


def test_retrieval_threshold_MESAFE_skalasinda_verilse_reddedilir() -> None:
    """0-1 kosinus şkalası. 1.8 kimi rəqəm SUT-da «heç nə keçmir» olur."""
    with pytest.raises(ConfigError, match="kosinus"):
        make_settings(retrieval_threshold=1.8)


def test_retrieval_top_k_SIFIR_reddedilir() -> None:
    with pytest.raises(ConfigError, match="RETRIEVAL_TOP_K"):
        make_settings(retrieval_top_k=0)


def test_retrieval_soft_floor_margin_MENFI_reddedilir() -> None:
    with pytest.raises(ConfigError, match="SOFT_FLOOR_MARGIN"):
        make_settings(retrieval_soft_floor_margin=-0.05)
