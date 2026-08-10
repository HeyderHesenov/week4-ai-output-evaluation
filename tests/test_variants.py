"""Prompt variantları — SUT-a toxunmadan çevirmə, sitat tələsi, idempotentlik."""

from __future__ import annotations

import pytest

from eval.errors import ConfigError
from eval.variants import (
    IDENTITY_VARIANT,
    FewShotExample,
    PromptVariant,
    load_variant,
    load_variants,
    variant_sentinel,
)

SYSTEM = "Sən sual-cavab köməkçisisən."


def variant(**overrides) -> PromptVariant:
    defaults = dict(
        id="v1",
        label="Test",
        system_suffix="Ədədi faktı mənbədəki dəqiqliklə yaz.",
        few_shot=(
            FewShotExample(
                user="Nümunə sual?",
                assistant="Nümunə cavab [1].",
                source_case_ids=("dev_a",),
            ),
        ),
    )
    defaults.update(overrides)
    return PromptVariant(**defaults)


def messages() -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Sual?"},
    ]


# --- tətbiq -----------------------------------------------------------------


def test_system_suffix_ve_few_shot_elave_olunur() -> None:
    result = variant().apply(messages())

    assert SYSTEM in result[0]["content"]
    assert "Ədədi faktı" in result[0]["content"]
    assert result[1] == {"role": "user", "content": "Nümunə sual?"}
    assert result[2] == {"role": "assistant", "content": "Nümunə cavab [1]."}
    assert result[3] == {"role": "user", "content": "Sual?"}


def test_ARQUMENT_hec_vaxt_deyisdirilmir() -> None:
    """`RagPipeline` `messages`-i korreksiya arasında YERİNDƏ genişləndirir.

    Arqument dəyişdirilsəydi, 2-ci cəhd ikiqat çevrilmiş girişlə gedərdi.
    """
    original = messages()
    snapshot = [dict(m) for m in original]

    variant().apply(original)

    assert original == snapshot


def test_sentinel_IKIQAT_tetbiqi_dayandirir() -> None:
    once = variant().apply(messages())
    twice = variant().apply(once)

    assert twice == once
    assert once[0]["content"].count(variant_sentinel("v1")) == 1


def test_baseline_variant_HEC_NE_deyismir() -> None:
    original = messages()
    assert IDENTITY_VARIANT.apply(original) == original
    assert IDENTITY_VARIANT.is_identity is True


def test_system_mesaji_yoxdursa_TOXUNULMUR() -> None:
    """Grounding hakiminin çağırışı bu yolla qorunur."""
    other = [{"role": "user", "content": "Sual?"}]
    assert variant().apply(other) == other


def test_bos_siyahi_ile_isleyir() -> None:
    assert variant().apply([]) == []


def test_yalniz_suffiks_olan_variant() -> None:
    result = PromptVariant(id="v2", label="Yalnız suffiks", system_suffix="Qısa cavab ver.").apply(
        messages()
    )
    assert len(result) == 2
    assert "Qısa cavab ver." in result[0]["content"]


# --- sitat tələsi -----------------------------------------------------------


def test_few_shot_YALNIZ_bir_istinadi_isledir() -> None:
    """SUT sitat nömrəsini HƏMİN sorğunun label-larına qarşı yoxlayır.

    Aralıqdan kənar nömrə invalid_citation → yenidən generasiya → imtina
    zəncirini işə salır; pass-rate düşməsi səhvən prompt məzmununa yazılardı.
    """
    bad = PromptVariant(
        id="v3",
        label="Sitat tələsi",
        few_shot=(FewShotExample(user="Sual?", assistant="Cavab [2]."),),
    )
    with pytest.raises(ConfigError, match=r"\[1\]"):
        bad.validate()


def test_bir_istinadli_numune_QEBUL_edilir() -> None:
    variant().validate()


def test_bos_numune_reddedilir() -> None:
    bad = PromptVariant(
        id="v4", label="Boş", few_shot=(FewShotExample(user="", assistant="Cavab [1]."),)
    )
    with pytest.raises(ConfigError, match="boş"):
        bad.validate()


def test_bos_id_reddedilir() -> None:
    with pytest.raises(ConfigError, match="id"):
        PromptVariant(id="  ", label="x").validate()


# --- kimlik və mənşə --------------------------------------------------------


def test_sha256_MEZMUNDAN_asilidir() -> None:
    a = variant()
    b = variant(system_suffix="Başqa qayda.")
    c = variant(label="Başqa etiket")

    assert a.sha256 != b.sha256
    assert a.sha256 == c.sha256, "etiket qiymətləndirməyə təsir etmir"
    assert len(a.sha256) == 64


def test_mense_id_leri_toplanir() -> None:
    v = PromptVariant(
        id="v5",
        label="İki nümunə",
        few_shot=(
            FewShotExample(user="a", assistant="c [1].", source_case_ids=("dev_a",)),
            FewShotExample(user="b", assistant="c [1].", source_case_ids=("dev_b", "dev_a")),
        ),
    )
    assert v.source_case_ids == frozenset({"dev_a", "dev_b"})


def test_metn_fraqmentleri_CIRKLENME_yoxlamasi_ucun_ayrilir() -> None:
    v = variant(system_suffix="Birinci cümlə. İkinci cümlə.")
    fragments = v.text_fragments()

    assert "Birinci cümlə." in fragments
    assert "İkinci cümlə." in fragments
    assert "Nümunə sual?" in fragments


# --- YAML yüklənməsi --------------------------------------------------------


def test_yaml_dan_yuklenir(tmp_path) -> None:
    path = tmp_path / "v1_test.yaml"
    path.write_text(
        "id: v1_test\n"
        "label: Test variantı\n"
        "description: Təsvir\n"
        "system_suffix: Ədədi faktı dəqiq yaz.\n"
        "few_shot:\n"
        "  - user: Sual?\n"
        "    assistant: Cavab [1].\n"
        "    source_case_ids: [dev_a]\n",
        encoding="utf-8",
    )
    v = load_variant(path)

    assert v.id == "v1_test"
    assert v.system_suffix == "Ədədi faktı dəqiq yaz."
    assert v.few_shot[0].source_case_ids == ("dev_a",)


def test_yaml_daki_sitat_telesi_YUKLEMEDE_tutulur(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "id: bad\nlabel: Pis\nfew_shot:\n  - user: Sual?\n    assistant: Cavab [3].\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_variant(path)


def test_variantlar_qovluqdan_yuklenir_ve_baseline_HEMISE_var(tmp_path) -> None:
    (tmp_path / "v1.yaml").write_text("id: v1\nlabel: Bir\n", encoding="utf-8")
    (tmp_path / "v2.yaml").write_text("id: v2\nlabel: İki\n", encoding="utf-8")

    variants = load_variants(tmp_path)
    assert set(variants) == {"baseline", "v1", "v2"}


def test_olmayan_qovluqda_yalniz_baseline(tmp_path) -> None:
    assert set(load_variants(tmp_path / "yoxdur")) == {"baseline"}


def test_tekrarlanan_variant_id_si_reddedilir(tmp_path) -> None:
    (tmp_path / "a.yaml").write_text("id: eyni\nlabel: A\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("id: eyni\nlabel: B\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Təkrarlanan"):
        load_variants(tmp_path)
