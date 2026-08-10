"""Test dəstinin sxemi, örtük qaydası və hash davranışı."""

from __future__ import annotations

import pytest
import yaml

from eval.dataset import (
    CATEGORIES,
    Dataset,
    dataset_hash,
    load_dataset,
    validate_dataset,
)
from eval.errors import DatasetError
from tests.conftest import PROJECT_ROOT, make_case


@pytest.fixture(scope="module")
def real_dataset() -> Dataset:
    return load_dataset(PROJECT_ROOT / "data" / "testset.yaml")


def test_dest_YIRMI_case_dir(real_dataset: Dataset) -> None:
    """Tapşırıq 15-20 sual tələb edir; 20 seçilib."""
    assert len(real_dataset) == 20
    assert len(real_dataset.subset("dev")) == 12
    assert len(real_dataset.subset("holdout")) == 8


def test_id_ler_TEKRARLANMIR(real_dataset: Dataset) -> None:
    assert len(real_dataset.ids()) == len(real_dataset)


def test_her_kateqoriya_TANINIR(real_dataset: Dataset) -> None:
    assert set(real_dataset.categories()) <= CATEGORIES


def test_normal_VE_kenar_hallar_hamisi_var(real_dataset: Dataset) -> None:
    """Tapşırıq 'normal + kənar hallar' tələb edir."""
    present = set(real_dataset.categories())
    for required in (
        "normal",
        "out_of_corpus",
        "false_premise",
        "ambiguous",
        "multi_hop",
        "prompt_injection",
        "language_mixed",
        "numeric_precision",
    ):
        assert required in present, f"'{required}' kateqoriyası yoxdur"


def test_hakim_case_lerinin_RUBRIKASI_var(real_dataset: Dataset) -> None:
    for case in real_dataset:
        if case.wants_judge:
            assert case.expected.rubric, f"{case.id}: rubrika yoxdur"


def test_deterministik_case_lerin_IDDIASI_var(real_dataset: Dataset) -> None:
    for case in real_dataset:
        if case.wants_deterministic:
            assert case.expected.has_assertion, f"{case.id}: iddiasız test heç nə ölçmür"


def test_bolgu_ortuyu_HER_IKI_bolgude(real_dataset: Dataset) -> None:
    counts = {cat: len(cases) for cat, cases in real_dataset.categories().items()}
    for cat, cases in real_dataset.categories().items():
        if counts[cat] >= 2:
            assert {c.split for c in cases} == {"dev", "holdout"}, cat


def test_ortuk_istisnalari_SEBEBI_ile_qeyd_olunub(real_dataset: Dataset) -> None:
    """Tək üzvlü kateqoriya istisnadır — səbəbi `note`-da yazılmalıdır."""
    counts = {cat: len(cases) for cat, cases in real_dataset.categories().items()}
    for cat, cases in real_dataset.categories().items():
        if counts[cat] == 1:
            assert "istisna" in cases[0].note.lower(), (
                f"{cases[0].id}: tək üzvlü kateqoriya, amma səbəb qeyd olunmayıb"
            )


def test_injection_markerleri_SUT_un_HEQIQI_promptundandir(real_dataset: Dataset) -> None:
    """Uydurma sızma markeri heç nə sübut etməzdi."""
    prompt = (
        PROJECT_ROOT / "vendor" / "week2-rag-document-qa" / "rag" / "pipeline.py"
    ).read_text(encoding="utf-8")
    for case in real_dataset:
        if case.category == "prompt_injection":
            assert case.expected.not_contains
            for marker in case.expected.not_contains:
                assert marker in prompt, f"'{marker}' SUT prompt-unda yoxdur"


def test_MEZUNIYYET_suali_deste_DAXIL_DEYIL(real_dataset: Dataset) -> None:
    """SUT-un system prompt-u '21 iş günü' faktını nümunə kimi ehtiva edir.

    Həmin sual retrieval-ı deyil, system prompt-u sınayardı — buna görə
    qəsdən çıxarılıb. Bu test onun geri qayıtmasının qarşısını alır.
    """
    for case in real_dataset:
        assert "məzuniyyət" not in case.question.lower(), (
            f"{case.id}: bu fakt SUT-un system prompt-unda nümunə kimi var, "
            "retrieval-ı sınamır"
        )


def test_hash_YAML_formatina_hessas_DEYIL(tmp_path) -> None:
    source = (PROJECT_ROOT / "data" / "testset.yaml").read_text(encoding="utf-8")
    original = load_dataset(PROJECT_ROOT / "data" / "testset.yaml")

    # Şərhləri at, sıralamanı və sitat üslubunu dəyiş — semantika eynidir.
    reformatted = tmp_path / "reformat.yaml"
    data = yaml.safe_load(source)
    data["cases"] = list(reversed(data["cases"]))
    reformatted.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=True), encoding="utf-8"
    )
    assert load_dataset(reformatted).sha256 == original.sha256


def test_hash_SEMANTIK_deyisikliyi_TUTUR() -> None:
    a = [make_case("c1", question="Birinci sual?")]
    b = [make_case("c1", question="Birinci sual!")]
    assert dataset_hash(a) != dataset_hash(b)


def test_note_hash_e_TESIR_ETMIR() -> None:
    """`note` insan üçün şərhdir; redaktəsi run-ları etibarsızlaşdırmamalıdır."""
    from dataclasses import replace

    base = make_case("c1")
    assert dataset_hash([base]) == dataset_hash([replace(base, note="yeni şərh")])


# --- validasiya qaydaları -------------------------------------------------


def test_tekrarlanan_id_reddedilir() -> None:
    with pytest.raises(DatasetError, match="Təkrarlanan"):
        validate_dataset([make_case("eyni"), make_case("eyni")])


def test_iddiasiz_deterministik_case_reddedilir() -> None:
    with pytest.raises(DatasetError, match="heç bir iddia yoxdur"):
        validate_dataset([make_case("c", gradable="deterministic")])


def test_rubrikasiz_hakim_case_reddedilir() -> None:
    with pytest.raises(DatasetError, match="rubric"):
        validate_dataset([make_case("c", gradable="judge")])


def test_rubrika_GOZLENILEN_cavabi_ehtiva_ede_bilmez() -> None:
    """Hakimə cavab vərəqəsi verilməməlidir."""
    case = make_case(
        "c",
        gradable="both",
        contains=["atlas-cache-edge"],
        rubric="Cavabda atlas-cache-edge adı olmalıdır.",
    )
    with pytest.raises(DatasetError, match="cavab vərəqəsi"):
        validate_dataset([case])


def test_rubrika_GOZLENILEN_ededi_ehtiva_ede_bilmez() -> None:
    case = make_case(
        "c",
        gradable="both",
        numeric=[{"value": 128, "unit": "GB"}],
        rubric="Cavabda 128 rəqəmi olmalıdır.",
    )
    with pytest.raises(DatasetError, match="cavab vərəqəsi"):
        validate_dataset([case])


def test_korpusda_olmayan_gold_source_reddedilir() -> None:
    with pytest.raises(DatasetError, match="korpusda yoxdur"):
        validate_dataset(
            [make_case("c", gold_sources=["uydurma.md"], numeric=[{"value": 1}])]
        )


def test_korpusdan_kenar_sualin_gold_source_u_ola_bilmez() -> None:
    with pytest.raises(DatasetError, match="gold_sources"):
        validate_dataset(
            [
                make_case(
                    "c",
                    category="out_of_corpus",
                    kind="refusal",
                    gold_sources=["atlas_api_senedi.md"],
                )
            ]
        )


def test_imtina_case_i_contains_dasiya_bilmez() -> None:
    with pytest.raises(DatasetError, match="imtina"):
        validate_dataset(
            [make_case("c", category="out_of_corpus", kind="refusal", contains=["x"], gold_sources=[])]
        )


def test_ortuk_qaydasi_pozulanda_xeta() -> None:
    cases = [
        make_case("a", category="multi_hop", split="dev", numeric=[{"value": 1}]),
        make_case("b", category="multi_hop", split="dev", numeric=[{"value": 2}]),
    ]
    with pytest.raises(DatasetError, match="Bölgü örtüyü"):
        validate_dataset(cases)
