"""dev/holdout intizamı — beş mexanizmin hər biri ayrıca.

Bu testlər layihənin ən vacib iddiasını qoruyur: «holdout optimallaşdırma
zamanı HEÇ VAXT görünmədi». İddia yalnız onu pozmağın mümkün olmadığı
test ilə sübut oluna bilər.
"""

from __future__ import annotations

import pytest

from eval.errors import ContaminationError
from eval.split import (
    JACCARD_THRESHOLD,
    ContaminationGuard,
    LedgerEntry,
    append_holdout_ledger,
    jaccard,
    load_split_manifest,
    read_holdout_ledger,
    seal_split,
    write_split_manifest,
)
from eval.variants import IDENTITY_VARIANT, FewShotExample, PromptVariant
from tests.conftest import make_case, make_dataset

GOLD = "atlas_api_senedi.md"


def build_dataset(holdout_question: str = "Xarici ezamiyyət gündəliyi neçə manatdır?"):
    return make_dataset(
        [
            make_case("dev_a", split="dev", question="Free planında limit nədir?",
                      gold_sources=[GOLD], numeric=[{"value": 60}]),
            make_case("dev_b", split="dev", question="Pro planı SLA faizi nədir?",
                      gold_sources=[GOLD], numeric=[{"value": 99.9}]),
            make_case("hold_a", split="holdout", question=holdout_question,
                      gold_sources=[GOLD], numeric=[{"value": 90}]),
        ]
    )


def build_guard(dataset=None) -> ContaminationGuard:
    dataset = dataset or build_dataset()
    manifest = seal_split(dataset, sealed_at="2026-08-10T00:00:00Z", sealed_by_commit="a" * 40)
    return ContaminationGuard(manifest, dataset)


# --- 1. möhürlənmiş bölgü ---------------------------------------------------


def test_dest_deyisibse_run_DAYANIR() -> None:
    original = build_dataset()
    manifest = seal_split(original, sealed_at="x", sealed_by_commit="y")
    edited = build_dataset(holdout_question="Redaktə edilmiş holdout sualı?")

    guard = ContaminationGuard(manifest, edited)
    with pytest.raises(ContaminationError, match="dəyişdirilib"):
        guard.assert_dataset_unchanged()


def test_dest_deyismeyibse_kecir() -> None:
    build_guard().assert_dataset_unchanged()


def test_manifest_disk_ile_GEDIS_GELIS_edir(tmp_path) -> None:
    dataset = build_dataset()
    manifest = seal_split(dataset, sealed_at="2026-08-10", sealed_by_commit="a" * 40)
    path = tmp_path / "split_manifest.json"

    write_split_manifest(manifest, path)
    loaded = load_split_manifest(path)

    assert loaded == manifest
    assert loaded.sha256 == manifest.sha256


def test_olmayan_manifest_SEAL_SPLITE_yonlendirir(tmp_path) -> None:
    with pytest.raises(ContaminationError, match="seal-split"):
        load_split_manifest(tmp_path / "yoxdur.json")


# --- 2. mənşə yoxlaması -----------------------------------------------------


def test_holdout_mensesi_olan_variant_REDD_edilir() -> None:
    variant = PromptVariant(
        id="v1",
        label="Sızmış",
        few_shot=(
            FewShotExample(user="s", assistant="c [1].", source_case_ids=("hold_a",)),
        ),
    )
    with pytest.raises(ContaminationError, match="hold_a"):
        build_guard().assert_variant_clean(variant)


def test_namelum_case_id_si_de_REDD_edilir() -> None:
    variant = PromptVariant(
        id="v1",
        label="Uydurma mənşə",
        few_shot=(
            FewShotExample(user="s", assistant="c [1].", source_case_ids=("uydurma_id",)),
        ),
    )
    with pytest.raises(ContaminationError, match="naməlum"):
        build_guard().assert_variant_clean(variant)


def test_dev_mensesi_QEBUL_edilir() -> None:
    variant = PromptVariant(
        id="v1",
        label="Təmiz",
        system_suffix="Ədədi faktı mənbədəki dəqiqliklə yaz.",
        few_shot=(
            FewShotExample(user="Sual?", assistant="Cavab [1].", source_case_ids=("dev_a",)),
        ),
    )
    build_guard().assert_variant_clean(variant)


def test_baseline_variant_yoxlanmadan_kecir() -> None:
    assert build_guard().assert_variant_clean(IDENTITY_VARIANT) == ()


# --- 3. parafraz sızması ----------------------------------------------------


def test_holdout_sualinin_PARAFRAZI_tutulur() -> None:
    """Elan edilmiş mənşə təmiz olsa belə, mətn holdout-dan törəmiş ola bilər."""
    variant = PromptVariant(
        id="v1",
        label="Parafraz",
        few_shot=(
            FewShotExample(
                user="Xarici ezamiyyət gündəliyi neçə manatdır?",
                assistant="Cavab [1].",
                source_case_ids=("dev_a",),
            ),
        ),
    )
    with pytest.raises(ContaminationError, match="Jaccard"):
        build_guard().assert_variant_clean(variant)


def test_ferqli_movzulu_metn_KECIR_ve_olculur() -> None:
    variant = PromptVariant(
        id="v1",
        label="Təmiz",
        few_shot=(
            FewShotExample(
                user="Keş qatı neçə zonada işləyir?",
                assistant="Üç zonada [1].",
                source_case_ids=("dev_a",),
            ),
        ),
    )
    hits = build_guard().assert_variant_clean(variant)

    assert hits, "oxşarlıq ÖLÇÜLMƏLİ və qaytarılmalıdır — iddia yox, rəqəm"
    assert max(h.score for h in hits) < JACCARD_THRESHOLD


def test_jaccard_eyni_metnde_bir() -> None:
    assert jaccard("bir iki üç", "bir iki üç") == pytest.approx(1.0)
    assert jaccard("bir iki üç", "") == pytest.approx(0.0)


# --- 4. struktur ayrılıq ----------------------------------------------------


def test_optimize_YALNIZ_dev_ile_isleyir() -> None:
    guard = build_guard()
    guard.assert_split_allowed("dev", action="optimize")

    for split in ("holdout", "all"):
        with pytest.raises(ContaminationError, match="dövri validasiya"):
            guard.assert_split_allowed(split, action="optimize")


def test_adi_run_butun_bolguleri_QEBUL_edir() -> None:
    guard = build_guard()
    for split in ("dev", "holdout", "all"):
        guard.assert_split_allowed(split, action="run")


# --- 5. yalnız-əlavə registr ------------------------------------------------


def test_registr_MOVCUD_qeydleri_saxlayir(tmp_path) -> None:
    path = tmp_path / "holdout_ledger.json"
    first = LedgerEntry(run_id="r1", at="2026-08-10", variant_id="baseline",
                        harness_commit="a", case_count=8)
    second = LedgerEntry(run_id="r2", at="2026-08-11", variant_id="v1",
                         harness_commit="b", case_count=8, note="ikinci")

    append_holdout_ledger(path, first)
    entries = append_holdout_ledger(path, second)

    assert len(entries) == 2
    assert read_holdout_ledger(path) == (first, second)


def test_olmayan_registr_BOS_qaytarir(tmp_path) -> None:
    assert read_holdout_ledger(tmp_path / "yoxdur.json") == ()
