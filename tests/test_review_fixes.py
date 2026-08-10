"""Code review tapıntılarının regressiya testləri.

Hər test MƏHZ bir tapıntıya bağlıdır və düzəlişdən ƏVVƏL uğursuz olurdu.
Adları uzundur, çünki testin adı tapıntının ifadəsidir: baq geri qayıtsa,
uğursuz testin adı nəyin sındığını izahsız deyir.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.artifacts import RunPaths, load_run
from eval.config import Settings
from eval.dataset import NumericClaim, load_dataset
from eval.errors import ConfigError, DatasetError
from eval.graders import extract_numbers, numeric_claim_satisfied
from eval.metrics import UNMEASURABLE
from eval.report import _outcomes
from eval.rootcause import RootCause
from eval.split import LedgerEntry, append_holdout_ledger, read_holdout_ledger
from eval.variants import load_variants
from tests.conftest import PROJECT_ROOT


# --- graders: ədəd ayrıştırma ------------------------------------------------


def test_SIYAHIDAKI_ededlerin_hamisi_tapilir() -> None:
    """«60, 600 və 6 000» — acgöz nişan üçünü də bir yerdə udub itirirdi."""
    values = [v for v, _, _ in extract_numbers("Limitlər: 60, 600 və 6 000 sorğu")]
    assert values == [60.0, 600.0, 6000.0]


def test_vergulle_ayrilmis_qisa_siyahi_ITMIR() -> None:
    assert [v for v, _, _ in extract_numbers("Zonalar: 1, 2, 3")] == [1.0, 2.0, 3.0]


def test_qrup_ve_onluq_ayiricilari_HELE_DE_isleyir() -> None:
    """Düzəliş köhnə davranışı pozmamalıdır."""
    assert [v for v, _, _ in extract_numbers("6 000 sorğu")] == [6000.0]
    assert [v for v, _, _ in extract_numbers("1.234,56 manat")] == [1234.56]


def test_SAAT_formati_ayri_eded_kimi_cixarilmir() -> None:
    """«01:00» vahidsiz `value=1`/`value=0` iddialarını yanlış təsdiqləyirdi."""
    assert extract_numbers("saat 01:00-da alınır") == ()
    assert not numeric_claim_satisfied(
        "Ehtiyat nüsxə saat 01:00-da alınır", NumericClaim(value=1.0, unit="")
    )


# --- graders: imtina səbəbi --------------------------------------------------


def _refusal_case(reason_in: tuple[str, ...] = ()):
    from eval.dataset import EvalCase, Expected

    return EvalCase(
        id="dev_ref",
        question="Gəlir nə qədərdir?",
        category="out_of_corpus",
        split="dev",
        gradable="deterministic",
        expected=Expected(kind="refusal", reason_in=reason_in),
    )


def _refused_observation(reason: str):
    from tests.conftest import make_observation

    obs = make_observation("dev_ref", answer_text="Sənədlərdə cavab tapılmadı.")
    return type(obs)(**{**vars(obs), "refused": True, "reason": reason})


def test_BOS_INDEKS_imtinasi_defolt_olaraq_KECMIR() -> None:
    """`reason_in` verilməyibsə, empty_index qəbul olunanlar arasında OLMAMALIDIR.

    Boş indeks səbəbindən imtina sistemin işlədiyini deyil, QURULMADIĞINI
    göstərir — `check_refusal`-ın öz docstring-i məhz bunu deyir, amma
    fallback bütün səbəbləri qəbul edib qurulma nasazlığını keçirirdi.
    """
    from eval.graders import check_refusal

    outcome = check_refusal(_refusal_case(), _refused_observation("empty_index"))
    assert outcome is not None and not outcome.ok


def test_ADI_imtina_sebebi_defolt_olaraq_KECIR() -> None:
    """Düzəliş qanuni imtinaları bloklamamalıdır."""
    from eval.graders import check_refusal

    outcome = check_refusal(_refusal_case(), _refused_observation("low_relevance"))
    assert outcome is not None and outcome.ok


# --- report: McNemar ölçülə bilməyənləri geriləmə saymır ----------------------


def _run_with_causes(categories: list[str]):
    class _FakeRun:
        causes = tuple(
            RootCause(case_id=f"c{i}", repeat=1, category=cat, detail="")
            for i, cat in enumerate(categories)
        )

    return _FakeRun()


def test_HAKIM_XETASI_McNemar_da_gerileme_sayilmir() -> None:
    """Bir 429 baş sətirdə «geriləmə» kimi görünürdü."""
    base = _outcomes(_run_with_causes(["ok", "ok"]))
    variant = _outcomes(_run_with_causes(["ok", "judge_error"]))

    from eval.metrics import mcnemar, stable_pass

    result = mcnemar(stable_pass(base), stable_pass(variant))
    assert result.regressed == 0
    assert "judge_error" in UNMEASURABLE


def test_olculebilen_HEQIQI_gerileme_HELE_DE_gorunur() -> None:
    base = _outcomes(_run_with_causes(["ok", "ok"]))
    variant = _outcomes(_run_with_causes(["ok", "generation"]))

    from eval.metrics import mcnemar, stable_pass

    assert mcnemar(stable_pass(base), stable_pass(variant)).regressed == 1


# --- variants: baseline əvəz edilə bilməz -------------------------------------


def test_BASELINE_id_si_fayl_ile_EVEZ_EDILE_BILMEZ(tmp_path) -> None:
    (tmp_path / "zz.yaml").write_text(
        "id: baseline\nlabel: saxta\nsystem_suffix: 'ƏLAVƏ QAYDA'\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="baseline"):
        load_variants(tmp_path)


# --- artifacts: qiymətlər təkrar üzrə ayrılır ---------------------------------


def test_qiymetler_TEKRAR_uzre_ayrilir() -> None:
    run = load_run(
        RunPaths.for_run(PROJECT_ROOT / "runs", "20260810T083111Z-holdout-v1_tam_cavab")
    )
    assert len(run.grade_map()) == len(run.grades), (
        "grade_map case_id üzrə yığsaydı, 21 qiymət 7-yə düşərdi və "
        "reclassify sonuncu təkrarın qiymətini hamısına şamil edərdi"
    )
    assert {g.repeat for g in run.grades} == {1, 2, 3}


# --- dataset: mühafizələr -----------------------------------------------------


@pytest.mark.parametrize("body", ["cases:\n", "cases: 5\n", "cases: 'x'\n"])
def test_bos_ve_ya_skalyar_cases_DATASETERROR_verir(tmp_path, body) -> None:
    """Xam TypeError `cli.main` tərəfindən tutulmur — istifadəçi traceback görür."""
    path = tmp_path / "t.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(DatasetError):
        load_dataset(path)


def test_rubrikadaki_SERBEST_reqem_yalanci_sizma_saymir(tmp_path) -> None:
    """«0-3 şkalası» ifadəsi `value: 3` olan case-i bloklamamalıdır."""
    path = tmp_path / "t.yaml"
    path.write_text(
        "cases:\n"
        "  - id: dev_x\n"
        "    question: 'Neçə zona?'\n"
        "    category: normal\n"
        "    split: dev\n"
        "    gradable: both\n"
        "    gold_sources: [atlas_infra_qeydleri.md]\n"
        "    expected:\n"
        "      kind: answer\n"
        "      numeric:\n"
        "        - value: 3\n"
        "          unit: 'zona'\n"
        "      rubric: 'Bal 0-3 şkalasındadır; cavab zona sayını deməlidir.'\n",
        encoding="utf-8",
    )
    # Bölgü örtüyü yoxlaması ayrı xəta verə bilər; bizi maraqlandıran
    # YALNIZ sızma mühafizəsinin işə düşməməsidir.
    try:
        load_dataset(path)
    except DatasetError as exc:
        assert "gözlənilən ədədi" not in str(exc), exc


def test_rubrika_HEQIQI_ededi_ehtiva_edirse_HELE_DE_bloklanir(tmp_path) -> None:
    path = tmp_path / "t.yaml"
    path.write_text(
        "cases:\n"
        "  - id: dev_x\n"
        "    question: 'Neçə zona?'\n"
        "    category: normal\n"
        "    split: dev\n"
        "    gradable: both\n"
        "    gold_sources: [atlas_infra_qeydleri.md]\n"
        "    expected:\n"
        "      kind: answer\n"
        "      numeric:\n"
        "        - value: 3\n"
        "          unit: 'zona'\n"
        "      rubric: 'Cavab 3 zona deməlidir.'\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="gözlənilən ədədi"):
        load_dataset(path)


# --- config: fallback modeli qabaqcadan yoxlanır ------------------------------


def test_QIYMETSIZ_fallback_modeli_run_dan_EVVEL_xeta_verir(monkeypatch) -> None:
    """Əks halda xəta yalnız hesabatda — pul xərcləndikdən sonra — çıxırdı."""
    monkeypatch.setenv("JUDGE_FALLBACK_MODEL", "model-yoxdur-9")
    monkeypatch.setenv("SUT_COMMIT", "0" * 40)
    with pytest.raises(ConfigError, match="qiymət cədvəlində yoxdur"):
        Settings.load()


# --- config: manifest maşından asılı deyil ------------------------------------


def test_manifest_yolu_REPOYA_NISBI_ve_hash_masindan_asili_deyil(monkeypatch) -> None:
    """Mütləq yol həm istifadəçi adını sızdırırdı, həm hash-i maşına bağlayırdı."""
    monkeypatch.setenv("SUT_COMMIT", "0" * 40)
    public = Settings.load().public_dict()

    assert public["sut_path"] == "vendor/week2-rag-document-qa"
    assert "/Users" not in json.dumps(public), "manifestə lokal yol düşməməlidir"


# --- split: registr atomik yazılır --------------------------------------------


def test_registr_ATOMIK_yazilir_ve_muveqqeti_fayl_qalmir(tmp_path) -> None:
    path = tmp_path / "holdout_ledger.json"
    entry = LedgerEntry(
        run_id="r1", at="2026-08-10T00:00:00Z", variant_id="baseline",
        harness_commit="abc", case_count=8, note="",
    )
    append_holdout_ledger(path, entry)
    append_holdout_ledger(path, LedgerEntry(**{**vars(entry), "run_id": "r2"}))

    assert len(read_holdout_ledger(path)) == 2
    assert not (tmp_path / "holdout_ledger.json.tmp").exists()
    assert json.loads(path.read_text(encoding="utf-8"))[0]["run_id"] == "r1"
