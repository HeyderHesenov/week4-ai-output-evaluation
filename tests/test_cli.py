"""CLI — exit kodları və mühafizələrin uçdan-uca davranışı.

Bu testlər HƏQİQİ `data/testset.yaml` üzərində işləyir, amma bütün yazılan
yollar `tmp_path`-a yönləndirilir: nə repo faylları dəyişir, nə şəbəkəyə
çıxılır.
"""

from __future__ import annotations

import json

import pytest

from eval.cli import load_human_labels, load_label_sources, main
from eval.errors import EvalError
from eval.metrics import LabelSource
from tests.conftest import PROJECT_ROOT


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Bütün yazılan yolları tmp_path-a yönləndirir."""
    monkeypatch.setenv("SPLIT_MANIFEST_PATH", str(tmp_path / "split_manifest.json"))
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("VARIANTS_DIR", str(tmp_path / "variants"))
    monkeypatch.setenv("SUT_COMMIT", "0" * 40)
    return tmp_path


# --- exit kodları -----------------------------------------------------------


def test_seal_split_ugurla_isleyir(sandbox, capsys) -> None:
    assert main(["seal-split"]) == 0

    manifest = json.loads((sandbox / "split_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["dev_ids"]) == 12
    assert len(manifest["holdout_ids"]) == 8
    assert len(manifest["dataset_sha256"]) == 64


def test_seal_split_TEKRAR_cagirisda_dinc_kecir(sandbox, capsys) -> None:
    main(["seal-split"])
    capsys.readouterr()
    assert main(["seal-split"]) == 0
    assert "artıq möhürlənib" in capsys.readouterr().out


def test_deyismis_hash_CIRKLENME_kodu_verir(sandbox, capsys) -> None:
    """Möhürlənmiş hash ilə cari dəst uyğun gəlmirsə, exit 2."""
    main(["seal-split"])
    path = sandbox / "split_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["dataset_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert main(["seal-split"]) == 2
    assert "ÇİRKLƏNMƏ" in capsys.readouterr().err


def test_force_ile_yeniden_mohurlemek_mumkundur(sandbox) -> None:
    main(["seal-split"])
    path = sandbox / "split_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["dataset_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert main(["seal-split", "--force"]) == 0
    rewritten = json.loads(path.read_text(encoding="utf-8"))
    assert rewritten["dataset_sha256"] != "0" * 64


def test_optimize_holdout_ile_EXIT_2(sandbox, capsys) -> None:
    """Struktur ayrılıq: optimallaşdırma holdout-u fiziki olaraq oxuya bilmir."""
    main(["seal-split"])
    assert main(["optimize", "--split", "holdout"]) == 2

    stderr = capsys.readouterr().err
    assert "ÇİRKLƏNMƏ" in stderr
    assert "dövri validasiya" in stderr


def test_namelum_variant_EXIT_1(sandbox, capsys) -> None:
    main(["seal-split"])
    assert main(["run", "--variant", "yoxdur"]) == 1
    assert "tapılmadı" in capsys.readouterr().err


def test_mohursuz_bolgu_ile_run_EXIT_2(sandbox, capsys) -> None:
    assert main(["run"]) == 2
    assert "seal-split" in capsys.readouterr().err


def test_olmayan_run_hesabati_EXIT_1(sandbox, capsys) -> None:
    assert main(["report", "yoxdur"]) == 1
    assert "list-runs" in capsys.readouterr().err


def test_olmayan_run_reclassify_EXIT_1(sandbox) -> None:
    assert main(["reclassify", "yoxdur"]) == 1


def test_list_runs_bos_qovluqda_isleyir(sandbox, capsys) -> None:
    assert main(["list-runs"]) == 0
    assert "Hələ run yoxdur" in capsys.readouterr().out


# --- insan etiketləri -------------------------------------------------------


def test_insan_etiketleri_oxunur(tmp_path) -> None:
    path = tmp_path / "human_labels.yaml"
    path.write_text("labels:\n  dev_a: 3\n  hold_a: 1\n", encoding="utf-8")
    assert load_human_labels(path) == {"dev_a": 3, "hold_a": 1}


def test_olmayan_etiket_fayli_BOS_qaytarir(tmp_path) -> None:
    assert load_human_labels(tmp_path / "yoxdur.yaml") == {}


def test_SKALADAN_KENAR_bal_xeta_verir(tmp_path) -> None:
    """Hakim 0-3 şkalasındadır; başqa şkalada etiket kappanı səssizcə pozar."""
    path = tmp_path / "human_labels.yaml"
    path.write_text("labels:\n  dev_a: 5\n", encoding="utf-8")
    with pytest.raises(EvalError, match="0-3"):
        load_human_labels(path)


def test_ede_bilmeyen_bal_xeta_verir(tmp_path) -> None:
    path = tmp_path / "human_labels.yaml"
    path.write_text("labels:\n  dev_a: yaxsi\n", encoding="utf-8")
    with pytest.raises(EvalError, match="tam ədəd"):
        load_human_labels(path)


def test_etiket_menbeleri_oxunur(tmp_path) -> None:
    path = tmp_path / "human_labels.yaml"
    path.write_text(
        "sources:\n"
        "  dev_a: {run_id: r1, repeat: 2}\n"
        "  hold_a: {run_id: r2}\n"
        "labels:\n  dev_a: 3\n",
        encoding="utf-8",
    )
    got = load_label_sources(path)
    assert got == {"dev_a": LabelSource("r1", 2), "hold_a": LabelSource("r2", 1)}


def test_etiket_menbesi_FORMATI_yanlisdirsa_xeta(tmp_path) -> None:
    path = tmp_path / "human_labels.yaml"
    path.write_text("sources:\n  dev_a: 20260810T-dev\n", encoding="utf-8")
    with pytest.raises(EvalError, match="run_id"):
        load_label_sources(path)


def test_LAYIHE_etiket_faylinin_doldurulma_mexanizmi_ISLEYIR(tmp_path) -> None:
    """Şərhdən çıxarılan `# case_id: 0` sətri HƏQİQƏTƏN `labels`-a düşür.

    Bu test regressiya üçündür: əvvəlki şablonda bal sətirləri `labels:`
    blokundan kənarda idi və təlimata əməl edən istifadəçi faylı doldursa
    da, `load_human_labels` boş qaytarırdı — kappa səssizcə hesablanmırdı.

    Mexanizm SİNTETİK şablon üzərində yoxlanır: real fayl artıq
    doldurulub (2026-08-12) və testin onun boş qalmasından asılı olması
    ölçünün gedişini testə bağlayardı.
    """
    template = (
        "sources:\n"
        "  dev_a: {run_id: R1, repeat: 1}\n"
        "\n"
        "labels:\n"
        "\n"
        "  #   SUAL: ...\n"
        "  # dev_a: 0\n"
    )
    path = tmp_path / "template.yaml"
    path.write_text(template, encoding="utf-8")
    assert load_human_labels(path) == {}, "şərhdə qalan sətir bal sayılmamalıdır"

    opened = tmp_path / "opened.yaml"
    opened.write_text(template.replace("  # dev_a: 0", "  dev_a: 2"), encoding="utf-8")
    assert load_human_labels(opened) == {"dev_a": 2}


def test_LAYIHE_doldurulmus_etiketler_menbeleri_ile_uzlasir() -> None:
    """Real fayl: hər etiketin `sources` qeydi var və bal şkaladadır.

    Bu, doldurulmadan sonrakı yeganə struktur zəmanətidir. Balın DƏYƏRİ
    burada yoxlanmır — o, müəllifin qərarıdır, testin işi deyil.
    """
    path = PROJECT_ROOT / "data" / "human_labels.yaml"
    sources = load_label_sources(path)
    labels = load_human_labels(path)
    assert len(sources) == 9, "fayl 9 case üçündür"

    orphan = sorted(set(labels) - set(sources))
    assert not orphan, f"`sources` qeydi olmayan etiket: {orphan}"
    off_scale = {c: v for c, v in labels.items() if v not in (0, 1, 2, 3)}
    assert not off_scale, f"şkaladan kənar bal (0-3 gözlənilir): {off_scale}"


# --- kömək mətnləri ---------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    ["seal-split", "run", "optimize", "reclassify", "report", "judge-bias", "list-runs"],
)
def test_her_alt_emrin_komeyi_var(command, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == 0


def test_hakim_prompt_hashi_manifestde_saxlanilir() -> None:
    """Sənədləşdirmə testi: hakim promptunun kimliyi kod bazasındadır."""
    from eval.judge import judge_prompt_sha256

    assert len(judge_prompt_sha256()) == 64
    assert (PROJECT_ROOT / "eval" / "judge.py").exists()
