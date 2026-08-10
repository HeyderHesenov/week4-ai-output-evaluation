"""Orkestrasiya — mühafizələrin SIRASI, artefaktlar, registr.

Ən vacib testlər «nə baş verdi» deyil, «nə baş VERMƏDİ» ilə bağlıdır:
çirklənmiş run heç bir pullu çağırış etməməlidir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from eval.artifacts import RunPaths, load_run
from eval.errors import ContaminationError, SutError
from eval.judge import Verdict
from eval.runner import Runner, reclassify
from eval.split import ContaminationGuard, read_holdout_ledger, seal_split
from eval.sut import SutInfo
from eval.variants import IDENTITY_VARIANT, FewShotExample, PromptVariant
from tests.conftest import make_case, make_chunk, make_dataset, make_observation, make_settings

GOLD = "atlas_api_senedi.md"


@dataclass
class FakeSut:
    observations: dict[str, Any] = field(default_factory=dict)
    preflight_error: Exception | None = None
    calls: list[tuple[str, int]] = field(default_factory=list)
    preflights: int = 0

    def preflight(self) -> SutInfo:
        self.preflights += 1
        if self.preflight_error is not None:
            raise self.preflight_error
        return SutInfo(commit="a" * 40, chunk_count=12, sut_path="/fake")

    def observe(self, case, *, repeat: int = 1):
        self.calls.append((case.id, repeat))
        template = self.observations.get(case.id)
        if template is None:
            template = make_observation(case.id, answer_text="Dəqiqədə 60 sorğu [1].")
        return replace(template, case_id=case.id, repeat=repeat)


@dataclass
class FakeJudge:
    score: int | None = 3
    calls: list[str] = field(default_factory=list)

    def judge(self, case, obs) -> Verdict:
        self.calls.append(obs.case_id)
        return Verdict(
            case_id=obs.case_id,
            repeat=obs.repeat,
            score=self.score,
            faithful=True,
            complete=True,
            reason="test",
            flags=(),
            judge_model="claude-opus-5",
            served_model="claude-opus-5",
            judge_prompt_sha256="b" * 64,
            threshold=2,
        )


def build_dataset():
    return make_dataset(
        [
            make_case(
                "dev_a",
                split="dev",
                gold_sources=[GOLD],
                numeric=[{"value": 60, "unit": "sorğu"}],
            ),
            make_case(
                "hold_a",
                split="holdout",
                gold_sources=[GOLD],
                numeric=[{"value": 60, "unit": "sorğu"}],
            ),
        ]
    )


def build_runner(tmp_path, *, dataset=None, sut=None, judge=None, **settings_overrides):
    dataset = dataset or build_dataset()
    settings = make_settings(runs_dir=tmp_path, logs_dir=tmp_path, **settings_overrides)
    manifest = seal_split(dataset, sealed_at="2026-08-10T00:00:00Z", sealed_by_commit="c" * 40)
    guard = ContaminationGuard(manifest, dataset)
    sut = sut or FakeSut()
    judge = judge or FakeJudge()
    runner = Runner(
        settings,
        dataset=dataset,
        guard=guard,
        sut=sut,
        judge=judge,
        now=lambda: "2026-08-10T12:00:00Z",
        harness_commit="d" * 40,
    )
    return runner, sut, judge, settings, dataset


# --- mühafizələr PULLU çağırışdan ƏVVƏL ------------------------------------


def test_optimize_holdouta_TOXUNA_BILMIR(tmp_path) -> None:
    runner, sut, judge, *_ = build_runner(tmp_path)

    with pytest.raises(ContaminationError, match="dev"):
        runner.run(split="holdout", action="optimize")

    assert sut.calls == [], "çirklənmiş run heç bir sorğu etməməli idi"
    assert sut.preflights == 0
    assert judge.calls == []


def test_deyisdirilmis_test_desti_RUNU_dayandirir(tmp_path) -> None:
    dataset = build_dataset()
    settings = make_settings(runs_dir=tmp_path)
    sealed = seal_split(dataset, sealed_at="x", sealed_by_commit="y")

    # Möhürləndikdən SONRA sual redaktə olunub.
    edited = make_dataset(
        [
            make_case("dev_a", question="Dəyişdirilmiş sual?", split="dev",
                      gold_sources=[GOLD], numeric=[{"value": 60}]),
            make_case("hold_a", split="holdout", gold_sources=[GOLD], numeric=[{"value": 60}]),
        ]
    )
    sut = FakeSut()
    runner = Runner(
        settings,
        dataset=edited,
        guard=ContaminationGuard(sealed, edited),
        sut=sut,
        judge=FakeJudge(),
    )

    with pytest.raises(ContaminationError, match="dəyişdirilib"):
        runner.run(split="dev")
    assert sut.calls == []


def test_holdout_id_li_variant_REDD_edilir(tmp_path) -> None:
    runner, sut, *_ = build_runner(tmp_path)
    variant = PromptVariant(
        id="v_bad",
        label="Sızmış variant",
        few_shot=(
            FewShotExample(
                user="Nümunə sual?",
                assistant="Nümunə cavab [1].",
                source_case_ids=("hold_a",),
            ),
        ),
    )

    with pytest.raises(ContaminationError, match="hold_a"):
        runner.run(split="dev", variant=variant)
    assert sut.calls == []


def test_SUT_preflight_xetasi_UGURSUZ_run_deyil_XETADIR(tmp_path) -> None:
    sut = FakeSut(preflight_error=SutError("commit uyğun gəlmir"))
    runner, *_ = build_runner(tmp_path, sut=sut)

    with pytest.raises(SutError):
        runner.run(split="dev")
    assert sut.calls == []


# --- normal icra ------------------------------------------------------------


def test_run_artefaktlari_yazir(tmp_path) -> None:
    runner, _, _, settings, _ = build_runner(tmp_path)
    result = runner.run(split="dev")

    loaded = load_run(RunPaths.for_run(settings.runs_dir, result.run_id))
    assert len(loaded.observations) == 1
    assert len(loaded.grades) == 1
    assert len(loaded.causes) == 1
    assert loaded.manifest["split"] == "dev"


def test_manifest_BUTUN_kimlikleri_dasiyir(tmp_path) -> None:
    runner, _, _, _, dataset = build_runner(tmp_path)
    manifest = runner.run(split="dev").manifest

    assert manifest["dataset_sha256"] == dataset.sha256
    assert manifest["variant_id"] == "baseline"
    assert manifest["harness_commit"] == "d" * 40
    assert manifest["sut_commit"] == "a" * 40
    assert len(manifest["judge_prompt_sha256"]) == 64
    assert manifest["config_hash"]
    assert manifest["max_holdout_similarity"] == 0.0


def test_manifest_ACAR_sizdirmir(tmp_path) -> None:
    runner, _, _, settings, _ = build_runner(tmp_path)
    result = runner.run(split="dev")

    raw = (result.paths.root / "manifest.json").read_text(encoding="utf-8")
    assert "sk-ant-test-fake-anthropic-key-000" not in raw
    assert "sk-test-fake-openai-key" not in raw
    assert json.loads(raw)["config"]["judge_model"] == "claude-opus-5"


def test_tekrarlar_ISLENIR(tmp_path) -> None:
    runner, sut, *_ = build_runner(tmp_path, repeats=3)
    result = runner.run(split="dev")

    assert sut.calls == [("dev_a", 1), ("dev_a", 2), ("dev_a", 3)]
    assert len(result.observations) == 3
    assert result.case_count == 1


# --- hakimin çağırılma qaydası ---------------------------------------------


def test_deterministik_case_HAKIME_gonderilmir(tmp_path) -> None:
    runner, _, judge, *_ = build_runner(tmp_path)
    runner.run(split="dev")
    assert judge.calls == []


def test_SUT_xetasi_olan_musahide_HAKIME_gonderilmir(tmp_path) -> None:
    """Qiymətləndiriləcək cavab yoxdursa, hakim çağırışı pul yandırmaqdır."""
    dataset = make_dataset(
        [
            make_case("dev_j", split="dev", gradable="judge", rubric="Meyar."),
            make_case("hold_j", split="holdout", gradable="judge", rubric="Meyar."),
        ]
    )
    sut = FakeSut(observations={"dev_j": make_observation("dev_j", error="RuntimeError: qopdu")})
    runner, _, judge, *_ = build_runner(tmp_path, dataset=dataset, sut=sut)

    result = runner.run(split="dev")

    assert judge.calls == []
    assert result.causes[0].category == "sut_error"


def test_hakim_case_i_QIYMETLENDIRILIR(tmp_path) -> None:
    dataset = make_dataset(
        [
            make_case("dev_j", split="dev", gradable="judge", rubric="Meyar."),
            make_case("hold_j", split="holdout", gradable="judge", rubric="Meyar."),
        ]
    )
    runner, _, judge, *_ = build_runner(tmp_path, dataset=dataset)
    result = runner.run(split="dev")

    assert judge.calls == ["dev_j"]
    assert result.causes[0].category == "ok"


# --- holdout registri -------------------------------------------------------


def test_holdout_icrasi_REGISTRE_yazilir(tmp_path) -> None:
    runner, _, _, settings, _ = build_runner(tmp_path)
    runner.run(split="holdout", note="ilk kor yoxlama")

    ledger = read_holdout_ledger(settings.runs_dir / "holdout_ledger.json")
    assert len(ledger) == 1
    assert ledger[0].variant_id == "baseline"
    assert ledger[0].note == "ilk kor yoxlama"


def test_registr_YALNIZ_ELAVE_dir(tmp_path) -> None:
    runner, _, _, settings, _ = build_runner(tmp_path)
    runner.run(split="holdout", note="birinci")
    runner.run(split="holdout", note="ikinci")

    ledger = read_holdout_ledger(settings.runs_dir / "holdout_ledger.json")
    assert [e.note for e in ledger] == ["birinci", "ikinci"]


def test_dev_run_u_registre_YAZILMIR(tmp_path) -> None:
    runner, _, _, settings, _ = build_runner(tmp_path)
    runner.run(split="dev")
    assert read_holdout_ledger(settings.runs_dir / "holdout_ledger.json") == ()


# --- yenidən təsnifat -------------------------------------------------------


def test_reclassify_SEBEKESIZ_isleyir(tmp_path) -> None:
    dataset = build_dataset()
    sut = FakeSut(
        observations={
            "dev_a": make_observation(
                "dev_a", answer_text="Yanlış [1].", chunks=[make_chunk(source=GOLD)]
            )
        }
    )
    runner, _, _, settings, _ = build_runner(tmp_path, dataset=dataset, sut=sut)
    result = runner.run(split="dev")

    run = load_run(RunPaths.for_run(settings.runs_dir, result.run_id))
    causes = reclassify(dataset, run.observations, run.grades, run.verdicts)

    assert len(causes) == 1
    assert causes[0].category == result.causes[0].category
    assert sut.calls == [("dev_a", 1)], "yenidən təsnifat SUT-u çağırmamalıdır"
