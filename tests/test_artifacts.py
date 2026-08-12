"""Artefaktların yazılması, geri oxunması və açar redaksiyası."""

from __future__ import annotations

import json

import pytest

from eval.artifacts import (
    OBSERVATIONS,
    REDACTED,
    RunArtifacts,
    RunPaths,
    RunWriter,
    list_runs,
    load_run,
    observation_from_json,
    redact,
)
from eval.errors import ArtifactError
from eval.graders import CheckOutcome, GradeResult
from eval.judge import Verdict
from eval.observation import RetrievalCall
from eval.rootcause import RootCause
from tests.conftest import make_chunk, make_llm_call, make_observation

SECRET = "sk-ant-heqiqi-acar-deyeri-12345"


def writer(tmp_path, *, secrets=(SECRET,)) -> RunWriter:
    return RunWriter(RunPaths.for_run(tmp_path, "run-001"), secrets=secrets)


def sample_verdict(**kwargs) -> Verdict:
    defaults = dict(
        case_id="c1",
        repeat=1,
        score=3,
        faithful=True,
        complete=True,
        reason="hər şey qaydasındadır",
        flags=("injection_attempt",),
        judge_model="claude-opus-5",
        served_model="claude-opus-5",
        judge_prompt_sha256="a" * 64,
        threshold=2,
        input_tokens=800,
        output_tokens=90,
        cached_read_tokens=700,
        cached_write_tokens=10,
        latency_ms=1234.5,
        stop_reason="end_turn",
    )
    defaults.update(kwargs)
    return Verdict(**defaults)


# --- gediş-gəliş ------------------------------------------------------------


def test_musahide_TAM_geri_oxunur(tmp_path) -> None:
    obs = make_observation(
        "c1",
        repeat=2,
        answer_text="Cavab [1].",
        chunks=[make_chunk(label=1, text="mənbə mətni", score=0.77)],
        llm_calls=[make_llm_call()],
        retrieval_calls=[
            RetrievalCall(mode="hybrid", k=4, latency_ms=120.0, returned=4, top_score=0.77, query_chars=30)
        ],
        invalid_citations=[7],
        unsupported_claims=["dəstəklənməyən iddia"],
    )
    w = writer(tmp_path)
    w.write_manifest({"run_id": "run-001"})
    w.append_observation(obs)

    loaded = load_run(RunPaths.for_run(tmp_path, "run-001")).observations[0]
    assert loaded == obs


def test_verdikt_ve_qiymet_geri_oxunur(tmp_path) -> None:
    grade = GradeResult(
        case_id="c1",
        passed=False,
        checks=(CheckOutcome(name="numeric", ok=False, detail="tapılmadı: 60"),),
    )
    cause = RootCause(case_id="c1", repeat=1, category="retrieval_miss", detail="gold yoxdur")

    w = writer(tmp_path)
    w.write_manifest({"run_id": "run-001"})
    w.append_grade(grade)
    w.append_verdict(sample_verdict())
    w.append_cause(cause)

    run = load_run(RunPaths.for_run(tmp_path, "run-001"))
    assert run.grades[0] == grade
    assert run.verdicts[0] == sample_verdict()
    assert run.causes[0] == cause


def test_hakim_XETASI_None_bal_ile_geri_oxunur(tmp_path) -> None:
    """`score=None` (ölçülə bilmədi) 0 bala ÇEVRİLMƏMƏLİDİR."""
    w = writer(tmp_path)
    w.write_manifest({})
    w.append_verdict(sample_verdict(score=None, error="hakim imtina etdi"))

    loaded = load_run(RunPaths.for_run(tmp_path, "run-001")).verdicts[0]
    assert loaded.score is None
    assert loaded.ok is False


def test_xeritelerle_axtaris(tmp_path) -> None:
    w = writer(tmp_path)
    w.write_manifest({})
    w.append_observation(make_observation("c1", repeat=1))
    w.append_observation(make_observation("c1", repeat=2))

    run = load_run(RunPaths.for_run(tmp_path, "run-001"))
    assert set(run.observation_map()) == {("c1", 1), ("c1", 2)}


# --- açar sızması -----------------------------------------------------------


def test_acar_MUSAHIDE_xetasindan_da_silinir(tmp_path) -> None:
    """Açar manifestə deyil, SUT-un xəta mətninə də düşə bilər."""
    obs = make_observation("c1", error=f"AuthenticationError: açar {SECRET} qəbul edilmədi")
    w = writer(tmp_path)
    w.write_manifest({})
    w.append_observation(obs)

    raw = (tmp_path / "run-001" / OBSERVATIONS).read_text(encoding="utf-8")
    assert SECRET not in raw
    assert REDACTED in raw


def test_acar_manifestden_de_silinir(tmp_path) -> None:
    w = writer(tmp_path)
    w.write_manifest({"note": f"debug: {SECRET}"})

    raw = (tmp_path / "run-001" / "manifest.json").read_text(encoding="utf-8")
    assert SECRET not in raw


def test_redact_bos_gizli_deyeri_ATIR() -> None:
    assert redact("mətn", ["", None or ""]) == "mətn"


# --- kəsilmiş run -----------------------------------------------------------


def test_yarimciq_run_OXUNA_BILIR(tmp_path) -> None:
    """JSONL-in bütün mənası budur: kəsilən run-lar araşdırılası run-lardır."""
    w = writer(tmp_path)
    w.write_manifest({})
    w.append_observation(make_observation("c1"))
    w.append_observation(make_observation("c2"))
    # 3-cü sətir yarımçıq qalıb (proses öldürülüb).
    with (tmp_path / "run-001" / OBSERVATIONS).open("a", encoding="utf-8") as handle:
        handle.write('{"case_id": "c3", "answer')

    with pytest.raises(ArtifactError, match="oxunmadı"):
        load_run(RunPaths.for_run(tmp_path, "run-001"))

    # Zədəli sətirdən ƏVVƏLKİLƏR isə hələ də yerindədir.
    lines = (tmp_path / "run-001" / OBSERVATIONS).read_text(encoding="utf-8").splitlines()
    assert observation_from_json(json.loads(lines[0])).case_id == "c1"
    assert observation_from_json(json.loads(lines[1])).case_id == "c2"


def test_olmayan_run_AYDIN_xeta_verir(tmp_path) -> None:
    with pytest.raises(ArtifactError, match="list-runs"):
        load_run(RunPaths.for_run(tmp_path, "yoxdur"))


# --- sxem inkişafı ----------------------------------------------------------


def test_KOHNE_artefakt_yeni_sahe_ile_de_oxunur() -> None:
    """`reclassify` tarixi run-ları oxumalıdır — oxuma qəsdən gevşəkdir."""
    minimal = {"case_id": "c1", "answer_text": "Cavab", "refused": False}
    obs = observation_from_json(minimal)

    assert obs.case_id == "c1"
    assert obs.answer_text == "Cavab"
    # Sonradan əlavə olunmuş sahələr defoltla dolur, oxuma sınmır.
    assert obs.repeat == 1
    assert obs.chunks == ()
    assert obs.llm_calls == ()
    assert obs.threshold == 0.0
    assert obs.error == ""


def test_causes_faylini_YENIDEN_yazmaq_mumkundur(tmp_path) -> None:
    """`reclassify` sistemi çağırmadan taksonomiyanı yeniləyir."""
    w = writer(tmp_path)
    w.write_manifest({})
    w.append_cause(RootCause(case_id="c1", repeat=1, category="ok", detail=""))

    w.write_causes([RootCause(case_id="c1", repeat=1, category="retrieval_miss", detail="yeni qayda")])

    run = load_run(RunPaths.for_run(tmp_path, "run-001"))
    assert len(run.causes) == 1
    assert run.causes[0].category == "retrieval_miss"


# --- siyahı -----------------------------------------------------------------


def test_run_siyahisi_MANIFESTI_olanlari_gosterir(tmp_path) -> None:
    RunWriter(RunPaths.for_run(tmp_path, "run-a")).write_manifest({})
    RunWriter(RunPaths.for_run(tmp_path, "run-b")).write_manifest({})
    (tmp_path / "manifestsiz").mkdir()

    assert list_runs(tmp_path) == ("run-a", "run-b")


def test_olmayan_qovluqda_siyahi_bosdur(tmp_path) -> None:
    assert list_runs(tmp_path / "yoxdur") == ()


def test_bos_run_artefakti_qurula_bilir() -> None:
    run = RunArtifacts(run_id="x", manifest={}, observations=(), grades=(), verdicts=())
    assert run.causes == ()


# --- NaN qoruması və probe artefaktı (2026-08-12) --------------------------


def test_manifestde_NaN_yazilmasi_XETA_verir(tmp_path) -> None:
    """Səssiz `NaN` yazmaq artefaktı diskdə saxlayıb sübut kimi öldürərdi."""
    from eval.artifacts import RunPaths, RunWriter
    from eval.errors import ArtifactError

    writer = RunWriter(RunPaths.for_run(tmp_path, "r1"))
    with pytest.raises(ArtifactError, match="NaN"):
        writer.write_manifest({"retrieval_soft_floor_margin": float("nan")})


def test_probe_MOVCUD_qovlugun_ustune_yazmir(tmp_path) -> None:
    """D1-in kök səbəbi: üç sweep bir faylı əvəzlədi, ikisi itdi."""
    from eval.artifacts import ProbePaths, ProbeWriter
    from eval.errors import ArtifactError

    paths = ProbePaths.for_probe(tmp_path, "20260812T000000Z-sweep-top_k")
    ProbeWriter(paths).write_manifest({"probe_tool": "sweep"})

    with pytest.raises(ArtifactError, match="üstünə yazılmır"):
        ProbeWriter(paths)


def test_probe_setirleri_AXINLA_yazilir(tmp_path) -> None:
    """Kəsilən sweep də oxunaqlı qalmalıdır — `runs/` ilə eyni zəmanət."""
    from eval.artifacts import PROBE_ROWS, ProbePaths, ProbeWriter

    paths = ProbePaths.for_probe(tmp_path, "20260812T000000Z-sweep-top_k")
    writer = ProbeWriter(paths)
    writer.append_row({"top_k": 4})
    # manifest hələ yazılmayıb — sətirlər buna baxmayaraq diskdədir
    lines = paths.file(PROBE_ROWS).read_text(encoding="utf-8").splitlines()
    assert [json.loads(l) for l in lines] == [{"top_k": 4}]


def test_probe_argv_ACARI_redaksiya_edir(tmp_path) -> None:
    """argv istifadəçidən gəlir — açar səhvən arqumentə düşə bilər."""
    from eval.artifacts import PROBE_MANIFEST, ProbePaths, ProbeWriter

    secret = "sk-ant-heqiqi-acar-deyeri-12345"
    paths = ProbePaths.for_probe(tmp_path, "20260812T000000Z-sweep-top_k")
    ProbeWriter(paths, secrets=[secret]).write_manifest({"argv": ["--key", secret]})

    text = paths.file(PROBE_MANIFEST).read_text(encoding="utf-8")
    assert secret not in text


def test_probe_id_OXLARI_ada_yazir() -> None:
    from eval.artifacts import probe_id

    assert probe_id(alet="sweep", oxlar=["top_k", "threshold"], indi="20260812T000000Z") == (
        "20260812T000000Z-sweep-top_k+threshold"
    )


# --- argv-də maşına aid yol -------------------------------------------------
#
# 2026-08-12: `logs/probes/...-eksperiment-.../manifest.json` `--workdir`-i
# olduğu kimi saxlamışdı, yəni ictimai artefaktda maşının müvəqqəti sessiya
# qovluğunun tam yolu qalmışdı. Yol SÜBUT DEYİL — hansı indeksin ölçüldüyünü
# `sut_index.sha256` və chunk ID barmaq izi qeyd edir — ona görə artefaktdan
# çıxarılır. Repo daxilindəki yol isə sübutun bir hissəsidir və nisbi qalır.


def test_argv_REPO_DAXILI_yolu_nisbi_edir(tmp_path) -> None:
    from eval.artifacts import argv_temizle

    daxili = tmp_path / "logs" / "probes"
    daxili.mkdir(parents=True)
    assert argv_temizle(["--probes-dir", str(daxili)], koke=tmp_path) == [
        "--probes-dir",
        "logs/probes",
    ]


def test_argv_REPODAN_KENAR_yolu_EVEZLEYIR(tmp_path) -> None:
    from eval.artifacts import MUVEQQETI_YOL, argv_temizle

    kenar = tmp_path.parent / "kenar-qovluq" / "idx"
    temiz = argv_temizle(["--workdir", str(kenar)], koke=tmp_path)
    assert temiz == ["--workdir", MUVEQQETI_YOL]
    assert str(kenar) not in " ".join(temiz)


def test_argv_BERABERLIKLI_forma_da_temizlenir(tmp_path) -> None:
    """`--workdir=/abs/yol` argparse-də qanunidir və eyni sızmanı verir."""
    from eval.artifacts import MUVEQQETI_YOL, argv_temizle

    kenar = tmp_path.parent / "kenar-qovluq"
    assert argv_temizle([f"--workdir={kenar}"], koke=tmp_path) == [
        f"--workdir={MUVEQQETI_YOL}"
    ]


def test_argv_YOL_OLMAYAN_tokenlere_toxunmur(tmp_path) -> None:
    """Ölçmə parametrləri toxunulmaz qalmalıdır — onlar sübutdur."""
    from eval.artifacts import argv_temizle

    argv = ["python", "tools/retrieval_sweep.py", "--top-k", "4", "6", "--threshold", "0.42"]
    assert argv_temizle(argv, koke=tmp_path) == argv


def test_probe_identity_argv_ni_TEMIZLENMIS_saxlayir(tmp_path) -> None:
    """Təmizləmə çağırana buraxılmır: unudulan yer sızmanın özüdür."""
    from eval.artifacts import MUVEQQETI_YOL, probe_identity

    kenar = tmp_path.parent / "kenar-qovluq" / "idx"
    kimlik = probe_identity(
        alet="eksperiment",
        argv=["python", "tools/retrieval_experiments.py", "--workdir", str(kenar)],
        started_at="20260812T000000Z",
        harness_commit="abc",
        sut_commit="def",
        config_hash="hash",
        dataset_sha256="sha",
        koke=tmp_path,
    )
    assert kimlik["argv"][-1] == MUVEQQETI_YOL
