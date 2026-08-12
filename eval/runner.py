"""Run-un orkestrasiyası — mühafizələr, icra, artefaktlar.

SIRALAMA QƏSDƏNDİR
------------------
Bütün pulsuz yoxlamalar İLK çağırışdan əvvəl bitir:

    dataset hash → bölgü icazəsi → variant təmizliyi → SUT commit → indeks

Bu sıra ilə çirklənmiş və ya yanlış qurulmuş run heç bir token xərcləmədən
dayanır. Əks sıra (əvvəl işlət, sonra yoxla) eyni xətanı 20 case-lik
xərcdən SONRA aşkarlayardı.

HOLDOUT REGİSTRİ NİYƏ KİLİD DEYİL
----------------------------------
`--split holdout` işlədilməsi qadağan edilmir, qeyd edilir. Kilid qoymaq
onu texniki maneəyə çevirərdi və maneələr aşılır (ətraf mühit dəyişəni,
əl ilə redaktə). Yalnız-əlavə registr isə sui-istifadəni GÖRÜNƏN edir:
hesabatda «holdout 9 dəfə işlədilib» sətri yoxlayıcıya öz hökmünü vermək
imkanı verir. Dürüstlük gizlədilməzlikdən keçir.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .artifacts import RunPaths, RunWriter
from .config import Settings
from .dataset import Dataset, EvalCase
from .graders import GradeResult, grade_deterministic
from .judge import Verdict, judge_prompt_sha256
from .observation import SutObservation
from .rootcause import RootCause, classify
from .split import ContaminationGuard, LedgerEntry, append_holdout_ledger
from .variants import IDENTITY_VARIANT, PromptVariant

HOLDOUT_LEDGER = "holdout_ledger.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_harness_commit(root: Path) -> str:
    """Harness-in öz commit-i. Commit yoxdursa run dayanmır — qeyd olunur."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "(git əlçatmazdır)"
    return result.stdout.strip() if result.returncode == 0 else "(commit yoxdur)"


@dataclass(frozen=True)
class RunResult:
    run_id: str
    paths: RunPaths
    manifest: dict[str, Any]
    observations: tuple[SutObservation, ...]
    grades: tuple[GradeResult, ...]
    verdicts: tuple[Verdict, ...]
    causes: tuple[RootCause, ...]

    @property
    def case_count(self) -> int:
        return len({o.case_id for o in self.observations})


class Runner:
    """Bir run-ı başdan-sona aparır.

    `sut` və `judge` inyeksiya olunur: test dəsti şəbəkəsiz işləyir və
    canlı yol ilə saxta yol EYNİ kodu keçir.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        dataset: Dataset,
        guard: ContaminationGuard,
        sut: Any,
        judge: Any,
        now: Callable[[], str] = utc_now,
        harness_commit: str | None = None,
        on_event: Callable[[str], None] = lambda _message: None,
    ) -> None:
        self.settings = settings
        self.dataset = dataset
        self.guard = guard
        self.sut = sut
        self.judge = judge
        self._now = now
        self._harness_commit = harness_commit
        self._emit = on_event

    # -- əsas axın ---------------------------------------------------------

    def run(
        self,
        *,
        split: str,
        variant: PromptVariant = IDENTITY_VARIANT,
        action: str = "run",
        note: str = "",
    ) -> RunResult:
        # --- 1. PULSUZ mühafizələr (heç bir token xərclənmir) -------------
        self.guard.assert_dataset_unchanged()
        self.guard.assert_split_allowed(split, action=action)
        similarity = self.guard.assert_variant_clean(variant)

        cases = tuple(self.dataset.subset(split))
        if not cases:
            self._emit(f"XƏBƏRDARLIQ: '{split}' bölgüsündə case yoxdur.")

        # --- 2. SUT preflight (yenə pulsuz) ------------------------------
        info = self.sut.preflight()
        self._emit(
            f"SUT hazırdır: commit {info.commit[:12]}, indeksdə {info.chunk_count} chunk."
        )

        started_at = self._now()
        run_id = f"{started_at.replace(':', '').replace('-', '')}-{split}-{variant.id}"
        paths = RunPaths.for_run(self.settings.runs_dir, run_id)
        writer = RunWriter(paths, secrets=self.settings.live_secrets)

        # --- 3. İcra -----------------------------------------------------
        observations: list[SutObservation] = []
        grades: list[GradeResult] = []
        verdicts: list[Verdict] = []
        causes: list[RootCause] = []

        total = len(cases) * self.settings.repeats
        done = 0
        for case in cases:
            for repeat in range(1, self.settings.repeats + 1):
                done += 1
                self._emit(f"[{done}/{total}] {case.id} (təkrar {repeat})")

                obs = self.sut.observe(case, repeat=repeat)
                observations.append(obs)
                writer.append_observation(obs)

                grade = None
                if case.wants_deterministic:
                    grade = grade_deterministic(case, obs)
                    grades.append(grade)
                    writer.append_grade(grade)

                verdict = None
                if case.wants_judge and not obs.error:
                    # SUT xətası olan müşahidəni hakimə göndərmək pul yandırmaqdır:
                    # qiymətləndiriləcək cavab yoxdur.
                    verdict = self.judge.judge(case, obs)
                    verdicts.append(verdict)
                    writer.append_verdict(verdict)

                cause = classify(case, obs, grade=grade, verdict=verdict)
                causes.append(cause)
                writer.append_cause(cause)

        # --- 4. Manifest və registr --------------------------------------
        manifest = self._manifest(
            run_id=run_id,
            started_at=started_at,
            split=split,
            variant=variant,
            info=info,
            case_count=len(cases),
            similarity=similarity,
            note=note,
        )
        writer.write_manifest(manifest)

        if split in ("holdout", "all"):
            entry = LedgerEntry(
                run_id=run_id,
                at=started_at,
                variant_id=variant.id,
                harness_commit=manifest["harness_commit"],
                case_count=len(cases),
                note=note,
            )
            entries = append_holdout_ledger(self.settings.runs_dir / HOLDOUT_LEDGER, entry)
            self._emit(
                f"Holdout registrinə yazıldı — bu, {len(entries)}-ci holdout icrasıdır."
            )

        return RunResult(
            run_id=run_id,
            paths=paths,
            manifest=manifest,
            observations=tuple(observations),
            grades=tuple(grades),
            verdicts=tuple(verdicts),
            causes=tuple(causes),
        )

    # -- manifest ----------------------------------------------------------

    def _manifest(
        self,
        *,
        run_id: str,
        started_at: str,
        split: str,
        variant: PromptVariant,
        info: Any,
        case_count: int,
        similarity: Sequence[Any],
        note: str,
    ) -> dict[str, Any]:
        commit = (
            self._harness_commit
            if self._harness_commit is not None
            else read_harness_commit(Path(__file__).resolve().parents[1])
        )
        worst = max((h.score for h in similarity), default=0.0)
        return {
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": self._now(),
            "split": split,
            "case_count": case_count,
            "repeats": self.settings.repeats,
            "note": note,
            # --- kimliklər: hansı iki run müqayisə edilə bilər ---
            "dataset_sha256": self.dataset.sha256,
            "split_manifest_sha256": self.guard.manifest.sha256,
            "variant_id": variant.id,
            "variant_label": variant.label,
            "variant_sha256": variant.sha256,
            # Variantın HƏQİQƏTƏN tətbiq olunduğunun sübutu. Sıfır olarsa,
            # run baseline ilə eynidir — hesabat bunu xəbərdarlıq kimi çap edir.
            # `getattr` ilə: SUT sarğısı test ikilisi ola bilər və runner
            # ölçmə üçün onun daxili sayğaclarından ASILI olmamalıdır.
            "variant_applications": getattr(self.sut, "variant_applications", None),
            "unknown_role_calls": getattr(self.sut, "unknown_role_calls", None),
            "judge_prompt_sha256": judge_prompt_sha256(),
            "harness_commit": commit,
            "sut_commit": info.commit,
            "sut_chunk_count": info.chunk_count,
            # SUT-un işlətdiyi FAKTİKİ retrieval parametrləri. `config` bloku
            # yalnız bizim override-ları göstərir; bu isə pipeline-dan geri
            # oxunandır, ona görə «override verməmişik, amma SUT nə işlədib?»
            # sualının cavabı artefaktda qalır.
            "sut_retrieval": dict(info.retrieval),
            # NİYYƏT (`sut_retrieval.chunk_size`) İLƏ SÜBUTU (barmaq izi)
            # ayrı saxlayırıq: birincisi indeksi təkrar qurmaq üçün lazımdır,
            # ikincisi isə hansı indeksin HƏQİQƏTƏN sorğulandığını göstərir.
            "sut_index": dict(info.index),
            "config_hash": self.settings.config_hash,
            "config": self.settings.public_dict(),
            # --- ölçülmüş çirklənmə riski: iddia yox, rəqəm ---
            "max_holdout_similarity": round(worst, 4),
        }


# ---------------------------------------------------------------------------
# Yenidən təsnifat
# ---------------------------------------------------------------------------


def reclassify(
    dataset: Dataset,
    observations: Sequence[SutObservation],
    grades: Sequence[GradeResult],
    verdicts: Sequence[Verdict],
) -> list[RootCause]:
    """Saxlanmış müşahidələr üzərində taksonomiyanı yenidən hesablayır.

    Nə şəbəkə, nə açar, nə pul. `observation.py`-dakı xam/törəmə ayrımının
    bütün mənası budur: taksonomiya kod, müşahidə isə fakt olduğu üçün
    kodun dəyişməsi faktın yenidən toplanmasını tələb etmir.
    """
    # Qiymət də verdikt kimi (case_id, repeat) ilə açarlanır. Yalnız case_id
    # işlədilsəydi, təkrarlı run-larda sonuncu təkrarın qiyməti bütün
    # təkrarlara şamil olunardı.
    grade_by_key = {(g.case_id, g.repeat): g for g in grades}
    verdict_by_key = {(v.case_id, v.repeat): v for v in verdicts}

    causes: list[RootCause] = []
    for obs in observations:
        case: EvalCase = dataset.by_id(obs.case_id)
        causes.append(
            classify(
                case,
                obs,
                grade=grade_by_key.get((obs.case_id, obs.repeat)),
                verdict=verdict_by_key.get((obs.case_id, obs.repeat)),
            )
        )
    return causes
