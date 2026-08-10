"""Run artefaktları — xam sübutun diskə yazılması və geri oxunması.

NİYƏ JSONL, NİYƏ BİR BÖYÜK JSON
--------------------------------
Sətir-sətir yazılan fayl run yarıda kəsiləndə də oxuna bilir: 20 case-in
14-ü işləyibsə, 14 müşahidə əldə qalır. Bir massiv kimi yazılsaydı, JSON
bağlanmadığı üçün fayl tamamilə oxunmaz olardı — və məhz kəsilən run-lar
ən çox araşdırılası run-lardır.

NİYƏ OXUMA GEVŞƏK, YAZMA SƏRT
------------------------------
Yazarkən bütün sahələr yazılır (`asdict`). Oxuyarkən isə hər sahə
`.get(...)` ilə, defolt dəyərlə alınır. Səbəb `reclassify` əmridir: bu gün
yazılmış artefakt sabah — `SutObservation`-a yeni sahə əlavə olunandan
sonra — hələ də oxuna bilməlidir. Əks halda taksonomiyanı yeniləmək bütün
tarixi run-ları zibilə çevirərdi.

AÇARIN REDAKSİYASI
------------------
`Settings.public_dict()` manifestin açar sızdırmamasını təmin edir, amma
açar başqa yolla da fayla düşə bilər: SUT-un xəta mətnində, hakimin
`reason` sətrində, traceback-də. Buna görə hər yazılan sətir
`settings.live_secrets` dəyərlərinə qarşı süzülür. Bu, «ola bilməz» deyil,
«olmasın» yanaşmasıdır — sızma bir dəfə baş verirsə, geri qaytarmaq mümkün
deyil.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .errors import ArtifactError
from .graders import CheckOutcome, GradeResult
from .judge import Verdict
from .observation import ChunkView, LlmCall, RetrievalCall, SutObservation
from .rootcause import RootCause

REDACTED = "***REDAKSİYA***"

MANIFEST = "manifest.json"
OBSERVATIONS = "observations.jsonl"
GRADES = "grades.jsonl"
VERDICTS = "verdicts.jsonl"
CAUSES = "causes.jsonl"


# ---------------------------------------------------------------------------
# Redaksiya
# ---------------------------------------------------------------------------


def redact(text: str, secrets: Sequence[str]) -> str:
    for secret in secrets:
        if secret and secret in text:
            text = text.replace(secret, REDACTED)
    return text


# ---------------------------------------------------------------------------
# Run qovluğu
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @classmethod
    def for_run(cls, runs_dir: Path, run_id: str) -> "RunPaths":
        return cls(root=runs_dir / run_id)

    @property
    def run_id(self) -> str:
        return self.root.name

    def file(self, name: str) -> Path:
        return self.root / name

    def exists(self) -> bool:
        return self.file(MANIFEST).exists()


def list_runs(runs_dir: Path) -> tuple[str, ...]:
    if not runs_dir.exists():
        return ()
    return tuple(
        sorted(p.name for p in runs_dir.iterdir() if (p / MANIFEST).exists())
    )


# ---------------------------------------------------------------------------
# Yazma
# ---------------------------------------------------------------------------


class RunWriter:
    """Artefaktları sətir-sətir yazır; run kəsilsə də oxunaqlı qalır."""

    def __init__(self, paths: RunPaths, *, secrets: Sequence[str] = ()) -> None:
        self.paths = paths
        self._secrets = tuple(s for s in secrets if s)
        self.paths.root.mkdir(parents=True, exist_ok=True)

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self._write_text(MANIFEST, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    def append_observation(self, obs: SutObservation) -> None:
        self._append(OBSERVATIONS, asdict(obs))

    def append_grade(self, grade: GradeResult) -> None:
        self._append(GRADES, asdict(grade))

    def append_verdict(self, verdict: Verdict) -> None:
        self._append(VERDICTS, asdict(verdict))

    def append_cause(self, cause: RootCause) -> None:
        self._append(CAUSES, asdict(cause))

    def write_causes(self, causes: Iterable[RootCause]) -> None:
        """`reclassify` üçün: mövcud faylı tam əvəz edir."""
        path = self.paths.file(CAUSES)
        path.write_text(
            "".join(self._line(asdict(c)) for c in causes), encoding="utf-8"
        )

    # -- daxili -----------------------------------------------------------

    def _line(self, payload: dict[str, Any]) -> str:
        return redact(json.dumps(payload, ensure_ascii=False), self._secrets) + "\n"

    def _append(self, name: str, payload: dict[str, Any]) -> None:
        with self.paths.file(name).open("a", encoding="utf-8") as handle:
            handle.write(self._line(payload))

    def _write_text(self, name: str, text: str) -> None:
        self.paths.file(name).write_text(redact(text, self._secrets), encoding="utf-8")


# ---------------------------------------------------------------------------
# Oxuma
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    manifest: dict[str, Any]
    observations: tuple[SutObservation, ...]
    grades: tuple[GradeResult, ...]
    verdicts: tuple[Verdict, ...]
    causes: tuple[RootCause, ...] = field(default=())

    def observation_map(self) -> dict[tuple[str, int], SutObservation]:
        return {(o.case_id, o.repeat): o for o in self.observations}

    def grade_map(self) -> dict[tuple[str, int], GradeResult]:
        return {(g.case_id, g.repeat): g for g in self.grades}

    def verdict_map(self) -> dict[tuple[str, int], Verdict]:
        return {(v.case_id, v.repeat): v for v in self.verdicts}

    # `run_id` açara daxildir ki, bir neçə run-ın xəritəsi qarışmadan
    # birləşdirilə bilsin — insan etiketi məhz bir run-ın bir təkrarına
    # bağlıdır (bax `metrics.judge_bias`).
    def keyed_verdicts(self) -> dict[tuple[str, str, int], Verdict]:
        return {(self.run_id, v.case_id, v.repeat): v for v in self.verdicts}

    def keyed_answer_lengths(self) -> dict[tuple[str, str, int], int]:
        return {
            (self.run_id, o.case_id, o.repeat): len(o.answer_text)
            for o in self.observations
        }


def _restore_grade_repeats(grades: Sequence[GradeResult]) -> tuple[GradeResult, ...]:
    """`repeat` sahəsi əlavə edilməzdən ƏVVƏL yazılmış run-lar üçün bərpa.

    Köhnə `grades.jsonl` fayllarında təkrar nömrəsi yoxdur, amma sətirlər
    müşahidələrlə EYNİ sırada yazılıb. Ona görə hər case_id üçün görünmə
    sırası (1, 2, 3…) düzgün təkrar nömrəsini verir. Bunu etməsək köhnə
    run-larda bütün təkrarlar `repeat=0` ilə toqquşub bir-birini əvəz edərdi
    — yəni düzəltdiyimiz baqın elə özü.
    """
    seen: dict[str, int] = {}
    restored: list[GradeResult] = []
    for grade in grades:
        if grade.repeat:
            restored.append(grade)
            continue
        seen[grade.case_id] = seen.get(grade.case_id, 0) + 1
        restored.append(replace(grade, repeat=seen[grade.case_id]))
    return tuple(restored)


def load_run(paths: RunPaths) -> RunArtifacts:
    if not paths.exists():
        raise ArtifactError(
            f"Run tapılmadı: {paths.root}\n"
            "Mövcud run-ları görmək üçün: python -m eval.cli list-runs"
        )
    return RunArtifacts(
        run_id=paths.run_id,
        manifest=_read_json(paths.file(MANIFEST)),
        observations=tuple(_read_lines(paths.file(OBSERVATIONS), observation_from_json)),
        grades=_restore_grade_repeats(_read_lines(paths.file(GRADES), grade_from_json)),
        verdicts=tuple(_read_lines(paths.file(VERDICTS), verdict_from_json)),
        causes=tuple(_read_lines(paths.file(CAUSES), cause_from_json)),
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"{path} oxunmadı: {exc}") from None


def _read_lines(path: Path, decode) -> Iterator[Any]:
    if not path.exists():
        return
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            yield decode(json.loads(line))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ArtifactError(f"{path}:{number} oxunmadı: {exc}") from None


# ---------------------------------------------------------------------------
# Dekoderlər — GEVŞƏK: çatışmayan sahə defoltla doldurulur
# ---------------------------------------------------------------------------


def observation_from_json(raw: dict[str, Any]) -> SutObservation:
    return SutObservation(
        case_id=str(raw["case_id"]),
        repeat=int(raw.get("repeat", 1)),
        question=str(raw.get("question", "")),
        answer_text=str(raw.get("answer_text", "")),
        refused=bool(raw.get("refused", False)),
        reason=str(raw.get("reason", "")),
        cited_labels=tuple(raw.get("cited_labels") or ()),
        invalid_citations=tuple(raw.get("invalid_citations") or ()),
        rescued_labels=tuple(raw.get("rescued_labels") or ()),
        unsupported_claims=tuple(raw.get("unsupported_claims") or ()),
        grounding_detail=str(raw.get("grounding_detail", "")),
        top_score=float(raw.get("top_score", 0.0)),
        top_lexical=float(raw.get("top_lexical", 0.0)),
        threshold=float(raw.get("threshold", 0.0)),
        retrieved_count=int(raw.get("retrieved_count", 0)),
        attempts=int(raw.get("attempts", 0)),
        chunks=tuple(_chunk_from_json(c) for c in raw.get("chunks") or ()),
        llm_calls=tuple(_llm_from_json(c) for c in raw.get("llm_calls") or ()),
        retrieval_calls=tuple(_retrieval_from_json(c) for c in raw.get("retrieval_calls") or ()),
        total_ms=float(raw.get("total_ms", 0.0)),
        error=str(raw.get("error", "")),
    )


def _chunk_from_json(raw: dict[str, Any]) -> ChunkView:
    return ChunkView(
        label=int(raw.get("label", 0)),
        source=str(raw.get("source", "")),
        page=int(raw.get("page", 0)),
        chunk_index=int(raw.get("chunk_index", 0)),
        score=float(raw.get("score", 0.0)),
        lexical_score=float(raw.get("lexical_score", 0.0)),
        chunk_id=str(raw.get("chunk_id", "")),
        text=str(raw.get("text", "")),
    )


def _llm_from_json(raw: dict[str, Any]) -> LlmCall:
    return LlmCall(
        call_id=str(raw.get("call_id", "")),
        role=str(raw.get("role", "")),
        provider=str(raw.get("provider", "")),
        model=str(raw.get("model", "")),
        input_tokens=int(raw.get("input_tokens", 0)),
        output_tokens=int(raw.get("output_tokens", 0)),
        cached_read_tokens=int(raw.get("cached_read_tokens", 0)),
        latency_ms=float(raw.get("latency_ms", 0.0)),
        ok=bool(raw.get("ok", True)),
        error_type=str(raw.get("error_type", "")),
        system_sha256=str(raw.get("system_sha256", "")),
        prompt_sha256=str(raw.get("prompt_sha256", "")),
        response_text=str(raw.get("response_text", "")),
        usage_source=str(raw.get("usage_source", "missing")),
    )


def _retrieval_from_json(raw: dict[str, Any]) -> RetrievalCall:
    return RetrievalCall(
        mode=str(raw.get("mode", "")),
        k=int(raw.get("k", 0)),
        latency_ms=float(raw.get("latency_ms", 0.0)),
        returned=int(raw.get("returned", 0)),
        top_score=float(raw.get("top_score", 0.0)),
        query_chars=int(raw.get("query_chars", 0)),
    )


def grade_from_json(raw: dict[str, Any]) -> GradeResult:
    return GradeResult(
        case_id=str(raw["case_id"]),
        # 0 = «faylda yoxdur» sentineli; `load_run` onu görünmə sırasına
        # görə bərpa edir (aşağıdakı `_restore_grade_repeats`).
        repeat=int(raw.get("repeat", 0)),
        passed=bool(raw.get("passed", False)),
        checks=tuple(
            CheckOutcome(
                name=str(c.get("name", "")),
                ok=bool(c.get("ok", False)),
                detail=str(c.get("detail", "")),
            )
            for c in raw.get("checks") or ()
        ),
    )


def verdict_from_json(raw: dict[str, Any]) -> Verdict:
    score = raw.get("score")
    return Verdict(
        case_id=str(raw["case_id"]),
        repeat=int(raw.get("repeat", 1)),
        score=int(score) if score is not None else None,
        faithful=bool(raw.get("faithful", False)),
        complete=bool(raw.get("complete", False)),
        reason=str(raw.get("reason", "")),
        flags=tuple(raw.get("flags") or ()),
        judge_model=str(raw.get("judge_model", "")),
        served_model=str(raw.get("served_model", "")),
        judge_prompt_sha256=str(raw.get("judge_prompt_sha256", "")),
        threshold=int(raw.get("threshold", 2)),
        input_tokens=int(raw.get("input_tokens", 0)),
        output_tokens=int(raw.get("output_tokens", 0)),
        cached_read_tokens=int(raw.get("cached_read_tokens", 0)),
        cached_write_tokens=int(raw.get("cached_write_tokens", 0)),
        latency_ms=float(raw.get("latency_ms", 0.0)),
        stop_reason=str(raw.get("stop_reason", "")),
        error=str(raw.get("error", "")),
    )


def cause_from_json(raw: dict[str, Any]) -> RootCause:
    return RootCause(
        case_id=str(raw["case_id"]),
        repeat=int(raw.get("repeat", 1)),
        category=str(raw.get("category", "ok")),
        detail=str(raw.get("detail", "")),
    )
