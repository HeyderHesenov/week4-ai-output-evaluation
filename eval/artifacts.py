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

# Probe artefaktı — sweep/eksperiment üçün, `runs/` ilə eyni konvensiya.
PROBE_MANIFEST = "manifest.json"
PROBE_ROWS = "rows.jsonl"
PROBE_SUMMARY = "summary.md"


def _json(payload: Any, *, indent: int | None = None) -> str:
    """`allow_nan=False` — NaN/Infinity artefakta HEÇ VAXT düşmür.

    Python-un defoltu çılpaq `NaN` / `Infinity` yazır. Bunlar JSON deyil
    (RFC 8259), yəni belə bir manifest standart parserlə oxunmur: artefakt
    diskdə var, amma sübut kimi yararsızdır. `eval/config.py`-dakı
    `yoxla_retrieval` bunu girişdə tutur; bura isə ikinci qatdır, çünki
    dəyər SUT-dan kopyalanaraq da gələ bilər (`_effective_retrieval`) və o
    yol bizim validatoru görmür.
    """
    try:
        return json.dumps(payload, indent=indent, ensure_ascii=False, allow_nan=False)
    except ValueError as exc:
        raise ArtifactError(
            f"Artefakta NaN/Infinity yazılmır: {exc}\n"
            "Nəticə etibarlı JSON olmazdı və artefakt sübut kimi oxunmazdı. "
            "Mənbə çox vaxt konfiqurasiyadır — RETRIEVAL_* dəyərlərini yoxlayın."
        ) from None


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
# Probe qovluğu — sweep və eksperiment artefaktları
# ---------------------------------------------------------------------------
#
# NİYƏ RUN İLƏ EYNİ MÜQAVİLƏ
# ---------------------------
# Sweep pul xərcləyir və sənəddə sitat gətirilən rəqəm istehsal edir — yəni
# sübutdur. Əvvəlki versiya onu sabit `--out` faylına yazırdı və üç ardıcıl
# sweep bir-birinin üstünə yazdı: sənəd dörd ox üzrə nəticə iddia etdi,
# diskdə isə yalnız sonuncu qaldı. Bu, transkripsiya səhvi deyil, artefakt
# müqaviləsindəki boşluq idi — `runs/` heç vaxt üstünə yazılmır, sweep isə
# yazılırdı. Asimmetriya aradan qaldırılır.


@dataclass(frozen=True)
class ProbePaths:
    root: Path

    @classmethod
    def for_probe(cls, probes_dir: Path, probe_id: str) -> "ProbePaths":
        return cls(root=probes_dir / probe_id)

    @property
    def probe_id(self) -> str:
        return self.root.name

    def file(self, name: str) -> Path:
        return self.root / name

    def exists(self) -> bool:
        # QOVLUĞUN ÖZÜ, manifest yox. Manifest ən son yazılan fayl idi, ona
        # görə yarımçıq kəsilmiş icradan qalan qovluq «mövcud deyil» sayılırdı
        # və növbəti icra onun `rows.jsonl`-ına ƏLAVƏ yazırdı — bir artefaktda
        # iki icranın sətirləri. Qovluq isə ilk andan mövcuddur.
        return self.root.exists()


# Probe artefaktının vəziyyəti — manifestdəki `status` sahəsi.
#
# Ölçmə kəsilə bilər: Ctrl-C, `ConfigError`, chroma xətası. Kəsilmiş icranın
# qovluğu diskdə qalır və onu SƏSSİZ saxlamaq iki cür yalan yaradırdı —
# yarımçıq rəqəmlər tam kimi oxunurdu, yaxud qovluq heç nə demədən bütün
# testləri qırmızı edirdi. Status hər iki halı adlandırır.
PROBE_YARIMCIQ = "yarımçıq"   # qovluq açılıb, ölçmə hələ bitməyib
PROBE_TAMAM = "tamam"         # ölçmə bitib, nəticələr etibarlıdır
PROBE_UGURSUZ = "uğursuz"     # ölçmə kəsilib; sətirlər natamamdır


def probe_id(*, alet: str, oxlar: Sequence[str], indi: str) -> str:
    """`20260812T131500Z-sweep-top_k+lexical_threshold`.

    Ox adları ada QƏSDƏN düşür: qovluq siyahısına baxan adam hansı sweep-in
    hansı oxu əhatə etdiyini faylı açmadan görür. Vaxt damğası isə eyni
    oxların təkrar ölçülməsini ƏVƏZ etmir, YANINA qoyur.
    """
    if not alet:
        raise ArtifactError("probe_id: alət adı boş ola bilməz.")
    suffiks = "+".join(oxlar) if oxlar else "sabit"
    return f"{indi}-{alet}-{suffiks}"


# Repo-dan KƏNAR mütləq yolun artefaktdakı əvəzi.
MUVEQQETI_YOL = "<müvəqqəti-qovluq>"


def _yol_kimi_gorunur(token: str) -> bool:
    return token.startswith(("/", "~"))


def _yolu_temizle(token: str, koke: Path) -> str:
    try:
        cozulmus = Path(token).expanduser().resolve()
    except (OSError, RuntimeError):
        return MUVEQQETI_YOL
    try:
        return str(cozulmus.relative_to(koke))
    except ValueError:
        return MUVEQQETI_YOL


def argv_temizle(argv: Sequence[str], *, koke: Path) -> list[str]:
    """argv-dən maşına aid MÜTLƏQ yolları çıxarır, qalanına toxunmur.

    NİYƏ: artefakt ictimai repo-ya düşür və mütləq yol maşını tanıdır —
    istifadəçi adı, müvəqqəti sessiya qovluğu, alət izləri. Bu yol SÜBUT
    DEYİL: hansı indeksin ölçüldüyü sualına `sut_index.sha256` və chunk ID
    barmaq izi cavab verir, `--workdir`-in harada olması isə nəticəyə təsir
    etmir — istənilən boş qovluq eyni cədvəli verir.

    Repo daxilindəki yol NİSBİ qalır, çünki o, sübutun bir hissəsidir
    (`logs/probes/...` oxucunun aça biləcəyi faktiki fayldır) və nisbi
    formada maşından asılı olmur — `config._repo_relative` ilə eyni qayda.

    Yol OLMAYAN tokenlər toxunulmur: `--top-k 4 6 8` ölçmənin özüdür.
    """
    koke = koke.resolve()
    out: list[str] = []
    for token in argv:
        if _yol_kimi_gorunur(token):
            out.append(_yolu_temizle(token, koke))
        elif token.startswith("--") and "=" in token:
            # `--workdir=/abs/yol` argparse-də `--workdir /abs/yol` ilə eyni
            # mənadadır; yalnız birini təmizləmək qapını açıq saxlayardı.
            ad, _, deyer = token.partition("=")
            out.append(
                f"{ad}={_yolu_temizle(deyer, koke)}"
                if _yol_kimi_gorunur(deyer)
                else token
            )
        else:
            out.append(token)
    return out


def probe_identity(
    *,
    alet: str,
    argv: Sequence[str],
    started_at: str,
    harness_commit: str,
    sut_commit: str,
    config_hash: str,
    dataset_sha256: str,
    koke: Path,
    split: str = "dev",
) -> dict[str, Any]:
    """Hər probe manifestinin ortaq kimlik bloku.

    `argv` qəsdən saxlanılır: «bu cədvəl hansı əmrlə alınıb?» sualının
    cavabı sənəddə deyil, artefaktda olmalıdır — sənəd köhnələ bilər.

    Təmizləmə BURADA aparılır, çağıranda yox: `koke` məcburi arqumentdir,
    ona görə yeni alət də onu ötürməyə məcburdur. Sızma məhz unudulan
    yerdə baş verir.
    """
    return {
        "probe_tool": alet,
        "argv": argv_temizle(argv, koke=koke),
        "started_at": started_at,
        "harness_commit": harness_commit,
        "sut_commit": sut_commit,
        "config_hash": config_hash,
        "dataset_sha256": dataset_sha256,
        "split": split,
    }


class ProbeWriter:
    """Sweep/eksperiment artefaktı — MÖVCUD qovluğun üstünə yazmır.

    `RunWriter` ilə eyni zəmanətlər: hər sətir yazılan anda diskə düşür
    (kəsilən icra da oxunaqlı qalır) və açar dəyərləri redaksiya olunur —
    argv istifadəçidən gəlir, açar oraya səhvən düşə bilər.

    KİMLİK İLK ANDA YAZILIR. Əvvəllər manifest yalnız uğurlu sonda yaranırdı,
    yəni qovluq mövcud olub manifestsiz qala bilirdi: kim tərəfindən, hansı
    əmrlə açıldığı bilinməzdi və `test_logs_iddialari` hamı üçün qırmızı
    olurdu. İndi qovluq yarananda `status: yarımçıq` manifesti düşür və
    icranın sonunda `tamam` (`write_manifest`) və ya `uğursuz`
    (`mark_failed`) ilə əvəzlənir.
    """

    def __init__(
        self,
        paths: ProbePaths,
        *,
        secrets: Sequence[str] = (),
        kimlik: dict[str, Any] | None = None,
    ) -> None:
        if paths.exists():
            raise ArtifactError(
                f"{paths.root} artıq mövcuddur — ölçmə qeydinin üstünə yazılmır.\n"
                "Hər icra öz qovluğunu alır; köhnə artefakt sübutdur, silinmir "
                "və əvəzlənmir."
            )
        self.paths = paths
        self._secrets = tuple(s for s in secrets if s)
        self._kimlik = dict(kimlik or {})
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self._status_yaz(PROBE_YARIMCIQ, self._kimlik)

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        """İcra UĞURLA bitdi — manifest `tamam` kimi möhürlənir."""
        self._status_yaz(PROBE_TAMAM, manifest)

    def mark_failed(self, error: str) -> None:
        """İcra kəsildi — qovluq qalır, amma nəticələri natamam elan olunur.

        Qovluq SİLİNMİR: yarımçıq sətirlər də «bu parametrlərdə nə baş verdi?»
        sualına cavabdır, və silmək kəsilmənin özünü gizlədərdi.
        """
        self._status_yaz(PROBE_UGURSUZ, {**self._kimlik, "error": error})

    def _status_yaz(self, status: str, manifest: dict[str, Any]) -> None:
        self._write_text(
            PROBE_MANIFEST, _json({**manifest, "status": status}, indent=2) + "\n"
        )

    def append_row(self, row: dict[str, Any]) -> None:
        with self.paths.file(PROBE_ROWS).open("a", encoding="utf-8") as handle:
            handle.write(redact(_json(row), self._secrets) + "\n")

    def write_summary(self, markdown: str) -> None:
        """Sənədə köçürüləcək cədvəl BURADA yaranır, əl ilə yazılmır.

        Transkripsiya səhvini struktur olaraq mümkünsüz edən budur:
        `tests/test_logs_iddialari.py` `logs/*.md`-dəki hər artefakt blokunun
        məhz bu faylda mövcud olduğunu yoxlayır.
        """
        self._write_text(PROBE_SUMMARY, markdown)

    def _write_text(self, name: str, text: str) -> None:
        self.paths.file(name).write_text(redact(text, self._secrets), encoding="utf-8")


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
        self._write_text(MANIFEST, _json(manifest, indent=2) + "\n")

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
        return redact(_json(payload), self._secrets) + "\n"

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
        # Dekoderdə UNUDULMUŞDU: `scores` yazılırdı, amma burada bərpa
        # olunmadığı üçün dataclass default-u susmadan `()` verirdi. Sahənin
        # bütün mənası saxlanmış artefaktdan astana sualına cavab verməkdir —
        # oxunmayan sahə isə yazılmamış sahə ilə eynidir.
        scores=tuple(float(x) for x in raw.get("scores") or ()),
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
