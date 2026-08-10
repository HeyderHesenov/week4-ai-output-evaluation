"""Qiymətləndirilən sistemin (Week 2 RAG) adapteri.

PREFLIGHT NİYƏ AYRICA ADDIMDIR
-------------------------------
Üç şey PULLU çağırışdan ƏVVƏL yoxlanılır:

  1. `git rev-parse HEAD` == `SUT_COMMIT` — «Week 2 sistemini dəyişdirmədən
     qiymətləndirdik» cümləsi beləliklə iddia olmaqdan çıxıb yoxlanan
     invarianta çevrilir.
  2. `store.count() > 0` — boş indekslə işlədilən run 20 case-in hamısında
     `empty_index` imtinası verər və nəticə «sistem imtina edir» kimi
     oxunardı. Halbuki bu, sistemin deyil, qurulmanın nəticəsidir.
  3. Yol və import — `rag` modulu görünmürsə, bunu 1-ci case-də deyil,
     başlanğıcda bilmək lazımdır.

Hər üçü şəbəkəyə çıxmır və pul xərcləmir; buna görə heç bir səbəb yoxdur ki,
onlar run-ın ortasında aşkarlansın.

BİR RUN = BİR VARİANT
---------------------
Variant `RagSut` qurularkən verilir və müşahidə başına dəyişmir. Səbəb
sadədir: variant müşahidənin kontekstidir, parametri deyil. Case başına
dəyişən variant «bu müşahidə hansı prompt altında alındı» sualını hər
sətirdə yenidən verməyi tələb edərdi; run manifestində bir dəfə yazmaq həm
daha ucuz, həm daha dürüstdür.

CASE SƏVİYYƏSİNDƏ İSTİSNA UDULUR — QƏSDƏN
------------------------------------------
`observe()` istisnanı `SutObservation(error=...)`-a çevirir. 19 case
işləyib 20-cidə şəbəkə qopanda bütün run-ı (və onun xərcini) itirmək
mənasızdır; xəta artefaktda görünür və `rootcause` onu `sut_error` kimi
təsnif edir — yəni keçid kimi sayılmır.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import Settings
from .dataset import EvalCase
from .errors import SutError
from .instrument import InstrumentedLLM, InstrumentedStore, RoleMap
from .observation import CallRecorder, ChunkView, SutObservation
from .variants import IDENTITY_VARIANT, PromptVariant

# `observe()` istisnanı bu səbəb kodu ilə yazır. SUT-un öz səbəb kodları
# arasında YOXDUR (`graders.REFUSAL_REASONS`) — qarışmasın deyə.
REASON_SUT_ERROR = "sut_error"


@dataclass(frozen=True)
class SutInfo:
    """Preflight-ın topladığı, manifestə yazılan həqiqətlər."""

    commit: str
    chunk_count: int
    sut_path: str


def read_git_commit(path: Path) -> str:
    """`git rev-parse HEAD` — submodule-un HƏQİQİ vəziyyəti."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SutError(f"`git rev-parse` işə salına bilmədi: {exc}") from None
    if result.returncode != 0:
        raise SutError(
            f"{path} git deposu kimi oxunmadı.\n"
            f"  git: {result.stderr.strip()}\n"
            "Həlli: git submodule update --init --recursive"
        )
    return result.stdout.strip()


class RagSut:
    """Week 2 RAG pipeline-ının ölçülən sarğısı."""

    def __init__(
        self,
        settings: Settings,
        *,
        variant: PromptVariant = IDENTITY_VARIANT,
        pipeline_factory: Callable[[Callable[[Any], Any], Callable[[Any], Any]], Any] | None = None,
        commit_reader: Callable[[Path], str] = read_git_commit,
        roles: RoleMap | None = None,
    ) -> None:
        self.settings = settings
        self.variant = variant
        self.recorder = CallRecorder()
        self._factory = pipeline_factory or self._default_factory
        self._commit_reader = commit_reader
        self._roles = roles
        self._pipeline: Any = None
        self._info: SutInfo | None = None
        self._llm_wrappers: list[InstrumentedLLM] = []

    @property
    def variant_applications(self) -> int:
        """Variantın SUT-un promptuna neçə dəfə tətbiq olunduğu.

        Variant qeyri-identity olduğu halda bu SIFIR qalırsa, run baseline
        ilə eynidir və «variant ölçüldü» iddiası yalandır. Manifestə yazılır
        ki, iddia sonradan yoxlana bilsin.
        """
        return sum(w.applied_count for w in self._llm_wrappers)

    @property
    def unknown_role_calls(self) -> int:
        return sum(w.unknown_role_count for w in self._llm_wrappers)

    # -- preflight ---------------------------------------------------------

    def preflight(self) -> SutInfo:
        """Bir dəfə işləyir; nəticəsi keşlənir."""
        if self._info is not None:
            return self._info

        expected = self.settings.sut_commit.strip()
        if not expected:
            raise SutError(
                "SUT_COMMIT təyin edilməyib.\n"
                "Pin edilmiş commit olmadan «sistemi dəyişdirmədən qiymətləndirdik» "
                "iddiası yoxlana bilmir və qiymətləndirmə təkrarlanmır.\n"
                "Həlli: .env faylında SUT_COMMIT= sətrini submodule-un commit-i ilə "
                "doldurun (git -C vendor/week2-rag-document-qa rev-parse HEAD)."
            )

        path = self.settings.sut_path
        if not path.exists():
            raise SutError(
                f"Qiymətləndirilən sistem tapılmadı: {path}\n"
                "Həlli: git submodule update --init --recursive"
            )

        actual = self._commit_reader(path)
        if actual != expected:
            raise SutError(
                "SUT pin edilmiş commit-də deyil — qiymətləndirmə dayandırıldı.\n"
                f"  gözlənilən: {expected}\n"
                f"  cari:       {actual}\n"
                "Ya submodule-u pin-ə qaytarın, ya da .env-dəki SUT_COMMIT-i "
                "yeniləyib bu dəyişikliyi hesabatda AÇIQ qeyd edin."
            )

        pipeline = self._ensure_pipeline()
        count = int(pipeline.store.count())
        if count <= 0:
            raise SutError(
                "Vektor indeksi boşdur — bütün case-lər `empty_index` imtinası verərdi "
                "və nəticə sistemin deyil, qurulmanın hesabatı olardı.\n"
                f"Həlli: cd {path} && python -m rag.cli ingest --path data/"
            )

        self._info = SutInfo(commit=actual, chunk_count=count, sut_path=str(path))
        return self._info

    # -- müşahidə ----------------------------------------------------------

    def observe(self, case: EvalCase, *, repeat: int = 1) -> SutObservation:
        self.preflight()
        pipeline = self._ensure_pipeline()

        # Əvvəlki müşahidədən qalan qeyd olmasın (SUT xəta atıb drain-ə
        # çatmadığı halda belə).
        self.recorder.drain()

        started = time.perf_counter()
        try:
            answer = pipeline.ask(case.question)
        except Exception as exc:  # noqa: BLE001 — case səviyyəsində udulur
            total_ms = (time.perf_counter() - started) * 1000.0
            llm_calls, retrieval_calls = self.recorder.drain()
            return SutObservation(
                case_id=case.id,
                repeat=repeat,
                question=case.question,
                answer_text="",
                refused=False,
                reason=REASON_SUT_ERROR,
                llm_calls=llm_calls,
                retrieval_calls=retrieval_calls,
                total_ms=total_ms,
                error=f"{type(exc).__name__}: {exc}",
            )

        total_ms = (time.perf_counter() - started) * 1000.0
        llm_calls, retrieval_calls = self.recorder.drain()
        return SutObservation(
            case_id=case.id,
            repeat=repeat,
            question=case.question,
            answer_text=str(answer.text),
            refused=bool(answer.refused),
            reason=str(answer.reason),
            cited_labels=tuple(answer.cited_labels),
            invalid_citations=tuple(answer.invalid_citations),
            rescued_labels=tuple(answer.rescued_labels),
            unsupported_claims=tuple(str(c) for c in answer.unsupported_claims),
            grounding_detail=str(answer.grounding_detail),
            top_score=float(answer.top_score),
            top_lexical=float(answer.top_lexical),
            threshold=float(answer.threshold),
            retrieved_count=int(answer.retrieved_count),
            attempts=int(answer.attempts),
            chunks=tuple(_to_chunk_view(c) for c in answer.chunks),
            llm_calls=llm_calls,
            retrieval_calls=retrieval_calls,
            total_ms=total_ms,
        )

    # -- qurulma -----------------------------------------------------------

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is None:
            self._pipeline = self._factory(self._wrap_llm, self._wrap_store)
        return self._pipeline

    def _wrap_llm(self, inner: Any) -> InstrumentedLLM:
        wrapper = InstrumentedLLM(
            inner=inner,
            recorder=self.recorder,
            roles=self._roles if self._roles is not None else self._live_roles(),
            variant=self.variant,
            provider="openai",
            model=self.settings.generator_model,
        )
        self._llm_wrappers.append(wrapper)
        return wrapper

    def _wrap_store(self, inner: Any) -> InstrumentedStore:
        return InstrumentedStore(inner=inner, recorder=self.recorder)

    def _live_roles(self) -> RoleMap:
        """Rol xəritəsi SUT-un HƏQİQİ sabitlərindən qurulur.

        Prompt mətnini bura köçürmək (graders-dəki `REFUSAL_REASONS` kimi)
        səhv olardı: orada məqsəd SUT-dan MÜSTƏQİL yoxlamaq idi, burada isə
        məqsəd MƏHZ SUT-un promptunu tanımaqdır. Kopyalanmış mətn bir boşluq
        fərqi ilə köhnəlsə, variant səssizcə tətbiq olunmamağa başlayardı.
        """
        self._add_sut_to_path()
        from rag.grounding import JUDGE_SYSTEM  # noqa: PLC0415
        from rag.pipeline import SYSTEM_INSTRUCTION  # noqa: PLC0415

        roles = RoleMap.from_prompts(synthesis=SYSTEM_INSTRUCTION, grounding=JUDGE_SYSTEM)
        self._roles = roles
        return roles

    def _add_sut_to_path(self) -> None:
        path = str(self.settings.sut_path)
        if path not in sys.path:
            sys.path.insert(0, path)

    def _default_factory(
        self,
        wrap_llm: Callable[[Any], Any],
        wrap_store: Callable[[Any], Any],
    ) -> Any:
        """Canlı pipeline — SUT import-ları YALNIZ burada baş verir.

        Modul səviyyəsində import etsək, bütün test dəsti langchain/chroma
        tələb edərdi və `conftest.py`-dakı «açarsız, şəbəkəsiz» prinsip
        pozulardı.
        """
        self._add_sut_to_path()
        try:
            from rag.config import Settings as SutSettings, build_llm  # noqa: PLC0415
            from rag.pipeline import RagPipeline  # noqa: PLC0415
            from rag.store import VectorStore  # noqa: PLC0415
        except ImportError as exc:
            raise SutError(
                f"SUT modulları import edilmədi: {exc}\n"
                "Həlli: pip install -r "
                f"{self.settings.sut_path}/requirements.txt"
            ) from None

        # AÇIQ override: SUT öz `.env`-ini oxuyur və biz onun nəyi oxuduğunu
        # təxmin etmək istəmirik. Manifestə yazdığımız model adı ilə həqiqətən
        # işlədilən model arasında fərq qalmamalıdır.
        sut_settings = SutSettings.load(
            openai_api_key=self.settings.require_openai_key(),
            llm_model=self.settings.generator_model,
            embedding_model=self.settings.embedding_model,
            grounding_mode=self.settings.grounding_mode,
        )
        return RagPipeline(
            sut_settings,
            store=wrap_store(VectorStore(sut_settings)),
            llm=wrap_llm(build_llm(sut_settings)),
        )


def _to_chunk_view(chunk: Any) -> ChunkView:
    return ChunkView(
        label=int(chunk.label),
        source=str(chunk.source),
        page=int(chunk.page),
        chunk_index=int(chunk.chunk_index),
        score=float(chunk.score),
        lexical_score=float(chunk.lexical_score),
        chunk_id=str(chunk.chunk_id),
        text=str(chunk.text),
    )
