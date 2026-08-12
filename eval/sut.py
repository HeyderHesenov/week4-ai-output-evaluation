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

import hashlib
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

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
    # SUT-un HƏQİQƏTƏN işlətdiyi retrieval dəyərləri — istədiyimiz deyil,
    # qurulmuş pipeline-dan geri oxunanı. Fərq ola bilər (SUT `validate()`
    # edir), və manifestə düşməli olan faktiki dəyərdir.
    retrieval: Mapping[str, Any] = field(default_factory=dict)
    # SORĞULANAN İNDEKSİN ÖZÜ haqqında fakt — `retrieval` isə NİYYƏTDİR.
    # Fərq vacibdir: `chunk_size` `sut_settings`-dən oxunur, yəni TƏZƏ
    # ingest-in nə edəcəyini bildirir. `CHUNK_SIZE=500` verib `PERSIST_DIR`-i
    # köhnə 800/200 indeksinə yönəltmək manifestə `chunk_size: 500` yazır,
    # halbuki 800/200 chunk-lar xidmət olunur. Barmaq izi bunu tutur.
    index: Mapping[str, Any] = field(default_factory=dict)


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

        self._info = SutInfo(
            commit=actual,
            chunk_count=count,
            sut_path=str(path),
            retrieval=_effective_retrieval(pipeline),
            index=_index_identity(pipeline),
        )
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
        # Retrieval override-ları eyni AÇIQ kanaldan gedir. Təyin olunmayan
        # parametr sözlükdə YOXDUR, yəni SUT öz dəyərini saxlayır.
        sut_settings = SutSettings.load(
            openai_api_key=self.settings.require_openai_key(),
            llm_model=self.settings.generator_model,
            embedding_model=self.settings.embedding_model,
            grounding_mode=self.settings.grounding_mode,
            **self.settings.retrieval_overrides(),
        )
        return RagPipeline(
            sut_settings,
            store=wrap_store(VectorStore(sut_settings)),
            llm=wrap_llm(build_llm(sut_settings)),
        )


def _effective_retrieval(pipeline: Any) -> dict[str, Any]:
    """Qurulmuş pipeline-dan faktiki retrieval dəyərlərini geri oxuyur.

    `getattr` müdafiəsi test saxtakarları üçündür: testlərdəki pipeline
    `settings` daşımır və daşımalı da deyil — o halda boş sözlük qayıdır və
    manifestdə sahə sadəcə görünmür (uydurulmuş dəyər yazılmır).
    """
    sut_settings = getattr(pipeline, "settings", None)
    if sut_settings is None:
        return {}
    out: dict[str, Any] = {}
    for name in (
        "top_k",
        "relevance_threshold",
        "soft_floor_margin",
        "lexical_threshold",
        "hybrid_retrieval",
        # İNDEKS KİMLİYİ. Chunking dəyişikliyi qapıya toxunmur, amma ölçülən
        # sistemi dəyişir: fərqli chunk-lanmış indeks fərqli mətn qaytarır.
        # Bunlar artefaktda olmasa, «hansı indeksdə ölçüldü?» sualının cavabı
        # yalnız işlədən adamın yaddaşında qalardı.
        "chunk_size",
        "chunk_overlap",
        "collection_name",
    ):
        value = getattr(sut_settings, name, None)
        if value is not None:
            out[name] = value
    # `persist_dir` MÜTLƏQ yoldur — manifestə maşından asılı yol yazmaq
    # `sut_path`-də düzəldilən səhvin təkrarı olardı. Yalnız adı saxlanılır.
    persist_dir = getattr(sut_settings, "persist_dir", None)
    if persist_dir is not None:
        out["persist_dir_name"] = Path(str(persist_dir)).name
    # `soft_floor` törəmə dəyərdir (astana − marja, 0-da kəsilir). Onu ayrıca
    # yazırıq, çünki hesabatı oxuyan adam üçün əsl qapı BUDUR.
    soft_floor = getattr(sut_settings, "soft_floor", None)
    if soft_floor is not None:
        out["soft_floor"] = round(float(soft_floor), 4)
    return out


def index_fingerprint(ids: Iterable[str]) -> str:
    """Sıralanmış chunk ID dəstinin sha256-sı (16 simvol).

    SUT-da `chunk_id` = sha1(mənbə|mətn) olduğu üçün ID-lər MƏZMUNDAN
    törəyir: fərqli chunking = fərqli mətn = fərqli ID dəsti = fərqli
    barmaq izi. `tools/retrieval_experiments.py` eyni düsturu işlədir, ona
    görə probe artefaktı ilə run manifesti birləşdirilə bilir — «bu run
    məhz eksperimentin ölçdüyü indeksdədir» iddiası yoxlanan olur.
    """
    digest = hashlib.sha256()
    for chunk_id in sorted(ids):
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def _index_identity(pipeline: Any) -> dict[str, Any]:
    """Sorğulanan kolleksiyanın ÖZÜ haqqında fakt.

    Şəbəkəyə çıxmır, embedding çağırmır, pul xərcləmir.

    Oxuna bilməsə preflight DAYANMIR: barmaq izi sübutdur, qapı deyil.
    Sahə boş qalır və hesabat «təsdiqlənə bilmir» deyir — uydurulmuş dəyər
    yazmaqdansa boşluğu etiraf etmək doğrudur.
    """
    store = getattr(pipeline, "store", None)
    if store is None:
        return {}
    out: dict[str, Any] = {}
    try:
        # SUT-un öz oxuma yolu (`rag/store.py`-dakı `_lexical_index` ilə eyni):
        # `include=[]` yalnız ID qaytarır, mətn və vektor yox.
        ids = store._store.get(include=[]).get("ids", [])
    except Exception:  # noqa: BLE001 — sübut əldə edilməsə, iddia da edilmir
        return out
    if ids:
        out["sha256"] = index_fingerprint(ids)
        out["id_count"] = len(ids)
    persist_dir = getattr(getattr(pipeline, "settings", None), "persist_dir", None)
    if persist_dir is not None:
        out["persist_dir_name"] = Path(str(persist_dir)).name
    return out


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
