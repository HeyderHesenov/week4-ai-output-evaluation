"""Bir sorğunun tam müşahidəsi — ölçmə qatının ortaq lüğəti.

Bu modul QƏSDƏN asılılıqsızdır (yalnız stdlib). `instrument`, `sut`,
`graders`, `rootcause`, `metrics` və `artifacts` hamısı buradakı tiplərlə
danışır; beləliklə qraderi sınamaq üçün nə LangChain, nə Chroma, nə də
şəbəkə lazım deyil — sintetik `SutObservation` qurmaq kifayətdir.

XAM SÜBUT / TÖRƏMƏ AYRIMI
--------------------------
`SutObservation` XAM sübutdur: sistemin nə qaytardığı. Qiymət, uğursuzluq
sinfi və metriklər bundan TÖRƏYİR və heç vaxt onun içinə yazılmır. Bu
ayrım `reclassify` əmrini mümkün edir: taksonomiya dəyişəndə saxlanmış
müşahidələr üzərində yenidən hesablamaq kifayətdir, sistemi (və pulu)
yenidən işlətmək lazım deyil.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChunkView:
    """Çəkilmiş bir chunk-ın qiymətləndirmə üçün lazım olan görüntüsü.

    `label` HƏR SORĞUDA yenidən nömrələnir (SUT `replace(chunk, label=...)`
    edir), yəni `label` retrieval sırası DEYİL və saxlanmış chunk siyahısı
    olmadan mənasızdır. Kök-səbəb qaydaları buna görə həmişə label → chunk
    həllini bu siyahı üzərindən aparır.
    """

    label: int
    source: str
    page: int
    chunk_index: int
    score: float
    lexical_score: float
    chunk_id: str
    text: str


@dataclass(frozen=True)
class LlmCall:
    """SUT-un etdiyi bir LLM çağırışı."""

    call_id: str
    role: str  # "synthesis" | "correction" | "grounding_judge"
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_read_tokens: int
    latency_ms: float
    ok: bool
    error_type: str = ""
    system_sha256: str = ""
    prompt_sha256: str = ""
    response_text: str = ""
    # "usage_metadata" | "response_metadata" | "missing"
    usage_source: str = "missing"


@dataclass(frozen=True)
class RetrievalCall:
    """SUT-un etdiyi bir vektor axtarışı."""

    mode: str  # "dense" | "hybrid"
    k: int
    latency_ms: float
    returned: int
    top_score: float
    query_chars: int
    # ÇƏKİLƏN BÜTÜN chunk-ların balları, azalan sırada.
    #
    # NİYƏ LAZIMDIR: `SutObservation.chunks` yalnız astanadan KEÇƏNLƏRİ saxlayır.
    # Atılanların balı heç yerdə qalmırdı, ona görə «astananı nə qədər endirsək
    # çatışmayan sənəd kontekstə düşərdi?» sualına yalnız yeni (pullu) run ilə
    # cavab vermək olurdu. Bu sahə həmin sualı SAXLANMIŞ artefaktdan
    # cavablandırır. Mətn saxlanmır — yalnız rəqəmlər.
    scores: tuple[float, ...] = ()


@dataclass(frozen=True)
class SutObservation:
    """Bir case-in bir icrasının xam nəticəsi."""

    case_id: str
    repeat: int
    question: str
    answer_text: str
    refused: bool
    reason: str
    cited_labels: tuple[int, ...] = ()
    invalid_citations: tuple[int, ...] = ()
    rescued_labels: tuple[int, ...] = ()
    unsupported_claims: tuple[str, ...] = ()
    grounding_detail: str = ""
    top_score: float = 0.0
    top_lexical: float = 0.0
    threshold: float = 0.0
    retrieved_count: int = 0
    attempts: int = 0
    chunks: tuple[ChunkView, ...] = ()
    llm_calls: tuple[LlmCall, ...] = ()
    retrieval_calls: tuple[RetrievalCall, ...] = ()
    total_ms: float = 0.0
    error: str = ""

    @property
    def llm_ms(self) -> float:
        return sum(c.latency_ms for c in self.llm_calls)

    @property
    def retrieval_ms(self) -> float:
        return sum(c.latency_ms for c in self.retrieval_calls)

    @property
    def overhead_ms(self) -> float:
        """Qalan vaxt — parse, grounding yoxlaması, obyekt qurulması.

        Üç sütun (retrieval + llm + overhead) hesabatda yan-yana çap olunur
        ki, cəmin bağlandığı GÖRÜNSÜN. Bağlanmayan hesab gizli ölçmə
        boşluğu deməkdir.
        """
        return max(0.0, self.total_ms - self.llm_ms - self.retrieval_ms)

    @property
    def cited_chunks(self) -> tuple[ChunkView, ...]:
        labels = set(self.cited_labels)
        return tuple(c for c in self.chunks if c.label in labels)

    @property
    def cited_sources(self) -> frozenset[str]:
        return frozenset(c.source for c in self.cited_chunks)

    @property
    def regenerated(self) -> bool:
        return self.attempts > 1


@dataclass
class CallRecorder:
    """Bir sorğu ərzində baş verən çağırışları toplayır.

    Dəyişkəndir (accumulator), qalan hər şey dəyişməzdir — bu qəsdəndir:
    yalnız bir dəyişkən obyekt olsa, "kim nəyi dəyişdi" sualı bir yerdə
    cavablanır.
    """

    llm_calls: list[LlmCall] = field(default_factory=list)
    retrieval_calls: list[RetrievalCall] = field(default_factory=list)

    def record_llm(self, call: LlmCall) -> None:
        self.llm_calls.append(call)

    def record_retrieval(self, call: RetrievalCall) -> None:
        self.retrieval_calls.append(call)

    def drain(self) -> tuple[tuple[LlmCall, ...], tuple[RetrievalCall, ...]]:
        llm = tuple(self.llm_calls)
        retrieval = tuple(self.retrieval_calls)
        self.llm_calls.clear()
        self.retrieval_calls.clear()
        return llm, retrieval
