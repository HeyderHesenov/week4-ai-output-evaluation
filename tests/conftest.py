"""Test yardımçıları — bütün suite açarsız və şəbəkəsiz işləyir.

PRİNSİP: heç bir test nə OpenAI, nə Anthropic açarı tələb etməməlidir.
Bu sadəcə rahatlıq deyil — CI-də açar olmadığı üçün suite həqiqətən
oradakı kimi işləyir və "lokalda keçir, CI-də keçmir" sinfi aradan qalxır.
Bütün xarici sərhədlər inyeksiya olunur: `InstrumentedLLM(inner=...)`,
`AnthropicJudge(client=...)`, `RagSut(pipeline_factory=...)`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pytest

from eval.config import Settings
from eval.dataset import Dataset, EvalCase, Expected, NumericClaim, dataset_hash
from eval.observation import ChunkView, LlmCall, RetrievalCall, SutObservation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fabriklər
# ---------------------------------------------------------------------------


def make_settings(**overrides: Any) -> Settings:
    """Şəbəkəsiz, saxta açarlı Settings."""
    defaults: dict[str, Any] = {
        "openai_api_key": "sk-test-fake-openai-key-0000000000",
        "anthropic_api_key": "sk-ant-test-fake-anthropic-key-000",
        "sut_path": PROJECT_ROOT / "vendor" / "week2-rag-document-qa",
        "sut_commit": "0" * 40,
        "grounding_mode": "strict",
        "generator_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "judge_model": "claude-opus-5",
        "judge_effort": "low",
        "judge_max_tokens": 1500,
        "judge_pass_threshold": 2,
        "judge_timeout": 5.0,
        "judge_max_retries": 2,
        "judge_retry_base_delay": 0.0,
        "judge_fallback_model": "",
        "dataset_path": PROJECT_ROOT / "data" / "testset.yaml",
        "split_manifest_path": PROJECT_ROOT / "data" / "split_manifest.json",
        "prices_path": PROJECT_ROOT / "data" / "prices.yaml",
        "variants_dir": PROJECT_ROOT / "data" / "variants",
        "runs_dir": PROJECT_ROOT / "runs",
        "logs_dir": PROJECT_ROOT / "logs",
        "repeats": 1,
    }
    defaults.update(overrides)
    settings = Settings(**defaults)
    settings.validate()
    return settings


def make_case(
    case_id: str = "case_test",
    *,
    question: str = "Test sualı nədir?",
    category: str = "normal",
    split: str = "dev",
    gradable: str = "deterministic",
    gold_sources: Sequence[str] = ("atlas_api_senedi.md",),
    gold_contains: Sequence[str] = (),
    **expected_kwargs: Any,
) -> EvalCase:
    numeric = tuple(
        n if isinstance(n, NumericClaim) else NumericClaim(**n)
        for n in expected_kwargs.pop("numeric", ())
    )
    expected = Expected(
        kind=expected_kwargs.pop("kind", "answer"),
        contains=tuple(expected_kwargs.pop("contains", ())),
        not_contains=tuple(expected_kwargs.pop("not_contains", ())),
        numeric=numeric,
        reason_in=tuple(expected_kwargs.pop("reason_in", ())),
        require_citation=expected_kwargs.pop("require_citation", True),
        min_distinct_cited_sources=expected_kwargs.pop("min_distinct_cited_sources", 0),
        rubric=expected_kwargs.pop("rubric", ""),
    )
    assert not expected_kwargs, f"naməlum arqument: {sorted(expected_kwargs)}"
    return EvalCase(
        id=case_id,
        question=question,
        category=category,
        split=split,
        gradable=gradable,
        expected=expected,
        gold_sources=tuple(gold_sources),
        gold_contains=tuple(gold_contains),
    )


def make_dataset(cases: Sequence[EvalCase]) -> Dataset:
    return Dataset(cases=tuple(cases), sha256=dataset_hash(cases))


def make_chunk(
    label: int = 1,
    *,
    source: str = "atlas_api_senedi.md",
    text: str = "Nümunə chunk mətni.",
    score: float = 0.80,
    lexical_score: float = 0.5,
    page: int = 0,
    chunk_index: int = 0,
) -> ChunkView:
    return ChunkView(
        label=label,
        source=source,
        page=page,
        chunk_index=chunk_index,
        score=score,
        lexical_score=lexical_score,
        chunk_id=f"chunk-{source}-{chunk_index}",
        text=text,
    )


def make_observation(
    case_id: str = "case_test",
    *,
    answer_text: str = "Nümunə cavab [1].",
    refused: bool = False,
    reason: str = "answered",
    cited_labels: Sequence[int] = (1,),
    chunks: Sequence[ChunkView] | None = None,
    llm_calls: Sequence[LlmCall] = (),
    retrieval_calls: Sequence[RetrievalCall] = (),
    **kwargs: Any,
) -> SutObservation:
    return SutObservation(
        case_id=case_id,
        repeat=kwargs.pop("repeat", 1),
        question=kwargs.pop("question", "Test sualı nədir?"),
        answer_text=answer_text,
        refused=refused,
        reason=reason,
        cited_labels=tuple(cited_labels),
        invalid_citations=tuple(kwargs.pop("invalid_citations", ())),
        rescued_labels=tuple(kwargs.pop("rescued_labels", ())),
        unsupported_claims=tuple(kwargs.pop("unsupported_claims", ())),
        grounding_detail=kwargs.pop("grounding_detail", ""),
        top_score=kwargs.pop("top_score", 0.80),
        top_lexical=kwargs.pop("top_lexical", 0.5),
        threshold=kwargs.pop("threshold", 0.42),
        retrieved_count=kwargs.pop("retrieved_count", 4),
        attempts=kwargs.pop("attempts", 1),
        chunks=tuple(chunks if chunks is not None else (make_chunk(),)),
        llm_calls=tuple(llm_calls),
        retrieval_calls=tuple(retrieval_calls),
        total_ms=kwargs.pop("total_ms", 1200.0),
        error=kwargs.pop("error", ""),
    )


def make_llm_call(
    *,
    role: str = "synthesis",
    model: str = "gpt-4o-mini",
    input_tokens: int = 900,
    output_tokens: int = 120,
    latency_ms: float = 800.0,
    cached_read_tokens: int = 0,
    ok: bool = True,
) -> LlmCall:
    return LlmCall(
        call_id=f"{role}-1",
        role=role,
        provider="openai",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_read_tokens=cached_read_tokens,
        latency_ms=latency_ms,
        ok=ok,
        usage_source="usage_metadata",
    )


# ---------------------------------------------------------------------------
# Fake-lər
# ---------------------------------------------------------------------------


@dataclass
class FakeAIMessage:
    """LangChain `AIMessage`-in ölçmə üçün lazım olan hissəsi."""

    content: str = "Saxta cavab [1]."
    usage_metadata: dict[str, Any] | None = None
    response_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeInnerLLM:
    """`InstrumentedLLM`-in sarıdığı model. Hər çağırışı qeyd edir."""

    responses: list[Any] = field(default_factory=list)
    calls: list[list[dict[str, Any]]] = field(default_factory=list)
    default_response: Any = None

    def invoke(self, messages: Any) -> Any:
        # Sarğının siyahını dəyişdirmədiyini yoxlaya bilmək üçün DƏRİN
        # nüsxə saxlanılır.
        self.calls.append([dict(m) for m in messages])
        if self.responses:
            item = self.responses.pop(0)
            # İstisna obyekti "cavab" kimi verilirsə ATILIR — şəbəkə xətası
            # ssenarilərini skriptləməyin yolu budur (`FakeAnthropicClient`
            # eyni konvensiyanı işlədir).
            if isinstance(item, BaseException):
                raise item
            return item
        if self.default_response is not None:
            return self.default_response
        return FakeAIMessage(
            usage_metadata={"input_tokens": 900, "output_tokens": 120}
        )


@dataclass
class FakeAnthropicResponse:
    """`client.messages.create()` cavabının lazımi hissəsi."""

    text: str = ""
    stop_reason: str = "end_turn"
    model: str = "claude-opus-5"
    input_tokens: int = 800
    output_tokens: int = 150
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    stop_details: Any = None

    @property
    def content(self) -> list[Any]:
        if not self.text:
            return []
        return [_FakeTextBlock(self.text)]

    @property
    def usage(self) -> Any:
        return _FakeUsage(
            self.input_tokens,
            self.output_tokens,
            self.cache_read_input_tokens,
            self.cache_creation_input_tokens,
        )


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeUsage:
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int


class FakeAnthropicClient:
    """Skriptlənə bilən Anthropic klienti; hər sorğunu saxlayır."""

    def __init__(self, responses: Sequence[Any] | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self._responses = list(responses or [])
        self.messages = _FakeMessages(self)

    def push(self, response: Any) -> None:
        self._responses.append(response)

    def _next(self, kwargs: dict[str, Any]) -> Any:
        self.requests.append(kwargs)
        if not self._responses:
            return FakeAnthropicResponse(
                text=json.dumps(
                    {
                        "score": 3,
                        "faithful": True,
                        "complete": True,
                        "reason": "hər şey qaydasındadır",
                        "flags": [],
                    },
                    ensure_ascii=False,
                )
            )
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeMessages:
    def __init__(self, client: FakeAnthropicClient) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._next(kwargs)


@pytest.fixture
def settings() -> Settings:
    return make_settings()
