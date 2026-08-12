"""Ölçmə sarğılarının davranışı — SUT-a toxunmadan token, gecikmə və rol.

Bu testlərin heç biri şəbəkəyə çıxmır: `FakeInnerLLM` sarğının içinə
inyeksiya olunur və hər çağırışın MƏHZ hansı mesajlarla getdiyini saxlayır.
"""

from __future__ import annotations

import pytest

from eval.instrument import (
    CORRECTION,
    GROUNDING_JUDGE,
    SYNTHESIS,
    UNKNOWN,
    InstrumentedLLM,
    InstrumentedStore,
    RoleMap,
    sha256_text,
)
from eval.observation import CallRecorder
from eval.variants import FewShotExample, PromptVariant, variant_sentinel
from tests.conftest import FakeAIMessage, FakeInnerLLM

SYS = "Sən sual-cavab köməkçisisən. Qaydalar: ..."
GROUNDING_SYS = "Sən NLI hakimisən. BLOKLAR və CAVAB verilir."

ROLES = RoleMap.from_prompts(synthesis=SYS, grounding=GROUNDING_SYS)

VARIANT = PromptVariant(
    id="v1",
    label="Test variantı",
    system_suffix="Əlavə qayda: ədədi faktı mənbədəki dəqiqliklə yaz.",
    few_shot=(
        FewShotExample(
            user="Nümunə sual?",
            assistant="Nümunə cavab [1].",
            source_case_ids=("dev_normal_rate_limit_free",),
        ),
    ),
)


def build(
    *,
    variant: PromptVariant | None = None,
    responses: list | None = None,
) -> tuple[InstrumentedLLM, FakeInnerLLM, CallRecorder]:
    inner = FakeInnerLLM(responses=responses or [])
    recorder = CallRecorder()
    llm = InstrumentedLLM(
        inner=inner,
        recorder=recorder,
        roles=ROLES,
        variant=variant or PromptVariant(id="baseline", label="Baseline"),
        model="gpt-4o-mini",
    )
    return llm, inner, recorder


def synthesis_messages() -> list[dict]:
    return [
        {"role": "system", "content": SYS},
        {"role": "user", "content": "Free planında limit nədir?"},
    ]


def correction_messages() -> list[dict]:
    return synthesis_messages() + [
        {"role": "assistant", "content": "Əvvəlki cavab [5]."},
        {"role": "user", "content": "Cavabında mövcud olmayan mənbəyə istinad var."},
    ]


def grounding_messages() -> list[dict]:
    return [
        {"role": "system", "content": GROUNDING_SYS},
        {"role": "user", "content": "BLOKLAR:\n[1] ...\n\nCAVAB:\n..."},
    ]


# --- rol təsnifatı ---------------------------------------------------------


def test_ilk_cagiris_SINTEZ_kimi_tanınır() -> None:
    llm, _, recorder = build()
    llm.invoke(synthesis_messages())
    assert recorder.llm_calls[0].role == SYNTHESIS


def test_assistant_novbesi_olan_cagiris_KORREKSIYA_dir() -> None:
    """Pipeline korreksiyanı eyni siyahıya əlavə edir — rol oradan oxunur."""
    llm, _, recorder = build()
    llm.invoke(correction_messages())
    assert recorder.llm_calls[0].role == CORRECTION


def test_grounding_hakimi_AYRI_rol_dur() -> None:
    llm, _, recorder = build()
    llm.invoke(grounding_messages())
    assert recorder.llm_calls[0].role == GROUNDING_JUDGE


def test_taninmayan_system_prompt_UNKNOWN_dur() -> None:
    llm, _, recorder = build()
    llm.invoke([{"role": "system", "content": "başqa bir prompt"}])
    assert recorder.llm_calls[0].role == UNKNOWN


# --- variantın tətbiqi -----------------------------------------------------


def test_variant_SINTEZ_cagirisina_tetbiq_olunur() -> None:
    llm, inner, _ = build(variant=VARIANT)
    llm.invoke(synthesis_messages())

    sent = inner.calls[0]
    assert VARIANT.system_suffix in sent[0]["content"]
    assert variant_sentinel("v1") in sent[0]["content"]
    assert sent[1]["content"] == "Nümunə sual?"
    assert sent[2]["content"] == "Nümunə cavab [1]."


def test_variant_GROUNDING_hakimine_TETBIQ_OLUNMUR() -> None:
    """Grounding hakiminin system prompt-u başqadır.

    Variantı ora da tətbiq etmək ölçülən sistemi qeyri-müəyyən edərdi:
    prompt dəyişikliyinin sintezə yoxsa grounding qatına təsir etdiyi
    ayırd edilə bilməzdi.
    """
    llm, inner, _ = build(variant=VARIANT)
    llm.invoke(grounding_messages())

    sent = inner.calls[0]
    assert sent == grounding_messages()


def test_variant_TANINMAYAN_prompta_tetbiq_olunmur() -> None:
    """Fail-safe: tanınmayan prompt heç vaxt səssizcə dəyişdirilmir."""
    llm, inner, _ = build(variant=VARIANT)
    original = [{"role": "system", "content": "naməlum"}]
    llm.invoke(original)
    assert inner.calls[0] == original


def test_giris_siyahisi_DEYISDIRILMIR() -> None:
    """Pipeline `messages`-i korreksiya arasında YERİNDƏ genişləndirir.

    Sarğı arqumenti dəyişdirsəydi, 2-ci cəhd ikiqat çevrilmiş girişlə
    gedərdi və few-shot nümunələri hər cəhddə təkrarlanardı.
    """
    llm, _, _ = build(variant=VARIANT)
    messages = synthesis_messages()
    snapshot = [dict(m) for m in messages]

    llm.invoke(messages)

    assert messages == snapshot


def test_tekrar_cagiris_few_shotu_IKIQAT_ELAVE_ETMIR() -> None:
    llm, inner, _ = build(variant=VARIANT)
    llm.invoke(synthesis_messages())
    first_length = len(inner.calls[0])

    llm.invoke(synthesis_messages())
    assert len(inner.calls[1]) == first_length


# --- token və gecikmə ------------------------------------------------------


def test_token_usage_metadata_dan_oxunur() -> None:
    llm, _, recorder = build(
        responses=[
            FakeAIMessage(
                content="Cavab [1].",
                usage_metadata={
                    "input_tokens": 1200,
                    "output_tokens": 90,
                    "input_token_details": {"cache_read": 300},
                },
            )
        ]
    )
    llm.invoke(synthesis_messages())

    call = recorder.llm_calls[0]
    assert (call.input_tokens, call.output_tokens) == (1200, 90)
    assert call.cached_read_tokens == 300
    assert call.usage_source == "usage_metadata"


def test_token_response_metadata_ya_GERI_CEKILIR() -> None:
    llm, _, recorder = build(
        responses=[
            FakeAIMessage(
                content="Cavab [1].",
                usage_metadata=None,
                response_metadata={
                    "token_usage": {"prompt_tokens": 800, "completion_tokens": 40}
                },
            )
        ]
    )
    llm.invoke(synthesis_messages())

    call = recorder.llm_calls[0]
    assert (call.input_tokens, call.output_tokens) == (800, 40)
    assert call.usage_source == "response_metadata"


def test_token_YOXDURSA_missing_kimi_isarelenir() -> None:
    """Sıfır uydurmaq xərc hesabatını SƏSSİZCƏ aşağı göstərərdi."""
    llm, _, recorder = build(
        responses=[FakeAIMessage(content="Cavab [1].", usage_metadata=None)]
    )
    llm.invoke(synthesis_messages())

    call = recorder.llm_calls[0]
    assert call.usage_source == "missing"
    assert (call.input_tokens, call.output_tokens) == (0, 0)


def test_gecikme_ve_hashler_qeyd_olunur() -> None:
    llm, _, recorder = build()
    llm.invoke(synthesis_messages())

    call = recorder.llm_calls[0]
    assert call.latency_ms >= 0.0
    assert call.system_sha256 == sha256_text(SYS)
    assert call.prompt_sha256
    assert call.model == "gpt-4o-mini"
    assert call.ok is True


def test_xeta_QEYD_OLUNUR_ve_yeniden_atilir() -> None:
    llm, _, recorder = build(responses=[RuntimeError("şəbəkə xətası")])

    with pytest.raises(RuntimeError):
        llm.invoke(synthesis_messages())

    call = recorder.llm_calls[0]
    assert call.ok is False
    assert call.error_type == "RuntimeError"


def test_call_id_ler_FERQLIDIR() -> None:
    llm, _, recorder = build()
    llm.invoke(synthesis_messages())
    llm.invoke(synthesis_messages())
    assert recorder.llm_calls[0].call_id != recorder.llm_calls[1].call_id


# --- retrieval sarğısı -----------------------------------------------------


class FakeChunk:
    def __init__(self, score: float) -> None:
        self.score = score


class FakeStore:
    def __init__(self, chunks: list | None = None, count: int = 7) -> None:
        self.chunks = chunks if chunks is not None else [FakeChunk(0.81), FakeChunk(0.55)]
        self._count = count
        self.calls: list[tuple[str, str, int]] = []

    def search(self, query: str, k: int) -> list:
        self.calls.append(("dense", query, k))
        return self.chunks

    def hybrid_search(self, query: str, k: int) -> list:
        self.calls.append(("hybrid", query, k))
        return self.chunks

    def count(self) -> int:
        return self._count


def test_hibrid_axtaris_qeyd_olunur() -> None:
    recorder = CallRecorder()
    store = InstrumentedStore(inner=FakeStore(), recorder=recorder)

    result = store.hybrid_search("Free plan limiti?", k=4)

    assert len(result) == 2
    call = recorder.retrieval_calls[0]
    assert call.mode == "hybrid"
    assert call.k == 4
    assert call.returned == 2
    assert call.top_score == pytest.approx(0.81)
    assert call.query_chars == len("Free plan limiti?")


def test_dense_axtaris_MODU_ile_ayrilir() -> None:
    recorder = CallRecorder()
    store = InstrumentedStore(inner=FakeStore(), recorder=recorder)
    store.search("sual", k=2)
    assert recorder.retrieval_calls[0].mode == "dense"


def test_count_OTURULUR_ve_qeyd_olunmur() -> None:
    """`count()` pulsuzdur və ölçülən çağırış deyil."""
    recorder = CallRecorder()
    store = InstrumentedStore(inner=FakeStore(count=42), recorder=recorder)
    assert store.count() == 42
    assert recorder.retrieval_calls == []


def test_bos_netice_top_score_SIFIR() -> None:
    recorder = CallRecorder()
    store = InstrumentedStore(inner=FakeStore(chunks=[]), recorder=recorder)
    store.hybrid_search("sual", k=4)
    assert recorder.retrieval_calls[0].top_score == 0.0
    assert recorder.retrieval_calls[0].returned == 0


# --- atılan chunk-ların balları qeyd olunur ---------------------------------


def test_retrieval_BUTUN_ballari_qeyd_olunur_astanadan_asagilar_da() -> None:
    """Astanadan aşağı ballar da saxlanmalıdır.

    `SutObservation.chunks` yalnız qəbul edilənləri daşıyır. Atılanların balı
    itsəydi, «astananı nə qədər endirmək lazımdır?» sualına yalnız yeni
    (pullu) run ilə cavab vermək olardı.
    """
    from eval.instrument import InstrumentedStore
    from eval.observation import CallRecorder

    class _Chunk:
        def __init__(self, score: float) -> None:
            self.score = score

    class _Store:
        def search(self, query, k):
            return [_Chunk(0.51), _Chunk(0.19), _Chunk(0.40), _Chunk(0.33)]

    recorder = CallRecorder()
    store = InstrumentedStore(inner=_Store(), recorder=recorder)
    store.search("sual", k=4)

    _llm, retrievals = recorder.drain()
    call = retrievals[0]
    assert call.scores == (0.51, 0.40, 0.33, 0.19), "azalan sırada olmalıdır"
    assert call.top_score == 0.51
    assert call.returned == 4


def test_bos_neticede_scores_bosdur_top_score_sifirdir() -> None:
    from eval.instrument import InstrumentedStore
    from eval.observation import CallRecorder

    class _Empty:
        def search(self, query, k):
            return []

    recorder = CallRecorder()
    InstrumentedStore(inner=_Empty(), recorder=recorder).search("sual", k=4)
    _llm, retrievals = recorder.drain()
    call = retrievals[0]
    assert call.scores == ()
    assert call.top_score == 0.0
