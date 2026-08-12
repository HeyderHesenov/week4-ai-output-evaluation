"""Ölçmə sarğıları — SUT-un bir sətrini belə dəyişmədən token, gecikmə və rol.

İKİ TİKİŞ, HƏR İKİSİ ONSUZ DA MÖVCUD İDİ
-----------------------------------------
`RagPipeline(settings, store=..., llm=...)` (vendor/.../rag/pipeline.py:262)
hər iki xarici sərhədi inyeksiya kimi qəbul edir — Week 2 bunu öz testləri
şəbəkəsiz işləsin deyə etmişdi. Biz həmin qapıdan girib arada dayanırıq:
monkey-patch yoxdur, SUT faylına redaktə yoxdur, pin edilmiş commit toxunulmaz
qalır.

NİYƏ ROL SİSTEM PROMPT-UNUN HASH-I İLƏ TƏYİN OLUNUR
----------------------------------------------------
Bir sorğu ərzində SUT eyni `llm` obyektini ÜÇ fərqli məqsədlə çağıra bilir:
sintez, korreksiya və (GROUNDING_MODE=llm olduqda) NLI hakimi. Onları
ayırmasaq:

  - token hesabatı "hansı qat nə qədər yeyir" sualına cavab verə bilməz;
  - və daha pisi, prompt VARİANTI grounding hakiminin promptuna da yapışardı.

İkincisi ölçmənin özünü korlayır: variantın pass-rate-ə təsiri sintez
prompt-undan yoxsa grounding qatının dəyişməsindən gəldiyi ayırd edilə
bilməzdi. Buna görə rol tanınmayan hər prompt üçün variant TƏTBİQ OLUNMUR
(fail-safe) — səssiz dəyişiklik ən bahalı ölçmə səhvidir.

TOKEN MƏNBƏYİ SAXLANILIR, SIFIR UYDURULMUR
-------------------------------------------
LangChain token sayını əvvəlcə `usage_metadata`-da, köhnə/fərqli
provayderlərdə `response_metadata["token_usage"]`-da verir, bəzən heç
vermir. Üçüncü halda 0 yazmaq xərc hesabatını səssizcə aşağı göstərərdi,
buna görə `usage_source="missing"` işarələnir və hesabat bunu banner kimi
çıxarır.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .observation import CallRecorder, LlmCall, RetrievalCall
from .variants import IDENTITY_VARIANT, PromptVariant

SYNTHESIS = "synthesis"
CORRECTION = "correction"
GROUNDING_JUDGE = "grounding_judge"
UNKNOWN = "unknown"

# Variantın tətbiq oluna biləcəyi YEGANƏ rollar. Siyahı ağdır (allow-list):
# gələcəkdə yeni rol əlavə olunsa, o, default olaraq TOXUNULMAZ qalır.
VARIANT_ROLES = frozenset({SYNTHESIS, CORRECTION})


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RoleMap:
    """System prompt hash-ından rola xəritə.

    Hash-lar `RagSut` tərəfindən SUT-un HƏQİQİ sabitlərindən qurulur
    (`rag.pipeline.SYSTEM_INSTRUCTION`, `rag.grounding.JUDGE_SYSTEM`).
    Beləliklə "tanınmayan prompt" halı yalnız SUT dəyişəndə yaranır —
    və o zaman commit pin yoxlaması onsuz da run-ı dayandırmış olur.
    """

    synthesis_sha256: str = ""
    grounding_sha256: str = ""

    @classmethod
    def from_prompts(cls, *, synthesis: str, grounding: str = "") -> "RoleMap":
        return cls(
            synthesis_sha256=sha256_text(synthesis) if synthesis else "",
            grounding_sha256=sha256_text(grounding) if grounding else "",
        )

    def role_for(self, system_text: str, *, has_assistant_turn: bool) -> str:
        digest = sha256_text(system_text)
        if self.synthesis_sha256 and digest == self.synthesis_sha256:
            # Korreksiya sintezdən yalnız bir şeylə fərqlənir: siyahıda artıq
            # modelin öz cavabı var (pipeline `_correction_messages` əlavə edib).
            return CORRECTION if has_assistant_turn else SYNTHESIS
        if self.grounding_sha256 and digest == self.grounding_sha256:
            return GROUNDING_JUDGE
        return UNKNOWN


def _system_text(messages: Sequence[Mapping[str, Any]]) -> str:
    if not messages:
        return ""
    head = messages[0]
    if head.get("role") != "system":
        return ""
    return str(head.get("content", ""))


def _prompt_digest(messages: Sequence[Mapping[str, Any]]) -> str:
    """Bütün mesaj siyahısının kanonik hash-ı — replay üçün kimlik."""
    payload = json.dumps(
        [{"role": str(m.get("role", "")), "content": str(m.get("content", ""))} for m in messages],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return sha256_text(payload)


def _extract_usage(response: Any) -> tuple[int, int, int, str]:
    """(input, output, cached_read, mənbə)."""
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, Mapping) and usage:
        details = usage.get("input_token_details") or {}
        cached = 0
        if isinstance(details, Mapping):
            cached = int(details.get("cache_read", 0) or 0)
        return (
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
            cached,
            "usage_metadata",
        )

    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, Mapping):
        token_usage = metadata.get("token_usage")
        if isinstance(token_usage, Mapping) and token_usage:
            cached_details = token_usage.get("prompt_tokens_details") or {}
            cached = 0
            if isinstance(cached_details, Mapping):
                cached = int(cached_details.get("cached_tokens", 0) or 0)
            return (
                int(token_usage.get("prompt_tokens", 0) or 0),
                int(token_usage.get("completion_tokens", 0) or 0),
                cached,
                "response_metadata",
            )

    return 0, 0, 0, "missing"


@dataclass
class InstrumentedLLM:
    """`.invoke(messages)` çağırışını tutan sarğı.

    Sarğı ölçür və (yalnız icazə verilən rollarda) variantı tətbiq edir;
    başqa heç nə etmir. Xəta udulmur — qeyd olunur və yenidən atılır, çünki
    SUT-un öz retry/korreksiya məntiqi ondan asılıdır.
    """

    inner: Any
    recorder: CallRecorder = field(default_factory=CallRecorder)
    roles: RoleMap = field(default_factory=RoleMap)
    variant: PromptVariant = IDENTITY_VARIANT
    provider: str = "openai"
    model: str = ""
    _counter: int = field(default=0, init=False, repr=False)
    # Variantın NEÇƏ DƏFƏ tətbiq olunduğu. `role_for` prompt-u tanımasa
    # (məsələn SUT-un system mesajı dinamik yığılırsa), variant səssizcə
    # tətbiq olunmur və run baseline ilə bayt-bayt eyni olur — hesabat isə
    # onu variant kimi göstərib McNemar hesablayır. Say sıfır qalırsa,
    # runner bunu görünən edir.
    applied_count: int = field(default=0, init=False)
    unknown_role_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.model:
            self.model = str(
                getattr(self.inner, "model_name", None)
                or getattr(self.inner, "model", None)
                or "naməlum"
            )

    def invoke(self, messages: Sequence[Mapping[str, Any]]) -> Any:
        system_text = _system_text(messages)
        has_assistant = any(m.get("role") == "assistant" for m in messages)
        role = self.roles.role_for(system_text, has_assistant_turn=has_assistant)

        if role in VARIANT_ROLES:
            outgoing = self.variant.apply(messages)
            if not self.variant.is_identity:
                self.applied_count += 1
        else:
            outgoing = [dict(m) for m in messages]
            if role == UNKNOWN and not self.variant.is_identity:
                self.unknown_role_count += 1

        self._counter += 1
        call_id = f"{role}-{self._counter}"
        started = time.perf_counter()
        try:
            response = self.inner.invoke(outgoing)
        except BaseException as exc:  # noqa: BLE001 — qeyd edib yenidən atırıq
            self.recorder.record_llm(
                LlmCall(
                    call_id=call_id,
                    role=role,
                    provider=self.provider,
                    model=self.model,
                    input_tokens=0,
                    output_tokens=0,
                    cached_read_tokens=0,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    ok=False,
                    error_type=type(exc).__name__,
                    system_sha256=sha256_text(system_text),
                    prompt_sha256=_prompt_digest(outgoing),
                )
            )
            raise

        latency_ms = (time.perf_counter() - started) * 1000.0
        input_tokens, output_tokens, cached_read, usage_source = _extract_usage(response)
        self.recorder.record_llm(
            LlmCall(
                call_id=call_id,
                role=role,
                provider=self.provider,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_read_tokens=cached_read,
                latency_ms=latency_ms,
                ok=True,
                system_sha256=sha256_text(system_text),
                prompt_sha256=_prompt_digest(outgoing),
                response_text=str(getattr(response, "content", "")),
                usage_source=usage_source,
            )
        )
        return response


@dataclass
class InstrumentedStore:
    """`VectorStore` proxy-si — axtarış çağırışlarını qeyd edir.

    `count()` qəsdən qeyd olunmur: o, lokal sayğacdır, nə şəbəkə, nə xərc.
    Onu da ölçmək retrieval statistikasını mənasız şişirdərdi.
    """

    inner: Any
    recorder: CallRecorder = field(default_factory=CallRecorder)

    def search(self, query: str, k: int) -> Any:
        return self._timed("dense", self.inner.search, query, k)

    def hybrid_search(self, query: str, k: int) -> Any:
        return self._timed("hybrid", self.inner.hybrid_search, query, k)

    def count(self) -> int:
        return int(self.inner.count())

    def _timed(self, mode: str, func: Any, query: str, k: int) -> Any:
        started = time.perf_counter()
        chunks = func(query, k=k)
        latency_ms = (time.perf_counter() - started) * 1000.0
        scores = tuple(sorted((float(c.score) for c in chunks), reverse=True))
        self.recorder.record_retrieval(
            RetrievalCall(
                mode=mode,
                k=k,
                latency_ms=latency_ms,
                returned=len(chunks),
                top_score=scores[0] if scores else 0.0,
                query_chars=len(query),
                scores=scores,
            )
        )
        return chunks

    def __getattr__(self, name: str) -> Any:
        """Qalan hər şey SUT-un öz store-una ötürülür."""
        try:
            inner = object.__getattribute__(self, "inner")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(inner, name)
