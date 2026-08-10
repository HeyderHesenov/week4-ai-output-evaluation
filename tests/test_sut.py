"""Qiymətləndirilən sistemin adapteri — preflight və müşahidəyə çevirmə.

SUT-un özü (langchain, chroma) burada import EDİLMİR: `pipeline_factory`
inyeksiyası sayəsində bütün test dəsti həmin asılılıqlar olmadan işləyir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from eval.errors import SutError
from eval.instrument import RoleMap
from eval.sut import RagSut
from tests.conftest import PROJECT_ROOT, FakeAIMessage, FakeInnerLLM, make_case, make_settings

SYS = "SUT sintez prompt-u"
GOOD_COMMIT = "0" * 40


@dataclass
class FakeChunk:
    label: int = 1
    text: str = "Free planı: dəqiqədə 60 sorğu."
    source: str = "atlas_api_senedi.md"
    page: int = 0
    chunk_index: int = 3
    score: float = 0.77
    lexical_score: float = 0.51
    chunk_id: str = "atlas-3"


@dataclass
class FakeAnswer:
    """`rag.pipeline.Answer`-in duck-type qarşılığı."""

    question: str = "Free planında limit nədir?"
    text: str = "Free planında dəqiqədə 60 sorğu [1]."
    refused: bool = False
    reason: str = "answered"
    chunks: list = field(default_factory=lambda: [FakeChunk()])
    cited_labels: list = field(default_factory=lambda: [1])
    invalid_citations: list = field(default_factory=list)
    top_score: float = 0.77
    threshold: float = 0.42
    retrieved_count: int = 4
    diagnostics: list = field(default_factory=list)
    unsupported_claims: list = field(default_factory=list)
    grounding_detail: str = "leksik örtük 0.62"
    top_lexical: float = 0.51
    rescued_labels: list = field(default_factory=list)
    attempts: int = 1


@dataclass
class FakeStore:
    chunk_count: int = 12

    def count(self) -> int:
        return self.chunk_count

    def hybrid_search(self, query: str, k: int) -> list:
        return [FakeChunk()]


@dataclass
class FakePipeline:
    store: Any
    llm: Any
    answer: Any = field(default_factory=FakeAnswer)
    raises: BaseException | None = None
    asked: list = field(default_factory=list)

    def ask(self, question: str, k: int | None = None) -> Any:
        self.asked.append(question)
        if self.raises is not None:
            raise self.raises
        # Həqiqi pipeline kimi: həm store, həm llm çağırılır ki, sarğıların
        # doğrudan qoşulduğu görünsün.
        self.store.hybrid_search(question, k=k or 4)
        self.llm.invoke([{"role": "system", "content": SYS}, {"role": "user", "content": question}])
        return self.answer


def build_sut(
    *,
    store: FakeStore | None = None,
    answer: Any = None,
    raises: BaseException | None = None,
    commit: str = GOOD_COMMIT,
    settings_overrides: dict | None = None,
) -> tuple[RagSut, dict]:
    made = {}

    def factory(wrap_llm, wrap_store):
        pipeline = FakePipeline(
            store=wrap_store(store or FakeStore()),
            llm=wrap_llm(FakeInnerLLM(default_response=FakeAIMessage())),
            answer=answer or FakeAnswer(),
            raises=raises,
        )
        made["pipeline"] = pipeline
        return pipeline

    sut = RagSut(
        make_settings(**(settings_overrides or {})),
        pipeline_factory=factory,
        commit_reader=lambda path: commit,
        roles=RoleMap.from_prompts(synthesis=SYS),
    )
    return sut, made


# --- preflight -------------------------------------------------------------


def test_commit_uygun_gelmirse_IMTINA() -> None:
    """«Week 2-ni dəyişdirmədən qiymətləndirdik» iddiası burada yoxlanır."""
    sut, made = build_sut(commit="a" * 40)

    with pytest.raises(SutError, match="commit"):
        sut.preflight()

    assert "pipeline" not in made, "pipeline pullu şəkildə qurulmamalı idi"


def test_bos_indeks_PULLU_cagirisdan_EVVEL_tutulur() -> None:
    sut, _ = build_sut(store=FakeStore(chunk_count=0))

    with pytest.raises(SutError, match="ingest"):
        sut.preflight()


def test_SUT_COMMIT_bos_ola_bilmez() -> None:
    sut, _ = build_sut(settings_overrides={"sut_commit": ""})

    with pytest.raises(SutError, match="SUT_COMMIT"):
        sut.preflight()


def test_olmayan_sut_yolu_tutulur() -> None:
    sut, _ = build_sut(settings_overrides={"sut_path": PROJECT_ROOT / "yoxdur"})

    with pytest.raises(SutError, match="tapılmadı"):
        sut.preflight()


def test_preflight_MELUMAT_qaytarir() -> None:
    sut, _ = build_sut()
    info = sut.preflight()
    assert info.commit == GOOD_COMMIT
    assert info.chunk_count == 12


def test_observe_preflighti_AVTOMATIK_isledir() -> None:
    sut, _ = build_sut(commit="b" * 40)
    with pytest.raises(SutError):
        sut.observe(make_case())


# --- müşahidəyə çevirmə ----------------------------------------------------


def test_cavab_musahideye_cevrilir() -> None:
    sut, _ = build_sut()
    obs = sut.observe(make_case("dev_normal_rate_limit_free"), repeat=2)

    assert obs.case_id == "dev_normal_rate_limit_free"
    assert obs.repeat == 2
    assert obs.answer_text == "Free planında dəqiqədə 60 sorğu [1]."
    assert obs.refused is False
    assert obs.reason == "answered"
    assert obs.cited_labels == (1,)
    assert obs.top_score == pytest.approx(0.77)
    assert obs.threshold == pytest.approx(0.42)
    assert obs.retrieved_count == 4
    assert obs.attempts == 1
    assert obs.grounding_detail == "leksik örtük 0.62"
    assert obs.error == ""


def test_chunklar_gorunuse_cevrilir() -> None:
    sut, _ = build_sut()
    obs = sut.observe(make_case())

    chunk = obs.chunks[0]
    assert chunk.label == 1
    assert chunk.source == "atlas_api_senedi.md"
    assert chunk.chunk_index == 3
    assert chunk.score == pytest.approx(0.77)
    assert chunk.lexical_score == pytest.approx(0.51)
    assert chunk.chunk_id == "atlas-3"
    assert obs.cited_sources == frozenset({"atlas_api_senedi.md"})


def test_cagirislar_musahideye_yazilir() -> None:
    sut, _ = build_sut()
    obs = sut.observe(make_case())

    assert len(obs.llm_calls) == 1
    assert len(obs.retrieval_calls) == 1
    assert obs.total_ms > 0.0


def test_cagirislar_MUSAHIDELER_arasinda_YIGILMIR() -> None:
    """`CallRecorder.drain()` hər müşahidədən sonra boşalmalıdır.

    Boşalmasaydı, 10-cu case-in hesabatında 10 sorğunun bütün çağırışları
    görünərdi və xərc case başına 10 dəfə şişərdi.
    """
    sut, _ = build_sut()
    sut.observe(make_case("c1"))
    second = sut.observe(make_case("c2"))

    assert len(second.llm_calls) == 1
    assert len(second.retrieval_calls) == 1


def test_pipeline_BIR_DEFE_qurulur() -> None:
    sut, made = build_sut()
    sut.observe(make_case("c1"))
    first = made["pipeline"]
    sut.observe(make_case("c2"))
    assert made["pipeline"] is first
    assert first.asked == ["Test sualı nədir?", "Test sualı nədir?"]


def test_istisna_RUNU_OLDURMUR_xeta_kimi_yazilir() -> None:
    """Bir pis case bütün run-ı (və onun xərcini) itirməməlidir."""
    sut, _ = build_sut(raises=RuntimeError("chroma bağlantısı qopdu"))
    obs = sut.observe(make_case("c1"))

    assert obs.error
    assert "RuntimeError" in obs.error
    assert "chroma" in obs.error
    assert obs.reason == "sut_error"


def test_imtina_cavabi_duz_kocurulur() -> None:
    answer = FakeAnswer(
        text="Sənədlərdə bu suala cavab tapılmadı.",
        refused=True,
        reason="low_relevance",
        cited_labels=[],
        top_score=0.31,
    )
    sut, _ = build_sut(answer=answer)
    obs = sut.observe(make_case("c1"))

    assert obs.refused is True
    assert obs.reason == "low_relevance"
    assert obs.cited_labels == ()
