# Generation Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `tools/generation_probe.py` — saxlanmış dev run-ının kontekstini
təkrar oynadaraq ölçür ki, prompt qaydası SUT-u «sualın strukturuna münasibət
bildirmək» davranışına məcbur edə bilirmi.

**Architecture:** Alət retrieval-i heç işlətmir: chunk-lar
`runs/20260812T094516Z-dev-baseline/observations.jsonl`-dan oxunur, prompt isə
SUT-un öz qurucuları ilə bərpa olunur — yenidən yazılmır. Dörd qol (nəzarət +
üç suffiks) hər case üçün 3 dəfə işlədilir; təsnifat tam determinist, hakim
yoxdur. Nəzarət qolu orijinal davranışı təkrar istehsal etmirsə alət 5 kodu ilə
çıxır və cədvəl etibarsız elan olunur.

**Tech Stack:** Python 3.11, mövcud `eval.*` paketi, SUT `rag.*` (pin edilmiş
submodule, redaktə OLUNMUR), pytest. Şəbəkə yalnız son addımda (paid run).

## Global Constraints

- **SUT kodu redaktə olunmur** — pin `19f14c38d619371dd337acaa0624031f3f855f30`.
- **Testlər açarsız, şəbəkəsiz VƏ langchain/chroma-sız işləyir.** CI yalnız
  `anthropic`, `python-dotenv`, `PyYAML` (+ `requirements-dev.txt`) quraşdırır,
  `rag/store.py:26` isə modul səviyyəsində `langchain_chroma`-nı import edir və
  `rag/pipeline.py:38` `rag.store`-u çəkir. Ona görə testlərdə `rag.*` **birbaşa
  import edilmir** — SUT qurucuları `SutQurucular` vasitəsilə inyeksiya olunur.
  Repo-nun CI şərhi bu sinfin mövcud olmadığını iddia edir; plan onu pozmur.
- **Repo-da Claude atribusiyası yoxdur** — `Co-Authored-By` yazılmır.
- **Artefakt üstünə yazılmır** — `ProbeWriter` müqaviləsi (`status` sahəsi,
  manifest ilk anda).
- **Sənəd dili Azərbaycan dilindədir**, mövcud fayllarla eyni üslubda.
- Artefakt yolu: `logs/probes/<probe_id>/`; sənəd: `logs/generation_cycle.md`.
- İş budağı: `generation-probe` (main-də deyil).
- Python icra edilir: `.venv/bin/python -m pytest ...`

## Fayl quruluşu

| fayl | məsuliyyət |
|---|---|
| `tools/generation_probe.py` (yeni) | qurucu inyeksiyası, qollar, təsnifat, ölçmə dövrü, CLI |
| `tests/test_generation_probe.py` (yeni) | hamısı, `rag.*` import etmədən |
| `logs/generation_cycle.md` (yeni, Task 5) | dövrün mətn qeydi |

`tools/probe_common.py` və `eval/artifacts.py` **dəyişmir** — artefakt
müqaviləsi `203df03`-də qurulub və olduğu kimi işlədilir.

### Niyə inyeksiya

`rag.pipeline` modul səviyyəsində chroma çəkir. Üç seçim var idi: (1) prompt
qurucularını probe-da təkrar yazmaq — o halda ölçdüyümüz prompt SUT-un prompt-u
olmazdı; (2) CI-yə langchain+chroma əlavə etmək — CI-ni ağırlaşdırır və repo
bundan qəsdən qaçınır; (3) **inyeksiya** — testlər saxta qurucu alır, CLI isə
həqiqi SUT-u. Üçüncü seçim repo-nun mövcud üslubudur (`ensure_index(store=…)`,
`RagPipeline(llm=…)`), ona görə o götürülüb.

Sadiqliyi qoruyan şey testlər deyil, **icra vaxtı qapısıdır**: `sut_qurucular()`
SUT-un `RetrievedChunk` dataclass-ının sahə dəstini yoxlayır və uyğunsuzluqda
ölçməni dayandırır. Bu, CI testindən güclüdür, çünki məhz ölçmə anında işləyir.

---

### Task 1: Qurucu inyeksiyası və sadiq prompt bərpası

**Files:**
- Create: `tools/generation_probe.py`
- Test: `tests/test_generation_probe.py`

**Interfaces:**
- Consumes: `eval.observation.ChunkView`, `eval.errors.EvalError`
- Produces:
  - `SUT_CHUNK_SAHELERI: frozenset[str]`
  - `class ProbeError(EvalError)`
  - `SutQurucular` dataclass: `system_instruction: str`,
    `build_user_message: Callable[[str, list, str], str]`, `chunk_cls: type`,
    `is_refusal: Callable[[str], bool]`, `make_nonce: Callable[[], str]`
  - `yoxla_saheler(chunk_cls: type) -> None`
  - `sut_chunk(cv: ChunkView, *, q: SutQurucular) -> Any`
  - `mesajlar(question: str, chunks: Sequence[ChunkView], *, suffix: str, nonce: str, q: SutQurucular) -> list[dict]`

- [ ] **Step 1: Write the failing test**

```python
"""Generation probe — saxlanmış kontekstin təkrar oynadılması.

`rag.*` BURADA İMPORT EDİLMİR: `rag/store.py` modul səviyyəsində
`langchain_chroma`-nı çəkir, CI isə onu quraşdırmır. SUT qurucuları
`SutQurucular` ilə inyeksiya olunur, ona görə bütün dəst açarsız, şəbəkəsiz
və chroma-sız işləyir.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from eval.observation import ChunkView


@dataclass
class _SahteChunk:
    """SUT-un `RetrievedChunk`-ı ilə EYNİ sahə dəsti."""

    label: int
    text: str
    source: str
    page: int
    chunk_index: int
    score: float
    lexical_score: float
    chunk_id: str


def _cv(**over) -> ChunkView:
    base = dict(
        label=1, source="atlas_api_senedi.md", page=0, chunk_index=3,
        score=0.61, lexical_score=0.28, chunk_id="abc123", text="Limit 200-dür.",
    )
    base.update(over)
    return ChunkView(**base)


def _qurucular(**over):
    from tools.generation_probe import SutQurucular

    base = dict(
        system_instruction="SİSTEM QAYDALARI",
        build_user_message=lambda soru, chunks, nonce: (
            f"<<<KONTEKST:{nonce}>>>\n"
            + "\n".join(c.text for c in chunks)
            + f"\n<<<SUAL>>>{soru}"
        ),
        chunk_cls=_SahteChunk,
        is_refusal=lambda t: t.strip().lower().startswith(
            "sənədlərdə bu suala cavab tapılmadı"
        ),
        make_nonce=lambda: "NONCE1",
    )
    base.update(over)
    return SutQurucular(**base)


def test_sahe_desti_uygun_gelmirse_OLCME_DAYANIR() -> None:
    """Sahə uyğunsuzluğu bərpanı SƏSSİZCƏ sadiqsiz edərdi.

    `RetrievedChunk` SUT-a aiddir; pin edilmiş commit dəyişib sahə əlavə
    olunsa, adapter köhnə dəstlə obyekt qurar, prompt fərqlənər və cədvəl
    «eyni kontekstdə ölçüldü» iddiasını yalandan daşıyar. Qapı testdə deyil,
    İCRA VAXTINDA olmalıdır — CI `rag.*`-ı import edə bilmir, ona görə orada
    həqiqi sinif heç vaxt yoxlanmazdı.
    """
    from tools.generation_probe import ProbeError, yoxla_saheler

    yoxla_saheler(_SahteChunk)  # uyğun gəlir → susur

    @dataclass
    class _Deyisib(_SahteChunk):
        yeni_sahe: str = ""

    with pytest.raises(ProbeError, match="sahə dəsti"):
        yoxla_saheler(_Deyisib)


def test_adapter_butun_saheleri_KOCURUR() -> None:
    from tools.generation_probe import sut_chunk

    chunk = sut_chunk(_cv(), q=_qurucular())

    assert chunk.label == 1
    assert chunk.text == "Limit 200-dür."
    assert chunk.source == "atlas_api_senedi.md"
    assert chunk.chunk_id == "abc123"
    assert chunk.score == 0.61
    assert chunk.lexical_score == 0.28


def test_mesajlar_INYEKSIYA_olunmus_qurucunu_cagirir() -> None:
    """Prompt probe-da yenidən yazılmır — qurucu kənardan gəlir."""
    from tools.generation_probe import mesajlar

    msgs = mesajlar("Limit nə qədərdir?", [_cv()], suffix="", nonce="NONCE1", q=_qurucular())

    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "<<<KONTEKST:NONCE1>>>" in msgs[1]["content"]
    assert "Limit nə qədərdir?" in msgs[1]["content"]
    assert "Limit 200-dür." in msgs[1]["content"]


def test_suffiks_system_mesajinin_SONUNA_elave_olunur() -> None:
    """Variant mexanizmi ilə eyni forma: qaydalar əvəzlənmir, əlavə olunur."""
    from tools.generation_probe import mesajlar

    q = _qurucular()
    baseline = mesajlar("S?", [_cv()], suffix="", nonce="N", q=q)[0]["content"]
    genis = mesajlar("S?", [_cv()], suffix="15. Əlavə qayda.", nonce="N", q=q)[0]["content"]

    assert genis.startswith(baseline)
    assert genis.endswith("15. Əlavə qayda.")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_generation_probe.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.generation_probe'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Generation probe — kontekst düzgün olduqda prompt qaydası nə dəyişir.

NİYƏ AYRICA ALƏT
----------------
Sual generasiya qatı haqqındadır, ona görə retrieval SABİT saxlanmalıdır. Tam
run işlətmək onu hər dəfə yenidən icra edər və qapı davranışı nəticəyə
qarışar. Burada chunk-lar SAXLANMIŞ artefaktdan oxunur: retrieval
konstruksiyaya görə sabitdir, embedding çağırılmır, indeks açılmır.

Hakim də çağırılmır — 2026-08-12-də kappa 0.13 ölçüldü, yəni hakimdən keçən
hər rəqəm ehtiyat bayrağı daşıyır. Bu cədvəl tam determinist qalır.

NİYƏ SUT QURUCULARI İNYEKSİYA OLUNUR
------------------------------------
`rag.pipeline` modul səviyyəsində `rag.store`-u, o isə `langchain_chroma`-nı
import edir. CI bu paketləri quraşdırmır (yalnız anthropic/dotenv/PyYAML), ona
görə testlərdə həqiqi import «lokalda keçir, CI-də keçmir» sinfini yaradardı —
repo-nun CI şərhi məhz bu sinfin mövcud olmadığını iddia edir.
Prompt qurucularını burada TƏKRAR YAZMAQ isə daha pis olardı: onda ölçdüyümüz
prompt SUT-un prompt-u olmazdı.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_common import (  # noqa: E402
    PROBES_DIRNAME,
    add_sut_to_path,
    cedvel,
    probe_yarat,
    summary_metni,
)

from eval.config import Settings  # noqa: E402
from eval.errors import EvalError  # noqa: E402
from eval.observation import ChunkView  # noqa: E402


class ProbeError(EvalError):
    """Probe ölçməni davam etdirə bilmir."""


# SUT-un `RetrievedChunk` dataclass-ının GÖZLƏNİLƏN sahə dəsti.
SUT_CHUNK_SAHELERI = frozenset(
    {"label", "text", "source", "page", "chunk_index", "score", "lexical_score", "chunk_id"}
)


@dataclass(frozen=True)
class SutQurucular:
    """SUT-dan götürülən hər şey — bir yerdə, inyeksiya oluna bilən formada."""

    system_instruction: str
    build_user_message: Callable[[str, list, str], str]
    chunk_cls: type
    is_refusal: Callable[[str], bool]
    make_nonce: Callable[[], str]


def yoxla_saheler(chunk_cls: type) -> None:
    """Sahə dəsti gözlənildiyi kimidirmi — ÖLÇMƏ QAPISI.

    Test deyil, icra vaxtı yoxlaması: CI `rag.*`-ı import edə bilmədiyi üçün
    həqiqi sinif orada heç vaxt yoxlanmazdı. Uyğunsuzluqda ölçmə dayanır,
    çünki səssiz davam etmək «eyni kontekstdə ölçüldü» iddiasını yalan edərdi.
    """
    var_olan = {f.name for f in fields(chunk_cls)}
    if var_olan != SUT_CHUNK_SAHELERI:
        artiq = sorted(var_olan - SUT_CHUNK_SAHELERI)
        eskik = sorted(SUT_CHUNK_SAHELERI - var_olan)
        raise ProbeError(
            f"{chunk_cls.__name__} sahə dəsti gözlənildiyi kimi deyil "
            f"(artıq: {artiq or '—'}, əskik: {eskik or '—'}).\n"
            "SUT-un pin edilmiş commit-i dəyişib. Adapter köhnə dəstlə obyekt "
            "qursa, prompt səssizcə fərqlənər və cədvəl «eyni kontekstdə "
            "ölçüldü» iddiasını yalandan daşıyar."
        )


def sut_chunk(cv: ChunkView, *, q: SutQurucular) -> Any:
    """`ChunkView` → SUT-un `RetrievedChunk`-ı."""
    return q.chunk_cls(
        label=cv.label,
        text=cv.text,
        source=cv.source,
        page=cv.page,
        chunk_index=cv.chunk_index,
        score=cv.score,
        lexical_score=cv.lexical_score,
        chunk_id=cv.chunk_id,
    )


def mesajlar(
    question: str,
    chunks: Sequence[ChunkView],
    *,
    suffix: str,
    nonce: str,
    q: SutQurucular,
) -> list[dict]:
    """SUT-un `_answer`-inin qurduğu mesajların EYNİSİ (+ suffiks)."""
    system = q.system_instruction + (f"\n\n{suffix}" if suffix.strip() else "")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": q.build_user_message(
            question, [sut_chunk(c, q=q) for c in chunks], nonce
        )},
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_generation_probe.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add tools/generation_probe.py tests/test_generation_probe.py
git commit -m "Generation probe: SUT qurucularının inyeksiyası və sahə qapısı"
```

---

### Task 2: Determinist təsnifat

**Files:**
- Modify: `tools/generation_probe.py`
- Test: `tests/test_generation_probe.py`

**Interfaces:**
- Consumes: `SutQurucular` (Task 1)
- Produces:
  - `ProbeCase` dataclass: `case_id: str`, `nezaret_davranisi: str`, `izah: str`
  - `PROBE_CASES: tuple[ProbeCase, ...]`
  - `tesnif(case_id: str, text: str, *, q: SutQurucular) -> str` — qaytardığı
    dəyərlər: `"imtina"`, `"tek_oxunus"`, `"oxunuslar_adlandi"`,
    `"muqeddime_qebul"`, `"muqeddime_duzeldildi"`, `"diger"`

- [ ] **Step 1: Write the failing test**

Faktlar korpusdan: `atlas-cache-edge` **yalnız Enterprise planında** ayrılır,
limiti **128 GB** (`atlas_infra_qeydleri.md`); korpusda bir neçə fərqli «limit»
var — sürət, səhifələmə, yaddaş, gündəlik (`atlas_api_senedi.md`).

```python
def test_imtina_INYEKSIYA_olunmus_askarlayici_ile_taninir() -> None:
    """SUT-un öz `_is_refusal`-ı işlədilir — ikinci tərif yaradılmır."""
    from tools.generation_probe import tesnif

    q = _qurucular()
    assert tesnif("dev_ambiguous_limit", "Sənədlərdə bu suala cavab tapılmadı.", q=q) == "imtina"


def test_muqeddime_QEBUL_ve_DUZELDILDI_ayird_olunur() -> None:
    """`dev_false_premise_free_cache`: doğru davranış Enterprise-ı ADLANDIRMAQ.

    Mənfi yoxlama (`not_contains: 128`) bunu ayırd edə bilmir — imtina da onu
    keçir. Probe-un meyarı buna görə daha dardır.
    """
    from tools.generation_probe import tesnif

    q = _qurucular()
    cid = "dev_false_premise_free_cache"
    assert tesnif(cid, "Sənədlərdə bu suala cavab tapılmadı.", q=q) == "imtina"
    assert tesnif(cid, "Free planı üçün yaddaş limiti 128 GB-dır [1].", q=q) == "muqeddime_qebul"
    assert tesnif(
        cid,
        "atlas-cache-edge yalnız Enterprise planında ayrılır, Free planına aid deyil [1].",
        q=q,
    ) == "muqeddime_duzeldildi"


def test_TEK_oxunus_ve_ADLANDIRILMIS_oxunuslar_ayird_olunur() -> None:
    """`dev_ambiguous_limit`: baseline «Limit 200-dür [1]» — tək oxunuş."""
    from tools.generation_probe import tesnif

    q = _qurucular()
    cid = "dev_ambiguous_limit"
    assert tesnif(cid, "Limit 200-dür [1].", q=q) == "tek_oxunus"
    assert tesnif(
        cid, "Korpusda bir neçə limit var: sürət limiti [1] və səhifələmə limiti [2].", q=q
    ) == "oxunuslar_adlandi"
    assert tesnif(cid, "Hansı limiti nəzərdə tutursunuz?", q=q) == "oxunuslar_adlandi"


def test_out_of_corpus_ucun_IMTINA_yaxsi_davranisdir() -> None:
    from tools.generation_probe import tesnif

    assert tesnif(
        "dev_out_of_corpus_graphql", "Sənədlərdə bu suala cavab tapılmadı.", q=_qurucular()
    ) == "imtina"


def test_probe_case_leri_MODELE_CATAN_uclukdur() -> None:
    """`dev_out_of_corpus_ceo` qəsdən yoxdur: qapı onu modelə çatmamış kəsir.

    Saxlanmış artefaktda `reason=low_relevance` və LLM çağırışı SIFIRDIR, yəni
    heç bir prompt dəyişikliyi onu sındıra bilməz. Cədvələ salmaq ölçülməmiş
    şeyi ölçülmüş kimi göstərərdi.
    """
    from tools.generation_probe import PROBE_CASES

    assert [c.case_id for c in PROBE_CASES] == [
        "dev_false_premise_free_cache",
        "dev_ambiguous_limit",
        "dev_out_of_corpus_graphql",
    ]
    assert [c.nezaret_davranisi for c in PROBE_CASES] == ["imtina", "tek_oxunus", "imtina"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_generation_probe.py -q`
Expected: FAIL — `ImportError: cannot import name 'tesnif'`

- [ ] **Step 3: Write minimal implementation**

```python
# --- Təsnifat -------------------------------------------------------------
#
# MEYARLAR QABAQCADAN SABİTDİR. Cədvələ baxdıqdan sonra meyar seçmək nəticəni
# istənilən istiqamətə çevirə bilər, ona görə onlar burada yazılır və
# artefaktda çap olunur.

# Korpus faktı (`atlas_infra_qeydleri.md`): `atlas-cache-edge` YALNIZ
# Enterprise planında ayrılır, limiti 128 GB. Müqəddiməni düzəltmək = doğru
# planı adlandırmaq.
_DUZGUN_PLAN = "enterprise"
_YANLIS_REQEM = "128"

# Korpusdakı fərqli «limit» oxunuşları (`atlas_api_senedi.md` + infra qeydləri).
_LIMIT_NOVLERI = ("sürət", "səhifələmə", "yaddaş", "gündəlik", "eyni vaxtda")
_AYDINLASDIRMA = ("hansı limit", "hansı limiti", "dəqiqləşdir")


def _fold(text: str) -> str:
    """Azərbaycan «I/ı» problemi üçün SUT ilə eyni qaydada kiçildir."""
    return text.replace("I", "ı").replace("İ", "i").casefold()


@dataclass(frozen=True)
class ProbeCase:
    case_id: str
    nezaret_davranisi: str   # qol 0-dan GÖZLƏNİLƏN sinif (etibarlılıq qapısı)
    izah: str


PROBE_CASES = (
    ProbeCase(
        "dev_false_premise_free_cache", "imtina",
        "mənfi yoxlama ilə keçir, amma imtina ilə — düzəliş bacarığı ölçülməyib",
    ),
    ProbeCase(
        "dev_ambiguous_limit", "tek_oxunus",
        "«Limit 200-dür [1]» — səssizcə bir oxunuş seçilib",
    ),
    ProbeCase(
        "dev_out_of_corpus_graphql", "imtina",
        "REQRESSİYA NƏZARƏTİ — qol 2 imtinanı sındırmamalıdır",
    ),
)


def tesnif(case_id: str, text: str, *, q: SutQurucular) -> str:
    """Cavab mətni → davranış sinfi. Tam determinist, hakimsiz."""
    if q.is_refusal(text):
        return "imtina"
    alt = _fold(text)

    if case_id == "dev_false_premise_free_cache":
        if _DUZGUN_PLAN in alt:
            return "muqeddime_duzeldildi"
        if _YANLIS_REQEM in alt:
            return "muqeddime_qebul"
        return "diger"

    if case_id == "dev_ambiguous_limit":
        if any(a in alt for a in _AYDINLASDIRMA):
            return "oxunuslar_adlandi"
        if sum(1 for n in _LIMIT_NOVLERI if n in alt) >= 2:
            return "oxunuslar_adlandi"
        return "tek_oxunus"

    return "diger"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_generation_probe.py -q`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add tools/generation_probe.py tests/test_generation_probe.py
git commit -m "Generation probe: determinist təsnifat, meyarlar qabaqcadan sabit"
```

---

### Task 3: Qollar, ölçmə dövrü və nəzarət qapısı

**Files:**
- Modify: `tools/generation_probe.py`
- Test: `tests/test_generation_probe.py`

**Interfaces:**
- Consumes: `mesajlar` (Task 1), `tesnif`/`PROBE_CASES` (Task 2)
- Produces:
  - `Qol` dataclass: `ad: str`, `system_suffix: str`
  - `QOLLAR: tuple[Qol, ...]`, `NEZARET_QOLU: str`
  - `olc(observations: dict[str, dict], *, llm, q: SutQurucular, tekrar: int = 3) -> list[dict]`
    — sətir açarları: `case_id`, `qol`, `tekrar`, `sinif`, `cavab`, `system_sha256`
  - `nezaret_sadiqdir(setirler: Sequence[dict]) -> tuple[bool, list[str]]`

`llm` inyeksiya olunur və yalnız `invoke(messages) -> obyekt(.content)` tələb
edir — testlər buna görə şəbəkəsizdir.

- [ ] **Step 1: Write the failing test**

```python
class _SahteLlm:
    """`invoke(messages)` — SUT-un `ChatOpenAI`-dən işlətdiyi yeganə metod."""

    def __init__(self, cavablar: dict[str, str]) -> None:
        self.cavablar = cavablar
        self.cagirislar: list[list[dict]] = []

    def invoke(self, messages):
        self.cagirislar.append(messages)
        system = messages[0]["content"]
        for acar, cavab in self.cavablar.items():
            if acar and acar in system:
                return type("R", (), {"content": cavab})()
        return type("R", (), {"content": self.cavablar[""]})()


def _obs(case_id: str, question: str) -> dict:
    return {
        "case_id": case_id, "question": question,
        "chunks": [_cv()], "system_sha256": "",
    }


def test_her_qol_her_case_ucun_UC_defe_isledilir() -> None:
    from tools.generation_probe import QOLLAR, olc

    llm = _SahteLlm({"": "Sənədlərdə bu suala cavab tapılmadı."})
    setirler = olc(
        {"dev_out_of_corpus_graphql": _obs("dev_out_of_corpus_graphql", "GraphQL?")},
        llm=llm, q=_qurucular(), tekrar=3,
    )

    assert len(setirler) == len(QOLLAR) * 3
    assert len(llm.cagirislar) == len(QOLLAR) * 3
    assert sorted({s["tekrar"] for s in setirler}) == [1, 2, 3]
    assert all(s["sinif"] == "imtina" for s in setirler)


def test_qollar_system_mesajini_FERQLENDIRIR() -> None:
    """Dörd qol dörd fərqli system prompt-u deməkdir — yoxsa müqayisə boşdur."""
    from tools.generation_probe import QOLLAR, olc

    llm = _SahteLlm({"": "Sənədlərdə bu suala cavab tapılmadı."})
    setirler = olc(
        {"dev_out_of_corpus_graphql": _obs("dev_out_of_corpus_graphql", "GraphQL?")},
        llm=llm, q=_qurucular(), tekrar=1,
    )

    assert len({s["system_sha256"] for s in setirler}) == len(QOLLAR)


def test_nezaret_qolu_orijinal_davranisi_TEKRAR_ISTEHSAL_etmese_qapi_baglanir() -> None:
    """Sadiq olmayan təkrar oynatmada bütün cədvəl mənasızdır."""
    from tools.generation_probe import NEZARET_QOLU, nezaret_sadiqdir

    yaxsi = [{"case_id": "dev_ambiguous_limit", "qol": NEZARET_QOLU,
              "tekrar": t, "sinif": "tek_oxunus"} for t in (1, 2, 3)]
    assert nezaret_sadiqdir(yaxsi) == (True, [])

    pis = [{**r, "sinif": "oxunuslar_adlandi"} for r in yaxsi]
    ok, sebebler = nezaret_sadiqdir(pis)
    assert ok is False
    assert "dev_ambiguous_limit" in sebebler[0]


def test_nezaret_COXLUQ_uzre_qiymetlendirilir() -> None:
    """Model determinist deyil — 3 təkrarın 2-si uyğun gəlirsə sadiqdir."""
    from tools.generation_probe import NEZARET_QOLU, nezaret_sadiqdir

    setirler = [
        {"case_id": "dev_ambiguous_limit", "qol": NEZARET_QOLU, "tekrar": 1, "sinif": "tek_oxunus"},
        {"case_id": "dev_ambiguous_limit", "qol": NEZARET_QOLU, "tekrar": 2, "sinif": "tek_oxunus"},
        {"case_id": "dev_ambiguous_limit", "qol": NEZARET_QOLU, "tekrar": 3, "sinif": "imtina"},
    ]
    assert nezaret_sadiqdir(setirler)[0] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_generation_probe.py -q`
Expected: FAIL — `ImportError: cannot import name 'QOLLAR'`

- [ ] **Step 3: Write minimal implementation**

`hashlib` faylın yuxarısındakı import blokuna əlavə olunur.

```python
@dataclass(frozen=True)
class Qol:
    ad: str
    system_suffix: str


# QOLLAR. Hər biri SUT-un 5/6-cı qaydalarının yaratdığı boşluğu bir
# istiqamətdən doldurur. Mətnlər variant `system_suffix`-i ilə eyni formadadır,
# ona görə qazanan qol birbaşa `data/variants/`-ə köçürülə bilir.
_MUQEDDIME = """ƏLAVƏ QAYDA — SUALIN MÜQƏDDİMƏSİ:
15. Sualın içindəki fərziyyə kontekstlə ZİDDİYYƏT təşkil edirsə (sual bir
    dəyəri, planı və ya şərti mövcud kimi göstərir, kontekst isə başqasını
    yazır), əvvəlcə düzgün faktı istinadla AÇIQ yaz və fərziyyəni düzəlt.
    Bu halda imtina kifayət deyil: kontekstdə cavab VAR, sadəcə sual onu
    səhv adlandırıb. 1 və 2-ci qaydalar qüvvədə qalır — düzəliş yalnız
    kontekstdə yazılana əsaslana bilər."""

_QEYRI_MUEYYEN = """ƏLAVƏ QAYDA — QEYRİ-MÜƏYYƏN SUAL:
16. Kontekst sualın BİRDƏN ÇOX fərqli oxunuşunu dəstəkləyirsə (eyni söz
    müxtəlif şeylərə aiddir və hər birinin öz dəyəri var), nə səssizcə
    birini seç, nə də imtina et: oxunuşları adlandır və mümkünsə hər biri
    üçün dəyəri istinadla yaz. Hansının nəzərdə tutulduğunu soruşmaq da
    qəbul olunandır."""

QOLLAR = (
    Qol("0-nəzarət", ""),
    Qol("1-müqəddimə", _MUQEDDIME),
    Qol("2-qeyri-müəyyənlik", _QEYRI_MUEYYEN),
    Qol("3-hər ikisi", f"{_MUQEDDIME}\n\n{_QEYRI_MUEYYEN}"),
)

NEZARET_QOLU = QOLLAR[0].ad


def olc(
    observations: dict[str, dict], *, llm, q: SutQurucular, tekrar: int = 3
) -> list[dict]:
    """Hər case × hər qol × `tekrar` — bir sətir hər çağırışa."""
    setirler: list[dict] = []
    for case in PROBE_CASES:
        obs = observations.get(case.case_id)
        if obs is None:
            continue
        for qol in QOLLAR:
            for n in range(1, tekrar + 1):
                msgs = mesajlar(
                    obs["question"], obs["chunks"],
                    suffix=qol.system_suffix, nonce=q.make_nonce(), q=q,
                )
                cavab = str(llm.invoke(msgs).content).strip()
                setirler.append({
                    "case_id": case.case_id,
                    "qol": qol.ad,
                    "tekrar": n,
                    "sinif": tesnif(case.case_id, cavab, q=q),
                    "cavab": cavab,
                    "system_sha256": hashlib.sha256(
                        msgs[0]["content"].encode("utf-8")
                    ).hexdigest()[:16],
                })
    return setirler


def nezaret_sadiqdir(setirler: Sequence[dict]) -> tuple[bool, list[str]]:
    """Qol 0 orijinal davranışı təkrar istehsal edirmi.

    ÇOXLUQ üzrə qiymətləndirilir, hər təkrar üzrə yox: model determinist
    deyil, ona görə tək sapma sadiqsizlik deyil. Amma çoxluq sapırsa, təkrar
    oynatma orijinal sistemi əks etdirmir və CƏDVƏL ETİBARSIZDIR.
    """
    gozlenilen = {c.case_id: c.nezaret_davranisi for c in PROBE_CASES}
    sebebler: list[str] = []
    for case_id, gozlenen in gozlenilen.items():
        siniflər = [
            s["sinif"] for s in setirler
            if s["case_id"] == case_id and s["qol"] == NEZARET_QOLU
        ]
        if not siniflər:
            continue
        uygun = sum(1 for s in siniflər if s == gozlenen)
        if uygun * 2 <= len(siniflər):
            sebebler.append(
                f"{case_id}: nəzarət qolu «{gozlenen}» gözlənilirdi, "
                f"{len(siniflər)} təkrarın yalnız {uygun}-i uyğun gəldi"
            )
    return (not sebebler), sebebler
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_generation_probe.py -q`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add tools/generation_probe.py tests/test_generation_probe.py
git commit -m "Generation probe: dörd qol, ölçmə dövrü və nəzarət qapısı"
```

---

### Task 4: CLI, həqiqi SUT və artefakt

**Files:**
- Modify: `tools/generation_probe.py`
- Test: `tests/test_generation_probe.py`

**Interfaces:**
- Consumes: hamısı yuxarıdan; `probe_yarat`, `cedvel`, `summary_metni`
- Produces:
  - `sut_qurucular(settings: Settings) -> SutQurucular` — **yeganə yer ki,
    `rag.*` import edilir**; `yoxla_saheler`-i çağırır
  - `musahideleri_oxu(run_id: str) -> dict[str, dict]` — açarlar `case_id`,
    `question`, `chunks`, `system_sha256`
  - `system_prompt_deyismeyib(setirler, observations) -> tuple[bool, list[str]]`
  - `main(argv: Sequence[str] | None = None) -> int` — çıxış kodları: `0` uğur,
    `2` run tapılmadı, `5` nəzarət sadiq deyil / system prompt dəyişib

- [ ] **Step 1: Write the failing test**

```python
def test_SYSTEM_PROMPT_run_dan_beri_deyisibse_qapi_baglanir() -> None:
    """Qol 0 SUT-un prompt-unu OLDUĞU KİMİ işlədir — hash uyğun gəlməlidir.

    Uyğun gəlmirsə, SUT-un system prompt-u saxlanmış run-dan bəri dəyişib
    (pin yenilənib və ya submodule sürüşüb). Bu, davranış yoxlamasından
    AYRIDIR və ondan güclüdür: davranış təsadüfən üst-üstə düşə bilər,
    hash düşməz.
    """
    from tools.generation_probe import NEZARET_QOLU, system_prompt_deyismeyib

    setirler = [{"case_id": "c1", "qol": NEZARET_QOLU, "system_sha256": "aaaa"}]
    assert system_prompt_deyismeyib(setirler, {"c1": {"system_sha256": "aaaa"}}) == (True, [])

    ok, sebebler = system_prompt_deyismeyib(setirler, {"c1": {"system_sha256": "bbbb"}})
    assert ok is False
    assert "aaaa" in sebebler[0] and "bbbb" in sebebler[0]


def test_system_prompt_yoxlamasi_yalniz_NEZARET_qoluna_aiddir() -> None:
    """Qol 1-3 prompt-u QƏSDƏN dəyişir — orada uyğunsuzluq gözləniləndir."""
    from tools.generation_probe import system_prompt_deyismeyib

    setirler = [{"case_id": "c1", "qol": "1-müqəddimə", "system_sha256": "ferqli"}]
    assert system_prompt_deyismeyib(setirler, {"c1": {"system_sha256": "aaaa"}}) == (True, [])


def test_nezaret_sadiq_deyilse_alet_BES_kodu_qaytarir(monkeypatch, tmp_path, capsys) -> None:
    """Etibarsız cədvəl SƏSSİZ yazılmır — çıxış kodu onu deyir."""
    import tools.generation_probe as mod

    monkeypatch.setattr(mod, "musahideleri_oxu", lambda run_id: {
        "dev_ambiguous_limit": _obs("dev_ambiguous_limit", "Limit nə qədərdir?"),
    })
    monkeypatch.setattr(mod, "sut_qurucular", lambda settings: _qurucular())
    # Nəzarət qolunda GÖZLƏNİLMƏYƏN davranış: «tek_oxunus» əvəzinə imtina.
    monkeypatch.setattr(mod, "llm_qur", lambda settings: _SahteLlm(
        {"": "Sənədlərdə bu suala cavab tapılmadı."}
    ))

    kod = mod.main(["--run", "r1", "--probes-dir", str(tmp_path)])

    assert kod == 5
    assert "nəzarət" in capsys.readouterr().err.lower()


def test_ugurlu_probe_ARTEFAKT_yazir(monkeypatch, tmp_path) -> None:
    import json

    import tools.generation_probe as mod

    monkeypatch.setattr(mod, "musahideleri_oxu", lambda run_id: {
        "dev_ambiguous_limit": _obs("dev_ambiguous_limit", "Limit nə qədərdir?"),
    })
    monkeypatch.setattr(mod, "sut_qurucular", lambda settings: _qurucular())
    monkeypatch.setattr(mod, "llm_qur", lambda settings: _SahteLlm({"": "Limit 200-dür [1]."}))

    assert mod.main(["--run", "r1", "--probes-dir", str(tmp_path)]) == 0

    qovluq = next(p for p in tmp_path.iterdir() if p.is_dir())
    manifest = json.loads((qovluq / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "tamam"
    assert manifest["nezaret_sadiqdir"] is True
    assert (qovluq / "summary.md").exists()
    assert (qovluq / "rows.jsonl").read_text(encoding="utf-8").count("\n") == 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_generation_probe.py -q`
Expected: FAIL — `ImportError: cannot import name 'system_prompt_deyismeyib'`

- [ ] **Step 3: Write minimal implementation**

`argparse` və `from eval.artifacts import RunPaths, load_run` yuxarıdakı import
blokuna əlavə olunur.

```python
def sut_qurucular(settings: Settings) -> SutQurucular:
    """HƏQİQİ SUT — `rag.*` import edilən YEGANƏ yer.

    Bura yalnız CLI yolundan çağırılır, ona görə testlər chroma tələb etmir.
    """
    add_sut_to_path(settings)
    from rag.pipeline import (  # noqa: PLC0415
        SYSTEM_INSTRUCTION,
        _is_refusal,
        build_user_message,
        make_nonce,
    )
    from rag.store import RetrievedChunk  # noqa: PLC0415

    yoxla_saheler(RetrievedChunk)
    return SutQurucular(
        system_instruction=SYSTEM_INSTRUCTION,
        build_user_message=build_user_message,
        chunk_cls=RetrievedChunk,
        # `has_citation=False`: probe sitat yoxlamasını təkrar etmir, yalnız
        # imtina cümləsini tanıyır. SUT-un öz funksiyası olduğu üçün tərif
        # bir dənədir.
        is_refusal=lambda text: _is_refusal(text, has_citation=False),
        make_nonce=make_nonce,
    )


def llm_qur(settings: Settings):
    """SUT-un öz LLM qurucusu — probe başqa model işlətməməlidir."""
    add_sut_to_path(settings)
    from rag.config import Settings as SutSettings  # noqa: PLC0415
    from rag.config import build_llm  # noqa: PLC0415

    return build_llm(SutSettings.load(openai_api_key=settings.require_openai_key()))


def musahideleri_oxu(run_id: str) -> dict[str, dict]:
    """Saxlanmış run-dan hər case üçün BİRİNCİ təkrarın kontekstini götürür.

    Birinci təkrar seçilir, çünki ölçülən şey qollar arasındakı fərqdir —
    başlanğıc kontekst hamısı üçün eyni olmalıdır.
    """
    settings = Settings.load()
    paths = RunPaths.for_run(settings.runs_dir, run_id)
    if not paths.exists():
        raise FileNotFoundError(paths.root)
    run = load_run(paths)
    out: dict[str, dict] = {}
    for obs in run.observations:
        if obs.repeat != 1 or obs.case_id in out:
            continue
        # Sintez çağırışının system hash-i: qol 0 onunla tutuşdurulur. Çağırış
        # olmayan case (qapı kəsib) bura düşsə, dəyər boş qalır və yoxlama onu
        # atlayır — uydurulmuş hash yazmaqdansa boşluğu etiraf etmək doğrudur.
        sintez = [c for c in obs.llm_calls if c.role == "synthesis"]
        out[obs.case_id] = {
            "case_id": obs.case_id,
            "question": obs.question,
            "chunks": list(obs.chunks),
            "system_sha256": sintez[0].system_sha256 if sintez else "",
        }
    return out


def system_prompt_deyismeyib(
    setirler: Sequence[dict], observations: dict[str, dict]
) -> tuple[bool, list[str]]:
    """Qol 0-ın system prompt-u saxlanmış run-dakı ilə eynidirmi.

    YALNIZ NƏZARƏT QOLU: qol 1-3 prompt-u qəsdən dəyişir.
    """
    sebebler: list[str] = []
    for setir in setirler:
        if setir["qol"] != NEZARET_QOLU:
            continue
        gozlenilen = (observations.get(setir["case_id"]) or {}).get("system_sha256", "")
        if not gozlenilen:
            continue
        if setir["system_sha256"] != gozlenilen:
            sebeb = (
                f"{setir['case_id']}: system prompt dəyişib — "
                f"run-da `{gozlenilen}`, indi `{setir['system_sha256']}`"
            )
            if sebeb not in sebebler:
                sebebler.append(sebeb)
    return (not sebebler), sebebler


def _summary_govdesi(
    setirler: Sequence[dict], sadiq: bool, sebebler: Sequence[str]
) -> str:
    """Sənədə köçürüləcək cədvəl — hər case × qol üçün üstün sinif."""
    qol_adlari = [q.ad for q in QOLLAR]
    matris: dict[tuple[str, str], list[str]] = {}
    for s in setirler:
        matris.setdefault((s["case_id"], s["qol"]), []).append(s["sinif"])

    cedvel_setirleri = []
    for case in PROBE_CASES:
        setir = [case.case_id]
        for qol in qol_adlari:
            siniflər = matris.get((case.case_id, qol), [])
            if not siniflər:
                setir.append("—")
                continue
            usta = max(sorted(set(siniflər)), key=siniflər.count)
            setir.append(f"{usta} ({siniflər.count(usta)}/{len(siniflər)})")
        cedvel_setirleri.append(setir)

    out = cedvel(["case", *qol_adlari], cedvel_setirleri)
    if not sadiq:
        out += "\n\n> ⚠️ **NƏZARƏT QOLU SADİQ DEYİL** — " + "; ".join(sebebler) + (
            ". Yuxarıdakı cədvəl orijinal sistemi əks etdirmir və nəticə "
            "çıxarmaq üçün işlədilə bilməz."
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="kontekstin götürüləcəyi dev run_id")
    ap.add_argument("--probes-dir", type=Path, default=None)
    ap.add_argument("--tekrar", type=int, default=3)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)

    settings = Settings.load()
    try:
        observations = musahideleri_oxu(args.run)
    except FileNotFoundError as exc:
        print(f"Run tapılmadı: {exc}", file=sys.stderr)
        return 2

    from eval.dataset import load_dataset  # noqa: PLC0415

    dataset = load_dataset(settings.dataset_path)
    probes_dir = args.probes_dir or (settings.logs_dir / PROBES_DIRNAME)
    writer, kimlik = probe_yarat(
        alet="generasiya",
        oxlar=[qol.ad for qol in QOLLAR[1:]],
        argv=["python", "tools/generation_probe.py", *argv],
        settings=settings,
        dataset=dataset,
        probes_dir=probes_dir,
    )

    try:
        setirler = olc(
            observations, llm=llm_qur(settings), q=sut_qurucular(settings),
            tekrar=args.tekrar,
        )
        for setir in setirler:
            writer.append_row(setir)

        davranis_ok, davranis_sebebleri = nezaret_sadiqdir(setirler)
        hash_ok, hash_sebebleri = system_prompt_deyismeyib(setirler, observations)
        sadiq = davranis_ok and hash_ok
        sebebler = [*hash_sebebleri, *davranis_sebebleri]

        writer.write_manifest({
            **kimlik, "nezaret_sadiqdir": sadiq,
            "nezaret_sebebleri": sebebler, "tekrar": args.tekrar,
        })
        writer.write_summary(summary_metni(
            basliq="Generasiya qolları (dev, saxlanmış kontekst)",
            probe_id_=writer.paths.probe_id, argv=kimlik["argv"],
            govde=_summary_govdesi(setirler, sadiq, sebebler),
        ))
    except BaseException as exc:  # noqa: BLE001 — KeyboardInterrupt da daxil
        writer.mark_failed(type(exc).__name__)
        raise

    print(f"\nArtefakt: {writer.paths.root.relative_to(PROJECT_ROOT)}")
    if not sadiq:
        print("\n⚠️ NƏZARƏT QOLU SADİQ DEYİL — cədvəl etibarsızdır:", file=sys.stderr)
        for sebeb in sebebler:
            print(f"   {sebeb}", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_generation_probe.py -q`
Expected: 17 passed

Sonra tam dəst və CI-nin əlavə addımı:

Run: `.venv/bin/python -m pytest -q && .venv/bin/python tools/generation_probe.py --help`
Expected: hamısı yaşıl; `--help` çap olunur

**CI müqaviləsinin yoxlanması** — testlərin chroma-sız keçdiyi TƏSDİQLƏNİR:

Run:
```bash
.venv/bin/python - <<'PY'
import builtins, subprocess, sys
# `langchain_chroma` import olunmağa cəhd edilsə test dəsti qırılmalı DEYİL.
kod = """
import builtins
_real = builtins.__import__
def _blok(ad, *a, **k):
    if ad.startswith(('langchain', 'chromadb')):
        raise ModuleNotFoundError(ad)
    return _real(ad, *a, **k)
builtins.__import__ = _blok
import pytest, sys
sys.exit(pytest.main(['-q', 'tests/test_generation_probe.py']))
"""
sys.exit(subprocess.run([sys.executable, '-c', kod]).returncode)
PY
```
Expected: 17 passed — langchain/chroma bloklandığı halda belə.

- [ ] **Step 5: Commit**

```bash
git add tools/generation_probe.py tests/test_generation_probe.py
git commit -m "Generation probe: CLI, həqiqi SUT qurucuları və artefakt"
```

---

### Task 5: Ölçməni apar və yaz

Yeganə **pullu** addım (~$0.015) və Heyderin açıq təsdiqini tələb edir.

**Files:**
- Create: `logs/generation_cycle.md`
- Artefakt: `logs/probes/<probe_id>/` (alət yazır, əl ilə yazılmır)

- [ ] **Step 1: Heyderdən təsdiq al**

Ölçmə şəbəkəyə çıxır və pul xərcləyir. Təsdiq alınmadan işlədilmir.

- [ ] **Step 2: Probe-u işlət**

```bash
.venv/bin/python tools/generation_probe.py --run 20260812T094516Z-dev-baseline
```

Expected: çıxış kodu `0` və artefakt yolu çap olunur.
Kod `5` olarsa: nəzarət qolu sadiq deyil — **nəticə çıxarılmır**, səbəb
`logs/generation_cycle.md`-ə yazılır və dövr orada dayanır.

- [ ] **Step 3: `logs/generation_cycle.md`-i yaz**

Skelet (başlıqlar sabitdir, məzmun ölçmədən gəlir):

```markdown
# Generasiya qatı dövrü (2026-08-…)

## Niyə bu dövr
[holdout-da qalan iki uğursuzluq + hər ikisinin kontekstində düzgün faktın
olduğu — artefaktdan chunk nömrələri və balları ilə]

## Hipotez: iki hərəkət, çatışmayan üçüncü
[cavab ver / imtina et; struktura münasibət yoxdur; 5-6-cı qaydalar]

## Ölçmədən əvvəl aşkarlanan test qüsuru
[`dev_false_premise_free_cache` imtina ilə keçir, çünki yoxlaması mənfidir;
bu, karkas haqqında tapıntıdır və case redaktə OLUNMUR]

## Probe metodu
[saxlanmış kontekst; retrieval işlədilmir; hakim yoxdur; nəzarət qolu qapısı;
`dev_out_of_corpus_ceo`-nun niyə kənarda olduğu]

## Nəticə

<!-- artefakt: <probe_id> -->
[cədvəl summary.md-dən OLDUĞU KİMİ köçürülür — əl ilə yazılmır]
<!-- /artefakt -->

## Oxunuş və növbəti addımın qapısı
[hansı qol nəyi dəyişdi; reqressiya varmı; dövr davam edir, yoxsa bağlanır]
```

- [ ] **Step 4: Sənəd iddialarını yoxla**

Run: `.venv/bin/python -m pytest tests/test_logs_iddialari.py -q`
Expected: PASS — sənəddəki hər sətir `summary.md`-də hərfi-hərfinə mövcuddur.

- [ ] **Step 5: Commit**

Commit mesajının başlığı **nəticəni deyir**, ölçmənin aparıldığını yox — repo-nun
mövcud üslubu budur (`Retrieval dövrü: chunking 500/150 ölçüldü və holdout-da
təsdiqləndi`). Nəticədən asılı olaraq:

```bash
git add logs/generation_cycle.md logs/probes
# müsbətdirsə:
git commit -m "Generasiya qolları ölçüldü: müqəddimə qaydası imtinanı düzəlişə çevirir"
# mənfidirsə:
git commit -m "Generasiya qolları ölçüldü: prompt qaydası davranışı dəyişmir, dövr bağlanır"
```

---

## Nəticənin qapısı

| probe nəticəsi | növbəti addım |
|---|---|
| nəzarət sadiq deyil (kod 5) | dövr dayanır; təkrar oynatmanın niyə sadiq olmadığı araşdırılır |
| heç bir qol davranışı dəyişmir | **dövr bağlanır və yazılır** — möhür toxunulmur, holdout toxunulmur |
| qol 1 və/və ya 2 işləyir, `graphql` sınmır | spesifikasiyanın addım 2-sinə keçilir (yeni dev case-ləri + `seal-split --force`) |
| qol işləyir, amma `graphql` sınır | qazanc reqressiya ilə gəlir; variant mətni daraldılır və probe təkrarlanır (~$0.015) |
