"""Chunking / embedding modeli / sorğu genişləndirmə — ölçmə dövrü.

`tools/retrieval_sweep.py` PARAMETR oxlarını sınadı və hamısını rədd etdi
(`logs/retrieval_sweep.md`). Bu alət növbəti üç DƏYİŞİKLİYİ eyni metrika ilə
ölçür — hər üçü indeksin və ya sorğunun özünü dəyişir, qapı parametrini yox.

İNDEKS TƏCRİDİ — BU FAYLIN ƏN VACİB HİSSƏSİ
--------------------------------------------
Hər eksperiment ÖZ `PERSIST_DIR`-inə yazır. Mövcud `storage/chroma`
2026-08-10 baseline run-larının ölçüldüyü indeksdir; onun üstünə yazmaq
həmin run-ları təkrar istehsal olunmaz edərdi və artefaktlardakı rəqəmlər
artıq heç bir mövcud indeksə aid olmazdı. Ona görə burada `--reset` heç vaxt
paylaşılan qovluğa tətbiq olunmur.

METRİKA — `retrieval_sweep.py` ilə EYNİ
----------------------------------------
örtük  : `gold_sources` sənədlərinin hamısı qəbul edilmiş chunk-lar arasındadırmı
sızma  : korpusda cavabı olmayan suala chunk qəbul edilirmi

Qapı parametrləri baseline-da SABİT saxlanılır (0.42 / 0.10 / 0.35). Səbəb:
burada ölçülən indeksin (və ya sorğunun) keyfiyyətidir. Qapını da eyni anda
tərpətsək, hansı dəyişikliyin nə verdiyi bilinməzdi.

DİQQƏT — YALNIZ DEV.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval.config import Settings  # noqa: E402
from eval.dataset import load_dataset  # noqa: E402

# Qapı — bütün eksperimentlərdə eyni qalır (SUT defaultları).
GATE = {"relevance_threshold": 0.42, "soft_floor_margin": 0.10, "lexical_threshold": 0.35}


@dataclass(frozen=True)
class Experiment:
    name: str
    chunk_size: int = 800
    chunk_overlap: int = 200
    embedding_model: str = ""       # boş = framework defaultu
    expand_queries: bool = False

    @property
    def slug(self) -> str:
        return self.name.replace(" ", "_").replace("/", "-")


def sut_settings_for(settings: Settings, exp: Experiment, persist_dir: Path):
    sys.path.insert(0, str(settings.sut_path))
    from rag.config import Settings as SutSettings

    # `PERSIST_DIR` env vasitəsilə oxunur və `Settings.load()` onu MÜTLƏQ yola
    # çevirir; ona görə override kimi deyil, env kimi verilir.
    os.environ["PERSIST_DIR"] = str(persist_dir)
    return SutSettings.load(
        openai_api_key=settings.require_openai_key(),
        embedding_model=exp.embedding_model or settings.embedding_model,
        chunk_size=exp.chunk_size,
        chunk_overlap=exp.chunk_overlap,
        top_k=4,
        **GATE,
    )


def ensure_index(settings: Settings, exp: Experiment, sut_settings) -> int:
    """İndeksi qurur. Artıq qurulubsa təkrar embedding etmir."""
    from rag.ingest import chunk_documents, load_documents
    from rag.store import VectorStore

    store = VectorStore(sut_settings)
    existing = store.count()
    if existing > 0:
        return existing

    documents = load_documents(settings.sut_path / "data")
    chunks = chunk_documents(documents, sut_settings)
    store.add(chunks)
    return store.count()


def sub_queries(question: str) -> list[str]:
    """Çox-sənədli sual üçün alt-sorğular — DETERMİNİSTİK, model çağırmır.

    Niyə model yox: sual bölgüsünü LLM-ə verdikdə ölçdüyümüz şey retrieval
    deyil, həmin LLM-in o gün necə böldüyü olur; nəticə təkrarlanmır və
    xərc gətirir. Burada sual sadəcə bağlayıcılardan bölünür — «kömək
    edərsə ümumiyyətlə dəyərmi?» sualına cavab vermək üçün bu kifayətdir.
    Kömək edərsə, model əsaslı variant AYRICA dövrdə sınanmalıdır.
    """
    parts = [question]
    for sep in (" və ", " həmçinin ", " eləcə də "):
        expanded = []
        for piece in parts:
            expanded.extend(piece.split(sep))
        parts = expanded
    parts = [p.strip(" ?,.") for p in parts]
    # Çox qısa parça sual deyil, qırıntıdır — retrieval-a vermək nəticəni
    # yaxşılaşdırmır, sadəcə səs-küy gətirir.
    parts = [p for p in parts if len(p.split()) >= 3]
    # Tam sual HƏMİŞƏ saxlanılır: genişləndirmə əlavədir, əvəzləmə deyil.
    # Yalnız hissələri işlətmək bölgü səhv olanda nəticəni gizlicə pisləşdirərdi.
    if question not in parts:
        parts.append(question)
    return parts


def evaluate(pipeline, question: str, gold: set[str], expand: bool) -> dict:
    queries = sub_queries(question) if expand else [question]
    accepted, retrieved_total = [], 0
    seen: set[str] = set()
    for q in queries:
        for chunk in pipeline._retrieve(q, k=pipeline.settings.top_k):
            retrieved_total += 1
            if not pipeline._accepts(chunk)[0]:
                continue
            key = str(getattr(chunk, "chunk_id", id(chunk)))
            if key not in seen:
                seen.add(key)
                accepted.append(chunk)
    got = {Path(str(getattr(c, "source", ""))).name for c in accepted}
    return {
        "queries": len(queries),
        "retrieved": retrieved_total,
        "accepted": len(accepted),
        "covered": not (gold - got),
        "missing": sorted(gold - got),
    }


def run(settings: Settings, exp: Experiment, cases, workdir: Path) -> dict:
    # SUT yolu import-lardan ƏVVƏL əlavə olunmalıdır.
    if str(settings.sut_path) not in sys.path:
        sys.path.insert(0, str(settings.sut_path))
    from rag.pipeline import RagPipeline
    from rag.store import VectorStore

    persist = workdir / exp.slug
    sut_settings = sut_settings_for(settings, exp, persist)
    count = ensure_index(settings, exp, sut_settings)
    pipeline = RagPipeline(sut_settings, store=VectorStore(sut_settings), llm=object())

    per_case, covered, total, leaked, leak_total = {}, 0, 0, 0, 0
    for case in cases:
        gold = {Path(s).name for s in (case.gold_sources or [])}
        row = evaluate(pipeline, case.question, gold, exp.expand_queries)
        per_case[case.id] = row
        if gold:
            total += 1
            covered += bool(row["covered"])
        elif case.category == "out_of_corpus":
            leak_total += 1
            leaked += bool(row["accepted"])

    print(
        f"{exp.name:34s} chunk={count:3d}  örtük: {covered}/{total}   "
        f"sızma: {leaked}/{leak_total}"
    )
    for cid, row in per_case.items():
        if row["missing"]:
            print(f"    ✗ {cid:32s} çatışmır: {', '.join(row['missing'])}")
    return {
        "name": exp.name,
        "chunk_size": exp.chunk_size,
        "chunk_overlap": exp.chunk_overlap,
        "embedding_model": exp.embedding_model or settings.embedding_model,
        "expand_queries": exp.expand_queries,
        "index_chunks": count,
        "covered": covered,
        "total": total,
        "leaked": leaked,
        "leak_total": leak_total,
        "cases": per_case,
    }


EXPERIMENTS = [
    Experiment("baseline (800/200)"),
    Experiment("chunking 500/150", chunk_size=500, chunk_overlap=150),
    Experiment("chunking 400/120", chunk_size=400, chunk_overlap=120),
    Experiment("chunking 300/90", chunk_size=300, chunk_overlap=90),
    Experiment("chunking 250/60", chunk_size=250, chunk_overlap=60),
    Experiment("embedding 3-large", embedding_model="text-embedding-3-large"),
    Experiment(
        "3-large + chunking 300/90",
        chunk_size=300,
        chunk_overlap=90,
        embedding_model="text-embedding-3-large",
    ),
    Experiment("sorğu genişləndirmə", expand_queries=True),
    Experiment(
        "genişləndirmə + 300/90",
        chunk_size=300,
        chunk_overlap=90,
        expand_queries=True,
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--workdir",
        type=Path,
        required=True,
        help="İndekslərin qurulacağı TƏCRİD qovluğu (paylaşılan storage/ OLMAMALIDIR)",
    )
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "logs" / "retrieval_experiments.json")
    args = ap.parse_args()

    settings = Settings.load()
    shared = (settings.sut_path / "storage").resolve()
    if shared == args.workdir.resolve() or shared in args.workdir.resolve().parents:
        print(
            f"İMTİNA: --workdir paylaşılan indeksin içindədir ({shared}).\n"
            "Baseline indeksinin üstünə yazmaq 2026-08-10 run-larını təkrar "
            "istehsal olunmaz edərdi.",
            file=sys.stderr,
        )
        return 2

    dataset = load_dataset(settings.dataset_path)
    cases = [c for c in dataset.cases if c.split == "dev"]
    args.workdir.mkdir(parents=True, exist_ok=True)

    print(f"{len(cases)} dev case × {len(EXPERIMENTS)} eksperiment")
    print(f"qapı sabitdir: {GATE}\n")

    results = [run(settings, exp, cases, args.workdir) for exp in EXPERIMENTS]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"split": "dev", "gate": GATE, "experiments": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nArtefakt: {args.out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
