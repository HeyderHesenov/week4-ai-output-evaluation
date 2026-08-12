"""Retrieval parametr sweep-i — GENERASİYA VƏ HAKİM YOXDUR.

NİYƏ AYRICA ALƏT
----------------
Tam run (dev) $0.042-dir və cavabın keyfiyyətini ölçür. Amma retrieval
parametrini seçmək üçün lazım olan sual daha dardır: «bu parametrlərlə
lazımi sənəd ümumiyyətlə kontekstə düşürmü?» Buna cavab vermək üçün nə
sintez, nə korreksiya, nə hakim lazımdır — yalnız sorğu embedding-i.
Ona görə bu alət ~$0.0002-yə namizədləri süzür və pullu run yalnız
sağ qalan 1-2 namizədə xərclənir.

NİYƏ SUT-UN ÖZ KODU ÇAĞIRILIR
------------------------------
Qəbul qapısı (`RagPipeline._accepts`) astana, yumşaq hədd və leksik
dəstəyi birlikdə tətbiq edir. Onu burada yenidən yazsaydıq, sweep
istehsalatdan FƏRQLİ bir şeyi ölçərdi və seçilən parametr tam run-da
gözlənilməz nəticə verərdi. Ona görə həm `_retrieve`, həm `_accepts`
SUT-un özündən çağırılır. LLM qurulmur (`llm=object()`): qapı qatı
model çağırmır.

ÖLÇÜLƏN METRİK: gold-source recall
-----------------------------------
Datasetdəki `gold_sources` cavab üçün LAZIM olan sənədləri sadalayır.
Sweep soruşur: qəbul edilən chunk-lar bu sənədlərin hamısını əhatə
edirmi? Bu, hakimdən asılı olmayan, tamamilə offline ölçüdür.

DİQQƏT — YALNIZ DEV. Holdout case-ləri bu alətə verilmir: parametr
seçimi holdout sübutuna baxaraq aparılsa, holdout dev-ə çevrilir.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval.config import Settings  # noqa: E402
from eval.dataset import load_dataset  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    top_k: int
    threshold: float
    soft_floor_margin: float
    lexical_threshold: float

    @property
    def label(self) -> str:
        return (
            f"k={self.top_k} astana={self.threshold:.2f} "
            f"marja={self.soft_floor_margin:.2f} leks={self.lexical_threshold:.2f}"
        )


def build_pipeline(settings: Settings, cand: Candidate):
    sys.path.insert(0, str(settings.sut_path))
    from rag.config import Settings as SutSettings
    from rag.pipeline import RagPipeline
    from rag.store import VectorStore

    sut_settings = SutSettings.load(
        openai_api_key=settings.require_openai_key(),
        embedding_model=settings.embedding_model,
        top_k=cand.top_k,
        relevance_threshold=cand.threshold,
        soft_floor_margin=cand.soft_floor_margin,
        lexical_threshold=cand.lexical_threshold,
    )
    # `llm=object()`: qapı qatı modelə toxunmur, ona görə saxta obyekt
    # kifayətdir. Əsl LLM qurmaq boş yerə açar tələb edərdi.
    return RagPipeline(sut_settings, store=VectorStore(sut_settings), llm=object())


def evaluate(pipeline, question: str, gold_sources: set[str], top_k: int) -> dict:
    retrieved = pipeline._retrieve(question, k=top_k)
    accepted = [c for c in retrieved if pipeline._accepts(c)[0]]
    got = {Path(str(getattr(c, "source", ""))).name for c in accepted}
    missing = gold_sources - got
    return {
        "retrieved": len(retrieved),
        "accepted": len(accepted),
        "scores": [round(float(c.score), 3) for c in retrieved],
        "covered": not missing,
        "missing": sorted(missing),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top-k", type=int, nargs="+", default=[4])
    ap.add_argument("--threshold", type=float, nargs="+", default=[0.42])
    ap.add_argument("--margin", type=float, nargs="+", default=[0.10])
    ap.add_argument("--lexical", type=float, nargs="+", default=[0.35])
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "logs" / "retrieval_sweep.json")
    args = ap.parse_args()

    settings = Settings.load()
    dataset = load_dataset(settings.dataset_path)
    cases = [c for c in dataset.cases if c.split == "dev"]
    if not cases:
        print("dev case tapılmadı", file=sys.stderr)
        return 1

    grid = [
        Candidate(k, t, m, lex)
        for k, t, m, lex in itertools.product(
            args.top_k, args.threshold, args.margin, args.lexical
        )
    ]
    print(f"{len(cases)} dev case × {len(grid)} namizəd\n")

    results = []
    for cand in grid:
        pipeline = build_pipeline(settings, cand)
        per_case, covered, total = {}, 0, 0
        # QAPININ O BİRİ TƏRƏFİ. Yalnız örtüyü ölçmək astananı sıfıra
        # endirməyi «mükəmməl» göstərərdi: hər sənəd hər suala düşər, örtük
        # 8/8 olar və sistem hallüsinasiya maşınına çevrilər. Korpusda cavabı
        # OLMAYAN case-lər üçün doğru davranış heç nə qəbul etməməkdir, ona
        # görə onlar da eyni cədvəldə sayılır.
        leaked, leak_total = 0, 0
        for case in cases:
            gold = {Path(s).name for s in (case.gold_sources or [])}
            row = evaluate(pipeline, case.question, gold, cand.top_k)
            per_case[case.id] = row
            if gold:
                total += 1
                covered += bool(row["covered"])
            elif case.category == "out_of_corpus":
                leak_total += 1
                leaked += bool(row["accepted"])
        print(
            f"{cand.label:38s} örtük: {covered}/{total}   "
            f"korpusdan kənara sızma: {leaked}/{leak_total}"
        )
        for cid, row in per_case.items():
            if row["missing"]:
                print(f"    ✗ {cid:32s} çatışmır: {', '.join(row['missing'])}")
        results.append(
            {
                "top_k": cand.top_k,
                "threshold": cand.threshold,
                "soft_floor_margin": cand.soft_floor_margin,
                "lexical_threshold": cand.lexical_threshold,
                "covered": covered,
                "total": total,
                "out_of_corpus_leaked": leaked,
                "out_of_corpus_total": leak_total,
                "cases": per_case,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"split": "dev", "candidates": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nArtefakt: {args.out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
