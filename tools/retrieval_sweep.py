"""Retrieval parametr sweep-i — GENERASİYA VƏ HAKİM YOXDUR.

NİYƏ AYRICA ALƏT
----------------
Tam run (dev) $0.042-dir və cavabın keyfiyyətini ölçür. Amma retrieval
parametrini seçmək üçün lazım olan sual daha dardır: «bu parametrlərlə
lazımi sənəd ümumiyyətlə kontekstə düşürmü?» Buna cavab vermək üçün nə
sintez, nə korreksiya, nə hakim lazımdır — yalnız sorğu embedding-i.
Ona görə bu alət ~$0.001-ə namizədləri süzür və pullu run yalnız sağ
qalan namizədə xərclənir.

NİYƏ SUT-UN ÖZ KODU ÇAĞIRILIR
------------------------------
Qəbul qapısı (`RagPipeline._accepts`) astana, yumşaq hədd və leksik
dəstəyi birlikdə tətbiq edir. Onu burada yenidən yazsaydıq, sweep
istehsalatdan FƏRQLİ bir şeyi ölçərdi və seçilən parametr tam run-da
gözlənilməz nəticə verərdi. Ona görə həm `_retrieve`, həm `_accepts`
SUT-un özündən çağırılır. LLM qurulmur (`llm=object()`): qapı qatı model
çağırmır.

ÖLÇÜLƏN İKİ RƏQƏM
------------------
örtük : `gold_sources` sənədlərinin hamısı qəbul edilmiş chunk-lar
        arasındadırmı (hakimdən asılı deyil, tamamilə offline)
sızma : korpusda cavabı OLMAYAN suala chunk qəbul edilirmi

İkincisi qəsdən var: təkcə örtüyə baxsaq, astananı sıfıra endirmək
«mükəmməl» görünərdi və sistem hallüsinasiya maşınına çevrilərdi.

ARTEFAKT MÜQAVİLƏSİ (2026-08-12 düzəlişi)
------------------------------------------
Hər icra `logs/probes/<probe_id>/` qovluğuna yazır və MÖVCUD qovluğun
üstünə yazmır. Əvvəlki versiya sabit `--out` faylına yazırdı; üç ardıcıl
sweep bir-birini əvəzlədi və sənəd dörd ox üzrə nəticə iddia etdiyi halda
diskdə yalnız sonuncu qaldı. Cədvəl də artıq `summary.md`-də generasiya
olunur, sənədə əl ilə köçürülmür.

DİQQƏT — YALNIZ DEV. Holdout case-ləri bu alətə verilmir: parametr seçimi
holdout sübutuna baxaraq aparılsa, holdout dev-ə çevrilir.

ÇIXIŞ KODLARI
  0  uğurlu
  1  konfiqurasiya xətası (arqumentlər SUT-a çatmadan yoxlanır)
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# `python tools/retrieval_sweep.py` çağırışında Python `tools/`-u sys.path-a
# özü qoyur, `import tools.retrieval_sweep` (testlər, CI) isə qoymur. Açıq
# yazılır ki, alət hər iki yolla eyni işləsin.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_common import (  # noqa: E402
    PROBES_DIRNAME,
    PROJECT_ROOT,
    add_sut_to_path,
    cedvel,
    probe_yarat,
    summary_metni,
)

from eval.config import Settings, yoxla_retrieval  # noqa: E402
from eval.dataset import load_dataset  # noqa: E402
from eval.errors import ConfigError  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    top_k: int
    threshold: float
    soft_floor_margin: float
    lexical_threshold: float

    @property
    def soft_floor(self) -> float:
        return max(0.0, self.threshold - self.soft_floor_margin)

    @property
    def qapi_sonur(self) -> bool:
        """Yumşaq hədd 0-a düşübsə, dense qapı praktiki olaraq sönür."""
        return self.soft_floor <= 0.0

    @property
    def label(self) -> str:
        etiket = (
            f"k={self.top_k} astana={self.threshold:.2f} "
            f"marja={self.soft_floor_margin:.2f} leks={self.lexical_threshold:.2f}"
        )
        if self.qapi_sonur:
            etiket += " (yumşaq hədd 0.00 — dense qapı sönür)"
        return etiket


def build_pipeline(settings: Settings, cand: Candidate):
    add_sut_to_path(settings)
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


def swept_axes(args: argparse.Namespace) -> list[str]:
    """Birdən çox dəyəri olan oxlar — `probe_id`-yə düşür."""
    pairs = (
        ("top_k", args.top_k),
        ("threshold", args.threshold),
        ("soft_floor_margin", args.margin),
        ("lexical_threshold", args.lexical),
    )
    return [ad for ad, deyerler in pairs if len(deyerler) > 1]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top-k", type=int, nargs="+", default=[4])
    ap.add_argument("--threshold", type=float, nargs="+", default=[0.42])
    ap.add_argument("--margin", type=float, nargs="+", default=[0.10])
    ap.add_argument("--lexical", type=float, nargs="+", default=[0.35])
    ap.add_argument(
        "--probes-dir",
        type=Path,
        default=None,
        help="default: <LOGS_DIR>/probes. Hər icra öz alt-qovluğunu alır.",
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)

    # ARQUMENTLƏR SUT-A ÇATMADAN YOXLANIR. Əks halda `--threshold 1.8`
    # işləyib 0/8 cədvəli çap edərdi — və o cədvəl tapıntı kimi oxunardı.
    try:
        for k, t, m, lex in itertools.product(
            args.top_k, args.threshold, args.margin, args.lexical
        ):
            yoxla_retrieval(
                top_k=k, threshold=t, soft_floor_margin=m, lexical_threshold=lex
            )
    except ConfigError as exc:
        print(f"Konfiqurasiya xətası: {exc}", file=sys.stderr)
        return 1

    settings = Settings.load()
    dataset = load_dataset(settings.dataset_path)
    cases = [c for c in dataset.cases if c.split == "dev"]
    if not cases:
        print("dev case tapılmadı", file=sys.stderr)
        return 1

    probes_dir = args.probes_dir or (settings.logs_dir / PROBES_DIRNAME)
    writer, kimlik = probe_yarat(
        alet="sweep",
        oxlar=swept_axes(args),
        argv=["python", "tools/retrieval_sweep.py", *argv],
        settings=settings,
        dataset=dataset,
        probes_dir=probes_dir,
    )

    # Eksperiment aləti ilə EYNİ müqavilə: qovluq açılıb, `status: yarımçıq`
    # manifesti diskdədir; hansı yolla çıxsaq da vəziyyət manifestdə deyilir.
    try:
        return _sweep_et(writer, kimlik, settings, cases, args)
    except BaseException as exc:  # noqa: BLE001 — KeyboardInterrupt da daxil
        writer.mark_failed(type(exc).__name__)
        raise


def _sweep_et(writer, kimlik: dict, settings: Settings, cases, args) -> int:
    grid = [
        Candidate(k, t, m, lex)
        for k, t, m, lex in itertools.product(
            args.top_k, args.threshold, args.margin, args.lexical
        )
    ]
    print(f"{len(cases)} dev case × {len(grid)} namizəd → {writer.paths.root}\n")

    setirler = []
    for cand in grid:
        pipeline = build_pipeline(settings, cand)
        per_case, covered, total, leaked, leak_total = {}, 0, 0, 0, 0
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

        print(f"{cand.label:60s} örtük: {covered}/{total}   sızma: {leaked}/{leak_total}")
        for cid, row in per_case.items():
            if row["missing"]:
                print(f"    ✗ {cid:32s} çatışmır: {', '.join(row['missing'])}")

        writer.append_row(
            {
                "top_k": cand.top_k,
                "threshold": cand.threshold,
                "soft_floor_margin": cand.soft_floor_margin,
                "lexical_threshold": cand.lexical_threshold,
                "soft_floor": round(cand.soft_floor, 4),
                "covered": covered,
                "total": total,
                "out_of_corpus_leaked": leaked,
                "out_of_corpus_total": leak_total,
                "cases": per_case,
            }
        )
        setirler.append(
            [
                cand.top_k,
                f"{cand.threshold:.2f}",
                f"{cand.soft_floor_margin:.2f}",
                f"{cand.lexical_threshold:.2f}",
                f"{covered}/{total}",
                f"{leaked}/{leak_total}",
            ]
        )

    writer.write_manifest({**kimlik, "namizəd_sayı": len(grid), "case_sayı": len(cases)})
    writer.write_summary(
        summary_metni(
            basliq="Retrieval parametr sweep-i (dev)",
            probe_id_=writer.paths.probe_id,
            argv=kimlik["argv"],
            govde=cedvel(
                ["TOP_K", "astana", "marja", "leksik", "örtük", "sızma"], setirler
            ),
        )
    )
    print(f"\nArtefakt: {writer.paths.root.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
