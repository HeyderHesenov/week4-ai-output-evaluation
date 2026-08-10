"""Komanda sətri interfeysi.

EXIT KODLARI QƏSDƏN AYRILIB
---------------------------
    0 — uğur
    1 — istifadəçi/konfiqurasiya xətası (`EvalError`)
    2 — dev/holdout intizamının pozulması (`ContaminationError`)

2-nin ayrıca kod olması CI üçündür: çirklənmiş qiymətləndirmə adi xətadan
FƏRQLİ hadisədir. Adi xəta run-ı dayandırır, çirklənmə isə artıq yazılmış
nəticələrin hamısını şübhə altına alır — və pipeline buna fərqli reaksiya
verməlidir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .artifacts import RunPaths, RunWriter, list_runs, load_run
from .config import PROJECT_ROOT, Settings, load_prices
from .dataset import load_dataset
from .errors import ContaminationError, EvalError
from .judge import AnthropicJudge
from .metrics import LabelSource, judge_bias
from .report import build_report
from .runner import HOLDOUT_LEDGER, Runner, read_harness_commit, reclassify, utc_now
from .split import (
    ContaminationGuard,
    load_split_manifest,
    read_holdout_ledger,
    seal_split,
    write_split_manifest,
)
from .sut import RagSut
from .variants import load_variants


LABEL_SCALE = (0, 3)


def load_human_labels(path: Path) -> dict[str, int]:
    """`case_id: bal` xəritəsi. Fayl yoxdursa boş — kappa hesablanmır.

    Bal aralıqdan kənardırsa xəta atılır: hakim 0-3 şkalasındadır və başqa
    şkalada yazılmış insan etiketi ilə kappa mənasız rəqəm verər (səssizcə).
    """
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    labels = raw.get("labels") or {}
    if not isinstance(labels, dict):
        raise EvalError(f"{path}: 'labels' sözlük olmalıdır.")
    low, high = LABEL_SCALE
    out: dict[str, int] = {}
    for key, value in labels.items():
        try:
            score = int(value)
        except (TypeError, ValueError):
            raise EvalError(f"{path}: '{key}' üçün bal tam ədəd olmalıdır: {value!r}")
        if not low <= score <= high:
            raise EvalError(
                f"{path}: '{key}' üçün bal {low}-{high} aralığında olmalıdır, "
                f"verilən: {score}. Hakim eyni şkalanı işlədir; fərqli şkala "
                f"kappanı mənasız edər."
            )
        out[str(key)] = score
    return out


def load_label_sources(path: Path) -> dict[str, LabelSource]:
    """`case_id -> (run_id, repeat)`: hər etiket MƏHZ hansı cavaba aiddir.

    Bax `metrics.judge_bias` — bu ünvan olmasa etiket ya təkrarlara görə
    çoxaldılar, ya insanın görmədiyi cavabla müqayisə olunar.
    """
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sources = raw.get("sources") or {}
    if not isinstance(sources, dict):
        raise EvalError(f"{path}: 'sources' sözlük olmalıdır.")
    out: dict[str, LabelSource] = {}
    for key, value in sources.items():
        if not isinstance(value, dict) or "run_id" not in value:
            raise EvalError(
                f"{path}: 'sources.{key}' `{{run_id: ..., repeat: N}}` formatında olmalıdır."
            )
        out[str(key)] = LabelSource(str(value["run_id"]), int(value.get("repeat", 1)))
    return out


# ---------------------------------------------------------------------------
# Əmrlər
# ---------------------------------------------------------------------------


def cmd_seal_split(args: argparse.Namespace) -> int:
    settings = Settings.load()
    dataset = load_dataset(settings.dataset_path)
    path = settings.split_manifest_path

    if path.exists() and not args.force:
        existing = load_split_manifest(path)
        if existing.dataset_sha256 == dataset.sha256:
            print(f"Bölgü artıq möhürlənib və test dəsti dəyişməyib: {path}")
            return 0
        raise ContaminationError(
            "Bölgü möhürlənib, lakin test dəsti O VAXTDAN dəyişib.\n"
            f"  möhürlənmiş: {existing.dataset_sha256[:16]}\n"
            f"  cari:        {dataset.sha256[:16]}\n"
            "Dəyişiklik qəsdəndirsə: --force ilə yenidən möhürləyin və bunu "
            "hesabatda AÇIQ qeyd edin — bütün əvvəlki run-lar etibarsız olur."
        )

    manifest = seal_split(
        dataset,
        sealed_at=utc_now(),
        sealed_by_commit=read_harness_commit(PROJECT_ROOT),
    )
    write_split_manifest(manifest, path)
    print(f"Bölgü möhürləndi: {path}")
    print(f"  dev:     {len(manifest.dev_ids)} case")
    print(f"  holdout: {len(manifest.holdout_ids)} case")
    print(f"  dataset sha256: {manifest.dataset_sha256[:16]}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    overrides: dict[str, Any] = {}
    if args.repeats is not None:
        overrides["repeats"] = args.repeats
    settings = Settings.load(**overrides)

    dataset = load_dataset(settings.dataset_path)
    variants = load_variants(settings.variants_dir)
    if args.variant not in variants:
        raise EvalError(
            f"'{args.variant}' variantı tapılmadı. "
            f"Mövcud: {', '.join(sorted(variants))}"
        )
    variant = variants[args.variant]
    guard = ContaminationGuard(load_split_manifest(settings.split_manifest_path), dataset)

    runner = Runner(
        settings,
        dataset=dataset,
        guard=guard,
        sut=RagSut(settings, variant=variant),
        judge=AnthropicJudge(settings),
        on_event=lambda message: print(message, flush=True),
    )
    result = runner.run(
        split=args.split, variant=variant, action=args.action, note=args.note
    )

    print(f"\nRun tamamlandı: {result.run_id}")
    print(f"Artefaktlar: {result.paths.root}")
    print(f"Hesabat üçün: python -m eval.cli report {result.run_id}")
    return 0


def cmd_reclassify(args: argparse.Namespace) -> int:
    settings = Settings.load()
    dataset = load_dataset(settings.dataset_path)
    paths = RunPaths.for_run(settings.runs_dir, args.run_id)
    run = load_run(paths)

    causes = reclassify(dataset, run.observations, run.grades, run.verdicts)
    RunWriter(paths, secrets=settings.live_secrets).write_causes(causes)

    print(f"{len(causes)} nəticə yenidən təsnif edildi (şəbəkəyə çıxılmadı).")
    changed = sum(1 for old, new in zip(run.causes, causes) if old.category != new.category)
    print(f"Kateqoriyası dəyişən: {changed}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    settings = Settings.load()
    dataset = load_dataset(settings.dataset_path)
    prices = load_prices(settings.prices_path)
    run = load_run(RunPaths.for_run(settings.runs_dir, args.run_id))
    baseline = (
        load_run(RunPaths.for_run(settings.runs_dir, args.compare))
        if args.compare
        else None
    )

    text = build_report(
        run,
        dataset,
        prices,
        baseline=baseline,
        human_labels=load_human_labels(PROJECT_ROOT / "data" / "human_labels.yaml"),
        label_sources=load_label_sources(PROJECT_ROOT / "data" / "human_labels.yaml"),
        ledger=read_holdout_ledger(settings.runs_dir / HOLDOUT_LEDGER),
    )

    out = Path(args.out) if args.out else settings.logs_dir / f"report_{run.run_id}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Hesabat yazıldı: {out}")
    return 0


def cmd_judge_bias(args: argparse.Namespace) -> int:
    """Hakim ↔ insan razılığı.

    Arqumentsiz işləyir: mənbə run-larını `human_labels.yaml`-ın `sources`
    blokundan götürür. Səbəbi budur ki, 9 etiket iki AYRI run-a bölünüb —
    tək run tələb etmək etiketlərin bir hissəsini həmişə kənarda qoyardı.
    """
    settings = Settings.load()
    labels_path = PROJECT_ROOT / "data" / "human_labels.yaml"
    labels = load_human_labels(labels_path)
    sources = load_label_sources(labels_path)

    # İKİ ƏHATƏ QƏSDƏN FƏRQLİDİR
    #   kappa      — yalnız etiketin bağlı olduğu run/təkrar (insan məhz onu gördü)
    #   verbosity  — mövcud BÜTÜN verdiktlər (sual «hakim uzun cavaba yüksək bal
    #                verirmi» — ona hər verdikt müstəqil müşahidə kimi cavab verir)
    # `run_id` açıq verilibsə hər ikisi həmin run ilə məhdudlaşır.
    out_of_scope: list[str] = []
    if args.run_id:
        kappa_runs = {args.run_id}
        verbosity_runs = {args.run_id}
        # AÇIQ FİLTR XƏTA DEYİL. Etiketlər qəsdən iki run-a bölünüb; istifadəçi
        # birini seçəndə digərinə aid etiketlər «qarşılığı tapılmadı» yox,
        # «əhatədən kənar» sayılmalıdır — əks halda tamamilə düzgün əmr
        # sıfırdan fərqli kodla dayanardı.
        out_of_scope = sorted(
            cid for cid, s in sources.items()
            if cid in labels and s.run_id != args.run_id
        )
        labels = {cid: v for cid, v in labels.items() if cid not in out_of_scope}
    else:
        kappa_runs = {s.run_id for cid, s in sources.items() if cid in labels}
        verbosity_runs = set(list_runs(settings.runs_dir))

    if not labels:
        print(f"İnsan etiketi: 0 case ({labels_path} doldurulmayıb)")
        print("  kappa hesablanmadı — bu boşluq logs/judge_bias.md-də qeyd olunub.")
        return 0
    if not kappa_runs:
        raise EvalError(
            f"{labels_path}: etiketlər var, amma heç birinin `sources` qeydi yoxdur. "
            "Hər etiket üçün `{run_id, repeat}` göstərilməlidir."
        )

    verdicts: dict[tuple[str, str, int], Verdict] = {}
    lengths: dict[tuple[str, str, int], int] = {}
    for run_id in sorted(kappa_runs | verbosity_runs):
        run = load_run(RunPaths.for_run(settings.runs_dir, run_id))
        verdicts.update(run.keyed_verdicts())
        lengths.update(run.keyed_answer_lengths())

    bias = judge_bias(labels, sources, verdicts, lengths)

    print(f"kappa mənbəyi: {', '.join(sorted(kappa_runs))}")
    print(f"verbosity əhatəsi: {len(verbosity_runs)} run")
    print(f"İnsan etiketi: {len(labels)} case, ölçülən: {bias.sample_size}")
    print(f"  {bias.describe()}")

    if out_of_scope:
        print(
            f"  Əhatədən kənar ({len(out_of_scope)}, başqa run-a aiddir): "
            f"{', '.join(out_of_scope)}"
        )
    unlabeled = sorted(set(sources) - set(labels) - set(out_of_scope))
    if unlabeled:
        print(f"  Etiketlənməyib ({len(unlabeled)}): {', '.join(unlabeled)}")
    if bias.unmeasurable:
        print(
            f"  Ölçülə bilmədi (hakim xətası/imtinası, 0 bala ÇEVRİLMİR): "
            f"{', '.join(bias.unmeasurable)}"
        )
    if bias.unmatched:
        # Səssiz atma kappanı «tam» göstərərdi — ona görə bu xətadır.
        print(f"  XƏTA — qarşılığı tapılmayan etiket: {', '.join(bias.unmatched)}")
        print("  Ya `sources` qeydi yanlışdır, ya həmin run/təkrar mövcud deyil.")
        return 1
    if bias.sample_size and bias.kappa < 0.6:
        print("  ⚠️ kappa < 0.60 — hakim-törəmə rəqəmlər ehtiyatla oxunmalıdır.")
    return 0


def cmd_list_runs(args: argparse.Namespace) -> int:
    settings = Settings.load()
    runs = list_runs(settings.runs_dir)
    if not runs:
        print("Hələ run yoxdur.")
        return 0
    for run_id in runs:
        print(run_id)
    ledger = read_holdout_ledger(settings.runs_dir / HOLDOUT_LEDGER)
    print(f"\nHoldout icraları: {len(ledger)}")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.cli",
        description="Week 2 RAG sisteminin qiymətləndirilməsi (Week 4).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_seal = sub.add_parser("seal-split", help="dev/holdout bölgüsünü möhürlə")
    p_seal.add_argument(
        "--force",
        action="store_true",
        help="test dəsti dəyişibsə belə yenidən möhürlə (əvvəlki run-lar etibarsız olur)",
    )
    p_seal.set_defaults(func=cmd_seal_split)

    p_run = sub.add_parser("run", help="qiymətləndirməni işlət")
    p_run.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    p_run.add_argument("--variant", default="baseline")
    p_run.add_argument("--repeats", type=int, default=None)
    p_run.add_argument("--note", default="")
    p_run.set_defaults(func=cmd_run, action="run")

    # `optimize` eyni koddur, YALNIZ bölgü məhdudiyyəti ilə: guard
    # `action="optimize"` görəndə holdout-u rədd edir.
    p_opt = sub.add_parser(
        "optimize",
        help="prompt optimallaşdırma dövrü (yalnız --split dev)",
    )
    p_opt.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    p_opt.add_argument("--variant", default="baseline")
    p_opt.add_argument("--repeats", type=int, default=None)
    p_opt.add_argument("--note", default="")
    p_opt.set_defaults(func=cmd_run, action="optimize")

    p_rec = sub.add_parser(
        "reclassify", help="saxlanmış müşahidələri yenidən təsnif et (şəbəkəsiz)"
    )
    p_rec.add_argument("run_id")
    p_rec.set_defaults(func=cmd_reclassify)

    p_rep = sub.add_parser("report", help="Markdown hesabat yarat")
    p_rep.add_argument("run_id")
    p_rep.add_argument("--compare", default=None, help="baseline run_id")
    p_rep.add_argument("--out", default=None)
    p_rep.set_defaults(func=cmd_report)

    p_bias = sub.add_parser("judge-bias", help="hakim ↔ insan razılığı (kappa)")
    p_bias.add_argument(
        "run_id",
        nargs="?",
        default=None,
        help="Yalnız bu run-la məhdudlaşdır. Verilməzsə mənbə run-ları "
             "data/human_labels.yaml-ın `sources` blokundan götürülür.",
    )
    p_bias.set_defaults(func=cmd_judge_bias)

    p_list = sub.add_parser("list-runs", help="run-ları sadala")
    p_list.set_defaults(func=cmd_list_runs)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ContaminationError as exc:
        print(f"\nÇİRKLƏNMƏ: {exc}", file=sys.stderr)
        return 2
    except EvalError as exc:
        print(f"\nXƏTA: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
