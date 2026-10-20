"""Command-line interface.

    python -m xaudit.cli benchmark            # ground-truth experiment -> results/
    python -m xaudit.cli audit --dataset breast_cancer --model random_forest --n 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from xaudit.config import DEFAULT_TOP_K, MODEL_LABELS

ROOT = Path(__file__).resolve().parents[1]


def _benchmark(args) -> None:
    from xaudit.benchmark import run_benchmark, summarise_benchmark

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def prog(p):
        sys.stdout.write(f"\r  progress {p:6.1%}")
        sys.stdout.flush()

    print("Running ground-truth benchmark…")
    df = run_benchmark(n_instances=args.instances, seed=args.seed, progress=prog)
    print(f"\n  done in {time.time() - t0:.1f}s")
    df.to_csv(out / "benchmark.csv", index=False)
    s = summarise_benchmark(df)
    lines = ["# Ground-truth benchmark results", "",
             f"Predictions per (rho, model): {args.instances} · seed {args.seed}", "",
             f"**Trust vs correctness (instance level): r = {s['trust_precision_corr']:+.3f}**", "",
             "## Mean Explanation Trust Score (rho x model)", "", "```\n" + s["trust"].round(1).to_string() + "\n```", "",
             "## True-driver precision (rho x method)", "", "```\n" + s["precision"].round(3).to_string() + "\n```", "",
             "## True-driver precision (model x method)", "", "```\n" + s["by_model_method"].round(3).to_string() + "\n```", ""]
    (out / "benchmark_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved {out / 'benchmark.csv'} and {out / 'benchmark_summary.md'}")


def _audit(args) -> None:
    from xaudit.audit import audit_dataset
    from xaudit.data import load_dataset
    from xaudit.models import train_model

    tm = train_model(load_dataset(args.dataset), args.model, seed=args.seed)
    print(f"{tm.label} on {args.dataset}: accuracy {tm.metrics['accuracy']:.3f}, AUC {tm.metrics['roc_auc']:.3f}")
    g = audit_dataset(tm, n=args.n, k=args.k, seed=args.seed)
    print(json.dumps(g.headline(), indent=2))
    print("\nFaithfulness (mean):\n", g.faithfulness.round(4).to_string())
    if g.ground_truth is not None:
        print("\nGround truth:\n", g.ground_truth.round(3).to_string())
    if args.out:
        g.per_instance.to_csv(args.out, index=False)
        print(f"\nSaved per-instance results to {args.out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="xaudit", description="XAI Disagreement Auditor")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("benchmark", help="Run the ground-truth correlation benchmark")
    b.add_argument("--instances", type=int, default=20)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", default=str(ROOT / "results"))
    b.set_defaults(func=_benchmark)

    a = sub.add_parser("audit", help="Global audit of one dataset/model")
    a.add_argument("--dataset", default="credit_risk")
    a.add_argument("--model", default="gradient_boosting", choices=list(MODEL_LABELS))
    a.add_argument("--n", type=int, default=40)
    a.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--out", default=None)
    a.set_defaults(func=_audit)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
