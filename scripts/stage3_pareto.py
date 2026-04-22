from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from threesa.analysis import aggregate_results, plot_pareto_frontier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate attack results and plot Pareto frontier.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregated = aggregate_results(args.csv)
    plot_pareto_frontier(aggregated, args.output)
    print(aggregated.to_string(index=False))
    print(f"plot={args.output}")


if __name__ == "__main__":
    main()
