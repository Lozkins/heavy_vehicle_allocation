"""
Запуск сравнительного отчёта (Greedy vs CP-SAT vs MIP/HiGHS) одной командой:

    python main.py
    python main.py --time-limit 90 --mip-gap 0.001

Печатает таблицу в консоль и сохраняет её в results/comparison.md.
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from data import Data
from prepare import Prepare
from report import Report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-limit", type=float, default=300.0, help="лимит времени CP-SAT/MIP, сек (по умолчанию 300)")
    parser.add_argument("--mip-gap", type=float, default=0.0, help="относительный gap CP-SAT/MIP (по умолчанию 0.0 — точное решение)")
    parser.add_argument("--out", type=Path, default=ROOT_DIR / "results" / "comparison.md", help="путь для сохранения markdown-таблицы")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prepare = Prepare(Data())
    report = Report(prepare).run(time_limit_sec=args.time_limit, mip_rel_gap=args.mip_gap)

    print(report.to_markdown())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report.to_markdown() + "\n")
    print(f"\n[main] Таблица сохранена в {args.out}")


if __name__ == "__main__":
    main()
