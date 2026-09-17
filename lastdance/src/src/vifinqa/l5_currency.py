"""Specialized monetary-position and FX-sensitivity plans for q0427-q0428."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import l5_screening as engine
from .l5_screening import Builder, L5Spec, series


CURRENCY_IDS = (427, 428)


def make_specs() -> dict[int, L5Spec]:
    specs: dict[int, L5Spec] = {}

    b = Builder(427)
    liabilities, assets, sensitivity = [], [], []
    for code in ("USD", "EUR", "JPY", "SGD"):
        liabilities.append(b.fact("FPT", 2016, f"cong no tien te {code}"))
        assets.append(b.fact("FPT", 2016, f"tai san tien te {code}"))
        sensitivity.append(b.fact("FPT", 2016, f"do nhay loi nhuan truoc thue {code}"))
    lv, av, sv = series(liabilities), series(assets), series(sensitivity)
    specs[427] = b.spec(f"float({sv}.abs()[{lv} > {av}].sum() / 1000000000.0)")

    b = Builder(428)
    positions = [
        b.fact("ACB", 2024, f"trang thai tien te noi ngoai bang {code}")
        for code in ("USD", "EUR", "JPY", "AUD", "CAD", "khac")
    ]
    profit_before_tax = b.fact("ACB", 2024, "tong loi nhuan truoc thue")
    pv = series(positions)
    specs[428] = b.spec(
        f"float({pv}.abs()[{pv} < 0.0].max() * 0.05 / abs({profit_before_tax}) * 100.0)"
    )
    return specs


CURRENCY_SPECS = make_specs()


def activate() -> None:
    engine.L5_SPECS = CURRENCY_SPECS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    candidates = sub.add_parser("candidates")
    candidates.add_argument("--questions", type=Path, default=Path("ViFinQA/questions/questions.jsonl"))
    candidates.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    candidates.add_argument("--output-dir", type=Path, default=Path("outputs/l5-currency-facts"))
    candidates.add_argument("--overrides", type=Path, default=Path("analysis/l5_currency_manual_overrides.csv"))
    submission = sub.add_parser("submission")
    submission.add_argument("--plans", type=Path, default=Path("outputs/l5-currency-facts/plans.jsonl"))
    submission.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    submission.add_argument("--base-submission", type=Path, default=Path("outputs/l1-l2-l3-l4-l5-scenario-submission-final/submission.json"))
    submission.add_argument("--output-dir", type=Path, default=Path("outputs/l1-l2-l3-l4-l5-full-submission"))
    validate = sub.add_parser("validate")
    validate.add_argument("--submission", type=Path, required=True)
    validate.add_argument("--plans", type=Path, default=Path("outputs/l5-currency-facts/plans.jsonl"))
    validate.add_argument("--base-submission", type=Path, required=True)
    validate.add_argument("--zip", dest="zip_path", type=Path)
    args = parser.parse_args(argv); activate()
    if args.command == "candidates":
        engine.run_candidates(args.questions, args.database, args.output_dir, args.overrides)
    elif args.command == "submission":
        engine.build_submission(args.plans, args.database, args.base_submission, args.output_dir)
    else:
        print(json.dumps(engine.validate_submission(args.submission, args.plans, args.base_submission, args.zip_path), ensure_ascii=False))


if __name__ == "__main__":
    main()
