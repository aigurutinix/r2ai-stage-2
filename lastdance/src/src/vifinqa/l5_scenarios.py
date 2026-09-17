"""Auditable scenario/stress-test plans for the final ViFinQA L5 questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import l5_screening as engine
from .l5_advanced import (
    asset_turnover,
    debt_assets,
    fixed_asset_average,
    fixed_asset_turnover,
    quick_ratio,
    roe,
)
from .l5_screening import (
    Builder,
    L5Spec,
    cogs,
    gross_profit,
    interest,
    interest_coverage,
    inventory,
    inventory_days,
    liabilities,
    operating_profit,
    pat,
    pbt,
    ratio,
    revenue,
    series,
    total_assets,
)


SCENARIO_IDS = (419, 423, 424, 425, 432, 433, 434, 435, 436)


def make_specs() -> dict[int, L5Spec]:
    specs: dict[int, L5Spec] = {}

    b = Builder(419); tickers = ("BSR", "PLX", "PVT")
    cover = [interest_coverage(b,t,2024) for t in tickers]
    scenario_margin = [
        ratio(f"(({pbt(b,t,2024)}) - 0.2 * abs({interest(b,t,2024)}))", revenue(b,t,2024), 100.0)
        for t in tickers
    ]
    cv = series(cover)
    specs[419] = b.spec(f"float({series(scenario_margin)}[{cv} > 2.0].min())")

    b = Builder(423); tickers = ("GEX", "HBC", "PC1", "SAM", "VGC")
    quick = [quick_ratio(b,t,2024) for t in tickers]; qv = series(quick)
    stressed = [f"(0.85 * (({pbt(b,t,2024)}) + abs({interest(b,t,2024)})) / abs({interest(b,t,2024)}))" for t in tickers]
    raw = f"{series(stressed)}[{qv} < {qv}.median()].min()"
    specs[423] = b.spec(f"float(pd.Series([{raw}]).round(2).iloc[0])")

    b = Builder(424); tickers = ("DCM", "GVR", "HT1")
    days = [inventory_days(b,t,2024) for t in tickers]; dv = series(days)
    release = [f"((({inventory_days(b,t,2024)}) - ({dv}.median())) * abs({cogs(b,t,2024)}) / 365.0 / 1000000000.0)" for t in tickers]
    excess = [f"(({inventory_days(b,t,2024)}) - ({dv}.median()))" for t in tickers]
    specs[424] = b.spec(f"float({series(release)}.iloc[int({series(excess)}.idxmax())])")

    b = Builder(425); years = (2021,2022,2023,2024)
    eps = [b.fact("FPT", y, "lai co ban tren co phieu") for y in years]
    # Issuing 10% more shares reduces EPS by old_EPS - old_EPS/1.1 = old_EPS/11.
    specs[425] = b.spec(
        f"float({series([f'(({x}) / 11.0 / 1000.0)' for x in eps])}.iloc[int({series([roe(b,'FPT',y) for y in years])}.idxmax())])"
    )

    b = Builder(432); tickers = ("ASM", "DBC", "MML", "MPC", "MSN", "OGC", "QNS", "SAB", "VNM", "VSF")
    turns = [fixed_asset_turnover(b,t,2023) for t in tickers]; tv = series(turns)
    needed = [f"((({tv}.median() / ({x})) - 1.0) * 100.0)" for x in turns]
    specs[432] = b.spec(f"float({series(needed)}[{tv} < {tv}.median()].max())")

    b = Builder(433); tickers = ("CRE", "HPX", "KBC", "KHG", "NVL", "SNZ", "SSH", "VIC", "VPI", "VRE")
    da = [debt_assets(b,t,2023) for t in tickers]; dav = series(da)
    debts = [liabilities(b,t,2023) for t in tickers]; debtv = series(debts)
    stressed_net = []
    for t in tickers:
        short = b.fact(t, 2023, "cac khoan phai thu ngan han")
        long = b.fact(t, 2023, "cac khoan phai thu dai han")
        stressed_net.append(
            f"(({total_assets(b,t,2023)}) - 0.3 * (abs({short}) + abs({long})) - "
            f"0.5 * abs({inventory(b,t,2023)}) - ({liabilities(b,t,2023)}))"
        )
    sv = series(stressed_net); high = f"({dav} > {dav}.median())"
    specs[433] = b.spec(f"float({debtv}[{high} & ({sv} < 0.0)].sum() / {debtv}[{high}].sum() * 100.0)")

    b = Builder(434); tickers = ("DIG", "HPX", "SNZ", "SSH", "VRE")
    cover = [interest_coverage(b,t,2023) for t in tickers]; cv = series(cover)
    headroom = [f"((({x}) / 2.0 - 1.0) * 100.0)" for x in cover]
    specs[434] = b.spec(f"float({series(headroom)}[{cv} > 2.0].min())")

    b = Builder(435); tickers = ("CRE", "DIG", "HPX", "KHG", "SNZ", "SSH", "VRE")
    stressed_cover = [
        f"((({pbt(b,t,2023)}) - 0.1 * ({gross_profit(b,t,2023)}) + abs({interest(b,t,2023)})) / abs({interest(b,t,2023)}))"
        for t in tickers
    ]
    specs[435] = b.spec(f"int(({series(stressed_cover)} < 1.5).sum())", "int")

    b = Builder(436); tickers = ("GEE", "GEX", "HHV", "SAM", "SJG", "VGC")
    margins = [ratio(operating_profit(b,t,2023), revenue(b,t,2023)) for t in tickers]
    needed = [
        f"(0.05 * abs({cogs(b,t,2023)}) / (({revenue(b,t,2023)}) - ({operating_profit(b,t,2023)})) * 100.0)"
        for t in tickers
    ]
    mv = series(margins)
    specs[436] = b.spec(f"float({series(needed)}[{mv} > 0.0].max())")

    if tuple(sorted(specs)) != SCENARIO_IDS:
        raise ValueError("Incomplete scenario registry")
    return specs


SCENARIO_SPECS = make_specs()


def activate() -> None:
    engine.L5_SPECS = SCENARIO_SPECS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    candidates = sub.add_parser("candidates")
    candidates.add_argument("--questions", type=Path, default=Path("ViFinQA/questions/questions.jsonl"))
    candidates.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    candidates.add_argument("--output-dir", type=Path, default=Path("outputs/l5-scenario-facts"))
    candidates.add_argument("--overrides", type=Path, default=Path("analysis/l5_scenario_manual_overrides.csv"))
    submission = sub.add_parser("submission")
    submission.add_argument("--plans", type=Path, default=Path("outputs/l5-scenario-facts/plans.jsonl"))
    submission.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    submission.add_argument("--base-submission", type=Path, default=Path("outputs/l1-l2-l3-l4-l5-advanced-submission-final/submission.json"))
    submission.add_argument("--output-dir", type=Path, default=Path("outputs/l1-l2-l3-l4-l5-scenario-submission"))
    validate = sub.add_parser("validate")
    validate.add_argument("--submission", type=Path, required=True)
    validate.add_argument("--plans", type=Path, default=Path("outputs/l5-scenario-facts/plans.jsonl"))
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
