"""P2 driver: questions.jsonl -> artifacts/questions_parsed.parquet + coverage report"""

from __future__ import annotations

import collections
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifin.query.parse import parse_all  # noqa: E402


def main() -> None:
    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    parsed = parse_all(root / "data" / "questions" / "questions.jsonl", root / "data" / "code_stock.csv")

    frame = pd.DataFrame([asdict(p) for p in parsed])
    frame["tickers"] = frame["tickers"].apply(list)
    frame["years"] = frame["years"].apply(list)
    out = root / "artifacts" / "questions_parsed.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)

    total = len(parsed)
    resolved = sum(1 for p in parsed if p.tickers)
    print(f"questions            {total}")
    print(f"ticker resolved      {resolved} ({resolved / total:.1%})")
    print(f"year resolved        {sum(1 for p in parsed if p.years)}")
    print(f"scope=separate       {sum(1 for p in parsed if p.scope == 'separate')}")
    print(f"multi_entity         {sum(1 for p in parsed if p.multi_entity)}")
    print(f"multi_year           {sum(1 for p in parsed if p.multi_year)}")
    print(f"point_in_time        {sum(1 for p in parsed if p.point_in_time)}")
    print(f"convertible unit     {sum(1 for p in parsed if p.unit_scale is not None)}")
    units = collections.Counter(p.target_unit or "<none>" for p in parsed)
    print("units                " + ", ".join(f"{k}={v}" for k, v in units.most_common()))
    print(f"\nwrote {out}")
    for p in parsed:
        if p.unresolved:
            print(f"  unresolved id={p.id}: {p.question[:88]}")


if __name__ == "__main__":
    main()
