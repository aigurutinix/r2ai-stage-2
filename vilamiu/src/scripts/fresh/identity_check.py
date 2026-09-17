"""Offline arithmetic identities over BCTC CSVs (outside model / retrieval).

Checks (Circular 200 + period continuity):
  1. Roll-up: 270 = 100 + 200, 440 = 300 + 400, 270 = 440
  2. Cross-year: current(code, year N) ≈ prior(code, year N+1)
  3. (optional) statement self-consistency score for a single table

These are facts about the OCR corpus, not opinions. Use to:
  - measure how clean the extract is (offline metric)
  - rerank / keep a table only when identities hold
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fresh"))

import parse_statements as ps  # noqa: E402

# Parent = sum(children). Values already in đồng after parse_statement.
ROLLUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("cdkt", "270", ("100", "200")),
    ("cdkt", "440", ("300", "400")),
)

# Balance sheet must balance.
BALANCES: tuple[tuple[str, str, str], ...] = (
    ("cdkt", "270", "440"),
)

# Cross-year continuity on these codes (end N ↔ begin N+1).
CROSS_CODES: tuple[str, ...] = ("100", "110", "140", "200", "270", "300", "310", "400", "440")


@dataclass
class IdResult:
    name: str
    ok: bool
    rel: float
    detail: str


def rel_err(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1.0)
    return abs(a - b) / denom


def close(a: float, b: float, tol: float = 1e-4) -> bool:
    """Relative tolerance; OCR/scale noise allowed at ~0.01%."""

    return rel_err(a, b) <= tol


def load_statement_maps(
    ticker: str, year: str, scope: str,
) -> dict[str, ps.Statement]:
    """kind -> best Statement (most codes) for that doc scope."""

    root = ROOT / "data" / "official_corpus" / ticker / year
    if not root.is_dir():
        return {}
    best: dict[str, ps.Statement] = {}
    for doc_dir in root.iterdir():
        if not doc_dir.is_dir():
            continue
        doc = doc_dir.name
        if scope == "consolidated" and "consolidated" not in doc and "aggregated" not in doc:
            continue
        if scope == "separate" and "separate" not in doc:
            continue
        table_dir = doc_dir / f"{doc}_extracted_tables"
        if not table_dir.is_dir():
            continue
        for csv_path in sorted(table_dir.glob("table_*.csv")):
            rows = list(csv.reader(csv_path.open(encoding="utf-8-sig")))
            if len(rows) < 2:
                continue
            scale = ps.scale_from_unit_text(
                ",".join(c for r in rows[:3] for c in r), strict=False)
            table = ps.Table(
                doc_name=doc, table_id=int(csv_path.stem.split("_")[-1]),
                page_no=0, header=rows[0], rows=rows[1:],
                unit_snippets=("Đơn vị tính: Đồng",) if scale is None else (),
            )
            stmt = ps.parse_statement(table)
            if stmt is None:
                continue
            prev = best.get(stmt.kind)
            if prev is None or len(stmt.current) > len(prev.current):
                best[stmt.kind] = stmt
    return best


def check_rollups(stmts: dict[str, ps.Statement], tol: float = 1e-4) -> list[IdResult]:
    out: list[IdResult] = []
    for kind, parent, kids in ROLLUPS:
        stmt = stmts.get(kind)
        if stmt is None:
            continue
        if parent not in stmt.current:
            continue
        if any(k not in stmt.current for k in kids):
            continue
        lhs = stmt.current[parent].value
        rhs = sum(stmt.current[k].value for k in kids)
        ok = close(lhs, rhs, tol)
        out.append(IdResult(
            f"rollup:{kind}:{parent}=sum{kids}",
            ok, rel_err(lhs, rhs),
            f"{lhs:.4g} vs {rhs:.4g}",
        ))
    for kind, left, right in BALANCES:
        stmt = stmts.get(kind)
        if stmt is None:
            continue
        if left not in stmt.current or right not in stmt.current:
            continue
        a, b = stmt.current[left].value, stmt.current[right].value
        ok = close(a, b, tol)
        out.append(IdResult(
            f"balance:{kind}:{left}={right}",
            ok, rel_err(a, b),
            f"{a:.4g} vs {b:.4g}",
        ))
    return out


def check_cross_year(
    cur: dict[str, ps.Statement],
    nxt: dict[str, ps.Statement],
    tol: float = 5e-3,
) -> list[IdResult]:
    """current(year N) vs prior(year N+1) — same economic stock."""

    out: list[IdResult] = []
    a = cur.get("cdkt")
    b = nxt.get("cdkt")
    if a is None or b is None:
        return out
    for code in CROSS_CODES:
        if code not in a.current or code not in b.prior:
            continue
        left = a.current[code].value
        right = b.prior[code].value
        ok = close(left, right, tol)
        out.append(IdResult(
            f"cross:cdkt:{code}",
            ok, rel_err(left, right),
            f"end={left:.4g} begin_next={right:.4g}",
        ))
    return out


def score_ticker_year(
    ticker: str, year: str, scope: str = "consolidated",
) -> dict:
    stmts = load_statement_maps(ticker, year, scope)
    roll = check_rollups(stmts)
    nxt_year = str(int(year) + 1)
    nxt = load_statement_maps(ticker, nxt_year, scope)
    cross = check_cross_year(stmts, nxt) if nxt else []
    checks = roll + cross
    n_ok = sum(1 for c in checks if c.ok)
    return {
        "ticker": ticker,
        "year": year,
        "scope": scope,
        "n_checks": len(checks),
        "n_ok": n_ok,
        "pass_rate": (n_ok / len(checks)) if checks else None,
        "rollup_ok": all(c.ok for c in roll) if roll else None,
        "cross_ok": all(c.ok for c in cross) if cross else None,
        "n_rollup": len(roll),
        "n_cross": len(cross),
        "fails": [c.name for c in checks if not c.ok][:8],
    }


def list_corpus_pairs(scope: str = "consolidated") -> list[tuple[str, str]]:
    root = ROOT / "data" / "official_corpus"
    pairs: list[tuple[str, str]] = []
    if not root.is_dir():
        return pairs
    for ticker_dir in sorted(root.iterdir()):
        if not ticker_dir.is_dir():
            continue
        for year_dir in sorted(ticker_dir.iterdir()):
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
            # skip if no matching scope doc
            if not any(
                scope in p.name or (scope == "consolidated" and "aggregated" in p.name)
                for p in year_dir.iterdir() if p.is_dir()
            ):
                continue
            pairs.append((ticker_dir.name, year_dir.name))
    return pairs


def table_identity_score(csv_path: Path, tol: float = 1e-4) -> dict | None:
    """Score one CSV table in isolation (roll-ups only)."""

    rows = list(csv.reader(csv_path.open(encoding="utf-8-sig")))
    if len(rows) < 2:
        return None
    scale = ps.scale_from_unit_text(
        ",".join(c for r in rows[:3] for c in r), strict=False)
    table = ps.Table(
        doc_name=csv_path.parent.parent.name,
        table_id=int(csv_path.stem.split("_")[-1]),
        page_no=0, header=rows[0], rows=rows[1:],
        unit_snippets=("Đơn vị tính: Đồng",) if scale is None else (),
    )
    stmt = ps.parse_statement(table)
    if stmt is None or stmt.kind != "cdkt":
        return None
    fake = {"cdkt": stmt}
    checks = check_rollups(fake, tol=tol)
    if not checks:
        return {
            "applicable": False, "n_ok": 0, "n": 0, "score": 0.0,
            "kind": stmt.kind, "n_codes": len(stmt.current), "stmt": stmt,
        }
    n_ok = sum(1 for c in checks if c.ok)
    return {
        "applicable": True,
        "n_ok": n_ok,
        "n": len(checks),
        "score": n_ok / len(checks),
        "fails": [c.name for c in checks if not c.ok],
        "kind": stmt.kind,
        "n_codes": len(stmt.current),
        "stmt": stmt,
    }


def doc_union_rollup_ok(parent: Path, tol: float = 1e-4) -> bool | None:
    """Union CDKT codes across page-split tables; test TT200 roll-ups."""

    merged: dict[str, ps.Cell] = {}
    for path in sorted(parent.glob("table_*.csv")):
        scored = table_identity_score(path, tol=tol)
        if scored is None:
            continue
        for code, cell in scored["stmt"].current.items():
            merged.setdefault(code, cell)
    if not merged:
        return None
    fake = ps.Statement(
        doc_name=parent.parent.name, table_id=-1, kind="cdkt",
        scale=1.0, current=merged, prior={},
    )
    checks = check_rollups({"cdkt": fake}, tol=tol)
    if not checks:
        return None
    return all(c.ok for c in checks)


def prefer_identity_binding(
    csv_path: Path,
    code: str,
    kind: str = "cdkt",
    tol: float = 1e-4,
    extra_paths: list[Path] | None = None,
) -> dict | None:
    """Prefer richest CDKT fragment carrying `code` when doc union is clean.

    Page-split BS: 100/110 live on one CSV, 270 on another — single-table
    roll-up is often N/A. Union the doc first; then among tables that have
    `code`, rank by (own rollup score, n_codes) and flip if winner is better.

    `extra_paths`: retrieval top-k (and their same-doc siblings) already filtered
    to the same ticker/year/scope by the caller. Never invent a number — only
    re-bind the Mã số onto an identity-cleaner fragment.
    """

    code = str(code).lstrip("t")
    if kind and kind.lower() not in ("cdkt", "bang", ""):
        return None
    if not csv_path.is_file():
        return None
    parent = csv_path.parent
    if not parent.is_dir():
        return None

    doc_ok = doc_union_rollup_ok(parent, tol=tol)

    pool: list[Path] = list(sorted(parent.glob("table_*.csv")))
    seen: set[Path] = {p.resolve() for p in pool}
    for path in extra_paths or ():
        if not path.is_file():
            continue
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        pool.append(path)

    candidates: list[tuple[float, int, Path, ps.Cell]] = []
    for path in pool:
        scored = table_identity_score(path, tol=tol)
        if scored is None:
            continue
        stmt: ps.Statement = scored["stmt"]
        if code not in stmt.current:
            continue
        own = float(scored["score"]) if scored.get("applicable") else 0.0
        candidates.append((own, int(scored["n_codes"]), path, stmt.current[code]))

    if not candidates:
        return None
    if not any(own >= 1.0 for own, *_ in candidates) and doc_ok is not True:
        return None

    candidates.sort(key=lambda t: (-t[0], -t[1]))
    best_own, best_n, best_path, cell = candidates[0]

    greedy_key = None
    for own, n_codes, path, _c in candidates:
        if path.resolve() == csv_path.resolve():
            greedy_key = (own, n_codes)
            break
    winner_key = (best_own, best_n)
    # No upgrade vs greedy → keep caller address (avoid churn on re-parse).
    if greedy_key is not None and winner_key <= greedy_key:
        if best_path.resolve() == csv_path.resolve():
            return {
                "path": best_path,
                "row": cell.row_idx,
                "col": cell.col_idx,
                "scale": cell.scale,
                "table_id": int(best_path.stem.split("_")[-1]),
                "score": best_own,
                "n_codes": best_n,
                "flipped": False,
                "n_cand": len(candidates),
                "doc_ok": doc_ok,
            }
        return None

    # Cross-doc flips require a perfect roll-up (retrieval can propose wrong scope).
    flipped = best_path.resolve() != csv_path.resolve()
    if flipped and best_path.parent.resolve() != parent.resolve() and best_own < 1.0:
        return None

    return {
        "path": best_path,
        "row": cell.row_idx,
        "col": cell.col_idx,
        "scale": cell.scale,
        "table_id": int(best_path.stem.split("_")[-1]),
        "score": best_own,
        "n_codes": best_n,
        "flipped": flipped,
        "n_cand": len(candidates),
        "doc_ok": doc_ok,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="consolidated")
    parser.add_argument("--limit", type=int, default=0, help="0 = all pairs")
    parser.add_argument("--tol-rollup", type=float, default=1e-4)
    parser.add_argument("--out", default="artifacts/fresh/identity_corpus.jsonl")
    args = parser.parse_args()

    pairs = list_corpus_pairs(args.scope)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"pairs={len(pairs)} scope={args.scope}", flush=True)

    rows: list[dict] = []
    roll_stats = Counter()
    cross_stats = Counter()
    for i, (ticker, year) in enumerate(pairs):
        rec = score_ticker_year(ticker, year, args.scope)
        rows.append(rec)
        if rec["rollup_ok"] is True:
            roll_stats["ok"] += 1
        elif rec["rollup_ok"] is False:
            roll_stats["fail"] += 1
        else:
            roll_stats["na"] += 1
        if rec["cross_ok"] is True:
            cross_stats["ok"] += 1
        elif rec["cross_ok"] is False:
            cross_stats["fail"] += 1
        else:
            cross_stats["na"] += 1
        if (i + 1) % 50 == 0:
            print(f"  …{i+1}/{len(pairs)}", flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )

    with_checks = [r for r in rows if r["n_checks"]]
    avg = (
        sum(r["pass_rate"] or 0 for r in with_checks) / len(with_checks)
        if with_checks else 0.0
    )
    print(f"-> {out}")
    print(f"ticker-years with checks: {len(with_checks)}/{len(rows)}")
    print(f"mean pass_rate: {avg:.3f}")
    print(f"rollup all-ok: {dict(roll_stats)}")
    print(f"cross  all-ok: {dict(cross_stats)}")

    # fail histogram
    fail_c: Counter[str] = Counter()
    for r in rows:
        for f in r.get("fails") or []:
            fail_c[f.split(":")[0] + ":" + f.split(":")[-1] if ":" in f else f] += 1
    print("top fails:", fail_c.most_common(12))


if __name__ == "__main__":
    main()
