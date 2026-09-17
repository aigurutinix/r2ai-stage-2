#!/usr/bin/env python3
"""Fuse cross-encoder scores from Kaggle back into the sparse ranking.

Every replacement tried on this task has lost and every fusion that earned its
place has won, so the reranker enters as one more reciprocal rank rather than as
the new order. That also bounds the damage if the model is wrong: at worst it
pulls the ranking halfway toward its own opinion, instead of replacing ours.

Writes reranked orderings keyed by question ID, which the evaluator and the
submission builder both read.
"""

import argparse
import json
import sys
from pathlib import Path

from vifinqa.fusion import first_stage, load_scores, prune_line_item_tables, rankings
from vifinqa.jsonl import load_jsonl
from vifinqa.slots import load_line_items, named_line_items


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--pairs", type=Path, default=root / "output" / "rerank" / "pairs.jsonl")
    parser.add_argument(
        "--scores", type=Path, action="append", default=[],
        help="score file from a Kaggle run; repeat once per shard",
    )
    parser.add_argument("--output", type=Path, default=root / "output" / "rerank" / "ranking.json")
    parser.add_argument("--mode", choices=("fuse", "replace"), default="fuse")
    parser.add_argument(
        "--weight", type=float, default=0.5,
        help="share of the reciprocal-rank sum given to the sparse rank; 0.5 is the shipped "
             "unweighted fusion and nested CV picks 0.3, a difference inside the noise floor",
    )
    parser.add_argument(
        "--first-stage-only", action="store_true",
        help="rank by the retrievers alone, with no reranker scores at all",
    )
    parser.add_argument(
        "--arm", choices=("sparse", "dense", "hybrid"), default="sparse",
        help="which first stage's candidates to rank: 'sparse' the gated BM25 pool, "
             "'dense' the embedding pool, 'hybrid' both fused. One scored pool, three "
             "rankings, so the comparison is free of the run-to-run scoring noise that "
             "two sessions would introduce",
    )
    parser.add_argument(
        "--line-item-prune", action="store_true",
        help="keep exact line-item carriers and two non-matching recall slots",
    )
    parser.add_argument("--extra-nonmatches", type=int, default=2)
    args = parser.parse_args()

    pairs = {record["id"]: record for record in load_jsonl(args.pairs)}
    # Scores must be named. There is no default path, because there used to be
    # one: a stale score file left beside the pairs was picked up silently and
    # turned three first-stage rankings into reranked ones without a word. No
    # scores is a legitimate configuration — it is the arm's first stage alone —
    # but it has to be asked for.
    if not args.scores and not args.first_stage_only:
        raise SystemExit("pass --scores, or --first-stage-only to rank without a model")
    scored = load_scores(args.scores)
    ranking = rankings(pairs, scored, args.mode, args.weight, args.arm)
    if args.line_item_prune:
        labels = load_line_items(root / "data" / "derived" / "line_items.json")
        ranking = {
            str(identifier): prune_line_item_tables(
                record, ranking[str(identifier)], named_line_items(record["question"], labels),
                args.extra_nonmatches,
            )
            for identifier, record in pairs.items()
        }
    # Against the arm's own first candidate, not the file's: under --arm dense the
    # file still opens with a sparse candidate the dense arm never returned.
    def leader(record):
        ordered = [c for c in record["candidates"] if first_stage(c, args.arm) > 0.0]
        best = min(ordered, key=lambda c: -first_stage(c, args.arm), default=None)
        return best["table_id"] if best else None

    moved = sum(
        1 for identifier, record in pairs.items()
        if (ranking[str(identifier)] or [None])[0] != leader(record)
    )
    empty = sum(1 for order in ranking.values() if not order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ranking, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "mode": args.mode,
        "weight": args.weight,
        "arm": args.arm,
        "questions": len(ranking),
        "questions_with_no_candidate_in_this_arm": empty,
        "scored": len(scored),
        "unscored_kept_as_is": len(set(pairs) - set(scored)),
        "top_table_changed": moved,
        "top_table_unchanged": len(ranking) - moved,
        "line_item_prune": args.line_item_prune,
        "extra_nonmatches": args.extra_nonmatches,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
