"""The prompt the Qwen3 reranker is scored with, for the demo's own scoring call.

Byte-identical to the copy in kaggle/rerank_qwen_8b.py, which is pasted into a
notebook and so cannot import from anywhere; a test pins the two together.

This lives under demo/ rather than in the package because the demo must not
change the submission methodology's own modules. Nothing here is new
behaviour: it is the string the frozen run already scored with, written down
where the demo can reach it.

Serving a different instruction scores the model on a rule it was never asked
to apply. vLLM's own rerank route substitutes a generic web-search
instruction, and on one check that pushed the gold table from rank 1 to rank
21 of 50.
"""

from __future__ import annotations

INSTRUCTION = (
    "The Query asks about a Vietnamese listed company's financial statements. "
    "The Document is one table from a filing whose first line identifies its "
    "ticker, reporting period, and scope (hợp nhất or công ty mẹ). The candidate "
    "pool is already narrowed to compatible filings, but reject a Document when "
    "that filing line conflicts with the Query. Answer yes only when the table "
    "reports an exact requested accounting item with a value for the requested "
    "period. For a multi-step question, a table is yes when it supplies any one "
    "required input. Do not accept a merely related label, a reference to an "
    "item, a related-party or subsidiary listing, a movement or allocation "
    "schedule, or a column header. Prefer the statement or note that reports the "
    "figure; a note or segment breakdown that validly restates it is yes."
)

PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)

SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def rerank_prompt(query: str, document: str) -> str:
    """One scored pair, rendered the way the frozen run rendered it."""
    return (
        f"{PREFIX}<Instruct>: {INSTRUCTION}\n<Query>: {query}"
        f"\n<Document>: {document}{SUFFIX}"
    )
