"""The reranker prompt exists twice; this pins the copies together.

kaggle/rerank_qwen_8b.py is pasted into a notebook and cannot import from the
package, so its INSTRUCTION, PREFIX and SUFFIX are duplicated in
demo/rerank_prompt.py. A change to one and not the other scores the model on a
prompt it was never asked to apply.
"""

from __future__ import annotations

import re
from pathlib import Path

from demo.rerank_prompt import INSTRUCTION, PREFIX, SUFFIX, rerank_prompt

KAGGLE = Path(__file__).resolve().parents[1] / "kaggle" / "rerank_qwen_8b.py"


def _literal(name: str) -> str:
    source = KAGGLE.read_text(encoding="utf-8")
    match = re.search(rf"^{name} = (\(.*?\n\)|\".*?\")\n", source, re.S | re.M)
    assert match, f"{name} not found in {KAGGLE}"
    namespace: dict[str, object] = {}
    exec(compile(f"{name} = {match.group(1)}", str(KAGGLE), "exec"), namespace)
    return namespace[name]


def test_instruction_matches_the_off_box_scorer():
    assert INSTRUCTION == _literal("INSTRUCTION")


def test_prefix_and_suffix_match_the_off_box_scorer():
    assert PREFIX == _literal("PREFIX")
    assert SUFFIX == _literal("SUFFIX")


def test_prompt_places_query_and_document_in_their_slots():
    prompt = rerank_prompt("câu hỏi", "bảng")
    assert prompt.startswith(PREFIX)
    assert prompt.endswith(SUFFIX)
    assert "<Query>: câu hỏi\n<Document>: bảng" in prompt
    assert prompt.index("<Instruct>:") < prompt.index("<Query>:") < prompt.index("<Document>:")
