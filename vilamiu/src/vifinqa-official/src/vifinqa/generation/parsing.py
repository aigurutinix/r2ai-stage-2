
from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json_object(text: str) -> dict:
    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    return json.loads(cleaned)
