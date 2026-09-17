"""Score how well a label matches an indicator, so the best match can be a single row.

Ranking candidates by match *kind* alone left 27 of 120 questions ambiguous, and the
ambiguity sat almost entirely in one kind: a long variant contains a short label, so
`Tiền` and `Cộng` match a variant like "Tiền và các khoản tương đương tiền" as readily as
the right row does.

The fix is not a stop-list of short words, which would be a rule per observed failure. It
is to score by how much of the label the match actually explains: containment of a long
label is strong evidence, containment of a three-letter label is almost none. So the kind
of match sets the band and the length of what matched orders within it.

    label == variant                 exact, the strongest thing available
    variant inside label             the report printed extra numbering or a suffix
    label inside variant             weakest — and ordered by label length, which is
                                     what stops a short generic row from winning
    tokens overlap                   last resort, ordered by how many tokens agree
"""

from __future__ import annotations

import re
import unicodedata

TOKEN_OVERLAP = 0.7


def fold(text: str) -> str:
    text = str(text).replace("đ", "d").replace("Đ", "D")
    flat = "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn").casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", flat)).strip()


def score(names: list[str], label: str) -> float:
    """0 when nothing matches; higher is a more specific match."""

    flat_label = fold(label)
    if not flat_label or not names:
        return 0.0
    label_words = set(flat_label.split())
    best = 0.0
    for name in names:
        flat = fold(name)
        if not flat:
            continue
        words = set(flat.split())
        if flat == flat_label:
            best = max(best, 4000.0 + len(flat_label))
        elif flat in flat_label:
            best = max(best, 3000.0 + len(flat))
        elif flat_label in flat:
            # Ordered by how much of the variant the label covers, so a short generic
            # row cannot outrank a long specific one.
            best = max(best, 2000.0 + len(flat_label))
        else:
            shared = words & label_words
            if shared and len(shared) / min(len(words), len(label_words)) >= TOKEN_OVERLAP:
                best = max(best, 1000.0 + 10 * len(shared))
    return best
