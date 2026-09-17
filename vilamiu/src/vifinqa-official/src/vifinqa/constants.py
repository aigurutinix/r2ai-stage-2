
from __future__ import annotations

# Keep LLM behavior within the declared contract.
ANSWER_ABS_TOL = 1e-2

MIN_ROWS = 3
MIN_NUMERIC_CELLS = 6

# Preserve the default behavior for compatibility.
RETRIEVAL_K_SWEEP = (1, 3, 5, 10, 20, 50, 100, 500)

# Keep LLM behavior within the declared contract.
E2E_K_SWEEP = (1, 3, 5, 10)

# Keep LLM behavior within the declared contract.
DEFAULT_MAX_CONTEXT_TABLES = 5

TABLE_HEADER_MAX_CHARS = 2000

EMBEDDING_MAX_SEQ_LENGTH = 1024

# Row-chunk representation (metadata_context_row_chunks). Character caps are intentionally
# model-independent; token length is audited separately with the concrete BGE-M3 tokenizer.
DEFAULT_CHUNK_ROWS = 10
DEFAULT_CHUNK_OVERLAP_ROWS = 2
DEFAULT_CONTEXT_MAX_CHARS = 800
DEFAULT_CHUNK_MAX_CHARS = 1_800
DEFAULT_SUMMARY_MAX_CHARS = 2_000

CHUNKED_INDEX_FORMAT_VERSION = 1
# Cache behavior is part of the run contract.
MULTI_VIEW_INDEX_FORMAT_VERSION = 2

# Cache behavior is part of the run contract.
INDEX_FORMAT_VERSION = 4

BM25_TOKENIZER_VERSION = 2
