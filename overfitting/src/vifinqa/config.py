"""Central paths & constants. Everything can be overridden by CLI args."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Input data (as downloaded from HuggingFace) ---
DATA_DIR = ROOT / "data" / "ViFinQA"
FS_DIR = DATA_DIR / "financial_statements"
QUESTIONS_JSONL = DATA_DIR / "questions" / "questions.jsonl"
CODE_STOCK_CSV = DATA_DIR / "code_stock.csv"

# --- Artifacts produced by the pipeline ---
ART_DIR = ROOT / "artifacts"
STORE_DIR = ART_DIR / "store"                 # tables/{ticker}.parquet, cells/{ticker}.parquet, reports.parquet
RETRIEVAL_JSONL = ART_DIR / "retrieval.jsonl"
CODEGEN_JSONL = ART_DIR / "codegen_results.jsonl"
SUBMISSION_DIR = ART_DIR / "submission"
KAGGLE_PAYLOAD_DIR = ART_DIR / "kaggle_payload"
VALIDATION_DIR = ART_DIR / "validation"

# --- Competition format knobs ---
# "vi tri bang trong bao cao" — CONFIRMED BY THE ORGANIZERS (and verified on the
# leaderboard): the position is the LINE NUMBER (1-based) where the table starts
# in the OCR .txt file. Internally tables are still keyed by table_pos (0-based
# order of appearance); the mapping table_pos -> line_no lives in the store
# (column line_no) and is applied at submission build time.
TABLE_POS_MODE = "line"   # "line" = official; "order" kept only for debugging
TABLE_POS_BASE = 0        # only used in legacy "order" mode

# Retrieval depth kept in retrieval.jsonl (candidates per question)
RETRIEVE_DEPTH = 20
# k tables actually submitted in relevant_tables.
# Leaderboard calibration (pos_mode=line):
#   k=10 -> P=0.176 R=0.684 F2=0.364; gold ~2.6 tables/question, ~1 doc/question
#   k=5  -> P=0.2621 R=0.5852 F2macro=0.4092, MRR5=0.5882
# k=5 is the best tested cutoff, but has not been compared with k=7. F2 is
# macro-averaged per question and cannot be reconstructed from aggregate P/R.
SUBMISSION_K = 5
# tables shown to the codegen LLM
CODEGEN_K = 6

# --- Units ---
UNIT_SCALES = {
    "dong": 1.0,
    "nghin": 1e3,
    "trieu": 1e6,
    "ty": 1e9,
    "tram_ty": 1e11,
    "nghin_ty": 1e12,
}

YEAR_MIN, YEAR_MAX = 2010, 2026
