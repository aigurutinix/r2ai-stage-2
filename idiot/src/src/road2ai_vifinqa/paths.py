"""Stable project paths used by every stage of the pipeline."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "data" / "source" / "ViFinQA"
REPORT_ROOT = SOURCE_ROOT / "financial_statements"
QUESTIONS_PATH = SOURCE_ROOT / "questions" / "questions.jsonl"
COMPANIES_PATH = SOURCE_ROOT / "code_stock.csv"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
INDEX_PATH = ARTIFACT_ROOT / "tables.sqlite3"
INDEX_MANIFEST_PATH = ARTIFACT_ROOT / "index_manifest.json"
PANEL_PATH = ARTIFACT_ROOT / "financial_panel.json"
PANEL_MANIFEST_PATH = ARTIFACT_ROOT / "financial_panel_manifest.json"
RUNS_ROOT = PROJECT_ROOT / "runs"
SUBMISSIONS_ROOT = PROJECT_ROOT / "submissions"
