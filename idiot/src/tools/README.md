# Công cụ audit, benchmark và release

Chạy mọi lệnh từ repository root với `PYTHONPATH=src`.

## Build/benchmark artifact

- `audit_artifacts.py`: schema, count, coordinate, raw/value và SQLite integrity.
- `benchmark_preprocessing.py`: so panel baseline/candidate và replay families.
- `benchmark_retrieval.py`: Recall@K/MRR trên source-audited Easy set.
- `benchmark_release_regression.py`: so release candidate với baseline.

## Source audit

- `audit_tables.py`: tìm row/table theo ticker/year/term.
- `audit_coordinate_consistency.py`: đối chiếu ZIP, CSV và source coordinates.
- `audit_column_qualifiers.py`, `audit_report_periods.py`,
  `audit_easy_periods.py`, `audit_easy_source_units.py`,
  `audit_selected_signs.py`: kiểm tra period/unit/sign/qualifier chuyên biệt.
- `audit_reasoning_sources.py`: Qwen source-only review; diagnostic, không build release.

## Build release an toàn

- `apply_manual_overrides.py`: áp correction có manifest lên build mới.
- `merge_submission_builds.py`, `select_submission_overlay.py`: ghép candidate
  mà không overwrite baseline.
- `build_source_repair_release.py`, `build_reviewed_variant.py`: release recipes
  có allowlist ID và audit trail.
- `release_audit.py`: acceptance gate cuối, repeated replay.
- `remap_table_refs_to_lines.py`: đổi table ordinal sang OCR start line có validation.

## Local model experiments

- `download_qwen35_checkpoint.py`: download pin/hash Qwen3.5-9B.
- `benchmark_reasoning_selector.py`, `benchmark_reasoning_arithmetic.py`:
  source-only experiments; không được dùng model approximation làm gold.

## Codabench

Các tool `codabench_*` có tác động external và cần credential/session hợp lệ.
Không upload chỉ vì local test pass; phải có full release audit và source-backed
change. Generated reports đặt trong `runs/`, không commit model/dataset/ZIP.
