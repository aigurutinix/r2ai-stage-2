# KINGPRO Question Audit Factory

Xưởng audit biến queue 100/150 hoặc toàn bộ 1.012 câu thành packet ưu tiên có
cache, runtime replay và bằng chứng vật lý. Xưởng **không tự đổi answer**.

## Chạy một lệnh

```powershell
$env:PYTHONPATH = "src"
python scripts/run_audit_factory_pipeline.py --target batch
```

Đầu ra chính:

- `build/v234_audit_factory_pipeline/fast/PRIORITY_QUEUE.json`
- `build/v234_audit_factory_pipeline/fast/DASHBOARD.md`
- `build/v234_audit_factory_pipeline/deep/DASHBOARD.md`
- `build/v234_audit_factory_pipeline/AGENT_SHARDS.json`
- `build/v234_audit_factory_pipeline/FACTORY_REPORT.md`

## Hai tầng

1. `fast`: quét 1.012 câu bằng fingerprint, AST/dataflow, source-key intent,
   named entity/project gate, dead evidence và exact source clusters.
2. `deep`: chạy string + official-typed runtime, compact physical audit hoặc
   positional fallback cho đúng queue ưu tiên.

Cache được khóa bằng nội dung row, SHA của evidence và SHA của các audit tools.
Nếu không có thay đổi, lần chạy sau chỉ đọc cache.

`AGENT_SHARDS.json` chia priority queue thành ba shard để ba agent audit song
song mà không trùng câu; agent chính giữ vai trò adjudicate và ghi ledger.

## Low-token agent mode

```powershell
python scripts/render_compact_audit_tasks.py --records <records.jsonl>
python scripts/collect_compact_audit_results.py build/v241_compact_agent_tasks
```

Task trung bình chỉ vài KB. Agent đọc task, ghi result JSON và chat chỉ trả
`DONE q<ID> <path>`. Với các kết quả `keep/high` đã được primary phê duyệt:

```powershell
python scripts/apply_approved_compact_results.py build/v241_compact_agent_tasks `
  --approved-ids 508,512
```

Apply tool vẫn fail-closed: cấm runtime/physical/intent/invariant flags, yêu cầu
explicit approved IDs và từ chối ghi đè verdict đã tồn tại.

## Fail-closed contract

- Không ghi ledger.
- Không sửa submission.
- Không suy answer từ cluster nếu semantic/source/runtime gates chưa đủ.
- `intent_source_mismatch`, `runtime_failure`, `physical_failure` và
  `positional_risk` luôn xếp trước cleanup hiệu suất.
- Mọi mutation cuối cùng vẫn phải có physical source proof và release gate.
