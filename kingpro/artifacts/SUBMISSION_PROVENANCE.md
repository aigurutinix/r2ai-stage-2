# Bài nộp và tuân thủ no-hardcode — KINGPRO V297

## Canonical public-final

```text
version: V297
submission ID: 3747
file: sub_v297_scope2_a.zip
SHA-256: 90F7CC5E678EF80EB2741C5B9F162853AAF72CEA5B0ECC94BF80E2B7EAC36AAC
submission.json SHA-256: E19E4748DDAFFAD28112292EAE17E9296F85BBC99265C9D52EEB27E8F8539C85
```

Public metrics đã đo:

| Metric | Score |
| --- | ---: |
| Execution Accuracy | 0.7115 |
| Answer Accuracy | 0.7115 |
| Tables F2 / P / R / MRR@5 | 0.6120 / 0.5928 / 0.6261 / 0.6514 |
| Docs F2 / P / R / MRR@5 | 0.9618 / 0.9587 / 0.9678 / 0.9806 |

## Cơ chế tính trực tiếp

Mỗi record có `evidence` ánh xạ biến DataFrame tới CSV. `pandas_query` đọc ô từ
DataFrame và gán kết quả số vào `result` khi chạy. Sandbox tái chạy chương trình;
stored answer chỉ dùng để đối chiếu sau execution, không dùng làm operand.

Các pattern bị cấm và bị gate loại:

```python
result = 2.06
result = 1.03 * 2
result = df["answer"].iloc[0]
```

Exporter/release gate kiểm:

- query không rỗng, không import/I/O/network/dunder escape;
- code thực thi được trên CSV đóng gói;
- kết quả runtime khớp answer lưu;
- DataFrame variable bind đúng evidence CSV;
- `relevant_tables/docs` suy ra từ nguồn thật;
- archive không duplicate path, CRC/full-read PASS;
- final ZIP A/B byte-identical.

## Truy vết tới dữ liệu BTC

Manifest/source audit giữ table reference, tọa độ và raw token. Với legacy
program, causal perturbation thay từng cell và chỉ nhận dependency khi kết quả
Pandas thực sự đổi; frame còn phải bằng đúng một bảng vật lý thuộc
`relevant_tables`. Audit cuối: 1.012/1.012 câu, 7.299 cells, 0 mismatch và 0
citation-unbound.

## Quy tắc vận hành

Chỉ upload `sub_v297_scope2_a.zip`. Không repack/rename. Không upload hồ sơ
thuyết minh, source/data/model bundles, runtime cube hay lineage sidecar vào cổng
test.

