# Kết quả thực nghiệm và cách diễn giải

## 1. Preprocessing

| Hạng mục | Kết quả |
|---|---:|
| Documents | 1.973 |
| Tables | 146.246 |
| Rows | 1.535.824 |
| Index build time | 584,849 s |
| Index size | 1.175.851.008 bytes |
| Financial panel cells | 52.066 |
| Panel build time | 63,918 s |
| Artifact audit | 0 errors, 0 warnings |

## 2. Retrieval benchmark

Benchmark gồm 124 câu Easy đã source-audit:

| Retriever | R@1 | R@5 | R@10 | R@20 | MRR |
|---|---:|---:|---:|---:|---:|
| Calibrated legacy row | .2661 | .5000 | .5726 | .6210 | .3728 |
| Semantic row (experimental) | .2419 | .4597 | .5403 | .5887 | .3373 |
| Table → legacy-row cascade | .2581 | .4919 | .5806 | .6210 | .3687 |
| Semantic table | .3548 | .8065 | .9194 | .9677 | .5534 |

Kết luận: giữ calibrated row làm ordering mặc định; semantic table dùng làm
shortlist/gate. Không bật semantic row global vì regression đã đo được.

## 3. Panel và runtime

Panel v2 khác historical panel ở 3.546 keys (3.472 numeric values và 1.756
source selections, có giao nhau). Replay Hard 159 câu không đổi answer; năm
Template family chỉ đổi Q632 có source-audit, không có lỗi. So với release
baseline, 594/594 Hard+Template được tái tạo đúng.

Caching issuer/year/scope candidates, table preview và `TableAnalyzer` giảm
Q579 từ 14,55 giây xuống 0,29 giây cold và 0,26 giây warm, giữ nguyên answer.

## 4. Test và release audit

Mốc VN67: 66 tests pass; replay 651 deterministic questions chỉ có đúng 3 thay
đổi dự kiến và 0 lỗi; full ZIP audit chạy 1.012/1.012 CSV query, không thiếu
reference hoặc orphan CSV. Các mốc VN68/VN70 lần lượt 69 và 75 tests pass tại
thời điểm ghi nhận. Số test có thể tăng theo repository; luôn chạy suite hiện
tại thay vì coi các con số lịch sử là acceptance criterion cố định.

## 5. Official leaderboard history

| Release | Thay đổi chính | Execution/Answer Accuracy | Quyết định |
|---|---|---:|---|
| VN53/VN67 | protected/source-repair baseline | 0,6937 | baseline |
| VN68 | signed temporal changes | 0,6996 | promote |
| VN69 | broader ordered comparison | 0,6937 | reject |
| VN70 | literal ratio + gross-sales basis | **0,7016** | best recorded |

VN70 tăng 0,0079 absolute so với 0,6937. Retrieval metrics không đổi trong các
release này, vì thay đổi nằm ở phép tính/semantic contract. Kết quả official là
aggregate; không được suy ngược correctness của từng câu. Local replay chứng
minh expression nhất quán với submitted answer, không chứng minh hidden gold.

Nguồn chi tiết và SHA-256 release nằm trong
[`reasoning_upgrade_20260827.md`](reasoning_upgrade_20260827.md); preprocessing
benchmark nằm trong [`optimization.md`](optimization.md).
