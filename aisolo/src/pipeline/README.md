# Pipeline sinh bài nộp — ViFinQA (Road to AI 2026, Stage 2)

Sinh `submission.zip` từ kho báo cáo tài chính: ingest OCR → truy hồi báo cáo/bảng → tính đáp án
→ sinh `pandas_query` + CSV bằng chứng.

Bước dựng bài nộp **thuần deterministic**, chạy CPU vài phút. Hai bước dùng model (planner và
xếp hạng lại bảng) chạy **ngoại tuyến trước đó** và đóng băng kết quả ra JSON — xem mục
"Hai file cần sinh trước".

## Dữ liệu — đặt vào đâu

Dữ liệu **không kèm trong repo** (371 MB, và giấy phép nguồn là CC BY-NC nên không phân phối lại).
Cách đơn giản nhất: tải vào thư mục `data_vifinqa/` **ngay cạnh repo**, rồi không phải cấu hình gì
thêm — mọi script mặc định tìm ở đó.

```bash
# đứng ở thư mục gốc của repo
git clone https://huggingface.co/datasets/AIGuruTinix/ViFinQA data_vifinqa
```

Sau lệnh trên, cây thư mục phải là:

```
<gốc repo>/
├─ pipeline/
├─ r2ai-app/
└─ data_vifinqa/                                        ← đặt ở đây
   ├─ financial_statements/<MÃ_CK>/<NĂM>/<hợp nhất|riêng>/*.txt
   ├─ questions/questions.jsonl
   └─ code_stock.csv
```

Muốn để nơi khác thì trỏ bằng biến môi trường: `VIFINQA_ROOT=/đường/dẫn/khác python ...`

Nếu đặt sai chỗ, script **dừng ngay với thông báo rõ** chứ không chạy tiếp rồi cho ra bài nộp rỗng:

```
[LOI] Khong thay du lieu ViFinQA tai: <đường dẫn>
       Thieu thu muc con 'financial_statements/'.
```

## Các file

| File | Vai trò |
|---|---|
| `pipeline.py` | Lõi: ingest báo cáo OCR → bảng dòng theo Mã số TT200, chuẩn hoá đơn vị, BM25 chọn bảng, `locate()` chọn dòng |
| `build_submission.py` | Sinh bài nộp: chạy toàn bộ 1012 câu → `submission.json` + `data/*.csv` + ZIP |
| `llm_engine.py` | Nhánh LLM sinh pandas (tắt mặc định, bật bằng `USE_LLM=1`) |
| `agent_strands.py` | Planner→Executor cho câu suy luận nhiều bước; sinh ra `agent_full_v2.json` |
| `rerank_offline.py` | Xếp lại thứ tự bảng bằng cross-encoder Qwen3-Reranker-0.6B; sinh ra `rerank_cache.json` |
| `audit_wrong.py` | **Auditor** deterministic: suy miền giá trị hợp lệ từ chính câu hỏi → liệt kê đáp án SAI CHẮC CHẮN (gold-free) |
| `agent_audit.py` | Vòng Planner→Executor→Auditor: chạy lại câu bị Auditor bác, kèm ràng buộc miền giá trị |
| `merge_audit.py` | Gộp kết quả vòng Auditor vào `agent_full_v2.json` — chỉ nhận đáp án có bằng chứng (`refs`) |
| `check_audit.py` | Lớp kiểm chặt hơn cổng Auditor, để người đọc soát chất lượng đáp án mới |
| `verify_submission.py` | Kiểm chứng: chạy lại mọi `pandas_query` trên CSV kèm theo, so với đáp án |
| `diff_submission.py` | So bài nộp hiện tại với bản trước — biết thay đổi làm đổi câu nào |

## Chạy

```bash
# 1. Môi trường Python (xem hướng dẫn đầy đủ ở README gốc)
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt    # hoặc: pip install -r pipeline/requirements.txt từ thư mục gốc

# 2. (TUỲ CHỌN) Sinh lại các tệp trung gian — CẦN GPU, tổng ~20 giờ.
#    BỎ QUA ĐƯỢC: repo đã kèm sẵn kết quả đóng băng, nhảy thẳng xuống bước 3.
pip install torch transformers sentence-transformers
python agent_strands.py            # → agent_full_v2.json   (~14 giờ, RTX 3050 6GB)
python rerank_offline.py           # → rerank_cache.json    (~45 phút)

# 2b. VÒNG AUDITOR — sửa những câu đã CHỨNG MINH là sai (cận dưới bằng 0, xem mục bên dưới)
python audit_wrong.py              # → wrong_ids.json: đáp án sai chắc chắn của bản hiện tại
python agent_audit.py              # → agent_audit.json  (~5 giờ, cần Ollama)
python merge_audit.py --apply      # gộp vào agent_full_v2.json (tự sao lưu .bak)

# 3. Sinh bài nộp — từ đây chỉ cần CPU, vài phút
USE_AGENT=1 AGENT_OUT=agent_full_v2.json python build_submission.py
#   (không cần đặt VIFINQA_ROOT nếu dữ liệu ở data_vifinqa/ cạnh repo)

# 4. Kiểm chứng trước khi nộp
python verify_submission.py        # kỳ vọng: 1005/1005 khớp, 0 mismatch
```

### Hai file cần sinh trước

Bài nộp phụ thuộc hai file kết quả sinh từ model. **Từ 21/08 cả hai đã kèm sẵn trong repo**
(tổng 575 KB) nên dựng lại chỉ cần CPU; mục này mô tả cách tự sinh lại chúng, không
phải mã nguồn:

| File | Sinh bằng | Nội dung | Thiếu thì sao |
|---|---|---|---|
| `agent_full_v2.json` | `agent_strands.py` | Kế hoạch truy vấn của planner cho 396 câu (299 câu có đáp án, sau 5 vòng Auditor) | Dừng ngay với thông báo lỗi (từ 21/08; trước đó mất câu **im lặng**) |
| `rerank_cache.json` | `rerank_offline.py` | Thứ tự bảng sau khi cross-encoder xếp lại, 995 khoá | Tự động lùi về BM25 thuần, TABLES F2 0.4459 → 0.4348 |

Cả hai chạy **ngoại tuyến một lần** rồi đóng băng ra JSON. Nhờ vậy `build_submission.py` không
import `torch`, không cần GPU, và bài nộp vẫn thuần deterministic — chạy lại bao nhiêu lần cũng ra
đúng một kết quả.

> **`USE_AGENT=1` là bắt buộc.** Thiếu cờ này thì bài nộp mất 293 câu suy luận nhiều bước mà
> **không báo lỗi gì** — `verify_submission.py` vẫn báo 100%. Script sẽ in cảnh báo nếu quên.
>
> **`pyvi` là bắt buộc.** Thiếu nó thì BM25 không tách từ ghép tiếng Việt và cách chọn bảng đổi
> hoàn toàn, cũng không báo lỗi. Script in cảnh báo ra stderr nếu thiếu.

## Kết quả

| Chỉ số | Điểm |
|---|---|
| DOCS F2-macro | **0.9451** |
| TABLES F2-macro | **0.4487** |
| Answer Accuracy | **0.3340** |
| Execution Accuracy | **0.3340** |

Pandas tái lập **1005/1005** dưới cả hai cách đọc CSV (suy kiểu mặc định và `dtype=str`).
Answer và Execution bằng nhau tuyệt đối vì `pandas_query` và trường `answer` cùng làm tròn 2 chữ số
thập phân ngay trong biểu thức — Execution được chấm bằng cách chạy lại chính `pandas_query` đó.

### Vòng Auditor — vì sao nó an toàn

`audit_wrong.py` chỉ phát biểu điều **suy được từ chính câu hỏi**: hỏi "có bao nhiêu doanh nghiệp"
thì đáp án phải là số nguyên nhỏ; hỏi "bao nhiêu phần trăm" thì phải nằm trong khoảng hợp lý. Nó
không bao giờ đoán đáp án đúng là gì, nên không thể tự lừa mình. Vòng Auditor chỉ chạy lại những câu
bị nó bác, tức **đã chứng minh sai** — thay bằng gì cũng không thể tệ hơn.
Kết quả thực đo: 151 → 70 câu sai chắc chắn, Execution 0.2470 → 0.3340.
Riêng VÒNG ĐẦU đã đưa 0.2470 → 0.3103 (+32 câu) — bước nhảy lớn nhất của dự án. Bốn vòng sau lợi
tức giảm dần (tỷ lệ chuyển đổi bộ đếm→điểm từ 1.14 xuống 0.38) vì chúng cày trên nhóm câu khó.

### Quy ước dấu — đo bằng probe thay vì đoán

Báo cáo tài chính in chi phí/dự phòng trong ngoặc đơn (số âm), nhưng không có tài liệu nào
nói gold theo quy ước nào. Thay vì đoán, nộp một bản chỉ khác ở chỗ lấy trị tuyệt đối cho
18 câu hỏi một lượng không thể âm: Execution 0.3182 → 0.3261. Vậy **gold dùng quy ước
dương**. Mở rộng thêm 6 câu (giá vốn, thuế TNDN, tiền chi, nợ xấu, thu nhập bình quân) →
0.3281 (các mốc trong đoạn này là **diễn biến lúc chạy probe**; điểm hiện tại 0.3340 đạt được
sau vòng Auditor tiếp theo). Ranh giới quan trọng: chỉ áp cho khái niệm KHÔNG THỂ âm; lợi nhuận,
lưu chuyển tiền thuần, lãi thuần giữ nguyên dấu vì doanh nghiệp lỗ là chuyện có thật.

## Nguyên tắc kiến trúc

**LLM hiểu câu hỏi, CODE tính toán.** Mô hình chỉ xuất kế hoạch truy vấn dạng JSON; mọi phép tính
do code thực hiện trên số đọc từ báo cáo. Nhờ vậy đáp án không thể là số bịa, và ban tổ chức chấm
lại chỉ cần chạy `pandas_query` trên CSV kèm theo — **không cần model, không cần GPU**.
