# Khai báo nguồn dữ liệu và mô hình

Lập theo yêu cầu của BTC: *"Người tham gia được phép sử dụng dữ liệu từ các nguồn bên ngoài,
tuy nhiên phải trích dẫn rõ ràng và cung cấp đầy đủ thông tin về nguồn gốc dữ liệu để Ban tổ
chức có thể kiểm tra, xác minh khi cần thiết."*

Cập nhật: 02/08/2026.

## 1. Dữ liệu chính thức của cuộc thi

| Nguồn | Định danh | Dùng làm gì |
|---|---|---|
| HuggingFace Datasets | `AIGuruTinix/ViFinQA` (commit `0450088`) | Toàn bộ corpus BCTC (1.973 file `.txt`), `code_stock.csv`, `questions.jsonl` |

Tải bằng `scripts/download_corpus.py`. Không sửa đổi dữ liệu gốc; mọi thứ phái sinh nằm trong
`data/processed/` và tái tạo được từ `data/raw/` bằng `scripts/build_tables.py`.

## 2. Mã nguồn bên ngoài

| Nguồn | Định danh | Dùng làm gì |
|---|---|---|
| GitHub | `DSKT-NOWJ/ViFinQA` (repo công khai của chính BTC) | **Chỉ đọc để rút quy ước đánh giá**, không sao chép mã vào bài nộp |

Cụ thể, những quy ước sau được rút từ repo này và cài đặt lại độc lập trong `src/vifin/`:
`ANSWER_ABS_TOL = 1e-2` với `rel_tol = 0`; làm tròn 2 chữ số thập phân một lần ở bước cuối;
chênh lệch không nêu chiều lấy trị tuyệt đối; ngưỡng bảng eligible `n_rows >= 3` và
`>= 6` ô số; cách sandbox nạp CSV (`dtype=str, keep_default_na=False`); công thức F2.

Repo được clone vào `reference/ViFinQA/` và **không** đóng gói vào bài nộp.

## 3. Thư viện phần mềm

`pandas`, `numpy`, `pyarrow`, `scipy`, `huggingface_hub`. Đều là thư viện mã nguồn mở phổ biến.
Không dùng thư viện nào chứa tri thức về doanh nghiệp Việt Nam.

## 4. Tri thức bên ngoài đưa vào hệ thống

Đây là phần duy nhất hệ thống dùng thông tin **không** suy được từ dữ liệu BTC cấp.
Toàn bộ nằm trong hằng số `ALIASES_EXTERNAL` tại `src/vifin/questions.py`, gồm **5 mục**:

| Cụm trong câu hỏi | Mã CK | Tên BTC cấp trong `code_stock.csv` | Số câu bị ảnh hưởng |
|---|---|---|---|
| Vinamilk | VNM | CTCP Sữa Việt Nam | 3 |
| MBBank | MBB | Ngân hàng TMCP Quân đội | 2 |
| Eximbank | EIB | Ngân hàng TMCP Xuất nhập khẩu Việt Nam | 2 |
| Đạm Phú Mỹ | DPM | Tổng công ty Phân bón và Hóa chất Dầu khí - CTCP | 1 |
| Đạm Cà Mau | DCM | CTCP - Tổng công ty Phân bón Dầu khí Cà Mau | 1 |

Nguồn: tên thương hiệu phổ thông của doanh nghiệp niêm yết Việt Nam. Cần thiết vì một số câu
hỏi gọi doanh nghiệp bằng thương hiệu thay vì tên pháp nhân trong `code_stock.csv`.

Ba mươi bí danh còn lại (`ALIASES_FROM_CORPUS`) **không** phải dữ liệu ngoài: mỗi bí danh là
một cụm con của chính tên công ty trong `code_stock.csv`, chỉ để khớp cách gọi rút gọn
("Hoà Phát" ↔ "CTCP Tập đoàn Hòa Phát").

### Bí danh đã chủ động loại bỏ

Bộ dữ liệu **cố tình đổi tên** một số doanh nghiệp, nên tri thức đời thực có thể sai:

- `"Đất Xanh" → DXG`: **loại bỏ vì sai.** BTC đặt DXG = "CTCP Bluemarq Group", trong khi cụm
  "Đất Xanh" nằm trong tên chính thức của **DXS** ("CTCP Dịch vụ Bất động sản Đất Xanh").
  Bí danh này từng khớp nhầm vào 11 câu hỏi vốn hỏi về DXS.
- `"Petrolimex" → PLX`: loại bỏ vì gây nhiễu ("Ngân hàng TMCP Xăng dầu Petrolimex" là pháp
  nhân khác).
- `"Sabeco"`, `"Novaland"`: loại bỏ vì không câu hỏi nào dùng.

## 5. Mô hình ngôn ngữ

### Đã dùng tới bài nộp v7: KHÔNG dùng mô hình nào

Toàn bộ pipeline là quy tắc tường minh: parser HTML, lọc theo thực thể, xếp hạng BM25,
trích số bằng heuristic. Kiểm chứng được: máy local không cài `torch`, `transformers`,
`sentence-transformers`, và pipeline không gọi API ra ngoài lúc chạy.

### Từ v8: cross-encoder xếp hạng lại

| Mục | Chi tiết |
|---|---|
| Model | **`BAAI/bge-reranker-v2-m3`** |
| Cách lấy | HuggingFace Hub: `AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")` |
| Kích thước | ~568 triệu tham số (≪ 14B) |
| Trọng số | mở, giấy phép Apache-2.0 |
| Phát hành | 2024 (trước mốc 01/06/2026) |
| Dùng làm gì | Chỉ **xếp hạng lại** top-20 bảng ứng viên. Không sinh văn bản, không sinh đáp án. |
| Chạy ở đâu | Kaggle Notebook, GPU T4 — `notebooks/rerank_kaggle.ipynb` |

Chính BTC dùng model này trong `configs/retrieval/bge_m3_rerank.yaml` của repo tham chiếu.

Đáp án và `pandas_query` **vẫn do mã tất định sinh ra** từ bảng được chọn; reranker chỉ
ảnh hưởng tới việc *chọn bảng nào*, không đụng tới con số.

### Từ v12: đổi sang `Qwen/Qwen3-8B-AWQ`

| Mục | Chi tiết |
|---|---|
| Model | **`Qwen/Qwen3-8B-AWQ`** |
| Cách lấy | HuggingFace Hub, kho chính chủ của Qwen. Cần `pip install gptqmodel` như bản 14B. |
| Kích thước | 8,2B tham số (6,95B không kể embedding) — dưới trần 14B |
| Trọng số | mở, giấy phép **Apache-2.0** |
| Phát hành | 2025 (báo cáo kỹ thuật arXiv:2505.09388, 14/05/2025) — trước mốc 01/06/2026 |
| Lượng tử hoá | AWQ 4-bit, checkpoint nén sẵn trên Hub |
| Sinh | `do_sample=False` — tất định, bài nộp tái lập được |
| Chạy ở đâu | Kaggle Notebook, GPU T4 — `notebooks/generate_pandas_awq.ipynb` |

**Vì sao đổi từ 14B xuống 8B** (quyết định dựa trên đo đạc, không phải cảm tính):

Trên T4 15,6 GB, bản 14B nặng 10,3 GB nên chỉ còn **5,3 GB** — không đủ cho prompt 8
bảng (dài nhất 7.980 token) cộng vòng tự sửa lỗi. Bằng chứng quyết định: bật vòng sửa
lỗi trên 14B làm **OOM nhảy từ 18,5% lên 87,5%** và tốc độ tụt từ 1,4 xuống 0,4
câu/phút, đổi lại chỉ cứu được 1 câu trên 120.

Bản 8B nặng ~6,5 GB nên còn ~9,1 GB, tức **gấp 1,7 lần chỗ trống**. Đủ cho cả 8 bảng
lẫn vòng sửa lỗi. Ngân sách token giờ **tự tính từ VRAM thật** lúc chạy
(`compute_safe_tokens()`), nên đổi model không cần chỉnh tay.

Bản 14B vẫn giữ trong code ở hằng số `MODEL_14B` để đối chứng khi cần.

### Từ v10 tới v11: `Qwen/Qwen3-14B-AWQ`

| Mục | Chi tiết |
|---|---|
| Model | **`Qwen/Qwen3-14B-AWQ`** |
| Cách lấy | HuggingFace Hub: `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-14B-AWQ", device_map={"": 0})`, cần `pip install gptqmodel` |
| Kích thước | 14B — đúng trần cho phép |
| Trọng số | mở, giấy phép Apache-2.0 |
| Phát hành | 2025 (trước mốc 01/06/2026) |
| Lượng tử hoá | AWQ 4-bit, **checkpoint đã nén sẵn trên Hub** (không tự nén lúc nạp). Footprint đo được 10,0 GB. Kernel `AwqExllamaV2Linear`. |
| Sinh | `do_sample=False` — **tất định**, nên bài nộp tái lập được |
| Dùng làm gì | Sinh `pandas_query` cho **411/1012 câu** mà pipeline quy tắc gần như chắc sai (câu nhiều bước, tỷ lệ, cực trị nhiều kỳ). 601 câu còn lại vẫn do quy tắc tất định xử lý. |
| Chạy ở đâu | Kaggle Notebook, GPU T4 — `notebooks/generate_pandas_kaggle.ipynb` |

Chính BTC dùng model này trong `reference/ViFinQA/configs/answering/` của họ.

Prompt hệ thống dùng lại nguyên văn `reference/ViFinQA/prompts/answering/program_system.txt`
của BTC, bổ sung một mục về đơn vị đáp án theo xác nhận của họ ngày 02/08/2026.

**Ghi chú vận hành**: chạy hai tiến trình song song trên 2×T4, mỗi tiến trình chiếm trọn
một GPU (`device_map={"": 0}` + `CUDA_VISIBLE_DEVICES`) và xử lý một nửa số câu — nhanh gần
gấp đôi so với `device_map="auto"` vốn chỉ chia lớp model.

**Vì sao dùng checkpoint AWQ có sẵn thay vì tự nén bằng bitsandbytes**: đã thử
`Qwen/Qwen3-14B` với `BitsAndBytesConfig(load_in_4bit=True)` trên T4 với GPU hoàn toàn sạch
(15,4 GB trống) và vẫn tràn bộ nhớ ở 48% quá trình nạp, tại mốc 14,18 GB. Lượng tử hoá CÓ
hoạt động (kiểm chứng bằng `notebooks/check_bitsandbytes.ipynb`: các lớp
`q_proj/k_proj/v_proj` có kiểu `Params4bit`), nhưng transformers nạp trọng số **fp16 lên GPU
trước rồi mới nén**, nên đỉnh bộ nhớ lúc nạp bằng cỡ model fp16 (28 GB) chứ không phải 10 GB
của trạng thái cuối.

Checkpoint AWQ giải quyết triệt để: file trên Hub **vốn đã ở 4-bit**, nạp thẳng vào GPU,
không bao giờ đi qua fp16. Đo thực tế trên Kaggle T4: footprint **10,0 GB**, VRAM dùng
10,3 GB trên tổng 15,6 GB.

Lưu ý cài đặt: `transformers` đọc checkpoint AWQ qua **`gptqmodel`**, không phải `autoawq`
(đã thử `autoawq` và vẫn báo thiếu `gptqmodel`).

**Cổng kiểm soát**: mọi code do LLM sinh đều được thực thi lại trong bản sao sandbox trước
khi chấp nhận (`scripts/apply_llm.py`). Code không chạy được, trả về không phải số hữu hạn,
có `import`, hay cho kết quả phi lý (hỏi % mà trả > 1000; hỏi "năm nào" mà trả năm không có
trong câu hỏi) đều bị bác và quay về đáp án của quy tắc. Nghĩa là **LLM không thể làm điểm
tụt so với bản thuần quy tắc**.

### Ràng buộc cho mọi mô hình bổ sung về sau

Bắt buộc: **trọng số mở, ≤ 14B, phát hành trước 01/06/2026 (giờ Việt Nam)**, và ghi rõ cách
lấy mô hình vào bài báo. Tuyệt đối không dùng mô hình đóng (GPT-4o, Gemini, Claude...) ở
bất kỳ khâu nào của hệ thống nộp bài.

## 6. Công cụ hỗ trợ lập trình

Mã nguồn trong `src/` và `scripts/` được viết với sự hỗ trợ của trợ lý lập trình AI
(Claude Code, Anthropic). Công cụ này đóng vai trò **soạn thảo mã**, tương tự IDE hay
trình gợi ý mã; nó **không phải một thành phần của hệ thống nộp bài** và không tham gia
sinh ra bất kỳ đáp án nào trong `submission.json` — mọi con số đều do mã tất định trong
repo tính ra từ dữ liệu BTC cấp.

Nếu BTC coi việc này cần khai báo hoặc hạn chế thêm, chúng tôi sẵn sàng tuân thủ.
