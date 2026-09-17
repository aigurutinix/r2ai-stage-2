# Model Card — lớp sinh Text-to-Pandas

## Vai trò trong hệ thống

Artifact nộp bài chứa chương trình và CSV evidence; khi chấm, query chạy offline
và không gọi model, API hay mạng. Trong sản phẩm live, model mở chỉ là nhánh tùy
chọn cho câu hỏi ngoài verified registry và grounded compiler, sau khi endpoint
được attested. Nếu chưa có attestation, nhánh này fail-closed thay vì đoán.

## Checkpoint đã thử nghiệm

| Checkpoint | Vai trò | License được model card công bố | Thời điểm phát hành |
| --- | --- | --- | --- |
| `Qwen/Qwen2.5-Coder-14B-Instruct` | model sinh code chính | Apache-2.0 | 11/2024 |
| `Qwen/Qwen3-14B` | ablation/review ở một số thử nghiệm | Apache-2.0 | 04/2025 |

Model live mặc định là `Qwen/Qwen2.5-Coder-14B-Instruct`: model card chính thức
công bố 14,7B tổng tham số, 13,1B non-embedding, Apache-2.0; Qwen công bố dòng
14B ngày 12/11/2024. Xem [model card](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct),
[release announcement](https://qwenlm.github.io/blog/qwen2.5-coder-family/) và
[runtime attestation](RUNTIME_ATTESTATION.md).

Các endpoint được phục vụ bằng vLLM qua giao thức HTTP tương thích OpenAI. “Tương thích OpenAI” chỉ mô tả protocol client; không có model đóng nào tham gia pipeline retrieval/Text-to-Pandas hoặc sinh kết quả lúc chấm. Trong quá trình phát triển, đội có sử dụng coding assistant như công cụ hỗ trợ viết, đọc và review mã nguồn; mọi thay đổi vẫn phải được kiểm chứng lại bằng dữ liệu BCTC, phép tính Pandas và các gate tái lập của repo. Coding assistant không phải dependency của hệ thống và không xuất hiện trong artifact runtime.

## Cấu hình suy luận điển hình

- temperature `0` cho một mẫu xác định;
- temperature khoảng `0.5–0.6` khi sinh nhiều mẫu;
- tối đa khoảng 1.200–1.500 output tokens;
- bỏ phiếu theo kết quả thực thi trong sai số tuyệt đối 0,01;
- tối đa 2–3 vòng self-repair khi chương trình lỗi.

Model nhận câu hỏi, schema và mẫu dữ liệu; kết quả model không được tin trực tiếp. Code phải vượt kiểm tra AST, chạy trong sandbox và tạo giá trị số từ DataFrame.

## Trạng thái fine-tune

Repo có script thử nghiệm LoRA/Unsloth nhưng candidate bàn giao không yêu cầu một checkpoint fine-tune riêng để chạy. Các query cuối là artifact tĩnh và được kiểm lại bằng Pandas.

## Đối chiếu giới hạn kích thước model

Tên hai checkpoint chứa `14B`; model card chính thức công bố tổng tham số xấp xỉ 14,7–14,8B và số tham số không tính embedding xấp xỉ 13,1–13,2B. Theo xác nhận trực tiếp của BTC do đội thi cung cấp, ngưỡng áp dụng là tổng tham số không quá 15B. Vì vậy cả hai checkpoint đều nằm trong giới hạn được BTC xác nhận.

Đội thi cần lưu ảnh chụp hoặc email chứa xác nhận 15B cùng hồ sơ nghiệm thu để chứng minh cách áp dụng điều khoản nếu văn bản công khai vẫn ghi `14B`.

## Hạn chế và rủi ro

- Model có thể chọn sai dòng, cột, năm, scope hoặc đơn vị dù code hợp cú pháp.
- Self-consistency chỉ đo đồng thuận, không chứng minh đúng.
- Dữ liệu OCR nhiễu làm tăng nguy cơ model bám nhầm ô.
- Không dùng output cho quyết định đầu tư nếu chưa đối chiếu báo cáo gốc.

## Bảo mật và vận hành

Không ghi API key vào code, log, hồ sơ hoặc ZIP. `.env` đã được ignore và không cần thiết để kiểm tra artifact. Endpoint inference chỉ cần mở trong giai đoạn phát triển; submission runtime phải độc lập mạng.
