# FinWhale — ứng dụng web

Trợ lý phân tích báo cáo tài chính: nạp Excel/CSV, hỏi bằng tiếng Việt, nhận **số kèm trích nguồn**
và truy vấn Pandas chạy lại được.

Giới thiệu đầy đủ (tính năng, kiến trúc, kết quả trên bộ dữ liệu thi) nằm ở
[README gốc của repo](../README.md). File này chỉ ghi những gì cần để **chạy và kiểm** phần app.

## Chạy

```bash
pnpm install
pnpm dev            # http://localhost:3000 — cần Ollama chạy sẵn ở :11434
```

Cần `.env.local` với `ADMIN_USER`, `ADMIN_PASS` (đăng nhập) và tuỳ chọn `OLLAMA_MODEL`
(mặc định `hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M`). Xem `.env.example`.

## Năm agent, chỉ hai agent gọi LLM

`lib/financial/agents.ts` chạy tuần tự: **Hiểu câu hỏi** (LLM) → **Truy xuất & tính** (code) →
**Tự kiểm** (code) → **Chọn biểu đồ** (code) → **Nhận định** (LLM).

Hai lớp gác dễ nhầm, đừng lẫn:

| | file | gác cái gì |
|---|---|---|
| `validate.ts` | dữ liệu **VÀO** | đẳng thức kế toán TT200 (270 = 300 + 400…), sai số 1% |
| `audit.ts` | kết quả **RA** | miền giá trị hợp lệ suy từ chính câu hỏi, và lời văn có khớp con số không |

`audit.ts` không bao giờ đoán đáp án đúng — nó chỉ phát biểu điều suy được từ câu hỏi và định nghĩa
chỉ tiêu, nên không thể tự lừa mình. Nó **cảnh báo, không chặn và không sửa số**.

**Bàn giao Tự kiểm → Nhận định.** Mỗi vi phạm mang một trường `source` (`value` / `answer` /
`insight`), và agent Nhận định hành xử theo nó:

- vi phạm ở **con số** hoặc **câu trả lời** → **không gọi LLM** viết nhận định. Viết một đoạn phân
  tích tự tin cho con số vừa bị chứng minh là sai thì phần nghe có thẩm quyền nhất của màn hình sẽ
  mâu thuẫn với cảnh báo đỏ ngay phía trên nó.
- vi phạm ở chính **nhận định** (LLM viết ngược chiều số liệu) → **bỏ hẳn nhận định**, không hiển
  thị kèm cảnh báo. Một nhận định sai không có giá trị gì; để lại chỉ làm người đọc phân vân.

Đây là chỗ bàn giao duy nhất giữa hai agent, và nó truyền **dữ liệu có kiểu**, không phải văn bản
tóm tắt — xem lưu ý cuối file.

**Hợp đồng thứ tự.** Mỗi agent khai `requires` / `provides` (các trường của `Ctx`), và
`checkPipelineOrder` chạy **ngay lúc nạp module**: đổi thứ tự `PIPELINE` hay chèn agent vào sai chỗ
sẽ làm app hỏng ngay khi khởi động, kèm thông báo nói rõ agent nào thiếu trường nào. Trước đây một
sai sót như vậy chỉ biểu hiện thành `undefined` đọc được ở đâu đó xa nguyên nhân — đã gặp thật khi
`audit` được đặt trước `narrate` mà lại soi `result.insight`.

## Dữ liệu mẫu để trình diễn Auditor

```bash
node make-audit-sample.mjs      # → public/samples/01_unit_error_financial_statements.csv
```

File sinh ra có **hai công ty**: `CORP_LECH` (vốn chủ sở hữu ghi nhầm theo nghìn đồng ⇒ ROE
20.000% ⇒ Auditor **kêu**) và `CORP_CHUAN` (số liệu chuẩn ⇒ Auditor **im**). Nhóm đối chứng là
phần quan trọng: không có nó thì không chứng minh được Auditor biết phân biệt, chứ không phải
lúc nào cũng kêu.

File này **dàn dựng có chủ ý** để trình diễn, do chính dự án sinh ra 100%, **không phải báo cáo
của một doanh nghiệp có thật**. Nút *"Thử dữ liệu lệch đơn vị"* trong app nạp chính file này.

## Kiểm chất lượng

Kiểm tự động luôn chạy trước mỗi lần commit: `tsc --noEmit`, `eslint`, một bộ ca kiểm cho các luật
của Auditor (bao gồm phần lớn là ca **chống báo oan**), một bộ cho hợp đồng thứ tự agent (có ca bắt
bộ kiểm phải kêu, vì một bộ kiểm chưa từng kêu thì không có bằng chứng nó biết kêu), và một vòng
E2E Playwright đi trọn hành trình người dùng rồi soi lại từng ảnh chụp.

Các script này là **dụng cụ đo nội bộ, không nằm trong repo** — chúng phụ thuộc `.env.local`, dev
server đang chạy và dữ liệu cục bộ, nên có đưa lên cũng không chạy được ở máy khác.

## Lưu ý khi sửa code

- **Đừng suy trạng thái từ `plan.intent`.** Nhánh tính tăng trưởng chạy khi
  `intent === "yoy_growth"` **hoặc** `years.length >= 2`; suy từ `intent` đã gây bug thật hai lần.
  Nguồn sự thật là `AgentResult.comparison = {from, to, direction}` do `computeAnswer` đặt.
- **E2E phải chờ theo TRẠNG THÁI, không theo mili-giây.** Ngủ cứng từng làm hai ảnh chỉ khác nhau
  0,01% pixel mà vẫn là PNG hợp lệ đúng kích thước.
- **Đừng chèn tóm tắt do LLM sinh giữa các agent.** Bàn giao hiện tại là một object `Ctx` dùng
  chung với dữ liệu có kiểu — không mất mát, không có chỗ cho ảo giác. Quyết định này đã được đo:
  với model 4B, mỗi lần bàn giao bằng văn bản là một lần sinh JSON có thể hỏng, nên nhân số agent
  là nhân lỗi và nhân độ trễ (xem `WORKING-NOTES.md` §5.2). Thêm một tầng tóm tắt bằng ngôn ngữ
  tự nhiên giữa hai bước deterministic là mở đúng cái cửa mà cả kiến trúc dựng lên để đóng lại.
