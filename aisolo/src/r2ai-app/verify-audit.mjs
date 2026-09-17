/**
 * Kiểm Auditor (lib/financial/audit.ts) — chạy luật thật trên các ca dựng sẵn.
 *
 * Vì sao cần: Auditor chỉ có giá trị nếu nó KÍCH HOẠT đúng ca sai và IM LẶNG ở ca đúng. Bài học
 * từ pipeline: luật `group_count` từng nằm im không bao giờ chạy vì thiếu một dòng đăng ký, mà
 * test gọi thẳng hàm nên vẫn xanh. Ở đây test cả hai chiều: phải bắt, và phải không báo oan.
 *
 * Chạy:  npx tsc lib/financial/{audit,registry,types}.ts --outDir <dir> --module commonjs \
 *          --target es2020 --moduleResolution node --skipLibCheck
 *        node verify-audit.mjs <dir>
 */
import { createRequire } from "node:module";
import path from "node:path";

const dir = process.argv[2];
if (!dir) {
  console.error("Thiếu tham số: node verify-audit.mjs <thư-mục-đã-compile>");
  process.exit(2);
}
const require_ = createRequire(import.meta.url);
const { auditAnswer, auditMessage } = require_(path.resolve(dir, "audit.js"));

/** TidyTable tối thiểu: VNM có Tổng tài sản (Mã 270) = 1.000 tỷ ở kỳ 2024. */
const tidy = {
  tickers: ["VNM"],
  periods: ["2024"],
  unit: "VND",
  sheetName: "BCTC",
  byTicker: { VNM: { 270: { maSo: "270", name: "Tổng tài sản", values: { 2024: 1_000e9 }, rowIndex: 1 } } },
  cleaningNotes: [],
  meta: { maSoKey: "Mã số", periodKeys: { 2024: "2024" } },
};

/** Cùng công ty nhưng DỮ LIỆU NGUỒN ĐÃ LỆCH đẳng thức (270 ≠ 300 + 400). Dùng để kiểm rằng luật
 *  "khoản mục > tổng tài sản" biết NHƯỜNG cho cảnh báo dữ liệu thay vì đổ lỗi cho đáp án —
 *  đo trên dữ liệu mẫu thật: 875/987 công ty (88,7%) rơi vào tình huống này. */
const tidyLech = {
  ...tidy,
  byTicker: {
    VNM: {
      270: { maSo: "270", name: "Tổng tài sản", values: { 2024: 1_000e9 }, rowIndex: 1 },
      300: { maSo: "300", name: "Nợ phải trả", values: { 2024: 900e9 }, rowIndex: 2 },
      400: { maSo: "400", name: "Vốn chủ sở hữu", values: { 2024: 500e9 }, rowIndex: 3 },
    },
  },
};

const cite = [{ ticker: "VNM", maSo: "140", itemName: "Hàng tồn kho", period: "2024", sheet: "BCTC" }];
const mk = (o) => ({
  ok: true, answer: "", valueLabel: "", unit: "VND", citations: cite, pandasCode: "",
  plan: { intent: "direct_retrieval", metric: "inventory", years: ["2024"], tickers: ["VNM"] },
  provider: "qwen3.5-4b", ...o,
});

const CASES = [
  // --- PHẢI BẮT ---
  ["bắt: hàng tồn kho > tổng tài sản", mk({ value: 1_500e9 }), true],
  ["bắt: hàng tồn kho âm", mk({ value: -5e9 }), true],
  ["bắt: biên lợi nhuận 250.000%", mk({ value: 250_000, unit: "%", plan: { intent: "direct_retrieval", metric: "gross_margin", years: ["2024"], tickers: ["VNM"] } }), true],
  ["bắt: hệ số thanh toán âm", mk({ value: -1.2, unit: "x", plan: { intent: "direct_retrieval", metric: "current_ratio", years: ["2024"], tickers: ["VNM"] } }), true],

  // --- PHẢI IM LẶNG (không báo oan) ---
  ["im: hàng tồn kho 300 tỷ < tổng tài sản", mk({ value: 300e9 }), false],
  ["im: chính tổng tài sản (không tự so với mình)", mk({ value: 1_000e9, plan: { intent: "direct_retrieval", metric: "total_assets", years: ["2024"], tickers: ["VNM"] } }), false],
  ["im: lợi nhuận sau thuế ÂM — doanh nghiệp lỗ là có thật", mk({ value: -80e9, plan: { intent: "direct_retrieval", metric: "net_income", years: ["2024"], tickers: ["VNM"] } }), false],
  ["im: vốn chủ sở hữu ÂM — công ty âm vốn là có thật", mk({ value: -20e9, plan: { intent: "direct_retrieval", metric: "equity", years: ["2024"], tickers: ["VNM"] } }), false],
  ["im: so sánh 2 công ty ra HIỆU âm", mk({ value: -400e9, plan: { intent: "multi_company", metric: "inventory", years: ["2024"], tickers: ["VNM", "HPG"] } }), false],
  ["im: tăng trưởng −120% (lãi chuyển thành lỗ)", mk({ value: -120, unit: "%", plan: { intent: "yoy_growth", metric: "net_income", years: ["2023", "2024"], tickers: ["VNM"] } }), false],
  ["im: ROE 45%", mk({ value: 45, unit: "%", plan: { intent: "direct_retrieval", metric: "roe", years: ["2024"], tickers: ["VNM"] } }), false],
  ["im: kết quả lỗi (ok=false) thì không kiểm", mk({ ok: false, value: 9e99 }), false],
  ["im: không có số liệu tổng tài sản để đối chiếu", { ...mk({ value: 9_000e9 }), citations: [{ ...cite[0], ticker: "XXX" }] }, false],
  // Dữ liệu nguồn ĐÃ lệch đẳng thức ⇒ vượt trần là lỗi của FILE, không phải của đáp án.
  // `validate.ts` đã báo rồi; Auditor phải im để không đổ lỗi sai đối tượng.
  ["im: vượt tổng tài sản NHƯNG dữ liệu nguồn đã lệch đẳng thức (nhường validate.ts)",
    mk({ value: 1_500e9 }), false, tidyLech],
];

// --- Luật 5: LỜI VĂN phải khớp CON SỐ (bug thật người dùng gặp: "2024 đến 2023" + nói "giảm"
//     trong khi số liệu tăng, vì đọc 2023 trước) ---
const cmp = (o) => mk({
  value: 12.5, unit: "%",
  comparison: { from: "2023", to: "2024", direction: "up" },
  plan: { intent: "direct_retrieval", metric: "net_revenue", years: ["2023", "2024"], tickers: ["VNM"] },
  ...o,
});
CASES.push(
  // PHẢI BẮT
  ["bắt: số liệu TĂNG nhưng nhận định viết giảm",
    cmp({ answer: "Doanh thu thuần của VNM năm 2024 tăng 12,5% so với năm 2023.", insight: "Doanh thu suy giảm cho thấy sức mua yếu đi." }), true],
  ["bắt: khoảng năm viết ngược (2024 đến 2023)",
    cmp({ answer: "Doanh thu thuần của VNM năm 2024 tăng 12,5% so với năm 2023.", insight: "Đà tăng từ 2024 đến 2023 phản ánh nhu cầu phục hồi." }), true],
  ["bắt: câu hai vế, chiều ĐẦU TIÊN sai (luật cũ bỏ lọt)",
    cmp({ answer: "Doanh thu thuần của VNM năm 2024 tăng 12,5% so với năm 2023.", insight: "Doanh thu giảm nhẹ dù biên lợi nhuận tăng." }), true],
  ["bắt: số liệu GIẢM nhưng nhận định viết tăng",
    cmp({ value: -8.2, comparison: { from: "2023", to: "2024", direction: "down" }, answer: "Doanh thu thuần của VNM năm 2024 giảm 8,2% so với năm 2023.", insight: "Doanh thu cải thiện rõ rệt." }), true],

  // PHẢI IM LẶNG
  ["im: câu trả lời và nhận định đều đúng chiều",
    cmp({ answer: "Doanh thu thuần của VNM năm 2024 tăng 12,5% so với năm 2023.", insight: "Đà tăng cho thấy nhu cầu tiêu dùng phục hồi tốt." }), false],
  ["im: 'năm 2024 ... so với năm 2023' là cách viết ĐÚNG, không phải khoảng ngược",
    cmp({ answer: "Doanh thu thuần của VNM năm 2024 tăng 12,5% so với năm 2023.", insight: "Mức tăng từ 2023 sang 2024 là tín hiệu tích cực." }), false],
  ["im: câu hai vế, chiều đầu tiên ĐÚNG",
    cmp({ answer: "Doanh thu thuần của VNM năm 2024 tăng 12,5% so với năm 2023.", insight: "Doanh thu tăng dù biên lợi nhuận giảm nhẹ." }), false],
  ["im: chưa có nhận định (narrate chưa chạy hoặc trả null)",
    cmp({ answer: "Doanh thu thuần của VNM năm 2024 tăng 12,5% so với năm 2023.", insight: null }), false],
  ["im: nhận định không nêu chiều nào",
    cmp({ answer: "Doanh thu thuần của VNM năm 2024 tăng 12,5% so với năm 2023.", insight: "Kết quả này cần đặt cạnh mặt bằng chung của ngành sữa." }), false],
  // Hồi quy: tăng trưởng ÂM của một chỉ tiêu "không thể âm" KHÔNG được gắn cờ luật dấu.
  // Trước khi sửa `isDelta`, ca này báo oan "Doanh thu thuần không thể âm" cho con số −8,2%.
  ["im: tăng trưởng −8,2% của doanh thu (intent=direct_retrieval) không phải giá trị âm",
    cmp({ value: -8.2, comparison: { from: "2023", to: "2024", direction: "down" }, answer: "Doanh thu thuần của VNM năm 2024 giảm 8,2% so với năm 2023.", insight: "Sức mua yếu đi rõ rệt." }), false],
);

let pass = 0, fail = 0;
for (const [name, result, shouldFlag, tidyCase] of CASES) {
  const vios = auditAnswer(result, tidyCase ?? tidy);
  const flagged = vios.length > 0;
  const ok = flagged === shouldFlag;
  if (ok) pass++; else fail++;
  console.log(`${ok ? "  OK  " : "  FAIL"} ${name}${flagged ? `  -> ${vios.map((v) => v.rule).join(", ")}` : ""}`);
  if (ok && flagged && !auditMessage(vios)) {
    console.log("  FAIL   ^ có vi phạm nhưng auditMessage trả null");
    fail++; pass--;
  }
}
console.log(`\n${pass}/${pass + fail} ca luật đạt`);

// ---- Bàn giao Auditor -> Nhận định ----
// agents.ts quyết định dựa trên `source`: vi phạm ở value/answer => KHÔNG gọi LLM viết nhận định;
// vi phạm ở insight => BỎ HẲN nhận định. Nếu phân loại nguồn sai thì quyết định đó sai theo,
// nên phải kiểm riêng chứ không chỉ kiểm "có bắt được vi phạm hay không".
console.log("\n--- bàn giao: nguồn vi phạm ---");
const SRC = [
  ["số ngoài miền -> value", mk({ value: 250000, unit: "%",
    plan: { intent: "direct_retrieval", metric: "gross_margin", years: ["2024"], tickers: ["VNM"] } }), "value"],
  ["câu trả lời ngược chiều -> answer",
    cmp({ answer: "Doanh thu thuần của VNM năm 2024 giảm 12,5% so với năm 2023.", insight: null }), "answer"],
  ["nhận định ngược chiều -> insight",
    cmp({ answer: "Doanh thu thuần của VNM năm 2024 tăng 12,5% so với năm 2023.",
          insight: "Doanh thu suy giảm rõ rệt." }), "insight"],
];
for (const [name, result, want] of SRC) {
  const got = auditAnswer(result, tidy).map((v) => v.source);
  const ok = got.includes(want);
  if (ok) pass++; else fail++;
  console.log(`${ok ? "  OK  " : "  FAIL"} ${name}${got.length ? `  -> [${got.join(", ")}]` : "  -> (không có vi phạm)"}`);
}

console.log(`\n${pass}/${pass + fail} ca đạt (luật + bàn giao)`);
process.exit(fail === 0 ? 0 : 1);
