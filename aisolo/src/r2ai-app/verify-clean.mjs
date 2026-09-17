/**
 * Kiểm `cleanNumber` (lib/financial/clean.ts) trên bộ ca nhiễu đã ĐO trên dữ liệu thật.
 *
 * Vì sao cần: ba lỗi được vá ở đây đều thuộc loại SAI TRONG IM LẶNG — hàm trả về một con số
 * trông hợp lệ nên không có gì báo động. Dấu trừ Unicode biến lỗ thành lãi; "1,5 tỷ" lệch 9 bậc;
 * "1e6" thành 16. Không đo thì không biết.
 *
 * Bộ ca chia hai phần và PHẢI kiểm cả hai:
 *   GIU_NGUYEN — 11 ca vốn đã đúng. Vá mà làm hỏng chúng thì tệ hơn không vá.
 *   PHAI_SUA   — các ca vá lần này.
 *
 * Chạy:  npx tsc lib/financial/clean.ts --outDir <dir> --module commonjs \
 *          --target es2020 --moduleResolution node --skipLibCheck
 *        node verify-clean.mjs <dir>
 */
import { createRequire } from "node:module";
import path from "node:path";

const dir = process.argv[2];
if (!dir) {
  console.error("Thiếu tham số: node verify-clean.mjs <thư-mục-đã-compile>");
  process.exit(2);
}
const require_ = createRequire(import.meta.url);
const { cleanNumber } = require_(path.resolve(dir, "clean.js"));

/** [đầu vào, kỳ vọng, mô tả] */
const GIU_NGUYEN = [
  ["1,234,567", 1234567, "phân cách , kiểu Mỹ"],
  ["1.234.567", 1234567, "phân cách . kiểu Việt"],
  ["(1,500,000)", -1500000, "ngoặc đơn = âm"],
  ["60,673,395 VND", 60673395, "hậu tố tiền tệ VND"],
  ["2,15", 2.15, "thập phân kiểu Việt"],
  ["2.15", 2.15, "thập phân kiểu Mỹ"],
  ["1.234,56", 1234.56, "hỗn hợp kiểu Việt"],
  ["1,234.56", 1234.56, "hỗn hợp kiểu Mỹ"],
  ["1,234,567.89", 1234567.89, "đầy đủ kiểu Mỹ"],
  ["1234567", 1234567, "không phân cách"],
  ["-1500", -1500, "âm hyphen ASCII"],
  ["N/A", null, "N/A"],
  ["-", null, "gạch ngang đơn"],
  ["", null, "rỗng"],
  ["nan", null, "nan"],
  [null, null, "null"],
  [1234.5, 1234.5, "đã là number thì trả nguyên"],
];

const PHAI_SUA = [
  // 1 — dấu trừ Unicode. Trước khi vá: +1500 (SAI DẤU, im lặng).
  ["−1500", -1500, "dấu trừ toán học U+2212"],
  ["–1500", -1500, "en dash U+2013"],
  ["—1500", -1500, "em dash U+2014"],
  ["−1.234.567", -1234567, "U+2212 kèm phân cách nghìn"],
  // Gạch ngang đơn lẻ (mọi biến thể) vẫn phải là null, không phải 0.
  ["–", null, "en dash đơn lẻ vẫn là null"],
  ["—", null, "em dash đơn lẻ vẫn là null"],
  // 2 — hậu tố đơn vị trong chính ô. Trước khi vá: 1.5 (lệch 9 bậc).
  ["1,5 tỷ", 1.5e9, "hậu tố tỷ"],
  ["2 triệu", 2e6, "hậu tố triệu"],
  ["3,5 nghìn", 3.5e3, "hậu tố nghìn"],
  ["(2 tỷ)", -2e9, "hậu tố tỷ + ngoặc âm"],
  ["−1,5 tỷ", -1.5e9, "hậu tố tỷ + dấu trừ Unicode"],
  // 3 — ký hiệu khoa học. Trước khi vá: 16 (SAI, trông hợp lệ).
  ["1e6", 1e6, "ký hiệu khoa học"],
  ["1.5e3", 1500, "khoa học có thập phân"],
  ["-2e3", -2000, "khoa học âm"],
];

let pass = 0;
const fails = [];
const run = (nhom, cases) => {
  for (const [inp, exp, note] of cases) {
    const got = cleanNumber(inp);
    // So sánh có dung sai để tránh nhiễu dấu phẩy động khi nhân hệ số.
    const ok = exp === null ? got === null
      : got !== null && Math.abs(got - exp) <= Math.max(1e-9, Math.abs(exp) * 1e-12);
    if (ok) pass++;
    else fails.push(`[${nhom}] ${JSON.stringify(inp)} -> ${got}, kỳ vọng ${exp}  (${note})`);
  }
};
run("GIỮ NGUYÊN", GIU_NGUYEN);
run("PHẢI SỬA", PHAI_SUA);

const tong = GIU_NGUYEN.length + PHAI_SUA.length;
console.log(`cleanNumber: ${pass}/${tong} ca đạt`);
if (fails.length) {
  console.log("\nCA HỎNG:");
  for (const f of fails) console.log("  " + f);
  process.exit(1);
}
console.log("Tất cả ca đạt.");
