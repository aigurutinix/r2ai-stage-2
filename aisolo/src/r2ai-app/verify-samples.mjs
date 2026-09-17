/**
 * Kiểm HỒI QUY trên các file mẫu thật trong `public/samples/`.
 *
 * Vì sao cần: `verify-clean` và `verify-preprocess` chạy trên ca dựng tay. Chúng không chứng minh
 * được rằng chuỗi nạp file THẬT vẫn ra đúng kết quả cũ. Bài học xuyên suốt dự án: mọi chỉ số nội
 * bộ đều dương mà kết quả thật vẫn giảm — nên phải đo trên chính dữ liệu sẽ dùng khi demo.
 *
 * Kiểm hai chiều:
 *   KHÔNG HỒI QUY — 3 file mẫu đang chạy tốt phải giữ nguyên số công ty/kỳ và các Mã số chủ chốt.
 *   MỞ KHOÁ       — các thứ trước đây hỏng: file không có cột Mã số phải nạp được; mã lệch chuẩn
 *                   TT200 phải được sửa; luật kiểm cân đối "Nợ = NNH + NDH" phải chạy được.
 *
 * Chạy:  npx tsc lib/financial/{normalize,preprocess,clean,registry,validate,types}.ts \
 *          --outDir <dir> --module commonjs --target es2020 --moduleResolution node --skipLibCheck
 *        node verify-samples.mjs <dir>
 */
import { createRequire } from "node:module";
import path from "node:path";
import fs from "node:fs";
import * as XLSX from "xlsx";

const dir = process.argv[2];
if (!dir) {
  console.error("Thiếu tham số: node verify-samples.mjs <thư-mục-đã-compile>");
  process.exit(2);
}
const require_ = createRequire(import.meta.url);
const { normalize } = require_(path.resolve(dir, "normalize.js"));
const { validateTicker } = require_(path.resolve(dir, "validate.js"));

const fails = [];
let pass = 0;
const check = (ok, msg) => (ok ? pass++ : fails.push(msg));

/** Đọc file mẫu thành ma trận đúng cách `xlsx-client.ts` làm. */
function loadMatrix(file) {
  const txt = new TextDecoder("utf-8").decode(fs.readFileSync(path.join("public/samples", file)));
  const wb = XLSX.read(txt, { type: "string" });
  const name = wb.SheetNames[0];
  const aoa = XLSX.utils.sheet_to_json(wb.Sheets[name], {
    header: 1,
    raw: false,
    defval: "",
    blankrows: false,
  });
  return { matrix: aoa.map((r) => (r ?? []).map((c) => (c == null ? "" : String(c)))), sheet: name };
}
const load = (file) => {
  const { matrix, sheet } = loadMatrix(file);
  return normalize(matrix, sheet);
};

// ---------- KHÔNG HỒI QUY ----------
{
  const t = load("01_corporate_financial_statements.csv");
  check(t.fatal == null, `[HỒI QUY] file mẫu chuẩn không được fatal: ${t.fatal}`);
  check(t.tickers.length > 0, `[HỒI QUY] file mẫu chuẩn phải có công ty, ra ${t.tickers.length}`);
  check(
    t.periods.join(",") === "2024,2023",
    `[HỒI QUY] kỳ phải là 2024,2023 — ra ${t.periods.join(",")}`,
  );
  check(t.prep?.headerRow === 0, `[HỒI QUY] header phải ở dòng 0, ra ${t.prep?.headerRow}`);
  const it = t.byTicker[t.tickers[0]];
  for (const ma of ["100", "270", "300", "400", "10", "60"]) {
    check(it?.[ma] != null, `[HỒI QUY] mất Mã số ${ma} của ${t.tickers[0]}`);
  }
  check(typeof it?.["270"]?.values["2024"] === "number", `[HỒI QUY] giá trị Mã 270 kỳ 2024 phải là số`);
}
{
  const t = load("01_dirty_corporate_financial_statements.csv");
  check(t.fatal == null, `[HỒI QUY] file bẩn phải nạp được: ${t.fatal}`);
  check(t.tickers.length > 0, `[HỒI QUY] file bẩn phải có công ty`);
  check(
    t.cleaningNotes.some((n) => /làm sạch/.test(n)),
    `[HỒI QUY] file bẩn phải còn note "đã tự làm sạch"`,
  );
  const v = t.byTicker[t.tickers[0]]?.["100"]?.values[t.periods[0]];
  check(typeof v === "number" && v > 0, `[HỒI QUY] ô "60,673,395 VND" phải đọc ra số dương, ra ${v}`);
}
{
  const t = load("01_unit_error_financial_statements.csv");
  check(t.fatal == null, `[HỒI QUY] file lệch đơn vị phải nạp được: ${t.fatal}`);
  check(
    t.tickers.includes("CORP_LECH") && t.tickers.includes("CORP_CHUAN"),
    `[HỒI QUY] file lệch đơn vị phải có cả CORP_LECH và CORP_CHUAN — ra ${t.tickers.join(",")}`,
  );
  // Nút demo "Thử dữ liệu lệch đơn vị" dựa vào việc ROE của CORP_LECH vọt lên phi lý.
  const eq = t.byTicker["CORP_LECH"]?.["400"]?.values["2024"];
  const ni = t.byTicker["CORP_LECH"]?.["60"]?.values["2024"];
  check(typeof eq === "number" && typeof ni === "number", `[HỒI QUY] CORP_LECH phải có Mã 400 và 60`);
  check(
    typeof eq === "number" && typeof ni === "number" && Math.abs((ni / eq) * 100) > 10_000,
    `[HỒI QUY] ROE của CORP_LECH phải vẫn vượt 10.000% để Auditor kích hoạt`,
  );
}

// ---------- MỞ KHOÁ ----------
{
  // File này là bảng TỶ SỐ dựng sẵn (Current_Ratio, ROE_Pct…): không có cột kỳ theo năm, cũng không
  // có chỉ tiêu TT200 nào. Từ chối là ĐÚNG — nó không phải báo cáo tài chính.
  // Điều bắt buộc, và cũng là thứ trước đây thiếu: phải NÓI RA thiếu gì, không phải một màn hình câm.
  const t = load("02_financial_ratios_valuation.csv");
  check(typeof t.fatal === "string" && t.fatal.length > 20, `[MỞ KHOÁ] phải có lời giải thích, ra: ${t.fatal}`);
  check(/kỳ|năm|Mã số|chỉ tiêu/i.test(t.fatal ?? ""), `[MỞ KHOÁ] lời giải thích phải nêu thiếu gì, ra: ${t.fatal}`);
  check(
    /Year_2024|Năm 2024|31\/12/.test(t.fatal ?? ""),
    `[MỞ KHOÁ] lời giải thích nên kèm ví dụ cụ thể để người dùng biết sửa file thế nào`,
  );
}
{
  // File CÓ cột kỳ nhưng KHÔNG có cột Mã số — dạng bị từ chối thẳng trước đây. Phải nạp được nhờ
  // suy Mã số từ tên chỉ tiêu.
  const csv = "Chi_Tieu,Year_2024,Year_2023\nTổng tài sản,1000,900\nHàng tồn kho,200,180\nChỉ tiêu lạ,5,4";
  const wb = XLSX.read(csv, { type: "string" });
  const aoa = XLSX.utils
    .sheet_to_json(wb.Sheets[wb.SheetNames[0]], { header: 1, raw: false, defval: "", blankrows: false })
    .map((r) => (r ?? []).map((c) => (c == null ? "" : String(c))));
  const t = normalize(aoa, "Sheet1");
  check(t.fatal == null, `[MỞ KHOÁ] file chỉ có tên chỉ tiêu phải nạp được, fatal=${t.fatal}`);
  check(t.prep?.maSoSource === "inferred", `[MỞ KHOÁ] maSoSource phải là "inferred", ra "${t.prep?.maSoSource}"`);
  check(t.byTicker["(toàn bộ)"]?.["270"]?.values["2024"] === 1000, `[MỞ KHOÁ] "Tổng tài sản" phải tra được ở Mã 270`);
  check(t.byTicker["(toàn bộ)"]?.["140"]?.values["2024"] === 200, `[MỞ KHOÁ] "Hàng tồn kho" phải tra được ở Mã 140`);
  check(
    t.prep?.coverage.mapped === 2 && t.prep?.coverage.total === 3,
    `[MỞ KHOÁ] độ phủ phải là 2/3, ra ${t.prep?.coverage.mapped}/${t.prep?.coverage.total}`,
  );
}
{
  // Mã lệch chuẩn TT200 trong chính file mẫu: TSCĐ ghi 210 (chuẩn 220), Nợ dài hạn ghi 320 (chuẩn 330).
  const t = load("01_corporate_financial_statements.csv");
  const it = t.byTicker[t.tickers[0]];
  check(it?.["220"] != null, `[MỞ KHOÁ] "Tài sản cố định" phải tra được ở Mã 220 (file ghi 210)`);
  check(it?.["330"] != null, `[MỞ KHOÁ] "Nợ dài hạn" phải tra được ở Mã 330 (file ghi 320)`);
  check(t.prep?.maSoSource === "corrected", `[MỞ KHOÁ] maSoSource phải là "corrected", ra "${t.prep?.maSoSource}"`);
  check(
    t.prep?.notes.some((n) => /lệch chuẩn/.test(n.text)),
    `[MỞ KHOÁ] phải có note báo đã sửa Mã số lệch chuẩn`,
  );
  // Luật "Nợ phải trả = Nợ ngắn hạn + Nợ dài hạn" cần Mã 330 — trước đây KHÔNG BAO GIỜ chạy.
  const sum = validateTicker(t, t.tickers[0]);
  check(sum.checked > 0, `[MỞ KHOÁ] validateTicker phải kiểm được ít nhất một đẳng thức`);
}
{
  // File có dòng tiêu đề rác phía trên header thật — trước đây hỏng hoàn toàn.
  const csv =
    "BAO CAO TAI CHINH HOP NHAT,,\nDon vi tinh: trieu dong,,\nMa_So,Chi_Tieu,Year_2024\n270,TONG TAI SAN,1000";
  const wb = XLSX.read(csv, { type: "string" });
  const aoa = XLSX.utils
    .sheet_to_json(wb.Sheets[wb.SheetNames[0]], { header: 1, raw: false, defval: "", blankrows: false })
    .map((r) => (r ?? []).map((c) => (c == null ? "" : String(c))));
  const t = normalize(aoa, "Sheet1");
  check(t.fatal == null, `[MỞ KHOÁ] file có dòng rác phải nạp được: ${t.fatal}`);
  check(t.prep?.headerRow === 2, `[MỞ KHOÁ] phải dò header ở dòng 2, ra ${t.prep?.headerRow}`);
  check(t.byTicker["(toàn bộ)"]?.["270"] != null, `[MỞ KHOÁ] phải đọc được Mã 270`);
  // "Đơn vị tính: triệu đồng" nằm TRÊN header — trước đây không bao giờ thấy vì chỉ quét trong bảng.
  check(
    t.byTicker["(toàn bộ)"]?.["270"]?.values["2024"] === 1000 * 1e6,
    `[MỞ KHOÁ] phải áp hệ số triệu từ dòng tiêu đề, ra ${t.byTicker["(toàn bộ)"]?.["270"]?.values["2024"]}`,
  );
}

const tong = pass + fails.length;
console.log(`file mẫu: ${pass}/${tong} ca đạt`);
if (fails.length) {
  console.log("\nCA HỎNG:");
  for (const f of fails) console.log("  " + f);
  process.exit(1);
}
console.log("Tất cả ca đạt.");
