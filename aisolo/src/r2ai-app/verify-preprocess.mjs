/**
 * Kiểm bộ tiền xử lý (lib/financial/preprocess.ts) trên các dáng file đã ĐO thật.
 *
 * Kiểm HAI CHIỀU, vì một bộ dò chỉ có giá trị khi nó vừa cứu được file lạ vừa KHÔNG đụng vào file
 * chuẩn. Bài học từ pipeline: nhiều can thiệp "cứu" được vài ca nhưng làm hỏng số ca nhiều hơn,
 * và chỉ đo một chiều thì không thấy.
 *
 * Ba nhóm ca:
 *   HEADER    — dò đúng dòng header thật, và giữ nguyên dòng 0 với file chuẩn.
 *   COT       — ánh xạ cột (Mã số / tên / ticker / kỳ), gồm ca hai cột trùng nhãn kỳ.
 *   ANH_XA    — tên chỉ tiêu → Mã số. Ràng buộc CỨNG: 0 ánh xạ SAI.
 *
 * Chạy:  npx tsc lib/financial/preprocess.ts --outDir <dir> --module commonjs \
 *          --target es2020 --moduleResolution node --skipLibCheck
 *        node verify-preprocess.mjs <dir>
 */
import { createRequire } from "node:module";
import path from "node:path";

const dir = process.argv[2];
if (!dir) {
  console.error("Thiếu tham số: node verify-preprocess.mjs <thư-mục-đã-compile>");
  process.exit(2);
}
const require_ = createRequire(import.meta.url);
const { preprocess, detectHeaderRow, maSoFromName } = require_(path.resolve(dir, "preprocess.js"));

const fails = [];
let pass = 0;
const check = (ok, msg) => (ok ? pass++ : fails.push(msg));

/** CSV nhiều dòng → ma trận, để ca kiểm đọc được như file thật. */
const M = (s) => s.trim().split("\n").map((r) => r.split(","));

// ---------- HEADER ----------
const HEADER_CASES = [
  ["header ở dòng 1 (chuẩn) — PHẢI giữ nguyên", `Ma_So,Chi_Tieu,Year_2024\n270,TONG TAI SAN,1000`, 0],
  [
    "2 dòng tiêu đề rác phía trên",
    `BAO CAO TAI CHINH HOP NHAT,,\nDon vi tinh: trieu dong,,\nMa_So,Chi_Tieu,Year_2024\n270,TONG TAI SAN,1000`,
    2,
  ],
  [
    "rác + header đảo thứ tự cột",
    `CONG TY CP ABC,,,\nChi tieu,Ma so,31/12/2024,31/12/2023\nTONG TAI SAN,270,1000,900`,
    1,
  ],
  ["không có cột Mã số — header vẫn ở dòng 0", `Chi_Tieu,Year_2024,Year_2023\nTONG TAI SAN,1000,900`, 0],
];
for (const [note, csv, want] of HEADER_CASES) {
  const got = detectHeaderRow(M(csv));
  check(got === want, `[HEADER] ${note}: dò ra dòng ${got}, kỳ vọng ${want}`);
}

// ---------- CỘT ----------
{
  const r = preprocess(M(`Ticker,Ma_So,Chi_Tieu,Year_2024,Year_2023\nC1,270,TONG TAI SAN,1000,900`));
  check(r.fatal === null, `[CỘT] file chuẩn không được fatal: ${r.fatal}`);
  check(r.columns.maSo === 1, `[CỘT] cột Mã số phải là 1, ra ${r.columns.maSo}`);
  check(r.columns.name === 2, `[CỘT] cột tên phải là 2, ra ${r.columns.name}`);
  check(r.columns.ticker === 0, `[CỘT] cột ticker phải là 0, ra ${r.columns.ticker}`);
  check(
    r.columns.periods.map((p) => p.label).join(",") === "2024,2023",
    `[CỘT] kỳ phải là 2024,2023 — ra ${r.columns.periods.map((p) => p.label).join(",")}`,
  );
  check(r.maSoSource === "column", `[CỘT] maSoSource phải là "column", ra "${r.maSoSource}"`);
}
{
  // Header tiếng Việt có dấu + cột kỳ dạng ngày.
  const r = preprocess(M(`Mã số,Chỉ tiêu,31/12/2024,31/12/2023\n270,TỔNG TÀI SẢN,1000,900`));
  check(r.fatal === null, `[CỘT] header tiếng Việt không được fatal: ${r.fatal}`);
  check(r.columns.maSo === 0 && r.columns.name === 1, `[CỘT] header tiếng Việt ánh xạ sai cột`);
  check(
    r.columns.periods.map((p) => p.label).join(",") === "2024,2023",
    `[CỘT] cột kỳ dạng ngày phải ra 2024,2023`,
  );
}
{
  // Hai cột trùng nhãn kỳ: PHẢI giữ cả hai, không được nuốt mất một cột.
  const r = preprocess(M(`Ma_So,Chi_Tieu,Year_2024,Year_2024\n270,TONG TAI SAN,1000,2000`));
  check(r.columns.periods.length === 2, `[CỘT] hai cột trùng nhãn phải giữ CẢ HAI, ra ${r.columns.periods.length}`);
  const labels = r.columns.periods.map((p) => p.label);
  check(new Set(labels).size === 2, `[CỘT] hai cột trùng nhãn phải có nhãn phân biệt, ra ${labels.join(",")}`);
  check(
    r.notes.some((n) => /trùng nhãn kỳ/.test(n.text)),
    `[CỘT] phải có note báo cột trùng nhãn kỳ`,
  );
}
{
  // Thiếu cột kỳ → fatal có lời giải thích, KHÔNG được im lặng.
  const r = preprocess(M(`Ma_So,Chi_Tieu,Nam nay,Nam truoc\n270,TONG TAI SAN,1000,900`));
  check(typeof r.fatal === "string" && r.fatal.length > 20, `[CỘT] thiếu cột kỳ phải fatal kèm giải thích`);
}

// ---------- ÁNH XẠ tên → Mã số ----------
{
  // Số 0 đứng đầu phải được giữ nguyên (registry khoá theo chuỗi "01").
  check(maSoFromName("Doanh thu bán hàng và cung cấp dịch vụ") === "01", `[ÁNH XẠ] "doanh thu bán hàng" phải ra "01"`);
  check(maSoFromName("Các khoản giảm trừ doanh thu") === "02", `[ÁNH XẠ] "giảm trừ doanh thu" phải ra "02"`);
  check(maSoFromName("tổng tài sản") === "270", `[ÁNH XẠ] khớp không phân biệt hoa thường`);
  check(maSoFromName("TONG TAI SAN") === "270", `[ÁNH XẠ] khớp khi bỏ dấu tiếng Việt`);
  // RÀNG BUỘC CỨNG: dòng con không được ăn mã của dòng cha.
  check(maSoFromName("Hàng tồn kho ròng") === null, `[ÁNH XẠ] "Hàng tồn kho ròng" KHÔNG được gán mã 140`);
  check(
    maSoFromName("Dự phòng giảm giá hàng tồn kho") === null,
    `[ÁNH XẠ] "Dự phòng giảm giá hàng tồn kho" KHÔNG được gán mã 140`,
  );
  check(maSoFromName("Tiền mặt tại quỹ") === null, `[ÁNH XẠ] "Tiền mặt tại quỹ" KHÔNG được gán mã 110`);
  check(maSoFromName("") === null, `[ÁNH XẠ] chuỗi rỗng phải ra null`);
}
{
  // File KHÔNG có cột Mã số: suy từ tên, và nói rõ độ phủ.
  const r = preprocess(
    M(`Chi_Tieu,Year_2024\nTổng tài sản,1000\nHàng tồn kho,200\nMột chỉ tiêu lạ không có trong registry,50`),
  );
  check(r.fatal === null, `[ÁNH XẠ] file chỉ có tên chỉ tiêu phải nạp được, không fatal`);
  check(r.maSoSource === "inferred", `[ÁNH XẠ] maSoSource phải là "inferred", ra "${r.maSoSource}"`);
  check(
    JSON.stringify(r.maSoPerRow) === JSON.stringify(["270", "140", null]),
    `[ÁNH XẠ] maSoPerRow sai: ${JSON.stringify(r.maSoPerRow)}`,
  );
  check(
    r.coverage.mapped === 2 && r.coverage.total === 3,
    `[ÁNH XẠ] coverage phải là 2/3, ra ${r.coverage.mapped}/${r.coverage.total}`,
  );
  check(r.notes.some((n) => /2\/3/.test(n.text)), `[ÁNH XẠ] phải có note nêu độ phủ 2/3`);
}
{
  // Mã lệch chuẩn TT200: tin theo TÊN và nói rõ đã sửa.
  const r = preprocess(M(`Ma_So,Chi_Tieu,Year_2024\n210,Tài sản cố định,500\n270,Tổng tài sản,1000`));
  check(r.maSoSource === "corrected", `[ÁNH XẠ] maSoSource phải là "corrected", ra "${r.maSoSource}"`);
  check(r.maSoPerRow[0] === "220", `[ÁNH XẠ] mã 210 "Tài sản cố định" phải sửa thành 220, ra ${r.maSoPerRow[0]}`);
  check(r.maSoPerRow[1] === "270", `[ÁNH XẠ] mã 270 đúng chuẩn thì PHẢI giữ nguyên, ra ${r.maSoPerRow[1]}`);
  check(r.notes.some((n) => /lệch chuẩn/.test(n.text)), `[ÁNH XẠ] phải có note báo đã sửa mã lệch chuẩn`);
}
{
  // Dòng có mã nhưng tên không nhận ra → GIỮ NGUYÊN mã trong file, không được bịa.
  const r = preprocess(M(`Ma_So,Chi_Tieu,Year_2024\n136,Phải thu ngắn hạn khác,50`));
  check(r.maSoPerRow[0] === "136", `[ÁNH XẠ] tên lạ + có mã thì giữ mã gốc, ra ${r.maSoPerRow[0]}`);
  check(r.maSoSource === "column", `[ÁNH XẠ] không sửa gì thì maSoSource phải là "column"`);
}
{
  // Mã số một chữ số đọc từ ma trận thô phải giữ "01", không thành "1".
  const r = preprocess(M(`Ma_So,Chi_Tieu,Year_2024\n01,Doanh thu bán hàng và cung cấp dịch vụ,5000`));
  check(r.maSoPerRow[0] === "01", `[ÁNH XẠ] mã "01" phải giữ số 0 đầu, ra ${JSON.stringify(r.maSoPerRow[0])}`);
}

const tong = pass + fails.length;
console.log(`preprocess: ${pass}/${tong} ca đạt`);
if (fails.length) {
  console.log("\nCA HỎNG:");
  for (const f of fails) console.log("  " + f);
  process.exit(1);
}
console.log("Tất cả ca đạt.");
