/**
 * Sinh mẫu "CSV lạ" để TRÌNH DIỄN bước Chuẩn hoá dữ liệu làm việc thật.
 *
 * VÌ SAO CẦN: cũng như Auditor chỉ kêu khi có lỗi, bộ chuẩn hoá chỉ có gì để kể khi file KHÔNG
 * chuẩn. Ba file mẫu hiện có đều đã đúng dáng (header dòng 1, có cột Mã số, cột kỳ có năm) nên
 * bước này luôn im — không bao giờ chụp được ảnh nó đang làm việc. Cùng tiền lệ với nút "Thử dữ
 * liệu bẩn" và "Thử dữ liệu lệch đơn vị": dữ liệu dàn dựng có chủ ý và DÁN NHÃN RÕ, hành vi hệ
 * thống thì thật nguyên vẹn.
 *
 * BỐN LỖI DÀN DỰNG — đều là dạng gặp thật khi tải báo cáo từ web hoặc xuất từ phần mềm kế toán:
 *   1. Hai dòng tiêu đề rác trên header thật ("BÁO CÁO TÀI CHÍNH…", "Đơn vị tính: triệu đồng").
 *      → app phải dò được header ở dòng 3, VÀ lấy được hệ số "triệu" từ chính dòng rác đó.
 *   2. Mã số lệch chuẩn TT200: Tài sản cố định ghi 210 (chuẩn 220), Nợ dài hạn ghi 320 (chuẩn 330).
 *      → app phải tin theo TÊN và nói rõ đã sửa. Đây là lỗi CÓ THẬT trong file mẫu chủ đạo của
 *        dự án — hệ quả là hỏi "tài sản cố định" trả "không tìm thấy" và một luật kiểm cân đối
 *        kế toán không bao giờ chạy được.
 *   3. Mã số có số 0 đứng đầu (01, 02) — nhóm Mã số của Báo cáo kết quả kinh doanh.
 *      → app phải giữ "01", không được để thành "1" rồi tra trượt.
 *   4. Ô số bẩn: dấu trừ Unicode "−" (Word/PDF sinh ra), hậu tố đơn vị viết trong ô ("1,5 tỷ").
 *      → dấu trừ Unicode là lỗi nguy hiểm nhất: trước khi vá, lỗ thành lãi mà không dấu hiệu nào.
 *
 * Bộ số vẫn CÂN ĐỐI theo TT200 để `validate.ts` không kêu oan — mục đích file này là trình diễn
 * khâu chuẩn hoá, không phải khâu tự kiểm.
 *
 * Chạy: node make-messy-sample.mjs
 */
import fs from "node:fs";
import path from "node:path";

const OUT = path.resolve("public/samples/01_messy_financial_statements.csv");

/** Đơn vị của file này là TRIỆU đồng (khai trong dòng tiêu đề), nên số ghi ra đã chia 1e6. */
const TR = 1e6;

/** Bộ số cân đối: 100+200=270, 300+400=270, 310+330=300, 20=10−11, 60=50−51. */
function bctc(k) {
  const s = (x) => Math.round((x * k) / TR);
  return {
    "100": s(420e9), "110": s(80e9), "140": s(120e9),
    "200": s(580e9), "210": s(410e9), // 210: MÃ LỆCH — chuẩn TT200 là 220
    "270": s(1000e9),
    "310": s(250e9), "320": s(150e9), // 320: MÃ LỆCH — chuẩn TT200 là 330
    "300": s(400e9), "400": s(600e9), "440": s(1000e9),
    "01": s(950e9), "02": s(50e9), "10": s(900e9), "11": s(630e9), "20": s(270e9),
    "50": s(150e9), "51": s(30e9), "60": s(120e9),
  };
}

/** Tên chỉ tiêu viết ĐÚNG chuẩn — đây là căn cứ để app sửa được hai mã lệch ở trên. */
const TEN = {
  "100": "TÀI SẢN NGẮN HẠN",
  "110": "Tiền và các khoản tương đương tiền",
  "140": "Hàng tồn kho",
  "200": "TÀI SẢN DÀI HẠN",
  "210": "Tài sản cố định",
  "270": "TỔNG CỘNG TÀI SẢN",
  "310": "Nợ ngắn hạn",
  "320": "Nợ dài hạn",
  "300": "NỢ PHẢI TRẢ",
  "400": "VỐN CHỦ SỞ HỮU",
  "440": "TỔNG CỘNG NGUỒN VỐN",
  "01": "Doanh thu bán hàng và cung cấp dịch vụ",
  "02": "Các khoản giảm trừ doanh thu",
  "10": "Doanh thu thuần về bán hàng và cung cấp dịch vụ",
  "11": "Giá vốn hàng bán",
  "20": "Lợi nhuận gộp",
  "50": "Tổng lợi nhuận kế toán trước thuế",
  "51": "Chi phí thuế thu nhập doanh nghiệp hiện hành",
  "60": "Lợi nhuận sau thuế thu nhập doanh nghiệp",
};

const rows = [
  // LỖI 1 — hai dòng tiêu đề rác. Dòng thứ hai khai đơn vị, app phải đọc được từ đây.
  ["BÁO CÁO TÀI CHÍNH HỢP NHẤT — CÔNG TY CỔ PHẦN MẪU", "", "", "", "", ""],
  ["Đơn vị tính: triệu đồng", "", "", "", "", ""],
  ["Ticker", "Sector", "Mã số", "Chỉ tiêu", "Năm 2024", "Năm 2023"],
];

for (const tk of ["CORP_MESSY"]) {
  const y24 = bctc(1);
  const y23 = bctc(0.88);
  for (const code of Object.keys(TEN)) {
    let v24 = String(y24[code]);
    const v23 = String(y23[code]);
    // LỖI 4 — "Các khoản giảm trừ doanh thu" ghi bằng dấu trừ Unicode (U+2212), kiểu Word/PDF hay
    // sinh ra. Chọn đúng chỉ tiêu mà giá trị âm KHÔNG phá đẳng thức kế toán nào: mã 02 không nằm
    // trong luật nào của `validate.ts`, nên file vẫn cân đối và bước Tự kiểm vẫn im — đúng ý đồ,
    // vì file này để trình diễn khâu CHUẨN HOÁ, không phải khâu tự kiểm.
    if (code === "02") v24 = "−" + v24;
    // LỖI 4 — tiền mặt ghi kèm hậu tố đơn vị ngay trong ô.
    if (code === "110") v24 = (y24[code] / 1000).toLocaleString("vi-VN") + " tỷ";
    rows.push([tk, "Hàng tiêu dùng", code, TEN[code], v24, v23]);
  }
}

const csv =
  rows.map((r) => r.map((c) => (/[",]/.test(c) ? `"${String(c).replace(/"/g, '""')}"` : c)).join(",")).join("\n") + "\n";
fs.writeFileSync(OUT, "﻿" + csv, "utf8");

console.log(`Đã ghi ${OUT}`);
console.log("  1. header thật ở dòng 3 (2 dòng tiêu đề phía trên) + đơn vị 'triệu đồng' nằm trong dòng rác");
console.log("  2. Mã 210 'Tài sản cố định' (chuẩn 220), Mã 320 'Nợ dài hạn' (chuẩn 330) — app phải sửa theo tên");
console.log("  3. Mã '01', '02' có số 0 đứng đầu — app phải giữ nguyên, không thành '1', '2'");
console.log("  4. ô Mã 02 dùng dấu trừ Unicode U+2212; ô Mã 110 ghi kèm hậu tố 'tỷ'");
