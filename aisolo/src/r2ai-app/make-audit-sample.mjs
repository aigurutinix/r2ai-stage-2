/**
 * Sinh mẫu "lệch đơn vị" để TRÌNH DIỄN bước Tự kiểm (Auditor) bắt lỗi thật.
 *
 * VÌ SAO CẦN MỘT FILE DÀN DỰNG: Auditor chỉ kêu khi có gì đó sai. Với dữ liệu đúng thì nó im —
 * đúng như thiết kế, nhưng thành ra không bao giờ chụp được ảnh nó đang làm việc. Đây là cùng
 * tiền lệ với nút "Thử dữ liệu bẩn" đã có sẵn: dữ liệu được dàn dựng có chủ ý và DÁN NHÃN RÕ,
 * còn hành vi của hệ thống là thật nguyên vẹn.
 *
 * KỊCH BẢN — lỗi nhập liệu phổ biến bậc nhất ngoài đời: một chỉ tiêu bị ghi theo đơn vị khác
 * phần còn lại. Ở đây Vốn chủ sở hữu (Mã 400) và Vốn góp (411) của CORP_LECH ghi theo NGHÌN đồng
 * trong khi cả sheet theo đồng.
 *
 * Hệ quả dây chuyền, và đây mới là điểm đáng trình bày:
 *   - `validate.ts` bắt ở ĐẦU VÀO : 270 ≠ 300 + 400  → "file nguồn có vấn đề"
 *   - `audit.ts`    bắt ở ĐẦU RA  : ROE = LNST/VCSH nổ lên ~20.000% → "con số này không dùng được"
 * Hai lớp gác hai đầu, nói hai điều khác nhau, cùng chỉ về một nguyên nhân.
 *
 * CORP_CHUAN trong cùng file là nhóm đối chứng: số liệu cân đối, ROE 20% — Auditor phải IM.
 * Không có nhóm đối chứng thì không chứng minh được là Auditor biết phân biệt, chứ không phải
 * lúc nào cũng kêu.
 *
 * Chạy: node make-audit-sample.mjs
 */
import fs from "node:fs";
import path from "node:path";

const OUT = path.resolve("public/samples/01_unit_error_financial_statements.csv");

/** Bộ số CÂN ĐỐI theo TT200 (đồng). 100+200=270, 300+400=270, 310+330=300, 20=10−11, 60=50−51. */
function bctc(scale) {
  const s = (x) => Math.round(x * scale);
  return {
    100: s(420e9), 110: s(80e9), 140: s(120e9), 200: s(580e9), 220: s(410e9), 270: s(1000e9),
    310: s(250e9), 330: s(150e9), 300: s(400e9), 400: s(600e9), 411: s(500e9), 440: s(1000e9),
    10: s(900e9), 11: s(630e9), 20: s(270e9), 50: s(150e9), 51: s(30e9), 60: s(120e9),
  };
}

const TEN = {
  100: "TÀI SẢN NGẮN HẠN", 110: "Tiền và các khoản tương đương tiền", 140: "Hàng tồn kho",
  200: "TÀI SẢN DÀI HẠN", 220: "Tài sản cố định", 270: "TỔNG CỘNG TÀI SẢN",
  310: "Nợ ngắn hạn", 330: "Nợ dài hạn", 300: "NỢ PHẢI TRẢ", 400: "VỐN CHỦ SỞ HỮU",
  411: "Vốn góp của chủ sở hữu", 440: "TỔNG CỘNG NGUỒN VỐN",
  10: "Doanh thu thuần về bán hàng và cung cấp dịch vụ", 11: "Giá vốn hàng bán",
  20: "Lợi nhuận gộp", 50: "Tổng lợi nhuận kế toán trước thuế",
  51: "Chi phí thuế thu nhập doanh nghiệp hiện hành", 60: "Lợi nhuận sau thuế thu nhập doanh nghiệp",
};

const rows = [["Ticker", "Sector", "Item_Code", "Item_Name", "Year_2024", "Year_2023"]];
for (const [tk, sector, loi] of [
  ["CORP_CHUAN", "Hàng tiêu dùng", false],
  ["CORP_LECH", "Hàng tiêu dùng", true],
]) {
  const y24 = bctc(1);
  const y23 = bctc(0.88);
  if (loi) {
    // LỖI DÀN DỰNG: hai dòng vốn ghi theo NGHÌN đồng thay vì đồng.
    for (const code of [400, 411]) {
      y24[code] = Math.round(y24[code] / 1000);
      y23[code] = Math.round(y23[code] / 1000);
    }
  }
  for (const code of Object.keys(TEN)) {
    rows.push([tk, sector, code, TEN[code], String(y24[code]), String(y23[code])]);
  }
}

const csv = rows.map((r) => r.map((c) => (/[",]/.test(c) ? `"${c.replace(/"/g, '""')}"` : c)).join(",")).join("\n") + "\n";
fs.writeFileSync(OUT, "﻿" + csv, "utf8");

const roe = (v) => (120e9 / v) * 100;
console.log(`Đã ghi ${OUT}`);
console.log(`  CORP_CHUAN: VCSH 600.000.000.000 → ROE ${roe(600e9).toFixed(1)}%  (Auditor phải IM)`);
console.log(`  CORP_LECH : VCSH ${(600e9 / 1000).toLocaleString("vi-VN")} → ROE ${roe(600e6).toFixed(0)}%  (Auditor phải KÊU, ngưỡng 10.000%)`);
