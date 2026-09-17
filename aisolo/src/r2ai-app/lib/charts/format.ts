/** Định dạng số tiền tiếng Việt cho trục/nhãn chart (tỷ/tr/N). */
export function fmtVN(v: number): string {
  if (v == null || !isFinite(v)) return "";
  const a = Math.abs(v);
  if (a >= 1e9) return (v / 1e9).toLocaleString("vi-VN", { maximumFractionDigits: 1 }) + " tỷ";
  if (a >= 1e6) return (v / 1e6).toLocaleString("vi-VN", { maximumFractionDigits: 1 }) + " tr";
  if (a >= 1e3) return (v / 1e3).toLocaleString("vi-VN", { maximumFractionDigits: 0 }) + "N";
  return v.toLocaleString("vi-VN", { maximumFractionDigits: 1 });
}

/** Phần trăm gọn kiểu VN: 75,4% (thay vì 75.3555%). */
export function fmtPct(p: number): string {
  return p.toLocaleString("vi-VN", { maximumFractionDigits: 1 }) + "%";
}
