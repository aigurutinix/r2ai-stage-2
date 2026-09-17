import type { TidyTable } from "./types";

/**
 * Validation đẳng thức kế toán (TT200) — bắt lỗi số/đơn vị/nhập liệu + guard chống hallucination.
 * Chạy deterministic sau normalize, sai số cho phép 1%.
 */
export interface ValidationIssue {
  rule: string;
  period: string;
  reported: number;
  expected: number;
  diffPct: number;
}
export interface ValidationSummary {
  checked: number;
  passed: number;
  issues: ValidationIssue[];
}

type Get = (maSo: string) => number | null;
const add = (a: number | null, b: number | null) => (a == null || b == null ? null : a + b);
/**
 * Trừ một khoản CÓ THỂ VẮNG: không khai báo nghĩa là bằng 0, không phải "không tính được".
 * Dùng cho thuế TNDN hoãn lại — nhiều báo cáo không có dòng này vì doanh nghiệp không phát sinh.
 */
const subOpt = (a: number | null, b: number | null) => (a == null ? null : a - (b ?? 0));
/**
 * Trừ một khoản KHẤU TRỪ theo ĐỘ LỚN. Báo cáo in khoản khấu trừ trong ngoặc (= số âm) không nhất
 * quán giữa các doanh nghiệp: AAA ghi giá vốn −8.215 tỷ, HPG ghi dương. Trừ thẳng theo dấu thì
 * luật báo sai ở nhóm thứ nhất — đã thấy thật trên AAA 2019/2020/2021.
 * Chỉ dùng cho khoản KHÔNG THỂ âm trên thực tế (giá vốn, chi phí thuế hiện hành); khoản có thể
 * âm thật như thuế TNDN hoãn lại thì giữ nguyên dấu vì dấu ở đó mang nghĩa.
 */
const subAbs = (a: number | null, b: number | null) => (a == null || b == null ? null : a - Math.abs(b));

const RULES: { name: string; target: string; expected: (g: Get) => number | null }[] = [
  { name: "Tổng tài sản = Nợ phải trả + Vốn chủ sở hữu", target: "270", expected: (g) => add(g("300"), g("400")) },
  { name: "Tổng tài sản = TS ngắn hạn + TS dài hạn", target: "270", expected: (g) => add(g("100"), g("200")) },
  { name: "Nợ phải trả = Nợ ngắn hạn + Nợ dài hạn", target: "300", expected: (g) => add(g("310"), g("330")) },
  { name: "Lợi nhuận gộp = Doanh thu thuần − Giá vốn", target: "20", expected: (g) => subAbs(g("10"), g("11")) },
  // TT200: 60 = 50 − 51 − 52. Bỏ sót thuế hoãn lại (52) làm luật này BÁO SAI ở mọi doanh nghiệp
  // có phát sinh khoản đó — đã thấy thật trên HPG 2019 (lệch 1,12%, đúng bằng phần thuế hoãn lại).
  { name: "LN sau thuế = LN trước thuế − Chi phí thuế", target: "60",
    expected: (g) => subOpt(subAbs(g("50"), g("51")), g("52")) },
];

const TOLERANCE = 0.01;

export function validateTicker(tidy: TidyTable, ticker: string): ValidationSummary {
  const items = tidy.byTicker[ticker] ?? {};
  const issues: ValidationIssue[] = [];
  let checked = 0;
  let passed = 0;

  for (const period of tidy.periods) {
    const g: Get = (maSo) => items[maSo]?.values[period] ?? null;
    for (const rule of RULES) {
      const reported = g(rule.target);
      const expected = rule.expected(g);
      if (reported == null || expected == null) continue; // thiếu số → không kiểm được rule này
      checked++;
      const denom = Math.max(Math.abs(reported), Math.abs(expected), 1);
      const diffPct = Math.abs(reported - expected) / denom;
      if (diffPct <= TOLERANCE) passed++;
      else issues.push({ rule: rule.name, period, reported, expected, diffPct: Math.round(diffPct * 10000) / 100 });
    }
  }
  return { checked, passed, issues };
}
