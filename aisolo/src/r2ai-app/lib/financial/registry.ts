/**
 * Metric Registry — trái tim chống hallucination.
 * Khoá theo Mã số chuẩn Thông tư 200/2014/TT-BTC.
 * LLM chỉ được chọn metric.key trong danh mục này; code tính deterministic theo đây.
 *
 * statementType: BS = Bảng cân đối kế toán, PL = Kết quả kinh doanh, CF = Lưu chuyển tiền tệ.
 * (Mã số BS 100–440 và PL 01–70 không trùng nhau nên tra theo Mã số an toàn trong 1 bảng gộp.
 *  CF dùng lại dải 01–70 → sẽ tra theo statementType khi có bảng CF riêng — bổ sung sau.)
 */
export type StatementType = "BS" | "PL" | "CF";

export interface MetricDef {
  key: string;
  label: string;
  aliases: string[];
  kind: "line" | "ratio";
  /** với line item: Mã số + loại báo cáo */
  maSo?: string;
  statementType?: StatementType;
  /** với ratio: công thức theo Mã số */
  formula?: { numer: string; denom: string; percent?: boolean };
  unit?: "VND" | "%" | "x";
}

// ---- Bảng cân đối kế toán (BS) ----
const BS_METRICS: MetricDef[] = [
  { key: "current_assets", label: "Tài sản ngắn hạn", aliases: ["tài sản ngắn hạn", "current assets", "tscd ngắn hạn"], kind: "line", maSo: "100", statementType: "BS", unit: "VND" },
  { key: "cash", label: "Tiền và tương đương tiền", aliases: ["tiền và các khoản tương đương tiền", "tiền mặt", "cash and cash equivalents", "cash"], kind: "line", maSo: "110", statementType: "BS", unit: "VND" },
  { key: "short_term_investments", label: "Đầu tư tài chính ngắn hạn", aliases: ["đầu tư tài chính ngắn hạn", "short-term investments"], kind: "line", maSo: "120", statementType: "BS", unit: "VND" },
  { key: "short_term_receivables", label: "Phải thu ngắn hạn", aliases: ["các khoản phải thu ngắn hạn", "phải thu ngắn hạn", "short-term receivables"], kind: "line", maSo: "130", statementType: "BS", unit: "VND" },
  { key: "inventory", label: "Hàng tồn kho", aliases: ["hàng tồn kho", "tồn kho", "inventory"], kind: "line", maSo: "140", statementType: "BS", unit: "VND" },
  { key: "long_term_assets", label: "Tài sản dài hạn", aliases: ["tài sản dài hạn", "long-term assets", "non-current assets"], kind: "line", maSo: "200", statementType: "BS", unit: "VND" },
  { key: "fixed_assets", label: "Tài sản cố định", aliases: ["tài sản cố định", "tscd", "fixed assets"], kind: "line", maSo: "220", statementType: "BS", unit: "VND" },
  { key: "total_assets", label: "Tổng tài sản", aliases: ["tổng tài sản", "tổng cộng tài sản", "total assets"], kind: "line", maSo: "270", statementType: "BS", unit: "VND" },
  { key: "total_liabilities", label: "Nợ phải trả", aliases: ["nợ phải trả", "total liabilities", "liabilities", "tổng nợ"], kind: "line", maSo: "300", statementType: "BS", unit: "VND" },
  { key: "current_liabilities", label: "Nợ ngắn hạn", aliases: ["nợ ngắn hạn", "current liabilities"], kind: "line", maSo: "310", statementType: "BS", unit: "VND" },
  { key: "long_term_liabilities", label: "Nợ dài hạn", aliases: ["nợ dài hạn", "long-term liabilities"], kind: "line", maSo: "330", statementType: "BS", unit: "VND" },
  { key: "equity", label: "Vốn chủ sở hữu", aliases: ["vốn chủ sở hữu", "vcsh", "equity", "owner equity", "owner's equity"], kind: "line", maSo: "400", statementType: "BS", unit: "VND" },
  { key: "charter_capital", label: "Vốn góp của chủ sở hữu", aliases: ["vốn góp của chủ sở hữu", "vốn điều lệ", "charter capital", "vốn góp", "vốn cổ phần"], kind: "line", maSo: "411", statementType: "BS", unit: "VND" },
  { key: "total_resources", label: "Tổng nguồn vốn", aliases: ["tổng cộng nguồn vốn", "tổng nguồn vốn", "total resources", "total equity and liabilities"], kind: "line", maSo: "440", statementType: "BS", unit: "VND" },
];

// ---- Kết quả kinh doanh (PL) ----
const PL_METRICS: MetricDef[] = [
  { key: "gross_revenue", label: "Doanh thu bán hàng và cung cấp dịch vụ", aliases: ["doanh thu bán hàng và cung cấp dịch vụ", "doanh thu bán hàng", "gross revenue"], kind: "line", maSo: "01", statementType: "PL", unit: "VND" },
  { key: "revenue_deductions", label: "Các khoản giảm trừ doanh thu", aliases: ["các khoản giảm trừ doanh thu", "giảm trừ doanh thu", "revenue deductions"], kind: "line", maSo: "02", statementType: "PL", unit: "VND" },
  { key: "net_revenue", label: "Doanh thu thuần", aliases: ["doanh thu thuần", "doanh thu", "net revenue", "revenue", "net sales", "doanh thu thuần về bán hàng và cung cấp dịch vụ"], kind: "line", maSo: "10", statementType: "PL", unit: "VND" },
  { key: "cogs", label: "Giá vốn hàng bán", aliases: ["giá vốn hàng bán", "giá vốn", "cogs", "cost of goods sold", "giá vốn hàng bán và dịch vụ cung cấp"], kind: "line", maSo: "11", statementType: "PL", unit: "VND" },
  { key: "gross_profit", label: "Lợi nhuận gộp", aliases: ["lợi nhuận gộp", "lãi gộp", "gross profit", "lợi nhuận gộp về bán hàng và cung cấp dịch vụ"], kind: "line", maSo: "20", statementType: "PL", unit: "VND" },
  { key: "financial_income", label: "Doanh thu hoạt động tài chính", aliases: ["doanh thu hoạt động tài chính", "doanh thu tài chính", "financial income"], kind: "line", maSo: "21", statementType: "PL", unit: "VND" },
  { key: "financial_expense", label: "Chi phí tài chính", aliases: ["chi phí tài chính", "financial expense"], kind: "line", maSo: "22", statementType: "PL", unit: "VND" },
  { key: "interest_expense", label: "Chi phí lãi vay", aliases: ["chi phí lãi vay", "lãi vay", "interest expense", "trong đó: chi phí lãi vay"], kind: "line", maSo: "23", statementType: "PL", unit: "VND" },
  { key: "selling_expense", label: "Chi phí bán hàng", aliases: ["chi phí bán hàng", "selling expense"], kind: "line", maSo: "25", statementType: "PL", unit: "VND" },
  { key: "admin_expense", label: "Chi phí quản lý doanh nghiệp", aliases: ["chi phí quản lý doanh nghiệp", "chi phí quản lý", "admin expense", "g&a"], kind: "line", maSo: "26", statementType: "PL", unit: "VND" },
  { key: "operating_profit", label: "Lợi nhuận thuần từ HĐKD", aliases: ["lợi nhuận thuần từ hoạt động kinh doanh", "lợi nhuận hoạt động", "operating profit"], kind: "line", maSo: "30", statementType: "PL", unit: "VND" },
  { key: "other_income", label: "Thu nhập khác", aliases: ["thu nhập khác", "other income"], kind: "line", maSo: "31", statementType: "PL", unit: "VND" },
  { key: "ebt", label: "Lợi nhuận trước thuế", aliases: ["tổng lợi nhuận kế toán trước thuế", "lợi nhuận kế toán trước thuế", "lợi nhuận trước thuế", "lntt", "ebt", "profit before tax"], kind: "line", maSo: "50", statementType: "PL", unit: "VND" },
  { key: "current_tax", label: "Chi phí thuế TNDN hiện hành", aliases: ["chi phí thuế thu nhập doanh nghiệp hiện hành", "thuế tndn", "current tax"], kind: "line", maSo: "51", statementType: "PL", unit: "VND" },
  { key: "deferred_tax", label: "Chi phí thuế TNDN hoãn lại", aliases: ["chi phí thuế thu nhập doanh nghiệp hoãn lại", "chi phí thuế tndn hoãn lại", "lợi ích thuế thu nhập doanh nghiệp hoãn lại", "lợi ích thuế tndn hoãn lại", "thu nhập thuế thu nhập doanh nghiệp hoãn lại", "thu nhập thuế tndn hoãn lại", "chi phí thu nhập thuế thu nhập doanh nghiệp hoãn lại", "chi phí thu nhập thuế tndn hoãn lại", "thu nhập chi phí thuế thu nhập doanh nghiệp hoãn lại", "thu nhập chi phí thuế tndn hoãn lại", "chi phí lợi ích thuế tndn hoãn lại", "lợi ích chi phí thuế tndn hoãn lại", "thuế tndn hoãn lại", "deferred tax"], kind: "line", maSo: "52", statementType: "PL", unit: "VND" },
  { key: "net_income", label: "Lợi nhuận sau thuế", aliases: ["lợi nhuận sau thuế thu nhập doanh nghiệp", "lợi nhuận sau thuế tndn", "lợi nhuận sau thuế", "lnst", "lãi ròng", "net profit", "net income", "profit after tax"], kind: "line", maSo: "60", statementType: "PL", unit: "VND" },
];

export const LINE_METRICS: MetricDef[] = [...BS_METRICS, ...PL_METRICS];

export const RATIO_METRICS: MetricDef[] = [
  { key: "gross_margin", label: "Biên lợi nhuận gộp", aliases: ["biên lợi nhuận gộp", "biên lãi gộp", "gross margin"], kind: "ratio", formula: { numer: "20", denom: "10", percent: true }, unit: "%" },
  { key: "net_margin", label: "Biên lợi nhuận ròng", aliases: ["biên lợi nhuận ròng", "biên lợi nhuận sau thuế", "ros", "net margin"], kind: "ratio", formula: { numer: "60", denom: "10", percent: true }, unit: "%" },
  { key: "operating_margin", label: "Biên lợi nhuận hoạt động", aliases: ["biên lợi nhuận hoạt động", "operating margin"], kind: "ratio", formula: { numer: "30", denom: "10", percent: true }, unit: "%" },
  { key: "roe", label: "ROE", aliases: ["roe", "tỷ suất lợi nhuận trên vốn chủ", "return on equity"], kind: "ratio", formula: { numer: "60", denom: "400", percent: true }, unit: "%" },
  { key: "roa", label: "ROA", aliases: ["roa", "tỷ suất lợi nhuận trên tài sản", "return on assets"], kind: "ratio", formula: { numer: "60", denom: "270", percent: true }, unit: "%" },
  { key: "de_ratio", label: "Nợ trên Vốn chủ sở hữu (D/E)", aliases: ["nợ trên vốn chủ sở hữu", "d/e", "de ratio", "debt to equity", "hệ số nợ trên vốn chủ"], kind: "ratio", formula: { numer: "300", denom: "400" }, unit: "x" },
  { key: "debt_to_assets", label: "Nợ trên Tổng tài sản", aliases: ["nợ trên tổng tài sản", "hệ số nợ", "debt to assets"], kind: "ratio", formula: { numer: "300", denom: "270", percent: true }, unit: "%" },
  { key: "current_ratio", label: "Hệ số thanh toán hiện hành", aliases: ["thanh toán hiện hành", "hệ số thanh toán hiện hành", "current ratio"], kind: "ratio", formula: { numer: "100", denom: "310" }, unit: "x" },
];

export const ALL_METRICS: MetricDef[] = [...LINE_METRICS, ...RATIO_METRICS];

/** Tập Mã số dùng bởi registry (để gửi roster gọn cho multi-company). */
export const REGISTRY_CODES: string[] = Array.from(
  new Set<string>([
    ...LINE_METRICS.map((m) => m.maSo as string),
    ...RATIO_METRICS.flatMap((m) => [m.formula!.numer, m.formula!.denom]),
  ]),
);

/**
 * Chuẩn hoá tên chỉ tiêu trước khi so khớp: bỏ dấu tiếng Việt, hạ chữ thường, bỏ ký tự
 * không phải chữ/số, gộp khoảng trắng.
 *
 * Vì sao cần: `findMetric` cũ so khớp CHÍNH XÁC nên trượt ở những gì model hay trả về —
 * đo được 5/14 biến thể hợp lệ bị bỏ ("doanh thu thuan" không dấu, "doanh thu  thuần" hai
 * dấu cách, "gross margin %"). Trượt thì người dùng nhận "chưa hỗ trợ chỉ số" dù dữ liệu có.
 */
export const normKey = (s: string) =>
  s
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/[đĐ]/g, "d")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();

export function findMetric(key: string): MetricDef | undefined {
  const k = normKey(key);
  if (!k) return undefined;
  return ALL_METRICS.find((m) => normKey(m.key) === k) || ALL_METRICS.find((m) => m.aliases.some((a) => normKey(a) === k));
}

/** Các loại báo cáo (BS/PL/CF) cần thiết cho một metric — dùng để union bảng khi retrieve (tối đa F2). */
export function sourceStatements(key: string): StatementType[] {
  const m = findMetric(key);
  if (!m) return [];
  if (m.kind === "line") return m.statementType ? [m.statementType] : [];
  if (m.formula) {
    const set = new Set<StatementType>();
    for (const code of [m.formula.numer, m.formula.denom]) {
      const lm = LINE_METRICS.find((x) => x.maSo === code);
      if (lm?.statementType) set.add(lm.statementType);
    }
    return [...set];
  }
  return [];
}

/** Danh mục gọn để nhét vào prompt (tên metric hợp lệ + Mã số). */
export function registryForPrompt(): string {
  const line = LINE_METRICS.map((m) => `${m.key}(Mã số ${m.maSo}, ${m.label})`).join(", ");
  const ratio = RATIO_METRICS.map((m) => `${m.key}(${m.label})`).join(", ");
  return `LINE: ${line}\nRATIO: ${ratio}`;
}
