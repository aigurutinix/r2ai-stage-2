import type { TidyTable } from "./types";

const S = 1e6; // triệu VND
const round = (x: number) => Math.round(x / S);

function val(tidy: TidyTable, ticker: string, maSo: string, period: string): number | null {
  return tidy.byTicker[ticker]?.[maSo]?.values[period] ?? null;
}

/** Cơ cấu nguồn vốn: Nợ phải trả (300) vs Vốn chủ sở hữu (400). */
export function capitalStructure(tidy: TidyTable, ticker: string, period: string) {
  const debt = val(tidy, ticker, "300", period);
  const eq = val(tidy, ticker, "400", period);
  if (debt == null || eq == null) return null;
  return [
    { category: "Nợ phải trả", value: round(Math.abs(debt)) },
    { category: "Vốn chủ sở hữu", value: round(Math.abs(eq)) },
  ];
}

/** Cầu nối lợi nhuận (profit bridge): Doanh thu → LN sau thuế. */
export function profitBridge(tidy: TidyTable, ticker: string, period: string) {
  const rev = val(tidy, ticker, "10", period);
  const gross = val(tidy, ticker, "20", period);
  const ebt = val(tidy, ticker, "50", period);
  const net = val(tidy, ticker, "60", period);
  if ([rev, gross, ebt, net].some((x) => x == null)) return null;
  return [
    { category: "Doanh thu", value: round(rev!) },
    { category: "Giá vốn", value: round(gross! - rev!) },
    { category: "LN gộp", isTotal: true },
    { category: "Chi phí HĐ", value: round(ebt! - gross!) },
    { category: "LN trước thuế", isTotal: true },
    { category: "Thuế", value: round(net! - ebt!) },
    { category: "LN sau thuế", isTotal: true },
  ];
}

/** Xu hướng biên lợi nhuận gộp & ròng theo kỳ (%). */
export function marginTrend(tidy: TidyTable, ticker: string) {
  const periods = [...tidy.periods].sort();
  const out: { time: string; value: number; group: string }[] = [];
  for (const p of periods) {
    const rev = val(tidy, ticker, "10", p);
    const gross = val(tidy, ticker, "20", p);
    const net = val(tidy, ticker, "60", p);
    if (rev && rev !== 0) {
      if (gross != null) out.push({ time: p, value: Math.round((gross / rev) * 1000) / 10, group: "Biên LN gộp" });
      if (net != null) out.push({ time: p, value: Math.round((net / rev) * 1000) / 10, group: "Biên LN ròng" });
    }
  }
  return out.length ? out : null;
}

/** Khả năng sinh lời: ROE, ROA (%) kỳ mới nhất. */
export function profitability(tidy: TidyTable, ticker: string, period: string) {
  const net = val(tidy, ticker, "60", period);
  const eq = val(tidy, ticker, "400", period);
  const assets = val(tidy, ticker, "270", period);
  const out: { category: string; value: number }[] = [];
  if (net != null && eq) out.push({ category: "ROE", value: Math.round((net / eq) * 1000) / 10 });
  if (net != null && assets) out.push({ category: "ROA", value: Math.round((net / assets) * 1000) / 10 });
  return out.length ? out : null;
}
