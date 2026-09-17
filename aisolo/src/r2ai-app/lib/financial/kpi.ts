import type { TidyTable } from "./types";
import { LINE_METRICS, RATIO_METRICS, type MetricDef } from "./registry";
import { formatMoney } from "./clean";

export interface Kpi {
  key: string;
  label: string;
  unit: string;
  value: number | null;
  valueLabel: string;
  deltaLabel: string | null;
  /** "pct" = tăng trưởng %, tô màu theo dấu; "pp" = chênh điểm %, trung tính */
  deltaKind: "pct" | "pp" | null;
  deltaSign: number; // 1/-1/0
}

/** Các KPI hiển thị mặc định + nhãn ngắn cho card. */
const KPI_LIST: { key: string; short: string }[] = [
  { key: "net_revenue", short: "Doanh thu thuần" },
  { key: "net_income", short: "LN sau thuế" },
  { key: "roe", short: "ROE" },
  { key: "roa", short: "ROA" },
  { key: "gross_margin", short: "Biên LN gộp" },
  { key: "de_ratio", short: "Nợ / VCSH" },
];

const ALL = [...LINE_METRICS, ...RATIO_METRICS];

function metricValue(tidy: TidyTable, ticker: string, m: MetricDef, period: string): number | null {
  if (m.kind === "line" && m.maSo) return tidy.byTicker[ticker]?.[m.maSo]?.values[period] ?? null;
  if (m.kind === "ratio" && m.formula) {
    const n = tidy.byTicker[ticker]?.[m.formula.numer]?.values[period] ?? null;
    const d = tidy.byTicker[ticker]?.[m.formula.denom]?.values[period] ?? null;
    if (n == null || d == null || d === 0) return null;
    let v = n / d;
    if (m.formula.percent) v *= 100;
    return Math.round(v * 100) / 100;
  }
  return null;
}

function fmtValue(v: number | null, m: MetricDef, unit: string): string {
  if (v == null) return "—";
  if (m.unit === "%") return `${v.toLocaleString("vi-VN", { maximumFractionDigits: 1 })}%`;
  if (m.unit === "x") return `${v.toLocaleString("vi-VN", { maximumFractionDigits: 2 })}x`;
  return formatMoney(v, unit);
}

export function computeKpis(tidy: TidyTable, ticker: string): Kpi[] {
  const p0 = tidy.periods[0];
  const p1 = tidy.periods[1];
  const out: Kpi[] = [];
  for (const { key, short } of KPI_LIST) {
    const m = ALL.find((x) => x.key === key);
    if (!m) continue;
    const v0 = metricValue(tidy, ticker, m, p0);
    const v1 = p1 ? metricValue(tidy, ticker, m, p1) : null;

    let deltaLabel: string | null = null;
    let deltaKind: Kpi["deltaKind"] = null;
    let deltaSign = 0;
    if (v0 != null && v1 != null) {
      if (m.unit === "%" || m.unit === "x") {
        const d = Math.round((v0 - v1) * 100) / 100;
        deltaKind = "pp";
        deltaSign = Math.sign(d);
        deltaLabel = `${d >= 0 ? "+" : ""}${d.toLocaleString("vi-VN", { maximumFractionDigits: 2 })}${m.unit === "%" ? " pp" : ""}`;
      } else if (v1 !== 0) {
        const g = Math.round(((v0 - v1) / Math.abs(v1)) * 10000) / 100;
        deltaKind = "pct";
        deltaSign = Math.sign(g);
        deltaLabel = `${g >= 0 ? "+" : ""}${g.toLocaleString("vi-VN", { maximumFractionDigits: 1 })}%`;
      }
    }
    out.push({ key, label: short, unit: m.unit ?? "", value: v0, valueLabel: fmtValue(v0, m, tidy.unit), deltaLabel, deltaKind, deltaSign });
  }
  return out;
}

/** Dữ liệu chart: doanh thu (Mã số 10) + LNST (Mã số 60) theo kỳ, sắp xếp tăng dần. */
export function chartSeries(tidy: TidyTable, ticker: string): { period: string; revenue: number | null; netIncome: number | null }[] {
  const periods = [...tidy.periods].sort();
  return periods.map((p) => ({
    period: p,
    revenue: tidy.byTicker[ticker]?.["10"]?.values[p] ?? null,
    netIncome: tidy.byTicker[ticker]?.["60"]?.values[p] ?? null,
  }));
}
