import { resolveTickers, stripVN } from "./tickers";
import { LINE_METRICS, RATIO_METRICS, sourceStatements, type MetricDef, type StatementType } from "./registry";

export interface Understanding {
  tickers: string[];
  years: number[];
  metric?: string; // metric.key
  statementTypes: StatementType[];
}

const CURRENT_YEAR_DEFAULT = 2024;

/** Trích các năm: tuyệt đối, range, tương đối ("năm ngoái", "3 năm gần nhất"). */
function extractYears(question: string, currentYear: number): number[] {
  const t = stripVN(question);
  const years = new Set<number>();

  const range = t.match(/(20\d{2})\s*(?:-|den|toi|to)\s*(20\d{2})/);
  if (range) {
    let a = +range[1];
    let b = +range[2];
    if (a > b) [a, b] = [b, a];
    for (let y = a; y <= b; y++) years.add(y);
  }
  for (const m of t.matchAll(/(?:19|20)\d{2}/g)) years.add(+m[0]);

  if (/nam ngoai|nam truoc/.test(t)) years.add(currentYear - 1);
  if (/nam nay|hien tai/.test(t)) years.add(currentYear);
  const nRecent = t.match(/(\d+)\s*nam\s*(?:gan nhat|gan day|qua|tro lai day)/);
  if (nRecent) {
    const n = Math.min(+nRecent[1], 10);
    for (let i = 0; i < n; i++) years.add(currentYear - i);
  }
  return [...years].sort((a, b) => a - b);
}

/** Khớp metric qua alias/label (ratio ưu tiên vì cụ thể hơn line). */
function matchMetric(question: string): MetricDef | undefined {
  const t = stripVN(question);
  for (const m of [...RATIO_METRICS, ...LINE_METRICS]) {
    if (t.includes(stripVN(m.label))) return m;
    for (const a of m.aliases) {
      const na = stripVN(a);
      if (na.length >= 3 && t.includes(na)) return m;
    }
  }
  return undefined;
}

/** Query Understanding: trích ticker + năm + metric + loại báo cáo cần thiết. */
export function understand(question: string, currentYear = CURRENT_YEAR_DEFAULT): Understanding {
  const tickers = resolveTickers(question);
  const years = extractYears(question, currentYear);
  const metric = matchMetric(question);
  const statementTypes = metric ? sourceStatements(metric.key) : [];
  return { tickers, years, metric: metric?.key, statementTypes };
}
