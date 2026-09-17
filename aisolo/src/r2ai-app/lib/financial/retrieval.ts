import { understand, type Understanding } from "./query-understanding";
import { stripVN, tickerInfo } from "./tickers";
import { LINE_METRICS, type StatementType } from "./registry";

/** Một "bảng" trong kho (1 doanh nghiệp × 1 năm × 1 loại báo cáo). */
export interface TableDoc {
  id: string; // "HPG_2024_BS"
  ticker: string;
  companyName: string;
  aliases: string[];
  year: number;
  statementType: StatementType;
  terms: string[]; // tên chỉ tiêu / từ khoá cho lexical scoring
}

export interface RepoIndex {
  docs: TableDoc[];
}

export interface RetrieveOpts {
  currentYear?: number;
  topK?: number; // trần khi không lọc được metadata (rộng cho F2)
  minScore?: number;
}

export interface RetrieveResult {
  docs: TableDoc[];
  understanding: Understanding;
}

function lexScore(queryTerms: string[], docTerms: string[]): number {
  if (!queryTerms.length) return 0;
  const set = docTerms.map(stripVN);
  let hit = 0;
  for (const q of queryTerms) {
    if (set.some((d) => d === q || d.includes(q))) hit++;
  }
  return hit / queryTerms.length;
}

/**
 * Table Retrieval (skeleton, data-independent):
 * metadata filter (ticker/year) → union bảng theo công thức → lexical rank → OVER-RETRIEVE (tối đa F2).
 * Dense embedding + reranker (Python) sẽ bổ sung khi có corpus thật.
 */
export function retrieve(question: string, repo: RepoIndex, opts: RetrieveOpts = {}): RetrieveResult {
  const currentYear = opts.currentYear ?? 2024;
  const topK = opts.topK ?? 20;
  const u = understand(question, currentYear);

  // 1) Metadata filter cứng
  let cands = repo.docs;
  if (u.tickers.length) cands = cands.filter((d) => u.tickers.includes(d.ticker));
  if (u.years.length) cands = cands.filter((d) => u.years.includes(d.year));

  // 2) Union bảng theo công thức (statementTypes cần cho metric); recall-safe fallback nếu rỗng
  const wantStmts = u.statementTypes.length ? u.statementTypes : (["BS", "PL", "CF"] as StatementType[]);
  let byStmt = cands.filter((d) => wantStmts.includes(d.statementType));
  if (!byStmt.length) byStmt = cands;

  // 3) Lexical scoring
  const qTerms = stripVN(question)
    .split(/\s+/)
    .filter((w) => w.length >= 3);
  const scored = byStmt.map((d) => ({ d, s: lexScore(qTerms, d.terms) }));
  scored.sort((a, b) => b.s - a.s);

  // 4) Over-retrieve: nếu đã thu hẹp bằng metadata → giữ HẾT (ưu tiên recall/F2);
  //    nếu không có ticker/year → chỉ giữ topK theo score.
  const out =
    u.tickers.length || u.years.length
      ? scored.map((x) => x.d)
      : scored.filter((x) => x.s > (opts.minScore ?? 0)).slice(0, topK).map((x) => x.d);

  return { docs: out, understanding: u };
}

/** Dựng mock repo (BS+PL cho mỗi ticker×year) — để test skeleton trước khi có data BTC. */
export function buildMockRepo(tickers: string[], years: number[]): RepoIndex {
  const docs: TableDoc[] = [];
  for (const ticker of tickers) {
    const info = tickerInfo(ticker);
    for (const year of years) {
      for (const stmt of ["BS", "PL"] as StatementType[]) {
        const terms = LINE_METRICS.filter((m) => m.statementType === stmt).map((m) => m.label);
        docs.push({
          id: `${ticker}_${year}_${stmt}`,
          ticker,
          companyName: info?.companyName ?? ticker,
          aliases: info?.aliases ?? [],
          year,
          statementType: stmt,
          terms,
        });
      }
    }
  }
  return { docs };
}
