import { buildMockRepo, retrieve } from "./retrieval";
import { TICKERS } from "./tickers";

interface EvalCase {
  question: string;
  expected: string[]; // "TICKER|STMT|YEAR"
}

/** Bộ câu hỏi tổng hợp (thay bằng test set thật của BTC khi có). */
function makeCases(): EvalCase[] {
  const cases: EvalCase[] = [];
  const sample = TICKERS.slice(0, 12);
  for (const { ticker, companyName } of sample) {
    cases.push({ question: `Tổng tài sản của ${companyName} năm 2024?`, expected: [`${ticker}|BS|2024`] });
    cases.push({ question: `Doanh thu thuần ${companyName} năm 2024?`, expected: [`${ticker}|PL|2024`] });
    cases.push({ question: `ROE của ${companyName} năm 2024?`, expected: [`${ticker}|BS|2024`, `${ticker}|PL|2024`] });
    cases.push({
      question: `Doanh thu ${companyName} giai đoạn 2020-2024?`,
      expected: [2020, 2021, 2022, 2023, 2024].map((y) => `${ticker}|PL|${y}`),
    });
  }
  cases.push({
    question: `So sánh doanh thu 2024 của ${sample[0].companyName} và ${sample[1].companyName}?`,
    expected: [`${sample[0].ticker}|PL|2024`, `${sample[1].ticker}|PL|2024`],
  });
  return cases;
}

/**
 * F2 (β=2): recall trọng số gấp 4 lần precision (β²=4).
 * KHỚP công thức thể lệ Stage 2: F2 = (5·P·R)/(4·P+R), tính PER-QUERY.
 * Macro = trung bình F2 của từng truy vấn (xem runRetrievalEval), KHÔNG phải
 * F2 tính lại từ macro-P/macro-R. (Cách "F2 của số gộp" bị bác bởi số Stage 1:
 * vd Articles P=0.565,R=0.722 cho F2=0.684 nhưng bảng ghi 0.655 → phải là
 * per-query-rồi-trung-bình.) Giữ nguyên interpretation này.
 */
export function f2Score(pred: Set<string>, exp: Set<string>): { f2: number; precision: number; recall: number } {
  if (!exp.size) return { f2: pred.size === 0 ? 1 : 0, precision: 0, recall: 0 };
  let tp = 0;
  for (const e of exp) if (pred.has(e)) tp++;
  const precision = pred.size ? tp / pred.size : 0;
  const recall = tp / exp.size;
  const b2 = 4;
  const f2 = precision + recall > 0 ? ((1 + b2) * precision * recall) / (b2 * precision + recall) : 0;
  return { f2, precision, recall };
}

/** Chạy eval retrieval trên mock repo → F2 macro + chi tiết. (Pure, không gọi LLM.) */
export function runRetrievalEval() {
  const years = [2020, 2021, 2022, 2023, 2024];
  const tickers = TICKERS.slice(0, 12).map((t) => t.ticker);
  const repo = buildMockRepo(tickers, years);
  const cases = makeCases();

  let sumF2 = 0;
  let sumP = 0;
  let sumR = 0;
  const details = [];
  for (const c of cases) {
    const { docs, understanding } = retrieve(c.question, repo, { currentYear: 2024 });
    const pred = new Set(docs.map((d) => `${d.ticker}|${d.statementType}|${d.year}`));
    const exp = new Set(c.expected);
    const { f2, precision, recall } = f2Score(pred, exp);
    sumF2 += f2;
    sumP += precision;
    sumR += recall;
    details.push({
      q: c.question,
      f2: +f2.toFixed(3),
      precision: +precision.toFixed(3),
      recall: +recall.toFixed(3),
      predicted: pred.size,
      expected: exp.size,
      understood: { tickers: understanding.tickers, years: understanding.years, metric: understanding.metric },
    });
  }
  const n = cases.length;
  return {
    repoDocs: repo.docs.length,
    count: n,
    macroF2: +(sumF2 / n).toFixed(4),
    macroPrecision: +(sumP / n).toFixed(4),
    macroRecall: +(sumR / n).toFixed(4),
    details,
  };
}
