/**
 * Kho doanh nghiệp niêm yết trích sẵn từ bộ dữ liệu ViFinQA.
 *
 * Vì sao tiền tính offline thay vì tra lúc hỏi: tra lúc hỏi nghĩa là thừa kế khâu chọn dòng của
 * pipeline thi (đúng 67,6%), tức demo sai một phần ba số câu. Trích một lần ngoại tuyến thì sai
 * sót vẫn còn nhưng xảy ra nơi đo và sửa được, còn lúc hỏi chỉ là một phép tra khoá chính.
 *
 * Sinh dữ liệu: `python pipeline/build_corpus.py`. Đầu ra KHÔNG nằm trong repo (nội dung phái sinh
 * từ corpus CC BY-NC), nên thư mục có thể vắng — mọi hàm ở đây phải chịu được điều đó.
 */

/** Một dòng trong `corpus/index.json`. */
export interface CorpusEntry {
  ticker: string;
  /** số niên độ đọc được */
  periods: number;
  /** số chỉ tiêu registry trích được */
  items: number;
  cells: number;
  /** số niên độ THOẢ mọi đẳng thức kế toán kiểm được (Tổng TS = Nợ + VCSH, …) */
  verifiedYears: number;
  /** đủ tin để đưa lên demo: ≥3 niên độ đã qua kiểm */
  inScope: boolean;
}

/** Kết quả kiểm đẳng thức của MỘT niên độ. */
export interface YearCheck {
  testable: number;
  passed: number;
  ok: boolean;
}

export interface CorpusMeta {
  ticker: string;
  periods: string[];
  /** niên độ → tên báo cáo gốc, để trích dẫn nguồn */
  sourceDoc: Record<string, string>;
  verified: Record<string, YearCheck>;
}

export const corpusCsvUrl = (ticker: string) => `/corpus/${ticker}.csv`;

/**
 * Danh mục doanh nghiệp. Trả mảng rỗng khi chưa sinh kho — gọi ở màn hình rỗng nên KHÔNG được
 * ném lỗi làm hỏng cả trang; không có kho thì đơn giản là không hiện mục tra cứu.
 */
export async function loadCorpusIndex(): Promise<CorpusEntry[]> {
  try {
    const res = await fetch("/corpus/index.json");
    if (!res.ok) return [];
    const data = (await res.json()) as { tickers?: CorpusEntry[] };
    return data.tickers ?? [];
  } catch {
    return [];
  }
}

/** Siêu dữ liệu một doanh nghiệp (nguồn + kết quả kiểm đẳng thức). `null` nếu không có. */
export async function loadCorpusMeta(ticker: string): Promise<CorpusMeta | null> {
  try {
    const res = await fetch(`/corpus/${ticker}.json`);
    if (!res.ok) return null;
    return (await res.json()) as CorpusMeta;
  } catch {
    return null;
  }
}

/**
 * Câu tóm tắt mức độ tin cậy, dùng nguyên văn cho cả toast lẫn thẻ dữ liệu.
 *
 * Nói bằng con số đo được, không bằng tính từ: "9/11 niên độ đã qua kiểm cân đối" kiểm chứng được,
 * còn "dữ liệu đáng tin" thì không.
 */
export function verifiedSummary(meta: CorpusMeta | null): string | null {
  if (!meta) return null;
  const years = Object.keys(meta.verified);
  if (!years.length) return null;
  const ok = years.filter((y) => meta.verified[y]?.ok);
  return `${ok.length}/${years.length} niên độ đã qua kiểm cân đối kế toán`;
}
