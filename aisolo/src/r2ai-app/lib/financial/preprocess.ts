/**
 * TIỀN XỬ LÝ DỮ LIỆU — làm khâu nạp file chịu được CSV/Excel đời thật.
 *
 * Vì sao tồn tại: `normalize()` chỉ nhận đúng MỘT dáng file (header ở dòng 1, có cột Mã số, cột kỳ
 * mang chữ số năm trong tên). Ngoài dáng đó nó trả bảng rỗng, và người dùng rơi vào ngõ cụt: nạp
 * file xong hỏi gì cũng không có phản hồi. File thật thì muôn hình vạn trạng — có dòng tiêu đề rác
 * phía trên, có file chỉ ghi tên chỉ tiêu mà không có cột Mã số, có file dùng mã lệch chuẩn TT200.
 *
 * NGUYÊN TẮC — kế thừa từ Auditor (`audit.ts`), đừng phá:
 *   **Thà im lặng còn hơn đoán sai.** Mọi bước ở đây chỉ phát biểu điều SUY ĐƯỢC từ dữ liệu, không
 *   đoán ý người dùng. Đo được trên 38 tên chỉ tiêu thật của file mẫu: khớp tên kiểu "chứa/dài nhất
 *   thắng" ánh xạ được nhiều hơn (31/38) nhưng SAI MÃ 9 dòng trong im lặng — "Hàng tồn kho ròng"
 *   bị gán mã 140 của "Hàng tồn kho", "Dự phòng giảm giá hàng tồn kho" cũng 140. Khớp BẰNG NHAU
 *   TUYỆT ĐỐI chỉ ánh xạ 20/38 nhưng **0 lỗi**. Chọn cái thứ hai: dòng không nhận ra vẫn nạp và
 *   hiện trong bảng, chỉ là không tra cứu được — và app nói rõ đã nhận ra bao nhiêu.
 *
 * Mọi bước đều ghi lại việc đã làm vào `notes`. Không bước nào được sửa dữ liệu trong im lặng —
 * đó là điều kiện để người dùng còn tin được con số.
 *
 * Chạy ở CLIENT, trước `normalize()`. Không thể đặt ở server: `app/api/agent/route.ts` chặn request
 * khi `tidy.tickers` rỗng, nên đúng những ca cần cứu thì server không bao giờ thấy.
 */
import { LINE_METRICS, normKey } from "./registry";

/** Vai trò một cột sau khi dò. */
export interface ColumnMap {
  /** chỉ số cột chứa Mã số (nếu file có) */
  maSo: number | null;
  /** chỉ số cột chứa tên chỉ tiêu */
  name: number | null;
  /** chỉ số cột chứa mã/tên công ty */
  ticker: number | null;
  /** các cột kỳ: chỉ số cột + nhãn kỳ đã chuẩn hoá + tên cột gốc */
  periods: { index: number; label: string; header: string }[];
}

/** Một việc bộ tiền xử lý đã làm, để thuật lại cho người dùng. */
export interface PreprocessNote {
  /** `info` = đã xử lý xong, `warn` = người dùng nên biết để đối chiếu */
  level: "info" | "warn";
  text: string;
}

export interface PreprocessResult {
  /** dòng được chọn làm header (0-based trong ma trận) */
  headerRow: number;
  /** tên các cột lấy từ dòng header */
  headers: string[];
  columns: ColumnMap;
  /** các dòng dữ liệu (sau header), mỗi dòng là mảng ô dạng chuỗi */
  rows: string[][];
  /** Mã số suy ra cho từng dòng dữ liệu — `null` nghĩa là KHÔNG dám đoán. */
  maSoPerRow: (string | null)[];
  /** Mã số đến từ đâu: cột sẵn có / suy từ tên / đã sửa vì lệch chuẩn TT200 */
  maSoSource: "column" | "inferred" | "corrected" | "none";
  /** đã nhận ra bao nhiêu chỉ tiêu trên tổng số dòng có tên */
  coverage: { mapped: number; total: number };
  notes: PreprocessNote[];
  /** Không dựng được bảng (thiếu cột kỳ, hoặc không có cả Mã số lẫn tên nhận ra được). */
  fatal: string | null;
}

// ---------- Bước 1: dò dòng header thật ----------

/**
 * Cột Mã số. KHÔNG dùng `\b` ở đuôi: `\b` xét theo [A-Za-z0-9_] nên sau ký tự có dấu ("số") nó
 * không bao giờ khớp — đo được: header "Mã số" trượt sạch trong khi "Ma_So" lại đạt. Cùng lớp bẫy
 * đã ghi ở `clean.ts` và `audit.ts`. Thay bằng: hết chuỗi, hoặc theo sau là ký tự KHÔNG phải chữ
 * (đủ để nhận "Mã số (TT200)", "Ma_So_Chi_Tieu" mà vẫn không nuốt một cột tên khác).
 */
const RE_MASO_CELL = /^(m[ãa][_\s]?s[ốo]|ma[_\s]?so|item[_\s]?code|code)(?=$|[^\p{L}])/iu;
const RE_NAME_CELL = /(item[_\s]?name|chi[_\s]?tieu|ch[ỉi][_\s]?ti[êe]u|indicator|kho[ảa]n\s?m[ụu]c|t[êe]n|name)/i;
const RE_TICKER_CELL = /(ticker|m[ãa][_\s]?ck|symbol|m[ãa]\s?c[ổo]\s?phi[ếe]u|company|doanh\s?nghi[ệe]p)/i;
const RE_YEAR = /(?:19|20)\d{2}/;

/** Bao nhiêu dòng đầu được xét làm header. File thật hiếm khi có quá ngần này dòng tiêu đề. */
const HEADER_SCAN_DEPTH = 8;

function scoreHeaderRow(cells: string[]): number {
  const filled = cells.map((c) => c.trim()).filter(Boolean);
  if (filled.length < 2) return -1;
  let score = 0;
  if (filled.some((c) => RE_MASO_CELL.test(c))) score += 3;
  if (filled.some((c) => RE_NAME_CELL.test(c))) score += 2;
  score += Math.min(3, filled.filter((c) => RE_YEAR.test(c)).length);
  if (filled.length >= 3) score += 2;
  // Dòng header là NHÃN, không phải dữ liệu: nếu quá nửa số ô đã là số thuần thì đây là dòng số
  // liệu bị chấm nhầm. Không có chặn này, một bảng không có tiêu đề sẽ lấy dòng số đầu làm header.
  const numeric = filled.filter((c) => /^[\d.,\s()+-]+$/.test(c)).length;
  if (numeric > filled.length / 2) score -= 4;
  return score;
}

/** Chọn dòng làm header. Hoà điểm thì lấy dòng TRÊN CÙNG — file chuẩn phải thắng file lạ. */
export function detectHeaderRow(matrix: string[][]): number {
  let best = 0;
  let bestScore = -Infinity;
  for (let i = 0; i < Math.min(HEADER_SCAN_DEPTH, matrix.length); i++) {
    const s = scoreHeaderRow(matrix[i] ?? []);
    if (s > bestScore) {
      bestScore = s;
      best = i;
    }
  }
  return best;
}

// ---------- Bước 2: ánh xạ cột ----------

/** Suy nhãn kỳ từ tên cột: "Year_2024"→"2024", "Q4_2024"→"Q4/2024", "31/12/2024"→"2024". */
export function periodLabelOf(key: string): string | null {
  const q = key.match(/q([1-4])[_\s-]?((?:19|20)\d{2})/i);
  if (q) return `Q${q[1]}/${q[2]}`;
  const y = key.match(RE_YEAR);
  if (y) return y[0];
  return null;
}

function mapColumns(headers: string[]): ColumnMap {
  const idx = (re: RegExp) => headers.findIndex((h) => re.test(h.trim()));
  const maSo = idx(RE_MASO_CELL);
  const nameCandidates = headers
    .map((h, i) => ({ h, i }))
    .filter(({ i }) => i !== maSo && RE_NAME_CELL.test(headers[i].trim()));
  const ticker = idx(RE_TICKER_CELL);

  const periods: ColumnMap["periods"] = [];
  const seen = new Map<string, number>();
  headers.forEach((h, i) => {
    const label = periodLabelOf(h);
    if (label == null) return;
    // Hai cột cùng nhãn kỳ (hợp nhất / công ty mẹ đặt cạnh nhau) là chuyện thường. Trước đây cột
    // sau ghi đè cột trước trong `periodKeys` nên MỘT CỘT BIẾN MẤT mà không dấu hiệu nào. Giữ cả
    // hai, phân biệt bằng hậu tố, để người dùng thấy và tự chọn.
    const n = (seen.get(label) ?? 0) + 1;
    seen.set(label, n);
    periods.push({ index: i, label: n === 1 ? label : `${label} (${n})`, header: h });
  });

  return {
    maSo: maSo >= 0 ? maSo : null,
    // Cột tên: ưu tiên cột KHÁC cột ticker, vì "Company Name" cũng khớp RE_NAME_CELL.
    name: (nameCandidates.find((c) => c.i !== ticker) ?? nameCandidates[0])?.i ?? null,
    ticker: ticker >= 0 ? ticker : null,
    periods,
  };
}

// ---------- Bước 3+4: Mã số theo tên chỉ tiêu ----------

/**
 * Index tên chỉ tiêu → Mã số, dựng từ registry (label + mọi alias).
 * CHỈ dùng cho khớp BẰNG NHAU tuyệt đối sau chuẩn hoá — xem ghi chú nguyên tắc ở đầu file.
 * Nhãn đầu tiên thắng khi trùng, để thứ tự registry là thứ tự quyết định.
 */
const NAME_TO_MASO: Map<string, string> = (() => {
  const m = new Map<string, string>();
  for (const metric of LINE_METRICS) {
    if (!metric.maSo) continue;
    for (const s of [metric.label, ...metric.aliases]) {
      const k = normKey(s);
      if (k && !m.has(k)) m.set(k, metric.maSo);
    }
  }
  return m;
})();

/** Mã số chuẩn TT200 cho một tên chỉ tiêu, hoặc `null` nếu không chắc chắn. */
export function maSoFromName(name: string): string | null {
  return NAME_TO_MASO.get(normKey(name)) ?? null;
}

// ---------- Bộ chuẩn hoá ----------

/**
 * Chạy toàn bộ chuỗi tiền xử lý trên ma trận thô của một sheet.
 * Không ném lỗi: mọi thất bại đều thành `fatal` + `notes` để tầng trên hiển thị được.
 */
export function preprocess(matrix: string[][]): PreprocessResult {
  const notes: PreprocessNote[] = [];
  const empty = (fatal: string): PreprocessResult => ({
    headerRow: 0,
    headers: [],
    columns: { maSo: null, name: null, ticker: null, periods: [] },
    rows: [],
    maSoPerRow: [],
    maSoSource: "none",
    coverage: { mapped: 0, total: 0 },
    notes,
    fatal,
  });

  if (!matrix.length) return empty("File hoặc sheet này không có dòng dữ liệu nào.");

  // 1 — dòng header thật
  const headerRow = detectHeaderRow(matrix);
  if (headerRow > 0) {
    notes.push({
      level: "info",
      text: `Bỏ qua ${headerRow} dòng tiêu đề phía trên, lấy dòng ${headerRow + 1} làm tên cột.`,
    });
  }
  const headers = (matrix[headerRow] ?? []).map((c) => String(c ?? "").trim());
  const body = matrix.slice(headerRow + 1);

  // 2 — ánh xạ cột
  const columns = mapColumns(headers);
  if (!columns.periods.length) {
    return empty(
      "Không tìm thấy cột số liệu theo kỳ. File cần ít nhất một cột có năm trong tên " +
        '(ví dụ "Year_2024", "Năm 2024", "31/12/2024").',
    );
  }
  const dup = columns.periods.filter((p) => /\(\d+\)$/.test(p.label));
  if (dup.length) {
    notes.push({
      level: "warn",
      text: `${dup.length} cột trùng nhãn kỳ — đã giữ cả hai và đánh số để phân biệt (${dup
        .map((p) => p.label)
        .join(", ")}).`,
    });
  }
  if (columns.maSo == null && columns.name == null) {
    return empty(
      "Không tìm thấy cột Mã số lẫn cột tên chỉ tiêu. File cần ít nhất một trong hai để tra cứu được.",
    );
  }

  // 3+4+5 — Mã số cho từng dòng
  const maSoPerRow: (string | null)[] = [];
  let mapped = 0;
  let total = 0;
  let corrected = 0;
  const correctedVd: string[] = [];

  for (const row of body) {
    const rawMa = columns.maSo != null ? String(row[columns.maSo] ?? "").trim() : "";
    const rawName = columns.name != null ? String(row[columns.name] ?? "").trim() : "";
    if (!rawMa && !rawName) {
      maSoPerRow.push(null);
      continue;
    }
    total++;
    const byName = rawName ? maSoFromName(rawName) : null;

    if (rawMa && byName && rawMa !== byName) {
      // 5 — file có CẢ hai cột và chúng mâu thuẫn. Tin theo TÊN: tên chỉ tiêu là thứ con người
      // đọc và kiểm được, còn mã số là quy ước dễ ghi lệch. Đo được trên chính file mẫu của dự án:
      // Tài sản cố định ghi 210 (chuẩn 220), Nợ dài hạn ghi 320 (chuẩn 330) — hậu quả là hỏi
      // "tài sản cố định" trả "không tìm thấy", và một luật kiểm cân đối kế toán không bao giờ chạy.
      maSoPerRow.push(byName);
      mapped++;
      corrected++;
      if (correctedVd.length < 3) correctedVd.push(`${rawMa}→${byName} "${rawName}"`);
      continue;
    }
    if (rawMa) {
      maSoPerRow.push(rawMa);
      mapped++;
      continue;
    }
    // Không có cột Mã số: chỉ nhận khi tên khớp tuyệt đối. Không khớp thì để null — dòng vẫn được
    // nạp và hiện trong bảng, chỉ là không tra cứu theo chỉ tiêu được.
    maSoPerRow.push(byName);
    if (byName) mapped++;
  }

  const maSoSource: PreprocessResult["maSoSource"] =
    columns.maSo == null ? "inferred" : corrected > 0 ? "corrected" : "column";

  if (maSoSource === "inferred") {
    notes.push({
      level: mapped === total ? "info" : "warn",
      text:
        `File không có cột Mã số — đã suy Mã số từ tên chỉ tiêu, nhận ra ${mapped}/${total} dòng. ` +
        (mapped < total
          ? `${total - mapped} dòng còn lại vẫn hiển thị nhưng chưa tra cứu theo tên được.`
          : ""),
    });
  }
  if (corrected > 0) {
    notes.push({
      level: "warn",
      text:
        `${corrected} dòng có Mã số lệch chuẩn Thông tư 200 — đã dùng mã theo tên chỉ tiêu ` +
        `(${correctedVd.join("; ")}${corrected > 3 ? "…" : ""}).`,
    });
  }

  return {
    headerRow,
    headers,
    columns,
    rows: body,
    maSoPerRow,
    maSoSource,
    coverage: { mapped, total },
    notes,
    fatal: null,
  };
}
