import { cleanNumber, hasOwnUnit } from "./clean";
import { preprocess } from "./preprocess";
import type { TidyItem, TidyTable } from "./types";

/** Phát hiện "Đơn vị tính: triệu/tỷ/nghìn đồng" từ tên cột hoặc các ô đầu bảng → hệ số quy đổi về VND. */
function detectUnitScale(candidates: string[]): { scale: number; label: string } {
  for (const c of candidates) {
    const t = c.toLowerCase();
    if (/đơn\s?vị|don\s?vi|\bunit\b/.test(t)) {
      if (/\btỷ\b|\bty\b|\btỉ\b|billion/.test(t)) return { scale: 1e9, label: "tỷ đồng" };
      if (/triệu|trieu|million/.test(t)) return { scale: 1e6, label: "triệu đồng" };
      if (/nghìn|nghin|ngàn|ngan|thousand/.test(t)) return { scale: 1e3, label: "nghìn đồng" };
    }
  }
  return { scale: 1, label: "VND" };
}

/**
 * Chuẩn hoá một sheet (ma trận ô dạng chuỗi) thành TidyTable khoá theo Mã số.
 *
 * Nhận MA TRẬN THÔ chứ không phải kết quả `sheet_to_json`: bộ tiền xử lý cần thấy các dòng phía
 * trên header để dò được header thật, và cần chuỗi nguyên bản để giữ Mã số "01" khỏi thành "1".
 *
 * Không còn cổng chặn cứng "phải có cột Mã số": file chỉ có tên chỉ tiêu vẫn nạp được, Mã số suy
 * từ tên (xem `preprocess.ts`). Khi thật sự không dựng được bảng thì trả bảng rỗng kèm `fatal` nói
 * rõ thiếu gì — để người dùng biết phải làm gì thay vì gặp một màn hình câm.
 */
export function normalize(matrix: string[][], sheetName: string): TidyTable {
  const cleaningNotes: string[] = [];
  const prep = preprocess(matrix);
  // `unitScale` điền sau, khi đã dò được đơn vị của sheet — mọi nơi giữ CÙNG một object nên
  // không phải dựng lại.
  const prepMeta: NonNullable<TidyTable["prep"]> = {
    headerRow: prep.headerRow,
    maSoSource: prep.maSoSource,
    coverage: prep.coverage,
    unitScale: 1,
    notes: prep.notes,
  };
  const empty: TidyTable = {
    tickers: [],
    periods: [],
    unit: "VND",
    sheetName,
    byTicker: {},
    cleaningNotes,
    prep: prepMeta,
    fatal: prep.fatal,
    meta: { maSoKey: "", periodKeys: {} },
  };
  if (prep.fatal) return empty;

  const { columns, headers, rows } = prep;
  const maSoKey = columns.maSo != null ? headers[columns.maSo] : "";
  const nameKey = columns.name != null ? headers[columns.name] : undefined;
  const tickerKey = columns.ticker != null ? headers[columns.ticker] : undefined;
  const periods = columns.periods.map((p) => p.label);

  // Nhãn đơn vị nằm ở tên cột HOẶC ở các dòng tiêu đề phía trên header ("Đơn vị tính: triệu đồng").
  // Trước đây chỉ quét 20 dòng ĐẦU BẢNG nên khi dòng đó nằm trên header thì không bao giờ thấy.
  const { scale, label: unitLabel } = detectUnitScale([
    ...headers,
    ...matrix.slice(0, prep.headerRow).flat(),
    ...rows.slice(0, 20).flat(),
  ]);
  prepMeta.unitScale = scale;
  const byTicker: Record<string, Record<string, TidyItem>> = {};
  let dirtyCount = 0;
  const dupNotes: string[] = [];

  rows.forEach((row, i) => {
    const maSo = prep.maSoPerRow[i];
    if (!maSo) return; // không có Mã số và không suy được — dòng này không tra cứu được
    const ticker = tickerKey != null && columns.ticker != null
      ? String(row[columns.ticker] ?? "").trim() || "(toàn bộ)"
      : "(toàn bộ)";
    const name = columns.name != null ? String(row[columns.name] ?? "").trim() || maSo : maSo;

    const values: Record<string, number | null> = {};
    for (const p of columns.periods) {
      const raw = row[p.index];
      const cleaned = cleanNumber(raw);
      // Ô tự khai đơn vị ("1,5 tỷ") thì cleanNumber đã quy đổi — áp thêm hệ số chung của sheet nữa
      // là NHÂN ĐÔI. Đo được: 80 tỷ thành 80 triệu tỷ khi sheet khai "Đơn vị tính: triệu đồng".
      values[p.label] = cleaned == null ? null : hasOwnUnit(raw) ? cleaned : cleaned * scale;
      // đánh dấu dữ liệu bẩn (có ký tự tiền tệ / ngoặc / phân cách bất thường)
      if (typeof raw === "string" && /[^\d\s.,()-]/.test(raw)) dirtyCount++;
    }

    byTicker[ticker] ??= {};
    // Mã số KHÔNG duy nhất trong một báo cáo: bảng lưu chuyển tiền tệ dùng lại dải 01–70 nên
    // mã 20 vừa là "Lợi nhuận gộp" (KQKD) vừa là "LCTT thuần từ HĐKD". File nguồn không có
    // cột loại báo cáo để phân biệt. Trước đây dòng sau ghi đè dòng trước KHÔNG BÁO GÌ —
    // đo được: hỏi "Lợi nhuận gộp" nhận về giá trị dòng lưu chuyển tiền, không dấu hiệu nào.
    // Giữ dòng ĐẦU (thứ tự BCTC là BS → KQKD → LCTT nên dòng đầu là báo cáo chính) và nêu rõ
    // dòng bị bỏ, để người dùng thấy được thay vì nhận số sai trong im lặng.
    const existing = byTicker[ticker][maSo];
    if (existing) {
      if (existing.name !== name) dupNotes.push(`Mã ${maSo}: dùng "${existing.name}", bỏ qua "${name}"`);
      return;
    }
    // rowIndex trỏ về dòng trong FILE GỐC (1-based), tính cả các dòng tiêu đề đã bỏ qua — nếu
    // không, trích dẫn sẽ chỉ sai chỗ ở đúng những file có dòng rác phía trên.
    byTicker[ticker][maSo] = { maSo, name, values, rowIndex: prep.headerRow + i + 2 };
  });

  if (scale !== 1) {
    cleaningNotes.push(`Đơn vị gốc: ${unitLabel} — đã quy đổi tất cả số về VND.`);
  }
  if (dirtyCount > 0) {
    cleaningNotes.push(`Đã tự làm sạch ${dirtyCount} ô có ký hiệu tiền tệ/định dạng bất thường (VND, $, ngoặc âm, phân cách nghìn).`);
  }
  if (dupNotes.length) {
    const head = dupNotes.slice(0, 3).join("; ");
    cleaningNotes.push(
      `${dupNotes.length} dòng trùng Mã số (khác loại báo cáo) — đã giữ dòng đầu: ${head}${dupNotes.length > 3 ? "…" : ""}`,
    );
  }

  const tickers = Object.keys(byTicker);
  const periodKeys: Record<string, string> = {};
  for (const p of columns.periods) periodKeys[p.label] = p.header;

  // Dựng được bảng nhưng không dòng nào tra cứu được: vẫn phải nói ra lý do, đừng để màn hình câm.
  const fatal =
    tickers.length === 0
      ? prep.coverage.total > 0
        ? `Đọc được ${prep.coverage.total} dòng nhưng không nhận ra chỉ tiêu nào theo chuẩn Thông tư 200. ` +
          "File cần cột Mã số, hoặc tên chỉ tiêu viết theo cách gọi chuẩn."
        : "Sheet này không có dòng số liệu nào đọc được."
      : null;

  return {
    tickers,
    periods,
    unit: "VND",
    sheetName,
    byTicker,
    cleaningNotes,
    prep: prepMeta,
    fatal,
    meta: { maSoKey, tickerKey, nameKey, periodKeys },
  };
}
