import * as XLSX from "xlsx";

export interface ParsedWorkbook {
  fileName: string;
  sheetNames: string[];
  sheets: Record<string, Record<string, unknown>[]>;
  /**
   * Ma trận thô từng sheet (mảng dòng × ô), đọc với `raw: false` — mọi ô là CHUỖI đúng như hiện
   * trên màn hình Excel.
   *
   * Vì sao cần thêm, không thay `sheets`:
   *  1. `sheet_to_json` mặc định ép DÒNG ĐẦU làm tên cột. File thật hay có 1-2 dòng tiêu đề
   *     ("BÁO CÁO TÀI CHÍNH", "Đơn vị: triệu đồng") phía trên header thật ⇒ toàn bộ tên cột hỏng.
   *     Có ma trận thì dò được dòng header thật.
   *  2. Mặc định SheetJS đọc ô `01` thành số `1`, mất số 0 đứng đầu vĩnh viễn. Registry khoá theo
   *     CHUỖI ("01" = Doanh thu bán hàng, "02" = Các khoản giảm trừ) nên `"1" !== "01"` ⇒ tra
   *     trượt cả nhóm Mã số một chữ số. `raw: false` lấy trường `w` (văn bản đã định dạng) nên
   *     giữ nguyên "01" — đọc đúng thứ file ghi, không phải đoán rồi chèn số 0.
   * Mọi ô thành chuỗi không phải vấn đề: `cleanNumber` vốn nhận chuỗi, và điều này khớp với
   * `dtype=str` mà mã Pandas sinh ra dùng, nên hai bên cho cùng kết quả.
   */
  matrices: Record<string, string[][]>;
}

/** Đọc workbook (.xlsx/.csv) từ File ở client. */
export async function parseWorkbookFromFile(file: File): Promise<ParsedWorkbook> {
  const buf = await file.arrayBuffer();
  return parseWorkbook(new Uint8Array(buf), file.name);
}

export function parseWorkbook(data: Uint8Array, fileName: string): ParsedWorkbook {
  const isCsv = /\.csv$/i.test(fileName);
  const wb = isCsv
    ? XLSX.read(new TextDecoder("utf-8").decode(data), { type: "string" })
    : XLSX.read(data, { type: "array" });
  const sheets: Record<string, Record<string, unknown>[]> = {};
  const matrices: Record<string, string[][]> = {};
  for (const name of wb.SheetNames) {
    sheets[name] = XLSX.utils.sheet_to_json(wb.Sheets[name], { defval: null });
    // blankrows:false bỏ dòng trống hoàn toàn — dòng phân cách trong file thật, không mang tin.
    const aoa = XLSX.utils.sheet_to_json<unknown[]>(wb.Sheets[name], {
      header: 1,
      raw: false,
      defval: "",
      blankrows: false,
    });
    matrices[name] = aoa.map((row) => (row ?? []).map((c) => (c == null ? "" : String(c))));
  }
  return { fileName, sheetNames: wb.SheetNames, sheets, matrices };
}
