/** Một chỉ tiêu (dòng) đã chuẩn hoá, key theo Mã số. */
export interface TidyItem {
  maSo: string;
  name: string;
  /** period (vd "2024", "2023", "Q4_2024") -> giá trị đã làm sạch */
  values: Record<string, number | null>;
  /** vị trí dòng trong sheet gốc (1-based, cho citation) */
  rowIndex: number;
}

/** Bảng tài chính đã chuẩn hoá từ 1 sheet. */
export interface TidyTable {
  tickers: string[];
  periods: string[];
  unit: string;
  sheetName: string;
  /** tên file CSV gốc (để header read_csv trong Pandas chạy được đúng file) */
  fileName?: string;
  /** ticker -> maSo -> item */
  byTicker: Record<string, Record<string, TidyItem>>;
  /** cảnh báo dữ liệu bẩn đã tự xử lý (nếu có) */
  cleaningNotes: string[];
  /**
   * Báo cáo của bộ tiền xử lý (`preprocess.ts`) — hiển thị ở panel dữ liệu VÀ được agent
   * "Chuẩn hoá dữ liệu" thuật lại trên timeline. Một nguồn sự thật cho cả hai nơi.
   */
  prep?: {
    /** dòng được lấy làm header (0-based); >0 nghĩa là đã bỏ qua dòng tiêu đề rác */
    headerRow: number;
    /** Mã số đến từ đâu — quyết định cách `pandasLookup` sinh điều kiện lọc */
    maSoSource: "column" | "inferred" | "corrected" | "none";
    /** đã nhận ra bao nhiêu chỉ tiêu trên tổng số dòng có nội dung */
    coverage: { mapped: number; total: number };
    /**
     * Hệ số quy đổi đơn vị của cả sheet ("Đơn vị tính: triệu đồng" → 1e6). Mọi giá trị trong
     * `byTicker` đã nhân sẵn hệ số này; ghi ra đây để mã Pandas nhân lại cho khớp — nó đọc file
     * GỐC nên thấy số chưa quy đổi.
     */
    unitScale: number;
    notes: { level: "info" | "warn"; text: string }[];
  };
  /**
   * Vì sao bảng rỗng, nói bằng lời người đọc hiểu. Có giá trị khi file không dựng được bảng —
   * để UI và câu trả lời trong khung chat nêu đúng thứ còn thiếu thay vì im lặng.
   */
  fatal?: string | null;
  /**
   * Dữ liệu đến từ KHO trích sẵn (không phải file người dùng nạp) — mang theo mã chứng khoán.
   * Có cờ này thì khi tra chỉ tiêu không có trong 30 chỉ tiêu của kho, agent được phép tra
   * TRỰC TIẾP trên báo cáo gốc (xem `live-lookup.ts`). Với file người dùng nạp thì KHÔNG, vì
   * dữ liệu của họ không nằm trong corpus ViFinQA.
   */
  fromCorpus?: { ticker: string };
  /** tên cột gốc trong sheet (để sinh Pandas khớp file thật) */
  meta: { maSoKey: string; tickerKey?: string; nameKey?: string; periodKeys: Record<string, string> };
}

/** Sự kiện tiến trình từng bước của agent (stream về client). */
export interface StepData {
  id: string;
  label: string;
  status: "running" | "done" | "error";
  /** Tóm tắt 1 dòng dạng người đọc (ý định/kết quả) — KHÔNG chứa mã/công thức. */
  detail?: string;
  /** Chi tiết phụ dạng gạch đầu dòng (vd danh sách số liệu đã dùng) — cũng không chứa mã. */
  items?: string[];
}

/** Kế hoạch truy vấn do LLM sinh (đã validate). */
export interface QueryPlan {
  intent: "direct_retrieval" | "yoy_growth" | "profit_margin" | "ratio" | "multi_company" | "unknown";
  metric: string;
  years: string[];
  tickers: string[];
}

export interface Citation {
  ticker: string;
  maSo: string;
  itemName: string;
  period: string;
  sheet: string;
}

/** Kết quả agent trả về cho UI. */
export interface AgentResult {
  ok: boolean;
  answer: string; // câu trả lời tiếng Việt
  value: number | null;
  valueLabel: string; // đã format (vd "125,12 tỷ" hoặc "25.6%")
  unit: string;
  citations: Citation[];
  pandasCode: string;
  plan: QueryPlan;
  provider: string; // model đã dùng, vd "qwen3.5-4b"
  error?: string;
  /** Chart gpt-vis do agent sinh khi câu hỏi mang tính trực quan (deterministic data). */
  chart?: { type: string; height: number; config: Record<string, unknown> } | null;
  /** Nhận định ngắn kiểu analyst (LLM). */
  insight?: string | null;
  /**
   * Giá trị lấy bằng cách tra TRỰC TIẾP trên báo cáo gốc, KHÔNG có đẳng thức kế toán xác nhận.
   * Khâu chọn dòng ở đó đo được 67,6% nên giao diện PHẢI nói rõ mức tin cậy khác với số lấy từ
   * kho đã kiểm chứng. Thiếu cờ này thì người xem tưởng hai tầng như nhau.
   */
  unverified?: boolean;
  /** Nguồn của số chưa kiểm chứng: tên báo cáo gốc + dòng, để trích dẫn được. */
  liveSource?: { doc: string; line: number | null; label: string } | null;
  /** Cảnh báo dữ liệu vi phạm đẳng thức kế toán (validateTicker) — luôn hiển thị xác định ở UI,
   * không phụ thuộc LLM có nhắc tới hay không. */
  dataQualityWarning?: string | null;
  /** Cảnh báo ĐÁP ÁN nằm ngoài miền giá trị hợp lệ suy từ câu hỏi (auditAnswer).
   * Khác `dataQualityWarning`: cái kia nói dữ liệu VÀO bẩn, cái này nói kết quả RA sai. */
  auditWarning?: string | null;
  /**
   * Đánh dấu kết quả là một phép SO SÁNH HAI KỲ có chiều — do `computeAnswer` đặt, KHÔNG suy lại
   * từ `plan.intent`.
   *
   * Vì sao phải có trường này: nhánh tính YoY chạy khi `intent==="yoy_growth" HOẶC years.length>=2`,
   * nhưng bộ kiểm chiều của insight lại chỉ bật khi `intent==="yoy_growth"`. Câu "doanh thu 2023 và
   * 2024" cho planner trả intent=direct_retrieval + 2 năm ⇒ vẫn ra % có dấu mà **bộ kiểm chiều bị
   * tắt** ⇒ model tự do viết ngược chiều. Ghi sự thật ra đây một lần, mọi nơi khác đọc chung.
   */
  comparison?: { from: string; to: string; direction: "up" | "down" } | null;
}
