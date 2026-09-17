/**
 * AUDITOR — kiểm tra ĐÁP ÁN, deterministic, không gọi LLM.
 *
 * Port từ `pipeline/audit_wrong.py` (bản dự thi ViFinQA, nơi vòng Planner→Executor→Auditor đưa
 * Execution 0.2470 → 0.3103, tức đúng thêm 32 câu).
 *
 * NGUYÊN TẮC — đây là lý do nó đáng tin, đừng phá:
 *   Auditor chỉ phát biểu điều **suy được từ chính câu hỏi và định nghĩa chỉ tiêu**. Nó KHÔNG BAO
 *   GIỜ đoán đáp án đúng là bao nhiêu. Hỏi "hàng tồn kho" thì biết chắc kết quả không thể lớn hơn
 *   tổng tài sản — mà không cần biết con số đúng. Vì không tự đoán nên **nó không thể tự lừa mình**,
 *   khác hẳn mọi cách dùng LLM chấm LLM.
 *
 * KHÁC BIỆT CÓ CHỦ Ý so với bản pipeline: ở đó Auditor **bác rồi chạy lại** câu sai (đáng làm vì
 * planner là LLM, chạy lại cho kết quả khác). Ở app, bước truy xuất là deterministic — chạy lại
 * ra đúng số cũ, nên vòng lặp vô nghĩa. Thay vào đó **cảnh báo cho người dùng thấy**: người dùng
 * đang ngồi đó, biết được "số này nằm ngoài miền hợp lệ" giá trị hơn một vòng lặp câm.
 *
 * Đây KHÔNG phải `validate.ts`. `validate.ts` kiểm đẳng thức kế toán trên **dữ liệu đầu vào**
 * (270 = 300 + 400) để bắt file nguồn bẩn. File này kiểm **miền giá trị của đáp án đầu ra**.
 */
import { findMetric } from "./registry";
import { validateTicker } from "./validate";
import type { AgentResult, TidyTable } from "./types";

export interface AuditViolation {
  /** Tên luật, đủ ngắn để hiện trên trace. */
  rule: string;
  /** Câu giải thích cho người dùng cuối — nói VÌ SAO chắc chắn sai, không nói đáp án đúng là gì. */
  detail: string;
  /**
   * Vi phạm nằm ở ĐÂU. Đây là thứ agent Nhận định cần để biết phải làm gì:
   *   value   — bản thân CON SỐ ngoài miền hợp lệ  ⇒ đừng viết phân tích cho một số đã biết là sai
   *   answer  — câu trả lời tự sinh mô tả sai chiều ⇒ như trên
   *   insight — chính phần nhận định của LLM sai    ⇒ bỏ nhận định đó đi, đừng hiển thị kèm cảnh báo
   */
  source: "value" | "answer" | "insight";
}

/** Chỉ tiêu KHÔNG THỂ âm theo định nghĩa kế toán.
 *  Cố tình BỎ: equity/vốn chủ sở hữu (công ty âm vốn là có thật), mọi chỉ tiêu lợi nhuận
 *  (lỗ là có thật), nợ dài hạn (có thể bằng 0 nhưng không âm — vẫn giữ vì an toàn). */
const NON_NEGATIVE = new Set([
  "total_assets", "total_resources", "current_assets", "long_term_assets", "fixed_assets",
  "cash", "inventory", "short_term_investments", "short_term_receivables",
  "total_liabilities", "current_liabilities", "long_term_liabilities",
  "charter_capital", "gross_revenue", "net_revenue",
]);

/** Trần cho mọi chỉ số dạng %. Biên lợi nhuận, ROE, tăng trưởng đều có thể vượt 100% một cách
 *  hợp lệ, nên ngưỡng đặt rất rộng — vượt 10.000% thì gần như chắc chắn là lỗi đơn vị/parse
 *  (số tiền lọt vào ô phần trăm), không phải doanh nghiệp tăng trưởng phi thường. */
const PCT_LIMIT = 10_000;

/** Sai số cho phép khi so với tổng tài sản (số liệu OCR/làm tròn có thể lệch chút). */
const TA_TOLERANCE = 1.02;

/** Từ chỉ chiều, dùng để đối chiếu LỜI VĂN với DẤU của con số code đã tính.
 *  `\b` không dùng được với tiếng Việt có dấu trong JS regex mặc định nên dò theo cụm. */
const UP_WORDS = /tăng trưởng|tăng lên|tăng|cải thiện|mở rộng|gia tăng|đi lên|khởi sắc|cao hơn|nhiều hơn|vượt/i;
const DOWN_WORDS = /suy giảm|sụt giảm|giảm|thu hẹp|đi xuống|sa sút|thụt lùi|thấp hơn|ít hơn|kém hơn/i;

/**
 * Chiều được NÊU RA ĐẦU TIÊN trong câu — đó là điều người đọc tiếp nhận, kể cả khi câu sau có
 * từ chiều ngược lại ("doanh thu tăng nhưng biên lợi nhuận giảm" ⇒ chiều chính là TĂNG).
 * Luật cũ đòi "có từ giảm VÀ không có từ tăng" nên câu hai vế luôn lọt lưới.
 */
export function leadingDirection(text: string): "up" | "down" | null {
  const up = text.search(UP_WORDS);
  const down = text.search(DOWN_WORDS);
  if (up < 0 && down < 0) return null;
  if (up < 0) return "down";
  if (down < 0) return "up";
  return up < down ? "up" : "down";
}

/**
 * Có nêu khoảng năm NGƯỢC chiều thời gian không, vd "từ 2024 đến 2023", "2024 → 2023".
 *
 * CHỈ bắt các liên từ CHỈ HƯỚNG ĐI TỚI. Cố tình KHÔNG bắt "so với": "năm 2024 tăng 5% so với năm
 * 2023" là cách viết ĐÚNG và phổ biến nhất — đưa "so với" vào đây sẽ báo oan chính câu mà
 * `computeAnswer` tự sinh ra.
 */
export function hasReversedRange(text: string): boolean {
  const re = /(20\d{2})\s*(?:->|→|-|–|đến|tới|sang)\s*(20\d{2})/gi;
  for (const m of text.matchAll(re)) {
    if (parseInt(m[1], 10) > parseInt(m[2], 10)) return true;
  }
  return false;
}

/**
 * Soi một kết quả đã tính. Trả về danh sách vi phạm — RỖNG nghĩa là không phát hiện gì
 * (KHÔNG có nghĩa là đáp án đúng; Auditor chỉ loại, không xác nhận).
 */
export function auditAnswer(result: AgentResult, tidy: TidyTable): AuditViolation[] {
  const v = result.value;
  if (!result.ok || v == null || !Number.isFinite(v)) return [];

  const out: AuditViolation[] = [];
  const metric = findMetric(result.plan.metric);
  /**
   * Kết quả là HIỆU hoặc TỐC ĐỘ THAY ĐỔI ⇒ âm là bình thường, không áp luật dấu và luật trần.
   *
   * Suy từ `result.comparison` (sự thật do computeAnswer ghi lại) chứ KHÔNG từ `plan.intent`.
   * Test bắt được: câu "doanh thu 2023 và 2024" cho intent=direct_retrieval nhưng vẫn chạy nhánh
   * YoY ra −8,2% ⇒ nếu xét theo intent thì `isDelta` sai thành false và −8,2 bị gắn cờ oan
   * "doanh thu thuần không thể âm". Đúng cùng một gốc lỗi với bộ kiểm chiều của insight.
   */
  const isDelta = !!result.comparison || result.plan.intent === "multi_company";

  // 1 — Đơn vị phần trăm mà giá trị lớn bất thường ⇒ gần như chắc chắn số tiền lọt vào ô %.
  if (result.unit === "%" && Math.abs(v) > PCT_LIMIT) {
    out.push({
      rule: "Phần trăm ngoài miền hợp lệ",
      source: "value",
      detail: `Kết quả ${v.toLocaleString("vi-VN", { maximumFractionDigits: 0 })}% vượt xa mọi mức hợp lý cho một tỷ lệ — nhiều khả năng đơn vị bị đọc sai.`,
    });
  }

  // 2 — Hệ số thanh toán hiện hành = TS ngắn hạn / Nợ ngắn hạn, hai vế đều không âm ⇒ không thể âm.
  if (metric?.key === "current_ratio" && v < 0) {
    out.push({
      rule: "Hệ số thanh toán âm",
      source: "value",
      detail: "Hệ số thanh toán hiện hành ghép từ hai chỉ tiêu đều không âm nên không thể ra số âm.",
    });
  }

  // 3 — Chỉ tiêu không thể âm theo định nghĩa mà lại ra âm.
  if (!isDelta && v < 0 && metric && NON_NEGATIVE.has(metric.key)) {
    out.push({
      rule: "Giá trị âm cho chỉ tiêu không thể âm",
      source: "value",
      detail: `"${metric.label}" không thể mang giá trị âm theo định nghĩa kế toán.`,
    });
  }

  // 4 — Khoản mục bảng cân đối lớn hơn TỔNG TÀI SẢN (Mã số 270) của chính công ty/kỳ đó.
  //
  //     CHỈ ÁP KHI DỮ LIỆU NGUỒN VỐN ĐÃ CÂN ĐỐI. Đây là bài học phải trả giá khi port luật từ
  //     pipeline sang app, đo được trên chính dữ liệu mẫu: 875/987 công ty (88,7%) có ít nhất một
  //     khoản mục vượt tổng tài sản, vì dữ liệu mẫu là số sinh ngẫu nhiên không cân đối kế toán.
  //     Nếu cứ kêu thì gần như MỌI câu hỏi bảng cân đối đều bị gắn cờ đỏ.
  //
  //     Và quan trọng hơn con số: kêu như vậy là **đổ lỗi sai đối tượng**. Ở pipeline, giá trị đi
  //     qua một chuỗi mong manh (chọn dòng → chọn cột → quy đổi đơn vị) nên vượt trần nghĩa là
  //     chuỗi đó hỏng. Ở app, giá trị là ô đọc thẳng theo Mã số — đọc đúng thì con số vượt trần
  //     chỉ chứng minh FILE NGUỒN mâu thuẫn, không phải đáp án sai. Việc đó `validate.ts` đã báo.
  //     Nên: nếu validateTicker thấy dữ liệu công ty đó đã lệch đẳng thức, NHƯỜNG cho cảnh báo
  //     dữ liệu; chỉ khi dữ liệu cân đối mà vẫn vượt trần thì mới là KHÂU XỬ LÝ CỦA TA hỏng
  //     (vd normalize.ts áp nhầm hệ số đơn vị cho một số dòng) — lúc đó cảnh báo mới đúng địa chỉ.
  if (!isDelta && metric?.statementType === "BS" && metric.kind === "line"
      && metric.key !== "total_assets" && metric.key !== "total_resources") {
    const c = result.citations[0];
    const ta = c ? tidy.byTicker[c.ticker]?.["270"]?.values[c.period] ?? null : null;
    if (ta != null && ta > 0 && Math.abs(v) > ta * TA_TOLERANCE
        && c && validateTicker(tidy, c.ticker).issues.length === 0) {
      out.push({
        rule: "Khoản mục lớn hơn tổng tài sản",
        source: "value",
        detail: `"${metric.label}" đang lớn hơn Tổng tài sản của ${c.ticker} kỳ ${c.period} dù dữ liệu nguồn vẫn cân đối — nhiều khả năng lỗi ở khâu đọc/quy đổi đơn vị.`,
      });
    }
  }

  // 5 — LỜI VĂN có khớp CON SỐ không. Vẫn đúng nguyên tắc: không đoán đáp án đúng, chỉ đối chiếu
  //     chữ với chính con số mà CODE đã tính. Mâu thuẫn ở đây là sai chứng minh được.
  if (result.comparison) {
    const { from, to, direction } = result.comparison;
    const SRC = [["Câu trả lời", result.answer, "answer"], ["Nhận định", result.insight, "insight"]] as const;
    for (const [nguon, text, src] of SRC) {
      if (!text) continue;
      const said = leadingDirection(text);
      if (said && said !== direction) {
        out.push({
          rule: "Lời văn ngược chiều với số liệu", source: src,
          detail: `${nguon} mô tả theo chiều ${said === "up" ? "tăng" : "giảm"} trong khi số liệu thực tế ${direction === "up" ? "tăng" : "giảm"} (${from} → ${to}).`,
        });
      }
      if (hasReversedRange(text)) {
        out.push({
          rule: "Khoảng thời gian bị viết ngược", source: src,
          detail: `${nguon} nêu khoảng năm ngược chiều thời gian; kỳ gốc là ${from}, kỳ sau là ${to}.`,
        });
      }
    }
  }

  return out;
}

/** Gộp các vi phạm thành một câu cảnh báo hiển thị cạnh đáp án. */
export function auditMessage(vios: AuditViolation[]): string | null {
  if (vios.length === 0) return null;
  const body = vios.map((x) => x.detail).join(" ");
  return `Kết quả này không vượt qua bước tự kiểm: ${body} Hãy đối chiếu lại với file nguồn trước khi sử dụng.`;
}
