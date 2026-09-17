/**
 * Làm sạch số kế toán Việt Nam:
 * - Ngoặc đơn = số âm: "(1,500,000)" -> -1500000
 * - Bỏ ký hiệu tiền tệ dính liền: "60,673,395 VND", "82.542.011 $"
 * - Xử lý phân cách nghìn (dấu . hoặc ,) và thập phân
 * - Hậu tố đơn vị trong chính ô: "1,5 tỷ" -> 1500000000
 * - null/NaN/N/A/"-"/khoảng trắng -> null
 *
 * PHẢI GIỮ ĐỒNG BỘ với bản Python `clean_number` trong `agent.ts` (pandasHeader): mã Pandas sinh
 * ra cho người dùng đọc FILE GỐC, nên hai bản lệch nhau nghĩa là copy code ra chạy được số khác
 * số app hiển thị — thứ giám khảo kiểm được bằng mã nguồn công khai.
 */

/**
 * Hệ số cho hậu tố đơn vị viết ngay trong ô ("1,5 tỷ"). Không gồm "đồng" trần vì đó đã là VND.
 *
 * KHÔNG dùng `\b`: trong JS regex mặc định `\b` xét theo [A-Za-z0-9_], nên "tỷ" (ký tự có dấu)
 * không bao giờ khớp `\b...\b` — đo được: "1,5 tỷ" trượt trong khi "2 triệu" lại đạt. Cùng lớp bẫy
 * đã ghi ở `audit.ts` với UP_WORDS/DOWN_WORDS. Thay bằng: hậu tố phải đứng SAU chữ số (có thể cách
 * khoảng trắng), và tận cùng chuỗi hoặc theo sau là "đồng"/"vnd".
 */
const SUFFIX_TAIL = /(?:\s*(?:đồng|dong|vnd|d))?\s*\)?\s*$/i;
const SUFFIX_SCALE: [RegExp, number][] = [
  [new RegExp(`\\d\\s*(?:tỷ|tỉ|ty|ti|billion|bn)${SUFFIX_TAIL.source}`, "i"), 1e9],
  [new RegExp(`\\d\\s*(?:triệu|trieu|million|tr)${SUFFIX_TAIL.source}`, "i"), 1e6],
  [new RegExp(`\\d\\s*(?:nghìn|nghin|ngàn|ngan|thousand|k)${SUFFIX_TAIL.source}`, "i"), 1e3],
];

/** Dấu trừ không phải ASCII hyphen: U+2212 toán học, en/em dash. Word/Excel/PDF sinh ra thường xuyên. */
const UNICODE_MINUS = /[−‒–—―]/g;

/**
 * Ô này có TỰ KHAI đơn vị trong chính nội dung của nó không ("1,5 tỷ")?
 *
 * Vì sao cần biết: `normalize` áp hệ số đơn vị CHUNG của cả sheet ("Đơn vị tính: triệu đồng") lên
 * mọi ô. Nếu ô đã tự khai đơn vị thì `cleanNumber` đã quy đổi rồi, áp thêm hệ số sheet là nhân đôi
 * — đo được trên file mẫu lạ: 80 tỷ thành 80 triệu tỷ. Ô tự khai thì lời khai của nó thắng, vì nó
 * cụ thể hơn nhãn chung của sheet.
 */
export function hasOwnUnit(raw: unknown): boolean {
  if (typeof raw !== "string") return false;
  const s = raw.trim().replace(UNICODE_MINUS, "-");
  return SUFFIX_SCALE.some(([re]) => re.test(s));
}

export function cleanNumber(raw: unknown): number | null {
  if (raw == null) return null;
  if (typeof raw === "number") return Number.isFinite(raw) ? raw : null;
  // Quy dấu trừ lạ về hyphen ASCII TRƯỚC mọi phép thử. Nếu không, `/^-/` không nhận ra số âm và
  // bước strip bên dưới xoá luôn ký tự đó — "−1500" ra +1500, sai dấu mà không một dấu hiệu nào.
  let s = String(raw).trim().replace(UNICODE_MINUS, "-");
  if (s === "" || /^(n\/?a|null|nan|-|\.\.\.)$/i.test(s)) return null;

  // Ký hiệu khoa học: xử lý TRƯỚC khi strip. Strip xoá chữ "e" rồi dán hai phần lại, nên "1e6"
  // thành "16" — một con số sai hoàn toàn nhưng trông hợp lệ. Thà trả đúng còn hơn trả bừa.
  if (/^[+-]?\d+(?:\.\d+)?[eE][+-]?\d+$/.test(s)) {
    const n = Number(s);
    return Number.isFinite(n) ? n : null;
  }

  // Hậu tố đơn vị đọc TRƯỚC khi strip, vì strip xoá sạch chữ. Chỉ áp khi có chữ số đứng trước,
  // để không bắt nhầm một ô văn bản thuần ("Đơn vị: tỷ đồng" là nhãn, không phải giá trị).
  let suffix = 1;
  if (/\d/.test(s)) {
    for (const [re, mul] of SUFFIX_SCALE) {
      if (re.test(s)) { suffix = mul; break; }
    }
  }

  const neg = /^\(.*\)$/.test(s) || /^-/.test(s);
  s = s.replace(/[()]/g, "").replace(/[^0-9.,]/g, ""); // giữ chữ số, dấu . ,
  if (s === "") return null;

  if (/^\d{1,3}([.,]\d{3})+$/.test(s)) {
    // Số nguyên có phân cách nghìn (60,673,395 hoặc 82.542.011)
    s = s.replace(/[.,]/g, "");
  } else if (/^\d+[.,]\d+$/.test(s)) {
    // Một nhóm thập phân duy nhất (2,15 hoặc 2.15)
    s = s.replace(",", ".");
  } else {
    // Hỗn hợp: xác định dấu thập phân là dấu xuất hiện SAU cùng
    const lc = s.lastIndexOf(",");
    const ld = s.lastIndexOf(".");
    if (lc > -1 && ld > -1) {
      const dec = lc > ld ? "," : ".";
      const tho = dec === "," ? "." : ",";
      s = s.split(tho).join("").replace(dec, ".");
    } else {
      s = s.replace(/[.,]/g, "");
    }
  }

  const n = Number(s);
  if (!Number.isFinite(n)) return null;
  const v = n * suffix;
  return neg ? -v : v;
}

/** Định dạng số tiền cho người Việt (kèm quy đổi tỷ/triệu khi đủ lớn). */
export function formatMoney(n: number | null, unit = "VND"): string {
  if (n == null) return "—";
  const abs = Math.abs(n);
  const sign = n < 0 ? "-" : "";
  const nf = (x: number) => x.toLocaleString("vi-VN", { maximumFractionDigits: 2 });
  if (unit === "VND") {
    if (abs >= 1e9) return `${sign}${nf(abs / 1e9)} tỷ`;
    if (abs >= 1e6) return `${sign}${nf(abs / 1e6)} triệu`;
  }
  return `${sign}${nf(abs)}${unit && unit !== "VND" ? " " + unit : ""}`;
}
