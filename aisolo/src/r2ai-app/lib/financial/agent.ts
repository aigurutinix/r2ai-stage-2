import { generateText } from "ai";
import { z } from "zod";
import { primaryModel } from "@/lib/providers";
import { findMetric, registryForPrompt, type MetricDef } from "./registry";
import { plannerSystem } from "./prompts";
import { formatMoney } from "./clean";
import { capitalStructure, profitBridge } from "./dashboard";
import type { AgentResult, Citation, QueryPlan, TidyItem, TidyTable } from "./types";

const INTENTS = ["direct_retrieval", "yoy_growth", "profit_margin", "ratio", "multi_company", "unknown"] as const;

const planSchema = z.object({
  intent: z.enum(INTENTS),
  metric: z.string(),
  years: z.array(z.string()),
  tickers: z.array(z.string()),
});

function extractJson(text: string): unknown {
  let s = text.trim();
  // Bỏ khối thinking nếu model rò <think>...</think> (Qwen3.x hybrid).
  s = s.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  const fence = s.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fence) s = fence[1].trim();
  const start = s.indexOf("{");
  const end = s.lastIndexOf("}");
  if (start > -1 && end > start) s = s.slice(start, end + 1);
  return JSON.parse(s);
}

/** Lập kế hoạch: Qwen3.5-4B local qua Ollama (generateText + parse JSON). */
export async function planQuery(
  question: string,
  context: string,
  prevPlan?: QueryPlan | null,
): Promise<{ plan: QueryPlan; provider: string }> {
  const system = plannerSystem(INTENTS, registryForPrompt());
  // Kế hoạch trước truyền dạng JSON CÓ KIỂU, không phải tóm tắt bằng văn xuôi: planner chỉ cần
  // biết những TRƯỜNG nào đang thiếu để kế thừa, và JSON nói điều đó không mất mát.
  const prev = prevPlan ? `Kế hoạch trước: ${JSON.stringify(prevPlan)}\n` : "";
  const prompt = `${context}\n${prev}\nCâu hỏi: ${question}`;
  const r = await generateText({ model: primaryModel(), system, prompt, temperature: 0.2 });
  const plan = planSchema.parse(extractJson(r.text));
  return { plan, provider: "qwen3.5-4b" };
}

// ---- Compute deterministic ----

export function pickTicker(tidy: TidyTable, plan: QueryPlan, activeTicker?: string): string | undefined {
  for (const t of plan.tickers) {
    if (tidy.byTicker[t]) return t;
    const f = tidy.tickers.find((x) => x.toLowerCase() === t.toLowerCase());
    if (f) return f;
  }
  if (activeTicker && tidy.byTicker[activeTicker]) return activeTicker;
  return tidy.tickers[0];
}

function pickPeriod(tidy: TidyTable, year?: string): string | undefined {
  if (!year) return tidy.periods[0];
  if (tidy.periods.includes(year)) return year;
  return tidy.periods.find((p) => p.includes(year)) ?? tidy.periods[0];
}

function lookup(tidy: TidyTable, ticker: string, maSo: string, period: string): { value: number | null; item?: TidyItem } {
  const item = tidy.byTicker[ticker]?.[maSo];
  return { value: item ? (item.values[period] ?? null) : null, item };
}

function cite(tidy: TidyTable, ticker: string, item: TidyItem | undefined, maSo: string, period: string): Citation {
  return { ticker, maSo, itemName: item?.name ?? maSo, period, sheet: tidy.sheetName };
}

/** Chuỗi Python an toàn (double-quoted, escape đúng). */
const pyStr = (s: string) => JSON.stringify(String(s));

/**
 * Header self-contained, chạy được trên ĐÚNG file người dùng vừa nạp.
 * - Đọc theo đuôi file: .xlsx/.xls dùng read_excel (app nhận cả Excel; read_csv sẽ hỏng ngay dòng đầu).
 * - Kèm clean_number: bản Python của `cleanNumber` trong clean.ts. Người dùng có thể nạp file
 *   bẩn ("60,673,395 VND", "82.542.011 $", "(65,406,519)") — app tự làm sạch nhưng code copy ra
 *   thì đọc file GỐC, nên không có bước này `pd.to_numeric` trả NaN (đo được bằng verify-pandas.mjs).
 */
function pandasHeader(tidy: TidyTable): string {
  const file = tidy.fileName ?? "bao_cao_tai_chinh.csv";
  const isExcel = /\.xlsx?$/i.test(file);
  // App tự dò dòng header thật khi file có dòng tiêu đề rác phía trên ("BÁO CÁO TÀI CHÍNH",
  // "Đơn vị tính: ..."). Pandas mặc định lấy dòng đầu, nên phải nói cho nó bỏ qua đúng ngần ấy dòng
  // — nếu không, mã copy ra đọc sai toàn bộ tên cột và không chạy được trên chính file người dùng.
  const skip = tidy.prep?.headerRow ?? 0;
  const skipArg = skip > 0 ? `, skiprows=${skip}` : "";
  const read = isExcel
    ? `pd.read_excel(${pyStr(file)}, sheet_name=${pyStr(tidy.sheetName)}, dtype=str${skipArg})`
    : `pd.read_csv(${pyStr(file)}, dtype=str, keep_default_na=False${skipArg})`;
  return `import re
import pandas as pd

# Bản Python của \`cleanNumber\` (lib/financial/clean.ts) — PHẢI khớp từng luật với bản TS.
# Hai bản lệch nhau nghĩa là code này chạy ra số khác số app hiển thị, mà đây chính là đoạn
# người đọc mã nguồn copy ra để kiểm chứng. \`verify-pandas.mjs\` đối chiếu hai bản.
SUFFIX_TAIL = r"(?:\\s*(?:đồng|dong|vnd|d))?\\s*\\)?\\s*$"
SUFFIX_SCALE = [
    (re.compile(r"\\d\\s*(?:tỷ|tỉ|ty|ti|billion|bn)" + SUFFIX_TAIL, re.I), 1e9),
    (re.compile(r"\\d\\s*(?:triệu|trieu|million|tr)" + SUFFIX_TAIL, re.I), 1e6),
    (re.compile(r"\\d\\s*(?:nghìn|nghin|ngàn|ngan|thousand|k)" + SUFFIX_TAIL, re.I), 1e3),
]

def clean_number(raw):
    """Số kế toán VN: ngoặc đơn = âm, bỏ ký hiệu tiền tệ, phân cách nghìn '.' hoặc ','."""
    # Quy dấu trừ lạ (U+2212, en/em dash) về hyphen ASCII trước mọi phép thử: nếu không,
    # "−1500" bị strip mất dấu và ra +1500 — sai dấu trong im lặng.
    s = re.sub(r"[−‒–—―]", "-", str(raw).strip())
    if s == "" or re.fullmatch(r"n/?a|null|nan|-|\\.\\.\\.", s, re.I):
        return float("nan")
    if re.fullmatch(r"[+-]?\\d+(?:\\.\\d+)?[eE][+-]?\\d+", s):   # 1e6 -> 1000000, KHÔNG phải 16
        try:
            return float(s)
        except ValueError:
            return float("nan")
    suffix = 1.0
    for rx, mul in SUFFIX_SCALE:                             # "1,5 tỷ" -> 1.5e9
        if rx.search(s):
            suffix = mul
            break
    neg = (s.startswith("(") and s.endswith(")")) or s.startswith("-")
    s = re.sub(r"[^0-9.,]", "", s.replace("(", "").replace(")", ""))
    if s == "":
        return float("nan")
    if re.fullmatch(r"\\d{1,3}([.,]\\d{3})+", s):          # 60,673,395 hoặc 82.542.011
        s = re.sub(r"[.,]", "", s)
    elif re.fullmatch(r"\\d+[.,]\\d+", s):                 # 2,15 hoặc 2.15
        s = s.replace(",", ".")
    else:
        lc, ld = s.rfind(","), s.rfind(".")
        if lc > -1 and ld > -1:                          # dấu SAU cùng là thập phân
            dec, tho = (",", ".") if lc > ld else (".", ",")
            s = s.replace(tho, "").replace(dec, ".")
        else:
            s = re.sub(r"[.,]", "", s)
    try:
        v = float(s) * suffix
    except ValueError:
        return float("nan")
    return -v if neg else v

def has_own_unit(raw):
    """Ô có TỰ KHAI đơn vị trong chính nội dung không ("1,5 tỷ")?"""
    if not isinstance(raw, str):
        return False
    s = re.sub(r"[−‒–—―]", "-", raw.strip())
    return any(rx.search(s) for rx, _ in SUFFIX_SCALE)

def scale_cell(raw, scale):
    """Quy đổi theo đơn vị CHUNG của sheet — trừ ô đã tự khai đơn vị, vì nhân nữa là nhân đôi."""
    v = clean_number(raw)
    return v if has_own_unit(raw) else v * scale

df = ${read}
`;
}

/**
 * 1 dòng lookup: lọc theo Ticker + chỉ tiêu, cast numeric (an toàn dưới dtype=str).
 *
 * Lọc theo cột nào phụ thuộc nguồn Mã số — đây là điều kiện để mã sinh ra CHẠY ĐƯỢC trên file gốc:
 *  - `column`    : file có cột Mã số và mã khớp chuẩn ⇒ lọc theo Mã số như cũ.
 *  - `inferred`  : file KHÔNG có cột Mã số (app suy từ tên) ⇒ lọc theo cột TÊN, vì cột Mã số không
 *                  tồn tại trong file; giữ nguyên cách cũ sẽ sinh `df["…"]` gây KeyError.
 *  - `corrected` : file có cột Mã số nhưng ghi lệch chuẩn TT200 (vd Tài sản cố định ghi 210 thay
 *                  vì 220). Mã app dùng đã được sửa nên KHÔNG khớp mã trong file ⇒ cũng phải lọc
 *                  theo tên, đúng cái tên mà app đã căn cứ vào để sửa.
 */
function pandasLookup(tidy: TidyTable, ticker: string, maSo: string, period: string, varName = "result"): string {
  const col = tidy.meta.periodKeys[period] ?? period;
  const tk = tidy.meta.tickerKey;
  const src = tidy.prep?.maSoSource ?? "column";
  const nameKey = tidy.meta.nameKey;

  // Tên chỉ tiêu đúng như ghi trong file, lấy từ chính dòng đã đọc.
  const itemName = tidy.byTicker[ticker]?.[maSo]?.name;
  const dungTen = src !== "column" && nameKey && itemName;

  const filter = dungTen
    ? `df[${pyStr(nameKey)}].str.strip() == ${pyStr(itemName)}`
    : `df[${pyStr(tidy.meta.maSoKey)}].str.strip() == ${pyStr(maSo)}`;
  const cond = tk ? `(df[${pyStr(tk)}] == ${pyStr(ticker)}) & (${filter})` : filter;

  // Sheet khai đơn vị chung ("Đơn vị tính: triệu đồng") thì app đã quy đổi hết về VND, còn Pandas
  // đọc file GỐC nên thấy số chưa quy đổi. Không nhân lại thì mã copy ra lệch app đúng 6 hoặc 9 bậc.
  // Dùng `scale_cell`: ô nào TỰ KHAI đơn vị trong nội dung ("1,5 tỷ") thì `clean_number` đã quy đổi
  // rồi, nhân thêm hệ số sheet nữa là nhân đôi — đúng lỗi đã đo được ở phía app.
  const scale = tidy.prep?.unitScale ?? 1;
  const cell = `df.loc[${cond}, ${pyStr(col)}].iloc[0]`;
  const expr = scale !== 1 ? `scale_cell(${cell}, ${scale.toExponential().replace("e+", "e")})` : `clean_number(${cell})`;
  return `${varName} = ${expr}`;
}

function fmt(value: number | null, unit: MetricDef["unit"], tableUnit: string): string {
  if (value == null) return "—";
  if (unit === "%") return `${value.toLocaleString("vi-VN", { maximumFractionDigits: 2 })}%`;
  if (unit === "x") return `${value.toLocaleString("vi-VN", { maximumFractionDigits: 2 })} lần`;
  return formatMoney(value, tableUnit);
}

/** Tính kết quả cuối theo plan + tidy (không để LLM tự tính số). */
export function computeAnswer(tidy: TidyTable, plan: QueryPlan, provider: string, activeTicker?: string): AgentResult {
  const base: AgentResult = {
    ok: false,
    answer: "",
    value: null,
    valueLabel: "—",
    unit: tidy.unit,
    citations: [],
    pandasCode: "",
    plan,
    provider,
  };

  const metric = findMetric(plan.metric);
  const ticker = pickTicker(tidy, plan, activeTicker);
  if (!metric) return { ...base, error: `Không nhận ra chỉ số "${plan.metric}".`, answer: `Xin lỗi, tôi chưa hỗ trợ chỉ số "${plan.metric}". Bạn thử hỏi doanh thu, lợi nhuận, ROE, ROA, biên lợi nhuận, D/E…` };
  if (!ticker) return { ...base, error: "Không có dữ liệu công ty.", answer: "Chưa có dữ liệu để trả lời. Bạn đính kèm lại file báo cáo nhé." };

  // Multi-company: 2 ticker, line metric
  if (plan.intent === "multi_company" && metric.kind === "line" && plan.tickers.length >= 2) {
    const [tA, tB] = plan.tickers.map((t) => tidy.tickers.find((x) => x.toLowerCase() === t.toLowerCase()) ?? t);
    const period = pickPeriod(tidy, plan.years[0]);
    if (period && tidy.byTicker[tA] && tidy.byTicker[tB]) {
      const a = lookup(tidy, tA, metric.maSo!, period);
      const b = lookup(tidy, tB, metric.maSo!, period);
      if (a.value != null && b.value != null) {
        const diff = a.value - b.value;
        return {
          ...base,
          ok: true,
          value: diff,
          valueLabel: fmt(diff, metric.unit, tidy.unit),
          answer: `${metric.label} năm ${period}: ${tA} (${fmt(a.value, metric.unit, tidy.unit)}) ${diff >= 0 ? "lớn hơn" : "nhỏ hơn"} ${tB} (${fmt(b.value, metric.unit, tidy.unit)}) là ${fmt(Math.abs(diff), metric.unit, tidy.unit)}.`,
          citations: [cite(tidy, tA, a.item, metric.maSo!, period), cite(tidy, tB, b.item, metric.maSo!, period)],
          pandasCode: `${pandasHeader(tidy)}${pandasLookup(tidy, tA, metric.maSo!, period, "a")}\n${pandasLookup(tidy, tB, metric.maSo!, period, "b")}\nresult = a - b`,
        };
      }
    }
  }

  // Ratio (roe, roa, margin, d/e...)
  if (metric.kind === "ratio" && metric.formula) {
    const period = pickPeriod(tidy, plan.years[0]);
    if (!period) return { ...base, error: "Thiếu kỳ số liệu.", answer: "Không xác định được năm cần tính." };
    const num = lookup(tidy, ticker, metric.formula.numer, period);
    const den = lookup(tidy, ticker, metric.formula.denom, period);
    if (num.value == null || den.value == null || den.value === 0) {
      return { ...base, error: "Thiếu số liệu thành phần.", answer: `Không đủ dữ liệu để tính ${metric.label} cho ${ticker} năm ${period}.` };
    }
    let val = num.value / den.value;
    if (metric.formula.percent) val *= 100;
    val = Math.round(val * 100) / 100;
    return {
      ...base,
      ok: true,
      value: val,
      unit: metric.unit ?? "",
      valueLabel: fmt(val, metric.unit, tidy.unit),
      answer: `${metric.label} của ${ticker} năm ${period} là ${fmt(val, metric.unit, tidy.unit)}.`,
      citations: [cite(tidy, ticker, num.item, metric.formula.numer, period), cite(tidy, ticker, den.item, metric.formula.denom, period)],
      pandasCode: `${pandasHeader(tidy)}${pandasLookup(tidy, ticker, metric.formula.numer, period, "numer")}\n${pandasLookup(tidy, ticker, metric.formula.denom, period, "denom")}\nresult = round(numer / denom${metric.formula.percent ? " * 100" : ""}, 2)`,
    };
  }

  // Line item
  if (metric.kind === "line" && metric.maSo) {
    // YoY
    if (plan.intent === "yoy_growth" || plan.years.length >= 2) {
      // Xác định kỳ SAU/kỳ GỐC theo GIÁ TRỊ NĂM, không theo thứ tự mảng years[] — planner liệt kê
      // theo thứ tự XUẤT HIỆN trong câu hỏi ("từ 2023 sang 2024" → ["2023","2024"]), không phải
      // theo quy ước "years[0] luôn là kỳ đến". Giả định cũ làm ĐẢO NGƯỢC hoàn toàn chiều tăng/giảm
      // với dạng câu phổ biến "từ X sang Y" (X cũ hơn Y) — đo được gây sai chiều 100% cho dạng này.
      // Fallback về thứ tự mảng gốc nếu không tách được năm 4 chữ số (kỳ dạng lạ, hiếm).
      const y0 = parseInt(plan.years[0]?.match(/\d{4}/)?.[0] ?? "", 10);
      const y1 = parseInt(plan.years[1]?.match(/\d{4}/)?.[0] ?? "", 10);
      const swap = !Number.isNaN(y0) && !Number.isNaN(y1) && y0 < y1;
      const p2 = pickPeriod(tidy, plan.years[swap ? 1 : 0]) ?? tidy.periods[0];
      const p1 = pickPeriod(tidy, plan.years[swap ? 0 : 1]) ?? tidy.periods[1] ?? tidy.periods[0];
      const v2 = lookup(tidy, ticker, metric.maSo, p2);
      const v1 = lookup(tidy, ticker, metric.maSo, p1);
      if (v1.value != null && v2.value != null && v1.value !== 0) {
        const g = Math.round(((v2.value - v1.value) / Math.abs(v1.value)) * 10000) / 100;
        return {
          ...base,
          ok: true,
          value: g,
          unit: "%",
          // Ghi rõ đây là so sánh CÓ CHIỀU + hai kỳ nào. Bộ kiểm chiều của insight và Auditor đọc
          // trường này thay vì đoán lại từ plan.intent (nhánh này chạy cả khi intent KHÁC yoy_growth).
          comparison: { from: p1, to: p2, direction: g >= 0 ? "up" : "down" },
          valueLabel: `${g.toLocaleString("vi-VN", { maximumFractionDigits: 2 })}%`,
          answer: `${metric.label} của ${ticker} năm ${p2} ${g >= 0 ? "tăng" : "giảm"} ${Math.abs(g).toLocaleString("vi-VN", { maximumFractionDigits: 2 })}% so với năm ${p1} (${fmt(v1.value, metric.unit, tidy.unit)} → ${fmt(v2.value, metric.unit, tidy.unit)}).`,
          citations: [cite(tidy, ticker, v2.item, metric.maSo, p2), cite(tidy, ticker, v1.item, metric.maSo, p1)],
          pandasCode: `${pandasHeader(tidy)}${pandasLookup(tidy, ticker, metric.maSo, p2, "v2")}\n${pandasLookup(tidy, ticker, metric.maSo, p1, "v1")}\nresult = round((v2 - v1) / v1 * 100, 2)`,
        };
      }
      return { ...base, error: "Thiếu số liệu 2 năm.", answer: `Không đủ dữ liệu ${metric.label} cho ${ticker} ở cả hai năm.` };
    }
    // Direct retrieval
    const period = pickPeriod(tidy, plan.years[0]);
    if (!period) return { ...base, error: "Thiếu kỳ.", answer: "Không xác định được năm cần lấy." };
    const r = lookup(tidy, ticker, metric.maSo, period);
    if (r.value == null) return { ...base, error: "Không có số liệu.", answer: `Không tìm thấy ${metric.label} (Mã số ${metric.maSo}) của ${ticker} năm ${period}.` };
    return {
      ...base,
      ok: true,
      value: r.value,
      valueLabel: fmt(r.value, metric.unit, tidy.unit),
      answer: `${metric.label} của ${ticker} năm ${period} là ${fmt(r.value, metric.unit, tidy.unit)}.`,
      citations: [cite(tidy, ticker, r.item, metric.maSo, period)],
      pandasCode: `${pandasHeader(tidy)}${pandasLookup(tidy, ticker, metric.maSo, period)}`,
    };
  }

  return { ...base, error: "Không xử lý được.", answer: "Tôi chưa xử lý được câu hỏi này. Bạn thử diễn đạt khác nhé." };
}

/**
 * Ngữ cảnh gọn nhét vào prompt (không gửi toàn bộ dữ liệu cho LLM).
 *
 * Có nêu CÁC MÃ SỐ THẬT SỰ CÓ trong file, không chỉ kỳ và công ty. Vì sao: danh mục chỉ tiêu trong
 * system prompt là danh mục TĨNH của registry, nên planner không hề biết file này thiếu chỉ tiêu
 * nào. Nó cứ chọn bừa một metric hợp lệ về mặt danh mục rồi bước tra số trả "không tìm thấy" —
 * người dùng nhận lời từ chối cho một câu hỏi mà dữ liệu vốn không trả lời được, và không hiểu vì
 * sao. Nêu ra thì planner tự tránh, còn nếu vẫn chọn thì ít ra ta biết lỗi ở planner.
 * Gửi Mã số (ngắn, khớp đúng thứ registry dùng) chứ không gửi tên chỉ tiêu (dài, tốn token).
 */
export function buildContext(tidy: TidyTable, activeTicker?: string): string {
  const tk = tidy.tickers.slice(0, 15).join(", ") + (tidy.tickers.length > 15 ? `, … (${tidy.tickers.length} công ty)` : "");
  const ref = activeTicker && tidy.byTicker[activeTicker] ? activeTicker : tidy.tickers[0];
  const codes = ref ? Object.keys(tidy.byTicker[ref] ?? {}).sort((a, b) => Number(a) - Number(b)) : [];
  const coDuoc = codes.length ? ` Mã số có trong file = [${codes.join(", ")}] — chỉ chọn metric có Mã số nằm trong danh sách này.` : "";
  return `Ngữ cảnh dữ liệu (sheet "${tidy.sheetName}"): các kỳ có sẵn = [${tidy.periods.join(", ")}]; công ty = [${tk}]${activeTicker ? `; công ty đang chọn = ${activeTicker}` : ""}.${coDuoc}`;
}

/**
 * Chọn biểu đồ theo DẠNG câu hỏi (classifier deterministic — không để LLM/gpt-vis tự quyết):
 *  - Cơ cấu / tỷ trọng nguồn vốn        → pie (donut)
 *  - So sánh NHIỀU công ty (1 kỳ)       → column (bar), trục x = công ty
 *  - Xu hướng theo THỜI GIAN (nhiều kỳ) → line, trục x = kỳ
 *  - 1 con số đơn                       → không chart (đã có KPI card)
 * Lưu ý: factory Line encode x="time" (khác Column/Pie dùng "category") → data key phải là `time`.
 */
export const PALETTE = ["#1c64e8", "#e07c00", "#0f9d58", "#8e44ad", "#c62828", "#00838f"];

export function buildChart(tidy: TidyTable, plan: QueryPlan, question: string, ticker: string): AgentResult["chart"] {
  const q = question.toLowerCase();
  const wantsChart = /vẽ|biểu đồ|chart|đồ thị|trực quan|minh ho[aạ]|visual/.test(q);
  const metric = findMetric(plan.metric);

  // 0) PHÂN RÃ tổng thành các khoản cộng/trừ nối tiếp → CẦU THANG (waterfall).
  //    Dạng riêng của tài chính, quy tắc biểu đồ phổ thông không có: đi từ doanh thu qua giá
  //    vốn, chi phí, thuế xuống lợi nhuận sau thuế — thấy được khoản nào ăn mòn lợi nhuận.
  if (/phân rã|cấu thành|cầu nối|cầu thang|waterfall|từ doanh thu (đến|xuống|tới)/.test(q)) {
    const bridge = profitBridge(tidy, ticker, pickPeriod(tidy, plan.years[0]) ?? tidy.periods[0]);
    if (bridge) return { type: "waterfall", height: 260, config: { data: bridge } };
  }

  // 1) CƠ CẤU / TỶ TRỌNG — tròn hay miền tuỳ SỐ KỲ.
  //    Quy tắc: 1–3 kỳ → tròn; từ 4 kỳ trở lên → miền chồng 100% (tròn không đọc được
  //    chuyển dịch qua chuỗi thời gian dài).
  //    Hai chặn riêng cho số liệu tài chính, quy tắc phổ thông không có:
  //    hình tròn KHÔNG biểu diễn được giá trị âm (lỗ, vốn chủ âm) và loãng nghĩa khi quá
  //    nhiều lát — cả hai trường hợp chuyển sang cột.
  if (/cơ cấu|tỷ trọng/.test(q) && /vốn|nợ|nguồn/.test(q)) {
    const periods = [...tidy.periods].sort();
    if (periods.length >= 4) {
      const data: { time: string; value: number; group: string }[] = [];
      for (const p of periods) {
        for (const part of capitalStructure(tidy, ticker, p) ?? [])
          data.push({ time: p, value: part.value, group: part.category });
      }
      if (data.length >= 4) return { type: "area", height: 240, config: { data, style: { palette: PALETTE } } };
    }
    const cap = capitalStructure(tidy, ticker, periods[periods.length - 1] ?? tidy.periods[0]);
    if (cap) {
      const bad = cap.some((d) => d.value < 0) || cap.length > 6;
      if (bad) return { type: "column", height: 240, config: { data: cap, style: { palette: PALETTE } } };
      return { type: "pie", height: 240, config: { data: cap, innerRadius: 0.5, style: { palette: PALETTE } } };
    }
  }

  // Các dạng dưới chỉ áp cho chỉ tiêu dòng (line item, có Mã số) — ratio/% dùng KPI, không vẽ.
  if (!(metric?.kind === "line" && metric.maSo)) return null;

  // 2) So sánh NHIỀU công ty → CỘT. Một kỳ thì cột đơn; nhiều kỳ thì cột NHÓM (mỗi kỳ một
  //    cụm, mỗi công ty một cột) — đúng quy tắc so sánh cùng đơn vị qua nhiều mốc.
  if (plan.intent === "multi_company" && plan.tickers.length >= 2) {
    const tks = plan.tickers.map((t) => tidy.tickers.find((x) => x.toLowerCase() === t.toLowerCase()) ?? t);
    const periods = plan.years.length >= 2 ? [...tidy.periods].sort() : [pickPeriod(tidy, plan.years[0]) ?? tidy.periods[0]];
    if (periods.length >= 2) {
      const data: { category: string; value: number; group: string }[] = [];
      for (const p of periods)
        for (const tk of tks) {
          const v = tidy.byTicker[tk]?.[metric.maSo]?.values[p];
          if (v != null) data.push({ category: p, value: Math.round(v / 1e6), group: tk });
        }
      if (data.length >= 4) {
        // Chuỗi thời gian DÀI (≥4 kỳ) đọc bằng đường rõ hơn cột nhóm — cụm cột dày quá thì
        // mắt không lần được xu hướng. Ngắn (2–3 kỳ) thì cột nhóm so sánh trực diện hơn.
        if (periods.length >= 4) {
          const line = data.map((d) => ({ time: d.category, value: d.value, group: d.group }));
          return { type: "line", height: 240, config: { data: line, style: { palette: PALETTE } } };
        }
        return { type: "column", height: 240, config: { data, group: true, style: { palette: PALETTE } } };
      }
    }
    const period = periods[0];
    const data: { category: string; value: number }[] = [];
    for (const tk of tks) {
      const v = tidy.byTicker[tk]?.[metric.maSo]?.values[period];
      if (v != null) data.push({ category: tk, value: Math.round(v / 1e6) });
    }
    if (data.length >= 2) return { type: "column", height: 240, config: { data, style: { palette: PALETTE } } };
  }

  // 3) Xu hướng theo THỜI GIAN qua nhiều kỳ → LINE (trục x = kỳ, data key "time")
  const isTrend = plan.intent === "yoy_growth" || /xu hướng|qua các năm|theo năm|từng năm|các kỳ|biến động/.test(q) || wantsChart;
  if (isTrend) {
    const periods = [...tidy.periods].sort();
    const data: { time: string; value: number }[] = [];
    for (const p of periods) {
      const v = tidy.byTicker[ticker]?.[metric.maSo]?.values[p];
      if (v != null) data.push({ time: p, value: Math.round(v / 1e6) });
    }
    if (data.length >= 2) return { type: "line", height: 240, config: { data, style: { palette: ["#1c64e8"] } } };
  }
  return null;
}

/** Ý định theo intent, bằng tiếng Việt cho người đọc (không phải khoá kỹ thuật). */
const INTENT_VI: Record<QueryPlan["intent"], string> = {
  direct_retrieval: "Lấy một chỉ tiêu",
  yoy_growth: "So sánh tăng trưởng qua năm",
  profit_margin: "Tính biên lợi nhuận",
  ratio: "Tính tỷ số",
  multi_company: "So sánh nhiều công ty",
  unknown: "Phân tích câu hỏi",
};

/** Mô tả kế hoạch bằng NGÔN NGỮ ý định (dùng nhãn chỉ tiêu, KHÔNG lộ khoá metric/mã/công thức). */
export function describePlan(plan: QueryPlan): string {
  const metricLabel = findMetric(plan.metric)?.label;
  const parts = [INTENT_VI[plan.intent]];
  if (metricLabel) parts.push(metricLabel);
  if (plan.tickers.length) parts.push(plan.tickers.join(", "));
  if (plan.years.length) parts.push("năm " + plan.years.join("–"));
  return parts.filter(Boolean).join(" · ");
}

/** Danh sách số liệu ĐÃ dùng, dạng người đọc (không mã Pandas): "AAA · Doanh thu thuần · 2024". */
export function usedDataLines(result: AgentResult): string[] {
  return result.citations.map(
    (c) => `${c.ticker !== "(toàn bộ)" ? c.ticker + " · " : ""}${c.itemName} · ${c.period}`,
  );
}

