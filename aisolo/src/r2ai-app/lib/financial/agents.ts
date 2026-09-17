/**
 * Multi-agent tuần tự: mỗi agent làm ĐÚNG MỘT việc, nhận đầu ra của agent trước và chuyển tiếp
 * cho agent sau, đến câu trả lời cuối cùng.
 *
 * Theo phân loại của Strands, đây là dạng **Graph/Workflow** chứ không phải Swarm: đường đi cố
 * định do code quyết định (có nhánh theo intent), không phải do agent tự chuyền tay nhau.
 *
 * Agent nào gọi LLM, agent nào không — và vì sao (SÁU agent, chỉ HAI gọi model):
 *   1. normalize  — CODE. Thuật lại việc bộ tiền xử lý đã làm với file; không tự sửa gì thêm.
 *   2. understand — LLM. Hiểu ngôn ngữ tự nhiên là việc chỉ model làm được.
 *   3. retrieve   — CODE. Tra số theo Mã số là tra cứu xác định; để LLM làm chỉ thêm đường bịa số.
 *   4. audit      — CODE. Tự kiểm miền giá trị của đáp án; xem `audit.ts` để biết vì sao đáng tin.
 *   5. chart      — CODE. Chọn dạng biểu đồ là bộ quy tắc rõ ràng (số kỳ, số đối tượng, đơn vị).
 *   6. narrate    — LLM. Diễn giải kết quả ĐÃ tính, không được tự tính lại.
 * Giữ đúng nguyên tắc xuyên suốt dự án: LLM hiểu và diễn giải, CODE tính toán.
 */
import {
  buildChart,
  buildContext,
  computeAnswer,
  describePlan,
  pickTicker,
  planQuery,
  usedDataLines,
} from "./agent";
import { auditAnswer, auditMessage } from "./audit";
import { formatMoney } from "./clean";
import { checkPipelineOrder, type AgentContract } from "./pipeline-contract";
import { genInsight } from "./insight-agent";
import { liveLookup } from "./live-lookup";
import { validateTicker } from "./validate";
import type { AgentResult, QueryPlan, StepData, TidyTable } from "./types";

/** Trạng thái chuyền giữa các agent. Agent chỉ đọc phần mình cần và ghi phần mình tạo ra. */
interface Ctx {
  tidy: TidyTable;
  question: string;
  activeTicker?: string;
  /** Kế hoạch của câu hỏi TRƯỚC — để planner kế thừa khi câu mới là hỏi nối tiếp
   *  ("còn 2023 thì sao?"). Truyền dữ liệu CÓ KIỂU, không phải tóm tắt bằng văn xuôi. */
  prevPlan?: QueryPlan | null;
  plan?: QueryPlan;
  provider?: string;
  result?: AgentResult;
}

/** Kết quả một lượt agent — dùng để hiển thị trace cho người dùng (ý định + kết quả, KHÔNG lộ mã). */
interface StepOut {
  status: StepData["status"];
  detail?: string;
  items?: string[];
}

interface Agent extends AgentContract {
  label: string;
  run(ctx: Ctx): Promise<StepOut>;
}

/**
 * 0 — CHUẨN HOÁ DỮ LIỆU: thuật lại việc bộ tiền xử lý đã làm với file người dùng nạp.
 *
 * KHÔNG tự xử lý gì ở đây, chỉ đọc và kể lại. Việc chuẩn hoá thật xảy ra ở CLIENT (`preprocess.ts`
 * gọi trong `normalize`), vì phải như vậy: `app/api/agent/route.ts` chặn request khi bảng rỗng,
 * nên đặt phần xử lý ở server thì đúng những file cần cứu lại không bao giờ tới được đây.
 *
 * Vì sao vẫn đáng có một agent: người dùng cần thấy dữ liệu của mình đã đi qua những gì trước khi
 * thành con số — nhất là khi app đã tự sửa Mã số hoặc bỏ qua dòng tiêu đề. Một nguồn sự thật:
 * client làm, chỗ này kể lại, không tính toán lại để khỏi có cơ hội lệch nhau.
 */
const normalizeData: Agent = {
  id: "normalize",
  label: "Chuẩn hoá dữ liệu",
  requires: ["tidy"],
  provides: [],
  async run(ctx) {
    const p = ctx.tidy.prep;
    if (!p) return { status: "done", detail: "dữ liệu đã ở dạng chuẩn" };

    const detail = [
      p.headerRow > 0 ? `header ở dòng ${p.headerRow + 1}` : null,
      p.coverage.total > 0 ? `${p.coverage.mapped}/${p.coverage.total} chỉ tiêu` : null,
      p.maSoSource === "inferred" ? "Mã số suy từ tên" : p.maSoSource === "corrected" ? "đã sửa Mã số lệch chuẩn" : null,
    ]
      .filter(Boolean)
      .join(" · ");

    const items = p.notes.map((n) => (n.level === "warn" ? `⚠ ${n.text}` : n.text));
    return {
      status: "done",
      detail: detail || "không phải sửa gì",
      items: items.length ? items : undefined,
    };
  },
};

/** 1 — HIỂU CÂU HỎI: câu hỏi tiếng Việt → kế hoạch truy vấn (JSON). */
const understand: Agent = {
  id: "understand",
  label: "Hiểu câu hỏi",
  requires: ["tidy", "question"],
  provides: ["plan", "provider"],
  async run(ctx) {
    const { plan, provider } = await planQuery(ctx.question, buildContext(ctx.tidy, ctx.activeTicker), ctx.prevPlan);
    ctx.plan = plan;
    ctx.provider = provider;
    return { status: "done", detail: describePlan(plan) };
  },
};

/** 2 — TRUY XUẤT & TÍNH: theo kế hoạch, tra số theo Mã số rồi tính (deterministic). */
const retrieve: Agent = {
  id: "retrieve",
  label: "Truy xuất & tính",
  requires: ["tidy", "plan", "provider"],
  provides: ["result"],
  async run(ctx) {
    const r = (ctx.result = computeAnswer(ctx.tidy, ctx.plan!, ctx.provider!, ctx.activeTicker));
    const used = usedDataLines(r);
    // Kiểm đẳng thức kế toán cho mọi ticker vừa dùng — validateTicker() đã có sẵn nhưng
    // trước đây chỉ được data-panel gọi, agent không hề biết dữ liệu có vấn đề hay không.
    const tickers = Array.from(new Set(r.citations.map((c) => c.ticker).filter((t) => t !== "(toàn bộ)")));
    const issues = tickers.flatMap((t) => validateTicker(ctx.tidy, t).issues);
    if (issues.length > 0) {
      const detail = issues
        .slice(0, 2)
        .map((i) => `${i.rule} lệch ${i.diffPct}% (${i.period})`)
        .join("; ");
      r.dataQualityWarning = `Dữ liệu ${tickers.join(", ")} vi phạm đẳng thức kế toán: ${detail}${issues.length > 2 ? "…" : ""} — kiểm tra lại file nguồn.`;
    }
    const items = r.dataQualityWarning ? [...used, `⚠ ${r.dataQualityWarning}`] : used;

    // TẦNG DỰ PHÒNG: kho chỉ giữ 30 chỉ tiêu registry, còn báo cáo gốc có ~650 dòng. Câu hỏi rơi
    // ra ngoài 30 chỉ tiêu đó thì trước đây app đành trả "chưa hỗ trợ chỉ số" dù số liệu nằm ngay
    // trong báo cáo. Nay tra thẳng báo cáo gốc — nhưng bằng khâu chọn dòng lexical (67,6%) và
    // KHÔNG có đẳng thức xác nhận, nên kết quả mang cờ `unverified` để giao diện nói rõ.
    // Chỉ làm khi dữ liệu đến TỪ KHO; file người dùng nạp thì không nằm trong corpus ViFinQA.
    if (!r.ok && ctx.tidy.fromCorpus && ctx.plan?.metric) {
      const ticker = ctx.tidy.fromCorpus.ticker;
      const period = ctx.plan.years?.[0] ?? ctx.tidy.periods[0];
      const doctype = /công ty mẹ|cong ty me/i.test(ctx.question ?? "") ? "separate" : "consolidated";
      const hit = await liveLookup(ticker, period, ctx.plan.metric, doctype);
      if (hit.ok && hit.value != null) {
        const label = formatMoney(hit.value, ctx.tidy.unit);
        Object.assign(r, {
          ok: true,
          value: hit.value,
          valueLabel: label,
          unit: ctx.tidy.unit,
          answer: `${hit.label} của ${ticker} năm ${period} là ${label}.`,
          citations: [{ ticker, maSo: hit.maSo ?? "—", itemName: hit.label, period, sheet: hit.doc }],
          // Không có CSV tập con cho dòng này (nó nằm trong báo cáo OCR gốc), nên không sinh mã
          // Pandas giả vờ chạy được — nói thẳng chỗ lấy số để người đọc tự mở ra đối chiếu.
          pandasCode: `# Số này đọc trực tiếp từ báo cáo gốc, không qua bảng CSV đã trích:
`
            + `#   ${hit.doc}, bảng ${hit.table}${hit.line ? `, dòng ${hit.line}` : ""}
`
            + `#   chỉ tiêu: ${hit.label}`,
          unverified: true,
          liveSource: { doc: hit.doc, line: hit.line, label: hit.label },
        });
        return {
          status: "done",
          detail: `${label} · tra trực tiếp báo cáo gốc (chưa kiểm chứng)`,
          items: [`⚠ Chưa qua kiểm đẳng thức kế toán — nguồn: ${hit.doc}${hit.line ? `, dòng ${hit.line}` : ""}`],
        };
      }
    }
    if (!r.ok) return { status: "error", detail: r.error ?? "không đủ số liệu", items: items.length ? items : undefined };
    // Báo cả KPI: gộp tra số và tính vào một agent nên trace phải nêu đủ cả hai, đừng để
    // người dùng mất dòng kết quả từng có.
    return { status: "done", detail: [used.length ? `${used.length} số liệu` : null, r.valueLabel].filter(Boolean).join(" · "), items: items.length ? items : undefined };
  },
};

/**
 * 3 — TỰ KIỂM: đáp án vừa tính có nằm trong miền giá trị hợp lệ suy từ câu hỏi không?
 *
 * KHÔNG chặn câu trả lời và KHÔNG sửa số — chỉ gắn cảnh báo để người dùng thấy. Ở pipeline dự thi
 * Auditor bác rồi chạy lại (planner là LLM nên chạy lại ra khác); ở đây bước tính là deterministic,
 * chạy lại ra đúng số cũ, nên việc đúng đắn là phơi bày chứ không giấu. Xem `audit.ts`.
 */
const audit: Agent = {
  id: "audit",
  label: "Tự kiểm",
  requires: ["tidy", "result"],
  provides: [],
  async run(ctx) {
    const r = ctx.result!;
    if (!r.ok) return { status: "done", detail: "bỏ qua" };
    const vios = auditAnswer(r, ctx.tidy);
    r.auditWarning = auditMessage(vios);
    if (vios.length === 0) return { status: "done", detail: "đáp án nằm trong miền hợp lệ" };
    return {
      status: "error",
      detail: `${vios.length} dấu hiệu sai`,
      items: vios.map((v) => `⚠ ${v.rule}`),
    };
  },
};

/**
 * 4 — BIỂU ĐỒ: chọn dạng trực quan phù hợp.
 * Chạy CẢ KHI bước tính không ra số: câu kiểu "vẽ cơ cấu nguồn vốn" không có chỉ số cụ thể
 * nhưng vẫn vẽ được — đây là nhánh điều kiện của Graph, không phải lỗi.
 */
const chart: Agent = {
  id: "chart",
  label: "Chọn biểu đồ",
  requires: ["tidy", "plan", "result"],
  provides: [],
  async run(ctx) {
    const r = ctx.result!;
    const ticker = pickTicker(ctx.tidy, ctx.plan!, ctx.activeTicker) ?? ctx.activeTicker ?? ctx.tidy.tickers[0];
    const c = buildChart(ctx.tidy, ctx.plan!, ctx.question, ticker);
    if (!c) return { status: "done", detail: "không cần biểu đồ" };
    r.chart = c;
    if (!r.ok) {
      // Cứu câu chỉ yêu cầu trực quan hoá: có biểu đồ là đã trả lời được.
      r.ok = true;
      r.error = undefined;
      r.answer = `Đây là biểu đồ cho ${ticker}:`;
    }
    return { status: "done", detail: c.type };
  },
};

/** 5 — NHẬN ĐỊNH: diễn giải kết quả đã có. Best-effort, hỏng thì bỏ qua chứ không chặn câu trả lời. */
const narrate: Agent = {
  id: "narrate",
  label: "Nhận định",
  requires: ["tidy", "question", "result"],
  provides: [],
  async run(ctx) {
    const r = ctx.result!;
    if (!r.ok) return { status: "done", detail: "bỏ qua" };

    // BÀN GIAO TỪ AUDITOR: nếu bước Tự kiểm đã chứng minh CON SỐ hoặc CÂU TRẢ LỜI sai, thì viết
    // một đoạn phân tích cho nó là phản tác dụng — phần nghe có thẩm quyền nhất của màn hình sẽ
    // mâu thuẫn với cảnh báo đỏ ngay phía trên. Không gọi LLM luôn: nhanh hơn, và không có cách
    // nào để model "lỡ" viết tự tin về một số đã biết là hỏng.
    const truoc = auditAnswer(r, ctx.tidy);
    if (truoc.some((x) => x.source === "value" || x.source === "answer")) {
      r.auditWarning = auditMessage(truoc);
      return { status: "done", detail: "bỏ qua — đáp án không qua được tự kiểm" };
    }

    r.insight = await genInsight(ctx.question, r);

    // Soi lại SAU khi có nhận định: bước `audit` chạy trước nên chưa nhìn thấy `insight`.
    // Nếu chính phần nhận định mâu thuẫn với số liệu thì BỎ HẲN nó đi, đừng hiển thị kèm cảnh báo
    // — một nhận định sai không có giá trị gì, để lại chỉ làm người đọc phân vân.
    const sau = auditAnswer(r, ctx.tidy);
    if (sau.some((x) => x.source === "insight")) {
      r.insight = null;
      r.auditWarning = auditMessage(auditAnswer(r, ctx.tidy));   // soi lần cuối, đã bỏ insight
      return { status: "done", detail: "đã bỏ nhận định — mâu thuẫn với số liệu" };
    }
    r.auditWarning = auditMessage(sau);
    return { status: "done", detail: r.insight ? undefined : "không có nhận định" };
  },
};

// normalizeData đứng ĐẦU: nó nói về dữ liệu ĐẦU VÀO, nên phải xuất hiện trước khi có bất kỳ con
// số nào — người dùng thấy nền móng trước, kết quả sau.
// audit đứng NGAY SAU retrieve: nó chỉ cần `result`, và đặt trước chart/narrate để nếu đáp án
// đã bị gắn cờ thì cảnh báo có mặt cùng lúc với con số, không đến sau.
const PIPELINE: Agent[] = [normalizeData, understand, retrieve, audit, chart, narrate];

/** Trường có sẵn ngay từ đầu, do `runAgent` khởi tạo trước khi agent nào chạy. */
const SEED = ["tidy", "question", "activeTicker", "prevPlan"] as const;

// Kiểm NGAY LÚC NẠP MODULE, không đợi tới lúc có request: đổi thứ tự PIPELINE hoặc chèn agent vào
// sai chỗ sẽ làm app hỏng ngay khi khởi động, kèm thông báo nói rõ agent nào thiếu trường nào.
// Trước đây một sai sót như vậy chỉ biểu hiện thành `undefined` đọc được ở đâu đó xa nguyên nhân.
checkPipelineOrder(SEED, PIPELINE);

/** Chạy tuần tự cả chuỗi agent; mỗi agent phát trace riêng để người dùng thấy đang làm gì. */
export async function runAgent(
  tidy: TidyTable,
  question: string,
  activeTicker?: string,
  onStep?: (s: StepData) => void,
  prevPlan?: QueryPlan | null,
): Promise<AgentResult> {
  const ctx: Ctx = { tidy, question, activeTicker, prevPlan };
  for (const a of PIPELINE) {
    onStep?.({ id: a.id, label: a.label, status: "running" });
    const out = await a.run(ctx);
    onStep?.({ id: a.id, label: a.label, ...out });
  }
  return ctx.result!;
}
