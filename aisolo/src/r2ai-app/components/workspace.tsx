"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { ChatPanel } from "@/components/chat/chat-panel";
import { DataPanel } from "@/components/data/data-panel";
import { parseWorkbookFromFile, parseWorkbook, type ParsedWorkbook } from "@/lib/xlsx-client";
import { normalize } from "@/lib/financial/normalize";
import { REGISTRY_CODES } from "@/lib/financial/registry";
import { corpusCsvUrl, loadCorpusIndex, loadCorpusMeta, verifiedSummary,
         type CorpusEntry } from "@/lib/financial/corpus";
import type { TidyTable, TidyItem, AgentResult, StepData, QueryPlan } from "@/lib/financial/types";

export type ChatMsg =
  | { id: string; role: "user"; text: string }
  | { id: string; role: "assistant"; loading?: boolean; steps?: StepData[]; result?: AgentResult; text?: string };

const SAMPLE_URL = "/samples/01_corporate_financial_statements.csv";
const DIRTY_URL = "/samples/01_dirty_corporate_financial_statements.csv";
// Mẫu DÀN DỰNG có chủ ý để trình diễn bước Tự kiểm bắt lỗi — xem make-audit-sample.mjs.
const UNIT_ERR_URL = "/samples/01_unit_error_financial_statements.csv";
// Mẫu DÀN DỰNG để trình diễn bước Chuẩn hoá dữ liệu — xem make-messy-sample.mjs.
const MESSY_URL = "/samples/01_messy_financial_statements.csv";

// crypto.randomUUID chỉ có trong secure context (HTTPS/localhost); fallback cho HTTP qua IP mạng (demo LAN).
function genId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") return crypto.randomUUID();
  return "id-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10);
}

function upsertStep(steps: StepData[], step: StepData): StepData[] {
  const idx = steps.findIndex((s) => s.id === step.id);
  if (idx >= 0) {
    const copy = [...steps];
    copy[idx] = step;
    return copy;
  }
  return [...steps, step];
}

export function Workspace() {
  const [parsed, setParsed] = useState<ParsedWorkbook | null>(null);
  const [activeSheet, setActiveSheet] = useState("");
  const [tidy, setTidy] = useState<TidyTable | null>(null);
  const [activeTicker, setActiveTicker] = useState("");
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [sending, setSending] = useState(false);
  const [loadingSample, setLoadingSample] = useState(false);
  /**
   * Kho doanh nghiệp niêm yết trích sẵn (xem lib/financial/corpus.ts). Danh mục nạp một lần lúc
   * mở trang; vắng kho thì mảng rỗng và mục tra cứu tự ẩn, không làm hỏng gì.
   */
  const [corpusIndex, setCorpusIndex] = useState<CorpusEntry[]>([]);
  /** Câu tóm tắt "N/M niên độ đã qua kiểm cân đối" của doanh nghiệp đang xem. */
  const [corpusSummary, setCorpusSummary] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  /**
   * Kế hoạch của câu hỏi TRƯỚC, để planner kế thừa khi người dùng hỏi nối tiếp
   * ("còn 2023 thì sao?"). Dùng ref chứ không phải state: nó chỉ được ĐỌC lúc gửi request,
   * không ảnh hưởng render, nên đưa vào state chỉ gây render thừa.
   * PHẢI xoá mỗi khi ngữ cảnh đổi (đổi sheet, đổi công ty, nạp file mới) — nếu không, kế hoạch
   * cũ sẽ rò sang câu mới và người dùng nhận số của công ty/kỳ mà họ không hỏi.
   */
  const prevPlanRef = useRef<QueryPlan | null>(null);

  // Roster gọn: mọi công ty nhưng chỉ các Mã số registry — để multi-company tra được công ty thứ 2.
  const compactRoster = useMemo(() => {
    const out: Record<string, Record<string, TidyItem>> = {};
    if (!tidy) return out;
    for (const t of tidy.tickers) {
      const items = tidy.byTicker[t];
      const slim: Record<string, TidyItem> = {};
      for (const c of REGISTRY_CODES) if (items[c]) slim[c] = items[c];
      out[t] = slim;
    }
    return out;
  }, [tidy]);

  function applySheet(wb: ParsedWorkbook, sheet: string): TidyTable {
    // Truyền MA TRẬN THÔ chứ không phải `sheets[...]`: bộ tiền xử lý trong normalize cần thấy các
    // dòng phía trên header để dò header thật, và cần chuỗi nguyên bản để giữ Mã số "01".
    const t = normalize(wb.matrices[sheet] ?? [], sheet);
    t.fileName = wb.fileName;
    setTidy(t);
    setActiveTicker(t.tickers[0] ?? "");
    prevPlanRef.current = null;
    return t;
  }

  function applyWorkbook(wb: ParsedWorkbook) {
    setParsed(wb);
    setMessages([]);
    // Xoá kết quả tự kiểm của kho. Nạp kho rồi nạp file người dùng thì thẻ "Trích từ ViFinQA —
    // 11/11 niên độ đã qua kiểm" còn nguyên và dán nhãn tin cậy lên dữ liệu KHÁC. `loadCorpus`
    // gọi hàm này trước rồi mới đặt lại tóm tắt, nên xoá ở đây là đúng chỗ.
    setCorpusSummary(null);
    const first = wb.sheetNames[0];
    setActiveSheet(first);
    const t = applySheet(wb, first);
    if (t.tickers.length) {
      // Nêu luôn việc bộ tiền xử lý đã làm, để người dùng biết con số họ sắp thấy đến từ đâu.
      const themVao = [
        t.prep && t.prep.headerRow > 0 ? `bỏ ${t.prep.headerRow} dòng tiêu đề` : null,
        t.prep?.maSoSource === "inferred" ? `nhận ra ${t.prep.coverage.mapped}/${t.prep.coverage.total} chỉ tiêu` : null,
        t.prep?.maSoSource === "corrected" ? "đã sửa Mã số lệch chuẩn" : null,
      ].filter(Boolean);
      toast.success(
        `Đã nạp "${wb.fileName}" — ${t.tickers.length} công ty · ${t.periods.join(", ")}` +
          (themVao.length ? ` (${themVao.join(", ")})` : ""),
      );
    } else {
      // Nói THIẾU GÌ, không nói chung chung: `fatal` do preprocess/normalize sinh ra đã nêu rõ.
      toast.error(t.fatal ?? "File chưa nhận ra định dạng báo cáo tài chính. Kiểm tra lại file/sheet.");
    }
  }

  async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    try {
      const wb = await parseWorkbookFromFile(f);
      applyWorkbook(wb);
    } catch {
      toast.error("Không đọc được file. Định dạng hỗ trợ: .xlsx, .xls, .csv");
    }
    e.target.value = "";
  }

  async function loadFromUrl(url: string, fileName: string) {
    setLoadingSample(true);
    try {
      const res = await fetch(url);
      const buf = new Uint8Array(await res.arrayBuffer());
      applyWorkbook(parseWorkbook(buf, fileName));
    } catch {
      toast.error("Không tải được dữ liệu mẫu. Thử lại nhé.");
    } finally {
      setLoadingSample(false);
    }
  }
  useEffect(() => {
    loadCorpusIndex().then(setCorpusIndex);
  }, []);

  /**
   * Nạp một doanh nghiệp từ kho. Dùng LẠI nguyên đường nạp file: CSV sinh ra có đúng dáng mà
   * `normalize` vốn nhận (cột ma_ck / ma_so / chi_tieu + mỗi niên độ một cột), nên không cần thêm
   * một nhánh phân tích thứ hai — và mã Pandas app sinh ra vẫn trỏ tới một file CÓ THẬT.
   */
  async function loadCorpus(ticker: string) {
    setCorpusSummary(null);
    await loadFromUrl(corpusCsvUrl(ticker), `${ticker}.csv`);
    // Đánh dấu dữ liệu đến TỪ KHO. Nhờ cờ này, khi người dùng hỏi chỉ tiêu nằm ngoài 30 chỉ tiêu
    // kho giữ, agent được phép tra thẳng báo cáo gốc (kèm cảnh báo chưa kiểm chứng). File người
    // dùng tự nạp thì không có cờ, vì dữ liệu của họ không nằm trong corpus ViFinQA.
    setTidy((t) => (t ? { ...t, fromCorpus: { ticker } } : t));
    const meta = await loadCorpusMeta(ticker);
    const tom_tat = verifiedSummary(meta);
    setCorpusSummary(tom_tat);
    if (tom_tat) toast.success(`${ticker} — ${tom_tat}.`);
  }

  const loadSample = () => loadFromUrl(SAMPLE_URL, "01_corporate_financial_statements.csv");
  const loadDirty = () => loadFromUrl(DIRTY_URL, "01_dirty_corporate_financial_statements.csv");
  const loadUnitErr = () => loadFromUrl(UNIT_ERR_URL, "01_unit_error_financial_statements.csv");
  const loadMessy = () => loadFromUrl(MESSY_URL, "01_messy_financial_statements.csv");

  function selectSheet(s: string) {
    setCorpusSummary(null);   // đổi sheet là đổi ngữ cảnh — số liệu kiểm cũ không còn nói về nó
    setActiveSheet(s);
    setMessages([]);
    if (parsed) applySheet(parsed, s);
  }

  async function onSend(q: string) {
    const uid = genId();
    const aid = genId();
    // NGÕ CỤT CŨ: chỗ này từng `return` câm khi bảng rỗng — người dùng nạp file lạ rồi gõ câu hỏi
    // và KHÔNG nhận được gì cả, không cả một dòng báo lỗi. Giờ trả lời ngay tại client, nói rõ
    // thiếu gì và cần gì. Không gọi API vì server chặn khi `tickers` rỗng (app/api/agent/route.ts).
    if (!tidy || !tidy.tickers.length) {
      const ly_do = tidy?.fatal ?? "Chưa có dữ liệu nào được nạp.";
      const goi_y = !parsed
        ? "Bạn đính kèm file Excel/CSV ở khung chat, hoặc bấm một trong các nút dữ liệu mẫu bên phải nhé."
        : (parsed.sheetNames.length > 1
            ? "Bạn thử chọn sheet khác ở khung bên phải, "
            : "Bạn kiểm tra lại file: ") +
          "file cần có cột kỳ mang năm (ví dụ \"Năm 2024\"), và cột Mã số hoặc cột tên chỉ tiêu.";
      setMessages((m) => [
        ...m,
        { id: uid, role: "user", text: q },
        { id: aid, role: "assistant", text: `Tôi chưa trả lời được câu hỏi này. ${ly_do} ${goi_y}` },
      ]);
      return;
    }
    setMessages((m) => [...m, { id: uid, role: "user", text: q }, { id: aid, role: "assistant", loading: true, steps: [] }]);
    setSending(true);
    try {
      // Gửi: active ticker đầy đủ + roster gọn các công ty khác (cho multi-company).
      const tidyToSend: TidyTable = {
        ...tidy,
        byTicker: { ...compactRoster, [activeTicker]: tidy.byTicker[activeTicker] },
      };
      const res = await fetch("/api/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tidy: tidyToSend, question: q, activeTicker, prevPlan: prevPlanRef.current }),
      });
      if (!res.ok || !res.body) throw new Error("stream");

      // Đọc NDJSON stream: từng step + result cuối.
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.trim()) continue;
          let msg: { type: string; step?: StepData; result?: AgentResult };
          try {
            msg = JSON.parse(line);
          } catch {
            continue;
          }
          if (msg.type === "step" && msg.step) {
            const step = msg.step;
            setMessages((m) =>
              m.map((x) => (x.id === aid && x.role === "assistant" ? { ...x, steps: upsertStep(x.steps ?? [], step) } : x)),
            );
          } else if (msg.type === "result" && msg.result) {
            const result = msg.result;
            // Chỉ nhớ kế hoạch của câu TRẢ LỜI ĐƯỢC. Kế thừa từ một kế hoạch hỏng thì câu nối
            // tiếp cũng hỏng theo, và người dùng không hiểu vì sao.
            if (result.ok && result.plan) prevPlanRef.current = result.plan;
            setMessages((m) =>
              m.map((x) =>
                x.id === aid
                  ? { id: aid, role: "assistant", result, steps: x.role === "assistant" ? x.steps : undefined }
                  : x,
              ),
            );
          }
        }
      }
    } catch {
      setMessages((m) =>
        m.map((x) => (x.id === aid ? { id: aid, role: "assistant", text: "Có lỗi kết nối. Bạn thử lại nhé." } : x)),
      );
      toast.error("Có lỗi khi hỏi agent. Thử lại nhé.");
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="flex min-h-0 flex-1">
      <section className="flex min-h-0 w-full flex-col border-r border-hairline lg:w-[35%]">
        <ChatPanel
          messages={messages}
          sending={sending}
          hasData={!!tidy?.tickers.length}
          // Đã nạp file thì luôn hỏi được — kể cả khi không dựng được bảng, vì lúc đó câu trả lời
          // là lời giải thích thiếu gì. Xem nhánh đầu `onSend`.
          canAsk={!!tidy?.tickers.length || !!parsed}
          onSend={onSend}
          onAttach={() => fileRef.current?.click()}
        />
      </section>
      <section className="hidden min-h-0 flex-col bg-surface-soft lg:flex lg:w-[65%]">
        <DataPanel
          parsed={parsed}
          activeSheet={activeSheet}
          onSelectSheet={selectSheet}
          tidy={tidy}
          activeTicker={activeTicker}
          onSelectTicker={(t) => { prevPlanRef.current = null; setActiveTicker(t); }}
          onLoadSample={loadSample}
          onLoadDirty={loadDirty}
          onLoadUnitErr={loadUnitErr}
          onLoadMessy={loadMessy}
          loadingSample={loadingSample}
          corpusIndex={corpusIndex}
          onLoadCorpus={loadCorpus}
          corpusSummary={corpusSummary}
        />
      </section>
      <input ref={fileRef} type="file" accept=".xlsx,.xls,.csv" hidden onChange={onFile} />
    </div>
  );
}
