"use client";

import {
  Activity,
  BarChart3,
  Check,
  CheckCircle2,
  ChevronDown,
  Code2,
  Download,
  FileSpreadsheet,
  FileUp,
  MessageSquarePlus,
  PanelLeft,
  Plus,
  Search,
  Send,
  ShieldCheck,
  Trash2,
  X,
} from "lucide-react";
import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import type {
  AgentTraceStage,
  AnsweredResponse,
  Citation,
  HealthResponse,
  ProductResponse,
  TablePreviewResponse,
  VerifiedSourceCell,
} from "@/lib/types";

type Mode = "chat" | "batch";

type Run = {
  id: string;
  question: string;
  state: "running" | "done" | "error";
  response?: ProductResponse;
  error?: string;
  liveTrace?: AgentTraceStage[];
  startedAt: number;
};

type BatchRecord = {
  id: string | number;
  question: string;
  status: "answered" | "refused" | "fallback" | "error";
  response?: ProductResponse;
  error?: string;
};

const SUGGESTIONS = [
  "Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC) là bao nhiêu triệu đồng?",
  "Tổng tài sản của FPT năm 2024 là bao nhiêu tỷ đồng?",
  "Năm 2022, trong nhóm HPG, HSG, MSR và NKG, công ty nào có hệ số thanh toán nhanh thấp nhất?",
];

function formatNumber(value: number) {
  return new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 6 }).format(value);
}

function compactTitle(value: string) {
  const clean = value.replace(/\s+/g, " ").trim();
  return clean.length > 42 ? `${clean.slice(0, 42)}…` : clean;
}

function answerMode(answer: AnsweredResponse) {
  if (answer.verification.mode === "deterministic_compiler") return "Compiled";
  if (answer.verification.mode === "verified_registry_paraphrase") return "Paraphrase verified";
  if (answer.verification.mode === "verified_registry") return "Verified replay";
  return "Generated + replayed";
}

function tablePresentation(preview: TablePreviewResponse) {
  const genericColumns = preview.columns.every((column, index) => {
    const clean = String(column ?? "").trim().toLocaleLowerCase("vi");
    return clean === String(index) || clean === "" || clean.startsWith("unnamed:");
  });
  if (!genericColumns || !preview.rows.length) {
    return { columns: preview.columns, rows: preview.rows, promoted: false, promotedRows: 0 };
  }

  const meaningful = (value: unknown) => {
    const clean = String(value ?? "").trim();
    return clean && !["-", "—"].includes(clean);
  };
  const headerLike = (value: unknown) => {
    const clean = String(value ?? "").trim();
    return Boolean(clean && (/[A-Za-zÀ-ỹĐđ]/u.test(clean) || /^(?:19|20)\d{2}$/.test(clean)));
  };
  const headerRows: string[][] = [];
  for (const row of preview.rows.slice(0, 4)) {
    const candidates = row.slice(1).filter(meaningful);
    if (!candidates.length || !candidates.every(headerLike)) break;
    headerRows.push(row);
  }
  if (!headerRows.length) {
    return { columns: preview.columns, rows: preview.rows, promoted: false, promotedRows: 0 };
  }
  const columns = preview.columns.map((_column, index) => {
    if (index === 0) return "Chỉ tiêu";
    const labels = headerRows
      .map((row) => String(row[index] ?? "").trim())
      .filter((value, position, all) => Boolean(meaningful(value)) && all.indexOf(value) === position)
      .map((value) => value.replace(/^(20\d{2})(VND)$/i, "$1 · $2"));
    return labels.length ? labels.join(" · ") : `Cột ${index + 1}`;
  });
  return {
    columns,
    rows: preview.rows.slice(headerRows.length),
    promoted: true,
    promotedRows: headerRows.length,
  };
}

function selectedRowLabel(answer: AnsweredResponse | null) {
  if (!answer) return "";
  const match = answer.pandas_query.match(/#\s*LẤY DÒNG:\s*['\"]([^'\"]+)['\"]/i);
  return match?.[1]?.trim().toLocaleLowerCase("vi") ?? "";
}

function verifiedCellsForTable(answer: AnsweredResponse | null, tableRef: string) {
  if (!answer) return [];
  return (answer.verification.source_cells ?? []).filter((cell): cell is VerifiedSourceCell => (
    cell.table_ref === tableRef
    && Number.isInteger(Number(cell.row_idx))
    && Number.isInteger(Number(cell.col_idx))
    && cell.verified !== false
  ));
}

function parseSseBlock(block: string): { event: string; data: unknown } | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!data.length) return null;
  return { event, data: JSON.parse(data.join("\n")) };
}

function asTraceStage(value: unknown): AgentTraceStage | null {
  if (!value || typeof value !== "object") return null;
  const stage = value as Partial<AgentTraceStage>;
  if (!stage.stage || !stage.title || !stage.status) return null;
  return {
    stage: stage.stage,
    title: stage.title,
    status: stage.status,
    elapsed_ms: Number(stage.elapsed_ms ?? 0),
    detail: String(stage.detail ?? ""),
    metadata: stage.metadata && typeof stage.metadata === "object" ? stage.metadata : {},
  };
}

function upsertTrace(stages: AgentTraceStage[], incoming: AgentTraceStage) {
  const next = [...stages];
  const index = next.findIndex((stage) => stage.stage === incoming.stage && stage.status !== "completed" && stage.status !== "error");
  if (index >= 0) next[index] = incoming;
  else if (incoming.status === "running") {
    const completedIndex = next.findIndex((stage) => stage.stage === incoming.stage);
    if (completedIndex >= 0) next[completedIndex] = incoming;
    else next.push(incoming);
  } else next.push(incoming);
  return next;
}

function TracePanel({ stages, live = false }: { stages: AgentTraceStage[]; live?: boolean }) {
  if (!stages.length) return null;
  return (
    <section className="tracePanel" aria-label="Luồng xử lý thật">
      <header><span className={live ? "live" : ""} /><strong>Luồng xử lý</strong><small>{live ? "Live backend trace" : "Measured backend trace"}</small></header>
      <div className="traceList">
        {stages.map((stage, index) => (
          <div className="traceRow" key={`${stage.stage}-${index}`}>
            <span>{index + 1}</span>
            <div><strong>{stage.title}</strong><small>{stage.detail}</small></div>
            <em>{stage.elapsed_ms} ms</em>
            {stage.status === "completed" ? <CheckCircle2 size={16} /> : <span className="stageSpinner" />}
          </div>
        ))}
      </div>
    </section>
  );
}

function SourceSection({ citations, selectedRef, onSelect }: {
  citations: Citation[];
  selectedRef?: string;
  onSelect: (citation: Citation) => void;
}) {
  return (
    <section className="sources">
      <div className="sourcesHead">
        <strong>Nguồn dữ liệu</strong>
        <span><ShieldCheck size={14} /> Citation bound</span>
      </div>
      <div className="sourceGrid">
        {citations.map((citation) => (
          <button
            className={selectedRef === citation.table_ref ? "sourceCard active" : "sourceCard"}
            key={citation.table_ref}
            onClick={() => onSelect(citation)}
          >
            <span className="sourceIcon"><FileSpreadsheet size={17} /></span>
            <span className="sourceText">
              <strong>{citation.ticker} · {citation.year} · {citation.scope}</strong>
              <small>{citation.section}</small>
              <em>{citation.table_ref}</em>
            </span>
            <ChevronDown size={15} />
          </button>
        ))}
      </div>
    </section>
  );
}

function EvidencePanel({ answer, citation, open, onToggle, onSelect }: {
  answer: AnsweredResponse | null;
  citation: Citation | null;
  open: boolean;
  onToggle: () => void;
  onSelect: (citation: Citation) => void;
}) {
  const [preview, setPreview] = useState<TablePreviewResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [tableOpen, setTableOpen] = useState(false);
  const tableWrapRef = useRef<HTMLDivElement>(null);
  const presentation = preview ? tablePresentation(preview) : null;
  const activeRowLabel = selectedRowLabel(answer);
  const verifiedCells = citation ? verifiedCellsForTable(answer, citation.table_ref) : [];
  const verifiedRows = new Set(verifiedCells.map((cell) => Number(cell.row_idx)));

  useEffect(() => {
    if (!open || !citation) {
      setPreview(null);
      setError("");
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    setError("");
    fetch(`/api/table?ref=${encodeURIComponent(citation.table_ref)}`, { cache: "no-store", signal: controller.signal })
      .then(async (response) => {
        const data = await response.json();
        if (!response.ok || data.status !== "ok") throw new Error(data.error ?? "Không tải được bảng nguồn");
        setPreview(data as TablePreviewResponse);
      })
      .catch((reason) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(reason instanceof Error ? reason.message : "Không tải được bảng nguồn");
      })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [citation, open]);

  useEffect(() => {
    setTableOpen(false);
  }, [citation?.table_ref, open]);

  useEffect(() => {
    if (!tableOpen) return;
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") setTableOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [tableOpen]);

  useEffect(() => {
    if (!tableOpen || !preview) return;
    const frame = window.requestAnimationFrame(() => {
      const wrap = tableWrapRef.current;
      const target = wrap?.querySelector<HTMLElement>(".selectedSourceCell")
        ?? wrap?.querySelector<HTMLElement>(".selectedSourceRow");
      if (!target || !wrap) return;
      const left = Math.min(
        Math.max(0, wrap.scrollWidth - wrap.clientWidth),
        Math.max(0, target.offsetLeft + target.offsetWidth / 2 - wrap.clientWidth / 2),
      );
      const top = Math.max(0, target.offsetTop - (wrap.clientHeight - target.offsetHeight) / 2);
      wrap.scrollTo({ behavior: "smooth", left, top });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [tableOpen, preview, citation?.table_ref]);

  return (
    <>
      <aside className={open ? "evidencePanel open" : "evidencePanel closed"} aria-label="Evidence panel">
        <button className="evidenceToggle" onClick={onToggle} aria-label={open ? "Thu evidence panel" : "Mở evidence panel"}><PanelLeft size={18} /></button>
        {open && (
          <div className="evidenceContent">
            <header><div><small>Evidence</small><strong>Nguồn kiểm chứng</strong></div></header>
            {!answer || !citation ? (
              <div className="evidenceEmpty"><span><FileSpreadsheet size={22} /></span><strong>Chưa có nguồn</strong><p>Chạy một câu hỏi. Bảng được chương trình dùng sẽ xuất hiện tại đây.</p></div>
            ) : (
              <>
                <div className="evidenceStats"><div><strong>{answer.retrieval.documents.length}</strong><span>Docs</span></div><div><strong>{answer.retrieval.tables.length}</strong><span>Tables</span></div><div><strong>{answer.citations.length}</strong><span>Cited</span></div></div>
                {answer.citations.length > 1 && <div className="sourceSwitcher">{answer.citations.map((item, index) => <button className={item.table_ref === citation.table_ref ? "active" : ""} onClick={() => onSelect(item)} key={item.table_ref}>{index + 1}</button>)}</div>}
                <div className="evidenceIdentity"><span>{citation.ticker}</span><div><strong>{citation.company}</strong><small>{citation.scope} · FY {citation.year}</small></div></div>
                <div className="evidenceMeta"><strong>{citation.document}</strong><div><span>Trang {citation.page ?? "—"}</span><span>Dòng {citation.table_line ?? "—"}</span><span>Table {citation.table_ref.split("|").at(-1)}</span></div></div>
                <div className="evidenceSection"><small>Chỉ tiêu trong bảng</small><p>{citation.section}</p></div>
                <button className="tableLaunch" disabled={loading || !preview} onClick={() => setTableOpen(true)}>
                  <span><FileSpreadsheet size={18} /></span>
                  <div><strong>Bảng dữ liệu đã dùng</strong><small>{loading ? "Đang tải…" : error ? error : preview ? `${preview.row_count} dòng × ${preview.column_count} cột${verifiedCells.length ? ` · ${verifiedCells.length} ô đã xác minh` : ""}` : "Chưa sẵn sàng"}</small></div>
                  <em>Mở bảng lớn</em>
                </button>
              </>
            )}
          </div>
        )}
      </aside>
      {tableOpen && preview && citation && (
        <div className="tableModalBackdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setTableOpen(false); }}>
          <section className="tableModal" role="dialog" aria-modal="true" aria-label={`Bảng nguồn ${citation.table_ref}`}>
            <header>
              <div><small>Bảng nguồn đã dùng</small><strong>{citation.company} · {citation.year} · {citation.scope}</strong><span>{citation.table_ref} · {preview.row_count} × {preview.column_count}{verifiedCells.length ? ` · ${verifiedCells.length} ô lineage đã xác minh` : ""}</span></div>
              <button onClick={() => setTableOpen(false)} aria-label="Đóng bảng lớn"><X size={20} /></button>
            </header>
            <div className="tableModalWrap" ref={tableWrapRef}><table><thead><tr>{(presentation?.columns ?? preview.columns).map((column, index) => <th key={`${column}-${index}`}>{column || `Cột ${index + 1}`}</th>)}</tr></thead><tbody>{(presentation?.rows ?? preview.rows).map((row, rowIndex) => {
              const rowText = row.map((cell) => String(cell ?? "")).join(" ").toLocaleLowerCase("vi");
              const sourceRowIndex = rowIndex + (presentation?.promotedRows ?? 0);
              const exactRow = verifiedRows.has(sourceRowIndex);
              const highlighted = exactRow || (!verifiedRows.size && Boolean(activeRowLabel && rowText.includes(activeRowLabel)));
              return <tr className={highlighted ? "selectedSourceRow" : ""} key={rowIndex}>{row.map((cell, cellIndex) => {
                const exactCell = verifiedCells.some((source) => Number(source.row_idx) === sourceRowIndex && Number(source.col_idx) === cellIndex);
                return <td className={exactCell ? "selectedSourceCell" : ""} key={cellIndex}>{cell || "—"}</td>;
              })}</tr>;
            })}</tbody></table></div>
          </section>
        </div>
      )}
    </>
  );
}

function AnswerMessage({ response, showTrace, selectedRef, onCitation }: {
  response: ProductResponse;
  showTrace: boolean;
  selectedRef?: string;
  onCitation: (citation: Citation) => void;
}) {
  if (response.status === "refused") {
    return (
      <div className="assistantMessage">
        <div className="refusal"><ShieldCheck size={19} /><div><strong>Chưa đủ căn cứ để trả lời</strong><p>{response.refusal.message}</p><code>{response.refusal.code}</code></div></div>
      </div>
    );
  }
  const answer = response;
  return (
    <div className="assistantMessage">
      <div className={showTrace ? "traceMotion open" : "traceMotion closed"}><div><TracePanel stages={answer.trace ?? []} /></div></div>
      <div className="answerLead">
        <span><Check size={16} /></span>
        <div><small>{answerMode(answer)}</small><strong>{formatNumber(answer.answer)} <em>{answer.unit}</em></strong></div>
      </div>
      <p className="answerCopy">
        Kết quả được tính từ {answer.retrieval.tables.length} bảng nguồn, chạy lại bằng Pandas và khóa với {answer.citations.length} citation.
      </p>
      <details className="codeDetails">
        <summary><Code2 size={16} /> Xem code Pandas đã thực thi</summary>
        <pre><code>{answer.pandas_query}</code></pre>
      </details>
      <SourceSection citations={answer.citations} selectedRef={selectedRef} onSelect={onCitation} />
    </div>
  );
}

export default function Home() {
  const [mode, setMode] = useState<Mode>("chat");
  const [runs, setRuns] = useState<Run[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [showTrace, setShowTrace] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [evidenceOpen, setEvidenceOpen] = useState(true);
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [batchName, setBatchName] = useState("");
  const [batchItems, setBatchItems] = useState<Array<{ id: string | number; question: string }>>([]);
  const [batchResults, setBatchResults] = useState<Array<BatchRecord | null>>([]);
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchJobId, setBatchJobId] = useState("");
  const [batchError, setBatchError] = useState("");
  const [submissionReady, setSubmissionReady] = useState(false);
  const [submissionHash, setSubmissionHash] = useState("");
  const [submissionExporting, setSubmissionExporting] = useState(false);
  const batchInput = useRef<HTMLInputElement>(null);
  const active = runs.find((run) => run.id === activeId) ?? null;
  const answered = active?.response?.status === "answered" ? active.response : null;
  const visibleRuns = useMemo(() => {
    const needle = searchTerm.trim().toLocaleLowerCase("vi");
    return runs.filter((run) => !needle || run.question.toLocaleLowerCase("vi").includes(needle));
  }, [runs, searchTerm]);
  const completedBatch = batchResults.filter(Boolean).length;

  useEffect(() => {
    const load = () => fetch("/api/health", { cache: "no-store" }).then((response) => response.json()).then(setHealth).catch(() => setHealth(null));
    load();
    const timer = window.setInterval(load, 15_000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    setSelectedCitation(answered?.citations[0] ?? null);
  }, [answered?.trace_id]);

  const submit = async (question = query) => {
    const clean = question.trim();
    if (!clean) return;
    const id = crypto.randomUUID();
    const run: Run = { id, question: clean, state: "running", startedAt: Date.now() };
    setMode("chat");
    setRuns((current) => [run, ...current]);
    setActiveId(id);
    setQuery("");
    try {
      const response = await fetch("/api/ask/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: clean }),
      });
      if (!response.ok || !response.body) throw new Error(`Pipeline stream HTTP ${response.status}`);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finalResponse: ProductResponse | null = null;
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split(/\r?\n\r?\n/);
        buffer = blocks.pop() ?? "";
        for (const block of blocks) {
          const event = parseSseBlock(block);
          if (!event) continue;
          if (event.event === "status") {
            const stage = asTraceStage(event.data);
            if (stage) setRuns((current) => current.map((item) => item.id === id ? { ...item, liveTrace: upsertTrace(item.liveTrace ?? [], stage) } : item));
          } else if (event.event === "result") {
            finalResponse = event.data as ProductResponse;
          } else if (event.event === "error") {
            const payload = event.data as { detail?: string };
            throw new Error(payload.detail ?? "Pipeline stream error");
          }
        }
      }
      if (!finalResponse) throw new Error("Pipeline stream kết thúc nhưng không có result");
      const resolved = finalResponse;
      setRuns((current) => current.map((item) => item.id === id ? { ...item, state: "done", response: resolved, liveTrace: resolved.status === "answered" ? resolved.trace : item.liveTrace } : item));
    } catch (reason) {
      setRuns((current) => current.map((item) => item.id === id ? { ...item, state: "error", error: reason instanceof Error ? reason.message : "Lỗi không xác định" } : item));
    }
  };

  const newAnalysis = () => {
    setMode("chat");
    setActiveId(null);
    setQuery("");
    setSelectedCitation(null);
  };

  const loadBatchFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    try {
      const payload = JSON.parse(await file.text());
      if (!Array.isArray(payload)) throw new Error("File phải là JSON array");
      const seen = new Set<string>();
      const items = payload.map((item, index) => {
        if (!item || typeof item !== "object" || typeof item.question !== "string" || !item.question.trim()) throw new Error(`Câu ${index + 1} thiếu question`);
        const id = typeof item.id === "string" || typeof item.id === "number" ? item.id : index + 1;
        const key = String(id);
        if (seen.has(key)) throw new Error(`Trùng ID ${key}`);
        seen.add(key);
        return { id, question: item.question.trim() };
      });
      setBatchName(file.name);
      setBatchItems(items);
      setBatchResults(Array(items.length).fill(null));
      setBatchJobId("");
      setBatchError("");
      setSubmissionReady(false);
      setSubmissionHash("");
    } catch (reason) {
      setBatchName(reason instanceof Error ? reason.message : "Không đọc được file");
      setBatchItems([]);
      setBatchResults([]);
      setBatchJobId("");
      setBatchError("");
      setSubmissionReady(false);
      setSubmissionHash("");
    }
  };

  const runBatch = async () => {
    if (!batchItems.length || batchRunning) return;
    setBatchRunning(true);
    setBatchError("");
    setSubmissionReady(false);
    setSubmissionHash("");
    const results: Array<BatchRecord | null> = Array(batchItems.length).fill(null);
    setBatchResults(results);
    try {
      const createdResponse = await fetch("/api/batch/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ items: batchItems, workers: 4, fallback_on_refusal: false }),
      });
      const created = await createdResponse.json();
      if (!createdResponse.ok || !created.job_id) throw new Error(created.error ?? "Không tạo được batch job");
      const jobId = String(created.job_id);
      setBatchJobId(jobId);
      const stream = await fetch(`/api/batch/jobs/${encodeURIComponent(jobId)}/events?cursor=0`, { cache: "no-store" });
      if (!stream.ok || !stream.body) throw new Error(`Batch stream HTTP ${stream.status}`);
      const reader = stream.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split(/\r?\n\r?\n/);
        buffer = blocks.pop() ?? "";
        for (const block of blocks) {
          const event = parseSseBlock(block);
          if (!event) continue;
          const data = event.data as { last?: { index?: number; id?: string | number; status?: BatchRecord["status"]; error?: string } };
          if (event.event === "item" && data.last && typeof data.last.index === "number") {
            const index = data.last.index;
            const item = batchItems[index];
            if (item) {
              results[index] = {
                ...item,
                status: data.last.status ?? "error",
                error: data.last.error ?? undefined,
              };
              setBatchResults([...results]);
            }
          } else if (event.event === "error") {
            throw new Error("Batch job thất bại");
          }
        }
      }
    } catch (reason) {
      setBatchError(reason instanceof Error ? reason.message : "Batch job thất bại");
    } finally {
      setBatchRunning(false);
    }
  };

  const downloadBatch = () => {
    const anchor = document.createElement("a");
    anchor.href = `/api/batch/jobs/${encodeURIComponent(batchJobId)}/download`;
    anchor.click();
  };

  const exportSubmission = async () => {
    if (!batchJobId || submissionExporting) return;
    setSubmissionExporting(true);
    setBatchError("");
    try {
      const response = await fetch(`/api/batch/jobs/${encodeURIComponent(batchJobId)}/submission`, { method: "POST" });
      const data = await response.json();
      if (!response.ok || data.status !== "PASS") throw new Error(data.error ?? "Submission export không vượt strict validation");
      setSubmissionHash(String(data.archive?.sha256 ?? ""));
      setSubmissionReady(true);
    } catch (reason) {
      setBatchError(reason instanceof Error ? reason.message : "Không xuất được submission");
    } finally {
      setSubmissionExporting(false);
    }
  };

  const downloadSubmission = () => {
    const anchor = document.createElement("a");
    anchor.href = `/api/batch/jobs/${encodeURIComponent(batchJobId)}/submission/download`;
    anchor.click();
  };

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    void submit();
  };

  return (
    <main className={`shell ${sidebarOpen ? "" : "sidebarClosed"} ${evidenceOpen ? "evidenceOpen" : "evidenceClosed"}`}>
      {sidebarOpen && (
        <aside className="sidebar">
          <div className="brand">
            <div><strong>KINGPRO</strong><small>Financial QA</small></div>
            <button className="iconButton" onClick={() => setSidebarOpen(false)} aria-label="Ẩn sidebar"><PanelLeft size={19} /></button>
          </div>
          <nav>
            <button className="navButton active" onClick={newAnalysis}><MessageSquarePlus size={20} /> Phân tích mới</button>
            <button className="navButton" onClick={() => setMode("batch")}><FileSpreadsheet size={20} /> Chạy batch test</button>
            <button className="navButton" onClick={() => setSearchOpen((value) => !value)}><Search size={20} /> Tìm lịch sử</button>
          </nav>
          {searchOpen && <input className="searchInput" value={searchTerm} onChange={(event) => setSearchTerm(event.target.value)} placeholder="Nhập từ khóa" autoFocus />}
          <div className="sectionTitle">Gần đây</div>
          <div className="history">
            {!visibleRuns.length && <span>Chưa có phân tích</span>}
            {visibleRuns.map((run) => (
              <div className={run.id === activeId ? "historyRow selected" : "historyRow"} key={run.id}>
                <button onClick={() => { setMode("chat"); setActiveId(run.id); }}><strong>{compactTitle(run.question)}</strong><small>{run.state === "running" ? "Đang chạy" : run.response?.status ?? run.state}</small></button>
                <button className="deleteButton" onClick={() => { setRuns((current) => current.filter((item) => item.id !== run.id)); if (activeId === run.id) setActiveId(null); }} aria-label="Xóa"><Trash2 size={15} /></button>
              </div>
            ))}
          </div>
          <div className="profile">
            <span>KP</span><div><strong>KINGPRO</strong><small><i /> Private-ready workspace</small></div>
          </div>
        </aside>
      )}

      <section className="workspace">
        <header className="topbar">
          {!sidebarOpen && <button className="iconButton sidebarToggle" onClick={() => setSidebarOpen(true)} aria-label="Hiện sidebar"><PanelLeft size={19} /></button>}
          <div className={mode === "batch" ? "modeSwitch batchActive" : "modeSwitch"}>
            <button className={mode === "chat" ? "active" : ""} onClick={() => setMode("chat")}>Chat</button>
            <button className={mode === "batch" ? "active" : ""} onClick={() => setMode("batch")}>Batch</button>
          </div>
          <button className={showTrace ? "traceToggle active" : "traceToggle"} onClick={() => setShowTrace((value) => !value)} aria-pressed={showTrace}><Activity size={17} /> <span>{showTrace ? "Ẩn luồng" : "Hiện luồng"}</span></button>
        </header>

        {mode === "batch" ? (
          <div className="batchPage">
            <span className="heroIcon"><FileSpreadsheet size={25} /></span>
            <h1>Chạy batch test</h1>
            <p>Tạo server job dùng chính runner checkpoint nguyên tử, chạy bốn câu song song và tiếp tục trên đĩa dù tab trình duyệt bị đóng.</p>
            <div className="batchCard">
              <button className="uploadButton" onClick={() => batchInput.current?.click()}><FileUp size={18} /> Chọn questions.json</button>
              <input ref={batchInput} type="file" accept="application/json,.json" hidden onChange={loadBatchFile} />
              <div><strong>{batchName || "Chưa chọn file"}</strong><small>{batchItems.length ? `${batchItems.length} câu hợp lệ` : "JSON array gồm id và question"}</small></div>
              <button className="runBatchButton" disabled={!batchItems.length || batchRunning} onClick={() => void runBatch()}>{batchRunning ? `${completedBatch}/${batchItems.length}` : "Bắt đầu"}</button>
            </div>
            {batchError && <div className="batchError">{batchError}</div>}
            {batchItems.length > 0 && <div className="batchProgress"><i style={{ width: `${(completedBatch / batchItems.length) * 100}%` }} /></div>}
            {completedBatch > 0 && (
              <div className="batchSummary">
                <div><strong>{completedBatch}</strong><span>Hoàn tất</span></div>
                <div><strong>{batchResults.filter((item) => item?.status === "answered" || item?.status === "fallback").length}</strong><span>Answered</span></div>
                <div><strong>{batchResults.filter((item) => item?.status === "refused").length}</strong><span>Refused</span></div>
                <div><strong>{batchResults.filter((item) => item?.status === "error").length}</strong><span>Error</span></div>
                {!batchRunning && completedBatch === batchItems.length && batchJobId && <button onClick={downloadBatch}><Download size={16} /> Trace JSON</button>}
                {!batchRunning && completedBatch === batchItems.length && batchJobId && !submissionReady && <button className="submissionButton" disabled={submissionExporting} onClick={() => void exportSubmission()}><FileSpreadsheet size={16} /> {submissionExporting ? "Đang kiểm tra…" : "Xuất ZIP nộp"}</button>}
                {submissionReady && <button className="submissionButton ready" onClick={downloadSubmission}><Check size={16} /> Tải submission ZIP</button>}
              </div>
            )}
            <code className="runnerHint">{submissionHash ? `ZIP SHA-256: ${submissionHash}` : batchJobId ? `Server job: ${batchJobId}` : "Durable runner · ordered output · atomic checkpoint · exact fallback only"}</code>
          </div>
        ) : (
          <>
            <div className="messages">
              {!active ? (
                <div className="emptyState">
                  <span className="heroIcon"><BarChart3 size={25} /></span>
                  <h1>Anh muốn phân tích báo cáo nào?</h1>
                  <p>KINGPRO tìm đúng bảng, chạy Pandas và chỉ trả lời khi nguồn có thể kiểm chứng.</p>
                  <div className="suggestions">{SUGGESTIONS.map((suggestion) => <button key={suggestion} onClick={() => void submit(suggestion)}>{suggestion}</button>)}</div>
                </div>
              ) : (
                <div className="conversation">
                  <div className="userMessage">{active.question}</div>
                  {active.state === "running" && (
                    active.liveTrace?.length ? (
                      <div className={showTrace ? "traceMotion open" : "traceMotion closed"}><div><TracePanel stages={active.liveTrace} live /></div></div>
                    ) : <div className="loadingMessage"><span /><span /><span /> Đang khởi tạo pipeline…</div>
                  )}
                  {active.state === "error" && <div className="errorMessage">{active.error}</div>}
                  {active.response && <AnswerMessage response={active.response} showTrace={showTrace} selectedRef={selectedCitation?.table_ref} onCitation={(citation) => { setSelectedCitation(citation); setEvidenceOpen(true); }} />}
                </div>
              )}
            </div>
            <div className="composerWrap">
              <form className="composer" onSubmit={onSubmit}>
                <button className="addButton" type="button" onClick={newAnalysis} aria-label="Phân tích mới"><Plus size={22} /></button>
                <textarea value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void submit(); } }} placeholder="Hỏi về báo cáo tài chính…" rows={1} />
                <span className="verified"><ShieldCheck size={16} /> Verified <ChevronDown size={14} /></span>
                <button className="sendButton" disabled={!query.trim()} type="submit" aria-label="Gửi"><Send size={18} /></button>
              </form>
              <small className="composerNote">{health?.demo_readiness?.stage_safe_now ? "Stage-safe · " : ""}{health?.replay_entries ?? 0} verified programs · nguồn thật hoặc từ chối</small>
            </div>
          </>
        )}
      </section>
      <EvidencePanel answer={mode === "chat" ? answered : null} citation={mode === "chat" ? selectedCitation : null} open={evidenceOpen} onToggle={() => setEvidenceOpen((value) => !value)} onSelect={setSelectedCitation} />
    </main>
  );
}
