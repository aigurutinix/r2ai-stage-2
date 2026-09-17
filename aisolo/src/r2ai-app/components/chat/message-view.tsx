"use client";

import { useState } from "react";
import { FileText, Code2, Lightbulb, Copy, Check, AlertTriangle, ShieldAlert } from "lucide-react";
import type { AgentResult } from "@/lib/financial/types";
import { GptChart, type GptChartType } from "@/components/data/gpt-chart";

/** Card code Pandas: bấm "Xem" để mở, có nút Copy. */
function PandasCard({ code }: { code: string }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard bị chặn — bỏ qua */
    }
  }
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 text-xs font-medium text-muted hover:text-ink"
      >
        <Code2 className="h-3.5 w-3.5" /> {open ? "Ẩn" : "Xem"} truy vấn Pandas
      </button>
      {open && (
        <div className="mt-1.5 overflow-hidden rounded-md border border-hairline bg-surface-soft">
          <div className="flex items-center justify-between border-b border-hairline bg-surface-strong px-3 py-1.5">
            <span className="font-mono text-[11px] text-muted-soft">pandas · chạy được trên CSV</span>
            <button
              type="button"
              onClick={copy}
              className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-medium text-muted hover:bg-surface-strong hover:text-primary"
            >
              {copied ? <Check className="h-3 w-3 text-primary" /> : <Copy className="h-3 w-3" />}
              {copied ? "Đã copy" : "Copy"}
            </button>
          </div>
          <pre className="overflow-x-auto px-3 py-2 font-mono text-xs leading-relaxed text-ink">
            <code>{code}</code>
          </pre>
        </div>
      )}
    </div>
  );
}

/** Hiển thị kết quả agent: số + nhận định + chart + citation + Pandas. */
export function AssistantResult({ result }: { result: AgentResult }) {
  return (
    <div className="space-y-3">
      <p className="leading-relaxed text-body">{result.answer}</p>

      {result.ok && result.valueLabel !== "—" && (
        <div className="font-mono text-[32px] font-bold leading-none tracking-tight text-primary">
          {result.valueLabel}
        </div>
      )}

      {/* MỨC TIN CẬY của con số. Số lấy từ kho đã qua đẳng thức kế toán; số tra thẳng báo cáo gốc
          thì chưa. Hai thứ đó KHÁC NHAU và người xem cần biết mình đang đứng ở đâu — giấu đi là
          bán rẻ chính điểm mạnh của sản phẩm. Đặt ngay dưới con số, trước mọi cảnh báo khác. */}
      {result.unverified && (
        <div className="flex items-start gap-2 rounded-md border border-warning/30 bg-warning-soft px-3 py-2 text-sm text-body">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" />
          <span>
            Số này <strong>đọc trực tiếp từ báo cáo gốc</strong>, chưa qua kiểm đẳng thức kế toán —
            khác với các chỉ tiêu lấy từ kho đã kiểm chứng.
            {result.liveSource && (
              <> Nguồn: {result.liveSource.doc}
                {result.liveSource.line ? `, dòng ${result.liveSource.line}` : ""} · «{result.liveSource.label}».</>
            )}
          </span>
        </div>
      )}

      {/* Tự kiểm ĐÁP ÁN — đặt ngay dưới con số vì nó nói về chính con số đó. Phải đứng TRƯỚC
          cảnh báo dữ liệu vào: nếu kết quả đã sai chắc chắn thì đó là điều cần đọc đầu tiên. */}
      {result.auditWarning && (
        <div className="flex items-start gap-2 rounded-md border border-negative/30 bg-negative-soft px-3 py-2 text-sm text-negative">
          <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" />
          <span>{result.auditWarning}</span>
        </div>
      )}

      {result.dataQualityWarning && (
        <div className="flex items-start gap-2 rounded-md bg-negative-soft px-3 py-2 text-sm text-negative">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>{result.dataQualityWarning}</span>
        </div>
      )}

      {result.insight && (
        <div className="flex items-start gap-2 rounded-md bg-primary-soft px-3 py-2 text-sm text-body">
          <Lightbulb className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
          <span>{result.insight}</span>
        </div>
      )}

      {result.chart && (
        <div className="rounded-md border border-hairline bg-canvas p-2">
          <GptChart type={result.chart.type as GptChartType} height={result.chart.height} config={result.chart.config} />
        </div>
      )}

      {result.citations.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {result.citations.map((c, i) => (
            <span
              key={i}
              className="inline-flex items-center gap-1 rounded-full bg-surface-strong px-2.5 py-1 text-xs font-medium text-muted"
              title={c.itemName}
            >
              <FileText className="h-3.5 w-3.5" />
              {c.ticker !== "(toàn bộ)" ? `${c.ticker} · ` : ""}Mã số {c.maSo} · {c.period}
            </span>
          ))}
        </div>
      )}

      {result.pandasCode && <PandasCard code={result.pandasCode} />}

      <p className="text-xs text-muted-soft">
        Nguồn tính: Qwen3.5-4B (local)
      </p>
    </div>
  );
}
