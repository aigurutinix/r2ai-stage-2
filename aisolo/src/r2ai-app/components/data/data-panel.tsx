import { AlertTriangle, CheckCircle2, Droplets, FileSpreadsheet, ShieldAlert, ShieldCheck, Sparkles, Table2, Wand2 } from "lucide-react";
import type { ParsedWorkbook } from "@/lib/xlsx-client";
import type { TidyTable } from "@/lib/financial/types";
import { cn } from "@/lib/utils";
import { formatMoney } from "@/lib/financial/clean";
import { computeKpis } from "@/lib/financial/kpi";
import { validateTicker } from "@/lib/financial/validate";
import { KpiRow } from "./kpi-row";
import { Dashboard } from "./dashboard";

import { CorpusPicker } from "@/components/data/corpus-picker";
import type { CorpusEntry } from "@/lib/financial/corpus";

interface Props {
  parsed: ParsedWorkbook | null;
  activeSheet: string;
  onSelectSheet: (s: string) => void;
  tidy: TidyTable | null;
  activeTicker: string;
  onSelectTicker: (t: string) => void;
  onLoadSample: () => void;
  onLoadDirty: () => void;
  onLoadUnitErr: () => void;
  onLoadMessy: () => void;
  loadingSample: boolean;
  /** Danh mục doanh nghiệp trích sẵn; rỗng khi chưa sinh kho → mục tra cứu tự ẩn. */
  corpusIndex: CorpusEntry[];
  onLoadCorpus: (ticker: string) => void;
  /** "N/M niên độ đã qua kiểm cân đối kế toán" của doanh nghiệp đang xem, nếu đến từ kho. */
  corpusSummary: string | null;
}

const MAX_ROWS = 80;

export function DataPanel({
  parsed,
  activeSheet,
  onSelectSheet,
  tidy,
  activeTicker,
  onSelectTicker,
  onLoadSample,
  onLoadDirty,
  onLoadUnitErr,
  onLoadMessy,
  loadingSample,
  corpusIndex,
  onLoadCorpus,
  corpusSummary,
}: Props) {
  if (!parsed) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-4 p-8 text-center">
        <span className="flex h-14 w-14 items-center justify-center rounded-full bg-primary-soft text-primary">
          <Table2 className="h-7 w-7" />
        </span>
        <div>
          <h2 className="text-lg font-semibold text-ink">Chưa có dữ liệu</h2>
          <p className="mt-1 max-w-xs text-sm text-muted">
            Chọn một doanh nghiệp niêm yết bên dưới để hỏi ngay, hoặc đính kèm file Excel/CSV
            báo cáo tài chính của bạn ở khung chat.
          </p>
        </div>
        <CorpusPicker entries={corpusIndex} onPick={onLoadCorpus} loading={loadingSample} />

        {corpusIndex.length > 0 && (
          <p className="text-xs text-muted">hoặc dùng dữ liệu mẫu</p>
        )}

        <div className="flex flex-col gap-2">
          <button
            onClick={onLoadSample}
            disabled={loadingSample}
            className="inline-flex items-center justify-center gap-2 rounded-sm bg-primary px-4 py-2.5 text-sm font-semibold text-on-primary hover:bg-primary-hover disabled:opacity-60"
          >
            <Sparkles className="h-4 w-4" /> {loadingSample ? "Đang tải…" : "Dùng dữ liệu mẫu"}
          </button>
          <button
            onClick={onLoadDirty}
            disabled={loadingSample}
            className="inline-flex items-center justify-center gap-2 rounded-sm border border-border-strong bg-canvas px-4 py-2.5 text-sm font-semibold text-body hover:bg-surface-soft disabled:opacity-60"
          >
            <Droplets className="h-4 w-4" /> Thử dữ liệu bẩn (test làm sạch)
          </button>
          <button
            onClick={onLoadUnitErr}
            disabled={loadingSample}
            className="inline-flex items-center justify-center gap-2 rounded-sm border border-border-strong bg-canvas px-4 py-2.5 text-sm font-semibold text-body hover:bg-surface-soft disabled:opacity-60"
          >
            <ShieldAlert className="h-4 w-4" /> Thử dữ liệu lệch đơn vị (test tự kiểm)
          </button>
          <button
            onClick={onLoadMessy}
            disabled={loadingSample}
            className="inline-flex items-center justify-center gap-2 rounded-sm border border-border-strong bg-canvas px-4 py-2.5 text-sm font-semibold text-body hover:bg-surface-soft disabled:opacity-60"
          >
            <Wand2 className="h-4 w-4" /> Thử CSV lạ (test chuẩn hoá)
          </button>
        </div>
      </div>
    );
  }

  const items = tidy && tidy.byTicker[activeTicker] ? Object.values(tidy.byTicker[activeTicker]) : [];
  const sorted = [...items].sort((a, b) => Number(a.maSo) - Number(b.maSo) || a.maSo.localeCompare(b.maSo));
  const shown = sorted.slice(0, MAX_ROWS);
  const kpis = tidy && items.length ? computeKpis(tidy, activeTicker) : [];
  const validation = tidy && items.length ? validateTicker(tidy, activeTicker) : null;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* File chip + selects */}
      <div className="shrink-0 space-y-3 border-b border-hairline p-4">
        <div className="flex items-center gap-2.5">
          <span className="flex h-9 w-9 items-center justify-center rounded-md bg-positive-soft text-positive">
            <FileSpreadsheet className="h-5 w-5" />
          </span>
          <div className="min-w-0">
            <div className="truncate text-sm font-semibold text-ink">{parsed.fileName}</div>
            <div className="text-xs text-muted">
              {parsed.sheetNames.length} sheet{tidy ? ` · ${tidy.tickers.length} công ty · ${tidy.periods.join(", ")}` : ""}
            </div>
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          {parsed.sheetNames.length > 1 && (
            <label className="flex items-center gap-1.5 text-xs text-muted">
              Sheet:
              <select
                value={activeSheet}
                onChange={(e) => onSelectSheet(e.target.value)}
                className="rounded-sm border border-hairline bg-canvas px-2 py-1 text-xs text-ink"
              >
                {parsed.sheetNames.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </label>
          )}
          {tidy && tidy.tickers.length > 1 && (
            <label className="flex items-center gap-1.5 text-xs text-muted">
              Công ty:
              <select
                value={activeTicker}
                onChange={(e) => onSelectTicker(e.target.value)}
                className="max-w-[180px] rounded-sm border border-hairline bg-canvas px-2 py-1 text-xs text-ink"
              >
                {tidy.tickers.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </label>
          )}
        </div>

        {/* Dữ liệu đến từ kho trích sẵn thì nói luôn mức độ tự kiểm. Đây là điều phân biệt với
            việc chỉ trích dẫn nguồn: không chỉ "lấy từ đâu" mà "và con số này tự đối chiếu được".
            Nói bằng con số đo được, không bằng tính từ. */}
        {corpusSummary && (
          <p className="flex items-start gap-1.5 rounded-sm bg-primary-soft px-2.5 py-2 text-xs text-body">
            <ShieldCheck className="mt-0.5 h-3.5 w-3.5 shrink-0 text-primary" />
            <span>
              Trích từ bộ dữ liệu ViFinQA · <strong>{corpusSummary}</strong> (Tổng tài sản = Nợ phải
              trả + Vốn chủ sở hữu).
            </span>
          </p>
        )}

        {/* Bộ tiền xử lý đã làm gì với file — hiện TỪNG việc một, phân theo mức độ. Trước đây mọi
            ghi chú bị nối bằng dấu cách thành một khối vàng phẳng, đọc không ra việc nào với việc
            nào. Người dùng cần biết con số họ sắp thấy đã đi qua những gì. */}
        {tidy?.prep && (tidy.prep.notes.length > 0 || tidy.prep.headerRow > 0) && (
          <ul className="space-y-1">
            {tidy.prep.notes.map((n, i) => (
              <li
                key={i}
                className={cn(
                  "flex items-start gap-2 rounded-md px-3 py-2 text-xs",
                  n.level === "warn" ? "bg-warning-soft text-warning" : "bg-surface-soft text-muted",
                )}
              >
                {n.level === "warn" ? (
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                ) : (
                  <Wand2 className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                )}
                <span>{n.text}</span>
              </li>
            ))}
          </ul>
        )}

        {tidy && tidy.cleaningNotes.length > 0 && (
          <div className="flex items-start gap-2 rounded-md bg-warning-soft px-3 py-2 text-xs text-warning">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>{tidy.cleaningNotes.join(" ")}</span>
          </div>
        )}

        {validation && validation.checked > 0 && (
          validation.issues.length === 0 ? (
            <div className="flex items-center gap-2 rounded-md bg-positive-soft px-3 py-2 text-xs text-positive">
              <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
              Đã kiểm tra cân đối kế toán: {validation.passed}/{validation.checked} đẳng thức khớp.
            </div>
          ) : (
            <div className="flex items-start gap-2 rounded-md bg-warning-soft px-3 py-2 text-xs text-warning">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>
                Phát hiện {validation.issues.length} điểm chưa khớp đẳng thức kế toán (nên kiểm tra đơn vị/nhập liệu). Ví dụ:{" "}
                {validation.issues[0].rule} ({validation.issues[0].period}, lệch {validation.issues[0].diffPct}%).
              </span>
            </div>
          )
        )}
      </div>

      {/* KPI + chart + table (cuộn chung) */}
      <div className="min-h-0 flex-1 overflow-auto">
        {shown.length === 0 ? (
          // Nói THIẾU GÌ và làm gì tiếp, thay vì một câu chung chung. `fatal` do bộ tiền xử lý sinh
          // ra đã nêu đúng nguyên nhân — dùng lại nó thay vì đoán lại ở đây.
          <div className="mx-auto max-w-md space-y-3 p-6 text-sm text-muted">
            <div className="flex items-start gap-2 rounded-md bg-warning-soft px-3 py-2 text-xs text-warning">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>{tidy?.fatal ?? "Sheet này chưa nhận ra định dạng báo cáo tài chính."}</span>
            </div>
            <div className="text-xs">
              Một file đọc được cần:
              <ul className="mt-1 list-disc space-y-0.5 pl-5">
                <li>ít nhất một cột số liệu có năm trong tên — ví dụ <span className="text-ink">Năm 2024</span>, <span className="text-ink">Year_2024</span>, <span className="text-ink">31/12/2024</span>;</li>
                <li>cột <span className="text-ink">Mã số</span> theo Thông tư 200, hoặc cột <span className="text-ink">Chỉ tiêu</span> ghi tên theo cách gọi chuẩn.</li>
              </ul>
            </div>
            {parsed.sheetNames.length > 1 && (
              <p className="text-xs">File có {parsed.sheetNames.length} sheet — bạn thử chọn sheet khác ở trên.</p>
            )}
          </div>
        ) : (
          <div className="space-y-5 p-4">
            <KpiRow kpis={kpis} />
            {tidy && <Dashboard tidy={tidy} ticker={activeTicker} />}
            <div>
              <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">Bảng số liệu</div>
              <div className="overflow-hidden rounded-md border border-hairline">
                <table className="w-full border-collapse text-sm">
                  <thead className="bg-surface-soft">
                    <tr className="text-left text-xs uppercase text-muted">
                      <th className="px-3 py-2 font-medium">Mã số</th>
                      <th className="px-3 py-2 font-medium">Chỉ tiêu</th>
                      {tidy!.periods.map((p) => (
                        <th key={p} className="px-3 py-2 text-right font-medium">{p}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {shown.map((it) => (
                      <tr key={it.maSo} className="border-t border-hairline-soft hover:bg-surface-soft">
                        <td className="px-3 py-1.5 font-mono text-xs text-muted">{it.maSo}</td>
                        <td className="px-3 py-1.5 text-ink">{it.name}</td>
                        {tidy!.periods.map((p) => (
                          <td key={p} className="px-3 py-1.5 text-right font-mono text-xs text-body">
                            {formatMoney(it.values[p], tidy!.unit)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {sorted.length > MAX_ROWS && (
                <div className="px-3 py-2 text-center text-xs text-muted-soft">
                  … còn {sorted.length - MAX_ROWS} dòng nữa
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
