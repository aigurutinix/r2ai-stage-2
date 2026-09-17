"use client";
import { useMemo, useState } from "react";
import { Building2, ShieldCheck } from "lucide-react";
import type { CorpusEntry } from "@/lib/financial/corpus";

interface Props {
  entries: CorpusEntry[];
  onPick: (ticker: string) => void;
  loading: boolean;
}

/**
 * Chọn một doanh nghiệp niêm yết từ kho đã trích sẵn.
 *
 * Chỉ liệt kê doanh nghiệp ĐÃ QUA KIỂM đẳng thức kế toán. Số còn lại vẫn được nêu ra kèm lý do
 * thay vì giấu đi — người xem cần biết phạm vi thật của hệ thống, không phải một danh sách đã
 * lọc sạch để trông đẹp hơn.
 */
export function CorpusPicker({ entries, onPick, loading }: Props) {
  const inScope = useMemo(
    () => entries.filter((e) => e.inScope).sort((a, b) => a.ticker.localeCompare(b.ticker)),
    [entries],
  );
  const [ticker, setTicker] = useState("");
  if (!entries.length) return null;

  const outCount = entries.length - inScope.length;
  return (
    <div className="w-full max-w-sm rounded-md border border-border-strong bg-canvas p-4 text-left">
      <p className="flex items-center gap-2 text-sm font-semibold text-ink">
        <Building2 className="h-4 w-4 text-primary" />
        Tra cứu doanh nghiệp niêm yết
      </p>
      <p className="mt-1 text-xs text-muted">
        {inScope.length} doanh nghiệp trích sẵn từ bộ dữ liệu ViFinQA — không cần tải file lên.
      </p>

      <div className="mt-3 flex gap-2">
        <select
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          aria-label="Chọn doanh nghiệp"
          className="min-w-0 flex-1 rounded-sm border border-border-strong bg-surface px-2 py-2 text-sm text-body"
        >
          <option value="">— Chọn mã chứng khoán —</option>
          {inScope.map((e) => (
            <option key={e.ticker} value={e.ticker}>
              {e.ticker} · {e.verifiedYears}/{e.periods} niên độ đã kiểm
            </option>
          ))}
        </select>
        <button
          onClick={() => ticker && onPick(ticker)}
          disabled={!ticker || loading}
          className="rounded-sm bg-primary px-3 py-2 text-sm font-semibold text-on-primary hover:bg-primary-hover disabled:opacity-60"
        >
          {loading ? "Đang tải…" : "Nạp"}
        </button>
      </div>

      <p className="mt-3 flex items-start gap-1.5 text-xs text-muted">
        <ShieldCheck className="mt-0.5 h-3.5 w-3.5 shrink-0 text-primary" />
        <span>
          Mỗi niên độ được đối chiếu bằng đẳng thức kế toán (Tổng tài sản = Nợ phải trả + Vốn chủ sở
          hữu). {outCount > 0 && `${outCount} doanh nghiệp còn lại chưa hỗ trợ: không chỉ tiêu nào khớp hệ Mã số Thông tư 200 vì tổ chức tín dụng dùng hệ tài khoản riêng.`}
        </span>
      </p>
    </div>
  );
}
