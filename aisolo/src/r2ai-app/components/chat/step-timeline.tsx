import { Loader2, Check, X, ChevronRight } from "lucide-react";
import type { StepData } from "@/lib/financial/types";
import { cn } from "@/lib/utils";

/**
 * Timeline tiến trình agent — render ĐỘNG theo thứ tự bước đến (live).
 * Chỉ hiển thị ý định + kết quả dạng người đọc; KHÔNG lộ mã/công thức/số thô.
 */
export function StepTimeline({ steps }: { steps: StepData[] }) {
  if (!steps.length) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted">
        <Loader2 className="h-4 w-4 animate-spin text-primary" /> Đang xử lý…
      </div>
    );
  }
  return (
    <ol className="space-y-2">
      {steps.map((s) => (
        <li key={s.id} className="text-sm">
          <div className="flex items-center gap-2">
            {s.status === "running" && <Loader2 className="h-4 w-4 shrink-0 animate-spin text-primary" />}
            {s.status === "done" && <Check className="h-4 w-4 shrink-0 text-positive" />}
            {s.status === "error" && <X className="h-4 w-4 shrink-0 text-negative" />}
            {/* whitespace-nowrap: tên bước là nhãn ngắn, để nó xuống dòng thì thành "Hiểu / câu / hỏi" */}
            <span className={cn("whitespace-nowrap", s.status === "error" ? "text-negative" : "text-body")}>{s.label}</span>
            {s.detail && <span className="truncate text-xs text-muted">· {s.detail}</span>}
          </div>
          {s.items && s.items.length > 0 && (
            <ul className="ml-6 mt-1 space-y-0.5 border-l border-hairline pl-3">
              {s.items.map((it, i) => (
                <li key={i} className="text-xs text-muted">
                  {it}
                </li>
              ))}
            </ul>
          )}
        </li>
      ))}
    </ol>
  );
}

/** Trace gấp gọn (mặc định đóng) trong tin nhắn đã xong — để xem lại "AI đã làm gì". */
export function TraceReview({ steps }: { steps: StepData[] }) {
  if (!steps.length) return null;
  return (
    <details className="group">
      <summary className="flex cursor-pointer list-none items-center gap-1 text-xs font-medium text-muted hover:text-ink">
        <ChevronRight className="h-3.5 w-3.5 transition-transform group-open:rotate-90" />
        Các bước xử lý
      </summary>
      <div className="mt-2 pl-1">
        <StepTimeline steps={steps} />
      </div>
    </details>
  );
}
