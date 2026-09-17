import { TrendingUp, TrendingDown } from "lucide-react";
import type { Kpi } from "@/lib/financial/kpi";
import { cn } from "@/lib/utils";

export function KpiRow({ kpis }: { kpis: Kpi[] }) {
  if (!kpis.length) return null;
  return (
    <div>
      <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">Chỉ số nổi bật</div>
      <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3">
        {kpis.map((k) => (
          <div key={k.key} className="rounded-md border border-hairline bg-canvas p-3 shadow-[var(--shadow-card)]">
            <div className="truncate text-xs font-medium text-muted">{k.label}</div>
            <div className="mt-1 font-mono text-[22px] font-bold leading-tight text-primary">{k.valueLabel}</div>
            {k.deltaLabel && (
              <div
                className={cn(
                  "mt-1 inline-flex items-center gap-0.5 text-xs font-medium",
                  k.deltaKind === "pct" && k.deltaSign > 0 && "text-positive",
                  k.deltaKind === "pct" && k.deltaSign < 0 && "text-negative",
                  (k.deltaKind === "pp" || k.deltaSign === 0) && "text-muted",
                )}
              >
                {k.deltaKind === "pct" && k.deltaSign !== 0 ? (
                  k.deltaSign > 0 ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />
                ) : null}
                {k.deltaLabel}
                <span className="text-muted-soft"> so với năm trước</span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
