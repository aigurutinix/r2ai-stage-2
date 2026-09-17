import { chartSeries } from "@/lib/financial/kpi";
import type { TidyTable } from "@/lib/financial/types";
import { GptChart } from "./gpt-chart";

/** Xu hướng doanh thu & LNST theo kỳ — gpt-vis column (grouped). */
export function TrendChart({ tidy, ticker }: { tidy: TidyTable; ticker: string }) {
  const raw = chartSeries(tidy, ticker);
  if (raw.length < 2) return null;

  const scale = 1e6; // triệu VND
  const data: { category: string; value: number; group: string }[] = [];
  for (const d of raw) {
    if (d.revenue != null) data.push({ category: d.period, value: Math.round(d.revenue / scale), group: "Doanh thu" });
    if (d.netIncome != null) data.push({ category: d.period, value: Math.round(d.netIncome / scale), group: "LN sau thuế" });
  }

  return (
    <div>
      <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">
        Xu hướng doanh thu &amp; lợi nhuận <span className="normal-case text-muted-soft">(triệu VND)</span>
      </div>
      <div className="rounded-md border border-hairline bg-canvas p-2 shadow-[var(--shadow-card)]">
        <GptChart
          type="column"
          height={320}
          config={{ data, group: true, style: { palette: ["#1c64e8", "#0e9e5c"] } }}
        />
      </div>
    </div>
  );
}
