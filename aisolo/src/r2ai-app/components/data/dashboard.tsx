import type { TidyTable } from "@/lib/financial/types";
import { cn } from "@/lib/utils";
import { capitalStructure, profitBridge, marginTrend, profitability } from "@/lib/financial/dashboard";
import { TrendChart } from "./trend-chart";
import { GptChart } from "./gpt-chart";

function Card({ title, className, children }: { title: string; className?: string; children: React.ReactNode }) {
  return (
    <div className={cn("rounded-md border border-hairline bg-canvas p-2.5 shadow-[var(--shadow-card)]", className)}>
      <div className="mb-1.5 px-1 text-xs font-semibold uppercase tracking-wide text-muted">{title}</div>
      {children}
    </div>
  );
}

/** Dashboard tự sinh (deterministic) — nhiều chart gpt-vis cho 1 công ty. */
export function Dashboard({ tidy, ticker }: { tidy: TidyTable; ticker: string }) {
  const period = tidy.periods[0];
  const cap = capitalStructure(tidy, ticker, period);
  const bridge = profitBridge(tidy, ticker, period);
  const margins = marginTrend(tidy, ticker);
  const prof = profitability(tidy, ticker, period);

  return (
    <div className="flex flex-col gap-4">
      <TrendChart tidy={tidy} ticker={ticker} />

      {cap && (
        <Card title="Cơ cấu nguồn vốn (triệu VND)">
          <GptChart type="pie" height={360} config={{ data: cap, innerRadius: 0.5, style: { palette: ["#e07c00", "#1c64e8"] } }} />
        </Card>
      )}

      {margins && (
        <Card title="Xu hướng biên lợi nhuận (%)">
          <GptChart type="line" height={320} config={{ data: margins, style: { palette: ["#1c64e8", "#0e9e5c"] } }} />
        </Card>
      )}

      {prof && (
        <Card title="Khả năng sinh lời — ROE / ROA (%)">
          <GptChart type="column" height={300} config={{ data: prof, style: { palette: ["#1c64e8"] } }} />
        </Card>
      )}

      {bridge && (
        <Card title="Cầu nối lợi nhuận: Doanh thu → LN sau thuế (triệu VND)">
          <GptChart type="waterfall" height={340} config={{ data: bridge }} />
        </Card>
      )}
    </div>
  );
}
