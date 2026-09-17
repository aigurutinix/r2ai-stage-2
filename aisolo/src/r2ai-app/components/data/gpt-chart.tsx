"use client";
import { useEffect, useRef } from "react";

/** Loại chart (vendor AntV G2) đang dùng. */
export type GptChartType = "line" | "column" | "pie" | "area" | "waterfall";

const FACTORY_NAME: Record<GptChartType, string> = {
  line: "Line",
  column: "Column",
  pie: "Pie",
  area: "Area",
  waterfall: "Waterfall",
};

/**
 * Wrapper React cho gpt-vis (@antv/gpt-vis) — các factory là imperative (G2).
 * Dynamic import để chỉ nạp ở client (tránh lỗi SSR window). Tự vẽ lại khi resize.
 */
export function GptChart({
  type,
  config,
  height = 240,
}: {
  type: GptChartType;
  config: Record<string, unknown>;
  height?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const configKey = JSON.stringify(config);

  useEffect(() => {
    let chart: { destroy?: () => void } | undefined;
    let ro: ResizeObserver | undefined;
    let disposed = false;

    (async () => {
      const el = ref.current;
      if (!el) return;
      const mod = (await import("@/components/charts/vendor")) as unknown as Record<
        string,
        (o: { container: HTMLElement; width: number; height: number }) => {
          render: (c: Record<string, unknown>) => void;
          destroy?: () => void;
        }
      >;
      const factory = mod[FACTORY_NAME[type]];
      if (!factory || disposed) return;

      const draw = () => {
        const width = el.clientWidth;
        if (!width || width < 10) return; // chờ có kích thước (ResizeObserver sẽ gọi lại)
        try {
          chart?.destroy?.();
        } catch {}
        try {
          const inst = factory({ container: el, width, height });
          inst.render({ theme: "default", ...config });
          chart = inst;
        } catch (e) {
          console.error("gpt-vis render error:", e);
        }
      };
      draw();
      ro = new ResizeObserver(() => draw());
      ro.observe(el);
    })();

    return () => {
      disposed = true;
      ro?.disconnect();
      try {
        chart?.destroy?.();
      } catch {}
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [type, configKey, height]);

  return <div ref={ref} style={{ width: "100%", height }} />;
}
