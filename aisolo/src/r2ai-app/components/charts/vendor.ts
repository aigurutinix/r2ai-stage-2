/**
 * Vendor 4 hàm render chart trên @antv/g2.
 *
 * Phái sinh (derivative work) từ @antv/gpt-vis, đã sửa để: bật ANIMATION entry,
 * format số tiếng Việt, nhãn pie KHÔNG tràn. Import động ở client (gpt-chart.tsx)
 * để tránh SSR window.
 *
 * Portions derived from @antv/gpt-vis — The MIT License (MIT)
 * Copyright (c) 2024 AntV. Xem THIRD-PARTY-NOTICES.md (toàn văn giấy phép MIT).
 */
import { Chart } from "@antv/g2";
import { fmtVN, fmtPct } from "@/lib/charts/format";

/* eslint-disable @typescript-eslint/no-explicit-any */
type Cfg = Record<string, any>;
export interface ChartInstance {
  render: (config: Cfg) => void;
  destroy: () => void;
}
interface Opts {
  container: HTMLElement;
  width: number;
  height: number;
}

const WHITE = "#ffffff";

export function Column(opts: Opts): ChartInstance {
  let chart: any = null;
  return {
    render(config) {
      const { data = [], title, axisXTitle, axisYTitle, group = false, stack = false, style = {} } = config;
      if (chart) chart.destroy();
      const hasGroup = data.length > 0 && data[0]?.group !== undefined;
      chart = new Chart({ container: opts.container, width: opts.width, height: opts.height, autoFit: true });
      const transform = hasGroup && group ? [{ type: "dodgeX" }] : hasGroup && stack ? [{ type: "stackY" }] : [];
      const encode = hasGroup
        ? { x: "category", y: "value", color: "group" }
        : { x: "category", y: "value", color: "category" };
      const scale: Cfg = { y: { nice: true } };
      if (style.palette) scale.color = { range: style.palette };
      chart.options({
        type: "interval",
        animate: { enter: { type: "scaleInY", duration: 700 } },
        data,
        title: title ?? "",
        encode,
        transform,
        scale,
        axis: {
          x: { title: axisXTitle || false, labelAutoRotate: true },
          y: { title: axisYTitle || false, labelFormatter: fmtVN },
        },
        legend: hasGroup ? { color: { position: "bottom" } } : false,
        tooltip: { items: [(d: any) => ({ name: hasGroup ? d.group : d.category, value: fmtVN(d.value) })] },
        style: { radiusTopLeft: 4, radiusTopRight: 4, columnWidthRatio: 0.7 },
        viewStyle: { viewFill: WHITE },
      });
      chart.render();
    },
    destroy() {
      if (chart) {
        chart.destroy();
        chart = null;
      }
    },
  };
}

/**
 * Miền chồng 100% — dạng chuẩn cho "chuyển dịch cơ cấu" qua từ 4 kỳ trở lên (biểu đồ tròn chỉ
 * đúng với 1–3 kỳ). normalizeY quy mỗi kỳ về 100% nên đọc được tỉ trọng, không phải giá trị.
 */
export function Area(opts: Opts): ChartInstance {
  let chart: any = null;
  return {
    render(config) {
      const { data = [], title, axisXTitle, axisYTitle, style = {} } = config;
      if (chart) chart.destroy();
      chart = new Chart({ container: opts.container, width: opts.width, height: opts.height, autoFit: true });
      const scale: Cfg = { y: { nice: true } };
      if (style.palette) scale.color = { range: style.palette };
      chart.options({
        type: "area",
        animate: { enter: { type: "growInX", duration: 900 } },
        data,
        title: title ?? "",
        encode: { x: "time", y: "value", color: "group" },
        transform: [{ type: "stackY" }, { type: "normalizeY" }],
        scale,
        axis: {
          x: { title: axisXTitle || false },
          // normalizeY trả tỉ lệ 0..1, còn fmtPct nhận sẵn thang 0..100 (pie dùng vậy) →
          // phải nhân 100 tại chỗ, nếu không trục hiện "0,8%" thay vì "80%".
          y: { title: axisYTitle || false, labelFormatter: (v: number) => fmtPct(v * 100) },
        },
        legend: { color: { position: "bottom" } },
        tooltip: { items: [(d: any) => ({ name: d.group, value: fmtVN(d.value) })] },
        viewStyle: { viewFill: WHITE },
      });
      chart.render();
    },
    destroy() {
      if (chart) {
        chart.destroy();
        chart = null;
      }
    },
  };
}

export function Line(opts: Opts): ChartInstance {
  let chart: any = null;
  return {
    render(config) {
      const { data = [], title, axisXTitle, axisYTitle, style = {} } = config;
      if (chart) chart.destroy();
      const hasGroup = data.length > 0 && data[0]?.group !== undefined;
      const lineWidth = style.lineWidth ?? 2.5;
      const c0 = style.palette?.[0];
      chart = new Chart({ container: opts.container, width: opts.width, height: opts.height, autoFit: true });
      const encode: Cfg = hasGroup ? { x: "time", y: "value", color: "group" } : { x: "time", y: "value" };
      const scale: Cfg = { y: { nice: true } };
      if (hasGroup && style.palette) scale.color = { range: style.palette };
      chart.options({
        type: "view",
        animate: { enter: { type: "growInX", duration: 1000 } },
        data,
        title: title ?? "",
        encode,
        children: [
          { type: "line", style: { lineWidth, ...(!hasGroup && c0 ? { stroke: c0 } : {}) } },
          { type: "point", style: { r: 3, ...(!hasGroup && c0 ? { fill: c0 } : {}) } },
        ],
        scale,
        axis: { x: { title: axisXTitle || false }, y: { title: axisYTitle || false, labelFormatter: fmtVN } },
        legend: hasGroup ? { color: { position: "bottom" } } : false,
        tooltip: { items: [(d: any) => ({ name: hasGroup ? d.group : d.time, value: fmtVN(d.value) })] },
        viewStyle: { viewFill: WHITE },
      });
      chart.render();
    },
    destroy() {
      if (chart) {
        chart.destroy();
        chart = null;
      }
    },
  };
}

export function Pie(opts: Opts): ChartInstance {
  let chart: any = null;
  return {
    render(config) {
      const { data = [], innerRadius = 0, title, style = {} } = config;
      if (chart) chart.destroy();
      const sum = data.reduce((s: number, it: any) => s + (it.value || 0), 0) || 1;
      chart = new Chart({ container: opts.container, width: opts.width, height: opts.height, autoFit: true });
      const scale: Cfg = {};
      if (style.palette) scale.color = { range: style.palette };
      chart.options({
        type: "interval",
        animate: { enter: { type: "waveIn", duration: 800 } },
        data,
        title: title ?? "",
        encode: { y: "value", color: "category" },
        transform: [{ type: "stackY" }],
        coordinate: { type: "theta", innerRadius: Math.max(0, Math.min(1, innerRadius)) },
        scale,
        legend: { color: { position: "bottom" } },
        labels: [
          {
            // nhãn NẰM TRONG lát → không bao giờ tràn card
            text: (d: any) => `${d.category}\n${fmtPct((d.value / sum) * 100)}`,
            position: "inside",
            fontSize: 12,
            fontWeight: 500,
            transform: [{ type: "overflowHide" }, { type: "contrastReverse" }],
          },
        ],
        tooltip: { items: [(d: any) => ({ name: d.category, value: fmtVN(d.value) })] },
        interaction: { elementSelect: { single: true } },
        style: { fillOpacity: 0.9 },
        viewStyle: { viewFill: WHITE },
      });
      chart.render();
    },
    destroy() {
      if (chart) {
        chart.destroy();
        chart = null;
      }
    },
  };
}

function transformWaterfall(data: any[]) {
  let cumulative = 0;
  let lastMid = 0;
  let totalSum = 0;
  return data.map((item) => {
    const value = item.value || 0;
    let start: number;
    let end: number;
    if (item.isTotal) {
      start = 0;
      end = totalSum;
    } else if (item.isIntermediateTotal) {
      start = lastMid;
      end = cumulative;
      lastMid = end;
    } else {
      start = cumulative;
      end = cumulative + value;
      cumulative = end;
      totalSum += value;
    }
    return { ...item, __start__: start, __end__: end, __value__: item.isTotal ? totalSum : item.isIntermediateTotal ? end - start : value };
  });
}

function linkData(data: any[]) {
  return data.reduce((res: any[], cur: any, i: number) => {
    if (i > 0) {
      const prev = data[i - 1];
      res.push({ x: [prev.category, cur.category], y: cur.isTotal || cur.isIntermediateTotal ? cur.__end__ : cur.__start__ });
    }
    return res;
  }, []);
}

export function Waterfall(opts: Opts): ChartInstance {
  let chart: any = null;
  return {
    render(config) {
      const { data = [], title, axisXTitle, axisYTitle, style = {} } = config;
      if (chart) chart.destroy();
      const palette = style.palette || [];
      const pos = palette[0] || "#F04438";
      const neg = palette[1] || "#12B76A";
      const tot = palette[2] || "#1C64E8";
      const td = transformWaterfall(data);
      const ld = linkData(td);
      chart = new Chart({ container: opts.container, width: opts.width, height: opts.height, autoFit: true });
      chart.options({
        type: "view",
        animate: { enter: { type: "scaleInY", duration: 700 } },
        data: td,
        title: title ?? "",
        marginRight: 28,
        axis: {
          x: { title: axisXTitle || false, labelAutoRotate: true },
          y: { title: axisYTitle || false, labelFormatter: fmtVN },
        },
        scale: { y: { nice: true } },
        children: [
          {
            type: "interval",
            data: td,
            encode: { x: "category", y: ["__start__", "__end__"], color: "category" },
            style: {
              maxWidth: 60,
              stroke: "#666",
              radius: 4,
              fill: (d: any) => (d.isTotal || d.isIntermediateTotal ? tot : d.__value__ > 0 ? pos : neg),
            },
            labels: [
              {
                text: "__value__",
                position: "inside",
                fontSize: 10,
                transform: [{ type: "overflowHide" }],
                formatter: fmtVN,
                fill: "#fff",
                fontWeight: 600,
              },
            ],
            tooltip: { title: (d: any) => d.category, items: [{ field: "__value__", name: "Giá trị", valueFormatter: fmtVN }] },
          },
          {
            type: "link",
            data: ld,
            encode: { x: "x", y: "y" },
            zIndex: -1,
            style: { stroke: "#ccc", lineDash: [4, 2], lineWidth: 1 },
            tooltip: false,
          },
        ],
        legend: false,
        viewStyle: { viewFill: WHITE },
      });
      chart.render();
    },
    destroy() {
      if (chart) {
        chart.destroy();
        chart = null;
      }
    },
  };
}
