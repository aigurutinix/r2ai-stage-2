import { getSession } from "@/lib/session";
import { runAgent } from "@/lib/financial/agents";
import type { QueryPlan, TidyTable } from "@/lib/financial/types";

export const runtime = "nodejs";
export const maxDuration = 30;

export async function POST(req: Request) {
  const session = await getSession();
  if (!session.isLoggedIn) return Response.json({ ok: false, error: "Unauthorized" }, { status: 401 });

  let body: { tidy?: TidyTable; question?: string; activeTicker?: string; prevPlan?: QueryPlan | null };
  try {
    body = await req.json();
  } catch {
    return Response.json({ ok: false, error: "Yêu cầu không hợp lệ" }, { status: 400 });
  }
  const { tidy, question, activeTicker, prevPlan } = body;
  if (!tidy || !tidy.tickers?.length || !question?.trim()) {
    return Response.json({ ok: false, error: "Thiếu dữ liệu hoặc câu hỏi" }, { status: 400 });
  }

  // Stream NDJSON: từng "step" của agent, rồi "result" cuối cùng.
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const write = (obj: unknown) => controller.enqueue(encoder.encode(JSON.stringify(obj) + "\n"));
      try {
        const result = await runAgent(tidy, question, activeTicker, (step) => write({ type: "step", step }), prevPlan);
        write({ type: "result", result });
      } catch (err) {
        write({
          type: "result",
          result: { ok: false, error: err instanceof Error ? err.message : "Lỗi agent", answer: "Có lỗi khi xử lý câu hỏi. Bạn thử lại nhé." },
        });
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: { "Content-Type": "application/x-ndjson; charset=utf-8", "Cache-Control": "no-store" },
  });
}
