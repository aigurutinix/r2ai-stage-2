import { backendHeaders } from "@/lib/backend-auth";

export const dynamic = "force-dynamic";

export async function GET(request: Request, context: { params: Promise<{ id: string }> }) {
  const upstream = process.env.KINGPRO_API_URL ?? "http://127.0.0.1:8080";
  const { id } = await context.params;
  const incoming = new URL(request.url);
  const cursor = incoming.searchParams.get("cursor") ?? "0";
  try {
    const response = await fetch(
      `${upstream}/batch/jobs/${encodeURIComponent(id)}/events?cursor=${encodeURIComponent(cursor)}`,
      { headers: backendHeaders(request), cache: "no-store", signal: AbortSignal.timeout(15 * 60_000) },
    );
    if (!response.body) return Response.json({ status: "error", error: "batch_stream_missing" }, { status: 502 });
    return new Response(response.body, {
      status: response.status,
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-store",
        "X-Accel-Buffering": "no",
      },
    });
  } catch (error) {
    return Response.json(
      { status: "error", error: error instanceof Error ? error.message : "Không kết nối được batch stream" },
      { status: 503 },
    );
  }
}
