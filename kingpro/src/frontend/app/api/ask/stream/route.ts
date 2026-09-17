import { NextRequest, NextResponse } from "next/server";
import { backendHeaders } from "@/lib/backend-auth";

export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  const upstream = process.env.KINGPRO_API_URL ?? "http://127.0.0.1:8080";
  try {
    const body = await request.text();
    const response = await fetch(`${upstream}/ask/stream`, {
      method: "POST",
      headers: backendHeaders(request, { "Content-Type": "application/json; charset=utf-8" }),
      body,
      cache: "no-store",
      signal: AbortSignal.timeout(180_000),
    });
    if (!response.body) {
      return NextResponse.json({ status: "error", error: "upstream_stream_missing" }, { status: 502 });
    }
    return new Response(response.body, {
      status: response.status,
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-store",
        "X-Accel-Buffering": "no",
      },
    });
  } catch (error) {
    return NextResponse.json(
      { status: "error", error: error instanceof Error ? error.message : "Không kết nối được KINGPRO stream" },
      { status: 503 },
    );
  }
}
