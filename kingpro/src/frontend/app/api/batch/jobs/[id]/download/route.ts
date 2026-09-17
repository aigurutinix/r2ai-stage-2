import { NextResponse } from "next/server";
import { backendHeaders } from "@/lib/backend-auth";

export const dynamic = "force-dynamic";

export async function GET(request: Request, context: { params: Promise<{ id: string }> }) {
  const upstream = process.env.KINGPRO_API_URL ?? "http://127.0.0.1:8080";
  const { id } = await context.params;
  try {
    const response = await fetch(`${upstream}/batch/jobs/${encodeURIComponent(id)}/download`, {
      headers: backendHeaders(request),
      cache: "no-store",
      signal: AbortSignal.timeout(30_000),
    });
    const payload = await response.json();
    if (!response.ok) return NextResponse.json(payload, { status: response.status });
    return new NextResponse(JSON.stringify(payload, null, 2), {
      status: 200,
      headers: {
        "Content-Type": "application/json; charset=utf-8",
        "Content-Disposition": `attachment; filename="kingpro-${id}.json"`,
      },
    });
  } catch (error) {
    return NextResponse.json(
      { status: "error", error: error instanceof Error ? error.message : "Không tải được batch result" },
      { status: 503 },
    );
  }
}
