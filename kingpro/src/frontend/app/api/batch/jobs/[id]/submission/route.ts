import { NextResponse } from "next/server";
import { backendHeaders } from "@/lib/backend-auth";

export const dynamic = "force-dynamic";

export async function POST(request: Request, context: { params: Promise<{ id: string }> }) {
  const upstream = process.env.KINGPRO_API_URL ?? "http://127.0.0.1:8080";
  const { id } = await context.params;
  try {
    const response = await fetch(`${upstream}/batch/jobs/${encodeURIComponent(id)}/submission`, {
      method: "POST",
      headers: backendHeaders(request, { "X-Kingpro-Approval": "user-confirmed" }),
      cache: "no-store",
      signal: AbortSignal.timeout(15 * 60_000),
    });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { status: "error", error: error instanceof Error ? error.message : "Không xuất được submission" },
      { status: 503 },
    );
  }
}
