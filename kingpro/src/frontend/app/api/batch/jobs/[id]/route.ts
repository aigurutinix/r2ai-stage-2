import { NextResponse } from "next/server";
import { backendHeaders } from "@/lib/backend-auth";

export const dynamic = "force-dynamic";

export async function GET(request: Request, context: { params: Promise<{ id: string }> }) {
  const upstream = process.env.KINGPRO_API_URL ?? "http://127.0.0.1:8080";
  const { id } = await context.params;
  try {
    const response = await fetch(`${upstream}/batch/jobs/${encodeURIComponent(id)}`, {
      headers: backendHeaders(request),
      cache: "no-store",
      signal: AbortSignal.timeout(10_000),
    });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { status: "error", error: error instanceof Error ? error.message : "Không đọc được batch job" },
      { status: 503 },
    );
  }
}
