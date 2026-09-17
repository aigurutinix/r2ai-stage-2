import { NextRequest, NextResponse } from "next/server";
import { backendHeaders } from "@/lib/backend-auth";

export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  const upstream = process.env.KINGPRO_API_URL ?? "http://127.0.0.1:8080";
  try {
    const body = await request.text();
    const response = await fetch(`${upstream}/batch/jobs`, {
      method: "POST",
      headers: backendHeaders(request, { "Content-Type": "application/json; charset=utf-8" }),
      body,
      cache: "no-store",
      signal: AbortSignal.timeout(30_000),
    });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { status: "error", error: error instanceof Error ? error.message : "Không tạo được batch job" },
      { status: 503 },
    );
  }
}
