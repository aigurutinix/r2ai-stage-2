import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET() {
  const upstream = process.env.KINGPRO_API_URL ?? "http://127.0.0.1:8080";
  try {
    const response = await fetch(`${upstream}/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(5_000),
    });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch {
    return NextResponse.json(
      { status: "offline", catalog: false, bm25_index: false, llm_configured: false, model_allowed: false },
      { status: 503 },
    );
  }
}
