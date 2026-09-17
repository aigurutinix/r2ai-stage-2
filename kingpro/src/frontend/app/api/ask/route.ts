import { NextRequest, NextResponse } from "next/server";
import { repairTextDeep } from "@/lib/text";
import { backendHeaders } from "@/lib/backend-auth";

export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  const upstream = process.env.KINGPRO_API_URL ?? "http://127.0.0.1:8080";
  try {
    const body = await request.json();
    const response = await fetch(`${upstream}/ask`, {
      method: "POST",
      headers: backendHeaders(request, { "Content-Type": "application/json; charset=utf-8" }),
      body: JSON.stringify(body),
      cache: "no-store",
      signal: AbortSignal.timeout(180_000),
    });
    const data = repairTextDeep(await response.json());
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      {
        status: "error",
        grounded: false,
        error: error instanceof Error ? error.message : "Không kết nối được KINGPRO API",
      },
      { status: 503 },
    );
  }
}
