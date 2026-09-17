import { NextRequest, NextResponse } from "next/server";
import { SESSION_COOKIE } from "@/lib/constants";

/**
 * Proxy (Next 16, thay cho middleware) — CHỈ để redirect UX.
 * Gate thật nằm ở app/(app)/layout.tsx (đọc session server-side).
 */
export function proxy(req: NextRequest) {
  const hasCookie = req.cookies.has(SESSION_COOKIE);
  const { pathname } = req.nextUrl;

  if (!hasCookie && !pathname.startsWith("/login")) {
    return NextResponse.redirect(new URL("/login", req.url));
  }
  if (hasCookie && pathname.startsWith("/login")) {
    return NextResponse.redirect(new URL("/", req.url));
  }
  return NextResponse.next();
}

export const config = {
  // Bỏ qua /api (API tự kiểm session), /login, static.
  matcher: ["/((?!login|api|_next|.*\\..*).*)", "/login"],
};
