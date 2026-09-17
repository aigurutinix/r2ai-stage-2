import "server-only";
import { timingSafeEqual } from "node:crypto";

/** So sánh chuỗi chống timing attack. */
function safeEqual(a: string, b: string): boolean {
  const ba = Buffer.from(a);
  const bb = Buffer.from(b);
  if (ba.length !== bb.length) return false;
  return timingSafeEqual(ba, bb);
}

/** Kiểm tra credential với biến môi trường (mock 1 admin, không có DB). */
export function checkCredentials(username: string, password: string): boolean {
  const u = process.env.ADMIN_USER ?? "";
  const p = process.env.ADMIN_PASS ?? "";
  if (!u || !p) return false;
  return safeEqual(username, u) && safeEqual(password, p);
}
