import { getSession } from "@/lib/session";
import { checkCredentials } from "@/lib/auth";

export const runtime = "nodejs";

export async function POST(req: Request) {
  let username = "";
  let password = "";
  try {
    const body = await req.json();
    username = String(body.username ?? "");
    password = String(body.password ?? "");
  } catch {
    return Response.json({ ok: false, error: "Yêu cầu không hợp lệ" }, { status: 400 });
  }

  if (!checkCredentials(username, password)) {
    return Response.json(
      { ok: false, error: "Tên đăng nhập hoặc mật khẩu không đúng" },
      { status: 401 },
    );
  }

  const session = await getSession();
  session.user = username;
  session.isLoggedIn = true;
  await session.save();

  return Response.json({ ok: true });
}
