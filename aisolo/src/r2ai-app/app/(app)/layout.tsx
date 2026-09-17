import { redirect } from "next/navigation";
import { getSession } from "@/lib/session";

/** GATE thật (server-side) — không phụ thuộc middleware. */
export default async function AppLayout({ children }: { children: React.ReactNode }) {
  const session = await getSession();
  if (!session.isLoggedIn) redirect("/login");
  return <div className="flex h-screen flex-col">{children}</div>;
}
