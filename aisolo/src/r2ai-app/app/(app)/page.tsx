import { getSession } from "@/lib/session";
import { LogoutButton } from "@/components/logout-button";
import { Workspace } from "@/components/workspace";
import { Brand } from "@/components/brand";

export default async function AppPage() {
  const session = await getSession();

  return (
    <>
      <header className="flex h-14 shrink-0 items-center justify-between border-b border-hairline bg-canvas px-4">
        <div className="flex items-center gap-2">
          <Brand iconSize={30} textClass="text-base" />
          <span className="ml-1 hidden text-xs text-muted sm:inline">Trợ lý Phân tích Báo cáo Tài chính</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="hidden text-xs text-muted sm:inline">{session.user}</span>
          <LogoutButton />
        </div>
      </header>

      <Workspace />
    </>
  );
}
