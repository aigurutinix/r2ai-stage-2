"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2, LogIn } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Brand } from "@/components/brand";

export function LoginForm() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        setError(data.error ?? "Đăng nhập thất bại");
        setLoading(false);
        return;
      }
      router.replace("/");
      router.refresh();
    } catch {
      setError("Không kết nối được máy chủ. Thử lại.");
      setLoading(false);
    }
  }

  return (
    <div className="w-full max-w-md">
      {/* Brand */}
      <Brand iconSize={44} textClass="text-2xl" className="mb-8" />

      {/* Marketing hero */}
      <p className="text-sm font-semibold uppercase tracking-wider text-primary">
        Trợ lý tài chính bằng AI
      </p>
      <h1 className="mt-3 text-[30px] font-bold leading-[1.2] tracking-tight text-ink lg:text-[34px]">
        Biến báo cáo tài chính thành <span className="text-primary">câu trả lời.</span>
      </h1>
      <p className="mt-4 text-[19px] leading-relaxed text-muted">
        Đính kèm file Excel, hỏi bằng tiếng Việt. Mọi con số được trích thẳng từ file và kèm
        dòng nguồn để bạn tự đối chiếu — không cần biết lập trình.
      </p>

      {/* Form */}
      <form onSubmit={onSubmit} className="mt-8 space-y-4">
        <div>
          <Label htmlFor="username">Tên đăng nhập</Label>
          <Input
            id="username"
            autoComplete="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Nhập tên đăng nhập"
            required
            autoFocus
          />
        </div>
        <div>
          <Label htmlFor="password">Mật khẩu</Label>
          <Input
            id="password"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Nhập mật khẩu"
            required
          />
        </div>

        {error && (
          <p className="rounded-sm bg-negative-soft px-3 py-2 text-sm text-negative" role="alert">
            {error}
          </p>
        )}

        <Button type="submit" variant="gradient" size="md" className="w-full" disabled={loading}>
          {loading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" /> Đang đăng nhập...
            </>
          ) : (
            <>
              <LogIn className="h-4 w-4" /> Đăng nhập
            </>
          )}
        </Button>
      </form>

      <p className="mt-8 text-xs text-muted-soft">Phiên bản demo · Road to AI 2026</p>
    </div>
  );
}
