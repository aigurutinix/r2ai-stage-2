import { LoginForm } from "./login-form";
import { LoginBackground } from "./login-background";

/**
 * Trang login kiosk: khoá đúng chiều cao viewport, KHÔNG scroll.
 * Tỷ lệ 3:7 — cột trái (form) 30%, cột phải (video lấp đầy + cắt chéo) 70%.
 * object-cover để lấp đầy panel (chuẩn split-login), cắt chéo mép trái như ảnh mẫu.
 * Dưới md (768px): chỉ hiện form full-width, ẩn video.
 */
export default function LoginPage() {
  return (
    <main className="flex h-dvh w-full overflow-hidden bg-surface-soft">
      {/* Cột trái: form — 30% */}
      <section className="flex w-full items-center justify-center overflow-y-auto px-6 py-8 sm:px-8 md:w-[30%]">
        <LoginForm />
      </section>

      {/* Cột phải: video lấp đầy, cắt chéo mép trái — 70%. Ẩn dưới md. */}
      <aside className="relative hidden h-full shrink-0 md:block md:w-[70%] [clip-path:polygon(11%_0,100%_0,100%_100%,0_100%)]">
        <LoginBackground />
      </aside>
    </main>
  );
}
