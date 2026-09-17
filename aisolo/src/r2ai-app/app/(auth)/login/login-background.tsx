/**
 * Cột phải trang login: ảnh nền R2AI (background_r2ai.png) LẤP ĐẦY panel.
 * object-right giữ trọn nhân vật. Lớp phủ đen mờ nhẹ (gradient) tạo chiều sâu, bóng.
 */
export function LoginBackground() {
  return (
    <div className="relative h-full w-full overflow-hidden bg-surface-strong">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src="/background_r2ai.png"
        alt="Minh hoạ R2AI phân tích báo cáo tài chính"
        className="absolute inset-0 h-full w-full object-cover object-right"
      />
      {/* Lớp phủ đen mờ nhẹ đè lên tạo chiều sâu / độ bóng */}
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-black/10 via-black/5 to-black/30" />
    </div>
  );
}
