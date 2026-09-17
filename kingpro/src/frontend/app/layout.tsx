import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "KINGPRO — Financial Evidence Engine",
  description: "Trợ lý truy vấn báo cáo tài chính bằng Pandas có dẫn nguồn kiểm chứng.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="vi">
      <body>{children}</body>
    </html>
  );
}
