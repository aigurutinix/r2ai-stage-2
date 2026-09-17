import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Toaster } from "sonner";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin", "latin-ext"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin", "latin-ext"],
});

export const metadata: Metadata = {
  title: "FinWhale — Trợ lý Phân tích Báo cáo Tài chính",
  description:
    "Tải lên báo cáo tài chính, hỏi bằng tiếng Việt, nhận số liệu kèm nguồn — bằng AI Agent.",
  icons: {
    icon: "/favicon.ico",
    apple: "/logo.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="vi"
      suppressHydrationWarning
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      {/* suppressHydrationWarning: một số browser extension chèn attribute vào <body>
          trước khi React hydrate, gây cảnh báo mismatch giả — bỏ qua an toàn. */}
      <body suppressHydrationWarning className="h-full">
        {children}
        <Toaster position="bottom-right" richColors closeButton />
      </body>
    </html>
  );
}
