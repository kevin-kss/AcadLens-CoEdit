import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AcadLens-CoEdit 工作台",
  description: "学术文本白盒诊断与人机协同精修",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
