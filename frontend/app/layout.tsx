import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Bike Assistant — Sarvam AI",
  description: "AI-powered bike troubleshooting from your manual",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full antialiased bg-slate-50 text-slate-900">{children}</body>
    </html>
  );
}
