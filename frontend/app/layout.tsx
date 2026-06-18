import type { ReactNode } from "react";

export const metadata = {
  title: "Gooaye RAG",
  description: "Ask the 股癌 podcast — pluggable RAG playground",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="zh-Hant">
      <body
        style={{
          margin: 0,
          fontFamily:
            "ui-sans-serif, system-ui, -apple-system, 'PingFang TC', 'Noto Sans TC', sans-serif",
          background: "#0b0c0f",
          color: "#e8eaed",
        }}
      >
        {children}
      </body>
    </html>
  );
}
