import "./globals.css";
import Link from "next/link";

export const metadata = { title: "UniPilot Mini", description: "完全ローカルの大学生活専用AI" };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="ja"><body className="min-h-screen font-sans">
    <nav className="border-b border-slate-800 bg-slate-950/80 px-6 py-4 backdrop-blur">
      <div className="mx-auto flex max-w-5xl items-center justify-between">
        <Link href="/" className="font-semibold tracking-wide">UniPilot Mini</Link>
        <div className="flex gap-5 text-sm text-slate-300"><Link href="/">チャット</Link><Link href="/settings">モデル情報</Link><Link href="/developer">比較</Link><Link href="/campus-eval">Campus v1評価</Link><Link href="/campus-v2-eval">Campus v2評価</Link></div>
      </div>
    </nav>{children}</body></html>;
}
