import "./globals.css";
import { AcademicShell } from "../components/AcademicShell";

export const metadata = { title: "UniPilot | Academic OS", description: "大学生の学習・課題・研究をつなぐAIワークスペース。外部LLM API接続なし。" };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="ja"><body><AcademicShell>{children}</AcademicShell></body></html>;
}
