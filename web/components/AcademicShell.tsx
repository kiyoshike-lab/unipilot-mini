"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { createContext, useContext, useState } from "react";
import type { Message } from "../types/academic";

type Session = { messages: Message[]; setMessages: React.Dispatch<React.SetStateAction<Message[]>> };
const SessionContext = createContext<Session | null>(null);
export function useAcademicSession() { const state = useContext(SessionContext); if (!state) throw new Error("AcademicShell required"); return state; }
export const NAV = [{href:"/",label:"Ask",ja:"大学生活を相談",mark:"01"},{href:"/study",label:"Study",ja:"科目を理解する",mark:"02"},
  {href:"/report",label:"Report",ja:"課題を整理する",mark:"03"},{href:"/research",label:"Research",ja:"研究を組み立てる",mark:"04"},{href:"/sources",label:"Sources",ja:"出典を確かめる",mark:"05"}];
export function FeatureStatusBadge({status}: {status:"Available"|"Beta"|"Coming next"}) { return <span className={`feature-badge status-${status.split(" ")[0].toLowerCase()}`}>{status}</span>; }
export function AcademicShell({children}: {children: React.ReactNode}) {
  const pathname = usePathname(); const [messages,setMessages] = useState<Message[]>([]);
  return <SessionContext.Provider value={{messages,setMessages}}><a className="skip-link" href="#main-content">本文へスキップ</a>
    <div className="academic-frame"><aside className="knowledge-rail" aria-label="Knowledge Rail">
      <Link href="/" className="brand" aria-label="UniPilot ホーム"><svg viewBox="0 0 40 40" aria-hidden="true"><path d="M7 9v17l13 8 13-8V9M7 9l13 8L33 9M20 17v17"/><circle cx="7" cy="9" r="3"/><circle cx="33" cy="9" r="3"/><circle cx="20" cy="17" r="3"/></svg><span>UniPilot<small>ACADEMIC OS / v1</small></span></Link>
      <p className="rail-label">YOUR KNOWLEDGE RAIL</p><nav aria-label="メインナビゲーション">{NAV.map(n => <Link key={n.href} href={n.href} aria-current={pathname===n.href?"page":undefined} className="rail-link"><span className="node-number">{n.mark}</span><span>{n.label}<small>{n.ja}</small></span><span aria-hidden="true">↗</span></Link>)}</nav>
      <div className="rail-bottom"><p className="eyebrow">BUILT FOR UNIVERSITY</p><p>学びから、次の問いへ。</p><details><summary>モデル・評価ツール</summary><div className="utility-links"><Link href="/settings">モデル情報</Link><Link href="/developer">Checkpoint比較</Link><Link href="/campus-eval">Campus v1評価</Link><Link href="/campus-v2-eval">Campus v2評価</Link><Link href="/campus-v21-quick-eval">v2.1かんたん評価</Link><Link href="/campus-v21-eval">v2.1詳細評価</Link><Link href="/campus-v21-known-issues">既知問題</Link><Link href="/campus-ai-review">AI改善レビュー</Link></div></details><span className="privacy-note">External AI API: OFF</span></div>
    </aside><div className="academic-body"><header className="top-bar"><Link href="/" className="mobile-brand">UniPilot</Link><span>UNIVERSITY WORKSPACE</span><span className="top-tag">自分の理解を、少し先へ <i aria-hidden="true" /></span></header><div id="main-content" tabIndex={-1}>{children}</div></div></div>
    <nav className="mobile-nav" aria-label="モバイルナビゲーション">{NAV.map(n=><Link key={n.href} href={n.href} aria-current={pathname===n.href?"page":undefined}><span aria-hidden="true">{n.mark}</span>{n.label}</Link>)}</nav>
  </SessionContext.Provider>;
}
