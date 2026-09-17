"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { createContext, useContext, useState, useEffect } from "react";
import type { Message } from "../types/academic";
import {ApiStatus} from "./ApiStatus";

type Session = { messages: Message[]; setMessages: React.Dispatch<React.SetStateAction<Message[]>> };
const SessionContext = createContext<Session | null>(null);
export function useAcademicSession() { const state = useContext(SessionContext); if (!state) throw new Error("AcademicShell required"); return state; }
export const NAV = [{href:"/",label:"Ask",ja:"大学生活を相談",mark:"01"},{href:"/study",label:"Study",ja:"科目を理解する",mark:"02"},
  {href:"/materials",label:"Materials",ja:"資料から学ぶ",mark:"03"},{href:"/exam",label:"Exam",ja:"試験に備える",mark:"04"},
  {href:"/report",label:"Report",ja:"研究・課題の準備",mark:"05"},{href:"/research",label:"Research",ja:"研究を組み立てる",mark:"06"},{href:"/sources",label:"Sources",ja:"出典を確かめる",mark:"07"},
  {href:"/gpa",label:"GPA",ja:"成績を試算する",mark:"08"},{href:"/degree",label:"Degree",ja:"単位の条件を整理",mark:"09"},
  {href:"/planner",label:"Planner",ja:"予定と出席を整理",mark:"10"},{href:"/email",label:"Email",ja:"教授への連絡を準備",mark:"11"},
  {href:"/memory",label:"Memory",ja:"学びを記録する",mark:"12"},{href:"/plan",label:"Plan",ja:"学ぶ時間を組み立てる",mark:"13"},
  {href:"/office-hours",label:"Office Hours",ja:"質問を整理する",mark:"14"},{href:"/official-search",label:"Official",ja:"公式の根拠を確認",mark:"15"},
  {href:"/career",label:"Career",ja:"経験から就活を準備",mark:"16"}];
export const NAV_GROUPS=[{name:'学ぶ',paths:['/','/study','/materials','/exam','/office-hours']},{name:'調べる・書く',paths:['/report','/research','/sources','/official-search']},{name:'大学生活を整える',paths:['/gpa','/degree','/planner','/email']},{name:'次の学びと将来',paths:['/memory','/plan','/career']}];
export function FeatureStatusBadge({status}: {status:"Available"|"Foundation"|"Beta"|"Coming next"}) { return <span className={`feature-badge status-${status.split(" ")[0].toLowerCase()}`}>{status}</span>; }
export function AcademicShell({children}: {children: React.ReactNode}) {
  const pathname = usePathname(); const [messages,setMessages] = useState<Message[]>([]);const [menuOpen,setMenuOpen]=useState(false);
  useEffect(()=>{if(menuOpen)document.querySelector<HTMLAnchorElement>('#mobile-feature-menu a')?.focus();},[menuOpen]);
  return <SessionContext.Provider value={{messages,setMessages}}><a className="skip-link" href="#main-content">本文へスキップ</a>
    <div className="academic-frame"><aside className="knowledge-rail" aria-label="Knowledge Rail">
      <Link href="/" className="brand" aria-label="UniPilot ホーム"><svg viewBox="0 0 40 40" aria-hidden="true"><path d="M7 9v17l13 8 13-8V9M7 9l13 8L33 9M20 17v17"/><circle cx="7" cy="9" r="3"/><circle cx="33" cy="9" r="3"/><circle cx="20" cy="17" r="3"/></svg><span>UniPilot<small>ACADEMIC OS / v1</small></span></Link>
      <p className="rail-label">YOUR KNOWLEDGE RAIL</p><nav aria-label="メインナビゲーション">{NAV_GROUPS.map(g=><section key={g.name} className="nav-group"><h2>{g.name}</h2>{NAV.filter(n=>g.paths.includes(n.href)).map(n => <Link key={n.href} href={n.href} aria-current={pathname===n.href?"page":undefined} className="rail-link"><span className="node-number">{n.mark}</span><span>{n.label}<small>{n.ja}</small></span><span aria-hidden="true">↗</span></Link>)}</section>)}</nav>
      <div className="rail-bottom"><p className="eyebrow">BUILT FOR UNIVERSITY</p><p>学びから、次の問いへ。</p><details><summary>モデル・評価ツール</summary><div className="utility-links"><Link href="/settings">モデル情報</Link><Link href="/developer">Checkpoint比較</Link><Link href="/campus-eval">Campus v1評価</Link><Link href="/campus-v2-eval">Campus v2評価</Link><Link href="/campus-v21-quick-eval">v2.1かんたん評価</Link><Link href="/campus-v21-eval">v2.1詳細評価</Link><Link href="/campus-v21-known-issues">既知問題</Link><Link href="/campus-ai-review">AI改善レビュー</Link></div></details><span className="privacy-note">External AI API: OFF</span></div>
    </aside><div className="academic-body"><header className="top-bar"><Link href="/" className="mobile-brand">UniPilot</Link><span>UNIVERSITY WORKSPACE / STAGE 8</span><ApiStatus/></header><div id="main-content" tabIndex={-1}>{children}</div></div></div>
    {menuOpen&&<nav id="mobile-feature-menu" className="mobile-feature-menu" aria-label="全機能メニュー" onKeyDown={e=>{if(e.key==='Escape'){setMenuOpen(false);document.getElementById('feature-menu-toggle')?.focus();}}}>{NAV_GROUPS.map(g=><section key={g.name}><h2>{g.name}</h2>{NAV.filter(n=>g.paths.includes(n.href)).map(n=><Link key={n.href} href={n.href} aria-current={pathname===n.href?'page':undefined} onClick={()=>setMenuOpen(false)}>{n.label} / {n.ja}</Link>)}</section>)}</nav>}
    <nav className="mobile-nav" aria-label="モバイルナビゲーション">{NAV.filter(n=>['/','/study','/planner','/plan','/career'].includes(n.href)).map(n=><Link key={n.href} href={n.href} aria-current={pathname===n.href?"page":undefined} onClick={()=>setMenuOpen(false)}><span aria-hidden="true">{n.mark}</span>{n.label}</Link>)}<button id="feature-menu-toggle" aria-expanded={menuOpen} aria-controls="mobile-feature-menu" onClick={()=>setMenuOpen(v=>!v)}>{menuOpen?'閉じる':'全機能'}</button></nav>
  </SessionContext.Provider>;
}
