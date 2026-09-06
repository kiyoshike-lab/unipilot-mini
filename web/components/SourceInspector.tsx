"use client";
import { cardSources, safeSourceUrl, sourceStatus } from "../lib/academic";
import { useAcademicSession } from "./AcademicShell";
import type { SourceItem } from "../types/academic";
export function SourceInspector({sources,compact=false}: {sources:SourceItem[]; compact?:boolean}) {
  return <section className={`source-inspector ${compact?"compact":""}`} aria-label="Source Trace"><div className="section-heading"><span className="eyebrow">SOURCE TRACE</span><span className="count-tag">{sources.length}</span></div><h2>回答の根拠を、確かめる。</h2>
    {!sources.length ? <div className="empty-source"><span aria-hidden="true">⌁</span><p>まだ出典はありません</p><small>APIが返した出典だけを表示します。引用・論文検索は今後の機能です。</small></div> : <ul>{sources.map((source,index)=>{const url=safeSourceUrl(source.url); return <li key={`${source.id}-${index}`}>
      <span className={`source-status ${source.stale?"stale":""}`}>{sourceStatus(source)}</span><h3>{url?<a href={url} target="_blank" rel="noreferrer">{source.title} ↗</a>:source.title}</h3>
      <dl><div><dt>発行元</dt><dd>{source.publisher || "不明"}</dd></div><div><dt>ライセンス</dt><dd>{source.license || "不明"}</dd></div><div><dt>最終確認</dt><dd>{source.last_verified_at || "不明"}</dd></div><div><dt>信頼度（API）</dt><dd>{source.confidence || "不明"}</dd></div><div><dt>更新状態</dt><dd>{source.stale===undefined?"不明":source.stale?"古い可能性あり":"APIではstaleなし"}</dd></div></dl>
      {!url&&<p className="muted">参照可能なHTTPS URL：不明</p>}</li>;})}</ul>}
    <p className="fine-print">掲載と検証は別です。重要な主張は原文・更新日を確認してください。</p></section>;
}
export function SourcesWorkspace() { const {messages}=useAcademicSession(); return <main className="workspace"><p className="eyebrow">05 / SOURCES</p><h1>Source Inspector</h1><p className="intro-copy">このタブの会話で返された出典を確認する場所。自動引用エンジンは準備中です。</p><SourceInspector sources={messages.flatMap(m=>cardSources(m.cards))}/></main>; }
