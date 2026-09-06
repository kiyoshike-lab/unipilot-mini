"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { FeatureStatusBadge } from "./AcademicShell";
const workflows={report:["Requirements","Research","Outline","Draft","Sources","Review"],research:["Topic","Research Question","Sources","Evidence","Outline","Draft","Citation Audit"]};
const labels={report:["テーマ","課題文","文字数","締切","レポート形式","参考資料"],research:["研究テーマ","研究質問","方法・分析のメモ","先行研究・参考資料","結果・考察のメモ"]};
export function PlanningWorkspace({kind}: {kind:"report"|"research"}) {
  const [values,setValues]=useState<Record<string,string>>({}); const [ready,setReady]=useState(false); const [notice,setNotice]=useState("");
  const key=`unipilot-${kind}-draft-v1`; const title=kind==="report"?"Report Workspace":"Research Workspace";
  useEffect(()=>{setReady(false);try {const raw=JSON.parse(sessionStorage.getItem(key)||"{}");setValues(Object.fromEntries(Object.entries(raw).filter(([k,v])=>labels[kind].includes(k)&&typeof v==="string")) as Record<string,string>);}catch{setValues({});}setReady(true);},[key,kind]);
  function save() {try{sessionStorage.setItem(key,JSON.stringify(values));setNotice("このタブに下書きを保存しました。タブを閉じると失われます。");}catch{setNotice("保存できませんでした。メモをコピーして保管してください。");}}
  async function copy() {try {await navigator.clipboard.writeText(labels[kind].map(label=>`${label}\n${values[label]||"（未入力）"}`).join("\n\n"));setNotice("入力メモをコピーしました。");}catch{setNotice("コピーできませんでした。各入力欄のテキストを選択してください。");}}
  return <main className="workspace"><p className="eyebrow">{kind==="report"?"03 / REPORT":"04 / RESEARCH"} · WORKSPACE FOUNDATION</p><div className="planning-title"><h1>{title}</h1><FeatureStatusBadge status="Coming next"/></div><p className="intro-copy">{kind==="report"?"書き始める前に、課題の条件を整理する。":"研究テーマを、一つの問いから組み立てる。"}</p>
    <div className="shell-notice"><strong>現在は入力整理用のshellです。</strong> 自動執筆・論文検索・引用生成は実装していません。入力は送信されません。</div>
    <ol className="workflow-rail" aria-label="今後のワークフロー">{workflows[kind].map((step,i)=><li key={step}><span>{String(i+1).padStart(2,"0")}</span><strong>{step}</strong><small>{i===0?"入力を整理":"Coming next"}</small></li>)}</ol>
    <div className="planning-grid"><section className="planning-form"><h2>考えを置く、最初のメモ。</h2><p className="muted">未入力の項目は、あとから考えても大丈夫です。</p>{labels[kind].map(label=><label key={label}>{label}{label==="締切"?<input aria-label={label} type="date" disabled={!ready} value={values[label]||""} onChange={e=>setValues(v=>({...v,[label]:e.target.value}))}/>:<textarea aria-label={label} rows={label==="課題文"?4:2} disabled={!ready} value={values[label]||""} onChange={e=>setValues(v=>({...v,[label]:e.target.value}))} placeholder={`${label}を入力（任意）`}/>}</label>)}<div className="planning-actions"><button className="primary-button" onClick={save} disabled={!ready}>このタブに保存</button><button className="secondary-button" onClick={copy}>入力メモをコピー</button></div><p role="status">{notice}</p></section>
      <aside className="planning-aside"><span className="eyebrow">YOUR NEXT QUESTION</span><h2>完成を急がず、<br/>問いを明確に。</h2><p>{kind==="report"?"何を説明する課題か。必要な根拠は何か。評価条件を分けて書き留めましょう。":"何を明らかにしたいか。どの方法で確かめるか。まだ不明な点も研究メモの一部です。"}</p><Link className="secondary-button" href="/study">関連科目を学ぶ ↗</Link><hr/><h3>出典は実在するものだけ。</h3><p>論文・DOI・著者・ページ番号を自動補完しません。将来のCitation Engineは原文検証と根拠箇所の照合を前提にします。</p><Link href="/sources">Source Inspectorを見る ↗</Link></aside></div></main>;
}
