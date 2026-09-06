"use client";
import Link from "next/link";
import { FormEvent, useEffect, useRef, useState } from "react";
import { FeatureStatusBadge, useAcademicSession } from "./AcademicShell";
import { SourceInspector } from "./SourceInspector";
import { API, sendChat } from "../lib/chat";
import { buildStudyPrompt, cardSources, LEVELS, METHODS, SUBJECTS } from "../lib/academic";
import type { ClarifyOption, ResponseMode, StudyOptions, ToolCard } from "../types/academic";

function ToolCardRenderer({card,onUse}: {card:ToolCard;onUse:(s:string)=>void}) {
  const [notice,setNotice]=useState("");
  async function useCard() { if (card.copy_text) { try { await navigator.clipboard.writeText(card.copy_text); setNotice("コピーしました"); } catch { setNotice("コピーできませんでした。本文を選択してコピーしてください。"); } }
    else { const example=card.fields?.find(f=>f.example)?.example; onUse(example?`${card.title}：${example}`:`${card.title}：必要項目を入力します。`); } }
  const options=Array.isArray(card.data?.options) ? (card.data.options as ClarifyOption[]).filter(o=>typeof o?.prompt==="string"&&typeof o?.label==="string") : [];
  return <section className="tool-card"><span className="eyebrow">UNIPILOT TOOL</span><h3>{card.title}</h3><p>{card.summary}</p>
    {!!card.fields?.length&&<p className="muted">必要項目：{card.fields.map(f=>f.label).join("、")}</p>}
    {!!options.length&&<div className="clarify-options">{options.map((o,i)=><button type="button" key={`${o.category}-${i}`} onClick={()=>onUse(o.prompt)}>{o.label}</button>)}</div>}
    {card.kind==="sources"&&<details><summary>出典と更新日を表示</summary><SourceInspector sources={cardSources([card])} compact/></details>}
    {(card.action_label||card.copy_text)&&<button className="secondary-button" type="button" onClick={useCard}>{card.action_label||"コピー"}</button>}
    <p role="status" className="fine-print">{notice}</p></section>;
}

export function ChatWorkspace({study=false}: {study?:boolean}) {
  const {messages,setMessages}=useAcademicSession(); const [input,setInput]=useState(""); const [busy,setBusy]=useState(false);
  const [error,setError]=useState(""); const [connection,setConnection]=useState("未接続 · 送信時に接続");
  const [responseMode,setResponseMode]=useState<ResponseMode>("normal"); const [studyOptions,setStudyOptions]=useState<StudyOptions>({subject:"線形代数",level:"標準",method:"考え方から"});
  const session=useRef(""); const controller=useRef<AbortController|null>(null); const busyRef=useRef(false); const inputRef=useRef<HTMLTextAreaElement>(null);
  const retry=useRef<{question:string;prompt:string;responseMode:ResponseMode}|null>(null);
  useEffect(()=>{ try { session.current=sessionStorage.getItem("unipilot-campus-session")||crypto.randomUUID(); sessionStorage.setItem("unipilot-campus-session",session.current); }
    catch { session.current=crypto.randomUUID(); } return ()=>controller.current?.abort(); },[]);
  const mode=study?"study":"ask"; const shown=messages.filter(m=>m.mode===mode); const latest=shown.filter(m=>m.role==="assistant").at(-1);
  function fill(text:string) {setInput(text); inputRef.current?.focus();}
  async function send(question:string,prompt:string,length:ResponseMode) {
    if (busyRef.current) return; busyRef.current=true; setBusy(true); setError(""); setConnection("接続中 · APIの起動に時間がかかる場合があります");
    retry.current={question,prompt,responseMode:length}; const id=crypto.randomUUID(); const ctrl=new AbortController(); controller.current=ctrl;
    const timer=setTimeout(()=>ctrl.abort(),90000); let hasText=false;
    setMessages(v=>[...v,{id:crypto.randomUUID(),role:"user",text:question,mode},{id,role:"assistant",text:"",mode}]);
    try { const transport=await sendChat(prompt,length,session.current,snapshot=>{ if (snapshot.text) hasText=true;
      setConnection("受信中"); setMessages(items=>items.map(m=>m.id===id?{...m,text:snapshot.text??m.text,cards:snapshot.cards??m.cards}:m)); },ctrl.signal);
      if (!hasText) setMessages(items=>items.map(m=>m.id===id?{...m,text:"応答が空でした。質問を短くして再試行してください。"}:m));
      setConnection(transport==="stream"?"応答完了 · streaming":"応答完了 · /chat fallback");
    } catch(reason) { setConnection("接続を確認してください"); setError(ctrl.signal.aborted?"接続待ちが長いため停止しました。APIを確認して再試行してください。":reason instanceof Error?reason.message:"APIに接続できません");
      setMessages(items=>items.filter(m=>m.id!==id||m.text.length>0));
    } finally {clearTimeout(timer); busyRef.current=false; setBusy(false); controller.current=null;}
  }
  function submit(event:FormEvent) {event.preventDefault(); const question=input.trim(); if (!question||busyRef.current) return;
    const prompt=study?buildStudyPrompt(question,studyOptions):question; setInput(""); void send(question,prompt,responseMode);}
  return <main className="workspace chat-workspace"><div className="workspace-heading"><div><p className="eyebrow">{study?"02 / STUDY LAB":"01 / ACADEMIC COMMAND CENTER"}</p>
    <h1>{study?"答えの先の、理解へ。":"大学の「わからない」を、"}{!study&&<><br/><span>一つにつなぐ。</span></>}</h1>
    <p className="intro-copy">{study?"科目と学び方を選んで、一つずつ考える。UniPilot Tutor Stage 1。":"学習・課題・研究の入口をまとめた、大学生のためのAIワークスペース。"}</p></div>
    <div className="orbit-mark" aria-hidden="true"><svg viewBox="0 0 160 120"><path d="M24 80 75 25 135 50 105 105 24 80 135 50M75 25l30 80"/><circle cx="24" cy="80" r="5"/><circle cx="75" cy="25" r="7"/><circle cx="135" cy="50" r="5"/><circle cx="105" cy="105" r="4"/></svg><span>ACADEMIC CONSTELLATION</span></div></div>
    {!study&&<div className="academic-nodes" aria-label="ワークスペースへの入口">{[{href:"/study",label:"Study",description:"科目を、理解に変える。",status:"Beta"},{href:"/report",label:"Report",description:"課題の条件を整理する。",status:"Coming next"},{href:"/research",label:"Research",description:"研究の問いを組み立てる。",status:"Coming next"}].map((n,i)=><Link href={n.href} className="academic-node" key={n.href}><div><span className="node-index">0{i+2}</span><FeatureStatusBadge status={n.status as "Beta"|"Coming next"}/></div><h2>{n.label}<span aria-hidden="true">↗</span></h2><p>{n.description}</p></Link>)}</div>}
    <div className="command-grid"><section className="command-panel"><div className="focus-dock"><div><span className="eyebrow">FOCUS DOCK</span><h2>{study?"UniPilot Tutor":"Ask UniPilot"}</h2></div><FeatureStatusBadge status={study?"Beta":"Available"}/></div>
      <nav className="mode-switcher" aria-label="AIモード"><Link aria-current={!study?"page":undefined} href="/">Ask</Link><Link aria-current={study?"page":undefined} href="/study">Study</Link><Link href="/report">Report <small>準備中</small></Link><Link href="/research">Research <small>準備中</small></Link></nav>
      {study&&<fieldset className="study-controls"><legend>学び方を選ぶ</legend>{([{key:"subject",label:"科目",values:SUBJECTS},{key:"level",label:"説明レベル",values:LEVELS},{key:"method",label:"学習方法",values:METHODS}] as const).map(field=><label key={field.key}>{field.label}<select aria-label={field.label} value={studyOptions[field.key]} disabled={busy} onChange={e=>setStudyOptions(v=>({...v,[field.key]:e.target.value}))}>{field.values.map(v=><option key={v}>{v}</option>)}</select></label>)}</fieldset>}
      <div className="chat-thread" role="log" aria-label="会話" aria-live="polite" aria-relevant="additions text" aria-busy={busy}>
        {!shown.length&&<div className="welcome-state"><span className="welcome-symbol" aria-hidden="true">✳</span><h3>{study?"どこから、一緒に考えよう？":"いま、何が気になっていますか？"}</h3><p>{study?"問題文と、わからなくなった箇所を書いてください。":"大学生活の疑問から、レポートの準備まで。まずは短い質問で。"}</p><div className="prompt-suggestions">{(study?["固有値の意味を、具体例で知りたい","途中式のどこで間違えたか確認したい"]:["先生への相談メールを考えたい","レポートを始める手順を知りたい"]).map(q=><button key={q} onClick={()=>fill(q)} type="button">{q}<span aria-hidden="true">↗</span></button>)}</div></div>}
        {shown.map(message=><article key={message.id} className={`message message-${message.role}`}><span className="message-label">{message.role==="user"?"YOU":"UNIPILOT"}</span><div className="answer-text">{message.text||"接続中…"}</div>{message.cards?.map((card,i)=><ToolCardRenderer key={i} card={card} onUse={fill}/>)}</article>)}
      </div>
      <form className="academic-composer" onSubmit={submit}><label htmlFor="academic-question">{study?"質問・問題文":"大学生活について聞く"}</label><textarea ref={inputRef} id="academic-question" value={input} onChange={e=>setInput(e.target.value)} placeholder={study?"例：行列の固有値は、何を表していますか？":"質問を入力してください…"} rows={2} maxLength={8000}/><div className="composer-actions"><label>回答の長さ<select aria-label="回答の長さ" value={responseMode} onChange={e=>setResponseMode(e.target.value as ResponseMode)}><option value="short">短く</option><option value="normal">標準</option><option value="detailed">詳しく</option></select></label><button className="primary-button" disabled={busy||!input.trim()}>{busy?"受信中…":"送信 ↗"}</button></div></form>
      <p className="connection-status" role="status">{connection}</p>{error&&<div className="api-error" role="alert"><p>{error}</p><button type="button" disabled={busy} onClick={()=>{const r=retry.current;if(r)void send(r.question,r.prompt,r.responseMode);}}>同じ質問を再試行</button><small>接続先：{API}（APIの稼働状態を確認してください）</small></div>}
    </section><aside className="context-column"><section className="study-pulse"><div className="section-heading"><span className="eyebrow">STUDY PULSE</span><span className="pulse-dot" aria-hidden="true"/></div><h2>{study?studyOptions.subject:"今日の問いを、ここから。"}</h2><p>{study?`${studyOptions.level} / ${studyOptions.method}`:"まず相談。必要に応じて学習・課題・研究のモードへ。"}</p><div className="pulse-line" aria-hidden="true"/><small>学習履歴の分析・最適化は未実装です。</small></section>
      <SourceInspector sources={cardSources(latest?.cards)} compact/><Link className="source-link" href="/sources">Sources · 会話の全出典を見る ↗</Link>
      <div className="capability-note"><span className="eyebrow">HONEST BY DESIGN</span><p>UIは学習の入口です。モデルは開発中で、全科目の正答を保証しません。</p><small>回答は教材・一次資料と照合してください。</small></div>
      {study&&<details className="future-answer"><summary>今後の解説レイアウト</summary><p>APIが構造化回答に対応した段階で追加予定。現在は受信テキストをそのまま表示します。</p><ul><li>結論・前提・考え方・確認ポイント</li><li>式・途中式・単位・計算・検算</li><li>概念・背景・論点・具体例・反対意見</li></ul></details>}
    </aside></div><footer className="workspace-footer"><span>UNIPILOT / ACADEMIC CONSTELLATION</span><span>外部LLM接続なし · External AI API: OFF</span></footer>
  </main>;
}
