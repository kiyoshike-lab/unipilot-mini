"use client";
import {useEffect,useState} from "react";
import {ChatWorkspace} from "./ChatWorkspace";
import {buildMaterialPrompt,buildExamPrompt,examDays,localDateKey,MATERIAL_ACTIONS,EXAM_ACTIONS,MATERIAL_CHARACTER_LIMIT,PROMPT_TOKEN_UPPER_LIMIT,tokenUpperBound,type ExamInput} from "../lib/learning";

export function MaterialsWorkspace() {
  const [material,setMaterial]=useState(""); const [action,setAction]=useState<string>("Ask about this material");
  let budget="";
  try {budget=`資料単体 ${tokenUpperBound(material)} token上界（UTF-8 bytes）。指示・質問を含む上限 ${PROMPT_TOKEN_UPPER_LIMIT}。`;} catch {budget="入力を確認してください";}
  return <ChatWorkspace workflow="materials" preparePrompt={q=>buildMaterialPrompt(material,action,q)} setup={<section className="learning-setup" aria-labelledby="materials-heading">
    <h3 id="materials-heading">講義ノートを貼り付ける</h3><p className="muted">Plain text / Markdown対応。まずは短い一段落・数式一つの抜粋から。資料はこの画面内だけで保持し、送信時にUniPilot APIへ渡します。</p>
    <label htmlFor="lecture-material">講義資料</label><textarea id="lecture-material" value={material} onChange={e=>setMaterial(e.target.value)} rows={6} placeholder="講義ノートの短い抜粋を貼り付けてください" aria-describedby="material-limit"/>
    <p id="material-limit" className="fine-print">{Array.from(material).length} / {MATERIAL_CHARACTER_LIMIT}文字。{budget} モデルcontext 512、出力32＋APIの指示文用192を予約。日本語では文字数上限より先にtoken上限になります。正確なtoken数ではなく安全側の上界です。超過は送信停止し、自動切り詰めしません。</p>
    <fieldset className="action-picker"><legend>資料から何をする？</legend>{MATERIAL_ACTIONS.map(value=><button type="button" key={value} aria-pressed={action===value} onClick={()=>setAction(value)}>{value}</button>)}</fieldset>
    <p className="learning-note">資料の根拠づけはpromptによる指示段階です。引用箇所の自動照合・検証は未実装。回答を原文と照合してください。</p>
    <details><summary>Coming next · PDF / PPTX / DOCX</summary><p className="muted">Document ingestion → Chunking → Local embedding / Retrieval → Source spans → Lecture RAG。今回はファイル抽出・検索・外部embeddingを実行しません。</p></details>
  </section>}/>;
}

export function ExamWorkspace() {
  const [exam,setExam]=useState<ExamInput>({subject:"",date:"",scope:"",minutesPerDay:""});
  const [action,setAction]=useState<string>("Make study plan"); const [today,setToday]=useState("");
  useEffect(()=>{const update=()=>setToday(localDateKey());update();const timer=setInterval(update,60000);window.addEventListener("focus",update);return()=>{clearInterval(timer);window.removeEventListener("focus",update);};},[]);
  let countdown="試験日を入力すると残り日数を表示します。";
  if(exam.date&&today){try {const days=examDays(exam.date,today);countdown=days===0?"試験は今日です。":`試験まで ${days} 日`;}catch(e){countdown=e instanceof Error?e.message:"日付が無効です";}}
  return <ChatWorkspace workflow="exam" preparePrompt={q=>buildExamPrompt(exam,action,q,localDateKey())} setup={<section className="learning-setup" aria-labelledby="exam-heading">
    <h3 id="exam-heading">自分の試験を設定する</h3><div className="exam-fields">{([{key:"subject",label:"試験科目",type:"text"},{key:"date",label:"試験日",type:"date"},{key:"scope",label:"試験範囲",type:"text"},{key:"minutesPerDay",label:"1日あたり学習時間（分）",type:"number"}] as const).map(field=><label key={field.key}>{field.label}<input aria-label={field.label} type={field.type} value={exam[field.key]} min={field.type==="number"?1:undefined} max={field.type==="number"?1440:undefined} onChange={e=>setExam(v=>({...v,[field.key]:e.target.value}))}/></label>)}</div>
    <p className="exam-countdown" role="status">{countdown}</p><p className="fine-print">端末の現在日付 {today||"確認中"} から暦日で計算。AIに日付計算は任せません。</p>
    <fieldset className="action-picker"><legend>試験に向けて</legend>{EXAM_ACTIONS.map(value=><button type="button" key={value} aria-pressed={action===value} onClick={()=>setAction(value)}>{value}</button>)}</fieldset>
    <p className="learning-note">計画は入力に基づく提案です。学習履歴・個人最適化・カレンダー連携は未実装。入力全体は安全上限 {PROMPT_TOKEN_UPPER_LIMIT} token相当以内。超過は送信を停止します。</p>
  </section>}/>;
}
