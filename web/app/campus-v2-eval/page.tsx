"use client";

import { useEffect, useMemo, useState } from "react";
import { usePathname } from "next/navigation";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
type Axis = "correctness" | "relevance" | "actionable" | "naturalness" | "would_use_again";
type PairAxis = "correctness" | "specificity" | "actionability" | "readability" | "would_use";
type PairChoice = "unipilot" | "competitor" | "tie" | "unscored";
type IssueKey = "critical_error" | "factual_error" | "unanswered" | "university_policy_assertion" |
  "unnecessary_information" | "unusable_answer" | "router_error" | "tool_error" | "faq_error" |
  "retrieval_error" | "model_error";
type UXKey = "tool_card" | "copy_action" | "input_flow" | "clarification" | "streaming" | "latency";
type UXResult = "pass" | "fail" | "not_applicable" | "not_evaluated";
type Pairwise = Record<"chatgpt" | "gemini", Record<PairAxis, PairChoice>>;
type Item = {
  id: string; question: string; category: string; difficulty: string; evaluation_bucket?: string;
  surface_type?: string; specialist_domain?: string | null; campus_answer: string;
  campus_metadata?: { action?: string; route?: string; tool?: string | null; latency_ms?: number; cards?: unknown[] };
  chatgpt_answer: string; gemini_answer: string; scores: Record<Axis, number | null>;
  competitor_scores: { chatgpt: number | null; gemini: number | null }; notes: string;
  issue_flags?: Record<IssueKey, boolean>; issues_reviewed?: boolean; pairwise?: Pairwise;
  ux?: Record<UXKey, UXResult>;
};

const AXES: Array<[Axis, string]> = [["correctness", "正確さ"], ["relevance", "質問との一致"],
  ["actionable", "すぐ行動できる"], ["naturalness", "自然さ"], ["would_use_again", "また使いたい"]];
const PAIR_AXES: Array<[PairAxis, string]> = [["correctness", "正確性"], ["specificity", "大学生活への具体性"],
  ["actionability", "行動可能性"], ["readability", "読みやすさ"], ["would_use", "また使いたい"]];
const ISSUES: Array<[IssueKey, string]> = [["critical_error", "重大エラー"], ["factual_error", "事実誤り"],
  ["unanswered", "質問未回答"], ["university_policy_assertion", "大学固有制度の断定"],
  ["unnecessary_information", "不要情報"], ["unusable_answer", "利用不能"], ["router_error", "Router誤り"],
  ["tool_error", "Tool誤り"], ["faq_error", "FAQ誤り"], ["retrieval_error", "Retrieval誤り"],
  ["model_error", "Model誤り"]];
const UX_AXES: Array<[UXKey, string]> = [["tool_card", "Tool結果カード"], ["copy_action", "コピー操作"],
  ["input_flow", "入力フロー"], ["clarification", "Clarification"], ["streaming", "Streaming表示"],
  ["latency", "体感待ち時間"]];
const EMPTY_PAIRWISE: Pairwise = {
  chatgpt: {correctness: "unscored", specificity: "unscored", actionability: "unscored", readability: "unscored", would_use: "unscored"},
  gemini: {correctness: "unscored", specificity: "unscored", actionability: "unscored", readability: "unscored", would_use: "unscored"},
};

export default function CampusV2Evaluation() {
  const isV21 = usePathname().includes("campus-v21-eval");
  const version = isV21 ? "v2.1" : "v2";
  const endpoint = isV21 ? "campus-v21" : "campus-v2";
  const [items, setItems] = useState<Item[]>([]); const [index, setIndex] = useState(0);
  const [message, setMessage] = useState(""); const [blind, setBlind] = useState(false);
  useEffect(() => { fetch(`${API}/human-eval/${endpoint}`).then(response => response.json()).then(data => setItems(data.items ?? []))
    .catch(() => setMessage(`Campus ${version}評価データを読み込めません`)); }, [endpoint, version]);
  const item = items[index];
  const answerOrder = useMemo(() => blindOrder(item?.id ?? ""), [item?.id]);
  function update(patch: Partial<Item>) { setItems(values => values.map((value, position) => position === index ? {...value, ...patch} : value)); }
  function score(axis: Axis) { return <select value={item.scores[axis] ?? ""} onChange={event => update({scores: {...item.scores, [axis]: Number(event.target.value)}})}
    className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2"><option value="">未採点</option>{[0,1,2,3,4,5].map(value => <option key={value}>{value}</option>)}</select>; }
  async function copyQuestion() { await navigator.clipboard.writeText(item.question); setMessage("質問をコピーしました"); }
  async function save() {
    if (AXES.some(([axis]) => item.scores[axis] == null)) { setMessage(`Campus ${version}の5項目をすべて採点してください`); return; }
    if (isV21 && !item.issues_reviewed) { setMessage("問題フラグを確認済みにしてください"); return; }
    const payload = {item_id: item.id, ...item.scores, chatgpt_score: item.competitor_scores.chatgpt,
      gemini_score: item.competitor_scores.gemini, chatgpt_answer: item.chatgpt_answer,
      gemini_answer: item.gemini_answer, notes: item.notes,
      ...(isV21 ? {issue_flags: item.issue_flags, issues_reviewed: item.issues_reviewed,
        pairwise: item.pairwise ?? EMPTY_PAIRWISE, ux: item.ux} : {})};
    const response = await fetch(`${API}/human-eval/${endpoint}`, {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)});
    setMessage(response.ok ? "保存しました" : "保存に失敗しました");
    if (response.ok) setIndex(value => Math.min(value + 1, items.length - 1));
  }
  if (!item) return <main className="mx-auto max-w-7xl px-5 py-10"><h1 className="text-3xl font-bold">Campus {version} 人手評価</h1><p className="mt-4 text-slate-400">{message || "読み込み中…"}</p></main>;
  const answers = {unipilot: item.campus_answer, chatgpt: item.chatgpt_answer, gemini: item.gemini_answer};
  return <main className="mx-auto max-w-7xl px-5 py-10">
    <p className="text-xs font-semibold uppercase tracking-[.24em] text-cyan-400">Manual evaluation · External AI API OFF</p>
    <div className="mt-2 flex flex-wrap items-end justify-between gap-4"><div><h1 className="text-3xl font-bold">Campus {version} / ChatGPT / Gemini</h1>
      <p className="mt-3 text-sm text-slate-400">{index + 1} / {items.length} · {item.evaluation_bucket ?? item.difficulty}{blind ? " · Blind A/B/C" : ` · ${item.category}`}</p></div>
      {isV21 && <label className="flex items-center gap-2 rounded-lg border border-slate-700 px-3 py-2 text-sm"><input type="checkbox" checked={blind} onChange={event => setBlind(event.target.checked)} /> Blind A/B/C</label>}</div>
    <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/60 p-5"><div className="flex justify-between gap-4"><h2 className="font-semibold">質問</h2>
      <button onClick={copyQuestion} className="rounded border border-slate-600 px-3 py-1 text-xs">コピー</button></div><p className="mt-3">{item.question}</p></section>
    {blind ? <div className="mt-5 grid gap-4 lg:grid-cols-3">{answerOrder.map((origin, position) => <AnswerPanel key={origin} title={`回答 ${String.fromCharCode(65 + position)}`}
      value={answers[origin]} editable={origin !== "unipilot"} onText={value => update(origin === "chatgpt" ? {chatgpt_answer: value} : {gemini_answer: value})} />)}</div> :
      <div className="mt-5 grid gap-4 lg:grid-cols-3"><AnswerPanel title={`UniPilot Campus ${version}`} value={item.campus_answer} />
        <AnswerPanel title="ChatGPT（手動貼付）" value={item.chatgpt_answer} editable onText={value => update({chatgpt_answer: value})} />
        <AnswerPanel title="Gemini（手動貼付）" value={item.gemini_answer} editable onText={value => update({gemini_answer: value})} /></div>}
    <section className="mt-5 rounded-2xl border border-slate-800 bg-slate-900/60 p-5"><h2 className="font-semibold">UniPilot 0–5評価</h2>
      <div className="mt-4 grid gap-3 md:grid-cols-5">{AXES.map(([axis, label]) => <label key={axis} className="flex flex-col gap-2 text-sm"><span>{label}</span>{score(axis)}</label>)}</div></section>
    {isV21 && <V21Fields item={item} blind={blind} update={update} />}
    {!isV21 && <div className="mt-5 grid gap-4 lg:grid-cols-2"><CompetitorScore title="ChatGPT 総合点" value={item.competitor_scores.chatgpt} onScore={value => update({competitor_scores: {...item.competitor_scores, chatgpt: value}})} />
      <CompetitorScore title="Gemini 総合点" value={item.competitor_scores.gemini} onScore={value => update({competitor_scores: {...item.competitor_scores, gemini: value}})} /></div>}
    <textarea value={item.notes} onChange={event => update({notes: event.target.value})} placeholder="評価メモ" className="mt-5 min-h-24 w-full rounded-lg border border-slate-700 bg-slate-950 p-3" />
    <div className="mt-4 flex flex-wrap items-center gap-3"><button onClick={() => setIndex(value => Math.max(0, value - 1))} className="rounded-lg border border-slate-600 px-5 py-2">前へ</button>
      <button onClick={save} className="rounded-lg bg-cyan-500 px-5 py-2 font-semibold text-slate-950">保存して次へ</button>
      <button onClick={() => setIndex(value => Math.min(items.length - 1, value + 1))} className="rounded-lg border border-slate-600 px-5 py-2">次へ</button><span className="text-sm text-slate-400">{message}</span></div>
  </main>;
}

function V21Fields({item, blind, update}: {item: Item; blind: boolean; update: (patch: Partial<Item>) => void}) {
  const flags = item.issue_flags ?? Object.fromEntries(ISSUES.map(([key]) => [key, false])) as Record<IssueKey, boolean>;
  const pairwise = item.pairwise ?? EMPTY_PAIRWISE;
  const ux = item.ux ?? Object.fromEntries(UX_AXES.map(([key]) => [key, "not_evaluated"])) as Record<UXKey, UXResult>;
  return <>
    <section className="mt-5 rounded-2xl border border-slate-800 bg-slate-900/60 p-5"><h2 className="font-semibold">問題チェック</h2>
      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{ISSUES.map(([key, label]) => <label key={key} className="flex items-center gap-2 text-sm"><input type="checkbox" checked={flags[key]}
        onChange={event => update({issue_flags: {...flags, [key]: event.target.checked}})} />{label}</label>)}</div>
      <label className="mt-5 flex items-center gap-2 border-t border-slate-800 pt-4 text-sm"><input type="checkbox" checked={item.issues_reviewed ?? false}
        onChange={event => update({issues_reviewed: event.target.checked})} />全問題フラグを確認済み</label></section>
    <section className="mt-5 rounded-2xl border border-slate-800 bg-slate-900/60 p-5"><h2 className="font-semibold">Pairwise比較</h2>
      {blind ? <p className="mt-3 text-sm text-amber-300">Blind評価を終えてからモデル名を表示し、Pairwiseを記録してください。</p> :
        <div className="mt-4 grid gap-6 lg:grid-cols-2">{(["chatgpt", "gemini"] as const).map(competitor => <div key={competitor}><h3 className="font-medium">UniPilot vs {competitor === "chatgpt" ? "ChatGPT" : "Gemini"}</h3>
          <div className="mt-3 grid gap-3">{PAIR_AXES.map(([axis, label]) => <label key={axis} className="grid grid-cols-[1fr_10rem] items-center gap-3 text-sm"><span>{label}</span>
            <select value={pairwise[competitor][axis]} onChange={event => update({pairwise: {...pairwise, [competitor]: {...pairwise[competitor], [axis]: event.target.value as PairChoice}}})}
              className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2"><option value="unscored">未比較</option><option value="unipilot">UniPilot</option><option value="competitor">{competitor === "chatgpt" ? "ChatGPT" : "Gemini"}</option><option value="tie">Tie</option></select></label>)}</div></div>)}</div>}
    </section>
    <section className="mt-5 rounded-2xl border border-slate-800 bg-slate-900/60 p-5"><h2 className="font-semibold">Web UX</h2>
      <div className="mt-4 grid gap-3 md:grid-cols-3">{UX_AXES.map(([axis, label]) => <label key={axis} className="grid grid-cols-[1fr_9rem] items-center gap-2 text-sm"><span>{label}</span>
        <select value={ux[axis]} onChange={event => update({ux: {...ux, [axis]: event.target.value as UXResult}})} className="rounded-lg border border-slate-700 bg-slate-950 px-2 py-2"><option value="not_evaluated">未評価</option><option value="pass">Pass</option><option value="fail">Fail</option><option value="not_applicable">対象外</option></select></label>)}</div></section>
  </>;
}

function AnswerPanel({title, value, editable = false, onText}: {title: string; value: string; editable?: boolean; onText?: (value: string) => void}) {
  return <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4"><div className="flex items-center justify-between gap-3"><h2 className="font-semibold">{title}</h2>
    <button onClick={() => navigator.clipboard.writeText(value)} className="rounded border border-slate-600 px-2 py-1 text-xs">コピー</button></div>
    {editable ? <textarea value={value} onChange={event => onText?.(event.target.value)} placeholder="同じ質問への回答を手動で貼り付け" className="mt-3 min-h-64 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm" /> :
      <p className="mt-3 min-h-64 whitespace-pre-wrap text-sm">{value}</p>}</section>;
}

function CompetitorScore({title, value, onScore}: {title: string; value: number | null; onScore: (value: number) => void}) {
  return <label className="flex items-center justify-between rounded-2xl border border-slate-800 bg-slate-900/60 p-4"><span>{title}</span>
    <select value={value ?? ""} onChange={event => onScore(Number(event.target.value))} className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2"><option value="">未採点</option>{[0,1,2,3,4,5].map(score => <option key={score}>{score}</option>)}</select></label>;
}

function blindOrder(id: string): Array<"unipilot" | "chatgpt" | "gemini"> {
  const orders: Array<Array<"unipilot" | "chatgpt" | "gemini">> = [
    ["unipilot", "chatgpt", "gemini"], ["unipilot", "gemini", "chatgpt"], ["chatgpt", "unipilot", "gemini"],
    ["chatgpt", "gemini", "unipilot"], ["gemini", "unipilot", "chatgpt"], ["gemini", "chatgpt", "unipilot"],
  ];
  return orders[[...id].reduce((sum, value) => sum + value.charCodeAt(0), 0) % orders.length];
}
