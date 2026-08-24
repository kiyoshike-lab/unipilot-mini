"use client";
import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
type Axis = "correctness" | "relevance" | "actionable" | "naturalness" | "would_use_again";
type Item = { id: string; question: string; category: string; difficulty: string; campus_answer: string;
  chatgpt_answer: string; gemini_answer: string; scores: Record<Axis, number | null>;
  competitor_scores: {chatgpt: number | null; gemini: number | null}; notes: string };

const AXES: Array<[Axis, string]> = [["correctness", "正確さ"], ["relevance", "質問との一致"],
  ["actionable", "すぐ行動できる"], ["naturalness", "自然さ"], ["would_use_again", "また使いたい"]];

export default function CampusV2Evaluation() {
  const isV21 = usePathname().includes("campus-v21-eval");
  const version = isV21 ? "v2.1" : "v2";
  const endpoint = isV21 ? "campus-v21" : "campus-v2";
  const [items, setItems] = useState<Item[]>([]); const [index, setIndex] = useState(0); const [message, setMessage] = useState("");
  useEffect(() => { fetch(`${API}/human-eval/${endpoint}`).then(response => response.json()).then(data => setItems(data.items ?? []))
    .catch(() => setMessage(`Campus ${version}評価データを読み込めません`)); }, [endpoint, version]);
  const item = items[index];
  function update(patch: Partial<Item>) { setItems(values => values.map((value, position) => position === index ? {...value, ...patch} : value)); }
  function score(axis: Axis) { return <select value={item.scores[axis] ?? ""} onChange={event => update({scores: {...item.scores, [axis]: Number(event.target.value)}})}
    className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2"><option value="">未採点</option>{[0,1,2,3,4,5].map(value => <option key={value}>{value}</option>)}</select>; }
  async function save() {
    if (AXES.some(([axis]) => item.scores[axis] == null)) { setMessage(`Campus ${version}の5項目をすべて採点してください`); return; }
    const response = await fetch(`${API}/human-eval/${endpoint}`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({
      item_id: item.id, ...item.scores, chatgpt_score: item.competitor_scores.chatgpt, gemini_score: item.competitor_scores.gemini,
      chatgpt_answer: item.chatgpt_answer, gemini_answer: item.gemini_answer, notes: item.notes,
    })});
    setMessage(response.ok ? "保存しました" : "保存に失敗しました"); if (response.ok) setIndex(value => Math.min(value + 1, items.length - 1));
  }
  if (!item) return <main className="mx-auto max-w-6xl px-5 py-10"><h1 className="text-3xl font-bold">Campus {version} 人手評価</h1><p className="mt-4 text-slate-400">{message || "読み込み中…"}</p></main>;
  return <main className="mx-auto max-w-6xl px-5 py-10"><p className="text-xs font-semibold uppercase tracking-[.24em] text-cyan-400">Manual evaluation · External AI API OFF</p>
    <h1 className="mt-2 text-3xl font-bold">Campus {version} / ChatGPT / Gemini</h1><p className="mt-3 text-sm text-slate-400">{index + 1} / {items.length} · {item.difficulty} · {item.category}</p>
    <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/60 p-5"><h2 className="font-semibold">質問</h2><p className="mt-3">{item.question}</p></section>
    <section className="mt-5 rounded-2xl border border-slate-800 bg-slate-900/60 p-5"><h2 className="font-semibold">Campus {version}</h2><p className="mt-3 whitespace-pre-wrap text-sm">{item.campus_answer}</p>
      <div className="mt-5 grid gap-3 md:grid-cols-5">{AXES.map(([axis, label]) => <label key={axis} className="flex flex-col gap-2 text-sm"><span>{label}</span>{score(axis)}</label>)}</div></section>
    <div className="mt-5 grid gap-4 lg:grid-cols-2">
      <Competitor title="ChatGPT（同じ質問をUIで実行して貼付）" value={item.chatgpt_answer} score={item.competitor_scores.chatgpt}
        onText={value => update({chatgpt_answer: value})} onScore={value => update({competitor_scores: {...item.competitor_scores, chatgpt: value}})} />
      <Competitor title="Gemini（同じ質問をUIで実行して貼付）" value={item.gemini_answer} score={item.competitor_scores.gemini}
        onText={value => update({gemini_answer: value})} onScore={value => update({competitor_scores: {...item.competitor_scores, gemini: value}})} />
    </div>
    <textarea value={item.notes} onChange={event => update({notes: event.target.value})} placeholder="評価メモ" className="mt-5 min-h-24 w-full rounded-lg border border-slate-700 bg-slate-950 p-3" />
    <button onClick={save} className="mt-4 rounded-lg bg-cyan-500 px-5 py-2 font-semibold text-slate-950">保存して次へ</button><span className="ml-3 text-sm text-slate-400">{message}</span>
  </main>;
}

function Competitor({title, value, score, onText, onScore}: {title: string; value: string; score: number | null; onText: (value: string) => void; onScore: (value: number) => void}) {
  return <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4"><div className="flex items-center justify-between gap-3"><h2 className="font-semibold">{title}</h2>
    <select value={score ?? ""} onChange={event => onScore(Number(event.target.value))} className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2"><option value="">未採点</option>{[0,1,2,3,4,5].map(value => <option key={value}>{value}</option>)}</select></div>
    <textarea value={value} onChange={event => onText(event.target.value)} className="mt-3 min-h-64 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm" /></section>;
}
