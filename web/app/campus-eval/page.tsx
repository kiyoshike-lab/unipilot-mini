"use client";
import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
type Winner = "campus" | "chatgpt" | "gemini" | "tie" | "unscored";
type Item = { id: string; question: string; category: string; campus_answer: string; chatgpt_answer: string | null;
  gemini_answer: string | null; scores: {campus: number | null; chatgpt: number | null; gemini: number | null};
  winners: Record<string, Winner>; notes: string };

export default function CampusEvaluation() {
  const [items, setItems] = useState<Item[]>([]); const [index, setIndex] = useState(0); const [message, setMessage] = useState("");
  useEffect(() => { fetch(`${API}/human-eval/campus`).then(response => response.json()).then(data => setItems(data.items ?? []))
    .catch(() => setMessage("Campus評価データを読み込めません")); }, []);
  const item = items[index];
  function update(patch: Partial<Item>) { setItems(values => values.map((value, position) => position === index ? {...value, ...patch} : value)); }
  async function save() {
    if (!item || item.scores.campus == null) { setMessage("Campus scoreを選択してください"); return; }
    const body = {item_id: item.id, campus_score: item.scores.campus, chatgpt_score: item.scores.chatgpt,
      gemini_score: item.scores.gemini, chatgpt_answer: item.chatgpt_answer ?? "", gemini_answer: item.gemini_answer ?? "",
      correct_winner: item.winners.correct ?? "unscored", specific_winner: item.winners.specific ?? "unscored",
      usable_winner: item.winners.usable ?? "unscored", fast_winner: item.winners.fast ?? "unscored",
      student_preference: item.winners.student_preference ?? "unscored", notes: item.notes ?? ""};
    const response = await fetch(`${API}/human-eval/campus`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    setMessage(response.ok ? "保存しました" : "保存に失敗しました"); if (response.ok) setIndex(value => Math.min(value + 1, items.length - 1));
  }
  if (!item) return <main className="mx-auto max-w-5xl px-5 py-10"><h1 className="text-3xl font-bold">Campus比較評価</h1><p className="mt-4 text-slate-400">{message || "読み込み中…"}</p></main>;
  const score = (target: "campus" | "chatgpt" | "gemini") => <select value={item.scores[target] ?? ""} onChange={event => update({scores: {...item.scores, [target]: Number(event.target.value)}})}
    className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2"><option value="">未採点</option>{[0,1,2,3,4,5].map(value => <option key={value}>{value}</option>)}</select>;
  const winner = (key: string) => <select value={item.winners[key] ?? "unscored"} onChange={event => update({winners: {...item.winners, [key]: event.target.value as Winner}})}
    className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2"><option value="unscored">未採点</option><option value="campus">Campus</option><option value="chatgpt">ChatGPT</option><option value="gemini">Gemini</option><option value="tie">同等</option></select>;
  return <main className="mx-auto max-w-5xl px-5 py-10"><p className="text-xs font-semibold uppercase tracking-[.24em] text-cyan-400">Manual only · External API OFF</p>
    <h1 className="mt-2 text-3xl font-bold">Campus / ChatGPT / Gemini 比較</h1><p className="mt-3 text-sm text-slate-400">{index + 1} / {items.length} · {item.category}</p>
    <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/60 p-5"><h2 className="font-semibold">質問</h2><p className="mt-3">{item.question}</p></section>
    <div className="mt-5 grid gap-4 lg:grid-cols-3">
      <Answer title="Campus v1" value={item.campus_answer} readOnly score={score("campus")} />
      <Answer title="ChatGPT（人が貼り付け）" value={item.chatgpt_answer ?? ""} score={score("chatgpt")} onChange={value => update({chatgpt_answer: value})} />
      <Answer title="Gemini（人が貼り付け）" value={item.gemini_answer ?? ""} score={score("gemini")} onChange={value => update({gemini_answer: value})} />
    </div>
    <section className="mt-5 rounded-2xl border border-slate-800 bg-slate-900/60 p-5"><h2 className="font-semibold">勝敗</h2>
      <div className="mt-4 grid gap-3 md:grid-cols-2">{[["correct","正確さ"],["specific","具体性"],["usable","使いやすさ"],["fast","速さ"],["student_preference","学生が使いたい"]].map(([key,label]) => <label key={key} className="flex items-center justify-between gap-3 text-sm"><span>{label}</span>{winner(key)}</label>)}</div>
      <textarea value={item.notes ?? ""} onChange={event => update({notes: event.target.value})} placeholder="確認メモ" className="mt-4 min-h-24 w-full rounded-lg border border-slate-700 bg-slate-950 p-3" />
      <button onClick={save} className="mt-4 rounded-lg bg-cyan-500 px-5 py-2 font-semibold text-slate-950">保存して次へ</button><span className="ml-3 text-sm text-slate-400">{message}</span>
    </section></main>;
}

function Answer({title, value, readOnly, score, onChange}: {title: string; value: string; readOnly?: boolean; score: React.ReactNode; onChange?: (value: string) => void}) {
  return <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4"><div className="flex items-center justify-between gap-3"><h2 className="font-semibold">{title}</h2>{score}</div>
    <textarea readOnly={readOnly} value={value} onChange={event => onChange?.(event.target.value)} className="mt-3 min-h-72 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm" /></section>;
}
