"use client";

import {useEffect, useMemo, useState} from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const POSITION_KEY = "unipilot-campus-ai-review-position";

type Decision = "adopt" | "revise" | "reject";
type ReviewItem = {
  item_id: string;
  ai_judge_score: number;
  improved_score: number;
  question: string;
  category: string;
  route: string;
  source_ids: string[];
  original_answer: string;
  problems: string[];
  review_reasons: string[];
  critique: string[];
  improved_answer: string;
  decision?: Decision;
  edited_answer?: string;
  notes?: string;
};

type Queue = {
  items: ReviewItem[];
  reviewed: number;
  pending: number;
  review_required: number;
  decision_counts: Record<Decision, number>;
  automatic_training: false;
  external_ai_api: "OFF";
};

export default function CampusAIReview() {
  const [queue, setQueue] = useState<Queue | null>(null);
  const [index, setIndex] = useState(0);
  const [editedAnswer, setEditedAnswer] = useState("");
  const [notes, setNotes] = useState("");
  const [message, setMessage] = useState("読み込み中…");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    fetch(`${API}/ai-review/campus`).then(response => {
      if (!response.ok) throw new Error("AIレビューキューを読み込めません");
      return response.json();
    }).then((data: Queue) => {
      setQueue(data);
      const storedId = localStorage.getItem(POSITION_KEY);
      const storedIndex = storedId ? data.items.findIndex(item => item.item_id === storedId && !item.decision) : -1;
      const firstPending = data.items.findIndex(item => !item.decision);
      setIndex(storedIndex >= 0 ? storedIndex : firstPending >= 0 ? firstPending : 0);
      setMessage("AI Quality GateはHuman Gateの代替ではありません");
    }).catch(error => setMessage(error instanceof Error ? error.message : "読み込みに失敗しました"));
  }, []);

  const item = queue?.items[index];
  useEffect(() => {
    if (!item) return;
    localStorage.setItem(POSITION_KEY, item.item_id);
    setEditedAnswer(item.edited_answer || item.improved_answer);
    setNotes(item.notes || "");
  }, [item]);

  const counts = useMemo(() => queue?.decision_counts ?? {adopt: 0, revise: 0, reject: 0}, [queue]);

  async function save(decision: Decision) {
    if (!queue || !item || saving) return;
    setSaving(true);
    setMessage("保存中…");
    const response = await fetch(`${API}/ai-review/campus`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({item_id: item.item_id, decision,
        edited_answer: decision === "reject" ? "" : editedAnswer, notes}),
    }).catch(() => null);
    if (!response?.ok) {
      setSaving(false);
      setMessage("保存できませんでした");
      return;
    }
    const previous = item.decision;
    const updatedItems = queue.items.map(candidate => candidate.item_id === item.item_id
      ? {...candidate, decision, edited_answer: decision === "reject" ? "" : editedAnswer, notes} : candidate);
    const nextCounts = {...counts};
    if (previous) nextCounts[previous] -= 1;
    nextCounts[decision] += 1;
    const reviewed = updatedItems.filter(candidate => candidate.decision).length;
    setQueue({...queue, items: updatedItems, reviewed, pending: updatedItems.length - reviewed,
      decision_counts: nextCounts});
    const next = updatedItems.findIndex((candidate, position) => position > index && !candidate.decision);
    const wrapped = updatedItems.findIndex(candidate => !candidate.decision);
    if (next >= 0 || wrapped >= 0) setIndex(next >= 0 ? next : wrapped);
    setSaving(false);
    setMessage(`${item.item_id} を保存しました（自動学習はしません）`);
  }

  if (!queue || !item) return <main className="mx-auto max-w-5xl px-5 py-10"><h1 className="text-3xl font-bold">Campus AI Review</h1><p className="mt-4 text-slate-400">{message}</p></main>;

  return <main className="mx-auto max-w-5xl px-5 py-10">
    <p className="text-xs font-semibold uppercase tracking-[.24em] text-cyan-400">Local deterministic judge · External AI API: OFF</p>
    <h1 className="mt-2 text-3xl font-bold">Campus AI 改善レビュー</h1>
    <p className="mt-3 text-sm text-amber-300">AI Quality GateはHuman Gateの代替ではありません。採用回答も自動学習・本番反映されません。</p>
    <div className="mt-5 grid grid-cols-5 gap-2 rounded-xl border border-slate-700 bg-slate-900/80 p-3 text-center">
      <Metric label="確認" value={`${queue.reviewed} / ${queue.review_required}`} />
      <Metric label="未確認" value={queue.pending} /><Metric label="採用" value={counts.adopt} />
      <Metric label="修正" value={counts.revise} /><Metric label="却下" value={counts.reject} />
    </div>

    <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
      <div className="flex flex-wrap justify-between gap-3"><div><p className="text-xs text-slate-400">{index + 1} / {queue.items.length} · {item.item_id}</p>
        <p className="mt-1 text-xs text-cyan-300">category: {item.category} · route: {item.route}</p></div>
        <p className="text-sm">AI score <span className="font-bold text-amber-300">{item.ai_judge_score}</span> → <span className="font-bold text-emerald-300">{item.improved_score}</span></p></div>
      <h2 className="mt-5 text-xl font-semibold">{item.question}</h2>

      <div className="mt-5 grid gap-4 lg:grid-cols-2">
        <Answer title="元の回答" value={item.original_answer} />
        <Answer title="改善案" value={item.improved_answer} />
      </div>
      <div className="mt-4 rounded-xl border border-amber-900/70 bg-amber-950/20 p-4 text-sm">
        <p className="font-semibold text-amber-200">検出理由: {item.review_reasons.join(" / ")}</p>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-slate-300">{item.critique.map(value => <li key={value}>{value}</li>)}</ul>
        <p className="mt-2 text-xs text-slate-400">source: {item.source_ids.length ? item.source_ids.join(", ") : "なし"}</p>
      </div>

      <details className="mt-5 rounded-xl border border-slate-700 p-4" open={item.decision === "revise"}>
        <summary className="cursor-pointer font-semibold">改善案を編集・メモ</summary>
        <label className="mt-4 block text-sm text-slate-300">採用候補回答
          <textarea value={editedAnswer} onChange={event => setEditedAnswer(event.target.value)} className="mt-2 min-h-48 w-full rounded-lg border border-slate-700 bg-slate-950 p-3" />
        </label>
        <label className="mt-3 block text-sm text-slate-300">人間メモ
          <textarea value={notes} onChange={event => setNotes(event.target.value)} className="mt-2 min-h-20 w-full rounded-lg border border-slate-700 bg-slate-950 p-3" />
        </label>
      </details>

      <div className="mt-5 grid gap-3 sm:grid-cols-3">
        <button disabled={saving} onClick={() => save("adopt")} className="rounded-lg border border-emerald-500 bg-emerald-950/40 px-4 py-3 font-semibold text-emerald-200 disabled:opacity-50">採用</button>
        <button disabled={saving} onClick={() => save("revise")} className="rounded-lg border border-amber-500 bg-amber-950/40 px-4 py-3 font-semibold text-amber-200 disabled:opacity-50">修正必要</button>
        <button disabled={saving} onClick={() => save("reject")} className="rounded-lg border border-rose-500 bg-rose-950/40 px-4 py-3 font-semibold text-rose-200 disabled:opacity-50">却下</button>
      </div>
      <div className="mt-4 flex justify-between"><button disabled={index === 0} onClick={() => setIndex(value => Math.max(0, value - 1))} className="rounded border border-slate-700 px-3 py-2 disabled:opacity-30">前へ</button>
        <button disabled={index >= queue.items.length - 1} onClick={() => setIndex(value => Math.min(queue.items.length - 1, value + 1))} className="rounded border border-slate-700 px-3 py-2 disabled:opacity-30">次へ</button></div>
    </section>
    <p className="mt-4 text-sm text-slate-400">{message}</p>
  </main>;
}

function Metric({label, value}: {label: string; value: string | number}) {
  return <div><p className="text-xs text-slate-400">{label}</p><p className="mt-1 font-bold">{value}</p></div>;
}

function Answer({title, value}: {title: string; value: string}) {
  return <div><h3 className="text-sm font-semibold text-slate-300">{title}</h3><pre className="mt-2 min-h-44 whitespace-pre-wrap rounded-xl bg-slate-950 p-4 text-sm leading-6">{value}</pre></div>;
}
