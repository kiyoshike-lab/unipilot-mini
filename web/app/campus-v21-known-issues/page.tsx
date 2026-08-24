"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
type Group = "hallucination" | "router" | "retrieval";
type Review = {status: "pending" | "confirmed" | "not_reproduced" | "accepted_risk";
  severity: "unreviewed" | "low" | "medium" | "high" | "critical"; blocks_production: boolean; notes: string};
type Issue = {id: string; question: string; answer?: string; automatic_flag: string; gold_category?: string;
  predicted_category?: string; expected_guardrail?: string; returned?: string[]; relevant?: string[]; human_review: Review; group: Group};

export default function KnownIssueReview() {
  const [items, setItems] = useState<Issue[]>([]); const [filter, setFilter] = useState<Group | "all">("all");
  const [message, setMessage] = useState("読み込み中…");
  useEffect(() => { fetch(`${API}/human-eval/campus-v21/known-issues`).then(response => response.json()).then(data => {
    const flat = (Object.entries(data.groups ?? {}) as Array<[Group, Omit<Issue, "group">[]]>).flatMap(([group, rows]) => rows.map(row => ({...row, group})));
    setItems(flat); setMessage(`${data.reviewed} / ${data.total} 確認済み`);
  }).catch(() => setMessage("既知問題キューを読み込めません")); }, []);
  function update(id: string, review: Review) { setItems(values => values.map(item => item.id === id ? {...item, human_review: review} : item)); }
  async function save(item: Issue) {
    const response = await fetch(`${API}/human-eval/campus-v21/known-issues`, {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({item_id: item.id, group: item.group, ...item.human_review})});
    setMessage(response.ok ? `${item.id} を保存しました` : `${item.id} の保存に失敗しました`);
  }
  const shown = filter === "all" ? items : items.filter(item => item.group === filter);
  return <main className="mx-auto max-w-6xl px-5 py-10"><p className="text-xs font-semibold uppercase tracking-[.24em] text-amber-300">Frozen RC review queue</p>
    <h1 className="mt-2 text-3xl font-bold">Campus v2.1 既知問題の人間確認</h1>
    <p className="mt-3 text-sm text-slate-400">回答ロジックは変更しません。23件を人間が確認するまでHuman GateはPENDINGです。{message}</p>
    <div className="mt-5 flex flex-wrap gap-2">{(["all", "hallucination", "router", "retrieval"] as const).map(value => <button key={value} onClick={() => setFilter(value)}
      className={`rounded-lg border px-3 py-2 text-sm ${filter === value ? "border-cyan-400 text-cyan-300" : "border-slate-700"}`}>{value}</button>)}</div>
    <div className="mt-6 grid gap-5">{shown.map(item => <IssueCard key={`${item.group}:${item.id}`} item={item} onUpdate={review => update(item.id, review)} onSave={() => save(item)} />)}</div>
  </main>;
}

function IssueCard({item, onUpdate, onSave}: {item: Issue; onUpdate: (review: Review) => void; onSave: () => void}) {
  const review = item.human_review;
  return <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5"><div className="flex flex-wrap justify-between gap-2"><div><p className="text-xs uppercase tracking-wider text-amber-300">{item.group} · {item.automatic_flag}</p><h2 className="mt-1 font-semibold">{item.id}</h2></div>
    <p className="text-xs text-slate-400">{item.gold_category && `gold: ${item.gold_category}`} {item.predicted_category && `→ predicted: ${item.predicted_category}`}</p></div>
    <p className="mt-4">{item.question}</p>{item.answer && <pre className="mt-3 whitespace-pre-wrap rounded-lg bg-slate-950 p-3 text-sm">{item.answer}</pre>}
    {item.expected_guardrail && <p className="mt-3 text-sm text-slate-300">期待: {item.expected_guardrail}</p>}
    {item.returned && <p className="mt-3 text-xs text-slate-400">returned: {item.returned.join(", ")} / relevant: {(item.relevant ?? []).join(", ")}</p>}
    <div className="mt-4 grid gap-3 md:grid-cols-[12rem_10rem_1fr]">
      <select value={review.status} onChange={event => onUpdate({...review, status: event.target.value as Review["status"]})} className="rounded-lg border border-slate-700 bg-slate-950 p-2"><option value="pending">未確認</option><option value="confirmed">問題を確認</option><option value="not_reproduced">問題なし</option><option value="accepted_risk">リスク受容</option></select>
      <select value={review.severity} onChange={event => onUpdate({...review, severity: event.target.value as Review["severity"]})} className="rounded-lg border border-slate-700 bg-slate-950 p-2"><option value="unreviewed">重大度未評価</option><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option><option value="critical">Critical</option></select>
      <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={review.blocks_production} onChange={event => onUpdate({...review, blocks_production: event.target.checked})} />本番昇格を停止</label>
    </div>
    <textarea value={review.notes} onChange={event => onUpdate({...review, notes: event.target.value})} placeholder="確認メモ" className="mt-3 min-h-20 w-full rounded-lg border border-slate-700 bg-slate-950 p-3" />
    <button onClick={onSave} className="mt-3 rounded-lg bg-cyan-500 px-4 py-2 font-semibold text-slate-950">この確認を保存</button></section>;
}
