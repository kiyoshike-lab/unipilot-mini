"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const POSITION_KEY = "unipilot-campus-v21-quick-eval-position";

type Rating = "good" | "close" | "bad";
type BadReason = "incorrect" | "unanswered" | "too_short" | "unclear" | "router" | "other";
type Item = {
  id: string;
  question: string;
  campus_answer: string;
  category: string;
  evaluation_bucket: string;
  focus: string;
  campus_metadata?: {action?: string; route?: string; tool?: string | null; latency_ms?: number};
  quick_rating: Rating | null;
  quick_reason: BadReason | null;
};
type Summary = {
  status: "PENDING" | "COMPLETE";
  completed: number;
  pending: number;
  total: number;
  counts: Record<Rating, number>;
  rates_percent: Record<Rating, number>;
  quick_human_gate: {status: "PENDING" | "PASS_CANDIDATE" | "NEEDS_IMPROVEMENT" | "FAIL"; label: string; is_simplified: true};
};

const RATINGS: Array<{value: Rating; symbol: string; label: string; key: string; className: string}> = [
  {value: "good", symbol: "◎", label: "良い", key: "1", className: "border-emerald-500 bg-emerald-950/50 text-emerald-200 hover:bg-emerald-900"},
  {value: "close", symbol: "△", label: "惜しい", key: "2", className: "border-amber-500 bg-amber-950/50 text-amber-200 hover:bg-amber-900"},
  {value: "bad", symbol: "×", label: "ダメ", key: "3", className: "border-rose-500 bg-rose-950/50 text-rose-200 hover:bg-rose-900"},
];
const BAD_REASONS: Array<[BadReason, string]> = [
  ["incorrect", "内容が間違い"], ["unanswered", "質問に答えていない"], ["too_short", "短すぎる"],
  ["unclear", "分かりにくい"], ["router", "Routerがおかしい"], ["other", "その他"],
];

export default function CampusV21QuickEvaluation() {
  const [items, setItems] = useState<Item[]>([]);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [index, setIndex] = useState(0);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("読み込み中…");
  const [lastBadId, setLastBadId] = useState<string | null>(null);
  const [reviewing, setReviewing] = useState(false);

  useEffect(() => {
    fetch(`${API}/human-eval/campus-v21/quick`).then(response => {
      if (!response.ok) throw new Error("簡易評価データを読み込めません");
      return response.json();
    }).then(data => {
      const loaded = (data.items ?? []) as Item[];
      setItems(loaded);
      setSummary(data as Summary);
      const storedId = localStorage.getItem(POSITION_KEY);
      const storedIndex = storedId ? loaded.findIndex(item => item.id === storedId && !item.quick_rating) : -1;
      const firstPending = loaded.findIndex(item => !item.quick_rating);
      setIndex(storedIndex >= 0 ? storedIndex : firstPending >= 0 ? firstPending : 0);
      setMessage(loaded.length === 20 ? "1・2・3キーでも評価できます" : `評価項目数が20ではありません（${loaded.length}件）`);
    }).catch(error => setMessage(error instanceof Error ? error.message : "読み込みに失敗しました"));
  }, []);

  const item = items[index];

  useEffect(() => {
    if (item) localStorage.setItem(POSITION_KEY, item.id);
  }, [item]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (target?.matches("input, textarea, select, button") || saving || !item || (summary?.completed === 20 && !reviewing)) return;
      const rating = event.key === "1" ? "good" : event.key === "2" ? "close" : event.key === "3" ? "bad" : null;
      if (rating) {
        event.preventDefault();
        void saveRating(rating);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });

  async function saveRating(rating: Rating, reason: BadReason | null = null, advance = true) {
    if (!item || saving) return;
    setSaving(true);
    setMessage("保存中…");
    const response = await fetch(`${API}/human-eval/campus-v21/quick`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({item_id: item.id, rating, reason}),
    }).catch(() => null);
    if (!response?.ok) {
      setSaving(false);
      setMessage("保存できませんでした。もう一度押してください");
      return;
    }
    const data = await response.json();
    const updated = items.map(candidate => candidate.id === item.id
      ? {...candidate, quick_rating: rating, quick_reason: rating === "bad" ? reason : null} : candidate);
    setItems(updated);
    setSummary(data.summary as Summary);
    setSaving(false);
    setMessage("保存しました");
    if (rating === "bad" && !reason) setLastBadId(item.id);
    if (advance) moveToNextPending(updated, item.id);
  }

  function moveToNextPending(updated: Item[], currentId: string) {
    const currentIndex = updated.findIndex(candidate => candidate.id === currentId);
    const later = updated.findIndex((candidate, position) => position > currentIndex && !candidate.quick_rating);
    const wrapped = updated.findIndex(candidate => !candidate.quick_rating);
    if (later >= 0 || wrapped >= 0) {
      setIndex(later >= 0 ? later : wrapped);
    } else {
      setReviewing(false);
    }
  }

  async function saveBadReason(reason: BadReason) {
    const badItem = items.find(candidate => candidate.id === lastBadId);
    if (!badItem) return;
    setSaving(true);
    const response = await fetch(`${API}/human-eval/campus-v21/quick`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({item_id: badItem.id, rating: "bad", reason}),
    }).catch(() => null);
    if (response?.ok) {
      const data = await response.json();
      setItems(values => values.map(candidate => candidate.id === badItem.id ? {...candidate, quick_reason: reason} : candidate));
      setSummary(data.summary as Summary);
      setMessage("×の理由を保存しました");
    } else {
      setMessage("理由を保存できませんでした");
    }
    setSaving(false);
    setLastBadId(null);
  }

  async function exportResults() {
    setMessage("結果を生成中…");
    const response = await fetch(`${API}/human-eval/campus-v21/quick/export`, {method: "POST"}).catch(() => null);
    if (!response?.ok) { setMessage("結果を生成できませんでした"); return; }
    const data = await response.json();
    setMessage(`${data.results_path} と ${data.report_path} を生成しました`);
  }

  if (!summary || !item) {
    return <main className="mx-auto max-w-4xl px-5 py-10"><h1 className="text-3xl font-bold">Campus v2.1 かんたん評価</h1><p className="mt-4 text-slate-400">{message}</p></main>;
  }

  if (summary.completed === 20 && !reviewing) {
    return <main className="mx-auto max-w-4xl px-5 py-10">
      <p className="text-xs font-semibold uppercase tracking-[.24em] text-cyan-400">Quick Human Evaluation · 20 / 20</p>
      <h1 className="mt-2 text-3xl font-bold">評価完了</h1>
      <section className="mt-6 rounded-2xl border border-slate-700 bg-slate-900/70 p-6 text-center">
        <p className="text-sm text-slate-400">簡易Human Gate</p><p className={`mt-2 text-4xl font-bold ${gateColor(summary.quick_human_gate.status)}`}>{summary.quick_human_gate.label}</p>
        <div className="mt-6 grid grid-cols-3 gap-3"><Result label="◎ 良い" count={summary.counts.good} rate={summary.rates_percent.good} />
          <Result label="△ 惜しい" count={summary.counts.close} rate={summary.rates_percent.close} />
          <Result label="× ダメ" count={summary.counts.bad} rate={summary.rates_percent.bad} /></div>
        <p className="mt-5 text-xs text-amber-300">これは20問による簡易Human Gateです。本番昇格を自動決定するものではありません。</p>
      </section>
      <div className="mt-5 flex flex-wrap gap-3"><button onClick={exportResults} className="rounded-lg bg-cyan-500 px-5 py-2 font-semibold text-slate-950">結果JSON・Reportを生成</button>
        <button onClick={() => {setReviewing(true); setIndex(0);}} className="rounded-lg border border-slate-600 px-5 py-2">回答を見直す</button></div>
      <p className="mt-3 text-sm text-slate-400">{message}</p>
      <BadReasonPrompt visible={Boolean(lastBadId)} disabled={saving} onSelect={saveBadReason} onSkip={() => setLastBadId(null)} />
    </main>;
  }

  return <main className="mx-auto max-w-4xl px-5 py-10">
    <p className="text-xs font-semibold uppercase tracking-[.24em] text-cyan-400">Campus v2.1 RC · かんたん品質確認</p>
    <div className="mt-3 grid grid-cols-4 gap-2 rounded-xl border border-slate-700 bg-slate-900/80 p-3 text-center">
      <Progress label="進捗" value={`${summary.completed} / 20`} /><Progress label="◎" value={summary.counts.good} />
      <Progress label="△" value={summary.counts.close} /><Progress label="×" value={summary.counts.bad} />
    </div>

    <BadReasonPrompt visible={Boolean(lastBadId)} disabled={saving} onSelect={saveBadReason} onSkip={() => setLastBadId(null)} />

    <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
      <p className="text-sm text-slate-500">質問 {index + 1}</p><h1 className="mt-2 text-xl font-semibold leading-relaxed">{item.question}</h1>
      <div className="mt-6 rounded-xl bg-slate-950/80 p-5"><p className="text-xs font-semibold uppercase tracking-wider text-cyan-400">UniPilot</p>
        <p className="mt-3 whitespace-pre-wrap leading-relaxed">{item.campus_answer}</p></div>
      <details className="mt-4 text-sm text-slate-400"><summary className="cursor-pointer">詳細を見る</summary>
        <div className="mt-2 grid gap-2 sm:grid-cols-2"><span>カテゴリ: {item.category}</span><span>難易度: {item.evaluation_bucket}</span>
          <span>route: {item.campus_metadata?.route ?? "—"}</span><span>type: {item.campus_metadata?.action ?? "—"}</span>
          <span>代表観点: {item.focus}</span><span>response time: {item.campus_metadata?.latency_ms?.toFixed(2) ?? "—"} ms</span></div>
      </details>
    </section>

    <div className="mt-6 grid gap-3 sm:grid-cols-3">{RATINGS.map(rating => <button key={rating.value} disabled={saving} onClick={() => saveRating(rating.value)}
      className={`rounded-2xl border px-4 py-6 text-center transition disabled:opacity-40 ${rating.className}`}>
      <span className="block text-4xl font-bold">{rating.symbol}</span><span className="mt-2 block font-semibold">{rating.label}</span>
      <span className="mt-1 block text-xs opacity-70">キー {rating.key}</span></button>)}</div>

    <div className="mt-5 flex items-center justify-between gap-3"><button disabled={index === 0 || saving} onClick={() => {setReviewing(true); setIndex(value => Math.max(0, value - 1));}}
      className="rounded-lg border border-slate-700 px-3 py-2 text-sm disabled:opacity-30">前の回答を見る</button><p className="text-sm text-slate-400">{message}</p></div>
    <p className="mt-6 text-center text-xs text-slate-500">◎ 80%以上かつ × 5%以下でPASS候補。これは簡易Human Gateです。</p>
  </main>;
}

function BadReasonPrompt({visible, disabled, onSelect, onSkip}: {visible: boolean; disabled: boolean; onSelect: (reason: BadReason) => void; onSkip: () => void}) {
  if (!visible) return null;
  return <section className="mt-4 rounded-xl border border-rose-900 bg-rose-950/30 p-4"><div className="flex flex-wrap items-center justify-between gap-2"><p className="text-sm font-semibold">前の「×」の理由（任意）</p>
    <button onClick={onSkip} className="text-xs text-slate-400">選ばず続ける</button></div><div className="mt-3 flex flex-wrap gap-2">{BAD_REASONS.map(([reason, label]) => <button key={reason} disabled={disabled} onClick={() => onSelect(reason)}
      className="rounded-full border border-rose-800 px-3 py-1.5 text-xs text-rose-200 disabled:opacity-40">{label}</button>)}</div></section>;
}

function Progress({label, value}: {label: string; value: string | number}) {
  return <div><span className="block text-xs text-slate-500">{label}</span><span className="text-lg font-bold">{value}</span></div>;
}

function Result({label, count, rate}: {label: string; count: number; rate: number}) {
  return <div className="rounded-xl bg-slate-950/80 p-4"><span className="block text-sm text-slate-400">{label}</span><span className="mt-1 block text-2xl font-bold">{count}</span><span className="text-sm">{rate.toFixed(1)}%</span></div>;
}

function gateColor(status: Summary["quick_human_gate"]["status"]): string {
  if (status === "PASS_CANDIDATE") return "text-emerald-300";
  if (status === "NEEDS_IMPROVEMENT") return "text-amber-300";
  if (status === "FAIL") return "text-rose-300";
  return "text-slate-300";
}
