"use client";
import { useEffect, useState } from "react";
const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
type Checkpoint = { path: string; size_bytes: number };
type Info = Record<string, string | number | boolean | null>;
type Evaluation = { metrics?: Record<string, number | null>; validation_loss?: number; perplexity?: number };
type Training = { step?: number; stage?: string; train_loss?: number; stage_validation_loss?: number; tokens_per_second?: number; eta_seconds?: number };
type Comparison = { metrics?: Array<Record<string, string | number | null>> };
type HumanItem = { id: string; prompt: string; model_answer: string; score: number | null; notes: string };

export default function Developer() {
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]); const [info, setInfo] = useState<Info>({});
  const [evaluation, setEvaluation] = useState<Evaluation>({}); const [training, setTraining] = useState<Training>({});
  const [comparison, setComparison] = useState<Comparison>({});
  const [humanItems, setHumanItems] = useState<HumanItem[]>([]); const [humanIndex, setHumanIndex] = useState(0);
  const [selected, setSelected] = useState(""); const [message, setMessage] = useState("");
  async function refresh() {
    const [checkpointData, modelData, evaluationData, trainingData, comparisonData, humanData] = await Promise.all([
      fetch(`${API}/checkpoints`).then(r => r.json()), fetch(`${API}/model-info`).then(r => r.json()),
      fetch(`${API}/evaluation/latest`).then(r => r.json()), fetch(`${API}/training/latest`).then(r => r.json()),
      fetch(`${API}/evaluation/v03-v04`).then(r => r.json()), fetch(`${API}/human-eval/v04`).then(r => r.json())]);
    setCheckpoints(checkpointData.checkpoints ?? []); setInfo(modelData); setEvaluation(evaluationData.result ?? {});
    setTraining(trainingData); setComparison({metrics: [
      {model: "v0.3-5000", ...comparisonData.v03_metrics}, {model: "v0.4-2000", ...comparisonData.v04_metrics}
    ]}); setHumanItems(humanData.items ?? []);
  }
  useEffect(() => { refresh().catch(() => setMessage("UniPilot Mini APIに接続できません")); }, []);
  async function load() {
    setMessage("読み込み中…"); const isV01 = selected.includes("v01") || selected.includes("sanity");
    const response = await fetch(`${API}/model/load`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ checkpoint: selected, tokenizer: isV01 ? "tokenizer/vocab.json" : "tokenizer/vocab-v02-512.json" }) });
    const data = await response.json(); setMessage(response.ok ? "切り替えました" : data.detail ?? "切替に失敗しました"); if (response.ok) setInfo(data);
  }
  async function saveHumanScore(score: number) {
    const item = humanItems[humanIndex]; if (!item) return;
    const response = await fetch(`${API}/human-eval/v04`, {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({item_id: item.id, score, notes: item.notes ?? ""})});
    if (response.ok) { setHumanItems(values => values.map((value, index) => index === humanIndex ? {...value, score} : value)); setHumanIndex(value => Math.min(value + 1, humanItems.length - 1)); }
  }
  const metrics = evaluation.metrics ?? {};
  const rows = [["Model", info.model], ["Checkpoint", info.checkpoint], ["Stage", info.stage], ["Training step", info.step],
    ["Validation loss", evaluation.validation_loss], ["Perplexity", evaluation.perplexity],
    ["Japanese ratio", metrics.japanese_character_ratio == null ? null : `${(metrics.japanese_character_ratio * 100).toFixed(2)}%`],
    ["Repetition", metrics.repetition_rate == null ? null : `${(metrics.repetition_rate * 100).toFixed(2)}%`],
    ["Keyword relevance", metrics.keyword_relevance == null ? null : `${Number(metrics.keyword_relevance).toFixed(2)}%`],
    ["Category accuracy", metrics.category_accuracy == null ? null : `${(metrics.category_accuracy * 100).toFixed(2)}%`],
    ["EOS rate", metrics.eos_reached_rate == null ? null : `${(metrics.eos_reached_rate * 100).toFixed(2)}%`], ["Human score", "未採点"]];
  return <main className="mx-auto max-w-5xl px-5 py-10"><p className="text-xs font-semibold uppercase tracking-[.24em] text-cyan-400">Development only</p>
    <h1 className="mt-2 text-3xl font-bold">Checkpoint比較</h1><div className="mt-7 grid gap-6 lg:grid-cols-2">
      <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5"><h2 className="font-semibold">現在のモデル</h2>
        <dl className="mt-4 space-y-3">{rows.map(([label, value]) => <div key={String(label)} className="grid grid-cols-[150px_1fr] gap-3 text-sm"><dt className="text-slate-400">{String(label)}</dt><dd className="break-all">{value == null ? "—" : String(value)}</dd></div>)}</dl></section>
      <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5"><h2 className="font-semibold">ローカルcheckpoint切替</h2>
        <select value={selected} onChange={event => setSelected(event.target.value)} className="mt-4 w-full rounded-lg border border-slate-700 bg-slate-950 p-3">
          <option value="">選択してください</option>{checkpoints.map(item => <option key={item.path}>{item.path}</option>)}</select>
        <button onClick={load} disabled={!selected} className="mt-3 rounded-lg bg-cyan-500 px-5 py-2 font-semibold text-slate-950 disabled:opacity-40">読み込む</button>
        <p className="mt-3 text-sm text-slate-400">{message}</p><p className="mt-5 text-xs text-amber-300">切替はUNIPILOT_DEV_MODE=1のローカル開発時のみ有効です。</p></section></div>
    <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/60 p-5"><h2 className="font-semibold">Training monitor</h2>
      <p className="mt-3 text-sm text-slate-300">Step {training.step ?? "—"} · Stage {training.stage ?? "—"} · Loss {training.train_loss?.toFixed(4) ?? "—"} · Validation {training.stage_validation_loss?.toFixed(4) ?? "—"} · {training.tokens_per_second?.toFixed(1) ?? "—"} tokens/sec · ETA {training.eta_seconds?.toFixed(0) ?? "—"} sec</p></section>
    <section className="mt-6 overflow-x-auto rounded-2xl border border-slate-800 bg-slate-900/60 p-5"><h2 className="font-semibold">v0.3 / v0.4 comparison</h2>
      <table className="mt-4 w-full text-left text-sm"><thead className="text-slate-400"><tr><th>Model</th><th>Keyword</th><th>Category</th><th>Meaningful</th><th>EOS</th></tr></thead><tbody>{(comparison.metrics ?? []).map((row, index) => <tr key={index} className="border-t border-slate-800"><td className="py-2">{String(row.model)}</td><td>{Number(row.keyword_relevance).toFixed(1)}%</td><td>{(Number(row.category_accuracy) * 100).toFixed(1)}%</td><td>{(Number(row.meaningful_response_rate) * 100).toFixed(1)}%</td><td>{(Number(row.eos_reached_rate) * 100).toFixed(1)}%</td></tr>)}</tbody></table></section>
    {humanItems[humanIndex] && <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/60 p-5"><h2 className="font-semibold">Human evaluation · PENDING</h2>
      <p className="mt-3 text-sm text-slate-400">{humanIndex + 1} / {humanItems.length}</p><p className="mt-3 font-medium">{humanItems[humanIndex].prompt}</p>
      <p className="mt-3 whitespace-pre-wrap rounded-xl bg-slate-950 p-4 text-sm">{humanItems[humanIndex].model_answer}</p>
      <div className="mt-4 flex flex-wrap gap-2">{[0,1,2,3,4].map(score => <button key={score} onClick={() => saveHumanScore(score)} className="rounded-lg border border-slate-700 px-3 py-2 hover:border-cyan-400">{score} {['意味不明','無関係','一部関連','意味が通る','良い'][score]}</button>)}</div></section>}</main>;
}
