"use client";
import { useEffect, useState } from "react";
const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
type Checkpoint = { path: string; size_bytes: number };
type Info = Record<string, string | number | boolean | null>;

export default function Developer() {
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]); const [info, setInfo] = useState<Info>({});
  const [selected, setSelected] = useState(""); const [message, setMessage] = useState("");
  async function refresh() {
    const [checkpointData, modelData] = await Promise.all([fetch(`${API}/checkpoints`).then(r => r.json()), fetch(`${API}/model-info`).then(r => r.json())]);
    setCheckpoints(checkpointData.checkpoints ?? []); setInfo(modelData);
  }
  useEffect(() => { refresh().catch(() => setMessage("Local APIに接続できません")); }, []);
  async function load() {
    setMessage("読み込み中…"); const isV02 = selected.includes("v02");
    const response = await fetch(`${API}/model/load`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ checkpoint: selected, tokenizer: isV02 ? "tokenizer/vocab-v02-512.json" : "tokenizer/vocab.json" }) });
    const data = await response.json(); setMessage(response.ok ? "切り替えました" : data.detail ?? "切替に失敗しました"); if (response.ok) setInfo(data);
  }
  const rows = [["Model version", info.model], ["Checkpoint", info.checkpoint], ["Parameters", info.parameters], ["Validation loss", info.validation_loss],
    ["Perplexity", info.validation_loss ? Math.exp(Number(info.validation_loss)).toFixed(2) : null], ["Training step", info.step],
    ["Generation settings", "temperature 0.7 / top-k 40 / top-p 0.9 / penalty 1.1"]];
  return <main className="mx-auto max-w-5xl px-5 py-10"><p className="text-xs font-semibold uppercase tracking-[.24em] text-cyan-400">Development only</p>
    <h1 className="mt-2 text-3xl font-bold">Checkpoint比較</h1><div className="mt-7 grid gap-6 lg:grid-cols-2">
      <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5"><h2 className="font-semibold">現在のモデル</h2>
        <dl className="mt-4 space-y-3">{rows.map(([label, value]) => <div key={String(label)} className="grid grid-cols-[150px_1fr] gap-3 text-sm"><dt className="text-slate-400">{String(label)}</dt><dd className="break-all">{value == null ? "—" : String(value)}</dd></div>)}</dl></section>
      <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5"><h2 className="font-semibold">ローカルcheckpoint切替</h2>
        <select value={selected} onChange={event => setSelected(event.target.value)} className="mt-4 w-full rounded-lg border border-slate-700 bg-slate-950 p-3">
          <option value="">選択してください</option>{checkpoints.map(item => <option key={item.path}>{item.path}</option>)}</select>
        <button onClick={load} disabled={!selected} className="mt-3 rounded-lg bg-cyan-500 px-5 py-2 font-semibold text-slate-950 disabled:opacity-40">読み込む</button>
        <p className="mt-3 text-sm text-slate-400">{message}</p><p className="mt-5 text-xs text-amber-300">切替はUNIPILOT_DEV_MODE=1のローカル開発時のみ有効です。</p></section></div></main>;
}
