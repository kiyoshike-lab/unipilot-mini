"use client";
import { useEffect, useState } from "react";
const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
type Info = Record<string, string | number | boolean | null>;

export default function Settings() {
  const [info, setInfo] = useState<Info>({});
  useEffect(() => { fetch(`${API}/model-info`).then(value => value.json()).then(setInfo).catch(() => setInfo({ loaded: false })); }, []);
  const rows = [["パラメータ数", info.parameters], ["Checkpoint", info.checkpoint], ["Tokenizer", info.tokenizer],
    ["Vocabulary", info.vocab_size], ["Context length", info.context_length], ["学習 step", info.step], ["最終 validation loss", info.validation_loss], ["Device", info.device]];
  return <main className="mx-auto max-w-4xl px-5 py-10"><p className="text-xs font-semibold uppercase tracking-[.24em] text-cyan-400">Local model</p>
    <h1 className="mt-2 text-3xl font-bold">モデル管理</h1><div className="mt-8 overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/60">
      {rows.map(([label, value]) => <div key={String(label)} className="grid grid-cols-[180px_1fr] border-b border-slate-800 px-5 py-4 last:border-0">
        <span className="text-slate-400">{String(label)}</span><span className="break-all">{value == null ? "—" : String(value)}</span></div>)}</div>
    <div className="mt-5 rounded-xl border border-emerald-800 bg-emerald-950/40 p-4 text-emerald-300">● Model: Local<br />External AI API: OFF</div></main>;
}
