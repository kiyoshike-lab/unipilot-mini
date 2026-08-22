"use client";
import { FormEvent, useState } from "react";

type Message = { role: "user" | "assistant"; text: string };
const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([{ role: "assistant", text: "大学生活について、何でも聞いてください。" }]);
  const [input, setInput] = useState(""); const [busy, setBusy] = useState(false); const [error, setError] = useState("");
  async function submit(event: FormEvent) {
    event.preventDefault(); if (!input.trim() || busy) return;
    const question = input.trim(); setMessages(value => [...value, { role: "user", text: question }]); setInput(""); setBusy(true); setError("");
    try {
      const requestBody = JSON.stringify({ prompt: question, max_new_tokens: 32, temperature: .7, top_k: 40, top_p: .9, repetition_penalty: 1.0 });
      const stream = await fetch(`${API}/chat/stream`, { method: "POST", headers: { "Content-Type": "application/json" }, body: requestBody });
      if (!stream.ok || !stream.body) {
        const response = await fetch(`${API}/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: requestBody });
        if (!response.ok) throw new Error(`Local API error: ${response.status}`);
        const data = await response.json(); setMessages(value => [...value, { role: "assistant", text: data.text || "応答が空でした。" }]);
      } else {
        setMessages(value => [...value, { role: "assistant", text: "" }]);
        const reader = stream.body.getReader(); const decoder = new TextDecoder(); let buffer = ""; let finalText = "";
        while (true) {
          const { done, value } = await reader.read(); buffer += decoder.decode(value, { stream: !done });
          const lines = buffer.split("\n"); buffer = lines.pop() ?? "";
          for (const line of lines) if (line.trim()) {
            const snapshot = JSON.parse(line); finalText = snapshot.text ?? finalText;
            setMessages(items => items.map((item, index) => index === items.length - 1 ? { ...item, text: finalText } : item));
          }
          if (done) break;
        }
        if (!finalText) setMessages(items => items.map((item, index) => index === items.length - 1 ? { ...item, text: "応答が空でした。" } : item));
      }
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Local APIに接続できません"); }
    finally { setBusy(false); }
  }
  return <main className="mx-auto flex min-h-[calc(100vh-65px)] max-w-5xl flex-col px-4 py-8">
    <header className="mb-7 flex items-end justify-between"><div><p className="mb-1 text-xs font-semibold uppercase tracking-[.24em] text-cyan-400">University Assistant</p>
      <h1 className="text-3xl font-bold">大学生活専用AI</h1></div><div className="rounded-full border border-emerald-700/60 bg-emerald-950/50 px-3 py-1 text-sm text-emerald-300"><span className="mr-2">●</span>Local</div></header>
    <section className="flex-1 space-y-4 rounded-2xl border border-slate-800 bg-slate-900/50 p-5 shadow-2xl shadow-cyan-950/20">
      {messages.map((message, index) => <div key={index} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
        <div className={`max-w-[82%] whitespace-pre-wrap rounded-2xl px-4 py-3 ${message.role === "user" ? "bg-cyan-600 text-white" : "bg-slate-800 text-slate-100"}`}>{message.text}</div></div>)}
      {busy && <p className="text-sm text-slate-400">UniPilot Miniがローカルで生成中…</p>}{error && <p className="text-sm text-rose-400">{error}</p>}
    </section>
    <form onSubmit={submit} className="mt-5 flex gap-3"><input value={input} onChange={event => setInput(event.target.value)} placeholder="質問を入力..."
      className="min-w-0 flex-1 rounded-xl border border-slate-700 bg-slate-900 px-4 py-3 outline-none focus:border-cyan-500" />
      <button disabled={busy} className="rounded-xl bg-cyan-500 px-6 font-semibold text-slate-950 disabled:opacity-50">送信</button></form>
    <p className="mt-3 text-center text-xs text-slate-500">External AI API: OFF · 推論データはこのPC内で処理されます</p>
  </main>;
}
