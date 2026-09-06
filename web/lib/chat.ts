import type { ResponseMode, ToolCard } from "../types/academic";
export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
export type Snapshot = { text?: string; cards?: ToolCard[] };
export async function readSnapshots(body: ReadableStream<Uint8Array>, onSnapshot: (snapshot: Snapshot) => void) {
  const reader = body.getReader(); const decoder = new TextDecoder(); let buffer = "";
  const consume = (line: string) => { if (!line.trim()) return; const value = JSON.parse(line) as Snapshot;
    if (!value || typeof value !== "object" || (value.text !== undefined && typeof value.text !== "string")) throw new Error("応答形式を確認できません");
    onSnapshot({text: value.text, cards: Array.isArray(value.cards) ? value.cards : undefined}); };
  try { while (true) { const {done, value} = await reader.read(); buffer += decoder.decode(value, {stream: !done});
    const lines = buffer.split("\n"); buffer = lines.pop() ?? ""; lines.forEach(consume);
    if (done) { consume(buffer); break; } }
  } finally { reader.releaseLock(); }
}
export async function sendChat(prompt: string, responseMode: ResponseMode, sessionId: string, onSnapshot: (s: Snapshot) => void, signal: AbortSignal) {
  const body = JSON.stringify({prompt, max_new_tokens: 32, temperature: .7, top_k: 40, top_p: .9, repetition_penalty: 1.0,
    response_mode: responseMode, session_id: sessionId || undefined});
  const options = {method: "POST", headers: {"Content-Type": "application/json"}, body, signal};
  let stream: Response | null = null;
  try { stream = await fetch(`${API}/chat/stream`, options); } catch (reason) { if (signal.aborted) throw reason; }
  if (stream?.ok && stream.body) { await readSnapshots(stream.body, onSnapshot); return "stream"; }
  const response = await fetch(`${API}/chat`, options);
  if (!response.ok) throw new Error(`UniPilot API (${response.status}) に接続できません。時間をおいて再試行してください。`);
  const data = await response.json(); onSnapshot({text: typeof data.text === "string" ? data.text : "応答が空でした。", cards: Array.isArray(data.cards) ? data.cards : []});
  return "fallback";
}
