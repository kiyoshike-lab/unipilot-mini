"use client";

import { useEffect, useMemo, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const POSITION_KEY = "unipilot-campus-v21-eval-position";
const BLIND_KEY = "unipilot-campus-v21-eval-blind";
const REVEALED_KEY = "unipilot-campus-v21-eval-revealed";
const BLIND_RATINGS_KEY = "unipilot-campus-v21-eval-blind-ratings";

type Axis = "correctness" | "relevance" | "actionable" | "naturalness" | "would_use_again";
type PairAxis = "correctness" | "specificity" | "actionability" | "readability" | "would_use";
type PairChoice = "unipilot" | "competitor" | "tie" | "unscored";
type Competitor = "chatgpt" | "gemini";
type BlindLabel = "A" | "B" | "C";
type AxisScores = Record<Axis, number | null>;
type IssueKey = "critical_error" | "factual_error" | "unanswered" | "university_policy_assertion" |
  "unnecessary_information" | "router_error" | "retrieval_error" | "tool_error" | "model_error" |
  "too_long" | "too_short" | "other_error" | "unusable_answer" | "faq_error";
type Pairwise = Record<Competitor, Record<PairAxis, PairChoice>>;
type IssueFlags = Record<IssueKey, boolean>;
type Card = {kind?: string; title?: string; data?: Record<string, unknown>};
type Item = {
  id: string;
  question: string;
  category: string;
  difficulty?: string;
  evaluation_bucket?: string;
  campus_answer: string;
  campus_metadata?: {
    action?: string;
    route?: string;
    tool?: string | null;
    latency_ms?: number;
    source?: unknown;
    sources?: unknown;
    cards?: Card[];
  };
  scores: Record<Axis, number | null>;
  issue_flags?: Partial<IssueFlags>;
  issues_reviewed?: boolean;
  pairwise?: Pairwise;
  competitor_scores?: {chatgpt: number | null; gemini: number | null};
  chatgpt_answer?: string;
  gemini_answer?: string;
  other_issue?: string;
  notes?: string;
};
type PairCounts = {win: number; tie: number; loss: number; unscored: number};
type Summary = {
  status: "PENDING" | "COMPLETE";
  completed: number;
  pending: number;
  total: number;
  averages_0_to_5: Record<Axis, number | null>;
  issue_counts: Record<"critical_error" | "university_policy_assertion" | "router_error" | "retrieval_error" | "tool_error" | "model_error", number>;
  pairwise: Record<Competitor, PairCounts>;
  human_gate: {
    status: "PENDING" | "PASS" | "FAIL";
    critical_error_rate: number | null;
    university_policy_assertion_rate: number | null;
  };
  automated_comparison: null | {
    automated_correctness_percent: number;
    human_correctness_percent: number;
    gap_percentage_points: number;
    analysis: string;
  };
  error_categories: Array<{category: string; count: number; v2_2_recommendation: string}>;
  v2_2_priorities: Array<{category: string; count: number; v2_2_recommendation: string}>;
};

const AXES: Array<[Axis, string]> = [
  ["correctness", "Correctness"], ["relevance", "Relevance"], ["actionable", "Actionable"],
  ["naturalness", "Naturalness"], ["would_use_again", "Would use again"],
];
const PAIR_AXES: Array<[PairAxis, string]> = [
  ["correctness", "正確性"], ["specificity", "具体性"], ["actionability", "行動しやすさ"],
  ["readability", "読みやすさ"], ["would_use", "実際に使いたい"],
];
const ISSUES: Array<[IssueKey, string]> = [
  ["critical_error", "重大誤回答"], ["factual_error", "事実誤り"], ["unanswered", "質問に答えていない"],
  ["unnecessary_information", "不要な情報追加"], ["university_policy_assertion", "大学固有制度を誤断定"],
  ["router_error", "Router誤り"], ["retrieval_error", "Retrieval誤り"], ["tool_error", "Tool誤り"],
  ["model_error", "Model回答品質"], ["too_long", "長すぎる"], ["too_short", "短すぎる"],
  ["other_error", "その他"],
];
const ALL_ISSUES: IssueKey[] = [...ISSUES.map(([key]) => key), "unusable_answer", "faq_error"];

function emptyIssues(): IssueFlags {
  return Object.fromEntries(ALL_ISSUES.map(key => [key, false])) as IssueFlags;
}

function emptyPairwise(): Pairwise {
  const axes = {correctness: "unscored", specificity: "unscored", actionability: "unscored",
    readability: "unscored", would_use: "unscored"} as const;
  return {chatgpt: {...axes}, gemini: {...axes}};
}

function isComplete(item: Item): boolean {
  return Boolean(item.issues_reviewed) && AXES.every(([axis]) => item.scores?.[axis] != null);
}

function emptyAxisScores(): AxisScores {
  return {correctness: null, relevance: null, actionable: null, naturalness: null, would_use_again: null};
}

export default function CampusV21Evaluation() {
  const [items, setItems] = useState<Item[]>([]);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [index, setIndex] = useState(0);
  const [message, setMessage] = useState("読み込み中…");
  const [saving, setSaving] = useState(false);
  const [blind, setBlind] = useState(false);
  const [revealed, setRevealed] = useState<Record<string, boolean>>({});
  const [blindRatings, setBlindRatings] = useState<Record<string, Record<BlindLabel, AxisScores>>>({});
  const [dirtyIds, setDirtyIds] = useState<Record<string, boolean>>({});

  useEffect(() => {
    fetch(`${API}/human-eval/campus-v21`).then(response => {
      if (!response.ok) throw new Error("評価データを読み込めません");
      return response.json();
    }).then(data => {
      const loaded = (data.items ?? []) as Item[];
      setItems(loaded);
      setSummary(data as Summary);
      const storedId = localStorage.getItem(POSITION_KEY);
      const storedIndex = storedId ? loaded.findIndex(item => item.id === storedId) : -1;
      const firstPending = loaded.findIndex(item => !isComplete(item));
      setIndex(storedIndex >= 0 ? storedIndex : firstPending >= 0 ? firstPending : 0);
      setBlind(localStorage.getItem(BLIND_KEY) === "true");
      try { setRevealed(JSON.parse(localStorage.getItem(REVEALED_KEY) ?? "{}")); } catch { setRevealed({}); }
      try { setBlindRatings(JSON.parse(localStorage.getItem(BLIND_RATINGS_KEY) ?? "{}")); } catch { setBlindRatings({}); }
      setMessage(loaded.length === 100 ? "評価データを読み込みました" : `評価項目数が100ではありません（${loaded.length}件）`);
    }).catch(error => setMessage(error instanceof Error ? error.message : "評価データを読み込めません"));
  }, []);

  const item = items[index];
  const answerOrder = useMemo(() => blindOrder(item?.id ?? ""), [item?.id]);
  const isRevealed = item ? Boolean(revealed[item.id]) : false;

  useEffect(() => {
    if (item) localStorage.setItem(POSITION_KEY, item.id);
  }, [item]);

  function update(patch: Partial<Item>) {
    if (!item) return;
    setItems(values => values.map((value, position) => position === index ? {...value, ...patch} : value));
    setDirtyIds(values => ({...values, [item.id]: true}));
  }

  function setBlindMode(value: boolean) {
    setBlind(value);
    localStorage.setItem(BLIND_KEY, String(value));
  }

  function revealCurrent() {
    if (!item) return;
    const ratings = blindRatings[item.id];
    const complete = ratings && (["A", "B", "C"] as const).every(label => AXES.every(([axis]) => ratings[label]?.[axis] != null));
    if (!complete || !item.chatgpt_answer || !item.gemini_answer) {
      setMessage("3回答を用意し、A/B/Cすべての5軸を採点してから開示してください");
      return;
    }
    const unipilotLabel = String.fromCharCode(65 + answerOrder.indexOf("unipilot")) as BlindLabel;
    update({scores: {...ratings[unipilotLabel]}});
    const next = {...revealed, [item.id]: true};
    setRevealed(next);
    localStorage.setItem(REVEALED_KEY, JSON.stringify(next));
    setMessage("回答元を開示しました。Pairwiseを採点できます");
  }

  function updateBlindRating(label: BlindLabel, axis: Axis, value: number | null) {
    if (!item) return;
    const current = blindRatings[item.id] ?? {A: emptyAxisScores(), B: emptyAxisScores(), C: emptyAxisScores()};
    const next = {...blindRatings, [item.id]: {...current, [label]: {...current[label], [axis]: value}}};
    setBlindRatings(next);
    localStorage.setItem(BLIND_RATINGS_KEY, JSON.stringify(next));
    setDirtyIds(values => ({...values, [item.id]: true}));
  }

  function move(target: number) {
    setIndex(Math.max(0, Math.min(items.length - 1, target)));
    setMessage("");
  }

  function moveToPending() {
    const pending = items.findIndex((candidate, position) => position > index && !isComplete(candidate));
    const wrapped = items.findIndex(candidate => !isComplete(candidate));
    move(pending >= 0 ? pending : wrapped >= 0 ? wrapped : index);
  }

  async function save(advance: boolean) {
    if (!item) return;
    if (AXES.some(([axis]) => item.scores[axis] == null)) {
      setMessage("5軸をすべて0〜5で採点してください");
      return;
    }
    if (!item.issues_reviewed) {
      setMessage("問題チェックを確認済みにしてください");
      return;
    }
    setSaving(true);
    setMessage("保存中…");
    const response = await fetch(`${API}/human-eval/campus-v21`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        item_id: item.id,
        ...item.scores,
        chatgpt_answer: item.chatgpt_answer ?? "",
        gemini_answer: item.gemini_answer ?? "",
        chatgpt_score: item.competitor_scores?.chatgpt ?? null,
        gemini_score: item.competitor_scores?.gemini ?? null,
        issue_flags: {...emptyIssues(), ...item.issue_flags},
        issues_reviewed: true,
        pairwise: item.pairwise ?? emptyPairwise(),
        other_issue: item.other_issue ?? "",
        notes: item.notes ?? "",
      }),
    }).catch(() => null);
    if (!response?.ok) {
      setSaving(false);
      setMessage("保存に失敗しました。入力内容はこの画面に残っています");
      return;
    }
    const data = await response.json();
    setSummary(data.summary as Summary);
    setDirtyIds(values => ({...values, [item.id]: false}));
    setSaving(false);
    setMessage("途中結果とエクスポートを保存しました");
    if (advance) {
      const nextPending = items.findIndex((candidate, position) => position > index && candidate.id !== item.id && !isComplete(candidate));
      move(nextPending >= 0 ? nextPending : Math.min(index + 1, items.length - 1));
    }
  }

  async function exportNow() {
    setMessage("エクスポート中…");
    const response = await fetch(`${API}/human-eval/campus-v21/export`, {method: "POST"}).catch(() => null);
    if (!response?.ok) { setMessage("エクスポートに失敗しました"); return; }
    const data = await response.json();
    setSummary(data.summary as Summary);
    setMessage(`${data.results_path} と ${data.report_path} を保存しました`);
  }

  if (!item || !summary) {
    return <main className="mx-auto max-w-7xl px-5 py-10"><h1 className="text-3xl font-bold">Campus v2.1 Human Evaluation</h1>
      <p className="mt-4 text-slate-400">{message}</p></main>;
  }

  const flags = {...emptyIssues(), ...item.issue_flags};
  const pairwise = item.pairwise ?? emptyPairwise();
  const answers = {unipilot: item.campus_answer, chatgpt: item.chatgpt_answer ?? "", gemini: item.gemini_answer ?? ""};
  const sources = sourceLabels(item);
  const currentBlindRatings = blindRatings[item.id] ?? {A: emptyAxisScores(), B: emptyAxisScores(), C: emptyAxisScores()};
  const canReveal = Boolean(item.chatgpt_answer && item.gemini_answer) && (["A", "B", "C"] as const)
    .every(label => AXES.every(([axis]) => currentBlindRatings[label][axis] != null));

  return <main className="mx-auto max-w-7xl px-5 py-10">
    <p className="text-xs font-semibold uppercase tracking-[.24em] text-cyan-400">Campus v2.1 RC · Human Gate · External AI API OFF</p>
    <div className="mt-2 flex flex-wrap items-end justify-between gap-4">
      <div><h1 className="text-3xl font-bold">100問 Human Evaluation</h1>
        <p className="mt-2 text-sm text-slate-400">回答ロジック固定: 0dc1878… · 本番v0.4は変更しません</p></div>
      <div className="flex flex-wrap gap-2">
        <label className="flex items-center gap-2 rounded-lg border border-slate-700 px-3 py-2 text-sm"><input type="checkbox" checked={blind} onChange={event => setBlindMode(event.target.checked)} /> A/B/C Blind</label>
        <button onClick={exportNow} className="rounded-lg border border-slate-600 px-3 py-2 text-sm">途中結果をエクスポート</button>
      </div>
    </div>

    <Dashboard summary={summary} />

    <div className="mt-6 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-700 bg-slate-900/80 px-4 py-3">
      <div><span className="font-semibold">現在 {index + 1} / {items.length}</span><span className="ml-3 text-sm text-amber-300">未評価 {summary.pending}件</span>
        {dirtyIds[item.id] && <span className="ml-3 text-sm text-rose-300">未保存</span>}</div>
      <button onClick={moveToPending} className="rounded border border-cyan-700 px-3 py-1.5 text-sm text-cyan-200">次の未評価へ</button>
    </div>

    <section className="mt-5 rounded-2xl border border-slate-800 bg-slate-900/60 p-5">
      <div className="flex flex-wrap items-start justify-between gap-4"><div><h2 className="font-semibold">質問</h2><p className="mt-3 text-lg">{item.question}</p></div>
        <button onClick={() => navigator.clipboard.writeText(item.question)} className="rounded border border-slate-600 px-3 py-1 text-xs">質問をコピー</button></div>
      <div className="mt-4 grid gap-2 text-sm text-slate-300 sm:grid-cols-2 lg:grid-cols-5">
        <Meta label="route" value={item.campus_metadata?.route ?? "—"} />
        <Meta label="category" value={item.category ?? "—"} />
        <Meta label="type" value={item.campus_metadata?.action ?? "—"} />
        <Meta label="tool" value={item.campus_metadata?.tool ?? "—"} />
        <Meta label="response time" value={formatLatency(item.campus_metadata?.latency_ms)} />
      </div>
      {sources.length > 0 && <div className="mt-3 text-sm text-slate-300"><span className="text-slate-500">source</span><span className="ml-2">{sources.join(" / ")}</span></div>}
    </section>

    {blind ? <section className="mt-5">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3"><p className="text-sm text-slate-400">比較回答の貼付はBlindをOFFにして行い、再度ONにして採点してください。</p>
        {!isRevealed && <button disabled={!canReveal} onClick={revealCurrent} className="rounded-lg border border-amber-600 px-3 py-2 text-sm text-amber-200 disabled:opacity-40">5軸採点後に回答元を開示</button>}</div>
      <div className="grid gap-4 lg:grid-cols-3">{answerOrder.map((origin, position) => {
        const label = String.fromCharCode(65 + position) as BlindLabel;
        return <div key={origin}><AnswerPanel title={`回答 ${label}${isRevealed ? ` · ${originLabel(origin)}` : ""}`} value={answers[origin]} />
          {!isRevealed && <BlindScoreEditor label={label} values={currentBlindRatings[label]} onChange={(axis, value) => updateBlindRating(label, axis, value)} />}</div>;
      })}</div></section> :
      <div className="mt-5 grid gap-4 lg:grid-cols-3"><AnswerPanel title="UniPilot Campus v2.1" value={item.campus_answer} />
        <AnswerPanel title="ChatGPT（手動貼付）" value={item.chatgpt_answer ?? ""} editable onText={value => update({chatgpt_answer: value})} />
        <AnswerPanel title="Gemini（手動貼付）" value={item.gemini_answer ?? ""} editable onText={value => update({gemini_answer: value})} /></div>}

    {(!blind || isRevealed) && <section className="mt-5 rounded-2xl border border-slate-800 bg-slate-900/60 p-5"><h2 className="font-semibold">UniPilot 5軸評価（各0〜5）{blind ? " · Blind採点から反映" : ""}</h2>
      <div className="mt-4 grid gap-3 md:grid-cols-5">{AXES.map(([axis, label]) => <label key={axis} className="flex flex-col gap-2 text-sm"><span>{label}</span>
        <select value={item.scores[axis] ?? ""} onChange={event => update({scores: {...item.scores, [axis]: event.target.value === "" ? null : Number(event.target.value)}})} className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2">
          <option value="">未採点</option>{[0, 1, 2, 3, 4, 5].map(value => <option key={value} value={value}>{value}</option>)}</select></label>)}</div>
    </section>}

    <section className="mt-5 rounded-2xl border border-slate-800 bg-slate-900/60 p-5"><h2 className="font-semibold">問題チェック（複数選択可）</h2>
      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{ISSUES.map(([key, label]) => <label key={key} className="flex items-center gap-2 text-sm"><input type="checkbox" checked={flags[key]}
        onChange={event => update({issue_flags: {...flags, [key]: event.target.checked}})} />{label}</label>)}</div>
      {flags.other_error && <input value={item.other_issue ?? ""} onChange={event => update({other_issue: event.target.value})} placeholder="その他の内容" className="mt-4 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm" />}
      <label className="mt-5 flex items-center gap-2 border-t border-slate-800 pt-4 text-sm"><input type="checkbox" checked={item.issues_reviewed ?? false}
        onChange={event => update({issues_reviewed: event.target.checked})} />問題チェックを確認済み（問題なしの場合もチェック）</label>
    </section>

    <section className="mt-5 rounded-2xl border border-slate-800 bg-slate-900/60 p-5"><h2 className="font-semibold">Pairwise: UniPilot基準の Win / Tie / Loss</h2>
      {blind && !isRevealed ? <p className="mt-3 text-sm text-amber-300">5軸採点後に回答元を開示するとPairwiseを入力できます。</p> :
        <div className="mt-4 grid gap-6 lg:grid-cols-2">{(["chatgpt", "gemini"] as const).map(competitor => <PairwiseEditor key={competitor} competitor={competitor}
          values={pairwise[competitor]} onChange={(axis, value) => update({pairwise: {...pairwise, [competitor]: {...pairwise[competitor], [axis]: value}}})} />)}</div>}
    </section>

    <section className="mt-5 rounded-2xl border border-slate-800 bg-slate-900/60 p-5"><h2 className="font-semibold">コメント</h2>
      <textarea value={item.notes ?? ""} onChange={event => update({notes: event.target.value})} placeholder="この質問への任意メモ" className="mt-3 min-h-28 w-full rounded-lg border border-slate-700 bg-slate-950 p-3" /></section>

    <div className="sticky bottom-3 mt-5 flex flex-wrap items-center gap-3 rounded-xl border border-slate-700 bg-slate-950/95 p-3 shadow-xl">
      <button disabled={index === 0} onClick={() => move(index - 1)} className="rounded-lg border border-slate-600 px-5 py-2 disabled:opacity-40">前へ</button>
      <button disabled={saving} onClick={() => save(false)} className="rounded-lg border border-cyan-500 px-5 py-2 font-semibold text-cyan-200 disabled:opacity-40">途中保存</button>
      <button disabled={saving} onClick={() => save(true)} className="rounded-lg bg-cyan-500 px-5 py-2 font-semibold text-slate-950 disabled:opacity-40">保存して次へ</button>
      <button disabled={index === items.length - 1} onClick={() => move(index + 1)} className="rounded-lg border border-slate-600 px-5 py-2 disabled:opacity-40">次へ</button>
      <span className="min-w-56 flex-1 text-sm text-slate-400">{message}</span>
    </div>
  </main>;
}

function Dashboard({summary}: {summary: Summary}) {
  const counts = summary.issue_counts;
  return <section className="mt-6 rounded-2xl border border-cyan-900/70 bg-cyan-950/20 p-5">
    <div className="flex flex-wrap items-center justify-between gap-3"><h2 className="font-semibold">進捗ダッシュボード</h2>
      <span className={`rounded-full px-3 py-1 text-sm font-semibold ${summary.human_gate.status === "PASS" ? "bg-emerald-900 text-emerald-200" : summary.human_gate.status === "FAIL" ? "bg-rose-900 text-rose-200" : "bg-amber-900 text-amber-200"}`}>Human Gate: {summary.human_gate.status}</span></div>
    <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-5"><Metric label="完了" value={`${summary.completed} / ${summary.total}`} /><Metric label="未評価" value={summary.pending} />
      {AXES.map(([axis, label]) => <Metric key={axis} label={label} value={summary.averages_0_to_5[axis]?.toFixed(2) ?? "—"} />)}</div>
    <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-6"><Metric label="重大誤回答" value={counts.critical_error} /><Metric label="Router error" value={counts.router_error} />
      <Metric label="Retrieval error" value={counts.retrieval_error} /><Metric label="Tool error" value={counts.tool_error} /><Metric label="Model error" value={counts.model_error} />
      <Metric label="大学制度誤断定" value={counts.university_policy_assertion} /></div>
    <div className="mt-3 grid gap-3 md:grid-cols-2"><PairSummary label="ChatGPT" counts={summary.pairwise.chatgpt} /><PairSummary label="Gemini" counts={summary.pairwise.gemini} /></div>
    <div className="mt-4 overflow-x-auto"><table className="w-full text-left text-sm"><thead className="text-slate-500"><tr><th className="py-2">分類</th><th>件数</th></tr></thead><tbody>
      {summary.error_categories.map(entry => <tr key={entry.category} className="border-t border-slate-800"><td className="py-2">{entry.category}</td><td>{entry.count}</td></tr>)}</tbody></table></div>
    {summary.automated_comparison && <div className="mt-4 rounded-lg border border-slate-700 p-3 text-sm"><p>Automated Correctness {summary.automated_comparison.automated_correctness_percent.toFixed(2)}% vs Human Correctness {summary.automated_comparison.human_correctness_percent.toFixed(2)}%</p>
      <p className="mt-1 text-slate-400">差 {summary.automated_comparison.gap_percentage_points.toFixed(2)}pt · {summary.automated_comparison.analysis}</p></div>}
    {summary.v2_2_priorities.length > 0 && <div className="mt-4"><h3 className="text-sm font-semibold">Campus v2.2 修正優先順位</h3><ol className="mt-2 list-inside list-decimal text-sm text-slate-300">
      {summary.v2_2_priorities.map(entry => <li key={entry.category}>{entry.category} ({entry.count}) — {entry.v2_2_recommendation}</li>)}</ol></div>}
    {summary.status === "PENDING" && <p className="mt-4 text-xs text-amber-300">Human Gate判定、自動評価との差、v2.2修正優先順位は100/100完了まで確定しません。</p>}
  </section>;
}

function PairwiseEditor({competitor, values, onChange}: {competitor: Competitor; values: Record<PairAxis, PairChoice>; onChange: (axis: PairAxis, value: PairChoice) => void}) {
  const label = competitor === "chatgpt" ? "ChatGPT" : "Gemini";
  return <div><h3 className="font-medium">UniPilot vs {label}</h3><div className="mt-3 grid gap-3">{PAIR_AXES.map(([axis, axisLabel]) => <label key={axis} className="grid grid-cols-[1fr_8rem] items-center gap-3 text-sm"><span>{axisLabel}</span>
    <select value={values[axis]} onChange={event => onChange(axis, event.target.value as PairChoice)} className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2">
      <option value="unscored">未比較</option><option value="unipilot">Win</option><option value="tie">Tie</option><option value="competitor">Loss</option></select></label>)}</div></div>;
}

function BlindScoreEditor({label, values, onChange}: {label: BlindLabel; values: AxisScores; onChange: (axis: Axis, value: number | null) => void}) {
  return <section className="mt-2 rounded-xl border border-amber-900/70 bg-amber-950/20 p-3"><h3 className="text-sm font-semibold text-amber-200">回答 {label} の5軸採点</h3>
    <div className="mt-2 grid gap-2">{AXES.map(([axis, axisLabel]) => <label key={axis} className="grid grid-cols-[1fr_5rem] items-center gap-2 text-xs"><span>{axisLabel}</span>
      <select value={values[axis] ?? ""} onChange={event => onChange(axis, event.target.value === "" ? null : Number(event.target.value))} className="rounded border border-slate-700 bg-slate-950 px-2 py-1">
        <option value="">—</option>{[0, 1, 2, 3, 4, 5].map(value => <option key={value} value={value}>{value}</option>)}</select></label>)}</div></section>;
}

function AnswerPanel({title, value, editable = false, onText}: {title: string; value: string; editable?: boolean; onText?: (value: string) => void}) {
  return <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4"><div className="flex items-center justify-between gap-3"><h2 className="font-semibold">{title}</h2>
    <button onClick={() => navigator.clipboard.writeText(value)} className="rounded border border-slate-600 px-2 py-1 text-xs">コピー</button></div>
    {editable ? <textarea value={value} onChange={event => onText?.(event.target.value)} placeholder="外部APIは使わず、同じ質問への回答を手動で貼り付け" className="mt-3 min-h-64 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm" /> :
      <p className="mt-3 min-h-64 whitespace-pre-wrap text-sm">{value || "（回答未入力）"}</p>}</section>;
}

function Meta({label, value}: {label: string; value: string}) {
  return <div className="rounded-lg bg-slate-950/70 px-3 py-2"><span className="block text-xs text-slate-500">{label}</span><span>{value}</span></div>;
}

function Metric({label, value}: {label: string; value: string | number}) {
  return <div className="rounded-lg bg-slate-950/70 px-3 py-2"><span className="block text-xs text-slate-500">{label}</span><span className="text-lg font-semibold">{value}</span></div>;
}

function PairSummary({label, counts}: {label: string; counts: PairCounts}) {
  return <div className="rounded-lg bg-slate-950/70 px-3 py-2 text-sm"><span className="text-slate-500">{label}</span><span className="ml-3 font-semibold">W {counts.win} / T {counts.tie} / L {counts.loss}</span></div>;
}

function formatLatency(value?: number): string {
  return value == null ? "—" : `${value.toFixed(2)} ms`;
}

function sourceLabels(item: Item): string[] {
  const metadata = item.campus_metadata;
  const result: string[] = [];
  const add = (value: unknown) => {
    if (typeof value === "string" && value.trim()) result.push(value);
    if (Array.isArray(value)) value.forEach(add);
  };
  add(metadata?.source);
  add(metadata?.sources);
  for (const card of metadata?.cards ?? []) {
    const data = card.data ?? {};
    add(data.source);
    add(data.source_id);
    add(data.faq_id);
  }
  return [...new Set(result)];
}

function originLabel(origin: "unipilot" | "chatgpt" | "gemini"): string {
  return origin === "unipilot" ? "UniPilot" : origin === "chatgpt" ? "ChatGPT" : "Gemini";
}

function blindOrder(id: string): Array<"unipilot" | "chatgpt" | "gemini"> {
  const orders: Array<Array<"unipilot" | "chatgpt" | "gemini">> = [
    ["unipilot", "chatgpt", "gemini"], ["unipilot", "gemini", "chatgpt"], ["chatgpt", "unipilot", "gemini"],
    ["chatgpt", "gemini", "unipilot"], ["gemini", "unipilot", "chatgpt"], ["gemini", "chatgpt", "unipilot"],
  ];
  return orders[[...id].reduce((sum, value) => sum + value.charCodeAt(0), 0) % orders.length];
}
