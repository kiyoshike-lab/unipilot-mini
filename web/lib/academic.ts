import type { SourceItem, StudyOptions, ToolCard } from "../types/academic";
export const SUBJECTS = ["線形代数", "微積分", "確率・統計", "物理", "化学", "情報", "プログラミング", "経済", "経営", "法律", "歴史", "心理", "社会", "英語"];
export const LEVELS = ["やさしく", "標準", "大学レベル"];
export const METHODS = ["考え方から", "途中式つき", "ヒントから", "解答を確認", "類題を作る"];
export const METHOD_NAMES = ["Teach me", "Step by step", "Hint only", "Check my answer", "Practice problem"];
export const DIFFICULTIES = ["基礎", "標準", "発展"];
export function buildStudyPrompt(question: string, options: StudyOptions): string {
  if (!SUBJECTS.includes(options.subject) || !LEVELS.includes(options.level) || !METHODS.includes(options.method)) throw new Error("学習設定が無効です");
  if (options.difficulty && !DIFFICULTIES.includes(options.difficulty)) throw new Error("難易度が無効です");
  const instructions = ["概念を具体例で説明。", "前提から途中式を一段階ずつ説明。", "最終解答は提示せず、まず小さなヒントを一つ。学生の応答を待って段階的に支援。", "正しい部分、最初の誤り、直し方を順に説明。", "選択科目の類題を一問。解答は先に示さない。"];
  if (options.method === "解答を確認" && !options.studentAnswer?.trim()) throw new Error("自分の回答を入力してください");
  return `大学科目の学習支援。不確かな内容は断定せず、出典を作らない。${instructions[METHODS.indexOf(options.method)]}\n学習設定: ${JSON.stringify(options)}\n元の質問 (JSON文字列): ${JSON.stringify(question)}`;
}
export function safeSourceUrl(value?: string): string | undefined {
  if (!value) return undefined;
  try { const url = new URL(value); return url.protocol === "https:" && !url.username && !url.password ? url.href : undefined; } catch { return undefined; }
}
export function sourceStatus(source: SourceItem): string {
  if (source.stale === true) return "Stale · 要再確認";
  // A timestamp/confidence alone cannot establish verification of a claim.
  if (source.verified === true && safeSourceUrl(source.url) && source.last_verified_at) return "Verified · API確認済み";
  return source.last_verified_at ? "Needs verification · 原文確認が必要" : "確認状態：不明";
}
export function cardSources(cards: ToolCard[] = []): SourceItem[] {
  return cards.filter(c => c.kind === "sources" && Array.isArray(c.data?.sources)).flatMap(c => (c.data!.sources as unknown[]).filter((v): v is SourceItem => !!v && typeof v === "object" && typeof (v as SourceItem).title === "string"));
}
