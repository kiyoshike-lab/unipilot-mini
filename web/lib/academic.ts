import type { SourceItem, StudyOptions, ToolCard } from "../types/academic";
export const SUBJECTS = ["線形代数", "微積分", "確率・統計", "物理", "化学", "情報", "プログラミング", "経済", "経営", "法律", "歴史", "心理", "社会", "英語"];
export const LEVELS = ["やさしく", "標準", "大学レベル"];
export const METHODS = ["考え方から", "途中式つき", "ヒントから", "解答を確認", "類題を作る"];
export function buildStudyPrompt(question: string, options: StudyOptions): string {
  if (!SUBJECTS.includes(options.subject) || !LEVELS.includes(options.level) || !METHODS.includes(options.method)) throw new Error("学習設定が無効です");
  return `大学科目の学習支援です。科目・説明レベル・解説方式を参考に、元の質問に答えてください。不確かな内容は断定せず、出典を作らないでください。\n学習設定: ${JSON.stringify(options)}\n元の質問 (JSON文字列): ${JSON.stringify(question)}`;
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
