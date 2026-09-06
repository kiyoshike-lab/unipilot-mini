export type ToolCard = { kind: string; title: string; summary: string; action_label?: string | null; copy_text?: string | null;
  fields?: Array<{name: string; label: string; example?: string}>; data?: Record<string, unknown> };
export type ClarifyOption = { category: string; label: string; prompt: string };
export type SourceItem = { id: string; title: string; publisher?: string; url?: string; license?: string;
  last_verified_at?: string; confidence?: string; stale?: boolean; verified?: boolean };
export type ResponseMode = "short" | "normal" | "detailed";
export type Message = { id: string; role: "user" | "assistant"; text: string; cards?: ToolCard[]; mode?: string };
export type AcademicMode = "ask" | "study" | "report" | "research";
export type StudyOptions = { subject: string; level: string; method: string };
