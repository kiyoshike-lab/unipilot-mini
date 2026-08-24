from typing import Any, Literal

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=100, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0, le=2)
    top_k: int = Field(default=40, ge=0, le=500)
    top_p: float = Field(default=0.95, gt=0, le=1)
    repetition_penalty: float = Field(default=1.1, ge=1, le=2)


class ChatRequest(GenerateRequest):
    response_mode: Literal["auto", "short", "normal", "detailed"] = "auto"
    session_id: str | None = Field(default=None, min_length=1, max_length=100)
    tool_inputs: dict[str, Any] | None = None


class ModelLoadRequest(BaseModel):
    checkpoint: str = Field(min_length=1, max_length=500)
    tokenizer: str = Field(default="tokenizer/vocab-v02-512.json", min_length=1, max_length=500)


class HumanScoreRequest(BaseModel):
    item_id: str = Field(min_length=1, max_length=100)
    score: int = Field(ge=0, le=4)
    notes: str = Field(default="", max_length=1000)


class CampusHumanScoreRequest(BaseModel):
    item_id: str = Field(min_length=1, max_length=100)
    campus_score: int = Field(ge=0, le=5)
    chatgpt_score: int | None = Field(default=None, ge=0, le=5)
    gemini_score: int | None = Field(default=None, ge=0, le=5)
    correct_winner: Literal["campus", "chatgpt", "gemini", "tie", "unscored"] = "unscored"
    specific_winner: Literal["campus", "chatgpt", "gemini", "tie", "unscored"] = "unscored"
    usable_winner: Literal["campus", "chatgpt", "gemini", "tie", "unscored"] = "unscored"
    fast_winner: Literal["campus", "chatgpt", "gemini", "tie", "unscored"] = "unscored"
    student_preference: Literal["campus", "chatgpt", "gemini", "tie", "unscored"] = "unscored"
    chatgpt_answer: str = Field(default="", max_length=10000)
    gemini_answer: str = Field(default="", max_length=10000)
    notes: str = Field(default="", max_length=2000)


class CampusV2HumanScoreRequest(BaseModel):
    item_id: str = Field(min_length=1, max_length=100)
    correctness: int = Field(ge=0, le=5)
    relevance: int = Field(ge=0, le=5)
    actionable: int = Field(ge=0, le=5)
    naturalness: int = Field(ge=0, le=5)
    would_use_again: int = Field(ge=0, le=5)
    chatgpt_score: int | None = Field(default=None, ge=0, le=5)
    gemini_score: int | None = Field(default=None, ge=0, le=5)
    chatgpt_answer: str = Field(default="", max_length=10000)
    gemini_answer: str = Field(default="", max_length=10000)
    notes: str = Field(default="", max_length=2000)
