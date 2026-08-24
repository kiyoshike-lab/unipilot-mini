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


class CampusV21IssueFlags(BaseModel):
    critical_error: bool = False
    factual_error: bool = False
    unanswered: bool = False
    university_policy_assertion: bool = False
    unnecessary_information: bool = False
    unusable_answer: bool = False
    router_error: bool = False
    tool_error: bool = False
    faq_error: bool = False
    retrieval_error: bool = False
    model_error: bool = False
    too_long: bool = False
    too_short: bool = False
    other_error: bool = False


class CampusV21PairwiseAxes(BaseModel):
    correctness: Literal["unipilot", "competitor", "tie", "unscored"] = "unscored"
    specificity: Literal["unipilot", "competitor", "tie", "unscored"] = "unscored"
    actionability: Literal["unipilot", "competitor", "tie", "unscored"] = "unscored"
    readability: Literal["unipilot", "competitor", "tie", "unscored"] = "unscored"
    would_use: Literal["unipilot", "competitor", "tie", "unscored"] = "unscored"


class CampusV21Pairwise(BaseModel):
    chatgpt: CampusV21PairwiseAxes = Field(default_factory=CampusV21PairwiseAxes)
    gemini: CampusV21PairwiseAxes = Field(default_factory=CampusV21PairwiseAxes)


UXResult = Literal["pass", "fail", "not_applicable", "not_evaluated"]


class CampusV21UXEvaluation(BaseModel):
    tool_card: UXResult = "not_evaluated"
    copy_action: UXResult = "not_evaluated"
    input_flow: UXResult = "not_evaluated"
    clarification: UXResult = "not_evaluated"
    streaming: UXResult = "not_evaluated"
    latency: UXResult = "not_evaluated"


class CampusV21HumanScoreRequest(CampusV2HumanScoreRequest):
    issue_flags: CampusV21IssueFlags = Field(default_factory=CampusV21IssueFlags)
    issues_reviewed: bool = True
    pairwise: CampusV21Pairwise = Field(default_factory=CampusV21Pairwise)
    ux: CampusV21UXEvaluation = Field(default_factory=CampusV21UXEvaluation)
    other_issue: str = Field(default="", max_length=1000)


class CampusV22HumanScoreRequest(BaseModel):
    item_id: str = Field(min_length=1, max_length=150)
    correctness: int = Field(ge=0, le=5)
    depth: int = Field(ge=0, le=5)
    grounding: int = Field(ge=0, le=5)
    usefulness: int = Field(ge=0, le=5)
    naturalness: int = Field(ge=0, le=5)
    would_use_again: int = Field(ge=0, le=5)
    notes: str = Field(default="", max_length=2000)


class CampusV21KnownIssueReviewRequest(BaseModel):
    item_id: str = Field(min_length=1, max_length=150)
    group: Literal["hallucination", "router", "retrieval"]
    status: Literal["pending", "confirmed", "not_reproduced", "accepted_risk"]
    severity: Literal["unreviewed", "low", "medium", "high", "critical"] = "unreviewed"
    blocks_production: bool = False
    notes: str = Field(default="", max_length=2000)
