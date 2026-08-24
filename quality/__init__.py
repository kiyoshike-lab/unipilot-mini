"""Deterministic, local-only quality evaluation for UniPilot Campus."""

from quality.campus_ai_judge import CampusAIJudge
from quality.campus_answer_improver import CampusAnswerImprover

__all__ = ["CampusAIJudge", "CampusAnswerImprover"]
