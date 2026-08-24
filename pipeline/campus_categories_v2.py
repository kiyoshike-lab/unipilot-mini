from __future__ import annotations

from pipeline.campus_categories import CAMPUS_CATEGORIES


LEVEL1_GROUPS = {
    "academic": (
        "exam", "assignment", "credit", "gpa", "grade_simulator", "attendance", "lateness", "registration",
    ),
    "communication": (
        "professor_email", "absence_email", "lateness_email", "late_submission_email",
    ),
    "planning": ("schedule", "study_plan", "assignment_priority", "deadline_organizer"),
    "writing": ("report_outline", "citation_check", "presentation_outline"),
    "career": ("career_schedule", "es_outline", "toeic_plan", "internship"),
    "campus_support": (
        "scholarship", "tuition", "part_time_job", "campus_life", "relationship", "university_policy",
    ),
    "skills": ("programming", "ai_usage", "math", "statistics"),
    "information": ("faq_search", "general"),
}

CATEGORY_TO_LEVEL1 = {
    category: level1 for level1, categories in LEVEL1_GROUPS.items() for category in categories
}

assert set(CATEGORY_TO_LEVEL1) == set(CAMPUS_CATEGORIES)

ROUTE_ACTIONS = (
    "FAQ", "TOOL", "RAG", "MODEL", "TOOL+MODEL", "RAG+MODEL", "CLARIFY",
)

TOOL_AVAILABLE = {
    "gpa", "grade_simulator", "professor_email", "absence_email", "lateness_email",
    "late_submission_email", "registration", "study_plan", "assignment_priority",
    "deadline_organizer", "report_outline", "citation_check", "presentation_outline",
    "career_schedule", "es_outline", "toeic_plan", "credit", "exam", "schedule",
}

