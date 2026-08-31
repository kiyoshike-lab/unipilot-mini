from __future__ import annotations

from evaluation.audit_foundation_v19_benchmark import numeric_audit
from foundation.base_tokenizer import FoundationTokenizer


def test_numeric_audit_separates_symbolic_and_actual_tokenizer_tasks():
    tokenizer = FoundationTokenizer.load("tokenizer/foundation-v11-base-4096.json")
    result = numeric_audit(tokenizer)
    assert len(result["standalone_samples"]) >= 20
    assert len(result["actual_numeric_pattern_samples"]) >= 20
    assert len(result["old_synthetic_numeric_samples"]) >= 20
    assert result["tasks_separated"]["tokenizer_independent_symbolic"].endswith(
        "architecture gate >=95%"
    )
    assert result["arithmetic_audit"]["deprecated_v4_numeric_requires_modular_addition"] is True
    assert result["arithmetic_audit"]["symbolic_repetition_requires_arithmetic"] is False
