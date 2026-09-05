from evaluation.diagnose_foundation_v29_generation import loop_details, loop_type, recovery_class


def test_loop_details_identifies_short_periods():
    result = loop_details([7, 8, 9, 9, 9, 9])
    assert result["loop_onset"] == 3
    assert result["loop_length"] == 1
    assert result["loop_type"] == "1_token_loop"


def test_loop_type_and_recovery_classification():
    assert loop_type(3) == "3_to_4_token_loop"
    assert recovery_class([False, False]) == "NOT_RECOVERABLE"
    assert recovery_class([True, False]) == "PARTIALLY_RECOVERABLE"
    assert recovery_class([True, True]) == "RECOVERABLE"
