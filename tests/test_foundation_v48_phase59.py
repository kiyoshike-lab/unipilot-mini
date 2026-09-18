"""Static PHASE59 boundaries; CUDA execution is intentionally not a unit test."""
from evaluation import run_foundation_v48_phase59 as p
from evaluation import generation_contract_v47 as c

def test_registered_budget_and_permutations_are_fixed():
    assert p.SEEDED if False else p.SEEDS == (123, 2026)
    assert p.EXPECTED_PERM[123] != p.EXPECTED_PERM[2026]
    assert p.START == 'fcdf87966e37aaa3be8cec94ab30f8552cd232ec'

def test_phase59_has_no_extension_or_intervention_cli():
    text=p.Path(p.__file__).read_text(encoding='utf8')
    assert "choices=('preflight','dry-run','train','evaluate')" in text
    assert "anti_loop" not in text

def test_contract_has_exactly_34_registered_comparisons():
    spec=p.read(p.SPEC)
    assert len(c.mapping(spec['all_safeguards'])) == 34
