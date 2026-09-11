from pathlib import Path
from evaluation.diagnose_foundation_v40 import read
from evaluation.build_foundation_v42_holdout import norm,digest
from evaluation.study_foundation_v42_generation import stop_index

def test_fresh_gate_keeps_minimum_and_unconsumed_reserve():
    m=read(Path('evaluation/phase53/fresh-holdout-manifest.json'))
    assert m['gate']=='FRESH_HOLDOUT_TOO_SMALL' and m['documents']==129
    assert sum(s['documents'] for s in m['splits'].values())==129
    assert all(not s['model_scored'] and not s['consumed_for_model_selection'] for s in m['splits'].values())
    assert m['reserve_sealed'] and not m['checkpoint_scoring_started']

def test_provenance_post_cutoff_hashes_and_identity_unique():
    p=read(Path('evaluation/phase53/fresh-data-provenance.json'));rows=p['documents']
    assert len({r['page_id'] for r in rows})==len(rows)
    assert all(r['revision_timestamp']>p['corpus_cutoff']['retrieval_upper_bound'] and r['license']=='CC BY-SA 4.0' for r in rows)
    assert all(len(r['content_sha256'])==64 and r['revision_id'] and r['canonical_source_reference'] for r in rows)
    assert digest(norm('Ａ B\nＣ'))==digest(norm('abc'))

def test_final_blind_not_parsed_and_no_fabricated_confirmation():
    leak=read(Path('evaluation/phase53/leakage-report.json'));assert not leak['FinalBlind_content_opened']
    assert all(not p['path'].endswith('final-blind-1000.json') for p in leak['input_files'])
    s=read(Path('evaluation/foundation-v42-confirmatory-summary.json'))
    assert s['5e-5_confirmatory_metrics'] is None and s['eos']=='NOT_RUN'
    assert not s['new_training'] and not s['consumed_for_model_selection']

def test_stop_replay_rule_is_fixed_bounded_and_not_fabricated_eos():
    assert stop_index([1,2]*8)==8
    assert stop_index(list(range(20))) is None
    assert stop_index([1]*7) is None
