from pathlib import Path
from evaluation.diagnose_foundation_v40 import read


def test_confirmatory_never_rebrands_used_or_uncertain_data():
    p=read(Path('evaluation/phase52/data-provenance.json'))
    assert p['availability']=='PROVENANCE_UNCERTAIN' and not p['clean_set_found']
    assert all(r['confirmatory_eligible']=='NO' for r in p['rows'])
    assert any('foundation_v11' in r['path'] and 'test' in r['path'] and r['selection_influence']=='YES' for r in p['rows'])
    s=read(Path('evaluation/foundation-v41-confirmatory-summary.json'))
    assert s['confirmatory_gate']=='CONFIRMATORY_DATA_INVALID'
    assert not s['new_training'] and not s['canonical_training'] and s['confirmatory_set'] is None


def test_blind_hash_only_and_attractor_no_causal_promotion():
    p=read(Path('evaluation/phase52/data-provenance.json'))
    blind=next(r for r in p['rows'] if r['path'].endswith('final-blind-1000.json'))
    assert blind['sha256']=='fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b'
    assert not p['final_blind_content_opened']
    a=read(Path('evaluation/foundation-v41-attractor-summary.json'))
    assert len(a['prompt_onset_comparisons'])==900
    assert len(a['top20_cycle_train_counts'])==20
    assert a['classification']=='MIXED_OR_UNKNOWN' and not a['new_gpu_inference']
    assert all(r['pre_onset_steps_available']<=32 for r in a['prompt_onset_comparisons'])
