"""A blocked data gate must never become a scored LR approval or training arm."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))

def test_supplemental_is_unqualified_despite_quantity():
    m=read('evaluation/phase54/fresh-holdout-manifest.json')
    assert m['documents']>=250 and m['tokens']>=200000
    assert m['gate']=='LEAKAGE_RISK_TOO_HIGH' and m['splits']=={}
    assert not m['checkpoint_scoring_started'] and not m['future_reserve2_scored']
    assert m['phase53_reserve']=='UNTOUCHED'

def test_no_confirmatory_or_lr_approval_without_leakage_clearance():
    spec=read('evaluation/phase54/lr-confirmatory-preregistration.json')
    result=read('evaluation/foundation-v43-confirmatory-summary.json')
    assert spec['documents']==[] and not spec['scoring_started']
    assert result['formal_lr_gate']=='FORMAL_LR_UNRESOLVED'
    assert result['approved_lr'] is None and result['confirmatory_gate']=='CONFIRMATORY_INVALID'
    assert result['C_metrics']==result['B_metrics']=='NOT_RUN'

def test_fix_design_cannot_authorize_training_or_recycle_failed_auxiliary():
    d=read('evaluation/foundation-v43-training-fix-preregistration.json')
    assert d['phase55_arms']==[] and d['approved_lr'] is None
    assert d['gate']=='MORE_ROOT_CAUSE_WORK_REQUIRED'
    assert 'excluded' in d['repetition_auxiliary']
    assert not any(d[k] for k in ('new_training','canonical','20m','foundation_base'))
    assert d['success_targets']['new_severe_normal_control_errors_max']==0
