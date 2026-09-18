"""Behavioral contract tests, including every registered safeguard's wiring."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from evaluation import generation_contract_v47 as c

ROOT = Path(__file__).resolve().parents[1]
TH = json.loads((ROOT/'evaluation/phase56/phase57-training-preregistration.json').read_text(encoding='utf8'))['quality_safeguards_vs_each_parent_and_control']


def legacy_fixture():
    gen = c.fixture(TH)['sampling']['44000']
    raw = {old: gen['metrics'][new] for new, old in c.CONVERSION.items()}
    raw.update(examples=2, mean_repetition_1=raw['repetition_1'])
    return raw


def test_explicit_conversion_roundtrip_and_no_silent_alias():
    raw = legacy_fixture()
    value = c.convert_legacy(raw, source_version=c.LEGACY_VERSION)
    assert value['metrics']['naturalness_rate'] == raw['naturalness_rate']
    assert value['metrics']['semantic_rate'] == raw['semantic_rate']
    assert value['metrics']['topic_retention_rate'] == raw['topic_retention_proxy']
    with pytest.raises(c.ContractError):
        c.convert_legacy(raw, source_version='auto')


@pytest.mark.parametrize('field', list(c.CONVERSION))
def test_missing_canonical_field_fails(field):
    value = c.fixture(TH)['sampling']['44000']
    del value['metrics'][field]
    with pytest.raises(c.ContractError): c.validate_generation(value)


@pytest.mark.parametrize('field,alias', [('naturalness_rate','natural_japanese_proxy'), ('semantic_rate','semantic_coherence_proxy')])
def test_old_mismatch_reproduced_and_rejected_before_gate(field,alias):
    raw=legacy_fixture();raw[alias]=raw.pop(field)
    with pytest.raises(c.ContractError): c.convert_legacy(raw, source_version=c.LEGACY_VERSION)
    value=c.fixture(TH);gen=value['sampling']['44000']['metrics'];gen[alias]=gen.pop(field)
    with pytest.raises(c.ContractError): c.safety_gate(value,c.fixture(TH),TH)


@pytest.mark.parametrize('bad',[1.2,-.1,float('nan'),float('inf'),float('-inf'),'0.72',True,None,72])
@pytest.mark.parametrize('field',[k for k in c.UNITS if c.UNITS[k] in ('ratio_0_1','probability')])
def test_invalid_units_types_and_nonfinite_rejected(bad,field):
    value=c.fixture(TH);value['sampling']['44000']['metrics'][field]=bad
    with pytest.raises(c.ContractError): c.safety_gate(value,c.fixture(TH),TH)


@pytest.mark.parametrize('row',c.mapping(TH),ids=lambda r:r['id'])
def test_every_registered_safeguard_is_live(row):
    parent=c.fixture(TH);candidate=deepcopy(parent);key=row['id'];limit=row['threshold']
    if key.startswith('sampling.'):
        for gen in candidate['sampling'].values():gen['metrics'][key.split('.')[1]]-=limit+.01
    else:
        b=candidate['scalars'][key]
        candidate['scalars'][key]=b+limit+.01 if row['comparison']=='increase' else b-limit-.01 if row['comparison']=='drop' else b*(limit-.01) if row['comparison']=='ratio_min' else limit+.01
        if key.startswith('context.'):
            s=candidate['scalars']
            if key=='context.long_benefit':s['context.512.ce']=s['context.128.ce']-s[key]
            else:s['context.long_benefit']=s['context.128.ce']-s['context.512.ce']
    result=c.safety_gate(candidate,parent,TH)
    assert result['gate']=='CONTROL_SAFETY_FAIL'
    assert result['checks'][key]['pass'] is False


def test_complete_mapping_and_zero_denominator_fail_closed():
    assert {r['threshold_key'] for r in c.mapping(TH)}==set(TH)-{'missing_metric'}
    parent=c.fixture(TH);parent['scalars']['eos.terminal_probability']=0
    with pytest.raises(c.ContractError):c.safety_gate(c.fixture(TH),parent,TH)


def test_unknown_scalar_and_schema_version_fail_closed():
    for mode in ('unknown','missing','version','nan'):
        value=c.fixture(TH)
        if mode=='unknown':value['scalars']['extra']=0
        elif mode=='missing':del value['scalars']['core.micro_ce']
        elif mode=='version':value['schema_version']='v1'
        else:value['scalars']['core.micro_ce']=float('nan')
        with pytest.raises(c.ContractError):c.safety_gate(value,c.fixture(TH),TH)


def test_sampling_consumer_cannot_fall_back_to_greedy():
    parent=c.fixture(TH);a=deepcopy(parent)
    for v in a['sampling'].values():v['metrics']['naturalness_rate']=.4
    assert not c.safety_gate(a,parent,TH)['checks']['sampling.naturalness_rate']['pass']
    a['greedy']=c.fixture(TH)['sampling']['44000']
    with pytest.raises(c.ContractError):c.safety_gate(a,parent,TH)


def test_inclusive_decimal_boundary_not_binary_roundoff_or_tolerance():
    parent=c.fixture(TH);a=deepcopy(parent)
    for v in a['sampling'].values():v['metrics']['japanese_validity_rate']=.49
    assert c.safety_gate(a,parent,TH)['checks']['sampling.japanese_validity_rate']['pass']
    for v in a['sampling'].values():v['metrics']['japanese_validity_rate']=.489999
    assert not c.safety_gate(a,parent,TH)['checks']['sampling.japanese_validity_rate']['pass']


def test_runner_adapter_wrapper_summary_gate_dry_run_fixture():
    raw=legacy_fixture();parent=c.fixture(TH);a=deepcopy(parent)
    for key in c.RNG_BASES:a['sampling'][key]=c.convert_legacy(raw,source_version=c.LEGACY_VERSION)
    projection={path:a['scalars'][key] for key,path in c.SCALAR_PRODUCERS.items()}
    a=c.from_measurement_fields(projection,a['sampling'],TH)
    result=c.safety_gate(a,parent,TH)
    assert result['schema_version']==c.SAFETY_VERSION and result['gate']=='CONTROL_SAFETY_PASS'
    assert len(result['checks'])==len(c.mapping(TH))


@pytest.mark.parametrize('source',list(c.SCALAR_PRODUCERS.values()))
def test_every_missing_producer_field_is_rejected(source):
    value=c.fixture(TH);projection={path:value['scalars'][key] for key,path in c.SCALAR_PRODUCERS.items()}
    del projection[source]
    with pytest.raises(c.ContractError):c.from_measurement_fields(projection,value['sampling'],TH)


def test_committed_contract_and_real_cuda_receipt_match_code():
    schema=json.loads((ROOT/'evaluation/phase58/evaluator-contract-v2.json').read_text(encoding='utf8'))
    assert schema==c.schema_document(TH)
    receipt=json.loads((ROOT/'evaluation/phase58/evaluator-contract-audit.json').read_text(encoding='utf8'))
    assert receipt['dry_run']=='PASS' and receipt['new_training'] is False
    value=c.fixture(TH);value['sampling']=receipt['generation']
    for generation in receipt['generation'].values():c.validate_generation(generation)
    assert c.safety_gate(value,value,TH)==receipt['wrapper_to_gate']
    assert receipt['all_safeguards_mapped']['concrete_comparisons']==len(c.mapping(TH))
