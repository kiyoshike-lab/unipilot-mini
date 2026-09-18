"""Versioned evaluation boundary for future studies; never rescores PHASE57.

No inference, training, file I/O, aliases, or missing-metric defaults here.
Legacy conversion is explicit and versioned. All consumers validate first.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import math

VERSION = 'generation-eval-v2'
SAFETY_VERSION = 'quality-safety-v2'
LEGACY_VERSION = 'phase51-metrics-v1'
FAMILIES = ('math', 'code', 'lists', 'definitions', 'terminology')
RNG_BASES = ('44000', '51000', '52000')
CONVERSION = {
    'runaway_rate': 'runaway_rate',
    'completion_rate': 'sentence_completion_rate',
    'naturalness_rate': 'naturalness_rate',
    'semantic_rate': 'semantic_rate',
    'topic_retention_rate': 'topic_retention_proxy',
    'japanese_validity_rate': 'japanese_validity',
    'eos_completion_rate': 'eos_rate',
    'mean_eos_probability': 'mean_eos_probability',
    'repetition_1': 'repetition_1', 'repetition_2': 'repetition_2',
    'repetition_3': 'repetition_3', 'repetition_4': 'repetition_4',
    'unique_token_ratio': 'unique_token_ratio',
    'loop_onset': 'loop_onset', 'loop_cycle_length': 'loop_cycle_length',
}
UNITS = {k: ('generated_token_index' if k == 'loop_onset' else
             'tokens' if k == 'loop_cycle_length' else
             'probability' if k == 'mean_eos_probability' else 'ratio_0_1')
         for k in CONVERSION}

# Explicit measurement producer paths. No consumer infers or guesses aliases.
SCALAR_PRODUCERS = {
    'validation.ce':'validation.ce',
    **{f'validation.top{k}':f'validation.top{k}' for k in (1,5,10)},
    **{f'context.{k}.ce':f'context.{k}.mean_ce' for k in (128,512)},
    'eos.terminal_probability':'eos.terminal_mean_probability',
    'eos.terminal_top1':'eos.terminal_top1',
    'eos.nonterminal_probability':'eos.nonterminal_mean_probability',
    'eos.premature_argmax':'eos.premature_argmax_eos',
    **{f'{group}.{k}':f'{group}.metrics.{source}' for group in ('core','supported_tail')
       for k,source in [('micro_ce','micro_ce'),('macro_ce','macro_per_token_ce'),('top1','top1'),('top5','top5'),('top10','top10')]},
    **{f'{group}.paired_ce_ci95_upper':f'paired_document_bootstrap_10000_seed5701.{group}.upper' for group in ('core','supported_tail')},
    'normal.mean_ce':'normal_controls.mean_ce',
    'normal.terminal_probability':'normal_controls.terminal_eos_probability',
    **{f'normal.{f}.ce':f'normal_controls.family_ce.{f}' for f in FAMILIES},
}


class ContractError(ValueError):
    def __init__(self, message):
        super().__init__('EVALUATOR_CONTRACT_FAIL: ' + message)


def exact_keys(value, keys, where):
    if type(value) is not dict or set(value) != set(keys):
        raise ContractError(f'{where}: missing/unknown fields')


def number(value, unit, where):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ContractError(f'{where}: nonfinite or wrong numeric type')
    if unit in ('probability', 'ratio_0_1') and not 0 <= value <= 1:
        raise ContractError(f'{where}: expected [0,1], not percentage')
    if unit in ('CE_nats', 'tokens', 'generated_token_index') and value < 0:
        raise ContractError(f'{where}: negative {unit}')


def validate_generation(value):
    exact_keys(value, ('schema_version', 'examples', 'metrics'), 'generation')
    if value['schema_version'] != VERSION or type(value['examples']) is not int or value['examples'] <= 0:
        raise ContractError('generation version/examples')
    exact_keys(value['metrics'], CONVERSION, 'generation.metrics')
    for k, v in value['metrics'].items():
        number(v, UNITS[k], k)
    return value


def convert_legacy(metrics, *, source_version):
    if source_version != LEGACY_VERSION:
        raise ContractError('explicit supported source version required')
    required = set(CONVERSION.values()) | {'examples', 'mean_repetition_1'}
    optional = {'entropy', 'top1_top2_margin'}
    if type(metrics) is not dict or not required <= set(metrics) or set(metrics) - required - optional:
        raise ContractError('legacy runner schema mismatch')
    for k in optional & metrics.keys():
        number(metrics[k], 'CE_nats' if k == 'entropy' else 'ratio_0_1', k)
    number(metrics['mean_repetition_1'], 'ratio_0_1', 'mean_repetition_1')
    if metrics['mean_repetition_1'] != metrics['repetition_1']:
        raise ContractError('legacy repetition duplicate inconsistent')
    value = {'schema_version': VERSION, 'examples': metrics['examples'],
             'metrics': {k: metrics[v] for k, v in CONVERSION.items()}}
    return validate_generation(value)


def mapping(thresholds):
    """One entry per concrete comparison, with an explicit source threshold."""
    rows = []
    def add(field, key, operation, unit, producer=None, threshold=None):
        rows.append({'id': field, 'producer_field': producer or SCALAR_PRODUCERS[field],
                     'consumer_field': field, 'unit': unit, 'comparison': operation,
                     'threshold_key': key, 'threshold': thresholds[key] if threshold is None else threshold,
                     'missing_behavior': 'EVALUATOR_CONTRACT_FAIL'})
    add('validation.ce', 'validation_ce_increase_max', 'increase', 'CE_nats')
    for k in (1, 5, 10):
        add(f'validation.top{k}', 'validation_top1_drop_absolute_max' if k == 1 else 'validation_top5_top10_drop_absolute_max', 'drop', 'ratio_0_1')
    for c in (128, 512):
        add(f'context.{c}.ce', 'context_ce_increase_each_128_512_max', 'increase', 'CE_nats')
    add('context.long_benefit', 'context_long_benefit_loss_max', 'drop', 'CE_difference_nats', 'context.128.ce - context.512.ce')
    add('eos.terminal_probability', 'terminal_mean_eos_probability_ratio_min', 'ratio_min', 'probability')
    add('eos.terminal_top1', 'terminal_eos_top1_drop_absolute_max', 'drop', 'ratio_0_1')
    add('eos.nonterminal_probability', 'nonterminal_eos_probability_increase_max', 'increase', 'probability')
    add('eos.premature_argmax', 'premature_argmax_eos_increase_max', 'increase', 'ratio_0_1')
    for group in ('core', 'supported_tail'):
        for metric in ('micro_ce', 'macro_ce'):
            add(f'{group}.{metric}', f'{group}_micro_macro_ce_increase_max', 'increase', 'CE_nats')
        for k in (1, 5, 10):
            add(f'{group}.top{k}', 'frequency_top1_drop_absolute_max' if k == 1 else 'frequency_top5_top10_drop_absolute_max', 'drop', 'ratio_0_1')
        add(f'{group}.paired_ce_ci95_upper', 'frequency_ce_paired_ci95_upper_max', 'upper', 'CE_difference_nats',
            'paired_document_bootstrap_10000_seed5701.' + group + '.upper', thresholds['frequency_ce_paired_ci95_upper_max'][group])
    add('normal.mean_ce', 'normal_mean_ce_increase_max', 'increase', 'CE_nats')
    add('normal.terminal_probability', 'normal_terminal_eos_probability_ratio_min', 'ratio_min', 'probability')
    for f in FAMILIES:
        add(f'normal.{f}.ce', 'normal_each_family_ce_increase_max', 'increase', 'CE_nats')
    for metric, key in [('naturalness_rate', 'sampling_naturalness_semantic_proxy_drop_max'),
                        ('semantic_rate', 'sampling_naturalness_semantic_proxy_drop_max'),
                        ('topic_retention_rate', 'sampling_topic_retention_drop_max'),
                        ('japanese_validity_rate', 'japanese_validity_drop_max')]:
        add('sampling.' + metric, key, 'drop', 'ratio_0_1', 'sampling[*].metrics.' + metric + ' (equal 100-prompt RNG mean)')
    expected = set(thresholds) - {'missing_metric'}
    if {r['threshold_key'] for r in rows} != expected:
        raise ContractError('unmapped threshold')
    return rows


def validate_safety(value, thresholds):
    exact_keys(value, ('schema_version', 'scalars', 'sampling'), 'safety')
    if value['schema_version'] != SAFETY_VERSION:
        raise ContractError('safety version')
    rows = [r for r in mapping(thresholds) if not r['id'].startswith('sampling.')]
    exact_keys(value['scalars'], [r['id'] for r in rows], 'safety.scalars')
    for r in rows:
        number(value['scalars'][r['id']], r['unit'], r['id'])
    exact_keys(value['sampling'], RNG_BASES, 'sampling RNGs')
    for v in value['sampling'].values():
        validate_generation(v)
    if len({v['examples'] for v in value['sampling'].values()}) != 1:
        raise ContractError('unequal sampling geometry')
    s = value['scalars']
    if abs(s['context.long_benefit'] - (s['context.128.ce'] - s['context.512.ce'])) > 1e-12:
        raise ContractError('context benefit inconsistent')
    return value


def from_measurement_fields(fields, sampling, thresholds):
    """Explicit flat projection from named producer outputs for a future runner.

    Caller projects exactly these paths from actual measured outputs. CI fields
    are candidate-minus-own-parent statistics, not independent model metrics.
    No PHASE57 artifact is passed here by the forensic auditor.
    """
    exact_keys(fields, SCALAR_PRODUCERS.values(), 'measurement projection')
    scalars={k:fields[path] for k,path in SCALAR_PRODUCERS.items()}
    number(scalars['context.128.ce'],'CE_nats','context128')
    number(scalars['context.512.ce'],'CE_nats','context512')
    scalars['context.long_benefit']=scalars['context.128.ce']-scalars['context.512.ce']
    return validate_safety({'schema_version':SAFETY_VERSION,'scalars':scalars,'sampling':deepcopy(sampling)},thresholds)


def safety_gate(candidate, parent, thresholds):
    validate_safety(candidate, thresholds)
    validate_safety(parent, thresholds)
    if any(candidate['sampling'][k]['examples'] != parent['sampling'][k]['examples'] for k in RNG_BASES):
        raise ContractError('unpaired sampling example count')
    def get(value, field):
        if field.startswith('sampling.'):
            metric = field.split('.')[1]
            return sum(Decimal(str(value['sampling'][k]['metrics'][metric])) for k in RNG_BASES) / 3
        return Decimal(str(value['scalars'][field]))
    checks = {}
    for r in mapping(thresholds):
        a, b = get(candidate, r['id']), get(parent, r['id'])
        op = r['comparison']
        if op == 'ratio_min' and b == 0:
            raise ContractError('zero probability denominator: ' + r['id'])
        value = a-b if op == 'increase' else b-a if op == 'drop' else a/b if op == 'ratio_min' else a
        limit = Decimal(str(r['threshold']))
        passed = value >= limit if op == 'ratio_min' else value <= limit
        checks[r['id']] = {'value': float(value), 'threshold': float(limit), 'pass': passed}
    return {'schema_version': SAFETY_VERSION, 'gate': 'CONTROL_SAFETY_PASS' if all(r['pass'] for r in checks.values()) else 'CONTROL_SAFETY_FAIL', 'checks': checks}


def fixture(thresholds):
    gen = {'schema_version': VERSION, 'examples': 2,
           'metrics': {k: 1. if UNITS[k] in ('tokens', 'generated_token_index') else .5 for k in CONVERSION}}
    scalars = {r['id']: (0. if r['comparison'] == 'upper' or r['id'] == 'context.long_benefit' else
                        4. if r['unit'] == 'CE_nats' else .5)
               for r in mapping(thresholds) if not r['id'].startswith('sampling.')}
    return {'schema_version': SAFETY_VERSION, 'scalars': scalars,
            'sampling': {k: deepcopy(gen) for k in RNG_BASES}}


def schema_document(thresholds):
    return {'schema_version': VERSION, 'safety_schema_version': SAFETY_VERSION,
            'canonical_fields': UNITS, 'explicit_source_version': LEGACY_VERSION,
            'explicit_conversion': CONVERSION, 'unknown_missing_type_nonfinite': 'FAIL_CLOSED',
            'ratio_range': [0, 1], 'percentage_input': 'REJECT',
            'completion_definition': 'sentence_completion_rate automatic proxy, not EOS',
            'loop_onset_definition': 'legacy mean; no-loop sentinel 129 retained',
            'safeguard_mapping': mapping(thresholds), 'thresholds': thresholds,
            'comparison_arithmetic': 'Decimal(str(value)); exact inclusive decimal thresholds, no epsilon or margin relaxation',
            'interpretation': 'automatic proxies; no human quality guarantee'}
