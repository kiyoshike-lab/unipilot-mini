"""PHASE51: preregistered read-only checkpoint diagnostics, never training."""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from training.checkpoint_paths import checkpoint_root, existing_checkpoint_path
from training.train_foundation_v21_ab import file_sha256
from training.run_foundation_v36_lr_review import verify_payload
from training.run_foundation_v35_thermal_gate import cooldown, Monitor
from evaluation.evaluate_foundation_v39_gate import evaluated_checkpoint, verify_preserved
from evaluation.diagnose_foundation_v29_generation import (
    build_prefixes, document_ranges, generate_batch, load_model, ngram_repetition,
    summarize_generation,
)
from evaluation.evaluate_foundation_v33_context_gate import GREEDY, SAMPLE_T07
from evaluation.investigate_foundation_v14 import language_proxy
from foundation.base_tokenizer import FoundationTokenizer

OUT = ROOT / 'evaluation/phase51'
PREREG = OUT / 'preregistration.json'
SEEDS = (42, 123, 2026)
ARMS = ('baseline', 'C', 'B')
BLIND = ROOT / 'data/foundation_v09/evaluation/final-blind-1000.json'
BLIND_SHA = 'fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def new_json(path, value):
    """Exclusive artifact creation: a stopped/partial artifact requires inspection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def data():
    tok = FoundationTokenizer.load(ROOT / 'tokenizer/foundation-v11-base-4096.json')
    val = np.memmap(ROOT / 'data/foundation_v11/packed/vocab-4096/validation.bin', dtype=np.uint16, mode='r')
    return tok, val


def resolver_gate():
    root = checkpoint_root(ROOT)
    assert str(root) == r'Z:\AI\unipilot-mini\checkpoints', 'PROCESS_ENV_RESOLVER_MISMATCH'
    manifest = read(ROOT / 'evaluation/phase50/z-migration.json')
    assert manifest['gate'] == 'Z_CHECKPOINT_MIGRATION_PASS'
    for row in manifest['rows']:
        resolved = existing_checkpoint_path(ROOT, *Path(row['relative_path']).parts[1:])
        assert resolved == Path(row['destination']) and resolved.is_relative_to(root) and resolved.is_file()
    verify_preserved()
    assert file_sha256(BLIND) == BLIND_SHA  # Hash only; never parse blind content.
    return manifest


def freeze():
    manifest = resolver_gate()
    expected = '7ebec725cbf1c7cddcf7b7953659f5f2e23e5025'
    git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    assert git('branch', '--show-current') == 'foundation-research'
    assert git('rev-parse', 'HEAD') == expected
    assert git('ls-remote', 'origin', 'refs/heads/foundation-research').split()[0] == expected
    tok, val = data()
    ranges = document_ranges(val, tok.bos_id, tok.eos_id)
    terminal = [int(p) for p in np.flatnonzero(val == tok.eos_id) if p >= 128]
    eligible = np.flatnonzero((val != tok.eos_id) & (val != tok.bos_id))
    eligible = eligible[eligible >= 128]
    nonterminal = np.random.default_rng(5101).choice(eligible, 500, replace=False)
    definition = ROOT / 'evaluation/foundation-v39-supported-tail.json'
    assert file_sha256(definition) == '7d4a890faf0b4d366ad147eb01fc7c2f4cfb7a4e610b305f616471e720cd2c85'
    spec = {
        'phase': 51, 'registered_at_utc': datetime.now(timezone.utc).isoformat(),
        'new_diagnostic_results_seen': False, 'new_training': False, 'device': 'cuda', 'dtype': 'float32',
        'cpu_heavy_parallel_evaluation': False, 'checkpoint_root': str(checkpoint_root(ROOT)),
        'seeds': list(SEEDS), 'arms': list(ARMS), 'terminal_positions': terminal,
        'nonterminal_positions': sorted(map(int, nonterminal)), 'context': 128,
        'prompts': build_prefixes(val, ranges, tok), 'sampling_rng_bases': [44000, 51000, 52000],
        'sampling_seed_rule': 'base + prompt index; independent CUDA generator per prompt',
        'sampling_decoder': SAMPLE_T07, 'sampling_max_new_tokens': 64,
        'greedy_decoder': GREEDY, 'greedy_max_new_tokens': 128,
        'bootstrap': {'unit': 'paired document; paired prompt averaged across RNG for sampling',
            'replicates': 10000, 'seeds': [4900, 5100, 2026], 'interval': 'percentile 95%, exploratory; no multiplicity correction'},
        'frequency_population_file': str(definition.relative_to(ROOT)), 'frequency_population_sha256': file_sha256(definition),
        'metrics': {'eos': ['P(EOS)', 'rank', 'top1/5/10', 'nonterminal P(EOS)', 'premature EOS', 'competitor', 'logit margin', 'entropy', 'document length', 'previous-token class'],
            'core': ['paired CE/probability delta by token/document', 'occurrences', 'training frequency', 'top20 positive CE contributions', 'leave-one-document CI', 'Monte Carlo CI stability'],
            'sampling': ['naturalness proxy', 'semantic proxy', 'completion proxy', 'topic token overlap', 'repetition 1..4', 'Japanese validity', 'EOS completion', 'runaway'],
            'greedy': ['loop onset', 'repetition 1..4', 'entropy', 'top1-top2 probability margin', 'unique token ratio', 'loop cycle length']},
        'classification_rules': {
            'eos': 'TRUE_EOS_REGRESSION if all three seeds paired document P(EOS) delta CI upper < 0 and mean ratio < .9 vs 15.872M; otherwise SEED_LOCAL_EOS_VARIANCE if negative CI only some seeds; otherwise OUTLIER_DOCUMENTS if top5 positive loss contributions >=50%; otherwise RELATIVE_THRESHOLD_SENSITIVITY if any old gate fails but all unique-doc baseline CIs include zero or improve; otherwise EOS_MEASUREMENT_UNCERTAINTY.',
            'core': 'For seed2026 positive CE contributions: top5 documents >=50% -> DOCUMENT_CONCENTRATED; else top20 token types >=50% -> TOKEN_CONCENTRATED; else SEED_LOCAL if other seeds have nonpositive means; else BROAD. Report seed pattern separately. LOO never removes data from gate.',
            'sampling': 'Paired prompt CI for naturalness and semantic, average three RNGs. TRUE_SAMPLING_REGRESSION if either metric CI upper < -.08 in every checkpoint seed; else SEED_LOCAL_VARIANCE if this occurs in some seeds; else PROMPT_LOCAL_REGRESSION if top10 prompts explain >=50% positive losses; else SAMPLING_RNG_VARIANCE if RNG means straddle -.08; else EVALUATOR_NOISE (proxy uncertainty, not proof of no regression).',
            'attractor': 'C minus B, averaged seeds, paired prompt differences: later onset, lower repetition1, higher unique ratio count as improvement; reverse as worsening; mixed directions MIXED, all tied STATIC, consistent direction IMPROVING/WORSENING. Runaway100% remains unresolved regardless.',
            'combined': 'STOP_AND_INVESTIGATE on integrity/determinism failure; TRUE_SAFETY_REGRESSION on all-seed true EOS/sampling regression; MEASUREMENT_UNCERTAINTY_REMAINS if unchanged per-seed gates fail without established resolution; ATTRACTOR_REVIEW_REQUIRED if safety resolves but runaway persists; only otherwise recommend superior paired LM/frequency candidate. Recommendation never permission.'},
        'acceptance_interpretation': 'PHASE49/50 thresholds/membership remain immutable. Diagnostics are explanatory, never replacement eligibility gates. Old EOS/sampling gate references own LR 256k control; new comparisons baseline 15.872M plus preserved old gate values. Automatic language proxies are not human quality or semantic relevance judgments.',
        'source_sha256': {str(p.relative_to(ROOT)): file_sha256(p) for p in [
            ROOT/'evaluation/diagnose_foundation_v29_generation.py', ROOT/'evaluation/investigate_foundation_v14.py',
            ROOT/'evaluation/evaluate_foundation_v33_context_gate.py', ROOT/'data/foundation_v11/packed/vocab-4096/validation.bin']},
    }
    new_json(PREREG, spec)
    new_json(OUT / 'resolver-gate.json', {'gate': 'PROCESS_ENV_Z_ROOT_PASS', 'user_root': str(checkpoint_root(ROOT)),
        'initial_process_root': r'D:\UniPilot\active\checkpoints', 'corrected_process_root': os.environ['UNIPILOT_CHECKPOINT_ROOT'],
        'python_resolver': str(checkpoint_root(ROOT)), 'migration_repeated': False, 'checkpoint_copy': 0,
        'manifest_reused': True, 'manifest_sha256': file_sha256(ROOT/'evaluation/phase50/z-migration.json'),
        'destinations_match': len(manifest['rows']), 'representative_integrity_previously_verified': ['baseline-2026', 'C-123', 'B-2026'],
        'representative_sha_strict_load_metadata': 'PASS', 'runtime_hardcoded_D_path': False,
        'head': expected, 'branch': git('branch','--show-current'), 'origin_match': True,
        'c_free_bytes': shutil.disk_usage('C:\\').free, 'z_free_bytes': shutil.disk_usage(checkpoint_root(ROOT)).free,
        'cuda_available': torch.cuda.is_available(), 'protected_files': manifest['protected_files'], 'final_blind_sha256': file_sha256(BLIND)})
    print('PHASE51 preregistration frozen; resolver PASS', flush=True)


def cluster_ci(delta, groups, seed=4900, replicates=10000):
    delta, groups = np.asarray(delta, dtype=float), np.asarray(groups)
    _, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=delta)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(replicates, len(sums)))
    boot = sums[picks].sum(1) / counts[picks].sum(1)
    return {'mean': float(delta.mean()), 'lower': float(np.quantile(boot,.025)), 'upper': float(np.quantile(boot,.975))}


@torch.inference_mode()
def eos_rows(model, tok, val, positions):
    docs = np.maximum(0, np.cumsum(val == tok.bos_id)-1)
    starts = np.flatnonzero(val == tok.bos_id)
    ends = np.flatnonzero(val == tok.eos_id)
    result = []
    for offset in range(0, len(positions), 16):
        batch = positions[offset:offset+16]
        x = torch.tensor(np.stack([val[p-128:p].astype(np.int64) for p in batch]), device='cuda')
        logits, _ = model(x)
        scores = logits[:,-1].float()
        probs = scores.softmax(-1)
        entropy = -(probs * probs.clamp_min(1e-30).log()).sum(-1)
        ranks = (scores > scores[:,tok.eos_id,None]).sum(-1)+1
        competitors = scores.clone(); competitors[:,tok.eos_id] = -torch.inf
        best = competitors.argmax(-1)
        for i,p in enumerate(batch):
            doc = int(docs[p]); previous = tok.decode([int(val[p-1])], skip_special=True)
            previous_class = 'punctuation' if re.search(r'[。！？.!?]$', previous) else 'newline' if '\n' in previous else 'other'
            result.append({'position': p, 'document': doc, 'document_length': int(ends[doc]-starts[doc]-1),
                'probability': float(probs[i,tok.eos_id]), 'rank': int(ranks[i]),
                'top1': bool(scores[i].argmax()==tok.eos_id), 'top5': bool(ranks[i]<=5), 'top10': bool(ranks[i]<=10),
                'competitor_id': int(best[i]), 'competitor_text': tok.decode([int(best[i])],skip_special=True),
                'competitor_minus_eos_logit': float(scores[i,best[i]]-scores[i,tok.eos_id]),
                'entropy': float(entropy[i]), 'previous_token_id': int(val[p-1]), 'previous_token_class': previous_class})
    return result


def metrics(rows, prompts):
    out = summarize_generation(rows)
    out['japanese_validity'] = float(np.mean([r['character_valid'] for r in rows]))
    out['topic_retention_proxy'] = float(np.mean([len(set(r['ids']) & set(p['prefix_ids']))/max(1,len(set(r['ids']))) for r,p in zip(rows,prompts)]))
    for n in range(1,5): out[f'repetition_{n}'] = float(np.mean([ngram_repetition(r['ids'],n) for r in rows]))
    out['unique_token_ratio'] = float(np.mean([len(set(r['ids']))/max(1,len(r['ids'])) for r in rows]))
    out['loop_onset'] = float(np.mean([r['loop']['loop_onset'] or 129 for r in rows]))
    out['loop_cycle_length'] = float(np.mean([r['loop']['loop_length'] or 0 for r in rows]))
    traces = [s for r in rows for s in r['trace']]
    if traces:
        out['entropy'] = float(np.mean([s['entropy'] for s in traces]))
        out['top1_top2_margin'] = float(np.mean([s['top1_top2_margin'] for s in traces]))
    return out


def evaluate(arm, seed):
    manifest = resolver_gate(); spec = read(PREREG)
    for path, sha in spec['source_sha256'].items(): assert file_sha256(ROOT/path)==sha
    target = OUT / 'raw' / f'{arm}-seed-{seed}.json'
    if target.exists():
        done = read(target)
        assert done['complete'] and done['preregistration_sha256']==file_sha256(PREREG)
        assert file_sha256(evaluated_checkpoint(arm,seed))==done['checkpoint_sha256']
        print(f'Reuse completed {arm}/{seed}', flush=True); return
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    path = evaluated_checkpoint(arm,seed)
    known = next(r for r in manifest['rows'] if Path(r['destination'])==path)
    sha = file_sha256(path); assert sha==known['sha256']
    payload = torch.load(path, map_location='cpu', weights_only=False)
    integrity = verify_payload(payload, seed, payload['tokens_processed'], payload.get('experimental_lr', 1e-4))
    assert integrity['pass']
    del payload; gc.collect()
    cool = cooldown(); assert cool['target_reached'], 'Thermal cooldown gate blocked'
    monitor = Monitor(2); monitor.start()
    try:
        model = load_model(path, torch.device('cuda'))
        assert all(p.dtype==torch.float32 for p in model.parameters())
        tok, val = data(); prompts = spec['prompts']; prefix = [p['prefix_ids'] for p in prompts]
        terminal = eos_rows(model,tok,val,spec['terminal_positions'])
        nonterminal = eos_rows(model,tok,val,spec['nonterminal_positions'])
        print(f'{arm}/{seed}: EOS completed, sampling', flush=True)
        samples = {}
        for base in spec['sampling_rng_bases']:
            rows = generate_batch(model,tok,prefix,SAMPLE_T07,[base+i for i in range(100)],64,trace=False)
            for i,row in enumerate(rows):
                row.update({'prompt_id':prompts[i]['id'], 'sampling_seed':base+i,
                    'topic_retention_proxy':len(set(row['ids']) & set(prefix[i]))/max(1,len(set(row['ids'])))})
            samples[str(base)] = {'metrics':metrics(rows,prompts), 'rows':rows}
        # Same batch geometry/generators: exact reproducibility, including deterministic proxy.
        repeat = generate_batch(model,tok,prefix,SAMPLE_T07,[44000+i for i in range(100)],64,trace=False)
        deterministic = all(a['ids']==b['ids'] and all(a[k]==v for k,v in language_proxy(a['text'],eos_reached=a['eos_reached']).items()) for a,b in zip(samples['44000']['rows'],repeat))
        assert deterministic, 'Generation/metric determinism failure'
        print(f'{arm}/{seed}: sampling completed, greedy', flush=True)
        greedy = generate_batch(model,tok,prefix,GREEDY,list(range(100)),128,trace=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    finally:
        thermal = monitor.finish()
    assert file_sha256(path)==sha
    result = {'complete':True,'arm':arm,'seed':seed,'device':'cuda','dtype':'float32','training':False,
        'checkpoint':str(path),'checkpoint_sha256':sha,'checkpoint_unchanged':True,'integrity':integrity,
        'preregistration_sha256':file_sha256(PREREG),'terminal':terminal,'nonterminal':nonterminal,
        'sampling':samples,'determinism':deterministic,'greedy':{'metrics':metrics(greedy,prompts),'rows':greedy},
        'cooldown':cool,'thermal':thermal}
    new_json(target,result)
    print(f'{arm}/{seed}: complete; thermal={thermal.get("thermal_classification")}',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze','evaluate'])
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action=='freeze': freeze()
    else:
        if not args.arm or args.seed is None: parser.error('evaluate requires --arm and --seed')
        evaluate(args.arm,args.seed)
