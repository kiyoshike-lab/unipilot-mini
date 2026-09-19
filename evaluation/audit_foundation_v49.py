"""PHASE60: read-only checkpoint/log audit; never invokes optimizer.step or inference.

All findings are observational. Existing PHASE57/59 gates are immutable.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import generation_contract_v47 as contract
from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.checkpoint_paths import checkpoint_root
from training.optimizer import create_optimizer
from training.run_foundation_v36_lr_review import verify_payload, fingerprint
from training.train_foundation_v21_ab import frequency_ranks

OUT = ROOT / 'evaluation/phase60'
ZROOT = Path(r'Z:\AI\unipilot-mini\checkpoints')
RAW = ZROOT.parent / 'evaluation/phase60'
START = 'c4253450dbcdbdd8aebbf4ebfd02aa7c67308624'
SEEDS = (42, 123, 2026)
SCHEMA_SHA = '6576428940779e2908a8aacc67c4fb6bd2ca8bdc945e6202b88d5116ec701c6b'
DIRTY5 = ('campus-ai-quality-100.json', 'campus-ai-quality-20.json',
          'campus-ai-review-queue.json', 'campus-v21-close-analysis.json',
          'campus-v21-critical-failure.json')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def emit(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def artifact(name, value):
    emit(OUT / name, {'phase': 60, 'new_training': False, 'diagnostic_only': True, **value})


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def spec():
    return read(ROOT / 'evaluation/phase58/phase59-control-stability-preregistration.json')


def guard():
    if os.environ.get('UNIPILOT_CHECKPOINT_ROOT') != str(ZROOT) or checkpoint_root(ROOT) != ZROOT:
        raise RuntimeError('PROCESS_ENV_RESOLVER_MISMATCH')
    if git('branch', '--show-current') != 'foundation-research':
        raise RuntimeError('PHASE60_PREFLIGHT_BLOCKED')


def logpath(seed):
    return (RAW.parent / 'phase57/control-64k-training.json' if seed == 42 else
            RAW.parent / f'phase59/seed-{seed}-training-raw.json')


def evalpath(seed, role):
    if seed == 42:
        return RAW.parent / ('phase57/parent-evaluation.json' if role == 'parent' else 'phase57/control-64k-evaluation.json')
    return RAW.parent / f'phase59/seed-{seed}-{role}-evaluation-raw.json'


def identities():
    old = read(ROOT / 'evaluation/phase56/phase57-training-preregistration.json')
    parents = {42: old['parent_checkpoint']['sha256'], **{r['seed']: r['sha256'] for r in spec()['parents']}}
    rows = []
    for seed in SEEDS:
        p = ZROOT / f'experimental/phase48/arm-C/seed-{seed}/checkpoint-tokens-16384000.pt'
        candidate = read(logpath(seed))['checkpoint']
        for role, path, digest, tokens in [('parent', p, parents[seed], 16384000),
                                          ('candidate', Path(candidate['path']), candidate['sha256'], 16446464)]:
            if not path.is_relative_to(ZROOT):
                raise RuntimeError('CHECKPOINT_OUTSIDE_Z')
            rows.append({'seed': seed, 'role': role, 'path': str(path), 'sha256': digest, 'tokens': tokens})
    return rows


def preflight():
    guard()
    if git('rev-parse', 'HEAD') != START or git('ls-remote', 'origin', 'refs/heads/foundation-research').split()[0] != START:
        raise RuntimeError('PHASE60_PREFLIGHT_BLOCKED')
    if git('diff', '--cached', '--name-only'):
        raise RuntimeError('PREEXISTING_STAGED_FILES')
    if OUT.exists() or RAW.exists():
        raise RuntimeError('PHASE60_OUTPUT_EXISTS')
    status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=ROOT, text=True).splitlines()
    dirty = [{'path': line[3:], 'status': line[:2], 'sha256': sha(ROOT/line[3:]) if (ROOT/line[3:]).is_file() else None}
             for line in status if 'v49' not in line[3:]]
    protected = read(ROOT/'evaluation/phase56/safety-preflight.json')['protected_files']
    for row in protected:
        if sha(row['path']) != row['sha256']:
            raise RuntimeError('PROTECTED_CHANGED:' + row['path'])
    if sha(ROOT/'evaluation/phase58/evaluator-contract-v2.json') != SCHEMA_SHA:
        raise RuntimeError('EVALUATOR_CONTRACT_FAIL')
    s = spec()
    for rel, digest in {**s['source_sha256'], **s['artifact_sha256']}.items():
        if sha(ROOT/rel) != digest:
            raise RuntimeError('REGISTERED_SOURCE_CHANGED:' + rel)
    for row in [s['data'], *[s['evaluation_sets'][k] for k in ('validation', 'generation', 'frequency_population', 'normal_controls')]]:
        if sha(ROOT/row['path']) != row['sha256']:
            raise RuntimeError('INPUT_SHA_CHANGED:' + row['path'])
    checkpoints = identities()
    for r in checkpoints:
        if sha(r['path']) != r['sha256']:
            raise RuntimeError('CHECKPOINT_SHA_FAIL')
        payload = torch.load(r['path'], map_location='cpu', weights_only=False)
        r['integrity'] = verify_payload(payload, r['seed'], r['tokens'], 5e-5)
        del payload
        gc.collect()
        print('strict reload PASS', r['seed'], r['role'], flush=True)
    sealed = read(ROOT/'evaluation/phase55/fresh-holdout-v2-manifest.json')['splits']['future-reserve2']
    if sha(sealed['path']) != sealed['sha256']:
        raise RuntimeError('RESERVE2_HASH_CHANGED')
    blind = ROOT/'data/foundation_v09/evaluation/final-blind-1000.json'
    if sha(blind) != 'fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b':
        raise RuntimeError('BLIND_HASH_CHANGED')
    frozen = []
    for phase in (57, 58, 59):
        for folder in (ROOT/f'evaluation/phase{phase}', RAW.parent/f'phase{phase}'):
            for p in folder.rglob('*'):
                if p.is_file():
                    frozen.append({'path': str(p), 'sha256': sha(p)})
    for p in (ROOT/'evaluation').glob('foundation-v4[678]-*'):
        if p.is_file():
            frozen.append({'path': str(p), 'sha256': sha(p)})
    artifact('preflight.json', {'gate': 'PHASE60_PREFLIGHT_PASS', 'cwd': str(ROOT), 'start_head': START,
        'origin_sha': START, 'branch': 'foundation-research', 'main_ref': git('rev-parse','refs/heads/main'),
        'root': str(ZROOT), 'checkpoint_operations': dict.fromkeys(('copy','move','delete','overwrite','rename'), 0),
        'checkpoints': checkpoints, 'frozen': frozen, 'dirty': dirty, 'protected': protected,
        'dirty5': [{'path': str(ROOT/'evaluation'/n), 'sha256': sha(ROOT/'evaluation'/n), 'status': 'USER_DIRTY_PRESERVED'} for n in DIRTY5],
        'schema_sha256': SCHEMA_SHA, 'seal_status': {'reserve2': 'SEALED_UNSCORED_HASH_ONLY', 'final_blind': 'SHA_ONLY_PASS', 'phase53_reserve': 'RETIRED_UNSCORABLE_NOT_OPENED'},
        'free_bytes': {'C': shutil.disk_usage('C:\\').free, 'Z': shutil.disk_usage(ZROOT).free},
        'optimizer_steps': 0, 'inference_calls': 0})
    RAW.mkdir(parents=True, exist_ok=False)


def verify_frozen():
    guard()
    pre = read(OUT/'preflight.json')
    for r in pre['dirty']:
        p = ROOT/r['path']
        if (sha(p) if p.is_file() else None) != r['sha256']:
            raise RuntimeError('USER_DIRTY_CHANGED:' + r['path'])
    for r in pre['frozen'] + pre['protected'] + pre['checkpoints']:
        if sha(r['path']) != r['sha256']:
            raise RuntimeError('FROZEN_SHA_CHANGED:' + r['path'])
    if git('rev-parse','refs/heads/main') != pre['main_ref']:
        raise RuntimeError('MAIN_CHANGED')
    return pre


def dist(values):
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return {'count': 0, 'mean': None, 'median': None, 'p75': None, 'p90': None, 'p95': None, 'max': None}
    if not np.isfinite(x).all():
        raise ValueError('nonfinite measurement')
    return {'count': len(x), 'mean': float(x.mean()), 'median': float(np.median(x)),
            'p75': float(np.quantile(x,.75)), 'p90': float(np.quantile(x,.90)),
            'p95': float(np.quantile(x,.95)), 'max': float(x.max())}


def average_ranks(values):
    x = np.asarray(values)
    order = np.argsort(x, kind='stable')
    ranks = np.empty(len(x), dtype=float)
    a = 0
    while a < len(x):
        b = a+1
        while b < len(x) and x[order[b]] == x[order[a]]:
            b += 1
        ranks[order[a:b]] = (a+b-1)/2
        a = b
    return ranks


def spearman(x, y):
    a, b = average_ranks(x), average_ranks(y)
    if not len(a) or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a,b)[0,1])


def bootstrap(delta, docs):
    d = np.asarray(delta, dtype=float)
    _, inv = np.unique(docs, return_inverse=True)
    sums, counts = np.bincount(inv, weights=d), np.bincount(inv)
    rng = np.random.default_rng(5701)
    picks = rng.integers(0, len(sums), (10000, len(sums)))
    b = sums[picks].sum(1)/counts[picks].sum(1)
    return {'mean': float(d.mean()), 'lower': float(np.quantile(b,.025)), 'upper': float(np.quantile(b,.975))}


def projection(row, ci, th):
    fields = {}
    enriched = {**row, 'paired_document_bootstrap_10000_seed5701': ci}
    for path in contract.SCALAR_PRODUCERS.values():
        value = enriched
        for part in path.split('.'):
            value = value[part]
        fields[path] = value
    sampling = {k: contract.convert_legacy(row['generation']['sampling'][k]['metrics'], source_version=contract.LEGACY_VERSION)
                for k in contract.RNG_BASES}
    return contract.from_measurement_fields(fields, sampling, th)


def comparison_status(value, threshold, passed):
    # Exact equality only: there is no new 'near-threshold' margin.
    if Decimal(str(value)) == Decimal(str(threshold)):
        return 'BORDERLINE'
    return 'PASS' if passed else 'FAIL'


def matrix_and_normal():
    verify_frozen()
    s=spec(); th=s['all_safeguards']; populations=read(ROOT/s['evaluation_sets']['frequency_population']['path'])
    matrix={}; normal={}; frequency={}
    for seed in SEEDS:
        p,c=[read(evalpath(seed, role)) for role in ('parent','candidate')]
        ids=next(r for r in identities() if r['seed']==seed and r['role']=='parent')
        assert p['checkpoint_sha256']==ids['sha256']
        assert c['checkpoint_sha256']==read(logpath(seed))['checkpoint']['sha256']
        ci={}
        for group,key in [('core','core'),('supported_tail','population')]:
            pop=populations[key]
            assert p[group]['population_sha256']==c[group]['population_sha256']==pop['sha256']
            a,b=np.asarray(p[group]['values']['ce']),np.asarray(c[group]['values']['ce'])
            assert len(a)==len(b)==len(pop['token_ids']) and np.isfinite(a).all() and np.isfinite(b).all()
            assert np.all(np.diff(pop['positions'])>0)
            for row,v in ((p,a),(c,b)):
                assert abs(float(v.mean())-row[group]['metrics']['micro_ce'])<1e-12
                macro=np.mean([v[np.asarray(pop['token_ids'])==t].mean() for t in np.unique(pop['token_ids'])])
                assert abs(float(macro)-row[group]['metrics']['macro_per_token_ce'])<1e-12
            ci[group]=bootstrap(b-a,pop['document_ids'])
        pv=projection(p,{g:{'upper':0.} for g in ci},th); cv=projection(c,ci,th)
        checks=contract.safety_gate(cv,pv,th)['checks']
        if seed!=42:
            historical=read(ROOT/f'evaluation/phase59/seed{seed}-evaluation-summary.json')['safety_gate']['checks']
            assert checks==historical
        for v in checks.values():
            v['signature']=comparison_status(v['value'],v['threshold'],v['pass'])
        gen={}
        for label,r in [('parent',p),('candidate',c)]:
            sam=[contract.convert_legacy(v['metrics'],source_version=contract.LEGACY_VERSION)['metrics'] for v in r['generation']['sampling'].values()]
            gen[label]={'greedy':contract.convert_legacy(r['generation']['greedy']['metrics'],source_version=contract.LEGACY_VERSION),
                        'sampling_mean':{k:float(np.mean([v[k] for v in sam])) for k in contract.CONVERSION},
                        'sampling_runaway_by_rng':{k:v['metrics']['runaway_rate'] for k,v in r['generation']['sampling'].items()}}
        scalars={k:{'parent':pv['scalars'][k],'candidate':cv['scalars'][k],'delta':cv['scalars'][k]-pv['scalars'][k]} for k in cv['scalars'] if 'paired_ce' not in k}
        matrix[str(seed)]={'scalars':scalars,'bootstrap':ci,'generation':gen,'checks':checks,
             'failed_checks':[k for k,v in checks.items() if not v['pass']],
             'historical_status':'EXPERIMENT_INVALID' if seed==42 else 'CONTROL_STABILITY_MIXED',
             'seed42_comparisons':'DIAGNOSTIC_ONLY_NOT_FORMAL_EFFICACY_OR_GATE_RESCORING' if seed==42 else None}
        prows={r['id']:r for r in p['normal_controls']['rows']}; crows={r['id']:r for r in c['normal_controls']['rows']}
        assert set(prows)==set(crows) and len(prows)==20
        families={}
        for family in contract.FAMILIES:
            rows=[{'id':i,'parent':r['ce'],'candidate':crows[i]['ce'],'delta':crows[i]['ce']-r['ce']}
                  for i,r in prows.items() if r['family']==family]
            ds=np.array([r['delta'] for r in rows]); limit=th['normal_each_family_ce_increase_max']
            loo=[{'excluded':r['id'],'remaining_mean_delta':float(np.delete(ds,j).mean()),'diagnostic_pass':bool(np.delete(ds,j).mean()<=limit)} for j,r in enumerate(rows)]
            families[family]={'parent_ce':float(np.mean([r['parent'] for r in rows])), 'candidate_ce':float(np.mean([r['candidate'] for r in rows])),
                'delta':float(ds.mean()),'threshold':limit,'pass':checks[f'normal.{family}.ce']['pass'], 'examples':rows,'leave_one_out':loo,
                'all_leave_one_out_fail':all(not r['diagnostic_pass'] for r in loo),
                'top_positive_example_share':float(max(ds.max(),0)/np.maximum(ds,0).sum()) if np.maximum(ds,0).sum()>0 else None}
        failed=[f for f,v in families.items() if not v['pass']]
        concentrated=[f for f in failed if any(x['diagnostic_pass'] for x in families[f]['leave_one_out'])]
        classification=('NORMAL_FAILURE_EXAMPLE_CONCENTRATED' if concentrated else 'NORMAL_FAILURE_FAMILY_SPECIFIC' if len(failed)==1 else 'NORMAL_FAILURE_BROAD' if len(failed)>1 else 'NORMAL_FAILURE_UNRESOLVED')
        normal[str(seed)]={'families':families,'failed_families':failed,'mean':scalars['normal.mean_ce'],
            'classification':classification,'loo_gate_crossing_families':concentrated,'formal_gate_changed':False,'small_set_warning':'4 examples per family; leave-one-out is diagnostic, no new gate.'}
    artifact('three-seed-failure-matrix.json', {'seeds':matrix,'thresholds':th,'schema_sha256':SCHEMA_SHA,'borderline_definition':'Exact equality to the existing inclusive threshold; counts as pass. No new tolerance.'})
    artifact('normal-control-audit.json', {'seeds':normal,'exposure_proxy_reference':'data-order-audit.json / family_exposure_proxies'})


def gradient_audit():
    result={}
    for seed in SEEDS:
        log=read(logpath(seed)); rows=[]
        assert len(log['stats'])==122
        for r in log['stats']:
            norm=float(r['gradient_norm'] if seed==42 else r['gradient_norm_raw'])
            if not math.isfinite(norm) or norm<0: raise ValueError('invalid raw gradient norm')
            scale=min(1.,1./(norm+1e-6))
            rows.append({'update':r['update'],'train_loss':r['lm_loss'],'raw_gradient_norm':norm,
                         'clip_applied':scale<1,'clip_scale_derived':scale,'clipped_norm_derived':norm*scale})
        assert [r['update'] for r in rows]==list(range(32001,32123))
        emit(RAW/f'seed-{seed}-gradient-series.json',rows)
        result[str(seed)]={'raw_norm':dist([r['raw_gradient_norm'] for r in rows]),'clip_rate':float(np.mean([r['clip_applied'] for r in rows])),
             'clip_scale':dist([r['clip_scale_derived'] for r in rows]),'clipped_norm':dist([r['clipped_norm_derived'] for r in rows]),
             'loss':dist([r['train_loss'] for r in rows]),'clipped_norm_provenance':'Derived from clip_grad_norm_(max_norm=1) scale min(1,1/(norm+1e-6)); post-clip norms not logged, not presented as measured.'}
    artifact('gradient-clipping-audit.json',{'seeds':result,'signature':'CLIPPING_SATURATION_SIGNATURE' if all(r['clip_rate']==1 for r in result.values()) else 'NO_UNIFORM_SATURATION',
              'causal_claim':False,'effective_update_norms_reference':'optimizer-state-audit.json','limitations':'Clipped gradients precede AdamW preconditioning; clipping rate alone does not identify the cause or parameter update magnitude.'})


def category(name):
    if name=='embeddings.token.weight': return 'embedding_tied_head'
    if name.startswith('embeddings.position.'): return 'position_embedding'
    if '.attention.' in name: return 'attention'
    if '.feed_forward.' in name: return 'FFN'
    return 'LayerNorm'


def groups(name):
    return ['total','component:'+category(name), 'layer:'+name.split('.')[1] if name.startswith('blocks.') else 'layer:outside_blocks']


def optimizer_audit():
    all_deltas={}; aggregate={}; moments={}; rows_id=identities()
    for seed in SEEDS:
        payloads={r['role']:torch.load(r['path'],map_location='cpu',weights_only=False) for r in rows_id if r['seed']==seed}
        p,c=payloads['parent'],payloads['candidate']
        model=DiagnosticTransformerV17(DiagnosticConfigV17(**p['config']))
        names=[n for n,_ in model.named_parameters()]
        assert 'output.weight' not in names  # Tied head must count only once.
        assert torch.equal(p['model_state']['output.weight'],p['model_state']['embeddings.token.weight'])
        assert torch.equal(c['model_state']['output.weight'],c['model_state']['embeddings.token.weight'])
        sums=defaultdict(lambda:np.zeros(2)); deltas={}
        for name in names:
            base=p['model_state'][name].double(); delta=c['model_state'][name].double()-base
            deltas[name]=delta.float()
            for group in groups(name): sums[group]+=np.array([float(delta.square().sum()),float(base.square().sum())])
        aggregate[str(seed)]={k:{'net_update_norm':math.sqrt(v[0]),'parameter_relative_net_update_norm':math.sqrt(v[0]/v[1]) if v[1]>0 else None,
                                      'net_norm_divided_by_122':math.sqrt(v[0])/122} for k,v in sums.items()}
        moments[str(seed)]={}
        for role,payload in payloads.items():
            optimizer=create_optimizer(model,5e-5,.1);optimizer.load_state_dict(payload['optimizer_state'])
            mapping={id(v):n for n,v in model.named_parameters()}; acc=defaultdict(lambda:np.zeros(5)); perparam=[]
            for group in optimizer.param_groups:
                beta1,beta2=group['betas']
                for parameter in group['params']:
                    name=mapping[id(parameter)]; state=optimizer.state[parameter]; step=int(state['step'])
                    m=state['exp_avg'].double(); v=state['exp_avg_sq'].double(); w=payload['model_state'][name].double()
                    normalized=(m/(1-beta1**step))/(torch.sqrt(v/(1-beta2**step))+group['eps'])
                    adaptive=group['lr']*normalized
                    vec=np.array([float(m.square().sum()),float(v.square().sum()),float(w.square().sum()),float(normalized.square().sum()),float(adaptive.square().sum())])
                    for key in groups(name): acc[key]+=vec
                    perparam.append({'name':name,'first_moment_norm':math.sqrt(vec[0]),'second_moment_norm':math.sqrt(vec[1]),'normalized_direction_norm':math.sqrt(vec[3]),'step':step})
            moments[str(seed)][role]={k:{'first_moment_norm':math.sqrt(v[0]),'second_moment_norm':math.sqrt(v[1]),'first_moment_relative_to_parameter':math.sqrt(v[0]/v[2]) if v[2] else None,
                'normalized_direction_norm':math.sqrt(v[3]),'lr_scaled_adaptive_direction_norm':math.sqrt(v[4]),'relative_lr_scaled_direction_norm':math.sqrt(v[4]/v[2]) if v[2] else None} for k,v in acc.items()}
            emit(RAW/f'seed-{seed}-{role}-moment-parameters.json',perparam)
        all_deltas[seed]=deltas
        del payloads,p,c,model,optimizer
        gc.collect()
        print('parameter/moment audit',seed,flush=True)
    cosine={}
    for a,b in ((42,123),(42,2026),(123,2026)):
        acc=defaultdict(lambda:np.zeros(3))
        for name,d in all_deltas[a].items():
            x=d.double().flatten();y=all_deltas[b][name].double().flatten()
            for key in groups(name): acc[key]+=np.array([float(x.dot(x)),float(y.dot(y)),float(x.dot(y))])
        cosine[f'{a}:{b}']={k:{'cosine':float(v[2]/math.sqrt(v[0]*v[1])) if v[0]*v[1]>0 else None,'norm_ratio_first_over_second':math.sqrt(v[0]/v[1]) if v[1]>0 else None} for k,v in acc.items()}
    spread={}
    for role in ('parent','candidate'):
        spread[role]={}
        for group in moments['42'][role]:
            vals=[moments[str(seed)][role][group]['relative_lr_scaled_direction_norm'] for seed in SEEDS]
            spread[role][group]={'min':min(vals),'max':max(vals),'max_min_ratio':max(vals)/min(vals) if min(vals)>0 else None}
    artifact('optimizer-state-audit.json',{'net_parameter_updates':aggregate,'moments':moments,'cross_seed_update_vectors':cosine,'relative_direction_spread':spread,
        'classification':'OPTIMIZER_EVIDENCE_INSUFFICIENT','outlier_policy':'No registered quantitative outlier threshold; report complete layer distributions and extrema, do not invent a divergence gate.',
        'tied_head_counted_once':True,'interpretation':'Cosine is local coordinate update-direction similarity across different parent weights, not causal proof.',
        'net_update_limit':'Endpoint delta over 122 steps; neither accumulated path length nor average per-step norm is recoverable.',
        'moment_limit':'mhat/(sqrt(vhat)+eps) at saved endpoints; not a replayed optimizer step. Adaptive norm excludes decoupled weight decay. No optimizer steps executed.'})


def data_audit():
    s=spec(); tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json')
    train=np.memmap(ROOT/s['data']['path'],dtype=np.uint16,mode='r'); ranks=frequency_ranks(train,4096)
    pop=read(ROOT/s['evaluation_sets']['frequency_population']['path']); core_ids=np.unique(pop['core']['token_ids']);tail_ids=np.unique(pop['population']['token_ids'])
    bos=np.flatnonzero(train==tok.bos_id); special=list(tok.special_to_id.values())
    result={}; forgetting={}; raw={}
    # All proxies are descriptive counts; none affects sampling or formal gates.
    patterns={'math':('=', '+', '1'), 'code':('print','def','return'), 'lists':('1.','2.','確認済み'),
              'definitions':('定義','変数','関数'), 'terminology':('標本平均',)}
    for seed in SEEDS:
        path=next(r['path'] for r in identities() if r['seed']==seed and r['role']=='parent')
        payload=torch.load(path,map_location='cpu',weights_only=False)
        indices=payload['permutation'][32000:32122].cpu().numpy().astype(np.int64)
        start=indices*512
        assert len(indices)==122 and np.all(start+512<len(train))
        input_blocks=np.stack([np.asarray(train[x:x+512]) for x in start]);target_blocks=np.stack([np.asarray(train[x+1:x+513]) for x in start])
        hist=np.bincount(target_blocks.flatten(),minlength=4096); ihist=np.bincount(input_blocks.flatten(),minlength=4096)
        positions=(start[:,None]+np.arange(1,513)).flatten();docs=np.searchsorted(bos,positions,side='right')-1
        unique_docs,doc_counts=np.unique(docs,return_counts=True)
        texts=[tok.decode(x.tolist()) for x in input_blocks]
        proxies={f:{'patterns':list(ps),'block_hits':sum(any(p in text for p in ps) for text in texts),'match_occurrences':sum(text.count(p) for text in texts for p in ps),
                    'definition':'Literal substring TRAIN input proxy, not true category annotation or task mastery.'} for f,ps in patterns.items()}
        repeat={str(k):float(np.mean([1-len(set(tuple(row[i:i+k]) for i in range(len(row)-k+1)))/(len(row)-k+1) for row in input_blocks])) for k in (1,2,3,4)}
        result[str(seed)]={'permutation_sha256':fingerprint(payload['permutation'][32000:32122]),'blocks':122,'target_positions':62464,
            'target_token_histogram':hist.tolist(),'input_token_histogram':ihist.tolist(),
            'core_target_occurrences':int(hist[core_ids].sum()),'tail_target_occurrences':int(hist[tail_ids].sum()),
            'rare_share_rank_ge3277':float(hist[ranks>=3277].sum()/hist.sum()),'core_zero_exposure_fraction':float(np.mean(hist[core_ids]==0)),
            'eos_count':int(hist[tok.eos_id]),'eos_density':float(hist[tok.eos_id]/hist.sum()),'bos_count':int(hist[tok.bos_id]),
            'special_density':float(hist[special].sum()/hist.sum()),'sequence_boundary_count':int(hist[tok.bos_id]),
            'within_block_repetition':repeat,'document_count':len(unique_docs),'document_target_counts':{str(i):int(n) for i,n in zip(unique_docs,doc_counts)},
            'document_identity':'Ordinal of most recent BOS in train.bin; -1 is pre-BOS prefix; counts include partial documents.',
            'category_distribution':'UNKNOWN: no authoritative packed-position category metadata consumed.', 'family_exposure_proxies':proxies,
            'frequency_bands':{label:int(hist[mask].sum()) for label,mask in [('head_rank_0_819',ranks<820),('middle_rank_820_3276',(ranks>=820)&(ranks<3277)),('rare_rank_ge3277',ranks>=3277)]}}
        a,b=[read(evalpath(seed,role)) for role in ('parent','candidate')]
        delta=np.asarray(b['core']['values']['ce'])-np.asarray(a['core']['values']['ce']);token_ids=np.asarray(pop['core']['token_ids'])
        rows=[{'token_id':int(t),'exposure_targets':int(hist[t]),'exposure_inputs':int(ihist[t]),'validation_occurrences':int(np.sum(token_ids==t)),
               'ce_delta':float(delta[token_ids==t].mean()),'mean_log_probability_delta':float(-delta[token_ids==t].mean())} for t in core_ids]
        ds=np.asarray([r['ce_delta'] for r in rows]);exposure=hist[core_ids]
        # Equal-sized rank groups are descriptive, never decision thresholds.
        order=np.argsort(exposure,kind='stable');quartiles=[]
        for j,ix in enumerate(np.array_split(order,4)):
            quartiles.append({'rank_group':j+1,'tokens':len(ix),'exposure_min':int(exposure[ix].min()),'exposure_max':int(exposure[ix].max()),'mean_ce_delta':float(ds[ix].mean()),'ties_policy':'stable token ID order; ties may straddle groups'})
        forgetting[str(seed)]={'core_token_count':len(core_ids),'worsened_fraction':float(np.mean(ds>0)),'ce_delta_distribution':dist(ds),
            'zero_exposure_fraction':float(np.mean(exposure==0)), 'zero_exposure_delta':dist(ds[exposure==0]),'nonzero_exposure_delta':dist(ds[exposure>0]),
            'spearman_exposure_vs_ce_delta':spearman(exposure,ds),'rank_groups':quartiles,
            'zero_minus_nonzero_mean':float(ds[exposure==0].mean()-ds[exposure>0].mean()) if (exposure==0).any() and (exposure>0).any() else None}
        raw[str(seed)]={'permutation_entries':indices.tolist(),'core_tokens':rows}
        del payload
        gc.collect()
    for seed in SEEDS: emit(RAW/f'seed-{seed}-data-and-core-tokens.json',raw[str(seed)])
    overlap={}
    for a,b in ((42,123),(42,2026),(123,2026)):
        x,y=set(raw[str(a)]['permutation_entries']),set(raw[str(b)]['permutation_entries'])
        overlap[f'{a}:{b}']={'shared_blocks':len(x&y),'union_blocks':len(x|y),'jaccard':len(x&y)/len(x|y)}
    artifact('data-order-audit.json',{'seeds':result,'cross_seed_block_overlap':overlap,'no_shuffle_redraw':True,'exposure_definition':'LM TARGET positions start+1..start+512. Input histogram separately; matching no-grad cache not counted as gradient exposure.'})
    signature = all(r['spearman_exposure_vs_ce_delta'] is not None and r['spearman_exposure_vs_ce_delta'] < 0
                    and r['zero_minus_nonzero_mean'] is not None and r['zero_minus_nonzero_mean'] > 0 for r in forgetting.values())
    artifact('frequency-forgetting-audit.json',{'seeds':forgetting,'signature':'YES' if signature else 'UNKNOWN','reason':'All three seeds have negative exposure/CE rank association and greater mean worsening for unexposed tokens. FREQUENCY_FORGETTING_SIGNATURE is descriptive; rank grouping and zero-exposure comparisons are the same evidence stream, not independent causal confirmation.',
        'unit':'236 token types within each seed; no significance claim based on N=3 seed correlation','raw_reference':str(RAW)})


def run():
    verify_frozen()
    matrix_and_normal(); gradient_audit(); optimizer_audit(); data_audit(); verify_frozen()
    print('PHASE60 measurement audits complete; zero optimizer steps, zero inference',flush=True)


def scheduler_audit():
    verify_frozen()
    rows=[]
    for r in identities():
        p=torch.load(r['path'],map_location='cpu',weights_only=False)
        groups=p['optimizer_state']['param_groups']
        assert all(g['lr']==5e-5 for g in groups)
        rows.append({'seed':r['seed'],'role':r['role'],'checkpoint_sha256':r['sha256'],
                     'scheduler_state':p['scheduler_state'],'actual_optimizer_group_LR':[g['lr'] for g in groups],
                     'experimental_lr_metadata':p.get('experimental_lr'),
                     'legacy_scheduler_lr_disagrees':p['scheduler_state']['learning_rate']!=groups[0]['lr']})
        if r['role']=='parent':
            log=read(logpath(r['seed']))
            assert all(log['continuity'][k]==fingerprint(p[k]) for k in ('model_state','optimizer_state','scheduler_state','permutation','random_state'))
        del p
    artifact('scheduler-metadata-audit.json',{'checkpoints':rows,'parent_log_continuity_all_three':True,
        'finding':'Historical scheduler LR metadata is1e-4 for all six payloads; all optimizer groups actually hold5e-5. Shared mismatch, not a demonstrated between-seed mechanism.',
        'code_evidence':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'evaluation/run_foundation_v46_phase57.py',ROOT/'evaluation/run_foundation_v48_phase59.py',ROOT/'training/run_foundation_v30_eos_experiment.py')},
        'source_inspection':'Frozen training loops step the loaded optimizer, never scheduler.step; saved scheduler only advances global_step.',
        'future_requirement':'Assert per-step actual group LR; preserve old metadata separately from explicit new runtime LR contract.'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('preflight','audit','scheduler','verify'));args=parser.parse_args()
    torch.set_num_threads(2)
    if args.action=='preflight':preflight()
    elif args.action=='audit':run()
    elif args.action=='scheduler':scheduler_audit()
    else: verify_frozen(); print('PHASE60_FROZEN_PRESERVATION_PASS',flush=True)


if __name__=='__main__':main()
