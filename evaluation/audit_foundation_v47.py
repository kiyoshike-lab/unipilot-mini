"""PHASE58: read-only forensic analysis and contract dry run; no optimizer step.

Outputs are exclusively new PHASE58 files. PHASE57 remains INVALID forever.
Reserve/Blind bodies are never parsed; only the approved hashes are read.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from copy import deepcopy
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import generation_contract_v47 as contract
from evaluation.diagnose_foundation_v40 import resolver_gate, metrics
from evaluation.diagnose_foundation_v29_generation import generate_batch,load_model
from evaluation.evaluate_foundation_v33_context_gate import GREEDY,SAMPLE_T07
from evaluation.generation_prefix_objective_v45 import cycle_negatives
from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import DiagnosticTransformerV17,DiagnosticConfigV17
from training.checkpoint_paths import checkpoint_root,existing_checkpoint_path
from training.run_foundation_v36_lr_review import verify_payload,fingerprint
from training.run_foundation_v35_thermal_gate import cooldown,Monitor,query_gpu
from training.train_foundation_v21_ab import random_state,restore_random_state,frequency_ranks

OUT=ROOT/'evaluation/phase58'
RAW=Path(r'Z:\AI\unipilot-mini\evaluation\phase58')
OLDRAW=RAW.parent/'phase57'
START='e17d53868ce67e4b55f72e23fb2c093948c0b460'
ZROOT=Path(r'Z:\AI\unipilot-mini\checkpoints')
VERSION='phase58-audit-v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def emit(name,value):write(OUT/name,{'schema_version':VERSION,'phase':58,'new_training':False,**value})
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def spec():return read(ROOT/'evaluation/phase56/phase57-training-preregistration.json')
def thresholds():return spec()['quality_safeguards_vs_each_parent_and_control']
def own(rel):return rel.startswith('evaluation/phase58/') or 'v47' in rel
def preserved():
    rows=read(ROOT/'evaluation/phase56/safety-preflight.json')['protected_files']
    for r in rows:
        if sha(r['path'])!=r['sha256']:raise RuntimeError('PROTECTED_SHA_CHANGED:'+r['path'])
    return rows
def sealed():
    reserve=read(ROOT/'evaluation/phase55/fresh-holdout-v2-manifest.json')['splits']['future-reserve2']
    assert sha(reserve['path'])==reserve['sha256']
    blind=ROOT/'data/foundation_v09/evaluation/final-blind-1000.json'
    assert sha(blind)=='fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b'
    return {'reserve2_sha256':reserve['sha256'],'reserve2':'SEALED_UNSCORED_HASH_ONLY','phase53_reserve':'RETIRED_UNSCORABLE_NOT_OPENED','final_blind_sha256':sha(blind),'final_blind':'SHA_ONLY'}
def guard():
    if os.environ.get('UNIPILOT_CHECKPOINT_ROOT')!=str(ZROOT) or checkpoint_root(ROOT)!=ZROOT:raise RuntimeError('PROCESS_ENV_RESOLVER_MISMATCH')
    if git('branch','--show-current')!='foundation-research':raise RuntimeError('BRANCH_MISMATCH')
    preserved();sealed()
def check_frozen():
    guard();pre=read(OUT/'preflight.json')
    for r in pre['frozen_files']:
        assert sha(r['path'])==r['sha256'],r['path']
    for r in pre['dirty_files']:
        p=ROOT/r['path'];assert (sha(p) if p.is_file() else None)==r['sha256'],r['path']
    return pre
def payload_path(label):
    if label=='parent':return existing_checkpoint_path(ROOT,'experimental','phase48','arm-C','seed-42','checkpoint-tokens-16384000.pt')
    return existing_checkpoint_path(ROOT,'experimental','phase57','arm-'+('control' if label=='control' else 'A'),'seed-42','checkpoint-tokens-16446464.pt')


def preflight():
    guard();resolver_gate()
    assert git('rev-parse','HEAD')==START and git('ls-remote','origin','refs/heads/foundation-research').split()[0]==START
    assert not git('diff','--cached','--name-only')
    for p in (OUT,RAW):
        if p.exists():raise FileExistsError(p)
    freeze=list((ROOT/'evaluation/phase57').glob('*'))+list(OLDRAW.glob('*'))
    freeze += [ROOT/'evaluation/foundation-v46-anti-loop-summary.json',ROOT/'evaluation/foundation-v46-anti-loop-report.md',ROOT/'evaluation/run_foundation_v46_phase57.py',ROOT/'evaluation/phase56/phase57-training-preregistration.json']
    frozen=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in freeze if p.is_file()]
    if any(r['bytes']==0 or r['path'].endswith('.tmp') for r in frozen):raise RuntimeError('OLD_PARTIAL_ARTIFACT')
    status=git('status','--porcelain','--untracked-files=all').splitlines()
    dirty=[]
    # Preserve leading two porcelain status columns (git().strip removes first leading space).
    status=subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=ROOT,text=True).splitlines()
    for line in status:
        rel=line[3:]
        if not own(rel):
            p=ROOT/rel;dirty.append({'status':line[:2],'path':rel,'sha256':sha(p) if p.is_file() else None})
    checkpoints=[]
    expected=read(ROOT/'evaluation/phase55/integrity.json')['rows']
    for label in ('parent','control','A'):
        p=payload_path(label);assert p.is_relative_to(ZROOT)
        expected_sha=spec()['parent_checkpoint']['sha256'] if label=='parent' else read(OLDRAW/('control-64k-training.json' if label=='control' else 'A-64k-training.json'))['checkpoint']['sha256']
        assert sha(p)==expected_sha
        payload=torch.load(p,map_location='cpu',weights_only=False)
        integrity=verify_payload(payload,42,16384000 if label=='parent' else 16446464,5e-5)
        if label!='parent':
            assert payload['experimental'] and payload['not_canonical'] and payload['not_production']
        checkpoints.append({'label':label,'path':str(p),'sha256':expected_sha,'integrity':integrity})
        del payload;gc.collect()
    for seed in (123,2026):
        r=next(r for r in expected if r['arm']=='C' and r['seed']==seed)
        p=existing_checkpoint_path(ROOT,'experimental','phase48','arm-C',f'seed-{seed}','checkpoint-tokens-16384000.pt');assert p.is_relative_to(ZROOT) and sha(p)==r['sha256']
        payload=torch.load(p,map_location='cpu',weights_only=False)
        integrity=verify_payload(payload,seed,16384000,5e-5)
        rng=payload['random_state'];restore_random_state(rng,'cuda');assert fingerprint(random_state('cuda'))==fingerprint(rng)
        checkpoints.append({'label':f'candidate-parent-{seed}','seed':seed,'relative_path':str(p.relative_to(ZROOT)),'path':str(p),'sha256':r['sha256'],'integrity':integrity,'rng_roundtrip':True,'next_122_permutation_sha256':fingerprint(payload['permutation'][32000:32122])})
        del payload;gc.collect()
    for r in checkpoints:frozen.append({'path':r['path'],'sha256':r['sha256'],'bytes':Path(r['path']).stat().st_size})
    for row in [spec()['training_data']['base'],*[spec()['evaluation_sets'][k] for k in ('generation','validation','frequency_population','normal_controls')]]:
        assert sha(ROOT/row['path'])==row['sha256']
    dry={'schema_version':'phase58-contract-fixture-v1','purpose':'contract plumbing only, no model-quality decision','prompts':['日本の大学では','数学の基礎は'],'greedy_tokens':8,'sampling_tokens':8,'sampling_rng_bases':[44000,51000,52000],'synthetic_non_generation_metrics':True,'training':False}
    emit('preflight.json',{'gate':'PHASE58_PREFLIGHT_PASS','start_head':START,'branch':git('branch','--show-current'),'origin_match':True,'main_ref':git('rev-parse','refs/heads/main'),'protected_files':preserved(),'sealed':sealed(),'frozen_files':frozen,'dirty_files':dirty,'checkpoints':checkpoints,'checkpoint_copy_move_delete_overwrite':[0,0,0,0],'root':str(ZROOT),'C_free':shutil.disk_usage('C:\\').free,'Z_free':shutil.disk_usage(ZROOT).free})
    write(OUT/'dry-run-preregistration.json',dry)
    write(OUT/'evaluator-contract-v2.json',contract.schema_document(thresholds()))
    RAW.mkdir(parents=True,exist_ok=False)
    print('PHASE58 PREFLIGHT PASS; 5 checkpoint integrity checks; zero training',flush=True)


@torch.inference_mode()
def dry_run(output_name='evaluator-contract-audit.json'):
    check_frozen();fixture=read(OUT/'dry-run-preregistration.json')
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 2070 SUPER'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cool=cooldown();assert cool['target_reached']
    model=load_model(payload_path('parent'),torch.device('cuda'));tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json')
    prompts=[{'prefix_ids':[tok.bos_id]+tok.encode(s)} for s in fixture['prompts']]
    prefix=[p['prefix_ids'] for p in prompts];mon=Monitor();mon.start()
    parent=contract.fixture(thresholds());candidate=deepcopy(parent);canonical={}
    try:
        for base in fixture['sampling_rng_bases']:
            q=query_gpu();assert q['gpu_temperature_c']<80 and not q['hardware_thermal_slowdown']
            rows=generate_batch(model,tok,prefix,SAMPLE_T07,[base+i for i in range(len(prompts))],fixture['sampling_tokens'],trace=False)
            converted=contract.convert_legacy(metrics(rows,prompts),source_version=contract.LEGACY_VERSION)
            canonical[str(base)]=converted;candidate['sampling'][str(base)]=converted;parent['sampling'][str(base)]=deepcopy(converted)
        rows=generate_batch(model,tok,prefix,GREEDY,[0,1],fixture['greedy_tokens'],trace=True)
        greedy=contract.convert_legacy(metrics(rows,prompts),source_version=contract.LEGACY_VERSION)
        # Demonstrate matching no-grad eval forward has no RNG or buffer mutation.
        rng_before=fingerprint(random_state('cuda'));state_before=fingerprint(model.state_dict())
        x=torch.tensor([prefix[0]],device='cuda');model(x)
        no_grad={'rng_unchanged':rng_before==fingerprint(random_state('cuda')),'state_unchanged':state_before==fingerprint(model.state_dict()),'mode':'eval/no_grad','weights_updated':False}
        projected={path:candidate['scalars'][key] for key,path in contract.SCALAR_PRODUCERS.items()}
        candidate=contract.from_measurement_fields(projected,candidate['sampling'],thresholds())
        gate=contract.safety_gate(candidate,parent,thresholds());assert gate['gate']=='CONTROL_SAFETY_PASS'
    finally:
        thermal=mon.finish();del model;gc.collect();torch.cuda.empty_cache()
    assert all(no_grad[k] for k in ('rng_unchanged','state_unchanged')) and thermal['gpu_temperature_c_max']<80
    bad=deepcopy(candidate);bad['sampling']['44000']['metrics']['natural_japanese_proxy']=bad['sampling']['44000']['metrics'].pop('naturalness_rate')
    try:contract.safety_gate(bad,parent,thresholds())
    except contract.ContractError:rename_pass=True
    else:raise RuntimeError('MISMATCH_WAS_NOT_CAUGHT')
    coverage=contract.mapping(thresholds())
    emit(output_name,{'gate':'EVALUATOR_CONTRACT_READY','generation_schema_version':contract.VERSION,'contract_sha256':sha(OUT/'evaluator-contract-v2.json'),'dry_fixture_sha256':sha(OUT/'dry-run-preregistration.json'),'old_mismatch_reproduced':True,'rename_fail_closed':rename_pass,'all_safeguards_mapped':{'threshold_families':len(set(r['threshold_key'] for r in coverage)),'threshold_families_required':len(thresholds())-1,'concrete_comparisons':len(coverage)},'dry_run':'PASS','dry_run_quality_claim':False,'synthetic_non_generation_metrics':True,'generation':canonical,'greedy':greedy,'wrapper_to_gate':gate,'no_grad_matching_forward':no_grad,'cooldown':cool,'thermal':thermal})
    check_frozen();print('EVALUATOR_CONTRACT_READY; real CUDA tiny dry run PASS',flush=True)


def bootstrap(delta,docs,ids):
    _,di=np.unique(docs,return_inverse=True);_,ti=np.unique(ids,return_inverse=True)
    sums=np.bincount(di,weights=delta);counts=np.bincount(di);rng=np.random.default_rng(5701)
    micro=[];macro=[]
    for _ in range(10000):
        multiplicities=np.bincount(rng.integers(0,len(sums),len(sums)),minlength=len(sums))
        w=multiplicities[di];micro.append(float(np.dot(w,delta)/w.sum()))
        tc=np.bincount(ti,weights=w);ts=np.bincount(ti,weights=w*delta);macro.append(float(np.mean(ts[tc>0]/tc[tc>0])))
    return {'schema_version':'diagnostic-bootstrap-v1','unit':'paired document, fixed population','replicates':10000,'seed':5701,'micro_ci95':np.quantile(micro,[.025,.975]).tolist(),'macro_ci95':np.quantile(macro,[.025,.975]).tolist(),'formal_gate':False}


def contributions(delta,groups,ids,docs,counts,ranks):
    out=[]
    for group in np.unique(groups):
        mask=groups==group;tokens=np.unique(ids[mask]);row={'id':int(group),'count':int(mask.sum()),'ce_delta':float(delta[mask].mean()),'micro_contribution':float(delta[mask].sum()/len(delta)),'document_support':int(len(np.unique(docs[mask]))),'token_types':int(len(tokens))}
        if len(tokens)==1:row.update(train_count=int(counts[tokens[0]]),frequency_rank=int(ranks[tokens[0]]),frequency_bucket='bottom20pct' if ranks[tokens[0]]>=3277 else 'other')
        if np.array_equal(groups,ids):row['macro_contribution']=float(delta[mask].mean()/len(np.unique(ids)))
        out.append(row)
    return sorted(out,key=lambda r:r['micro_contribution'],reverse=True)


def parameter_deltas(parent,control,arm):
    model=DiagnosticTransformerV17(DiagnosticConfigV17(**parent['config']));names=[n for n,_ in model.named_parameters()]
    # named_parameters deduplicates the tied LM head; state_dict does not.
    tied=parent['config']['weight_tying'];assert tied and torch.equal(parent['model_state']['embeddings.token.weight'],parent['model_state']['output.weight'])
    buckets=defaultdict(lambda:np.zeros(5));individual=[]
    for name in names:
        p=parent['model_state'][name].double().reshape(-1);c=control['model_state'][name].double().reshape(-1)-p;a=arm['model_state'][name].double().reshape(-1)-p
        values=np.array([float(c.dot(c)),float(a.dot(a)),float((a-c).dot(a-c)),float(c.dot(a)),float(p.dot(p))])
        layer=name.split('.')[1] if name.startswith('blocks.') else 'outside_blocks'
        category='tied_embedding_and_lm_head' if name=='embeddings.token.weight' else 'position_embedding' if name.startswith('embeddings.') else 'attention' if '.attention.' in name else 'FFN' if '.feed_forward.' in name else 'LayerNorm'
        for key in ('total','layer:'+layer,'component:'+category):buckets[key]+=values
        individual.append({'name':name,'control_norm':math.sqrt(values[0]),'arm_norm':math.sqrt(values[1]),'difference_norm':math.sqrt(values[2])})
    result={k:{'control_update_norm':math.sqrt(v[0]),'arm_update_norm':math.sqrt(v[1]),'control_arm_difference_norm':math.sqrt(v[2]),'cosine_updates':float(v[3]/math.sqrt(v[0]*v[1])) if v[0]*v[1]>0 else None,'control_relative_to_parent':math.sqrt(v[0]/v[4]) if v[4]>0 else None} for k,v in buckets.items()}
    write(RAW/'parameter-deltas.json',{'schema_version':'parameter-delta-v1','diagnostic_only':True,'named_parameters':individual,'aggregates':result,'tied_head_counted_once':True})
    return {'tied_head_counted_once':True,'aggregates':result}


def base_candidates(generated,reference,special):
    """Diagnostic pre-veto denominator only; never creates a training mask."""
    result=[]
    for t,token in enumerate(generated):
        if token in special or t>=len(reference):continue
        for width in range(1,9):
            if t<4*width:continue
            cycle=generated[t-width:t]
            if any(x in special for x in cycle):continue
            if generated[t-4*width:t]==cycle*4 and token==cycle[0]:result.append((t,token));break
    return result


def forensic():
    check_frozen();reg=spec();pop=read(ROOT/reg['evaluation_sets']['frequency_population']['path'])['core']
    old={k:read(OLDRAW/(label+'-evaluation.json')) for k,label in [('parent','parent'),('control','control-64k'),('A','arm-a-64k')]}
    vals={k:np.array(r['core']['values']['ce'],float) for k,r in old.items()};positions=np.asarray(pop['positions']);ids=np.asarray(pop['token_ids']);docs=np.asarray(pop['document_ids'])
    val=np.memmap(ROOT/reg['evaluation_sets']['validation']['path'],dtype=np.uint16,mode='r');train=np.memmap(ROOT/reg['training_data']['base']['path'],dtype=np.uint16,mode='r');tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json')
    assert np.all(positions[:-1]<positions[1:]) and np.array_equal(val[positions],ids)
    assert all(len(x)==len(ids) and np.isfinite(x).all() for x in vals.values())
    reproduction={}
    for k,v in vals.items():
        micro=float(v.mean());macro=float(np.mean([v[ids==i].mean() for i in np.unique(ids)]));stored=old[k]['core']['metrics']
        assert old[k]['core']['population_sha256']==pop['sha256'] and abs(micro-stored['micro_ce'])<1e-12 and abs(macro-stored['macro_per_token_ce'])<1e-12
        reproduction[k]={'micro_ce':micro,'macro_ce':macro,'stored_exact_reproduction':True}
    delta=vals['control']-vals['parent'];counts=np.bincount(train,minlength=4096);ranks=frequency_ranks(train,4096)
    tokens=contributions(delta,ids,ids,docs,counts,ranks);documents=contributions(delta,docs,ids,docs,counts,ranks)
    positive_token=sum(max(0,r['micro_contribution']) for r in tokens);positive_doc=sum(max(0,r['micro_contribution']) for r in documents)
    concentration={'top20_token_positive_mass_share':sum(max(0,r['micro_contribution']) for r in tokens[:20])/positive_token,'top5_document_positive_mass_share':sum(max(0,r['micro_contribution']) for r in documents[:5])/positive_doc,'positive_occurrence_fraction':float(np.mean(delta>0)),'positive_token_fraction':sum(r['ce_delta']>0 for r in tokens)/len(tokens)}
    boot=bootstrap(delta,docs,ids)
    write(RAW/'core-contributions.json',{'schema_version':'core-diagnostic-v1','tokens':tokens,'documents':documents,'formal_gate':False})
    payloads={k:torch.load(payload_path(k),map_location='cpu',weights_only=False) for k in ('parent','control','A')}
    parameter=parameter_deltas(payloads['parent'],payloads['control'],payloads['A'])
    logs={k:read(OLDRAW/label) for k,label in [('control','control-64k-training.json'),('A','A-64k-training.json')]}
    continuity={}
    for name in ('model_state','optimizer_state','scheduler_state','permutation','random_state'):
        continuity[name]=all(logs[k]['continuity'][name]==fingerprint(payloads['parent'][name]) for k in logs)
    continuity['final_rng_equal']=fingerprint(payloads['control']['random_state'])==fingerprint(payloads['A']['random_state'])
    continuity['final_permutation_equal']=all(torch.equal(payloads['parent']['permutation'],payloads[k]['permutation']) for k in ('control','A'))
    trajectory={}
    for k,log in logs.items():
        rows=log['stats'];assert [r['update'] for r in rows]==list(range(32001,32123))
        trajectory[k]={'updates':len(rows),'lr':5e-5,'mean_lm_loss':float(np.mean([r['lm_loss'] for r in rows])),'first_16_lm_mean':float(np.mean([r['lm_loss'] for r in rows[:16]])),'last_16_lm_mean':float(np.mean([r['lm_loss'] for r in rows[-16:]])),'gradient_norm_mean':float(np.mean([r['gradient_norm'] for r in rows])),'gradient_norm_max':max(r['gradient_norm'] for r in rows),'clip_rate_norm_gt_1':float(np.mean([r['gradient_norm']>1 for r in rows])),'norm_gt10_count':sum(r['gradient_norm']>10 for r in rows),'gradient_kind':'LM' if k=='control' else 'combined; separate LM/aux gradients NOT_AVAILABLE'}
    blocks=payloads['parent']['permutation'][32000:32122].numpy().astype(int);ys=np.stack([train[i*512+1:i*512+513] for i in blocks]);stream=ys.flatten();stream_counts=np.bincount(stream,minlength=4096)
    bucket=lambda r:'top50pct' if r<2048 else 'mid30pct' if r<3277 else 'bottom20pct'
    order={'updates':122,'lm_targets':len(stream),'blocks_sha256':fingerprint(payloads['parent']['permutation'][32000:32122]),'EOS_density':float(np.mean(stream==tok.eos_id)),'train_EOS_density':float(counts[tok.eos_id]/len(train)),'adjacent_repetition_density':float(np.mean(ys[:,1:]==ys[:,:-1])),'frequency_buckets':{b:float(sum(stream_counts[i] for i in range(4096) if bucket(ranks[i])==b)/len(stream)) for b in ('top50pct','mid30pct','bottom20pct')},'core_occurrences':int(stream_counts[np.unique(ids)].sum()),'core_types_present':int(np.sum(stream_counts[np.unique(ids)]>0)),'category_proxy':'NOT_AVAILABLE: packed tokens have no reliable category metadata','causal_data_order_claim':False}
    core_ids=np.unique(ids);per_token_delta=np.array([delta[ids==i].mean() for i in core_ids]);cor=float(np.corrcoef(stream_counts[core_ids],per_token_delta)[0,1]);order['core_exposure_delta_pearson_descriptive']=cor
    classification=['CONTROL_DRIFT_TOKEN_CONCENTRATED' if concentration['top20_token_positive_mass_share']>=.5 else 'CONTROL_DRIFT_BROAD','CONTROL_DRIFT_SEED_LOCAL_POSSIBLE','CONTROL_DRIFT_CAUSE_UNRESOLVED']
    emit('control-drift-audit.json',{'diagnostic_only':True,'complete':True,'no_new_inference_for_core':True,'population_sha256':pop['sha256'],'geometry':'sorted positions, block=floor((position-1)/512), input [block*512:block*512+512], logits[position-start-1], original token IDs','evaluator_path_audit':'PHASE57 frequency_values matches frozen evaluate_foundation_v39_gate geometry; stored metrics reproduce exactly','reproduction':reproduction,'control_micro_delta':reproduction['control']['micro_ce']-reproduction['parent']['micro_ce'],'control_macro_delta':reproduction['control']['macro_ce']-reproduction['parent']['macro_ce'],'unchanged_margin':.1,'bootstrap':boot,'concentration':concentration,'top_tokens':tokens[:20],'top_documents':documents[:10],'parameter_delta':parameter,'trajectory':trajectory,'continuity':continuity,'data_order':order,'classification':classification,'causal_limit':'single seed, no counterfactual data-order or optimization experiment; concentration is descriptive, not causal identification'})
    cache=read(OLDRAW/'cache-raw.json')['episodes'];special=set(tok.special_to_id.values());aux=[r for r in logs['A']['stats'] if r['update']%8==0];dose=[]
    for row in aux:
        idx=(row['update']//8-1)%len(cache);ep=cache[idx];eligible=not ep['eos_reached'] and len(ep['generated'])==32
        pairs=cycle_negatives(ep['generated'],ep['reference'],special) if eligible else []
        assert pairs==[tuple(p) for p in ep['negative_pairs']] and len(pairs)==row['aux_negative_count']
        candidates=base_candidates(ep['generated'],ep['reference'],special) if eligible else []
        dose.append({'update':row['update'],'episode_index':idx,'negative_events':len(pairs),'masked_positions':len(set(t for t,_ in pairs)),'pre_veto_candidates':len(candidates),'vetoed':len(candidates)-len(pairs),'empty_mask':not pairs,'raw_auxiliary_loss':row['aux_loss'],'coefficient':.05,'weighted_contribution':row['aux_loss']*.05,'LM_loss':row['lm_loss'],'weighted_to_LM_ratio':row['aux_loss']*.05/row['lm_loss'],'combined_gradient_norm':row['gradient_norm']})
    write(RAW/'dose-by-update.json',{'schema_version':'intervention-dose-v1','diagnostic_only':True,'rows':dose})
    denominator=sum(r['pre_veto_candidates'] for r in dose);weighted=sum(r['weighted_contribution'] for r in dose)
    emit('intervention-dose-audit.json',{'diagnostic_only':True,'complete':True,'auxiliary_updates':len(aux),'unique_episodes_consumed':len(set(r['episode_index'] for r in dose)),'episode_indices':[r['episode_index'] for r in dose],'cache_index_rule':'(absolute_update//8-1)%128; starts at 32, not zero','registration_ambiguity':'registration says next frozen episode but did not freeze initial cursor; absolute-update implementation selected 32..46; do not repair or rerun','negative_events_used':sum(r['negative_events'] for r in dose),'mean_negative_events_per_aux_update':float(np.mean([r['negative_events'] for r in dose])),'masked_positions':sum(r['masked_positions'] for r in dose),'veto_denominator':denominator,'veto_rate':sum(r['vetoed'] for r in dose)/denominator if denominator else None,'veto_definition':'aligned target or reference snippet veto among nonspecial cycle candidates, eligible consumed episodes only','empty_mask_rate':float(np.mean([r['empty_mask'] for r in dose])),'raw_aux_loss_mean_aux_updates':float(np.mean([r['raw_auxiliary_loss'] for r in dose])),'raw_aux_loss_mean_all_updates':logs['A']['training']['mean_auxiliary_loss'],'coefficient':.05,'weighted_contribution_mean_aux_updates':weighted/len(aux),'weighted_contribution_mean_all_updates':weighted/122,'weighted_to_LM_ratio_mean_aux_updates':float(np.mean([r['weighted_to_LM_ratio'] for r in dose])),'weighted_to_LM_ratio_all_updates':weighted/sum(r['lm_loss'] for r in logs['A']['stats']),'LM_gradient_norm':'NOT_AVAILABLE for A separately','aux_gradient_norm':'NOT_AVAILABLE','cosine_LM_aux':'NOT_AVAILABLE','combined_gradient_norm':trajectory['A'],'parameter_delta':parameter['aggregates']['total'],'classification':['INTERVENTION_CONFOUNDED_BY_CONTROL_DRIFT','INSUFFICIENT_DOSE_EVIDENCE'],'coefficient_tuned':False,'efficacy_claim':'NO VALID CLAIM'})
    descriptive={k:{'greedy':r['generation']['greedy']['metrics'],'sampling':{base:x['metrics'] for base,x in r['generation']['sampling'].items()},'eos':{key:v for key,v in r['eos'].items() if key not in ('terminal','nonterminal')}} for k,r in old.items()}
    emit('phase57-forensic-summary.json',{'complete':True,'phase57_status':'EXPERIMENT_INVALID','automatic_gate':'CONTROL_DRIFT_REVIEW_REQUIRED','128k':'NOT_RUN','intervention_promising':'UNDETERMINED / NO VALID CLAIM','old_outputs_rescored_for_efficacy':False,'observed_descriptive_metrics':descriptive,'registered_valid_efficacy_decision':'INVALID','findings':['naturalness/semantic expected aliases absent; .get(...,0) silently erased their safeguards','generation_metric reads greedy first, including topic and Japanese validity labeled sampling','binary-float boundary 0.010000000000000009 triggered <=0.01 failure; never retroactively repaired','math/code/list shape regexes double-escaped in PHASE57; descriptive 0/20 is not validated capability evidence','absolute-update cache cursor starts at episode32; initial cursor not explicit in registration','64k full-success stop/conditional-extension rules differ from initial gate64 implementation intent; no 128k executed'],'primary_independent_blocker':'Control Core micro and macro both exceed unchanged 0.1 safeguard','new_contract_not_applied_to_old_outputs':True})
    del payloads;gc.collect();check_frozen();print('FORENSIC COMPLETE; old Gate untouched; zero training',flush=True)


def contribution_detail():
    check_frozen();population=read(ROOT/'evaluation/foundation-v39-supported-tail.json')['core'];ids=np.asarray(population['token_ids'])
    parent=read(OLDRAW/'parent-evaluation.json')['core']['values']['ce'];control=read(OLDRAW/'control-64k-evaluation.json')['core']['values']['ce'];delta=np.asarray(control)-np.asarray(parent)
    token_rows=read(RAW/'core-contributions.json')['tokens'];types=len(np.unique(ids));rows=[]
    for r in token_rows:rows.append({**r,'macro_contribution':r['ce_delta']/types})
    assert abs(sum(r['macro_contribution'] for r in rows)-(read(OUT/'control-drift-audit.json')['control_macro_delta']))<1e-12
    buckets={}
    for lo,hi in ((3277,3481),(3481,3686),(3686,3891),(3891,4096)):
        members=[r for r in rows if lo<=r['frequency_rank']<hi]
        buckets[f'train_rank_{lo}_{hi-1}']={'types':len(members),'occurrences':sum(r['count'] for r in members),'micro_contribution':sum(r['micro_contribution'] for r in members),'macro_contribution':sum(r['macro_contribution'] for r in members)}
    emit('core-frequency-decomposition.json',{'diagnostic_only':True,'micro_sum':sum(r['micro_contribution'] for r in rows),'macro_sum':sum(r['macro_contribution'] for r in rows),'macro_definition':'Equal mean of per-token CE deltas, no occurrence weighting','all_core_token_types':types,'frequency_groups':buckets,'top20_macro_contributors':sorted(rows,key=lambda r:r['macro_contribution'],reverse=True)[:20],'formal_gate':False})
    print('MICRO_MACRO_FREQUENCY_DECOMPOSITION_COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('preflight','dry-run','dry-run-final','forensic','contribution-detail','verify'));args=parser.parse_args();torch.set_num_threads(2)
    if args.action=='preflight':preflight()
    elif args.action=='dry-run':dry_run()
    elif args.action=='dry-run-final':dry_run('evaluator-contract-final-audit.json')
    elif args.action=='forensic':forensic()
    elif args.action=='contribution-detail':contribution_detail()
    else:check_frozen();print('FROZEN_FILES_PROTECTED_DIRTY_AND_SEALED_HASH_PASS')
if __name__=='__main__':main()
