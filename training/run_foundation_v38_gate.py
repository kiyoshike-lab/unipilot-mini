"""PHASE49 isolated LR extension and approval-guarded canonical candidates."""
from __future__ import annotations
import argparse
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from training import run_foundation_v36_lr_review as v36
from training import run_foundation_v37_stability as v37
from training.run_foundation_v35_thermal_gate import Monitor,cooldown
from training.run_foundation_v30_eos_experiment import load
from training.train_foundation_v21_ab import file_sha256,random_state
from training.train_foundation_v15_controlled import macro_batch
from training.foundation_v31_objective import weighted_lm_loss
from foundation.base_tokenizer import FoundationTokenizer

EVAL=ROOT/'evaluation/phase49'
OUT=ROOT/'checkpoints/experimental/phase49'
SEEDS=v37.SEEDS
read=v36.read_json
write=v36.write_json


def checkpoint(arm,seed,candidate=False):
    if arm not in ('B','C') or seed not in SEEDS: raise ValueError('unregistered arm/seed')
    if candidate: return OUT/f'canonical-candidate/arm-{arm}/seed-{seed}/checkpoint-tokens-16128000.pt'
    if arm=='C': return v37.checkpoint('C',seed,512000)
    return OUT/f'arm-B/seed-{seed}/checkpoint-tokens-16384000.pt'


def evaluation_path(arm,seed):
    return v37.evaluation_path(arm,seed,512000) if arm=='C' else EVAL/f'arm-B/seed-{seed}-evaluation.json'


def training_path(arm,seed,candidate=False):
    if candidate: return EVAL/f'canonical-candidate/arm-{arm}/seed-{seed}-training.json'
    return v37.training_path(arm,seed,512000) if arm=='C' else EVAL/f'arm-B/seed-{seed}-training.json'


def preflight():
    old=read(v37.EVAL/'preflight.json'); rows=list(old['immutable_checkpoints'])
    for path in v37.EVAL.glob('arm-*/*/*-training.json'):
        r=read(path); rows.append({'path':r['checkpoint'],'sha256':r['sha256']})
    for r in rows:
        assert file_sha256(ROOT/r['path'])==r['sha256']
        p=torch.load(ROOT/r['path'],map_location='cpu',weights_only=False)
        r['integrity']=v36.verify_payload(p,p['seed'],p['tokens_processed'],p.get('experimental_lr',1e-4))
    for r in old['preserved_files']: assert file_sha256(Path(r['path']))==r['sha256']
    assert file_sha256(ROOT/'data/foundation_v09/evaluation/final-blind-1000.json')==old['final_blind_sha256']
    assert shutil.disk_usage(ROOT).free>=20*1024**3 and torch.cuda.is_available()
    write(EVAL/'preflight.json',{'phase':49,'expected_head':'130f1db67c705a691cd7b365b9fba0ba3014cb9d',
        'branch':'foundation-research','origin_verified':True,'immutable_checkpoints':rows,
        'preserved_files':old['preserved_files'],'final_blind_sha256':old['final_blind_sha256'],
        'free_bytes':shutil.disk_usage(ROOT).free,'gpu':torch.cuda.get_device_name(0),'pass':True})
    print('PHASE49 preflight PASS',len(rows),flush=True)


def train(seed,candidate=False):
    assert read(EVAL/'tests-preflight.json')['failed']==0
    assert shutil.disk_usage(ROOT).free>=20*1024**3
    arm='B'
    if candidate:
        decision=read(EVAL/'decision.json')
        assert decision['approved'] and decision['gate'] in ('FORMAL_LR_APPROVED_5E5','FORMAL_LR_APPROVED_7_5E5')
        arm=decision['selected_arm']; source=v36.official(seed); tokens=16128000
    else: source=v37.checkpoint256('B',seed); tokens=16384000
    target=checkpoint(arm,seed,candidate); temporary=target.with_suffix('.pt.tmp')
    if target.exists() or temporary.exists(): raise FileExistsError(target)
    source_sha=file_sha256(source)
    expected=next(r['sha256'] for r in read(EVAL/'preflight.json')['immutable_checkpoints'] if Path(r['path'])==source.relative_to(ROOT))
    assert source_sha==expected
    lr=v37.ARMS[arm]; device=torch.device('cuda')
    # Integrity checking instantiates a model and consumes RNG. Do it BEFORE load restores RNG.
    original=torch.load(source,map_location='cpu',weights_only=False)
    v36.verify_payload(original,seed,tokens-256000,1e-4 if candidate else lr)
    del original
    cooling=cooldown()
    p,model,opt=load(source,device)
    continuity={k:v36.fingerprint(p[k]) for k in ('optimizer_state','scheduler_state','permutation','random_state')}
    assert v36.fingerprint(opt.state_dict())==continuity['optimizer_state']
    assert v36.fingerprint(random_state(device))==continuity['random_state']
    if candidate: v36.set_lr_only(opt,lr)
    else: assert all(g['lr']==lr for g in opt.param_groups)
    tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json')
    data=np.memmap(ROOT/'data/foundation_v11/packed/vocab-4096/train.bin',dtype=np.uint16,mode='r')
    end=p['update']+500; losses=[]; norms=[]; model.train()
    monitor=Monitor(); monitor.start(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    try:
        for step in range(p['update']+1,end+1):
            x,y=macro_batch(data,int(p['permutation'][step-1]),512); x,y=x.to(device),y.to(device)
            opt.zero_grad(set_to_none=True); z,_=model(x); loss,_,_=weighted_lm_loss(z,y,tok.eos_id,1.5)
            assert torch.isfinite(loss); loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); assert torch.isfinite(norm)
            opt.step(); losses.append(float(loss.detach())); norms.append(float(norm))
        torch.cuda.synchronize()
    finally: seconds=time.perf_counter()-start; telemetry=monitor.finish()
    saved={**p,'model_state':model.state_dict(),'optimizer_state':opt.state_dict(),'update':end,'tokens_processed':tokens,
           'scheduler_state':{**p['scheduler_state'],'global_step':end},'random_state':random_state(device),
           'phase':49,'experimental':True,'promoted':False,'formal_research':False,'canonical_candidate':candidate,
           'parent_checkpoint':source.relative_to(ROOT).as_posix(),'parent_sha256':source_sha,
           'recipe_id':f'phase49-eos1.5-fp32-lr{lr:g}','experimental_lr':lr,'arm':arm,'device':'cuda','precision':'FP32'}
    target.parent.mkdir(parents=True,exist_ok=True); torch.save(saved,temporary)
    check=torch.load(temporary,map_location='cpu',weights_only=False)
    integrity=v36.verify_payload(check,seed,tokens,lr)
    assert v36.fingerprint(check['permutation'])==continuity['permutation'] and file_sha256(source)==source_sha
    temporary.replace(target)
    write(training_path(arm,seed,candidate),{'phase':49,'arm':arm,'seed':seed,'lr':lr,'checkpoint':target.relative_to(ROOT).as_posix(),
        'sha256':file_sha256(target),'parent_checkpoint':source.relative_to(ROOT).as_posix(),'parent_sha256':source_sha,
        'start_tokens':tokens-256000,'end_tokens':tokens,'added_tokens':256000,'canonical_candidate':candidate,
        'source_unchanged':True,'integrity':integrity,'continuity':continuity,'cooldown':cooling,'telemetry':telemetry,
        'tokens_per_second':256000/seconds,'peak_vram_mib':torch.cuda.max_memory_allocated()/1048576,
        'loss':float(np.mean(losses)),'gradient_std':float(np.std(norms)),'device':'CUDA','precision':'FP32',
        'eos_weight':1.5,'repetition_auxiliary':False,'parallel_cpu_evaluation':'DISABLED'})
    print({'seed':seed,'tokens':tokens,'integrity':'PASS','tps':256000/seconds},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--preflight',action='store_true'); parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--candidate',action='store_true'); args=parser.parse_args(); torch.set_num_threads(4)
    if args.preflight: preflight()
    else: train(args.seed,args.candidate)
