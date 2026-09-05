"""PHASE 42 isolated GPU-training / CPU-evaluation pipeline."""
from __future__ import annotations
import argparse, json, math, os, subprocess, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.foundation_v31_objective import repetition_negative_candidates, unlikelihood_loss, weighted_lm_loss, JAPANESE_FUNCTION_WORDS
from training.run_foundation_v30_eos_experiment import load, lm
from training.train_foundation_v15_controlled import macro_batch
from training.train_foundation_v21_ab import file_sha256, random_state
from evaluation.diagnose_foundation_v29_generation import build_prefixes, document_ranges, generate_batch, target_metrics, ngram_repetition

BASE=ROOT/'checkpoints/foundation-v28-current/current'; OUT=ROOT/'checkpoints/experimental/phase42'; EVAL=ROOT/'evaluation/phase42/cpu-worker'; LOG=ROOT/'logs/phase42'; BUDGET=256000
ARMS={'A':(1.0,0.0),'B':(1.5,0.0),'C':(1.5,0.01),'D':(1.5,0.03),'E':(1.5,0.05)}
GREEDY={'name':'greedy','kind':'greedy','temperature':1.,'top_k':None,'top_p':None,'repetition_penalty':1.,'no_repeat_ngram':None,'eos_threshold':None}
SAMPLE={**GREEDY,'name':'temperature_0.7','kind':'sampling','temperature':.7}

def checkpoint(arm,seed=42): return OUT/f'arm-{arm}'/f'seed-{seed}'/f'checkpoint-tokens-{15360000+BUDGET}.pt'

def gpu_worker(arm, seed=42, budget=BUDGET):
    weight,lam=ARMS[arm]; device=torch.device('cuda'); source=BASE/f'seed-{seed}/checkpoint-tokens-15360000.pt'; source_hash=file_sha256(source)
    payload,model,opt=load(source,device); train=np.memmap(ROOT/'data/foundation_v11/packed/vocab-4096/train.bin',dtype=np.uint16,mode='r'); tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json')
    start=int(payload['update']); stats=[]; began=time.perf_counter(); model.train()
    torch.cuda.reset_peak_memory_stats()
    for update in range(start+1,start+budget//512+1):
        x,y=macro_batch(train,int(payload['permutation'][update-1]),512); x=x.to(device); y=y.to(device); opt.zero_grad(True); logits,_=model(x)
        ce,eos_ce,non_ce=weighted_lm_loss(logits,y,tok.eos_id,weight); negatives=repetition_negative_candidates(x,y) if lam else []; rep=unlikelihood_loss(logits,negatives); total=ce+lam*rep; total.backward(); norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)); opt.step()
        stats.append((float(ce.detach()),float(eos_ce.detach()),float(non_ce.detach()),float(rep.detach()),float(total.detach()),norm,len(negatives)))
    torch.cuda.synchronize(); elapsed=time.perf_counter()-began; target=checkpoint(arm,seed); target.parent.mkdir(parents=True,exist_ok=True); temp=target.with_suffix('.pt.tmp')
    saved={**payload,'model_state':model.state_dict(),'optimizer_state':opt.state_dict(),'update':start+budget//512,'tokens_processed':15360000+budget,'random_state':random_state(device),'experimental':True,'phase':42,'arm':arm,'eos_loss_weight':weight,'repetition_lambda':lam,'source_sha256':source_hash}
    torch.save(saved,temp); verify=torch.load(temp,map_location='cpu',weights_only=False); test=DiagnosticTransformerV17(DiagnosticConfigV17(**verify['config'])); test.load_state_dict(verify['model_state'],strict=True); temp.replace(target)
    result={'arm':arm,'seed':seed,'eos_weight':weight,'lambda_rep':lam,'tokens':budget,'seconds':elapsed,'tokens_per_second':budget/elapsed,'peak_vram_mib':torch.cuda.max_memory_allocated()/1048576,'source_unchanged':file_sha256(source)==source_hash,'checkpoint_sha256':file_sha256(target),'strict_reload':True,'loss_contributions':dict(zip(('lm_eos_weighted','eos_ce','non_eos_ce','repetition_aux','total','gradient_norm','negative_count'),np.mean(stats,axis=0).tolist()))}
    marker=target.with_suffix('.READY.json'); marker.write_text(json.dumps(result,indent=2),encoding='utf8'); return result

@torch.inference_mode()
def context_loss(model,val,device,length):
    losses=[]
    for p in range(512,4608,512):
        x=torch.tensor(np.asarray(val[p-length:p],dtype=np.int64),device=device)[None]; y=torch.tensor([int(val[p])],device=device); z,_=model(x); losses.append(float(torch.nn.functional.cross_entropy(z[:,-1],y)))
    return float(np.mean(losses))

def cpu_worker(arm,seed=42,threads=2):
    torch.set_num_threads(threads); target=checkpoint(arm,seed); marker=target.with_suffix('.READY.json')
    if not marker.exists(): raise RuntimeError('READY marker missing')
    before=file_sha256(target); p=torch.load(target,map_location='cpu',weights_only=False); model=DiagnosticTransformerV17(DiagnosticConfigV17(**p['config'])); model.load_state_dict(p['model_state'],strict=True); model.eval(); tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json'); val=np.memmap(ROOT/'data/foundation_v11/packed/vocab-4096/validation.bin',dtype=np.uint16,mode='r')
    unique=np.asarray([int(x) for x in np.flatnonzero(val==tok.eos_id) if x>=128]); terminal=np.resize(unique,500); non=np.linspace(128,len(val)-2,500,dtype=int); ranges=document_ranges(val,tok.bos_id,tok.eos_id); prefixes=build_prefixes(val,ranges,tok)
    began=time.perf_counter(); greedy=generate_batch(model,tok,[p['prefix_ids'] for p in prefixes],GREEDY,list(range(100)),128,True); sample=generate_batch(model,tok,[p['prefix_ids'] for p in prefixes],SAMPLE,list(range(100)),64,False)
    onsets=[r['loop']['loop_onset'] for r in greedy if r['loop']['loop_onset']]; spans=[r['loop']['maximum_repeated_span'] for r in greedy]; traces=[step for r in greedy for step in r['trace'] if r['loop']['loop_onset'] and step['step']==r['loop']['loop_onset']]
    function_ids={w:tok.encode(w)[0] for w in JAPANESE_FUNCTION_WORDS}; generated=[i for r in greedy for i in r['ids']]
    result={'arm':arm,'seed':seed,'cpu_threads':threads,'checkpoint_sha256':before,'checkpoint_unchanged':file_sha256(target)==before,'seconds':time.perf_counter()-began,'lm':lm(model,val,torch.device('cpu')),'terminal_eos':target_metrics(model,val,terminal,tok.eos_id),'nonterminal_eos':target_metrics(model,val,non,tok.eos_id),'greedy':{'runaway_rate':float(np.mean([r['runaway'] for r in greedy])),'first_break':any(not r['runaway'] for r in greedy),'median_loop_onset':float(np.median(onsets)) if onsets else None,'mean_loop_onset':float(np.mean(onsets)) if onsets else None,'repetition':{str(n):float(np.mean([ngram_repetition(r['ids'],n) for r in greedy])) for n in (1,2,3,4)},'mean_maximum_repeated_span':float(np.mean(spans)),'loop_taxonomy':dict(Counter(r['loop']['loop_type'] for r in greedy)),'onset_distribution':{'entropy':float(np.mean([x['entropy'] for x in traces])) if traces else None,'top1_probability':float(np.mean([x['top5'][0]['probability'] for x in traces])) if traces else None,'margin':float(np.mean([x['top1_top2_margin'] for x in traces])) if traces else None,'eos_probability':float(np.mean([x['eos_probability'] for x in traces])) if traces else None}},'sampling_t07':{'naturalness':float(np.mean([r['natural_japanese_proxy'] for r in sample])),'semantic':float(np.mean([r['semantic_coherence_proxy'] for r in sample])),'completion':float(np.mean([r['completion_proxy'] for r in sample])),'repetition_1':float(np.mean([r['repetition_1'] for r in sample]))},'japanese_function_word_rates':{w:generated.count(i)/len(generated) for w,i in function_ids.items()},'context_loss':{str(n):context_loss(model,val,torch.device('cpu'),n) for n in (512,64,16,2,1)}}
    EVAL.mkdir(parents=True,exist_ok=True); (EVAL/f'arm-{arm}-seed-{seed}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8'); return result

def run_subprocess(args,log):
    log.parent.mkdir(parents=True,exist_ok=True); handle=log.open('w',encoding='utf8'); return subprocess.Popen([sys.executable,__file__,*args],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT),handle

def orchestrate():
    started=time.perf_counter(); gpu=[]; cpu=[]; pending=None; pending_handle=None; sequential=0.; first_tps=None
    for arm in ARMS:
        g=gpu_worker(arm); gpu.append(g); sequential+=g['seconds']; first_tps=first_tps or g['tokens_per_second']
        if pending is not None: pending.wait(); pending_handle.close(); pending=None
        pending,pending_handle=run_subprocess(['--worker','cpu','--arm',arm],LOG/'cpu-worker'/f'arm-{arm}.log')
    if pending is not None: pending.wait(); pending_handle.close()
    cpu=[json.loads((EVAL/f'arm-{a}-seed-42.json').read_text(encoding='utf8')) for a in ARMS]; sequential+=sum(x['seconds'] for x in cpu); wall=time.perf_counter()-started
    parallel_tps=gpu[1]['tokens_per_second']; pipeline={'gpu_only_tok_s':first_tps,'parallel_tok_s':parallel_tps,'throughput_loss_pct':100*(1-parallel_tps/first_tps),'cpu_threads':2,'sequential_wall_seconds':sequential,'parallel_wall_seconds':wall,'total_speedup':sequential/wall,'peak_vram_mib':max(x['peak_vram_mib'] for x in gpu),'peak_ram_mib':None,'max_gpu_temp_c':None}; (ROOT/'evaluation/foundation-v31-arm-results.json').write_text(json.dumps(cpu,ensure_ascii=False,indent=2),encoding='utf8'); (ROOT/'evaluation/foundation-v31-parallel-pipeline-report.md').write_text('# PHASE 42 parallel pipeline\n\n'+json.dumps(pipeline,indent=2),encoding='utf8'); return cpu,pipeline

def finalize_existing():
    rows=json.loads((ROOT/'evaluation/foundation-v31-arm-results.json').read_text(encoding='utf8'))
    pipeline={'gpu_only_tok_s':13479.790697604965,'threads_2_parallel_tok_s':12501.740537734117,'threads_2_throughput_loss_pct':7.255677642269509,'threads_1_parallel_tok_s':12845.30210497219,'threads_1_throughput_loss_pct':4.706959,'selected_cpu_threads':1,'sequential_wall_seconds':258.6668898,'parallel_wall_seconds':358.5707429,'total_speedup':0.7213831438,'peak_ram_mib':2092.6,'peak_vram_mib':558.188,'max_gpu_temp_c':71,'verdict':'PARALLEL_PIPELINE_NOT_BENEFICIAL'}
    decisions={'A':'SAFE_BUT_NO_EFFECT','B':'SAFE_BUT_NO_EFFECT','C':'TOO_WEAK','D':'SAFE_BUT_NO_EFFECT','E':'SAFE_BUT_NO_EFFECT'}
    for row in rows: row['arm_judgment']=decisions[row['arm']]
    summary={'phase':42,'tested_arms':list(ARMS),'best_arm':None,'selected_eos_weight':1.5,'selected_repetition_lambda':None,'three_seed_confirmation':False,'greedy_attractor_correction_validated':False,'next_phase_gate':'EOS_FIXED_BUT_ATTRACTOR_TRAINING_FIX_FAILED','formal_continuation_permission':'NONE','foundation_base_complete':False,'parallel_pipeline':pipeline,'arms':rows}
    (ROOT/'evaluation/foundation-v31-greedy-attractor-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    report=['# Foundation v3.1 Greedy attractor research','','All five 256k-token seed-42 arms completed from the same official 15.360M checkpoint and optimizer state. Official checkpoints were not modified.','','| Arm | EOS weight | lambda | loss | runaway | median onset | repetition-1 | judgment |','|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows: report.append(f"| {r['arm']} | {ARMS[r['arm']][0]} | {ARMS[r['arm']][1]} | {r['lm']['loss']:.6f} | {r['greedy']['runaway_rate']:.0%} | {r['greedy']['median_loop_onset']:.1f} | {r['greedy']['repetition']['1']:.4f} | {r['arm_judgment']} |")
    report += ['','C/D/E did not improve runaway versus EOS-corrected B (all 100%), and median loop onset remained 19 for B/C/D and regressed to 18 for E. The tiny repetition differences are not sufficient for a SAFE_AND_HELPFUL selection. Three-seed confirmation was therefore not run.','',f"Gate: **{summary['next_phase_gate']}**. Formal 20M permission: **NONE**. Foundation Base completion: **NO**."]
    (ROOT/'evaluation/foundation-v31-greedy-attractor-report.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    (ROOT/'evaluation/foundation-v31-parallel-pipeline-report.md').write_text('# PHASE 42 parallel pipeline\n\n'+json.dumps(pipeline,indent=2)+'\n',encoding='utf8')

def main():
 a=argparse.ArgumentParser(); a.add_argument('--worker',choices=['gpu','cpu','all','finalize'],default='all'); a.add_argument('--arm',choices=list(ARMS)); a.add_argument('--threads',type=int,default=2); args=a.parse_args()
 if args.worker=='gpu': print(json.dumps(gpu_worker(args.arm))); return
 if args.worker=='cpu': print(json.dumps(cpu_worker(args.arm,threads=args.threads))); return
 if args.worker=='finalize': finalize_existing(); return
 rows,pipeline=orchestrate(); base=rows[0]; candidates=rows[2:]; best=max(candidates,key=lambda x:(-x['greedy']['runaway_rate'],x['greedy']['median_loop_onset'],-x['greedy']['repetition']['1'])); helpful=best['greedy']['median_loop_onset']>base['greedy']['median_loop_onset'] and best['lm']['loss']<=base['lm']['loss']+.05; gate='ATTRACTOR_WEAKENED_BUT_MORE_TESTING_REQUIRED' if helpful else 'EOS_FIXED_BUT_ATTRACTOR_TRAINING_FIX_FAILED'; summary={'phase':42,'best_arm':best['arm'] if helpful else None,'selected_eos_weight':1.5,'selected_repetition_lambda':ARMS[best['arm']][1] if helpful else None,'three_seed_confirmation':False,'greedy_attractor_correction_validated':False,'next_phase_gate':gate,'formal_continuation_permission':'NONE','foundation_base_complete':False,'parallel_pipeline':pipeline,'arms':rows}; (ROOT/'evaluation/foundation-v31-greedy-attractor-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8'); (ROOT/'evaluation/foundation-v31-greedy-attractor-report.md').write_text(f'# Foundation v3.1 Greedy attractor research\n\nBest arm: {summary["best_arm"]}. Gate: **{gate}**. No formal 20M training was run or authorized. Foundation Base completion: **NO**.\n',encoding='utf8')
if __name__=='__main__': main()
