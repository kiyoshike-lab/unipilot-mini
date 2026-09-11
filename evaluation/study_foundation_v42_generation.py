"""Bounded existing-prompt decoder study, separately authorized from fresh confirmation."""
from __future__ import annotations
import gc,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from evaluation.diagnose_foundation_v40 import resolver_gate,read,new_json,file_sha256,evaluated_checkpoint,verify_payload,cooldown,Monitor
from evaluation.diagnose_foundation_v29_generation import GREEDY,generate_batch,load_model,ngram_repetition
from evaluation.investigate_foundation_v14 import language_proxy
from foundation.base_tokenizer import FoundationTokenizer
from training.run_foundation_v35_thermal_gate import query_gpu
OUT=ROOT/'evaluation/phase53';PREREG=OUT/'generation-preregistration.json'

def stop_index(ids):
    """First 3 identical adjacent cycles, width1..4, after >=8 generated tokens."""
    for end in range(8,len(ids)+1):
        for width in range(1,5):
            if end>=3*width and ids[end-width:end]==ids[end-2*width:end-width]==ids[end-3*width:end-2*width]:return end
    return None

def freeze():
    resolver_gate();old=read(ROOT/'evaluation/phase51/preregistration.json')
    prompts=sorted(old['prompts'],key=lambda p:p['id'])[:24]
    modes=[GREEDY,{**GREEDY,'name':'temperature_0.7','kind':'sampling','temperature':.7},
           {**GREEDY,'name':'top_k_50','kind':'sampling','temperature':.7,'top_k':50},
           {**GREEDY,'name':'top_p_0.90','kind':'sampling','temperature':.7,'top_p':.9},
           {**GREEDY,'name':'repetition_1.10','repetition_penalty':1.1}]
    new_json(PREREG,{'phase':53,'scope':'Track C only; historical fixed prompts, not fresh/confirmatory/reserve data',
        'execution_authorization':'PENDING_CLARIFICATION: fresh gate TOO_SMALL; do not assume broad inference permission',
        'checkpoint_seed':42,'arms':['C','B'],'prompts':prompts,'prompt_selection':'First24 PHASE51 IDs sorted lexically, without new decoder outputs',
        'checkpoint_sha256':{arm:file_sha256(evaluated_checkpoint(arm,42)) for arm in ('C','B')},
        'modes':modes,'rng_bases':[53000,53100],'max_new_tokens':64,'batch_size':8,
        'stop_policy':'Offline prefix replay of greedy and temperature0.7: first3 adjacent identical cycles width1..4, minimum8 tokens; separately labelled forced truncation, not EOS',
        'metrics':['budget_exhaustion_runaway','forced_stop_rate','natural_japanese_proxy','semantic_coherence_proxy','completion_proxy','character_valid','japanese_character_ratio','topic_overlap_proxy','repetition1'],
        'fixed_screen':'Each arm: budget runaway <=0.20 and reduced>=0.50 absolute vs greedy; completion>=0.50, semantic proxy>=0.50, character valid>=0.95; completion/semantic/Japanese ratio not worse than temperature0.7 by>0.05. Automated screen only; human quality must be independently verified before staging recommendation.',
        'gate':'GENERATION_POLICY_UNSAFE if all non-greedy settings fail quality floors; otherwise ATTRACTOR_CAUSE_STILL_UNKNOWN until human safety supports DECODER_MITIGATION_FOR_STAGING. No MODEL_FIXED claim.',
        'cuda':'FP32, TF32 disabled','thermal':'cooldown before each mode; stop>=80C or telemetry error; no parallel heavy QA','fresh_data_read':False,'new_training':False})

def summarize(rows,prompts):
    result={key:float(np.mean([float(r[key]) for r in rows])) for key in ('runaway','forced_stop','natural_japanese_proxy','semantic_coherence_proxy','completion_proxy','character_valid','japanese_character_ratio','repetition_1')}
    result['topic_overlap_proxy']=float(np.mean([len(set(r['ids'])&set(p['prefix_ids']))/max(1,len(set(r['ids']))) for r,p in zip(rows,prompts)]))
    return result

def replay(rows,tok):
    output=[]
    for r in rows:
        at=stop_index(r['ids']);ids=r['ids'][:at] if at else r['ids'];eos=bool(ids and ids[-1]==tok.eos_id);text=tok.decode(ids,skip_special=True)
        output.append({**r,'ids':ids,'text':text,'forced_stop':at is not None and at<len(r['ids']),'eos_reached':eos,'runaway':len(ids)>=64 and not eos,'repetition_1':ngram_repetition(ids),**language_proxy(text,eos_reached=eos)})
    return output

def run(approved=False):
    assert approved,'Explicit authorization for existing-prompt inference required while fresh gate is TOO_SMALL'
    resolver_gate();spec=read(PREREG);sha=file_sha256(PREREG);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json');records=[]
    for arm in spec['arms']:
        path=evaluated_checkpoint(arm,42);assert file_sha256(path)==spec['checkpoint_sha256'][arm]
        payload=torch.load(path,map_location='cpu',weights_only=False);integrity=verify_payload(payload,42,payload['tokens_processed'],payload['experimental_lr']);assert integrity['pass'];del payload;gc.collect()
        model=load_model(path,torch.device('cuda'));assert all(p.dtype==torch.float32 for p in model.parameters())
        for mode in spec['modes']:
            for rng in spec['rng_bases']:
                target=OUT/'raw'/f"generation-{arm}-{mode['name']}-{rng}.json"
                if target.exists():r=read(target);assert r['complete'] and r['preregistration_sha256']==sha;records.append(r);continue
                cool=cooldown();assert cool['target_reached'];monitor=Monitor(1);monitor.start();rows=[]
                try:
                    for start in range(0,len(spec['prompts']),8):
                        assert query_gpu()['gpu_temperature_c']<80,'Thermal safety stop'
                        batch=spec['prompts'][start:start+8]
                        rows.extend(generate_batch(model,tok,[p['prefix_ids'] for p in batch],mode,[rng+start+i for i in range(len(batch))],64))
                finally:thermal=monitor.finish()
                assert thermal.get('gpu_temperature_c_max',999)<80,'Thermal safety stop'
                for row in rows:row['forced_stop']=False
                replayed=replay(rows,tok) if mode['name'] in ('greedy','temperature_0.7') else None
                record={'complete':True,'arm':arm,'rng':rng,'mode':mode['name'],'preregistration_sha256':sha,'checkpoint_sha256':spec['checkpoint_sha256'][arm],'integrity':integrity,'metrics':summarize(rows,spec['prompts']),'rows':rows,'stop_replay':{'metrics':summarize(replayed,spec['prompts']),'rows':replayed} if replayed else None,'thermal':thermal,'new_training':False}
                new_json(target,record);records.append(record);print(arm,mode['name'],rng,record['metrics'],flush=True)
        del model;gc.collect();torch.cuda.empty_cache();assert file_sha256(path)==spec['checkpoint_sha256'][arm]
    new_json(OUT/'generation-run-complete.json',{'preregistration_sha256':sha,'raw_files':20,'scope':'historical fixed prompts only','new_training':False})

if __name__=='__main__':
    if sys.argv[1:] == ['freeze']:freeze()
    elif sys.argv[1:] == ['run','--approved-existing-prompts']:run(True)
    else:raise SystemExit('Use freeze; run --approved-existing-prompts only after explicit authorization')
