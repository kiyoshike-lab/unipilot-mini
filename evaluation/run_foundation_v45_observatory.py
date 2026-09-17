"""Frozen PHASE56 inference only; original checkpoints and sealed sets immutable."""
from __future__ import annotations
import gc,hashlib,os,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.diagnose_foundation_v40 import read,new_json,file_sha256,resolver_gate,evaluated_checkpoint,verify_payload,load_model,cooldown,Monitor
from evaluation.confirm_foundation_v44 import thermal_guard
from evaluation.generation_observatory_v45 import trajectory,Observer
from foundation.base_tokenizer import FoundationTokenizer
from evaluation.build_foundation_v42_holdout import now,digest
OUT=ROOT/'evaluation/phase56';RAW=Path(r'Z:\AI\unipilot-mini\evaluation\phase56');SPEC=OUT/'observability-preregistration.json'

def gate():
    assert os.environ.get('UNIPILOT_CHECKPOINT_ROOT')==r'Z:\AI\unipilot-mini\checkpoints','PROCESS_ENV_RESOLVER_MISMATCH'
    m=resolver_gate();assert read(ROOT/'evaluation/foundation-v44-confirmatory-summary.json')['formal_lr_gate']=='FORMAL_LR_APPROVED_5E5'
    for r in [read(ROOT/'evaluation/phase55/phase53-reserve-retirement.json'),read(ROOT/'evaluation/phase55/fresh-holdout-v2-manifest.json')['splits']['future-reserve2']]:assert file_sha256(Path(r['path']))==r['sha256']
    return m

def freeze():
    m=gate();manifest=read(ROOT/'evaluation/phase55/fresh-holdout-v2-manifest.json');diagnostic=manifest['splits']['diagnostic']
    metadata=read(ROOT/'evaluation/phase55/fresh-holdout-v2-fingerprints.json')['documents'];admitted=set(diagnostic['document_ids'])
    chosen=sorted([r['document_id'] for r in metadata if r['document_id'] in admitted and r['tokens']>=320],key=lambda id:digest('phase56-5601|'+id))[:24]
    assert len(chosen)==24
    old=read(ROOT/'evaluation/phase51/preregistration.json')['prompts'][:8]
    source_files=[Path(__file__),ROOT/'evaluation/generation_observatory_v45.py',ROOT/'foundation/diagnostic_transformer_v17.py',ROOT/'model/attention.py',ROOT/'evaluation/diagnose_foundation_v29_generation.py',ROOT/'evaluation/phase55/normal-controls-preregistration.json',ROOT/'tokenizer/foundation-v11-base-4096.json']
    identities=[];torch.set_num_threads(2)
    for arm,seed in [('C',42),('C',123),('C',2026),('baseline',42)]:
        path=evaluated_checkpoint(arm,seed);known=next(r for r in m['rows'] if Path(r['destination'])==path);assert file_sha256(path)==known['sha256']
        payload=torch.load(path,map_location='cpu',weights_only=False);integrity=verify_payload(payload,seed,16384000 if arm=='C' else 15872000,5e-5 if arm=='C' else 1e-4)
        identities.append({'arm':arm,'seed':seed,'path':str(path),'sha256':known['sha256'],'integrity':integrity,'layers':payload['config']['n_layers']});del payload;gc.collect()
    new_json(SPEC,{'phase':56,'frozen_at':now(),'new_results_seen':False,'approved_lr':5e-5,'confirmatory_rerun':False,
        'diagnostic_source':diagnostic,'selected_document_ids':chosen,'selection':'First24 SHA256(phase56-5601|ID), among frozen Diagnostic IDs tokens>=320, before outcomes',
        'main_prefix':'BOS + first127 text tokens; correct continuation text[127:255]. No confirmatory/reserve inputs.',
        'position_comparison':'Same continuation boundary; short prefix BOS+text[64:127] vs long BOS+text[:127]. Natural suffix, no padding/embedding edits. Earlier-content and position remain confounded; cannot assert pure positional cause.',
        'legacy_prompt_ids':[r['id'] for r in old],'legacy_rule':'First8 existing PHASE51 prompts; C3seeds only, descriptive transfer check',
        'checkpoints':identities,'baseline_scope':'first8 selected Diagnostic docs, free/teacher only; no new B inference',
        'max_new_tokens':128,'decoder':'greedy, mask same special tokens except EOS as PHASE51; no repetition constraints or forced EOS stopping',
        'loop_definition':'Unchanged diagnose_foundation_v29_generation.loop_details: largest contiguous periodic span, widths1..32, >=2 cycles, earliest tie. Budget runaway reported separately.',
        'perturbation':{'offsets_before_onset':[4,8,16],'types':['top2','ground_truth'],'rule':'1-based step onset-offset if>=1, replay unmodified free choices before this step, replace exactly one next token by second-highest allowed raw-logit token or aligned teacher token, then greedy. No chosen outcome-dependent alternatives. Identical replacement marked no-op, not evidence.',
            'post_reloop':'apply same unchanged loop definition to suffix AFTER replaced token; require >=16 remaining steps for persistent-basin assay'},
        'observed':['token_id/text','absolute input position','EOS p/rank/logit/top1 margin','top1/2 probability/margin','entropy','selected-token probability','logit norm','all block hidden norms/previous/cycle cosine','raw and centered logit cosine','all block attention recent1..4/previous-cycle/earlier/BOS/expected position'],
        'hooks':'Read-only block output and attention_dropout output hooks; eval mode; returnNone; verify identical logits before/with hooks and unchanged parameter tensors. No model code rewrite.',
        'attention_definition':'Attention vectors already computed by existing eager module; dropout hook observes them safely, averaged heads. Previous-cycle band keys[-2w:-w]; recent keys include current input position.',
        'vectors':'Z-only selected step1,32,64,96,128 and onset-4..onset+4. Full-vocab vectors NOT saved at all positions; no Git raw.',
        'windows':'strictly32 steps before/after free-run onset; teacher uses free alignment. Boundary counts explicit. No-loop aligned at65, no invented onset.',
        'cycle_alignment':'compare each layer state to state w steps earlier using FREE cycle width, for free and teacher; representation predicts selected NEXT token',
        'normal_controls':'Reuse20 PHASE55 concretizations of PHASE54 frozen controls. New teacher-forced reference strings defined before scoring; all3Cseeds. Report CE and terminal/nonterminal EOS, not instruction-following accuracy.',
        'root_rules':{'feedback_signal':'At least2of3 Cseeds: equal-doc free minus teacher repetition4>=.20 AND post-onset free-minus-teacher cycle cosine>=.05 in at least5 layers.',
            'persistent_basin_signal':'Among changed perturbations with>=16 remaining steps, >=80% show a suffix loop; at least24 changed eligible runs overall.',
            'novel_signal':'At least50% unique generated6grams in post-onset region absent from complete training corpus; report occurrence-weighted and per-n rates too.',
            'actionable':'Only if all3 signals pass: MIXED_CAUSE_WITH_ACTIONABLE_TARGET (generated-prefix self-feedback target, not proven single dominant mechanism). Otherwise INSUFFICIENT_EVIDENCE. Dominant labels require additional independent causal separation not supplied by this design.',
            'layer_cycle_lock':'earliest layer with mean post-onset cycle cosine>=.95 AND free-minus-teacher>=.05; else NOT_ESTABLISHED. Descriptive, not a causal source-layer claim.'},
        'new_training':False,'canonical':False,'20m':False,'thermal':'cooldown65 before each checkpoint; monitor each trajectory, pause>=80, stop>=85/hardware slowdown','cpu_heavy_parallel_evaluation':False,
        'source_sha256':{str(p.relative_to(ROOT)):file_sha256(p) for p in source_files}})
    new_json(OUT/'preregistration-freeze.json',{'sha256':file_sha256(SPEC),'at':now()})
    new_json(OUT/'safety-preflight.json',{'protected_files':m['protected_files'],'root':os.environ['UNIPILOT_CHECKPOINT_ROOT'],'resolver':'PASS','checkpoint_copy_move_delete_overwrite':[0,0,0,0],'reserved_sets':'hash only PASS'})
    print('PHASE56 preregistered',file_sha256(SPEC),flush=True)

def normal_truth(family,n):
    if family=='math':return '+'.join(['1']*n)+'='+str(n)+'。'
    if family=='code':return '\n'.join(f'print({j})' for j in range(1,n+1))
    if family=='lists':return '\n'.join(f'{j}. 確認済み' for j in range(1,n+1))
    if family=='definitions':return '\n'.join('定義: '+s for s in ['変数は値を表す記号。','関数は入力に出力を対応させる規則。','集合は対象の集まり。','写像は各要素に対応先を一つ指定する。','命題は真偽を判断できる文。'][:n])
    return ''.join(['標本平均は観測値の合計を個数で割った値です。','標本平均はデータの中心を表す指標です。','標本平均は標本によって変動します。','標本平均は母平均の推定に使われます。','標本平均は極端な値の影響を受けます。'][:n])

def run():
    gate();spec=read(SPEC);sha=file_sha256(SPEC);assert sha==read(OUT/'preregistration-freeze.json')['sha256']
    for p,h in spec['source_sha256'].items():assert file_sha256(ROOT/p)==h
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert torch.cuda.is_available()
    split=spec['diagnostic_source'];assert file_sha256(Path(split['path']))==split['sha256']
    if not (OUT/'diagnostic-consumption.json').exists():new_json(OUT/'diagnostic-consumption.json',{'consumed_for_diagnostics':True,'source_documents':299,'selected_documents':24,'selected_ids':spec['selected_document_ids'],'formal_lr_selection':False,'at':now()})
    docs=read(split['path']);lookup={f"jawiki:{r['page_id']}:{r['revision_id']}":r for r in docs}
    tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json');encoded={id:tok.encode(lookup[id]['text']) for id in spec['selected_document_ids']}
    legacy=read(ROOT/'evaluation/phase51/preregistration.json')['prompts'][:8]
    controls=read(ROOT/'evaluation/phase55/normal-controls-preregistration.json')['prompts'];index=[]
    RAW.mkdir(parents=True,exist_ok=True)
    for identity in spec['checkpoints']:
        arm,seed=identity['arm'],identity['seed'];marker=OUT/f'{arm}-{seed}-complete.json'
        if marker.exists():index.extend(read(marker)['runs']);continue
        assert file_sha256(Path(identity['path']))==identity['sha256'];assert cooldown()['target_reached']
        model=load_model(Path(identity['path']),torch.device('cuda'));m=Monitor();m.start();records=[]
        # Test hooks are observational at exact same input, no weights changed.
        state={k:v.detach().clone() for k,v in model.state_dict().items()};sample=torch.tensor([[tok.bos_id,32,205]],device='cuda')
        with torch.inference_mode():
            plain=model(sample)[0];observer=Observer(model);tapped=model(sample)[0];observer.close();assert torch.equal(plain,tapped),'HOOK_CHANGED_LOGITS'
        def save(name,prefix,**kwargs):
            thermal_guard(m);target=RAW/f'{arm}-{seed}-{name}.json';vp=target.with_suffix('.npz')
            if target.exists():
                r=read(target);assert r['preregistration_sha256']==sha and file_sha256(vp)==r['vectors_sha256']
            else:
                assert not vp.exists(),'Incomplete vector artifact: inspect, never overwrite'
                result,vectors=trajectory(model,tok,prefix,**kwargs)
                with vp.open('xb') as handle:np.savez_compressed(handle,**vectors)
                r={**result,'preregistration_sha256':sha,'checkpoint_sha256':identity['sha256'],'vectors_path':str(vp),'vectors_sha256':file_sha256(vp)};new_json(target,r)
            records.append({'name':name,'arm':arm,'seed':seed,'path':str(target),'sha256':file_sha256(target),'vectors_sha256':r['vectors_sha256']})
            return r
        try:
            selected=spec['selected_document_ids'][:8] if arm=='baseline' else spec['selected_document_ids']
            for number,id in enumerate(selected):
                ids=encoded[id];prefix=[tok.bos_id]+ids[:127];truth=ids[127:255];name=f'doc{number:02}'
                free=save(name+'-free',prefix);save(name+'-teacher',prefix,truth=truth,reference_loop=free['loop'])
                if arm=='C':
                    save(name+'-short', [tok.bos_id]+ids[64:127])
                    onset=free['loop']['loop_onset']
                    for offset in (4,8,16):
                        step=(onset or 0)-offset
                        if step<1:continue
                        for kind,replacement in [('top2','top2'),('truth',truth[step-1])]:save(name+f'-{kind}-{offset}',prefix,replay=free['ids'],replace_step=step,replacement=replacement,reference_loop=free['loop'])
                print('Observed',arm,seed,name,'runs',len(records),flush=True)
            if arm=='C':
                for i,p in enumerate(legacy):save(f'legacy{i:02}',p['prefix_ids'])
                for p in controls:save('normal-'+p['id'],[tok.bos_id]+tok.encode(p['prompt']),truth=tok.encode(normal_truth(p['family'],p['n']))+[tok.eos_id])
            assert all(torch.equal(state[k],v) for k,v in model.state_dict().items()),'MODEL_STATE_CHANGED'
        finally:
            telemetry=m.finish();del model,state;gc.collect();torch.cuda.empty_cache()
        assert telemetry['samples'] and telemetry['gpu_temperature_c_max']<85 and not telemetry['hardware_thermal_slowdown']
        new_json(marker,{'complete':True,'runs':records,'thermal':telemetry,'hook_logits_equal':True,'model_tensors_unchanged':True,'new_training':False});index.extend(records)
    new_json(OUT/'run-index.json',{'preregistration_sha256':sha,'runs':index,'new_training':False});print('OBSERVATORY COMPLETE',len(index),flush=True)

if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
