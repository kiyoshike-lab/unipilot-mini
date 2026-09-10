"""PHASE52 provenance and observational attractor audit; no training/inference."""
from __future__ import annotations
import gzip
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from evaluation.diagnose_foundation_v40 import read,new_json,file_sha256,resolver_gate,SEEDS,ARMS,ngram_repetition
from training.run_foundation_v36_lr_review import write_json
OUT=ROOT/'evaluation/phase52'
FORBIDDEN=ROOT/'data/foundation_v09/evaluation/final-blind-1000.json'


def logit_dynamics():
    """Recover relative logit differences from raw softmax probabilities; no inference."""
    source=OUT/'raw/attractor-time-series.json';rows=read(source);derived=[];groups={}
    for r in rows:
        window=[]
        for t in r['pre_onset_window']:
            a,b,e=t['top1_probability'],t['top2_probability'],t['eos_probability']
            window.append({'step':t['step'],'top1_minus_top2_logit':math.log(a/b) if a>0 and b>0 else None,
                'top1_minus_eos_logit':math.log(a/e) if a>0 and e>0 else None})
        item={'arm':r['arm'],'seed':r['seed'],'prompt_id':r['prompt_id'],'onset':r['loop_onset'],'window':window}
        derived.append(item);groups.setdefault(f"{r['arm']}-{r['seed']}",[]).append(item)
    path=OUT/'raw/pre-onset-logit-dynamics.json';new_json(path,derived)
    summary=read(ROOT/'evaluation/foundation-v41-attractor-summary.json')
    summary['relative_logit_dynamics']={'method':'log(p1/p2) = logit1-logit2 for the same raw softmax distribution. Zero probabilities produce null, not invented finite values.',
        'path':path.relative_to(ROOT).as_posix(),'sha256':file_sha256(path),'source_sha256':file_sha256(source),
        'by_checkpoint':{key:{field:float(np.mean([r['window'][-1][field] for r in group if r['window'][-1][field] is not None])) for field in ('top1_minus_top2_logit','top1_minus_eos_logit')} for key,group in groups.items()}}
    write_json(ROOT/'evaluation/foundation-v41-attractor-summary.json',summary)


def complete_counts():
    """Enrich only this phase's generated audit, never any dataset or checkpoint."""
    audit=read(OUT/'data-provenance.json');present={r['path'] for r in audit['rows']}
    for p in (ROOT/'data').rglob('*'):
        rel=p.relative_to(ROOT).as_posix()
        if not p.is_file() or '/raw/' in rel or p.suffix not in ('.json','.jsonl'):continue
        if not re.search(r'/(evaluation|holdouts|benchmarks|blind)/|prompt|question',rel,re.I) or rel in present:continue
        history=subprocess.check_output(['git','log','--all','--reverse','--format=%h %s','--',rel],cwd=ROOT,text=True).splitlines()
        audit['rows'].append({'name':p.name,'path':rel,'sha256':file_sha256(p),'documents_or_items':None,
            'count_basis':'not yet inspected','first_phase_used':'NOT_ESTABLISHED','first_tracked_commit':history[0] if history else None,
            'purposes_used':['Repository heldout/benchmark family; first-use custody not established'],'selection_influence':'UNKNOWN','confirmatory_eligible':'NO',
            'reason':'Folder-based inventory extension; no evidence of fresh sealed custody','references':[]})
    for row in audit['rows']:
        p=ROOT/row['path']
        if p==FORBIDDEN or row['documents_or_items'] is not None:continue
        if p.name.endswith('.jsonl.gz'):
            with gzip.open(p,'rt',encoding='utf-8') as f:row['documents_or_items']=sum(bool(line.strip()) for line in f)
            row['count_basis']='JSONL nonempty record count; no source prose displayed'
        elif p.suffix=='.jsonl':
            with p.open(encoding='utf-8') as f:row['documents_or_items']=sum(bool(line.strip()) for line in f)
            row['count_basis']='JSONL nonempty record count; no source prose displayed'
        elif p.suffix=='.json':
            payload=read(p)
            if isinstance(payload,list):row['documents_or_items']=len(payload);row['count_basis']='top-level array count'
            elif isinstance(payload,dict):
                for key in ('items','questions','prompts','cases','samples','records','results'):
                    if isinstance(payload.get(key),list):row['documents_or_items']=len(payload[key]);row['count_basis']=f'{key} array count';break
                if row['documents_or_items'] is None:row['count_basis']='metadata/result object, no unambiguous item population; not an independent clean set'
    audit['audit_scope']+=' Supplement: folder-based evaluation/holdouts/benchmarks/blind inventory; non-protected dataset structures counted locally, no prose displayed. FinalBlind remained hash-only.'
    write_json(OUT/'data-provenance.json',audit)
    s=read(ROOT/'evaluation/foundation-v41-confirmatory-summary.json');s['data_provenance_sha256']=file_sha256(OUT/'data-provenance.json')
    write_json(ROOT/'evaluation/foundation-v41-confirmatory-summary.json',s)
    print('Provenance count/coverage supplement:',len(audit['rows']),'files',flush=True)


def inventory():
    resolver_gate()
    files=[]
    for base in (ROOT/'data',ROOT/'evaluation'):
        for p in base.rglob('*'):
            rel=p.relative_to(ROOT).as_posix()
            if not p.is_file() or '/raw/' in rel or re.search(r'/phase\d+/',rel):continue
            if re.search(r'(validation|test|holdout|blind|human|fixed_prompts|base-completion)',p.name,re.I) and (p.suffix in ('.json','.jsonl','.bin') or p.name.endswith('.jsonl.gz')):
                if base.name=='evaluation' and not p.name.startswith(('fixed_prompts','human-','test_prompts')):continue
                files.append(p)
    # Code/report references only; never search FinalBlind content or large corpora.
    sources=[]
    for folder in ('training','scripts','evaluation','docs','tests','foundation'):
        for p in (ROOT/folder).rglob('*'):
            if p.suffix in ('.py','.md') and 'v41' not in p.name:
                sources.append((p.relative_to(ROOT).as_posix(),p.read_text(encoding='utf-8',errors='replace')))
    packed=read(ROOT/'data/foundation_v11/packed/vocab-4096/manifest.json')
    rows=[]
    for path in sorted(set(files)):
        rel=path.relative_to(ROOT).as_posix();refs=[]
        for name,text in sources:
            for no,line in enumerate(text.splitlines(),1):
                if rel in line or (path.name.startswith(('fixed_prompts','base-completion','human-')) and path.name in line):
                    refs.append({'path':name,'line':no,'excerpt':line.strip()[:350]})
        count=None;count_basis='not established without reading dataset content'
        # Do not open dataset bodies during provenance audit; counts from manifests only.
        for split,meta in packed['splits'].items():
            if rel==meta['path'] or rel==f'data/foundation_v11/documents/{split}.jsonl.gz':count=meta['documents'];count_basis='packed manifest documents'
        if path==FORBIDDEN:count=1000;count_basis='declared user-protected item count; hash only'
        if path.name=='base-completion-50.json':count=50;count_basis='build_prompts count assertion'
        known='foundation_v11' in rel and ('validation' in rel or 'test' in rel or 'base-completion' in rel)
        if 'foundation_v11' in rel and 'test' in rel:
            refs.extend([{'path':'evaluation/evaluate_foundation_v11_completion.py','line':58,'excerpt':'test.jsonl.gz used to build heldout_continuation prompts'},
                {'path':'evaluation/investigate_foundation_v14.py','line':1038,'excerpt':'base-completion-50 reused in language-emergence failure diagnosis'}])
        forbidden=path==FORBIDDEN
        history=subprocess.check_output(['git','log','--all','--reverse','--format=%h %s','--',rel],cwd=ROOT,text=True).splitlines()
        rows.append({'name':path.name,'path':rel,'sha256':file_sha256(path),'documents_or_items':count,'count_basis':count_basis,
            'first_phase_used':'Foundation v1.1 (defa0da); diagnostic reuse v1.4 onward' if known else 'PROTECTED_NOT_USED' if forbidden else 'NOT_ESTABLISHED; see first tracked commit and references',
            'first_tracked_commit':history[0] if history else None,
            'purposes_used':['generation/completion and failure diagnosis; subsequent architecture/LR decisions'] if known else ['excluded by user; hash only'] if forbidden else ['historical references found; exact execution/selection influence not independently established'],
            'selection_influence':'YES' if known else 'NO' if forbidden else 'UNKNOWN',
            'confirmatory_eligible':'NO','reason':'User-protected FinalBlind excluded' if forbidden else 'Entire split already exposed to decision/diagnostic use; do not carve remaining documents post hoc' if known else 'Lack of recorded use is not proof of a clean, sealed set; no first-use custody/access record',
            'references':refs[:25]})
    # Repository-wide history searches are evidence of use/provenance, not a guarantee of off-repo non-use.
    history={needle:subprocess.check_output(['git','log','--all','--format=%h %s','-S',needle,'--','*.py','*.md'],cwd=ROOT,text=True).splitlines() for needle in ('documents/test.jsonl.gz','base-completion-50.json','validation.bin','fixed_prompts')}
    record={'phase':52,'availability':'PROVENANCE_UNCERTAIN','rows':rows,'history_searches':history,
        'audit_scope':'All candidate-named datasets under data, fixed/human prompt banks under evaluation; current code/reports plus all local Git refs. Dataset contents not opened; hashes and metadata only. Off-repository access history cannot be proved.',
        'clean_set_found':False,'unknown_policy':'UNKNOWN selection influence is explicitly unresolved, never silently NO or eligible. Item counts not in manifests remain null rather than invented.',
        'known_foundation_test_ineligible':True,'known_validation_ineligible':True,'final_blind_content_opened':False}
    new_json(OUT/'data-provenance.json',record)
    new_json(OUT/'confirmatory-preregistration.json',{'phase':52,'status':'NOT_REGISTERED_NO_PROVEN_CLEAN_SET','availability':record['availability'],
        'reason':'Foundation validation and test-derived completion prompts already used; other candidate banks lack verifiable fresh custody. FinalBlind prohibited. No post-hoc gate/population created.',
        'confirmatory_evaluation_started':False,'checkpoint_set_if_future_authorized':['C/B 16.384M seeds42/123/2026'],
        'future_holdout_design':{'collection':'Independent licensed/consented Japanese documents and student-authored prompts not in existing corpora/banks; no collection performed now',
            'custody':'Independent custodian seals IDs/content hashes/access log before model outputs; deduplicate source IDs and text against all train/eval corpora',
            'sampling':'Stratified source/topic/document-length design fixed before labels or model scores; sample size determined prospectively by power/precision simulation, not observed candidate advantage',
            'human':'Randomized anonymized C/B outputs, multiple independent raters, written naturalness/relevance/completion rubric; disclose disagreement and adjudication',
            'analysis':'Preregister paired document/prompt metrics, multiplicity policy, noninferiority safety margins, bootstrap RNG, fixed decoder and missing-data handling before first model access',
            'promotion':'One-shot held-out confirmatory readout; failed/uncertain results remain failures; no canonical permission from confirmation alone'}})
    summary={'phase':52,'availability':record['availability'],'confirmatory_set':None,'confirmatory_gate':'CONFIRMATORY_DATA_INVALID',
        'gate_meaning':'No admissible fresh set was established; evaluation not run (not a measured model failure)',
        'eos_confirmatory':'NOT_RUN','core_confirmatory':'NOT_RUN','sampling_confirmatory':'NOT_RUN',
        'formal_lr_candidate_gate':'FORMAL_LR_UNRESOLVED','new_training':False,'canonical_training':False,'20m_permission':False,'foundation_base':False,
        'resolver_gate':'PROCESS_ENV_Z_ROOT_PASS','z_root':str(resolver_gate()['destination_root']),'final_blind_sha_only':True,
        'copy':0,'move':0,'delete':0,'overwrite':0,'data_provenance_sha256':file_sha256(OUT/'data-provenance.json'),
        'prior_evidence':'PHASE51 comparative preference5e-5; not confirmed by independent evidence. Existing EOS/Core/Sampling failures and100% greedy runaway remain.'}
    new_json(ROOT/'evaluation/foundation-v41-confirmatory-summary.json',summary)
    print('Fresh audit:',record['availability'],len(rows),'candidate files; no confirmatory inference',flush=True)


def attractor():
    resolver_gate()
    old=read(ROOT/'evaluation/foundation-v40-eos-core-sampling-summary.json')
    spec=read(ROOT/'evaluation/phase51/preregistration.json')
    plan={'source':'PHASE51 fixed900 greedy traces; observational reuse only',
        'families':'Retain existing prefix taxonomy and lengths; no inferred factual/procedural/intention relabeling',
        'onset_window':'up to32 generated tokens before detected onset, plus onset; early onsets have fewer than32 (no invented trace)',
        'ngram_selection':'Top20 exact generated loop cycles by example count; length1..32, from all9 checkpoints; counts in frozen training corpus, not pasted source prose',
        'causal_policy':'Association only. No intervention proves EOS/position/data cause. MIXED_OR_UNKNOWN unless mechanism independently identified.',
        'decoder_mitigation':'Reuse existing temperature.7 observations only; no new decoder parameter search or beta claim'}
    if (OUT/'attractor-analysis-plan.json').exists():assert read(OUT/'attractor-analysis-plan.json')==plan
    else:new_json(OUT/'attractor-analysis-plan.json',plan)
    traces=[];cycles=Counter();by_checkpoint={};families={};used=[]
    for arm in ARMS:
        for seed in SEEDS:
            path=ROOT/f'evaluation/phase51/raw/{arm}-seed-{seed}.json'
            expected=next(r['sha256'] for r in old['raw_artifacts'] if Path(r['path'])==path.relative_to(ROOT))
            assert file_sha256(path)==expected
            raw=read(path);assert raw['complete'] and raw['integrity']['pass']
            assert file_sha256(Path(raw['checkpoint']))==raw['checkpoint_sha256']
            used.append({'checkpoint':raw['checkpoint'],'sha256':raw['checkpoint_sha256'],'integrity':'PASS; matching SHA, reuse strict/resume check'})
            rows=[]
            for i,r in enumerate(raw['greedy']['rows']):
                onset=r['loop']['loop_onset'];width=r['loop']['loop_length'];start=(onset or 1)-1
                cycle=tuple(r['ids'][start:start+width]) if width else ()
                if cycle:cycles[cycle]+=1
                series=[]
                for j,t in enumerate(r['trace']):
                    prefix=r['ids'][:j+1]
                    series.append({'step':j+1,'token_id':prefix[-1],'entropy':t['entropy'],
                        'top1_probability':t['top5'][0]['probability'],'top2_probability':t['top5'][1]['probability'],
                        'margin':t['top1_top2_margin'],'eos_probability':t['eos_probability'],
                        'unique_token_ratio':len(set(prefix))/len(prefix),'repeated_ngrams':{str(n):ngram_repetition(prefix,n) for n in range(1,5)}})
                window=series[max(0,start-32):start+1]
                item={'arm':arm,'seed':seed,'prompt_id':spec['prompts'][i]['id'],'family':spec['prompts'][i]['taxonomy'],
                    'prefix_length':spec['prompts'][i]['prefix_length'],'loop_onset':onset,'cycle_length':width,'cycle_token_ids':list(cycle),
                    'pre_onset_window':window,'full_time_series':series}
                traces.append(item);rows.append(item)
            by_checkpoint[f'{arm}-{seed}']={'mean_loop_onset':float(np.mean([r['loop_onset'] for r in rows])),
                'mean_cycle_length':float(np.mean([r['cycle_length'] for r in rows])),
                'at_onset':{k:float(np.mean([r['pre_onset_window'][-1][k] for r in rows])) for k in ('entropy','top1_probability','top2_probability','margin','eos_probability','unique_token_ratio')},
                'sampling_reused':{rng:v['metrics'] for rng,v in raw['sampling'].items()}}
    for family in sorted({r['family'] for r in traces}):
        subset=[r for r in traces if r['family']==family]
        families[family]={'examples':len(subset),'mean_onset':float(np.mean([r['loop_onset'] for r in subset]))}
    trainpath=ROOT/'data/foundation_v11/packed/vocab-4096/train.bin'
    manifest=read(ROOT/'data/foundation_v11/packed/vocab-4096/manifest.json')
    assert file_sha256(trainpath)==manifest['splits']['train']['sha256']
    train=np.memmap(trainpath,dtype=np.uint16,mode='r');ngram_rows=[]
    for cycle,n in cycles.most_common(20):
        hits=0;width=len(cycle)
        for offset in range(0,len(train)-width+1,1000000):
            size=min(1000000,len(train)-width+1-offset);mask=np.ones(size,dtype=bool)
            for j,token in enumerate(cycle):mask &= train[offset+j:offset+j+size]==token
            hits+=int(mask.sum())
        ngram_rows.append({'token_ids':list(cycle),'loop_examples':n,'train_occurrences':hits,'occurrences_per_million_windows':hits/max(1,len(train)-width+1)*1e6})
    new_json(OUT/'raw/attractor-time-series.json',traces)
    compact=[{k:v for k,v in r.items() if k not in ('full_time_series','pre_onset_window')}|{'pre_onset_steps_available':len(r['pre_onset_window'])-1,
        'before32_or_first':r['pre_onset_window'][0],'at_onset':r['pre_onset_window'][-1]} for r in traces]
    summary={'phase':52,'classification':'MIXED_OR_UNKNOWN','attractor_gate':'ATTRACTOR_CAUSE_STILL_UNKNOWN',
        'evidence':{'local_ngram_attractor':'Observed repetitive cycles in900/900 greedy examples; descriptive, not causal intervention',
            'eos_suppression':'Low EOS probability at onset; association, no ablation proves cause',
            'overconfident_top1':'Top1/Top2, entropy and margins recorded; probability concentration alone does not establish causal overconfidence',
            'position_context':'Prefix lengths/taxonomy retained; content and length confounded, no randomized position intervention',
            'data_repetition_signature':'Top20 generated cycle frequencies measured in train; frequency is not proof of memorized-loop causation'},
        'by_checkpoint':by_checkpoint,'families':families,'prompt_onset_comparisons':compact,'top20_cycle_train_counts':ngram_rows,
        'train_sha256':file_sha256(trainpath),'source_provenance':'Existing Foundation v1.1 manifest and data audit retained; Wikimedia/Wikibooks source licensing unchanged; no source prose displayed',
        'raw_timeseries_path':'evaluation/phase52/raw/attractor-time-series.json','raw_timeseries_sha256':file_sha256(OUT/'raw/attractor-time-series.json'),
        'decoder_mitigation':'No new decoder experiment. Existing temperature.7 multi-RNG sampling is descriptive; no claim of model repair or sufficient Beta safety.',
        'checkpoint_integrity':used,'new_training':False,'new_gpu_inference':False,'foundation_base_complete':False}
    new_json(ROOT/'evaluation/foundation-v41-attractor-summary.json',summary)
    print('Attractor audit complete;900 traces; no new GPU inference/training',flush=True)


if __name__=='__main__':
    if sys.argv[1:] == ['provenance']:inventory()
    elif sys.argv[1:] == ['attractor']:attractor()
    elif sys.argv[1:] == ['counts']:complete_counts()
    elif sys.argv[1:] == ['logits']:logit_dynamics()
    else:raise SystemExit('Use provenance or attractor')
