"""PHASE55: exclusively create v2 metadata/splits; NEVER open retired Reserve.

Commands are deliberately separate: freeze MUST precede build.
The old supplemental pool must not be reopened after Reserve2 sealing.
"""
from __future__ import annotations
import hashlib, os, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.diagnose_foundation_v40 import read,new_json,file_sha256,resolver_gate,BLIND_SHA
from evaluation.build_foundation_v42_holdout import norm,digest,now
from foundation.base_tokenizer import FoundationTokenizer
OUT=ROOT/'evaluation/phase55'
STORE=Path(r'Z:\AI\unipilot-mini\data\phase55-fresh-v2')
POOL=Path(r'Z:\AI\unipilot-mini\data\phase54-supplemental\unqualified-pool.json')

def gate():
    assert os.environ.get('UNIPILOT_CHECKPOINT_ROOT')==r'Z:\AI\unipilot-mini\checkpoints','PROCESS_ENV_RESOLVER_MISMATCH'
    return resolver_gate()

def grams(text):
    text=norm(text)
    return {text[i:i+5] for i in range(max(0,len(text)-4))}

def fingerprint(gs,k=256):
    # Domain-separated one-way hashes, never store plaintext ngrams.
    hashes=sorted({hashlib.sha256(('phase55-char5|'+g).encode()).hexdigest()[:16] for g in gs})
    return {'algorithm':'SHA256-64 bottom-k KMV','k':k,'distinct_grams':len(gs),'values':hashes[:k]}

def sketch_jaccard(a,b):
    limit=min(a['k'],b['k']); union=sorted(set(a['values'])|set(b['values']))[:limit]
    return sum(x in a['values'] and x in b['values'] for x in union)/max(1,len(union))

def freeze():
    m=gate()
    old=read(ROOT/'evaluation/phase53/fresh-holdout-manifest.json')['splits']['future-reserve']
    assert file_sha256(Path(old['path']))==old['sha256'] # bytes hash ONLY
    new_json(OUT/'phase53-reserve-retirement.json',{'phase':55,'status':'RETIRED_UNSCORABLE','documents':27,
        'path':old['path'],'sha256':old['sha256'],'retired_at':now(),
        'reason':'No presealed near-duplicate fingerprints; independence cannot be established without violating seal.',
        'permanent_prohibited_uses':['validation','test','confirmatory','future blind','model selection','threshold setting','hyperparameter setting','quality evaluation'],
        'body_opened':False,'body_scored':False,'prompt_generation':False,'fingerprint_reconstruction':False,
        'supplemental_near_relation':'UNKNOWN; excluded from eligibility requirement by PHASE55 user authorization; never claim independence between these sets'})
    inputs=[ROOT/'evaluation/phase54'/p for p in ('acquisition-plan.json','supplemental-provenance.json','leakage-report.json','fresh-holdout-manifest.json')]
    new_json(OUT/'holdout-preregistration.json',{'phase':55,'frozen_at':now(),'internal_results_seen':False,
        'pool_path':str(POOL),'pool_sha256':file_sha256(POOL),'inputs':{str(p.relative_to(ROOT)):file_sha256(p) for p in inputs},
        'internal_rule':'All unordered pairs: exact SHA, normalized SHA; near if full normalized-character5gram set Jaccard >= 0.8 (no SimHash prefilter). Connected components retain lowest page/revision ID.',
        'fingerprint':'Before split: SHA256-64 domain phase55-char5|, bottom256 KMV; stored hashes only. Future candidates compare shared hashes in smallest256 union. Estimated Jaccard >=0.75 flags exclusion/review. Sketch estimate is not proof of no paraphrase; low-entropy grams may permit dictionary membership inference, never raw text reconstruction.',
        'split_seed':'phase55-v2-5501','split_algorithm':'Within each category sort SHA256(seed|page_id|revision_id); floor(.5*n) diagnostic, floor(.3*n) confirmatory, remainder reserve2. Categories n<6: whole category to diagnostic to avoid artificial small strata. Final lists sort by numeric page/revision ID.',
        'ready_documents_min':250,'ready_tokens_min':200000,'reacquisition':False,'window_change':False,
        'historical_reuse':'PHASE54 historical exact/normalized/near + PHASE53 identity metadata; verify all source file hashes.',
        'retired_reserve_near_comparison':'NOT_REQUIRED_UNKNOWN','new_training':False})
    new_json(OUT/'preflight.json',{'resolver_gate':'PROCESS_ENV_Z_ROOT_PASS','root':os.environ['UNIPILOT_CHECKPOINT_ROOT'],
        'protected_files':m['protected_files'],'final_blind_sha_only':BLIND_SHA,
        'checkpoint_copy_move_delete_overwrite':[0,0,0,0],'expected_head':'10d74480cf059b6ef4d1484675a72de5e09a2c13'})
    print('Retired Reserve unopened; internal/split preregistration frozen',flush=True)

def build():
    gate(); assert not (OUT/'fresh-holdout-v2-manifest.json').exists(),'Already sealed: do not reopen pool'
    spec=read(OUT/'holdout-preregistration.json')
    assert file_sha256(POOL)==spec['pool_sha256']
    for p,sha in spec['inputs'].items():assert file_sha256(ROOT/p)==sha,p
    historical=read(ROOT/'evaluation/phase54/leakage-report.json')
    for r in historical['input_files']:assert file_sha256(ROOT/r['path'])==r['sha256'],r['path']
    meta=read(ROOT/'evaluation/phase54/supplemental-provenance.json')['documents']
    pool=read(POOL);assert len(pool)==len(meta)==591
    tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json')
    for r,m in zip(pool,meta):
        assert {k:v for k,v in r.items() if k!='text'}==m
        assert digest(r['text'])==r['content_sha256'] and digest(norm(r['text']))==r['normalized_sha256']
        assert len(tok.encode(r['text']))+2==r['tokens']
        assert all(r.get(k) for k in ('revision_id','page_id','category','license','license_url','canonical_source_reference','retrieved_at','revision_timestamp'))
    gs=[grams(r['text']) for r in pool];parent=list(range(len(pool)));pairs=[]
    def find(i):
        while parent[i]!=i:i=parent[i]
        return i
    for i in range(len(pool)):
        for j in range(i):
            reason='exact' if pool[i]['content_sha256']==pool[j]['content_sha256'] else 'normalized' if pool[i]['normalized_sha256']==pool[j]['normalized_sha256'] else None
            intersection=len(gs[i]&gs[j]);sim=intersection/max(1,len(gs[i])+len(gs[j])-intersection)
            if reason or sim>=.8:
                pairs.append({'a':pool[i]['page_id'],'b':pool[j]['page_id'],'kind':reason or 'near','jaccard':sim});parent[find(i)]=find(j)
    components=defaultdict(list)
    for i in range(len(pool)):components[find(i)].append(i)
    keep=sorted((min(v,key=lambda i:(pool[i]['page_id'],pool[i]['revision_id'])) for v in components.values()),key=lambda i:pool[i]['page_id'])
    eligible=[pool[i] for i in keep];metadata=[{**meta[i],'document_id':f"jawiki:{pool[i]['page_id']}:{pool[i]['revision_id']}",'near_fingerprint':fingerprint(gs[i])} for i in keep]
    n=len(eligible);tokens=sum(r['tokens'] for r in eligible)
    state='FRESH_HOLDOUT_V2_READY' if n>=250 and tokens>=200000 else 'FRESH_SET_INVALID'
    new_json(OUT/'fresh-holdout-v2-leakage.json',{'gate':state,'preregistration_sha256':file_sha256(OUT/'holdout-preregistration.json'),
        'historical_checks':'PASS_REUSED_INPUT_SHA_VERIFIED','historical_texts':historical['historical_texts'],
        'historical_rule':historical['historical_near_duplicate_rule'],'internal_pairs_checked':n*(n-1)//2 if not pairs else len(pool)*(len(pool)-1)//2,
        'duplicate_pairs':pairs,'counts':{k:sum(r['kind']==k for r in pairs) for k in ('exact','normalized','near')},
        'excluded_documents':len(pool)-n,'provenance_complete':True,'license_complete':True,
        'retired_reserve_relation':'UNKNOWN; no present/future independence claim','FinalBlind_opened':False,'phase53_reserve_opened':False,
        'limitation':'Checks cover local known sources and stated lexical rules, not arbitrary paraphrases/off-repository contamination.'})
    new_json(OUT/'fresh-holdout-v2-fingerprints.json',{'created_before_sealing':True,'documents':metadata})
    splits={}
    if state=='FRESH_HOLDOUT_V2_READY':
        categories=defaultdict(list)
        for r in eligible:categories[r['category']].append(r)
        parts={k:[] for k in ('diagnostic','confirmatory','future-reserve2')}
        for cat,rows in sorted(categories.items()):
            rows=sorted(rows,key=lambda r:digest(f"{spec['split_seed']}|{r['page_id']}|{r['revision_id']}"))
            a=int(len(rows)*.5) if len(rows)>=6 else len(rows);b=a+int(len(rows)*.3) if len(rows)>=6 else a
            for name,chunk in zip(parts,(rows[:a],rows[a:b],rows[b:])):parts[name].extend(chunk)
        for name,rows in parts.items():
            rows.sort(key=lambda r:(r['page_id'],r['revision_id']));p=STORE/f'{name}.json';new_json(p,rows)
            ids=[f"jawiki:{r['page_id']}:{r['revision_id']}" for r in rows]
            splits[name]={'path':str(p),'sha256':file_sha256(p),'document_ids':ids,'membership_sha256':digest('\n'.join(ids)),
                'documents':len(rows),'tokens':sum(r['tokens'] for r in rows),'categories':dict(Counter(r['category'] for r in rows)),
                'state':'SEALED_WITH_FINGERPRINTS' if name=='future-reserve2' else 'FROZEN_UNSCORED','model_scored':False}
    new_json(OUT/'fresh-holdout-v2-manifest.json',{'phase':55,'gate':state,'documents':n,'tokens':tokens,'splits':splits,
        'split_seed':spec['split_seed'],'preregistration_sha256':file_sha256(OUT/'holdout-preregistration.json'),
        'fingerprints_sha256':file_sha256(OUT/'fresh-holdout-v2-fingerprints.json'),'reserve2_sealed_at':now(),
        'body_storage_git':False,'future_reserve2_inference':False,'do_not_reopen_pool_after_sealing':True,
        'diagnostic_primary_evidence':False,'new_training':False})
    print(state,n,tokens,{k:(r['documents'],r['tokens']) for k,r in splits.items()},flush=True)

if __name__=='__main__':{'freeze':freeze,'build':build}[sys.argv[1]]()
