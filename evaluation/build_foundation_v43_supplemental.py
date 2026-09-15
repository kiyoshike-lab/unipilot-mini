"""Bounded PHASE54 acquisition; sealed PHASE53 Reserve is never read.

No evaluation is permitted while the sealed-set near-duplicate check is unresolved.
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import build_foundation_v42_holdout as base
from evaluation.diagnose_foundation_v40 import new_json, read, resolver_gate, file_sha256
OUT = ROOT / 'evaluation/phase54'
STORE = Path(r'Z:\AI\unipilot-mini\data\phase54-supplemental')

def freeze():
    resolver_gate()
    old = read(ROOT/'evaluation/phase53/acquisition-plan.json')
    spec = {**old, 'phase':54, 'frozen_at':base.now(), 'storage':str(STORE),
        'candidate_window':{'after':'2026-08-28T00:00:00Z', 'before':base.now(), 'maximum':3000,
            'type':'namespace 0 recentchanges new pages; exact initial revision, newest first; fixed before retrieval'},
        'ready_rule':{'documents':250, 'tokenizer_tokens':200000},
        'split':'Only if all leakage checks PASS: SHA256(phase54-v1|page_id), 60/20/20; confirmatory >=50 documents and >=40000 tokens',
        'phase53_reserve_policy':'Hash/metadata only. No text, raw reconstruction or content-derived new fingerprints. Missing precomputed near-duplicate features blocks READY.',
        'phase53_metadata_sha256':file_sha256(ROOT/'evaluation/phase53/fresh-data-provenance.json'),
        'old_phase53_splits_unchanged':True, 'new_training':False}
    new_json(OUT/'acquisition-plan.json', spec)
    print('PHASE54 acquisition preregistered: 3000 candidates, 250 docs / 200000 tokens',flush=True)

def acquire():
    resolver_gate()
    spec=read(OUT/'acquisition-plan.json')
    old_meta=read(ROOT/'evaluation/phase53/fresh-data-provenance.json')['documents']
    old_manifest=read(ROOT/'evaluation/phase53/fresh-holdout-manifest.json')
    for split in old_manifest['splits'].values():
        assert file_sha256(Path(split['path']))==split['sha256']
    old_ids={r['page_id'] for r in old_meta}
    old_revisions={r['revision_id'] for r in old_meta}
    STORE.mkdir(parents=True,exist_ok=True)
    inventory=STORE/'candidates.json'
    excluded=[]
    if not inventory.exists():
        candidates=[]; continuation={}
        while len(candidates)<spec['candidate_window']['maximum']:
            slot=len(candidates)//500; target=STORE/f'candidate-page-{slot:02}.json'
            if target.exists(): result=read(target)
            else:
                result=base.api({'action':'query','list':'recentchanges','rctype':'new','rcnamespace':0,
                    'rclimit':min(500,spec['candidate_window']['maximum']-len(candidates)),
                    'rcprop':'title|ids|timestamp','rcstart':spec['candidate_window']['before'],
                    'rcend':spec['candidate_window']['after'], **continuation})
                new_json(target,result)
            candidates.extend(result['query']['recentchanges'])
            print('Candidate pages',len(candidates),flush=True)
            if 'continue' not in result:break
            continuation=result['continue']
        admitted=[]
        for row in candidates:
            if row['pageid'] in old_ids or row['revid'] in old_revisions:
                excluded.append({'page_id':row['pageid'],'revision_id':row['revid'],'reason':'phase53_identity'})
            else: admitted.append(row)
        new_json(inventory,admitted)
        new_json(STORE/'inventory-summary.json',{'requested_limit':3000,'returned':len(candidates),'phase53_identity_exclusions':excluded})
    # Reuse unchanged PHASE53 cleaner and historical near-duplicate rule. Its
    # only writing endpoint is redirected to the exclusively new PHASE54 store.
    base.OUT=OUT; base.STORE=STORE
    def finish(plan,accepted,rejected,index,errors,metadata):
        # SHA comparison for all129 old documents requires metadata only.
        old_exact={r['content_sha256'] for r in old_meta}
        old_normalized={r['normalized_sha256'] for r in old_meta}
        accepted2=[]; meta2=[]
        for row in accepted:
            reason='phase53_exact' if row['content_sha256'] in old_exact else 'phase53_normalized' if row['normalized_sha256'] in old_normalized else None
            if reason:rejected.append({'page_id':row['page_id'],'reason':reason})
            else:accepted2.append(row);meta2.append({k:v for k,v in row.items() if k!='text'})
        inventory_info=read(STORE/'inventory-summary.json')
        rejected.extend(inventory_info['phase53_identity_exclusions'])
        new_json(STORE/'unqualified-pool.json',accepted2)
        tokens=sum(r['tokens'] for r in meta2)
        gate='ACQUISITION_BLOCKED' if errors else 'SUPPLEMENTAL_TOO_SMALL' if len(meta2)<250 or tokens<200000 else 'LEAKAGE_RISK_TOO_HIGH'
        limitations=['PHASE53 sealed Reserve lacks precomputed SimHash/Jaccard features; mandated near-duplicate comparison cannot PASS from SHA/page metadata alone.',
            'No PHASE53 content was reopened. PHASE53 near-duplicate check remains NOT_VERIFIED across all129 documents.',
            'Historical source matching inherits PHASE53 provenance coverage; exact matching cannot detect arbitrary paraphrases.']
        new_json(OUT/'supplemental-provenance.json',{'phase':54,'source':base.API,'license':plan['license'],
            'plan_sha256':file_sha256(OUT/'acquisition-plan.json'),'candidate_inventory':inventory_info,
            'documents':meta2,'errors':errors,'corpus_cutoff':plan['corpus_cutoff']})
        new_json(OUT/'leakage-report.json',{**index,'counts':dict(Counter(r['reason'] for r in rejected)),
            'excluded':rejected,'phase53_exact_normalized_page_revision':'CHECKED_METADATA_ONLY',
            'phase53_near_duplicate':'NOT_VERIFIED','historical_near_duplicate_rule':plan['leakage']['near_duplicate'],
            'limitations':limitations,'FinalBlind_content_opened':False,'phase53_reserve_content_opened':False})
        new_json(OUT/'fresh-holdout-manifest.json',{'gate':gate,'documents':len(meta2),'tokens':tokens,
            'category_distribution':dict(Counter(r['category'] for r in meta2)),
            'splits':{},'new_diagnostic_size':0,'new_confirmatory_size':0,'future_reserve2_size':0,
            'future_reserve2_scored':False,'phase53_reserve':'UNTOUCHED',
            'checkpoint_scoring_started':False,'new_training':False,'limitations':limitations})
        print(gate,len(meta2),'documents',tokens,'tokens',flush=True)
    base.finish=finish
    base.acquire()

if __name__=='__main__':
    {'freeze':freeze,'acquire':acquire}[sys.argv[1]]()
