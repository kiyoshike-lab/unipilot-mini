"""PHASE53 checkpoint-independent acquisition. Never reads Final Blind content."""
from __future__ import annotations
import gzip, hashlib, json, re, sys, unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from evaluation.diagnose_foundation_v40 import resolver_gate, new_json, read, file_sha256
from scripts.collect_foundation_v10_wikimedia import USER_AGENT, clean_extract
from foundation.mediawiki_cleaner import clean_mediawiki, strict_quality_reason
from scripts.prepare_foundation_v10 import simhash, category_for
from foundation.base_tokenizer import FoundationTokenizer

OUT = ROOT/'evaluation/phase53'
STORE = Path(r'Z:\AI\unipilot-mini\data\phase53-fresh')
API = 'https://ja.wikipedia.org/w/api.php'
BLIND = ROOT/'data/foundation_v09/evaluation/final-blind-1000.json'
def now(): return datetime.now(timezone.utc).isoformat()
def digest(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()
def norm(s): return re.sub(r'\s+', '', unicodedata.normalize('NFKC', s)).casefold()
def api(params):
    url=API+'?'+urlencode({'format':'json','formatversion':2,**params})
    with urlopen(Request(url,headers={'User-Agent':USER_AGENT}),timeout=30) as r: result=json.load(r)
    if 'error' in result: raise RuntimeError(str(result['error']))
    return result
def freeze():
    resolver_gate()
    reports=[ROOT/f'evaluation/foundation-v11-{p}-dump.json' for p in ('wikipedia','wikibooks')]
    sources=[{'path':str(p.relative_to(ROOT)),'sha256':file_sha256(p),**{k:read(p)[k] for k in ('dump_url','dump_sha256','retrieved_at','license')}} for p in reports]
    revisions=[];retrieved=[]
    for path in sorted((ROOT/'data/foundation_v11/documents').glob('*.gz')):
        with gzip.open(path,'rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                if r.get('revision_timestamp'):revisions.append(r['revision_timestamp'])
                if r.get('retrieved_at'):retrieved.append(r['retrieved_at'])
    cutoff=max(retrieved)
    plan={'phase':53,'frozen_at':now(),'checkpoint_outputs_seen':False,'source':API,
        'corpus_cutoff':{'retrieval_upper_bound':cutoff,'dump_date':'UNKNOWN: reports reference latest, not an immutable dated dump','revision_min':min(revisions),'revision_max':max(revisions),'sources':sources},
        'candidate_window':{'after':'2026-08-28T00:00:00Z','before':'2026-09-10T15:47:15Z','type':'new namespace0 pages, not merely downloaded today','maximum':500},
        'quality':'Existing strict_quality_reason and cleaner; 600..12000 cleaned characters; no output-dependent selection',
        'leakage':{'normalization':'NFKC whitespace removal casefold','exact_sha256':True,'source_identity':'exclude any known source page, revision or canonical URL',
            'near_duplicate':'existing 32-bit 5-character SimHash Hamming<=3 AND normalized character5-gram Jaccard>=0.8; deterministic candidate retrieval, no random sampling',
            'historical':'all Foundation v11 train/validation/test documents and text fields from PHASE52 provenance candidate banks; Final Blind excluded'},
        'split':'SHA256(phase53-v1|page_id) ascending; first floor(.6*N) diagnostic, next floor(.2*N) confirmatory, remainder reserve; group by page',
        'ready_rule':{'documents':250,'tokenizer_tokens':100000},'reserve':'metadata/hash only in Git; no model scoring or inference',
        'storage':str(STORE),'license':'CC BY-SA 4.0','license_url':'https://creativecommons.org/licenses/by-sa/4.0/',
        'license_policy':'https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use','no_external_ai':True,'new_training':False}
    assert cutoff < plan['candidate_window']['after']
    if (OUT/'acquisition-plan.json').exists(): raise RuntimeError('Plan already frozen; use acquire to resume')
    new_json(OUT/'acquisition-plan.json',plan)
    print('Acquisition plan frozen; retrieval bound',cutoff,flush=True)

def strings(value):
    if isinstance(value,dict):
        for k,v in value.items():
            if k in ('text','question','prompt','input','completion','answer','content') and isinstance(v,str) and len(v)>=40:yield v
            elif isinstance(v,(dict,list)):yield from strings(v)
    elif isinstance(value,list):
        for v in value:yield from strings(v)

def acquire():
    resolver_gate();plan=read(OUT/'acquisition-plan.json');STORE.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'fresh-holdout-manifest.json').exists(),'Completed/blocked run exists; inspect, do not overwrite'
    inventory=STORE/'candidates.json'
    errors=[]
    if not inventory.exists():
        params={'action':'query','list':'recentchanges','rctype':'new','rcnamespace':0,'rclimit':500,'rcprop':'title|ids|timestamp','rcstart':plan['candidate_window']['before'],'rcend':plan['candidate_window']['after']}
        try: result=api(params);new_json(inventory,result['query']['recentchanges'])
        except Exception as e: errors.append({'stage':'candidate_inventory','error':str(e),'at':now()})
    candidates=read(inventory) if inventory.exists() else []
    # Immutable initial revisions avoid updated pages slipping in after the selection freeze.
    for start in range(0,len(candidates),25):
        batch=candidates[start:start+25];target=STORE/f'revisions-{start:04d}.json'
        if target.exists(): read(target);continue
        try:
            result=api({'action':'query','prop':'revisions','revids':'|'.join(str(r['revid']) for r in batch),'rvprop':'ids|timestamp|content','rvslots':'main'})
            new_json(target,{'retrieved_at':now(),'pages':result['query']['pages']})
            print('Fetched initial revisions',start+len(batch),flush=True)
        except Exception as e:errors.append({'stage':f'revisions-{start}','error':str(e),'at':now()})
    if not candidates:
        finish(plan,[],[],{},errors,[]);return
    historical=[];known_ids=set();known_urls=set();input_files=[]
    for p in sorted((ROOT/'data/foundation_v11/documents').glob('*.gz')):
        input_files.append({'path':str(p.relative_to(ROOT)),'sha256':file_sha256(p)})
        with gzip.open(p,'rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);historical.append(r['text'])
                if 'wikipedia' in r.get('source_type',''):known_ids.add(r.get('page_id'))
                known_urls.add(r.get('source_url'))
    for row in read(ROOT/'evaluation/phase52/data-provenance.json')['rows']:
        p=ROOT/row['path']
        if p==BLIND or p.suffix!='.json' or not p.is_file():continue
        historical.extend(strings(read(p)));input_files.append({'path':str(p.relative_to(ROOT)),'sha256':file_sha256(p)})
    historical=list(dict.fromkeys(historical));exact={digest(t) for t in historical};normalized={digest(norm(t)) for t in historical}
    hashes=[];buckets=[defaultdict(list) for _ in range(4)]
    for i,t in enumerate(historical):
        h=simhash(t);hashes.append(h)
        for b in range(4):buckets[b][(h>>(8*b))&255].append(i)
        if i%5000==0:print('Leakage index',i,'/',len(historical),flush=True)
    tok=FoundationTokenizer.load(ROOT/'tokenizer/foundation-v11-base-4096.json');accepted=[];excluded=[];metadata=[];seen=set()
    gram_cache={}
    def grams(t):
        n=norm(t);return {n[i:i+5] for i in range(max(0,len(n)-4))}
    new_ids={r['pageid']:r for r in candidates}
    for p in sorted(STORE.glob('revisions-*.json')):
        response=read(p)
        for page in response['pages']:
            for revision in page.get('revisions',[]):
                candidate=new_ids.get(page['pageid'])
                if not candidate or revision['revid']!=candidate['revid']:continue
                source_text=revision.get('slots',{}).get('main',{}).get('content','')
                cleaned,_=clean_mediawiki(source_text);cleaned,_=clean_extract(cleaned)
                r={'page_id':page['pageid'],'revision_id':revision['revid'],'revision_timestamp':revision['timestamp'],'title':page['title'],
                   'project':'jawiki','source':API,'canonical_source_reference':f"https://ja.wikipedia.org/w/index.php?oldid={revision['revid']}",
                   'attribution_reference':f"https://ja.wikipedia.org/w/index.php?title={page['title']}&action=history",'retrieved_at':response['retrieved_at'],
                   'license':plan['license'],'license_url':plan['license_url'],'content_sha256':digest(cleaned),'source_content_sha256':digest(source_text),'normalized_sha256':digest(norm(cleaned)),
                   'modification':'MediaWiki markup and reference sections removed using existing Foundation cleaner', 'category':category_for({'title':page['title'],'categories':re.findall(r'\[\[(?:Category|カテゴリ):([^\]|]+)',source_text)})}
                reason=strict_quality_reason(page['title'],cleaned)
                if not 600<=len(cleaned)<=12000:reason='character_bound'
                if not plan['candidate_window']['after']<=revision['timestamp']<=plan['candidate_window']['before']:reason='timestamp_outside_frozen_window'
                if page['pageid'] in seen:reason='duplicate_candidate_page'
                seen.add(page['pageid'])
                if page['pageid'] in known_ids:reason='known_source_identity'
                if r['content_sha256'] in exact:reason='exact_duplicate'
                elif r['normalized_sha256'] in normalized:reason='normalized_duplicate'
                if not reason:
                    h=simhash(cleaned);possible=set()
                    for b in range(4):possible.update(buckets[b][(h>>(8*b))&255])
                    gs=None
                    for i in sorted(possible):
                        if (h^hashes[i]).bit_count()>3:continue
                        if gs is None:gs=grams(cleaned)
                        if i not in gram_cache:gram_cache[i]=grams(historical[i])
                        old=gram_cache[i];j=len(gs&old)/max(1,len(gs|old))
                        if j>=.8:reason='near_duplicate';break
                if reason:excluded.append({'page_id':page['pageid'],'revision_id':revision['revid'],'reason':reason});continue
                r['tokens']=len(tok.encode(cleaned))+2
                # Check pool duplicates by adding accepted candidates to the same historical index.
                i=len(historical);historical.append(cleaned);hashes.append(h);exact.add(r['content_sha256']);normalized.add(r['normalized_sha256'])
                for b in range(4):buckets[b][(h>>(8*b))&255].append(i)
                accepted.append({**r,'text':cleaned});metadata.append(r)
    finish(plan,accepted,excluded,{'historical_texts':len(historical)-len(accepted),'input_files':input_files},errors,metadata)

def finish(plan,accepted,excluded,index,errors,metadata):
    ordered=sorted(accepted,key=lambda r:digest('phase53-v1|'+str(r['page_id'])))
    n=len(ordered);a=int(.6*n);b=a+int(.2*n);parts={'diagnostic':ordered[:a],'confirmatory':ordered[a:b],'future-reserve':ordered[b:]}
    splits={}
    for name,rows in parts.items():
        path=STORE/f'{name}.json'
        new_json(path,rows)
        splits[name]={'path':str(path),'sha256':file_sha256(path),'documents':len(rows),'tokens':sum(r['tokens'] for r in rows),'model_scored':False,'consumed_for_model_selection':False}
    tokens=sum(r['tokens'] for r in accepted)
    gate='FRESH_HOLDOUT_READY' if n>=250 and tokens>=100000 else 'FRESH_ACQUISITION_BLOCKED' if not n and errors else 'FRESH_HOLDOUT_TOO_SMALL'
    new_json(OUT/'fresh-data-provenance.json',{'plan_sha256':file_sha256(OUT/'acquisition-plan.json'),'corpus_cutoff':plan['corpus_cutoff'],'source':API,'license':plan['license'],'documents':metadata,'errors':errors})
    new_json(OUT/'leakage-report.json',{'plan_sha256':file_sha256(OUT/'acquisition-plan.json'),**index,'excluded':excluded,'counts':dict(Counter(r['reason'] for r in excluded)),'FinalBlind_content_opened':False,'limitations':'Local known text/source matching is not proof against undetectable paraphrases or off-repository use; only initial post-cutoff revisions admitted.'})
    new_json(OUT/'fresh-holdout-manifest.json',{'gate':gate,'documents':n,'tokens':tokens,'splits':splits,'plan_sha256':file_sha256(OUT/'acquisition-plan.json'),'reserve_sealed':True,'checkpoint_scoring_started':False,'new_training':False,'errors':errors})
    print('Fresh Holdout Gate',gate,n,'documents',tokens,'tokens',flush=True)

if __name__=='__main__':
    if sys.argv[1:] == ['freeze']:freeze()
    elif sys.argv[1:] == ['acquire']:acquire()
    else:raise SystemExit('Use freeze or acquire')
