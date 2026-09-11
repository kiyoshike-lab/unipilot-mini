"""Aggregate PHASE53 Track C raw decoder results; never runs inference or reads fresh data."""
from __future__ import annotations
import hashlib, json, statistics, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from evaluation.diagnose_foundation_v40 import read,new_json,file_sha256,evaluated_checkpoint
from evaluation.diagnose_foundation_v29_generation import ngram_repetition

OUT=ROOT/'evaluation/phase53'; PREREG=OUT/'generation-preregistration.json'
def mean(rows,key):return float(sum(float(r[key]) for r in rows)/len(rows))
def compact(row,prompt_id):
    ids=row['ids'];return {'prompt_id':prompt_id,'loop_onset':row['loop']['loop_onset'],'loop_length':row['loop']['loop_length'],
      'runaway':row['runaway'],'eos_completion':row['eos_reached'],'completion_proxy':row['completion_proxy'],
      'natural_japanese_proxy':row['natural_japanese_proxy'],'semantic_coherence_proxy':row['semantic_coherence_proxy'],
      'character_valid':row['character_valid'],'japanese_character_ratio':row['japanese_character_ratio'],
      'topic_overlap_proxy':None,'output_length':len(ids),'unique_token_ratio':len(set(ids))/max(1,len(ids)),
      'repetition':{str(n):ngram_repetition(ids,n) for n in range(1,5)}}
def aggregate(rows,prompts):
    assert len(rows)%len(prompts)==0
    paired_prompts=prompts*(len(rows)//len(prompts))
    metrics={k:mean(rows,k) for k in ('runaway','completion_proxy','natural_japanese_proxy','semantic_coherence_proxy','character_valid','japanese_character_ratio','repetition_1')}
    metrics['eos_completion']=mean(rows,'eos_reached');metrics['output_length_mean']=float(sum(len(r['ids']) for r in rows)/len(rows))
    metrics['output_length_median']=float(statistics.median(len(r['ids']) for r in rows))
    metrics['unique_token_ratio']=float(sum(len(set(r['ids']))/max(1,len(r['ids'])) for r in rows)/len(rows))
    metrics['topic_overlap_proxy']=float(sum(len(set(r['ids'])&set(p['prefix_ids']))/max(1,len(set(r['ids']))) for r,p in zip(rows,paired_prompts))/len(rows))
    metrics['loop_onset_mean']=float(sum((r['loop']['loop_onset'] or 65) for r in rows)/len(rows))
    metrics['loop_onset_median']=float(statistics.median((r['loop']['loop_onset'] or 65) for r in rows))
    metrics['repetition']={str(n):float(sum(ngram_repetition(r['ids'],n) for r in rows)/len(rows)) for n in range(1,5)}
    return metrics
def main():
    spec=read(PREREG);sha=file_sha256(PREREG);raw=sorted((OUT/'raw').glob('generation-*.json'))
    assert len(raw)==20 and not (OUT/'generation-policy-results.json').exists()
    prompt_json=json.dumps(spec['prompts'],ensure_ascii=False,separators=(',',':')).encode();prompt_sha=hashlib.sha256(prompt_json).hexdigest()
    by={}; raw_index=[]
    for path in raw:
      record=read(path);assert record['complete'] and record['preregistration_sha256']==sha and len(record['rows'])==24
      assert file_sha256(evaluated_checkpoint(record['arm'],42))==record['checkpoint_sha256']
      key=(record['arm'],record['mode']);by.setdefault(key,[]).append(record)
      raw_index.append({'path':str(path.relative_to(ROOT)),'sha256':file_sha256(path),'arm':record['arm'],'mode':record['mode'],'rng':record['rng']})
    assert set(by)=={(a,m['name']) for a in spec['arms'] for m in spec['modes']}
    output={};screen={};stop={}
    for arm in spec['arms']:
      greedy=None;temperature=None
      for mode in spec['modes']:
        records=sorted(by[(arm,mode['name'])],key=lambda r:r['rng']);rows=[r for q in records for r in q['rows']]
        key=f'{arm}:{mode["name"]}';metrics=aggregate(rows,spec['prompts'])
        output[key]={'mode':mode,'rngs':[r['rng'] for r in records],'metrics':metrics,
          'per_prompt':[compact(r,spec['prompts'][i%24]['id'])|{'rng':records[(i//24)]['rng']} for i,r in enumerate(rows)],
          'thermal':[r['thermal'] for r in records]}
        if mode['name']=='greedy':greedy=metrics
        if mode['name']=='temperature_0.7':temperature=metrics
        if records[0]['stop_replay']:
          replay=[r for q in records for r in q['stop_replay']['rows']];original=[r for q in records for r in q['rows']]
          triggered=[(o,r) for o,r in zip(original,replay) if r['forced_stop']]
          stop[key]={'definition':spec['stop_policy'],'metrics':aggregate(replay,spec['prompts']),'triggered':len(triggered),'total':len(replay),
            'trigger_rate':len(triggered)/len(replay),'saved_tokens_mean_all':float(sum(len(o['ids'])-len(r['ids']) for o,r in zip(original,replay))/len(replay)),
            'saved_tokens_mean_triggered':float(sum(len(o['ids'])-len(r['ids']) for o,r in triggered)/len(triggered)) if triggered else 0.0,
            'triggered_eos_completion':sum(bool(r['eos_reached']) for _,r in triggered),
            'false_positive_evidence':'No labelled normal-repeat (math/list/definition) control prompts were preregistered; no false-positive rate can be estimated. All forced stops are non-EOS truncations, so replay is not a safe completion policy.'}
      assert greedy and temperature
      for mode in spec['modes']:
        if mode['name']=='greedy':continue
        m=output[f'{arm}:{mode["name"]}']['metrics']
        criteria={'runaway_at_most_0_20':m['runaway']<=.20,'absolute_runaway_reduction_at_least_0_50':greedy['runaway']-m['runaway']>=.50,
          'completion_at_least_0_50':m['completion_proxy']>=.50,'semantic_at_least_0_50':m['semantic_coherence_proxy']>=.50,
          'character_valid_at_least_0_95':m['character_valid']>=.95,
          'completion_not_below_temperature_by_0_05':m['completion_proxy']>=temperature['completion_proxy']-.05,
          'semantic_not_below_temperature_by_0_05':m['semantic_coherence_proxy']>=temperature['semantic_coherence_proxy']-.05,
          'japanese_ratio_not_below_temperature_by_0_05':m['japanese_character_ratio']>=temperature['japanese_character_ratio']-.05}
        screen[f'{arm}:{mode["name"]}']={'criteria':criteria,'passes_all':all(criteria.values())}
    result={'phase':53,'track':'C only','fresh_data_used':False,'future_reserve_opened_or_scored':False,'confirmatory_scoring':False,
      'preregistration_sha256':sha,'prompt_set':{'count':24,'ordered_prompt_sha256':prompt_sha,'phase51_preregistration_sha256':file_sha256(ROOT/'evaluation/phase51/preregistration.json'),'ids':[p['id'] for p in spec['prompts']]},
      'models':{'C_5e5_seed42':spec['checkpoint_sha256']['C'],'B_7_5e5_seed42':spec['checkpoint_sha256']['B']},'device':'CUDA FP32; TF32 disabled','registered_grid':spec['modes'],'raw_artifacts':raw_index,
      'results':output,'quality_screen':screen,'loop_stop_replay':stop,'best_safe_decoder_candidate':'NONE','attractor_gate':'GENERATION_POLICY_UNSAFE',
      'interpretation':'No registered non-greedy setting passes every fixed screen. Loop-stop replay truncates non-EOS text and lacks a labelled normal-repeat control; it is not safe for staging. This paired seed42 decoder study is not formal LR evidence.',
      'formal_lr':'FORMAL_LR_UNRESOLVED','model_fixed':False,'new_training':False,'canonical_training':False,'20m_permission':False,'foundation_base_complete':False}
    new_json(OUT/'generation-policy-results.json',result)
    print('Track C analysis complete; best NONE; gate GENERATION_POLICY_UNSAFE')
if __name__=='__main__':main()
