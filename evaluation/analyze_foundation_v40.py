"""Analyze frozen PHASE51 diagnostics. Reuse PHASE50 gates without retuning."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from evaluation.diagnose_foundation_v40 import (
    OUT, PREREG, SEEDS, ARMS, read, new_json, resolver_gate, cluster_ci, data,
    file_sha256, ngram_repetition,
)
from training.run_foundation_v37_stability import evaluation_path


def contribution(delta, probability, groups, train_counts, tokens, tokenizer, is_token):
    rows=[]
    for group in np.unique(groups):
        mask=groups==group
        freqs=train_counts[tokens[mask]]
        rows.append({'id':int(group),'occurrences':int(mask.sum()),'ce_delta':float(delta[mask].mean()),
            'probability_delta':float(probability[mask].mean()),'ce_sum_delta':float(delta[mask].sum()),
            'contribution_to_total_mean':float(delta[mask].sum()/len(delta)),
            'training_frequency':int(train_counts[group]) if is_token else {'min':int(freqs.min()),'max':int(freqs.max()),'occurrence_weighted_mean':float(freqs.mean())},
            **({'token_text':tokenizer.decode([int(group)],skip_special=True)} if is_token else {})})
    return sorted(rows,key=lambda r:r['ce_sum_delta'],reverse=True)


def positive_share(rows, n):
    positive=[max(0,r['ce_sum_delta']) for r in rows]
    return float(sum(positive[:n])/max(sum(positive),1e-30))


def core_analysis(spec, old):
    pop=read(ROOT/spec['frequency_population_file'])['core']
    assert file_sha256(ROOT/spec['frequency_population_file'])==spec['frequency_population_sha256']
    docs=np.array(pop['document_ids']);tokens=np.array(pop['token_ids'])
    assert (len(set(tokens)),len(tokens),len(set(docs)))==(236,4436,146)
    tok,_=data();train=np.memmap(ROOT/'data/foundation_v11/packed/vocab-4096/train.bin',dtype=np.uint16,mode='r')
    counts=np.bincount(train,minlength=4096)
    result={'population':{k:v for k,v in pop.items() if not isinstance(v,list)},'threshold_unchanged':.10,'arms':{}}
    for arm in ('C','B'):
        seed_rows={}
        for seed in SEEDS:
            base=read(ROOT/f'evaluation/phase49/frequency/baseline-seed-{seed}.json')['core']
            trial=read(ROOT/f'evaluation/phase49/frequency/{arm}-seed-{seed}.json')['core']
            assert base['population_sha256']==trial['population_sha256']==pop['sha256']
            delta=np.array(trial['values']['ce'])-base['values']['ce']
            probability=np.array(trial['values']['probabilities'])-base['values']['probabilities']
            cis={str(rng):cluster_ci(delta,docs,rng) for rng in spec['bootstrap']['seeds']}
            row={'mean':float(delta.mean()),'bootstrap_10000':cis,'ci_upper_range':[min(c['upper'] for c in cis.values()),max(c['upper'] for c in cis.values())],
                'old_gate':old['comparisons'][arm]['safety'][str(seed)]['checks']['core_ci']}
            if seed==2026:
                by_token=contribution(delta,probability,tokens,counts,tokens,tok,True)
                by_doc=contribution(delta,probability,docs,counts,tokens,tok,False)
                loo=[]
                for doc in np.unique(docs):
                    keep=docs!=doc
                    ci=cluster_ci(delta[keep],docs[keep])
                    loo.append({'excluded_document':int(doc),'remaining_occurrences':int(keep.sum()),'ci':ci,'upper_change':ci['upper']-cis['4900']['upper']})
                row.update({'tokens':by_token,'documents':by_doc,'top20_tokens':by_token[:20],'top20_documents':by_doc[:20],
                    'top20_token_positive_loss_share':positive_share(by_token,20),'top5_document_positive_loss_share':positive_share(by_doc,5),
                    'leave_one_document':sorted(loo,key=lambda r:r['upper_change']),
                    'loo_upper_range':[min(r['ci']['upper'] for r in loo),max(r['ci']['upper'] for r in loo)],
                    'loo_diagnostic_only':True,'loo_below_old_threshold_count':sum(r['ci']['upper']<=.10 for r in loo)})
            seed_rows[str(seed)]=row
        focused=seed_rows['2026']
        focused['classification']='DOCUMENT_CONCENTRATED' if focused['top5_document_positive_loss_share']>=.5 else 'TOKEN_CONCENTRATED' if focused['top20_token_positive_loss_share']>=.5 else 'SEED_LOCAL' if all(seed_rows[str(s)]['mean']<=0 for s in (42,123)) else 'BROAD'
        focused['seed_local_pattern']=all(seed_rows[str(s)]['mean']<=0 for s in (42,123)) and focused['mean']>0
        result['arms'][arm]=seed_rows
    return result


def mean_rows(rows, key): return float(np.mean([r[key] for r in rows]))


def eos_analysis(raw,old):
    result={'independent_document_ends':146,'terminal_occurrences':146,'old_requested_occurrences':500,
        'audit':'Old evaluator np.resize repeated 146 positions to 500, unequal document weighting; this diagnostic evaluates each once. Old nonterminal linspace did not exclude special markers; this diagnostic excludes EOS/BOS. No historical gate replaced.',
        'checkpoint_metrics':{},'comparison':{}}
    for arm in ARMS:
        for seed in SEEDS:
            r=raw[arm,seed];terminal=r['terminal'];nonterminal=r['nonterminal']
            assert len(terminal)==146 and len({x['document'] for x in terminal})==146
            result['checkpoint_metrics'][f'{arm}-{seed}']={'terminal':{key:mean_rows(terminal,key) for key in ('probability','rank','top1','top5','top10','competitor_minus_eos_logit','entropy')},
                'nonterminal':{'probability':mean_rows(nonterminal,'probability'),'premature_eos_top1':mean_rows(nonterminal,'top1'),'count':len(nonterminal)},'per_document':terminal}
    for arm in ('C','B'):
        comparisons={}
        for seed in SEEDS:
            a=raw[arm,seed]['terminal'];b=raw['baseline',seed]['terminal']
            delta=np.array([r['probability'] for r in a])-np.array([r['probability'] for r in b])
            ci=cluster_ci(delta,[r['document'] for r in a])
            losses=np.sort(np.maximum(-delta,0))[::-1]
            control=read(evaluation_path(arm,seed,256000))
            comparisons[str(seed)]={'delta_vs_baseline_ci':ci,'baseline_ratio':mean_rows(a,'probability')/mean_rows(b,'probability'),
                'top5_loss_share':float(losses[:5].sum()/max(losses.sum(),1e-30)),
                'old_256k_terminal':control['terminal_eos']['mean_probability'],
                'old_512k_terminal':old['comparisons'][arm]['legacy_metrics']['terminal_eos']['by_seed'][str(seed)],
                'old_gate':old['comparisons'][arm]['safety'][str(seed)]['checks']['eos']}
        negative=[r['delta_vs_baseline_ci']['upper']<0 for r in comparisons.values()]
        if all(negative) and all(r['baseline_ratio']<.9 for r in comparisons.values()): label='TRUE_EOS_REGRESSION'
        elif any(negative): label='SEED_LOCAL_EOS_VARIANCE'
        elif comparisons['2026']['top5_loss_share']>=.5: label='OUTLIER_DOCUMENTS'
        elif any(not r['old_gate'] for r in comparisons.values()): label='RELATIVE_THRESHOLD_SENSITIVITY'
        else: label='EOS_MEASUREMENT_UNCERTAINTY'
        result['comparison'][arm]={'classification':label,'by_seed':comparisons}
    return result


def sampling_analysis(raw, spec, old):
    result={'automatic_proxy_only':True,'determinism':all(r['determinism'] for r in raw.values()),
        'evaluator_audit':{'naturalness':'Deterministic regex/character/repetition thresholds; >=20 visible chars, Japanese >=.35, punctuation <=.25, newline <=.30, character repetition <.35 and content runs>=3.',
            'semantic':'Naturalness + sentence boundary + >=5 content runs; does not compare meaning to prompt. Semantic relevance cannot be established without human review.',
            'completion':'Naturalness + EOS or final punctuation; not proof of logical completeness.',
            'topic_retention':'Unique generated token overlap with prompt / unique generated token count; lexical proxy only.',
            'rng_device':'CUDA; same full batch geometry replayed exactly for base44000 at all nine checkpoints.',
            'historical_device':'Historical Phase46/48/49 generation used CPU generators (evaluation_execution.device=cpu). Identical numeric seeds do not imply identical CPU/CUDA samples; new matched comparisons use CUDA on both sides. Historical gate values remain untouched.',
            'old_gate_control':'Own-LR 256k single RNG retained; no replicated 256k inference performed, so historical FAIL is not nullified by new baseline comparison.'},
        'checkpoints':{},'comparisons':{}}
    keys=['natural_japanese_proxy','semantic_coherence_proxy']
    for arm in ARMS:
        for seed in SEEDS:
            item=raw[arm,seed]
            result['checkpoints'][f'{arm}-{seed}']={rng:{'metrics':v['metrics'],'rows':[{**{k:r[k] for k in ('prompt_id','sampling_seed','text','ids','natural_japanese_proxy','semantic_coherence_proxy','completion_proxy','character_valid','japanese_character_ratio','topic_retention_proxy','eos_reached','runaway')},'repetition':{str(n):ngram_repetition(r['ids'],n) for n in range(1,5)}} for r in v['rows']]} for rng,v in item['sampling'].items()}
    for arm in ('C','B'):
        comparisons={}
        for seed in SEEDS:
            stats={}
            for key in keys:
                deltas=[]
                for rng in spec['sampling_rng_bases']:
                    a=raw[arm,seed]['sampling'][str(rng)]['rows'];b=raw['baseline',seed]['sampling'][str(rng)]['rows']
                    deltas.append(np.array([r[key] for r in a],dtype=float)-np.array([r[key] for r in b],dtype=float))
                prompt=np.mean(deltas,axis=0);loss=np.sort(np.maximum(-prompt,0))[::-1]
                stats[key]={'paired_prompt_ci':cluster_ci(prompt,np.arange(100)),
                    'rng_delta_means':np.mean(deltas,axis=1).tolist(),'per_prompt_mean_delta':prompt.tolist(),
                    'top10_loss_share':float(loss[:10].sum()/max(loss.sum(),1e-30)),
                    'rng_straddles_minus_08':float(np.min(np.mean(deltas,axis=1)))<=-.08<=float(np.max(np.mean(deltas,axis=1)))}
            control=read(evaluation_path(arm,seed,256000))
            comparisons[str(seed)]={'metrics':stats,'old_256k_sampling':control['generation']['temperature_0.7'],
                'old_512k_sampling':{k:old['comparisons'][arm]['legacy_metrics'][k]['by_seed'][str(seed)] for k in ('naturalness','semantic')},
                'old_gate':old['comparisons'][arm]['safety'][str(seed)]['checks']['sampling']}
        true_seeds=[any(v['paired_prompt_ci']['upper']<-.08 for v in r['metrics'].values()) for r in comparisons.values()]
        priority=comparisons['123']['metrics']
        label='TRUE_SAMPLING_REGRESSION' if all(true_seeds) else 'SEED_LOCAL_VARIANCE' if any(true_seeds) else 'PROMPT_LOCAL_REGRESSION' if any(v['top10_loss_share']>=.5 for v in priority.values()) else 'SAMPLING_RNG_VARIANCE' if any(v['rng_straddles_minus_08'] for v in priority.values()) else 'EVALUATOR_NOISE'
        result['comparisons'][arm]={'classification':label,'by_seed':comparisons}
    return result


def greedy_analysis(raw):
    rows={f'{a}-{s}':raw[a,s]['greedy']['metrics'] for a in ARMS for s in SEEDS}
    avg={a:{k:float(np.mean([rows[f'{a}-{s}'][k] for s in SEEDS])) for k in rows[f'{a}-42']} for a in ARMS}
    delta={k:avg['C'][k]-avg['B'][k] for k in avg['C']}
    signs=np.sign([delta['loop_onset'],-delta['repetition_1'],delta['unique_token_ratio']])
    label='STATIC' if all(signs==0) else 'IMPROVING' if all(signs>=0) else 'WORSENING' if all(signs<=0) else 'MIXED'
    prompt_rows={f'{a}-{s}':[{'prompt':i,'loop':r['loop'],'unique_token_ratio':len(set(r['ids']))/len(r['ids']),
        'repetition':{str(n):ngram_repetition(r['ids'],n) for n in range(1,5)},
        'entropy':mean_rows(r['trace'],'entropy'),'top1_top2_probability_margin':mean_rows(r['trace'],'top1_top2_margin'),
        'runaway':r['runaway']} for i,r in enumerate(raw[a,s]['greedy']['rows'])] for a in ARMS for s in SEEDS}
    return {'classification':label,'comparison':'C minus B; mean loop onset, not historical median; no loop censored at129',
        'by_checkpoint':rows,'arm_means':avg,'C_minus_B':delta,'prompt_rows':prompt_rows,'runaway_unresolved':True,
        'trace_scope':'entropy/margin over all128 generation steps; historical summary used detected loop-onset step only'}


def main():
    resolver_gate();spec=read(PREREG);old=read(ROOT/'evaluation/foundation-v39-frequency-gate-v3-summary.json')
    raw={(a,s):read(OUT/'raw'/f'{a}-seed-{s}.json') for a in ARMS for s in SEEDS}
    for r in raw.values():
        assert r['complete'] and r['preregistration_sha256']==file_sha256(PREREG)
        assert file_sha256(Path(r['checkpoint']))==r['checkpoint_sha256'] and r['integrity']['pass']
    print('PHASE51 Core paired bootstrap / LOO',flush=True)
    core=core_analysis(spec,old);eos=eos_analysis(raw,old);sampling=sampling_analysis(raw,spec,old);greedy=greedy_analysis(raw)
    gate='TRUE_SAFETY_REGRESSION' if any(r['classification']=='TRUE_EOS_REGRESSION' for r in eos['comparison'].values()) or any(r['classification']=='TRUE_SAMPLING_REGRESSION' for r in sampling['comparisons'].values()) else 'MEASUREMENT_UNCERTAINTY_REMAINS'
    assert sampling['determinism']
    summary={'phase':51,'lr_recommendation_gate':gate,'recommended_lr':None,'comparative_lm_preference':5e-5,
        'formal_training_permission':False,'new_gpu_training':False,'next_canonical_target':'N/A until PHASE52','20m_permission':False,'foundation_base':False,
        'ml_reevaluation_reused':True,'checkpoint_integrity':'9/9 PASS, unchanged SHA, strict load and metadata',
        'preflight_tests':'480 passed,5 warnings,92.77s','resolver_gate':'PROCESS_ENV_Z_ROOT_PASS',
        'eos_classification':{a:r['classification'] for a,r in eos['comparison'].items()},
        'core_seed2026_classification':{a:r['2026']['classification'] for a,r in core['arms'].items()},
        'sampling_seed123_classification':{a:r['classification'] for a,r in sampling['comparisons'].items()},
        'attractor_classification':greedy['classification'],'greedy_runaway':{a:r['runaway_rate'] for a,r in greedy['arm_means'].items()},
        'phase50_comparative_evidence':old['comparisons'],
        'paired_core_C_minus_B':old['seed_mean_paired_core_C_minus_B_ci95'],
        'paired_supported_C_minus_B':old['seed_mean_paired_supported_C_minus_B_ci95'],
        'unchanged_failed_gates':{a:{s:[k for k,v in r['checks'].items() if not v] for s,r in old['comparisons'][a]['safety'].items()} for a in ('C','B')},
        'limitations':['Existing safety FAILs are not cleared by diagnostic relabeling.','Multi-RNG compares 15.872M baseline, not replicated own-LR256k controls; old relative-gate uncertainty remains.','Semantic proxy has no prompt meaning comparison and is not human semantic relevance.','100% greedy runaway remains a separate major blocker.','Paired bootstrap exploratory, no multiplicity-adjusted causal claim.'],
        'raw_artifacts':[{'path':str((OUT/'raw'/f'{a}-seed-{s}.json').relative_to(ROOT)),'sha256':file_sha256(OUT/'raw'/f'{a}-seed-{s}.json')} for a in ARMS for s in SEEDS],
        'thermal':{f'{a}-{s}':raw[a,s]['thermal'] for a in ARMS for s in SEEDS}}
    for suffix,value in [('core-influence',core),('eos-diagnostic',eos),('sampling-diagnostic',sampling),('greedy-diagnostic',greedy),('eos-core-sampling-summary',summary)]:
        new_json(ROOT/f'evaluation/foundation-v40-{suffix}.json',value)
    report=['# PHASE51 — EOS / Core / Sampling diagnostic','',f'LR Recommendation Gate: **{gate}**. Recommended LR: none. Comparative LM preference: 5e-5, not a safety approval.',
        '', 'Formal training permission: NO. New GPU training: NO. Next canonical target: N/A until PHASE52. 20M: NO. Foundation Base: NO.',
        '', 'Preregistration fixed before inference. CUDA FP32, no concurrent heavy CPU evaluation. Existing PHASE50 metrics/gates reused, no threshold or population changes. Nine checkpoints passed strict reload/metadata and before/after SHA checks. No checkpoint COPY/MOVE/DELETE/overwrite.',
        '', '## EOS','', 'All146 independent document ends measured once, plus500 distinct nonterminal positions excluding EOS/BOS. Historical500 EOS observations repeated146 documents; they were not500 independent ends. Historical gate uses own-LR256k control, not the formal baseline.',
        '', '| Arm | Seed | Baseline P(EOS) | Candidate P(EOS) | Delta 95% CI | Old EOS gate |','|---|---|---|---|---|---|']
    for arm in ('C','B'):
        for seed in SEEDS:
            row=eos['comparison'][arm]['by_seed'][str(seed)];ci=row['delta_vs_baseline_ci']
            report.append(f"| {arm} | {seed} | {eos['checkpoint_metrics'][f'baseline-{seed}']['terminal']['probability']:.6f} | {eos['checkpoint_metrics'][f'{arm}-{seed}']['terminal']['probability']:.6f} | [{ci['lower']:.6f}, {ci['upper']:.6f}] | {row['old_gate']} |")
    report+=['',f"EOS classifications: {summary['eos_classification']}. Per-document ranks, top-k, competitors, margins, lengths and entropy are in the EOS artifact.",'','## Core seed2026','', 'Core236 types /4436 occurrences /146 documents unchanged;10000 paired document resamples with RNG4900,5100,2026. LOO is influence diagnosis only, not a replacement gate.']
    for arm in ('C','B'):
        r=core['arms'][arm]['2026']
        report.append(f"- {arm}: {r['classification']}; CE delta {r['mean']:.6f}; CI upper range {r['ci_upper_range']}; top5 documents positive contribution {r['top5_document_positive_loss_share']:.1%}; top20 tokens {r['top20_token_positive_loss_share']:.1%}; LOO upper range {r['loo_upper_range']}. {r['loo_below_old_threshold_count']}/146 omissions would cross .10, without authorizing any omission.")
    report+=['','## Sampling and evaluator audit','', 'Same100 fixed prompts ×3 CUDA sampling RNG bases per checkpoint; temperature.7, no top-k/top-p filtering, max64 unchanged. Base44000 repeated exactly for all9 checkpoints. All2700 generated rows and paired prompt distributions are retained. Naturalness/semantic/completion are deterministic automatic proxies, not human quality. Semantic proxy checks surface structure, not prompt relevance.', '', 'Historical generation used CPU (evaluation_execution.device=cpu); numeric seeds are not equivalent CPU/CUDA random streams. New comparisons are CUDA versus CUDA. Differences from the historical single-seed aggregate are not a determinism failure, but cannot clear its gate.']
    for arm in ('C','B'):
        r=sampling['comparisons'][arm]
        report.append(f"- {arm} classification: {r['classification']}; seed123 paired metrics: {r['by_seed']['123']['metrics'] | {}}")
    # Keep report compact: detailed prompt vectors live in JSON, not prose.
    report=[line.split('; seed123 paired metrics:')[0]+'.' if '; seed123 paired metrics:' in line else line for line in report]
    report+=['','## Attractor and LR','',f"C versus B attractor: {greedy['classification']}; C-minus-B mean loop onset {greedy['C_minus_B']['loop_onset']:.3f}, repetition1 {greedy['C_minus_B']['repetition_1']:.6f}. Greedy runaway remains100% in all arms. Entropy/probability margin here average all steps; old summary measured loop-onset steps.",
        '', '5e-5 leads 7.5e-5 in loss, top-k, Middle, Core, Supported Tail and full-context CE. Existing failures remain: C123 Sampling; C2026 EOS/Core; B123 Middle/Sampling; B2026 EOS/Sampling/Core. Diagnostics do not establish a safe canonical LR. No automatic promotion.',
        '', '## Artifacts and reproducibility','', 'See phase51/preregistration.json, resolver-gate.json and the four detailed diagnostic JSONs. Raw full-step traces remain local under phase51/raw; SHA256 inventory is in the summary. Compact per-prompt/per-document evidence is versioned. FinalBlind content was not opened (hash only). Preflight pytest480 PASS; final QA is recorded separately after ML/Web completion.']
    report_path=ROOT/'evaluation/foundation-v40-eos-core-sampling-report.md'
    with report_path.open('x',encoding='utf-8') as handle:handle.write('\n'.join(report)+'\n')
    print(summary['lr_recommendation_gate'],summary['eos_classification'],summary['core_seed2026_classification'],summary['sampling_seed123_classification'],summary['attractor_classification'],flush=True)


if __name__=='__main__': main()
