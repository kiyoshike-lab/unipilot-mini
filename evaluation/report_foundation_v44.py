"""Publish body-free views/reports; preserve all frozen one-shot artifacts."""
import json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.diagnose_foundation_v40 import read,new_json,file_sha256
OUT=ROOT/'evaluation/phase55'

def main():
    spec=read(OUT/'lr-confirmatory-preregistration.json');frozen=file_sha256(OUT/'lr-confirmatory-preregistration.json')
    s=read(ROOT/'evaluation/foundation-v44-confirmatory-summary.json');a=read(OUT/'attractor-root-cause.json');m=read(OUT/'fresh-holdout-v2-manifest.json')
    assert frozen==s['preregistration_sha256']==read(OUT/'confirmatory-freeze.json')['sha256']
    # Tokenized fresh prefixes are body-derived: keep original frozen spec LOCAL,
    # publish only IDs/construction rule + original digest. No post-hoc amendment.
    public={**spec,'prompts':[{'id':p['id'],'prefix_token_count':len(p['prefix_ids'])} for p in spec['prompts']],
        'publication_note':'Body-free derivative, NOT an amended preregistration. Original frozen local artifact is excluded from Git because it contains tokenized fresh prefixes.',
        'original_frozen_sha256':frozen,'original_local_path':'evaluation/phase55/lr-confirmatory-preregistration.json'}
    new_json(OUT/'lr-confirmatory-preregistration-public.json',public)
    lines=['# Foundation v4.4 — one-shot confirmatory report','',f"Formal LR Gate: **{s['formal_lr_gate']}**. Approved LR: 5e-5 for future research use only; no training or model promotion.",'',
        '## Custody and holdout','',
        'PHASE53 Reserve27 permanently RETIRED_UNSCORABLE, unopened/unscored; near relation to supplemental remains UNKNOWN. These sets must never be claimed independent. No retired Reserve use for validation, selection, thresholding, hyperparameters or quality assessment.',
        'PHASE54 supplemental591 /1,078,706 tokens reused, acquisition window unchanged. All historical comparison source hashes verified. Internal174,345 unordered pairs: exact0, normalized0, near0 at full normalized5-character Jaccard>=.8. Historical reuse retains the narrower SimHash<=3 AND Jaccard>=.8 rule and its lexical/coverage limitations.',
        '**FRESH_HOLDOUT_V2_READY**. SHA256, normalized SHA and SHA256-64 bottom256 KMV fingerprints computed before split. Future sketch comparison is approximate (review/exclude at estimatedJaccard>=.75); it cannot certify absence of paraphrases. No raw or normalized text is in the fingerprints.',
        '', '| Split | Documents | Tokens | State |','|---|---:|---:|---|']
    for name,r in m['splits'].items():lines.append(f"| {name} | {r['documents']} | {r['tokens']} | {'SEALED_WITH_FINGERPRINTS / no inference' if name=='future-reserve2' else 'consumed for model selection' if name=='confirmatory' else 'pipeline-only; not primary LR evidence'} |")
    lines+=['','Category-stratified SHA ordering, seed phase55-v2-5501, memberships frozen before scoring. Categories<6 assigned entirely to Diagnostic (math3, language5, procedure1); no claim of broad subject coverage. Reserve2 contents were not reopened after sealing. Source metadata/license references and per-split distributions are in the manifest/fingerprint artifacts.',
        '', '## Preregistered comparison','',f'Original frozen SHA: `{frozen}`.',
        'Original local preregistration includes tokenized fresh prefixes and is intentionally NOT staged. The public derivative retains all metric/threshold/RNG rules and prompt IDs, with this original digest; no rule was amended after observing results.',
        'Six existing512k checkpoints C/B×seeds42/123/2026: SHA and12 integrity checks each PASS. CUDA FP32, TF32 OFF, no gradients/optimizer updates. Both LRs retain EOS1.5 and auxiliaryOFF. Per-checkpoint complete raw inference saved on Z, not Git. Thermal maximum82°C across confirmatory runs; no hardware throttle; monitored cooldown pauses used.',
        'Equal-document CE, all tokens including EOS target excluding BOS. Fixed128-target chunks with up to384-token overlap for max512 context; max128 short blocks score identical targets. These are bounded-context curves, not exactly512 history tokens at document starts. Seed-average paired document C−B, percentile95% bootstrap,10,000 replicates, RNG550055. Seeds fixed; CIs do not imply generalization over all training seeds.',
        '', '| Metric (equal3-seed mean) | C /5e-5 | B /7.5e-5 |','|---|---:|---:|']
    c,b=s['metrics']['C'],s['metrics']['B']
    for label,x,y in [('CE full',c['full']['ce'],b['full']['ce']),('Perplexity exp(mean CE)',math.exp(c['full']['ce']),math.exp(b['full']['ce'])),('CE short',c['short_ce'],b['short_ce']),*[(f'Top{k} rate',c['full'][f'top{k}'],b['full'][f'top{k}']) for k in (1,5,10)],('Terminal P(EOS)',c['terminal_probability'],b['terminal_probability']),('Nonterminal P(EOS)',c['full']['nonterminal_eos_probability'],b['full']['nonterminal_eos_probability']),('Premature EOS argmax rate',c['full']['premature_eos'],b['full']['premature_eos']),*[(k,c['sampling'][k],b['sampling'][k]) for k in ('runaway_rate','eos_rate','sentence_completion_rate','japanese_validity','topic_retention_proxy','naturalness_rate','semantic_rate','repetition_1','repetition_2','repetition_3','repetition_4')]]:lines.append(f'| {label} | {x:.6f} | {y:.6f} |')
    p=s['primary_paired_ci'];lines+=['',f"Primary paired CE: {p['mean']:.8f};95% CI [{p['lower']:.8f},{p['upper']:.8f}]. All3 seed-specific CIs also below0 (JSON).",'',
        '| Arm /seed | EOS probability | Mean rank | Top1 | Top5 | Top10 |','|---|---:|---:|---:|---:|---:|']
    for arm in ('C','B'):
        for r in s['metrics'][arm]['by_seed']:
            e=r['terminal'];lines.append(f"| {arm}/{r['seed']} | {e['probability']:.6f} | {e['rank']:.2f} | {e['top1']:.4f} | {e['top5']:.4f} | {e['top10']:.4f} |")
    lines+=['','## Relative safety, not generation safety','',
        'Registered clear-material-regression rule requires the entire paired95% interval to exceed a worsening boundary: context CE +.03; loss of long-context benefit+.03; terminal EOS relative drop20%; premature EOS+1pp; sampling runaway+8pp or naturalness/semantic/Japanese-validity drop8pp. All8 flags are false. Full uncertainty and per-seed distributions remain in JSON; absence of a flag is not a proof of no harm.',
        '64 checkpoint-independent fresh prompts ×2 fixed RNG ×3 seeds per LR. Sampling decoder temperature.7, max64, no forced-stop credit. Both LR runaway98.9583%, EOS completion1.0417%: **GENERATION_POLICY_UNSAFE** remains. Naturalness/semantic/topic/completion are automatic proxies, not human correctness or actual relevance validation.',
        'Confirmatory consumed_for_model_selection=true from scoring start and confirmed complete here; never reuse as blind. Diagnostic is not a replacement primary set. PHASE53 Reserve and Reserve2 remain unavailable to model scoring.',
        '','## Outcome','',
        'Approved5e-5 research LR does NOT authorize training start, Foundation Base completion, generation safety, canonical,20M or Production deployment. New training NO, canonical NO,20M NO, Foundation Base NO.',
        'Attractor audit: INSUFFICIENT_EVIDENCE / MORE_ROOT_CAUSE_WORK_REQUIRED. No PHASE56 arms registered; see foundation-v44-attractor-root-cause.md.']
    with (ROOT/'evaluation/foundation-v44-confirmatory-report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    root=['# Foundation v4.4 — attractor root-cause audit','',
        '**INSUFFICIENT_EVIDENCE** for a dominant cause. **MORE_ROOT_CAUSE_WORK_REQUIRED**. Research LR5e-5 is approved separately. No training, canonical or20M; no executable PHASE56 arm preregistration was created.',
        '', '## Integrated observations','',
        'Reused PHASE51 fixed900 greedy traces from baseline/C/B×3seeds and PHASE52 source-hash-verified time series. PHASE53 policy study remains GENERATION_POLICY_UNSAFE; decoder mitigation did not demonstrate safe natural termination. New PHASE55 normal controls are separate from fresh LR selection.',
        'Analyzed up to32 steps strictly before and after loop onset, retaining prompt taxonomy, prefix length, cycle length, repetition/unique-token ratio, entropy, top1/top2 margin and EOS. Early onsets have only14.83–19.69 pre-onset steps on average by checkpoint; post windows31.70–32.00. No invented missing steps. Source series contain cumulative repetition1–4; detailed repeated-ngrams retained in the hash-bound PHASE52 raw evidence.',
        'Across all9 checkpoints after-onset top1 increases and unique-token ratio falls; EOS decreases. For C2026: top1 .28091→.33632, entropy4.78123→4.45774, margin.21972→.26912, EOS.000612→.000378. This supports a descriptive self-copy attractor with probability concentration and weak stopping, but does not identify which factor causes another.',
        'Full logit-vector/hidden-state cosine: NOT_MEASURED. Historical records retain top5 probabilities, not full vectors; reconstructing cosine would be invalid. No architecture hooks or model changes introduced.',
        '', '## Copy signature versus training','',
        'Top20 generated post-onset ngrams per width, chosen by occurrence count then token-ID tie break. All33,402,759 packed training tokens examined, cross-document hits excluded. All10,012 JSONL documents were retokenized and matched exactly to packed boundaries before joining category labels. Per-pattern frequency, document support and category support are in phase55/attractor-root-cause.json. No source prose is quoted.',
        '', '| n | Selected | Zero training frequency | Max training frequency |','|---|---:|---:|---:|']
    for n in range(3,7):
        rs=[r for r in a['copy_support'] if r['n']==n];root.append(f"| {n} | {len(rs)} | {sum(r['train_frequency']==0 for r in rs)} | {max(r['train_frequency'] for r in rs)} |")
    root+=['',
        'Many frequent longer generated patterns are absent from training, weakening a simple direct-copy explanation. It does NOT rule out learned data-induced repetition. Frequent3grams may reflect ordinary grammar. No calibration, randomized position control or causal objective ablation distinguishes overconfidence, EOS suppression, self-copy and exposure effects.',
        '', '## Normal repetition control risk','',
        'Reused all20 frozen PHASE54 templates: math, lists, definitions, code and repeated terminology ×n2/3/4/5. Concrete prompts and rules frozen before new greedy inference; baseline42/C42/B42. All60 outputs exhausted128 tokens without EOS; observable shape proxy0/20 for each model. This is not a human semantic score. The pretraining model cannot currently satisfy the finite instructional templates, so this control set has a floor effect: unchanged0 does not establish non-regression. Independent boundary review remains pending.',
        '', '## Candidate comparison (hypotheses, not measured benefits)','',
        '| Candidate | Expected benefit | Failure risk | Normal-repeat risk | Complexity | Reversibility |','|---|---|---|---|---|---|',
        '| Approved LR continuation control | Isolate objective change at5e-5 | Shared loops likely remain | Existing failure floor | Low | New experimental checkpoint only |',
        '| Generated-prefix ngram unlikelihood | Target repeated self-generated contexts absent in teacher forcing | Bad negative mining; probability mass diversion | High: math/list/code repetition | Medium; exact negative labels needed | Disable loss, retain parent checkpoint |',
        '| Sequence-level anti-loop | Penalize pathological completed trajectories | High-variance/sparse signal; force-shortening shortcut | High: legitimate repeated procedures | High; trajectory scoring and masks | Separate arm, no canonical change |',
        '| Data repeated-pattern reweighting | Reduce objectively excessive source patterns if implicated | Remove useful language; longer generated grams often absent | High: textbook/code/definitions | Medium; corpus provenance audit | New weights only, original corpus immutable |',
        '| EOS-aware auxiliary | Improve stopping where endings are justified | Premature termination, apparent runaway improvement without quality | Medium/high on long explanations | Medium; terminal/nonterminal supervision | Disable auxiliary, preserve baseline |',
        '',
        'Unlikelihood is a published objective family with token/sequence variants; its reported success elsewhere does not prove suitability here. [Welleck et al., Neural Text Generation with Unlikelihood Training](https://arxiv.org/abs/1908.04319). The benefit/risk judgments above are hypotheses from this audit, not replicated paper results.',
        'The exact failed PHASE42 teacher-forced repeated3/4gram alternative-token auxiliary (weights.01/.03/.05, runaway100%) is explicitly excluded. A differently supervised generated-prefix proposal is not yet registered or authorized.',
        '', '## Gate and next evidence required','',
        'No PHASE56 arms: LR condition passes, sufficient-cause/safe-control condition does not. Do not create foundation-v44-training-fix-preregistration.json. Next bounded non-training research should separate EOS/position/self-prefix effects on fixed historical prompts and establish a normal-repeat control with measurable nonfloor correctness; never reuse fresh Confirmatory as blind or open either Reserve.',
        'If a later phase establishes both conditions, a future design should use control+max2 interventions, initial64–128k/max256k, greedy runaway<=50% minimum (<=25% desired), substantial sampling decline, and preregistered LM/Core/SupportedTail/EOS/context/normal-control non-degradation. These are planning constraints, not executable arms or permission to start.']
    with (ROOT/'evaluation/foundation-v44-attractor-root-cause.md').open('x',encoding='utf-8') as f:f.write('\n'.join(root)+'\n')
    new_json(OUT/'completion-state.json',{'confirmatory_complete':True,'consumed_for_model_selection':True,'fresh_gate':m['gate'],
        'formal_lr_gate':s['formal_lr_gate'],'training_fix_gate':a['training_fix_gate'],'phase56_arms':[],'new_training':False,
        'preregistration_sha256':frozen,'body_derived_preregistration_original_git':'NOT_STAGED; body-free public derivative provided'})

if __name__=='__main__':main()
