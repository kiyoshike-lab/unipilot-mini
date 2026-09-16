"""Pure invariants: no holdout body or checkpoint inference in tests."""
import numpy as np
from evaluation.build_foundation_v44_holdout import fingerprint,grams,sketch_jaccard
from evaluation.confirm_foundation_v44 import ci

def test_fingerprint_deterministic_nonplaintext():
    g=grams('学習の定義を確認します。'*30)
    a=fingerprint(g)
    assert a==fingerprint(set(reversed(sorted(g))))
    assert all(len(x)==16 and set(x)<=set('0123456789abcdef') for x in a['values'])
    assert sketch_jaccard(a,a)==1

def test_small_fingerprints_exact_jaccard():
    a=fingerprint({'alpha','beta'});b=fingerprint({'beta','gamma'})
    assert sketch_jaccard(a,b)==1/3

def test_ci_equal_document_not_token_weighted():
    result=ci([-1.,0.,1.])
    assert result['mean']==0 and result['lower']<=0<=result['upper']
    assert ci(np.repeat(-.03,20))['upper']<0
    assert ci([1,2,3])==ci([1,2,3])

def test_normalization_and_fixed_ngrams():
    assert grams('ＡＢＣＤＥＦ G')==grams('abcdefg')
