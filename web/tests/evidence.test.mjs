import test from 'node:test';
import assert from 'node:assert/strict';
import {emptyDraft,newEvidence,editEvidence,checkSpan,exactSpan,removeEvidence,linkedEvidence,bibliography,restoreDraft} from '../lib/evidence.ts';

test('evidence add/edit/delete cleans claim and literature links',()=>{
  let d=emptyDraft();d.evidence.push(newEvidence('e1'),newEvidence('e2'));d.evidence[0]=editEvidence(d.evidence[0],{title:'Source'});
  d.claims.push({id:'c',text:'Claim',evidenceIds:['e1','e2']});d.literature.push({id:'l',evidenceId:'e1',summary:'Student summary',provenance:'Student',question:'Why?'});
  assert.equal(d.evidence[0].status,'User supplied');assert.equal(linkedEvidence(d.claims[0],d.evidence).length,2);
  d=removeEvidence(d,'e1');assert.deepEqual(d.claims[0].evidenceIds,['e2']);assert.equal(d.literature[0].evidenceId,'');
  d=removeEvidence(d,'e2');assert.equal(linkedEvidence(d.claims[0],d.evidence).length,0);
});
test('exact substring has no whitespace normalization or empty-match promotion',()=>{
  assert.equal(exactSpan('A日本語B','日本語'),true);assert.equal(exactSpan('A B','A  B'),false);assert.equal(exactSpan('text',''),false);assert.equal(exactSpan('  ',' '),false);
  const item=editEvidence(newEvidence('e'),{sourceText:'原文です。',span:'原文'});const checked=checkSpan(item);assert.equal(checked.status,'Span confirmed');
  assert.equal(checkSpan({...item,span:'非一致'}).status,'Needs verification');assert.equal(editEvidence(checked,{sourceText:'変更'}).status,'User supplied');
});
test('missing bibliography metadata stays missing; URLs/DOIs never inferred',()=>{
  const item=editEvidence(newEvidence('e'),{title:'Known title'});assert.equal(bibliography(item).text,'Known title');assert.deepEqual(bibliography(item).missing,['author / organization','year/date','source type']);
  assert.equal(item.author,'');assert.equal(item.url,'');assert.equal(item.doi,'');assert.equal(item.page,'');
});
test('storage never accepts client Verified or stale span status',()=>{
  const item={...newEvidence('e'),status:'Verified',sourceText:'abc',span:'b'};
  assert.equal(restoreDraft({evidence:[item]}).evidence[0].status,'User supplied');
  assert.equal(restoreDraft({evidence:[{...item,status:'Span confirmed',span:'z'}]}).evidence[0].status,'User supplied');
  assert.equal(restoreDraft({evidence:[{...item,status:'Span confirmed'}]}).evidence[0].status,'Span confirmed');
});
test('report and research draft round trip preserves structured provenance',()=>{
  const d=emptyDraft();d.values={'テーマ':'研究','研究質問':'Why?'};d.evidence=[editEvidence(newEvidence('e'),{title:'Input source',span:'original'})];
  d.requirements=[{id:'r',text:'No guessing',state:'未確認条件'}];d.outline=[{id:'o',title:'Custom structure',text:'Student notes'}];d.literature=[{id:'l',evidenceId:'e',summary:'AI source summary',provenance:'AI',question:'Why?'}];
  assert.deepEqual(restoreDraft(JSON.parse(JSON.stringify(d))),d);assert.equal(d.evidence[0].span,'original');assert.notEqual(d.evidence[0].span,d.literature[0].summary);
});
test('untrusted storage shapes, duplicate IDs and dangling links are bounded',()=>{
  assert.deepEqual(restoreDraft(null),emptyDraft());const d=restoreDraft({values:{a:42},evidence:[newEvidence('e'),newEvidence('e')],claims:[{id:'c',text:'x',evidenceIds:['e','missing','e']}],literature:[{id:'l',evidenceId:'missing'}]});
  assert.deepEqual(d.values,{});assert.equal(d.evidence.length,1);assert.deepEqual(d.claims[0].evidenceIds,['e']);assert.equal(d.literature[0].evidenceId,'');
});
