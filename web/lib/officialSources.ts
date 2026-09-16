/** Official-source contract. Registry is application-controlled, never user input.
 * No live retrieval implemented: fixture results cannot attest to a real university.
 */
import type {Evidence} from './evidence';
export type OfficialStatus='Verified official'|'Official but stale'|'Official year mismatch'|'Unofficial'|'Unknown'|'Conflict';
export type DomainRule={university:string;domain:string;allowSubdomains:boolean};
export type OfficialSource={id:string;university:string;domain:string;title:string;type:string;url:string;academicYear:string;effectiveDate:string;lastVerifiedAt:string;termsNote:string;status:'registered'|'disabled';fixture:boolean};
export type OfficialSourceRegistry={domains:readonly DomainRule[];sources:readonly OfficialSource[]};
export type RetrievalEvidence={sourceId:string;finalUrl:string;text:string;span:string;claimKey:string;assertion:string;retrieved:boolean};
export type OfficialResult={source:OfficialSource|null;evidence:RetrievalEvidence|null;status:OfficialStatus;reason:string;citation:Evidence|null};
export type OfficialRetrieval={retrieve:(source:OfficialSource)=>Promise<RetrievalEvidence|null>};
export function safeOfficialUrl(raw:string):URL|null {
  try{const u=new URL(raw);if(u.protocol!=='https:'||u.username||u.password||u.port||u.hostname.endsWith('.')||!u.hostname.includes('.')||! /^[a-z0-9.-]+$/.test(u.hostname))return null;return u;}catch{return null;}
}
function validDomain(domain:string):boolean{return /^[a-z0-9]+(?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9]+(?:[a-z0-9-]*[a-z0-9])?)+$/.test(domain)&&! /^\d+(?:\.\d+){3}$/.test(domain);}
export function officialDomain(url:string,university:string,registry:OfficialSourceRegistry):boolean {
  const u=safeOfficialUrl(url);return !!u&&registry.domains.some(r=>r.university===university&&validDomain(r.domain)&&(u.hostname===r.domain||(r.allowSubdomains&&u.hostname.endsWith('.'+r.domain))));
}
function date(value:string):number|null {if(!/^\d{4}-\d{2}-\d{2}$/.test(value))return null;const t=Date.parse(value+'T00:00:00Z');return Number.isFinite(t)&&new Date(t).toISOString().slice(0,10)===value?t:null;}
export function assessOfficial(source:OfficialSource|null,evidence:RetrievalEvidence|null,registry:OfficialSourceRegistry,year:string,today:string):OfficialResult {
  const finish=(status:OfficialStatus,reason:string):OfficialResult=>({source,evidence,status,reason,citation:source&&evidence?toCitation(source,evidence,status):null});
  if(!source||source.status!=='registered'||!registry.sources.some(r=>r===source))return finish('Unknown','登録済みsourceがありません。公式窓口で確認してください。');
  const declared=safeOfficialUrl(source.url);
  if(!declared||declared.hostname!==source.domain||!officialDomain(source.url,source.university,registry))return finish('Unofficial','登録された大学公式domainと一致しません。');
  if(!evidence||evidence.sourceId!==source.id||!evidence.retrieved)return finish('Unknown','根拠本文が取得・確認されていません。推測で回答しません。');
  const final=safeOfficialUrl(evidence.finalUrl);
  // Until a validated redirect chain exists, require exact URL binding, not just domain.
  if(!final||final.href!==declared.href||!officialDomain(evidence.finalUrl,source.university,registry))return finish('Unknown','取得先URLの対応を検証できません。redirectを自動的に信頼しません。');
  if(!evidence.span.trim()||!evidence.text.includes(evidence.span)||!evidence.claimKey.trim()||!evidence.assertion.trim())return finish('Unknown','対応するEvidence spanまたはclaimが不足しています。');
  const checked=date(source.lastVerifiedAt),effective=date(source.effectiveDate),current=date(today);
  if(checked===null||effective===null||current===null||checked>current||effective>current||!source.termsNote.trim()||!/^\d{4}$/.test(source.academicYear)||!/^\d{4}$/.test(year))return finish('Unknown','日付・利用条件・年度の確認が不十分です。');
  if(source.academicYear!==year)return finish('Official year mismatch','希望年度と異なります。該当年度の資料を確認してください。');
  if(current-checked>90*86400000)return finish('Official but stale','最終確認から90日超です。最新の公式案内へ確認してください。');
  return finish('Verified official',source.fixture?'DEMO契約上のみ確認済み。実在大学の公式情報ではありません。':'登録domain・根拠span・年度・確認日が契約を満たします。実際の内容適用は担当窓口でも確認してください。');
}
export function toCitation(source:OfficialSource,evidence:RetrievalEvidence,status:OfficialStatus):Evidence {
  return {id:source.id,title:source.title,author:source.university,date:source.academicYear,sourceType:source.type,url:source.url,doi:'',page:'',sourceText:evidence.text,span:evidence.span,
    note:`${status}; ${source.fixture?'DEMO fixture; ':''}last_verified_at=${source.lastVerifiedAt}; effective=${source.effectiveDate}; ${source.termsNote}`,
    // Registry-domain verification does not bypass Citation Engine's authenticity gate.
    status:evidence.span.trim()&&evidence.text.includes(evidence.span)?'Span confirmed':'Needs verification'};
}
export function markConflicts(results:OfficialResult[]):OfficialResult[] {
  const eligible=(r:OfficialResult)=>r.source&&r.evidence&&['Verified official','Official but stale','Official year mismatch'].includes(r.status);
  return results.map(r=>eligible(r)&&results.some(other=>eligible(other)&&other.source!.university===r.source!.university&&other.source!.academicYear===r.source!.academicYear&&other.evidence!.claimKey===r.evidence!.claimKey&&other.evidence!.assertion!==r.evidence!.assertion)?{...r,status:'Conflict',reason:`${r.status}: 同じ大学・年度・項目の公式資料が矛盾しています。統合しません。最新資料または担当窓口へ確認してください。`,citation:r.source&&r.evidence?toCitation(r.source,r.evidence,'Conflict'):null}:r);
}
const base={university:'TEST UNIVERSITY（架空）',domain:'official.unipilot.example',type:'大学公式案内 DEMO',academicYear:'2026',effectiveDate:'2026-04-01',lastVerifiedAt:'2026-09-14',termsNote:'Local fictional fixture; no real university policy or license claim.',status:'registered' as const,fixture:true};
export const DEMO_REGISTRY:OfficialSourceRegistry={domains:[{university:base.university,domain:base.domain,allowSubdomains:false}],sources:[
  {...base,id:'credits',title:'履修の確認方法',url:'https://official.unipilot.example/credits'},
  {...base,id:'stale',title:'奨学金の案内（古い確認）',url:'https://official.unipilot.example/stale',lastVerifiedAt:'2026-01-01'},
  {...base,id:'year',title:'試験案内（旧年度）',url:'https://official.unipilot.example/exam',academicYear:'2025'},
  {...base,id:'deadline-a',title:'期限についての案内A',url:'https://official.unipilot.example/deadline-a'},
  {...base,id:'deadline-b',title:'期限についての案内B',url:'https://official.unipilot.example/deadline-b'},
  {...base,id:'unknown',title:'教室の案内（未取得）',url:'https://official.unipilot.example/rooms'},
  {...base,id:'unofficial',title:'非公式な掲示板',domain:'student.unipilot.example',url:'https://student.unipilot.example/board'},
]};
export const DEMO_EVIDENCE:readonly RetrievalEvidence[]=DEMO_REGISTRY.sources.filter(r=>r.id!=='unknown').map(r=>{const span=r.id==='deadline-a'?'DEMO期限は10月1日。':r.id==='deadline-b'?'DEMO期限は10月2日。':'DEMO: 最新の案内は大学窓口で確認してください。';return {sourceId:r.id,finalUrl:r.url,text:span,span,claimKey:r.id.startsWith('deadline')?'deadline':r.id,assertion:span,retrieved:true};});
export function searchOfficialFixtures(query:string,year:string,today:string):OfficialResult[] {
  if(!query.trim())return [];
  const q=query.trim().toLocaleLowerCase(),sources=DEMO_REGISTRY.sources.filter(r=>(r.title+' '+r.university).toLocaleLowerCase().includes(q));
  if(!sources.length)return [assessOfficial(null,null,DEMO_REGISTRY,year,today)];
  return markConflicts(sources.map(source=>assessOfficial(source,DEMO_EVIDENCE.find(e=>e.sourceId===source.id)||null,DEMO_REGISTRY,year,today)));
}
