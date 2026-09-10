/** Local evidence: no network lookups, invented metadata or client trust promotion. */
export type VerificationStatus = "Unverified" | "User supplied" | "Span confirmed" | "Needs verification" | "Verified";
export type Evidence = {id:string;title:string;author:string;date:string;sourceType:string;url:string;doi:string;page:string;sourceText:string;span:string;note:string;status:VerificationStatus};
export type Claim = {id:string;text:string;evidenceIds:string[]};
export type OutlineSection = {id:string;title:string;text:string};
export type Requirement = {id:string;text:string;state:"必須条件"|"未確認条件"|"確認済み"};
export type LiteratureNote = {id:string;evidenceId:string;summary:string;provenance:"Student"|"AI";question:string};
export type PlanningDraft = {values:Record<string,string>;evidence:Evidence[];claims:Claim[];outline:OutlineSection[];requirements:Requirement[];literature:LiteratureNote[]};
export const emptyDraft = ():PlanningDraft => ({values:{},evidence:[],claims:[],outline:[],requirements:[],literature:[]});
export const newEvidence = (id:string):Evidence => ({id,title:"",author:"",date:"",sourceType:"",url:"",doi:"",page:"",sourceText:"",span:"",note:"",status:"Unverified"});
export const exactSpan = (source:string,candidate:string):boolean => candidate.trim().length>0 && source.includes(candidate);
export const checkSpan = (item:Evidence):Evidence => ({...item,status:exactSpan(item.sourceText,item.span)?"Span confirmed":"Needs verification"});
export const editEvidence = (item:Evidence,patch:Partial<Omit<Evidence,"id"|"status">>):Evidence => ({...item,...patch,status:"User supplied"});
export function removeEvidence(draft:PlanningDraft,id:string):PlanningDraft {
  return {...draft,evidence:draft.evidence.filter(e=>e.id!==id),claims:draft.claims.map(c=>({...c,evidenceIds:c.evidenceIds.filter(e=>e!==id)})),literature:draft.literature.map(n=>n.evidenceId===id?{...n,evidenceId:""}:n)};
}
export const linkedEvidence = (claim:Claim,entries:Evidence[]) => entries.filter(e=>claim.evidenceIds.includes(e.id));
export function bibliography(item:Evidence):{text:string;missing:string[]} {
  const missing=([['title','title'],['author','author / organization'],['date','year/date'],['sourceType','source type']] as const).filter(([key])=>!item[key].trim()).map(([,label])=>label);
  return {text:[item.author,item.date&&`(${item.date})`,item.title,item.sourceType,item.page&&`page: ${item.page}`,item.doi&&`DOI: ${item.doi}`,item.url].filter(Boolean).join('. '),missing};
}
const record=(v:unknown):Record<string,unknown>=>v!==null&&typeof v==='object'&&!Array.isArray(v)?v as Record<string,unknown>:{};
const text=(v:unknown):string=>typeof v==='string'?v.slice(0,20000):'';
function rows(v:unknown):Record<string,unknown>[] {if(!Array.isArray(v))return [];const seen=new Set<string>();return v.slice(0,40).map(record).filter(r=>{const id=text(r.id);if(!id||seen.has(id))return false;seen.add(id);return true;});}
/** Storage is untrusted: reserved Verified state cannot be restored from a client. */
export function restoreDraft(value:unknown):PlanningDraft {
  const raw=record(value),values=record(raw.values);
  const evidence=rows(raw.evidence).map(row=>{const item=newEvidence(text(row.id));for(const key of Object.keys(item) as (keyof Evidence)[]){if(key!=='status')item[key]=text(row[key]);}
    item.status=row.status==='Span confirmed'&&exactSpan(item.sourceText,item.span)?'Span confirmed':row.status==='Unverified'?'Unverified':row.status==='Needs verification'?'Needs verification':'User supplied';return item;});
  const ids=new Set(evidence.map(e=>e.id));
  return {values:Object.fromEntries(Object.entries(values).slice(0,40).filter(([,v])=>typeof v==='string').map(([k,v])=>[k,text(v)])),evidence,
    claims:rows(raw.claims).map(r=>({id:text(r.id),text:text(r.text),evidenceIds:[...new Set(Array.isArray(r.evidenceIds)?r.evidenceIds.filter((id):id is string=>typeof id==='string'&&ids.has(id)):[])]})),
    outline:rows(raw.outline).map(r=>({id:text(r.id),title:text(r.title),text:text(r.text)})),
    requirements:rows(raw.requirements).map(r=>({id:text(r.id),text:text(r.text),state:r.state==='必須条件'?'必須条件':r.state==='確認済み'?'確認済み':'未確認条件'})),
    literature:rows(raw.literature).map(r=>({id:text(r.id),evidenceId:ids.has(text(r.evidenceId))?text(r.evidenceId):'',summary:text(r.summary),provenance:r.provenance==='AI'?'AI':'Student',question:text(r.question)}))};
}
