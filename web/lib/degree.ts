/** Degree schema foundation. Never blend curriculum versions or infer official rules. */
export type DegreeSource={url?:string;documentId?:string;verifiedDate?:string;year?:number;version?:string;independentlyVerified?:boolean};
export type DegreeRule={university:string;faculty:string;department:string;curriculumYear:number;ruleVersion:string;effectiveFrom:string;effectiveTo?:string;category:string;requiredCredits:number;courseConstraints:string[];groupConstraints:{courseIds:string[];minimumCredits:number}[];source:DegreeSource;fixture?:boolean};
export type CreditCourse={id:string;name:string;category:string;credits:number;completed:boolean;included:boolean};
const dateValid=(s:string)=>/^\d{4}-\d{2}-\d{2}$/.test(s)&&Number.isFinite(Date.parse(s))&&new Date(s).toISOString().slice(0,10)===s;
export function degreeAudit(rules:DegreeRule[],courses:CreditCourse[]){
  const counted=courses.filter(c=>c.completed&&c.included),totalCredits=counted.reduce((s,c)=>s+c.credits,0);
  const categoryCredits:Record<string,number>=Object.create(null);for(const c of counted)categoryCredits[c.category]=(categoryCredits[c.category]||0)+c.credits;
  const base={totalCredits,categoryCredits,requirements:[] as {category:string;required:number;earned:number;remaining:number;missingCourses:string[];groups:{minimum:number;earned:number;remaining:number}[]}[]};
  if(courses.some(c=>!c.id||!Number.isFinite(c.credits)||c.credits<0||c.credits>1000)||new Set(courses.map(c=>c.id)).size!==courses.length)return {...base,totalCredits:0,categoryCredits:{},status:'INVALID_INPUT',reason:'単位の非負有限値・科目IDの重複を確認してください。'};
  if(!rules.length)return {...base,status:'UNKNOWN',reason:'Unknown — 検証可能な規則がありません。'};
  const identities=new Set(rules.map(r=>[r.university,r.faculty,r.department,r.curriculumYear,r.ruleVersion,r.effectiveFrom,r.effectiveTo||''].join('|')));
  if(identities.size!==1||new Set(rules.map(r=>r.category)).size!==rules.length)return {...base,status:'VERSION_CONFLICT',reason:'大学・課程・年度・version・有効期間が競合しています。規則を混ぜず対象を確認してください。'};
  if(rules.some(r=>!r.university||!r.faculty||!r.department||!r.ruleVersion||!Number.isInteger(r.curriculumYear)||!r.category||!Number.isFinite(r.requiredCredits)||r.requiredCredits<0||!dateValid(r.effectiveFrom)||(r.effectiveTo&&(!dateValid(r.effectiveTo)||r.effectiveTo<r.effectiveFrom))||r.groupConstraints.some(g=>!Number.isFinite(g.minimumCredits)||g.minimumCredits<0)))return {...base,status:'INVALID_RULE',reason:'規則の必須項目・単位・有効期間を確認してください。'};
  const fixture=rules.every(r=>r.fixture&&r.university==='TEST UNIVERSITY');
  if(!fixture&&rules.some(r=>!r.source||(!r.source.url&&!r.source.documentId)||!r.source.verifiedDate||!dateValid(r.source.verifiedDate)||r.source.year!==r.curriculumYear||r.source.version!==r.ruleVersion||r.source.independentlyVerified!==true))return {...base,status:'UNKNOWN',reason:'Unknown — 出典URL/document ID・検証日・年度/version・独立した検証証跡が必要です。入力された日付だけではVerifiedになりません。'};
  const ids=new Set(counted.map(c=>c.id));
  const requirements=rules.map(r=>{const earned=r.category==='TOTAL'?totalCredits:categoryCredits[r.category]||0;return {category:r.category,required:r.requiredCredits,earned,remaining:Math.max(0,r.requiredCredits-earned),missingCourses:r.courseConstraints.filter(id=>!ids.has(id)),groups:r.groupConstraints.map(g=>{const earned=counted.filter(c=>g.courseIds.includes(c.id)).reduce((s,c)=>s+c.credits,0);return {minimum:g.minimumCredits,earned,remaining:Math.max(0,g.minimumCredits-earned)};})};});
  return {...base,requirements,status:fixture?'FIXTURE_ONLY':'CALCULATED_FROM_VERIFIED_RULE',reason:fixture?'架空fixture上の条件付き計算です。実在大学の卒業可否は判定しません。':'指定された検証済み規則の算術結果のみ。特例・最終卒業可否は担当窓口で確認してください。'};
}
export function fixtureRules():DegreeRule[]{
  const common={university:'TEST UNIVERSITY',faculty:'TEST FACULTY',department:'TEST DEPARTMENT',curriculumYear:2026,ruleVersion:'fixture-v1',effectiveFrom:'2026-04-01',effectiveTo:'2030-03-31',courseConstraints:[],groupConstraints:[],source:{documentId:'FICTITIOUS-TEST-ONLY',year:2026,version:'fixture-v1'},fixture:true};
  return [{...common,category:'TOTAL',requiredCredits:12},{...common,category:'一般',requiredCredits:4},{...common,category:'専門',requiredCredits:8}];
}
