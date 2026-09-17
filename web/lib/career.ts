/** Local structured drafts from student statements, never inferred achievements. */
export const CAREER_KEY='unipilot-career-v1';
export type CareerFact={id:string;kind:'course'|'project'|'research'|'user statement';title:string;detail:string;source:string;sourceId:string};
export type CareerSkill={id:string;name:string;evidenceId:string};
export type CareerProfile={schema:typeof CAREER_KEY;interests:string;courses:string;projects:string;research:string;activities:string;goals:string;facts:CareerFact[];skills:CareerSkill[];strength:string;example:string;result:string;prEvidenceId:string;esQuestion:string;esLimit:number;esFacts:string;interviewNotes:string};
export const emptyCareer=():CareerProfile=>({schema:CAREER_KEY,interests:'',courses:'',projects:'',research:'',activities:'',goals:'',facts:[],skills:[],strength:'',example:'',result:'',prEvidenceId:'',esQuestion:'',esLimit:400,esFacts:'',interviewNotes:''});
const string=(v:unknown)=>typeof v==='string'&&v.length<=6000;
export function restoreCareer(value:unknown):CareerProfile {
  const p=value as CareerProfile;
  if(!p||p.schema!==CAREER_KEY||!Array.isArray(p.facts)||!Array.isArray(p.skills)||p.facts.length>100||p.skills.length>100)throw Error('Career保存形式・上限を確認してください。');
  for(const k of ['interests','courses','projects','research','activities','goals','strength','example','result','prEvidenceId','esQuestion','esFacts','interviewNotes'] as const)if(!string(p[k]))throw Error('Careerの文字列形式・上限が不正です。');
  if(!Number.isInteger(p.esLimit)||p.esLimit<1||p.esLimit>10000)throw Error('ES文字数は1〜10000の整数で入力してください。');
  for(const f of p.facts)if(!f||!['id','title','detail','source','sourceId'].every(k=>string(f[k as keyof CareerFact]))||!f.id.trim()||!f.title.trim()||!['course','project','research','user statement'].includes(f.kind))throw Error('Evidenceの形式・タイトルを確認してください。');
  const ids=new Set(p.facts.map(f=>f.id));
  if(ids.size!==p.facts.length||new Set(p.skills.map(s=>s?.id)).size!==p.skills.length)throw Error('IDが重複しています。');
  for(const s of p.skills)if(!s||!string(s.id)||!s.id.trim()||!string(s.name)||!s.name.trim()||!string(s.evidenceId)||(s.evidenceId&&!ids.has(s.evidenceId)))throw Error('スキルとEvidenceの対応を確認してください。');
  if(p.prEvidenceId&&!ids.has(p.prEvidenceId))throw Error('自己PRのEvidenceがありません。');
  // Whitelist fields on restore; unrelated persisted keys cannot become claims.
  const blank=emptyCareer(),clean=Object.fromEntries(Object.keys(blank).map(k=>[k,p[k as keyof CareerProfile]])) as CareerProfile;
  clean.facts=p.facts.map(({id,kind,title,detail,source,sourceId})=>({id,kind,title,detail,source,sourceId}));
  clean.skills=p.skills.map(({id,name,evidenceId})=>({id,name,evidenceId}));return clean;
}
export function importSelected(p:CareerProfile,candidates:CareerFact[],ids:string[],consent:boolean):CareerProfile {
  if(consent!==true||!ids.length)throw Error('取り込む項目を明示的に選んでください。');
  const chosen=new Set(ids);if(ids.some(id=>!candidates.some(c=>c.id===id)))throw Error('選択した候補が見つかりません。');
  const existing=new Set(p.facts.map(f=>f.id));return restoreCareer({...p,facts:[...p.facts,...candidates.filter(c=>chosen.has(c.id)&&!existing.has(c.id)).map(c=>({...c}))]});
}
export function removeFact(p:CareerProfile,id:string):CareerProfile{return {...p,facts:p.facts.filter(f=>f.id!==id),skills:p.skills.map(s=>s.evidenceId===id?{...s,evidenceId:''}:s),prEvidenceId:p.prEvidenceId===id?'':p.prEvidenceId};}
export function selfPR(p:CareerProfile){
  restoreCareer(p);const evidence=p.facts.find(f=>f.id===p.prEvidenceId);
  return {status:'Template / structured draft',evidence:evidence?`${evidence.title} — ${evidence.source||'学生入力・未検証'} (${evidence.sourceId||'manual'})`:'Evidence未入力・要確認',strength:p.strength||'Strength未入力',example:p.example||'Example未入力',result:p.result||'Result未入力。実際の結果が確認できるまで補完しません。',verified:false};
}
export function entrySheet(p:CareerProfile){
  restoreCareer(p);if(!p.esQuestion.trim())throw Error('ESの質問を入力してください。');
  const text=p.esFacts.trim(),characters=[...text].length;
  return {status:'Template / student facts only',question:p.esQuestion,text,characters,limit:p.esLimit,withinLimit:characters<=p.esLimit,remaining:p.esLimit-characters,missingEvidence:!p.prEvidenceId,warning:'Unicode code pointsで計数。提出先の改行・空白・全角等の数え方は別途確認してください。自動短縮・応募・送信はしません。'};
}
export function interview(p:CareerProfile){return {status:'Template / general practice questions, not predicted company selection questions',questions:[p.goals?`「${p.goals}」を目指す理由を、自分の経験から説明できますか？`:'どの仕事・分野に関心がありますか？',p.example?`「${p.example}」で、自分が担当したことは何ですか？`:'具体的な経験で、自分が担当したことは何ですか？','その説明を裏付ける資料・経験は何ですか？'],answerNotes:p.interviewNotes,followUp:['結果が未確認なら、どの事実を確認する必要がありますか？','次に改善したい点と、実際にできる行動は何ですか？']};}
export function exportCareer(p:CareerProfile){return JSON.stringify(restoreCareer(p),null,2);}
