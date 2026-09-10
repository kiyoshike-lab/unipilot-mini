/** Generic arithmetic only. No university-specific policy or LLM arithmetic. */
export type RetakeMode='replace'|'both'|'exclude-old';
export type Grade={id:string;label:string;points:number;included:boolean};
export type GradingPolicy={name:string;grades:Grade[];retakeMode:RetakeMode;rounding?:{digits:number;mode:'nearest'|'floor'}};
export type Course={id:string;name:string;credits:number;grade:string;included:boolean;retakeOf?:string};
export const genericPolicy=():GradingPolicy=>({name:'Generic example — 大学公式制度ではありません',grades:[{id:'a',label:'A',points:4,included:true},{id:'b',label:'B',points:3,included:true},{id:'c',label:'C',points:2,included:true},{id:'d',label:'D',points:1,included:true},{id:'f',label:'F',points:0,included:true},{id:'p',label:'Pass (excluded)',points:0,included:false}],retakeMode:'both'});
const finite=(value:number,max:number)=>Number.isFinite(value)&&value>=0&&value<=max;
export function validateGpa(policy:GradingPolicy,courses:Course[]):string[] {
  const errors:string[]=[];
  if(!policy.grades.length||policy.grades.length>30)errors.push('評価区分は1〜30件必要です。');
  if(!['replace','both','exclude-old'].includes(policy.retakeMode))errors.push('再履修方針が不正です。');
  const grades=new Map(policy.grades.map(g=>[g.id,g]));
  if(grades.size!==policy.grades.length||policy.grades.some(g=>!g.id||!g.label.trim()||!finite(g.points,100)||typeof g.included!=='boolean'))errors.push('評価ラベルは必須です。grade pointsは0〜100の有限値にしてください。');
  if(policy.rounding&&(!Number.isInteger(policy.rounding.digits)||policy.rounding.digits<0||policy.rounding.digits>8||!['nearest','floor'].includes(policy.rounding.mode)))errors.push('Policy roundingは0〜8桁とnearest/floorで指定してください。');
  if(courses.length>100)errors.push('科目は100件までです。');
  const ids=new Map(courses.map(c=>[c.id,c]));const parents=new Set<string>();
  if(ids.size!==courses.length||courses.some(c=>!c.id))errors.push('科目IDが不正または重複しています。');
  for(const c of courses){
    if(!finite(c.credits,1000)||!grades.has(c.grade)||typeof c.included!=='boolean')errors.push(`${c.name||'科目'}: 単位は0〜1000の有限値、成績は登録済み評価から指定してください。`);
    if(c.retakeOf){const old=ids.get(c.retakeOf);
      if(!old||old.id===c.id||parents.has(c.retakeOf))errors.push(`${c.name||'科目'}: 再履修元が不明・自己参照・複数分岐です。`);
      parents.add(c.retakeOf);
      if(old&&policy.retakeMode==='replace'&&old.credits!==c.credits)errors.push('Replace old gradeは同じ単位数に限ります。異なる単位数はExclude oldを明示選択してください。');
      const seen=new Set([c.id]);let cursor=old;
      while(cursor){if(seen.has(cursor.id)){errors.push('再履修の参照が循環しています。');break;}seen.add(cursor.id);cursor=cursor.retakeOf?ids.get(cursor.retakeOf):undefined;}
    }
  }
  return [...new Set(errors)];
}
export function calculateGpa(policy:GradingPolicy,courses:Course[]) {
  const errors=validateGpa(policy,courses);if(errors.length)return {errors,qualityPoints:0,credits:0,rawGpa:null,policyGpa:null,rows:[]};
  const replaced=new Set(policy.retakeMode==='both'?[]:courses.map(c=>c.retakeOf).filter(Boolean));
  const rows=courses.map(course=>{const grade=policy.grades.find(g=>g.id===course.grade)!;const reason=replaced.has(course.id)?'再履修で旧成績を除外':!course.included?'科目を手動除外':!grade.included?'評価区分を除外':course.credits===0?'0単位':'算入';
    return {...course,gradeLabel:grade.label,points:grade.points,counted:reason==='算入',reason};});
  const counted=rows.filter(c=>c.counted);const credits=counted.reduce((sum,c)=>sum+c.credits,0),qualityPoints=counted.reduce((sum,c)=>sum+c.credits*c.points,0);
  const rawGpa=credits>0?qualityPoints/credits:null;
  const policyGpa=rawGpa===null?null:policy.rounding?(policy.rounding.mode==='floor'?Math.floor(rawGpa*10**policy.rounding.digits):Math.round(rawGpa*10**policy.rounding.digits))/10**policy.rounding.digits:rawGpa;
  return {errors,qualityPoints,credits,rawGpa,policyGpa,rows};
}
export function targetGpa(qualityPoints:number,credits:number,futureCredits:number,target:number,maxPoints:number) {
  if(!finite(qualityPoints,1e9)||!finite(credits,100000)||!finite(futureCredits,100000)||!finite(target,100)||!finite(maxPoints,100))return {error:'目標・単位・pointsには非負の有限値を入力してください。',requiredAverage:null,status:'INVALID'};
  if(credits===0&&qualityPoints!==0)return {error:'GPA単位が0の場合、累積quality pointsも0である必要があります。',requiredAverage:null,status:'INVALID'};
  if(futureCredits===0)return {error:'',requiredAverage:null,status:credits>0&&qualityPoints/credits>=target?'TARGET_ALREADY_ACHIEVED':'NO_FUTURE_CREDITS'};
  const requiredAverage=Math.max(0,(target*(credits+futureCredits)-qualityPoints)/futureCredits);
  return {error:'',requiredAverage,status:requiredAverage>maxPoints+1e-12?'IMPOSSIBLE':requiredAverage===0?'TARGET_ALREADY_FUNDED':'POSSIBLE'};
}
export function whatIf(policy:GradingPolicy,courses:Course[],grades:Record<string,string>){return calculateGpa(policy,courses.map(c=>({...c,grade:grades[c.id]||c.grade})));}
export type GpaDraft={policy:GradingPolicy;courses:Course[];scenario:Record<string,string>;target:number;futureCredits:number};
export const emptyGpa=():GpaDraft=>({policy:genericPolicy(),courses:[],scenario:{},target:3,futureCredits:0});
export function restoreGpa(value:unknown):GpaDraft {
  if(!value||typeof value!=='object')throw Error('保存データが不正です。');
  const d=value as GpaDraft;
  if(!d.policy||typeof d.policy.name!=='string'||!Array.isArray(d.policy.grades)||!Array.isArray(d.courses)||!d.scenario||typeof d.scenario!=='object'||Array.isArray(d.scenario))throw Error('保存データの形式が不正です。');
  if(d.policy.grades.some(g=>!g||typeof g.id!=='string'||typeof g.label!=='string')||d.courses.some(c=>!c||typeof c.id!=='string'||typeof c.name!=='string'||typeof c.grade!=='string'||(c.retakeOf!==undefined&&typeof c.retakeOf!=='string')))throw Error('保存データの項目が不正です。');
  if(validateGpa(d.policy,d.courses).length||!finite(d.target,100)||!finite(d.futureCredits,100000)||Object.entries(d.scenario).some(([id,g])=>!d.courses.some(c=>c.id===id)||typeof g!=='string'||(g!==''&&!d.policy.grades.some(x=>x.id===g))))throw Error('保存データの値を確認してください。');
  return d;
}
export const exportGpa=(draft:GpaDraft)=>JSON.stringify({schema:'unipilot-gpa-v1',scope:'Generic user policy, not official university validation',draft,current:calculateGpa(draft.policy,draft.courses),whatIf:whatIf(draft.policy,draft.courses,draft.scenario)},null,2);
