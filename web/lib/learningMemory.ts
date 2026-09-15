/** Student-confirmed local records. Saving is an explicit user action. */
export const MEMORY_KEY='unipilot-learning-memory-v1';
export type MemoryItem={id:string;subject:string;topic:string;confidence:number;lastStudied:string;mistakes:string;weakPoints:string;masteredItems:string;note:string;sourceType:'user entered'|'quiz result'|'exam practice'|'tutor session';sourceReference:string;weaknessConfirmed:boolean;confidenceSource:'student';};
export type LearningMemory={schema:'unipilot-learning-memory-v1';items:MemoryItem[]};
export const emptyMemory=():LearningMemory=>({schema:'unipilot-learning-memory-v1',items:[]});
export const emptyMemoryItem=():MemoryItem=>({id:'',subject:'',topic:'',confidence:3,lastStudied:'',mistakes:'',weakPoints:'',masteredItems:'',note:'',sourceType:'user entered',sourceReference:'',weaknessConfirmed:false,confidenceSource:'student'});
export function validateMemoryItem(x:MemoryItem){
  if(!x||['id','subject','topic','lastStudied','mistakes','weakPoints','masteredItems','note','sourceReference'].some(k=>typeof x[k as keyof MemoryItem]!=='string'||String(x[k as keyof MemoryItem]).length>2000))throw Error('学習記憶の文字列形式・上限を確認してください。');
  if(!x.id.trim()||!x.subject.trim()||!x.topic.trim())throw Error('科目とtopicが必要です。');
  if(!Number.isInteger(x.confidence)||x.confidence<1||x.confidence>5||x.confidenceSource!=='student')throw Error('Confidenceは学生自身の1〜5の評価です。');
  if(!/^20\d\d-\d\d-\d\d$/.test(x.lastStudied)||!Number.isFinite(Date.parse(x.lastStudied))||new Date(x.lastStudied).toISOString().slice(0,10)!==x.lastStudied)throw Error('学習日を正しく入力してください。');
  if(!['user entered','quiz result','exam practice','tutor session'].includes(x.sourceType))throw Error('情報源を選択してください。');
  if(x.sourceType!=='user entered'&&!x.sourceReference.trim())throw Error('結果・セッションの出典メモが必要です。');
  if(typeof x.weaknessConfirmed!=='boolean'||(x.weakPoints.trim()&&!x.weaknessConfirmed))throw Error('弱点は本人による確認が必要です。');
}
export function restoreMemory(v:unknown):LearningMemory{
  const m=v as LearningMemory;if(!m||m.schema!==MEMORY_KEY||!Array.isArray(m.items)||m.items.length>200)throw Error('学習記憶の保存形式・上限を確認してください。');
  m.items.forEach(validateMemoryItem);if(new Set(m.items.map(x=>x.id)).size!==m.items.length)throw Error('学習記憶IDが重複しています。');return m;
}
export function upsertMemory(m:LearningMemory,item:MemoryItem,consent:boolean):LearningMemory{
  if(consent!==true)throw Error('保存操作による同意が必要です。');validateMemoryItem(item);
  return restoreMemory({...m,items:[...m.items.filter(x=>x.id!==item.id),{...item}]});
}
export function deleteMemory(m:LearningMemory,id:string):LearningMemory{return {...m,items:m.items.filter(x=>x.id!==id)};}
export function exportMemory(m:LearningMemory){return JSON.stringify(restoreMemory(m),null,2);}
