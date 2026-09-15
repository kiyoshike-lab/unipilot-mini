"use client";
import {useEffect,useState} from 'react';
import {MEMORY_KEY,emptyMemory,emptyMemoryItem,restoreMemory,upsertMemory,deleteMemory,exportMemory} from '../lib/learningMemory';
import type {LearningMemory,MemoryItem} from '../lib/learningMemory';
export function MemoryWorkspace(){
  const [memory,setMemory]=useState<LearningMemory>(emptyMemory),[draft,setDraft]=useState<MemoryItem>(emptyMemoryItem),[ready,setReady]=useState(false),[error,setError]=useState(''),[notice,setNotice]=useState('');
  useEffect(()=>{try{const raw=localStorage.getItem(MEMORY_KEY);if(raw)setMemory(restoreMemory(JSON.parse(raw)));}catch{setError('保存内容を読み込めません。自動上書きしません。削除する前にブラウザーのデータを確認してください。');}setReady(true);},[]);
  function persist(next:LearningMemory){localStorage.setItem(MEMORY_KEY,exportMemory(next));setMemory(next);setError('');}
  function save(){try{const next=upsertMemory(memory,{...draft,id:draft.id||crypto.randomUUID()},true);persist(next);setDraft(emptyMemoryItem());setNotice('学習記憶をこのブラウザーに保存しました。');}catch(e){setError((e as Error).message);}}
  function remove(id:string){try{persist(deleteMemory(memory,id));if(draft.id===id)setDraft(emptyMemoryItem());setNotice('学習記憶を削除しました。');}catch(e){setError((e as Error).message);}}
  function clear(){if(!confirm('学習記憶と編集中の内容をすべて削除しますか？'))return;try{localStorage.removeItem(MEMORY_KEY);setMemory(emptyMemory());setDraft(emptyMemoryItem());setError('');setNotice('すべての学習記憶を削除しました。');}catch{setError('削除できません。ブラウザーのサイトデータ設定を確認してください。');}}
  function download(){const url=URL.createObjectURL(new Blob([exportMemory(memory)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='unipilot-learning-memory.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);setNotice('保存済み学習記憶をExportしました。');}
  const field=(key:keyof MemoryItem,value:string|boolean|number)=>setDraft(d=>({...d,[key]:value}));
  return <main className="workspace writing-workspace planner"><p className="eyebrow">11 / LEARNING MEMORY · LOCAL BETA</p><h1>学びを、自分の言葉で残す。</h1><p className="intro-copy">科目・理解度・振り返りを、自分で確かめて保存。</p><p className="shell-notice">Local only / このブラウザーのlocalStorageに保存。サーバーには送信しません。保存ボタンを押すまで入力は保存されません。共有端末では削除してください。</p><p>Confidenceは学生自身の自己評価です。弱点のAI自動判定はありません。</p>
    <p id="memory-errors" role="alert">{error}</p><p role="status" aria-live="polite">{notice}</p>
    <fieldset disabled={!ready} className="writing-controls"><legend>{draft.id?'記憶を編集':'新しい学習記憶'}</legend><div className="evidence-fields">
    {([['subject','科目'],['topic','Topic']] as const).map(([k,label])=><label key={k}>{label}<input aria-describedby="memory-errors" maxLength={200} value={draft[k]} onChange={e=>field(k,e.target.value)}/></label>)}
    <label>自分のConfidence<select aria-label="自分のConfidence" value={draft.confidence} onChange={e=>field('confidence',Number(e.target.value))}>{[1,2,3,4,5].map(n=><option key={n} value={n}>{n} / 5</option>)}</select></label>
    <label>最後に学習した日<input aria-describedby="memory-errors" type="date" value={draft.lastStudied} onChange={e=>field('lastStudied',e.target.value)}/></label>
    <label>情報源<select aria-label="情報源" value={draft.sourceType} onChange={e=>field('sourceType',e.target.value)}>{['user entered','quiz result','exam practice','tutor session'].map(s=><option key={s}>{s}</option>)}</select></label>
    <label>出典・結果のメモ<input aria-describedby="memory-errors" maxLength={2000} value={draft.sourceReference} onChange={e=>field('sourceReference',e.target.value)}/></label></div>
    {([['mistakes','間違えたこと'],['weakPoints','確認した弱点'],['masteredItems','習得したこと'],['note','学生メモ']] as const).map(([k,label])=><label key={k}>{label}<textarea aria-describedby="memory-errors" maxLength={2000} value={draft[k]} onChange={e=>{field(k,e.target.value);if(k==='weakPoints')field('weaknessConfirmed',false);}}/></label>)}
    <label className="consent-row"><input type="checkbox" checked={draft.weaknessConfirmed} onChange={e=>field('weaknessConfirmed',e.target.checked)}/>弱点の内容を自分で確認しました</label>
    <div className="draft-actions"><button className="primary-button" onClick={save}>Save to learning memory</button><button className="secondary-button" onClick={()=>setDraft(emptyMemoryItem())}>入力をリセット</button></div></fieldset>
    <section className="evidence-section"><h2>保存済み学習記憶 ({memory.items.length}/200)</h2><ul className="planner-list">{memory.items.map(item=><li key={item.id}><h3>{item.subject} / {item.topic}</h3><p>自己評価 {item.confidence}/5 · {item.lastStudied} · {item.sourceType}</p><p>{item.sourceReference}</p><dl><dt>間違い</dt><dd>{item.mistakes||'未記入'}</dd><dt>本人確認済みの弱点</dt><dd>{item.weakPoints||'未記入'}</dd><dt>習得事項</dt><dd>{item.masteredItems||'未記入'}</dd><dt>学生メモ</dt><dd>{item.note||'未記入'}</dd></dl><button className="secondary-button" aria-label={`${item.topic}を編集`} onClick={()=>{setDraft({...item});setNotice('編集内容は保存するまで反映されません。');}}>編集</button><button className="secondary-button" aria-label={`${item.topic}を削除`} onClick={()=>remove(item.id)}>削除</button></li>)}</ul><div className="draft-actions"><button className="secondary-button" onClick={download}>Export learning memory</button><button className="secondary-button" onClick={clear}>Clear all learning memory</button></div><p>Study Planへ取り込んだ情報は別の保存コピーです。削除したい場合はPlan側でも削除してください。</p></section>
  </main>;
}
