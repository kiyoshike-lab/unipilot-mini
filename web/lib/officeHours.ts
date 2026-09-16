/** Template-only foundation. No model call, persistence, or professor impersonation. */
export const OFFICE_MODES = ['Explain','Ask me questions','Hint first','Check my reasoning','Prepare professor question'] as const;
export type OfficeMode = typeof OFFICE_MODES[number];
export type OfficeInput = {course:string;topic:string;question:string;understood:string;stuck:string;excerpt:string;source:string;mode:OfficeMode};
export const emptyOffice = ():OfficeInput => ({course:'',topic:'',question:'',understood:'',stuck:'',excerpt:'',source:'',mode:'Explain'});
export type OfficeResult = {material:{quote:string;reference:string;status:string}|null;general:string[];escalation:string[];professorDraft:string|null;disclaimer:string;input:OfficeInput};
export function officeErrors(input:OfficeInput):Partial<Record<keyof OfficeInput,string>> {
  const errors:Partial<Record<keyof OfficeInput,string>>={};
  if(!input.course.trim())errors.course='科目を入力してください。';
  if(!input.question.trim())errors.question='質問を入力してください。';
  if(!OFFICE_MODES.includes(input.mode))errors.mode='対応するモードを選んでください。';
  for(const key of ['course','topic','question','understood','stuck','excerpt','source'] as const)if(input[key].length>(key==='excerpt'?6000:2000))errors[key]='入力が長すぎます。短い抜粋にしてください。';
  return errors;
}
export function prepareOffice(input:OfficeInput):OfficeResult {
  if(Object.keys(officeErrors(input)).length)throw new Error(Object.values(officeErrors(input)).join(' '));
  const combined=[input.topic,input.question,input.stuck].join(' ');
  const policy=/シラバス|採点|成績|単位認定|締切|締め切り|期限|延長|例外|履修規則|授業.*(?:規則|方針)|syllabus|grad(?:e|ing)|deadline|exception|course.?policy/i.test(combined);
  const escalation:string[]=[];
  if(policy)escalation.push('授業固有の規則・採点・期限の判断は教授/TAへ確認してください。このツールでは決定しません。');
  if(!input.excerpt.trim()||!input.source.trim())escalation.push('根拠が不足しています。資料名・該当箇所を確認し、必要なら教授/TAへ質問してください。');
  const general:Record<OfficeMode,string[]>={
    Explain:['一般的な学習手順: 問いの用語を定義し、前提と結論を分けてみましょう。','具体例を一つ作り、資料のどの記述が各段階を支えるか確かめましょう。これは科目内容への検証済み解答ではありません。'],
    'Ask me questions':['この問いで使う用語を、自分の言葉でどう説明しますか？','分かっている前提と、まだ確かめられていない前提は何ですか？','どの一段階を確かめると先へ進めそうですか？'],
    'Hint first':['ヒント1: 質問を「分かっている条件」と「求めること」に分けてください。','答えを先に出さず、資料から使えそうな定義を一つ選び、自分で次の一手を書いてみましょう。'],
    'Check my reasoning':[input.understood.trim()?'自己点検: 書いた説明の各段階に理由・根拠を付けてください。正誤の自動判定は行っていません。':'まず「ここまで理解したこと」に自分の考えを書いてください。', '前提の抜け、反例、単位や記号の変化がないか確認しましょう。分からない段階は教授/TAへの質問に切り出せます。'],
    'Prepare professor question':['学生からの質問案です。宛先・事実・敬称を自分で確認してください。送信はしません。'],
  };
  const draft=input.mode==='Prepare professor question'?`${input.course}の${input.topic||'授業内容'}について質問があります。\n質問: ${input.question}\n自分の理解: ${input.understood||'まだ整理できていません'}\n困っている点: ${input.stuck||'質問の進め方'}\n参照資料: ${input.source||'未確認'}\nどの前提・資料を確認するとよいか、ご教示いただけますでしょうか。`:null;
  return {input:{...input},material:input.excerpt.trim()?{quote:input.excerpt,reference:input.source||'出典未確認',status:'学生提供の抜粋 / 真正性・内容未検証'}:null,
    general:general[input.mode],escalation,professorDraft:draft,disclaimer:'Template Foundation / 教授の代理ではありません。科目の正解・採点・教授の意見を生成または代弁しません。AIモデルによる解説品質は未検証です。'};
}
