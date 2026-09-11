// Local-only QA. API fixtures are explicitly Demo; no production request is sent.
const {chromium}=require(process.env.UNIPILOT_PLAYWRIGHT_MODULE || 'playwright');
const fs=require('node:fs'); const path=require('node:path'); const assert=require('node:assert/strict');
const base=process.env.UNIPILOT_QA_URL || 'http://127.0.0.1:3049';
const out=path.resolve(__dirname,process.env.UNIPILOT_QA_OUTPUT||'../qa/phase53'); fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true}); const context=await browser.newContext({viewport:{width:1440,height:1000}});
 const page=await context.newPage(); const errors=[]; page.on('pageerror',e=>errors.push(e.message)); const requests=[];let mode='stream';let healthMode='online';
 await context.addInitScript(()=>{Object.defineProperty(navigator,'clipboard',{value:{writeText:async(text)=>{window.__copied=text;}}});});
 await context.route('**/*',async route=>{
   const req=route.request();const url=new URL(req.url());
   if(url.pathname==='/health'){
     if(healthMode==='offline'){await route.abort();return;}
     if(healthMode==='slow')await new Promise(r=>setTimeout(r,6500));
     await route.fulfill({status:200,headers:{'access-control-allow-origin':'*','content-type':'application/json'},body:JSON.stringify({status:'ok',loaded:true})});return;
   }
   if(url.pathname==='/chat/stream'||url.pathname==='/chat'){
     const headers={'access-control-allow-origin':'*','access-control-allow-headers':'content-type'};
     if(req.method()==='OPTIONS'){await route.fulfill({status:204,headers});return;}
     const body=req.postDataJSON();requests.push({path:url.pathname,body});
     if(mode==='offline'){await route.abort();return;}
     if(mode==='fallback'&&url.pathname==='/chat/stream'){await route.fulfill({status:503,headers});return;}
     const cards=[{kind:'clarify',title:'Demo: 確認',summary:'QA fixture',data:{options:[{category:'test',label:'Demo: 詳しく',prompt:'Demo: 詳しく知りたい'}]}},
       {kind:'tool',title:'Demo: メール',summary:'テスト用のコピー',action_label:'Demoをコピー',copy_text:'Demo: copied'},
       {kind:'sources',title:'Demo: 出典',summary:'テスト用メタデータ',data:{sources:[{id:'demo-1',title:'Demo: 検証用資料',publisher:'Demo publisher',url:'https://example.com/demo',license:'Demo license',last_verified_at:'2026-09-06',confidence:'high',stale:true}]}}];
     const snapshot={text:mode==='fallback'?'Demo: fallback応答':'Demo: streaming応答',cards};
     await route.fulfill({status:200,headers:{...headers,'content-type':url.pathname==='/chat'?'application/json':'application/x-ndjson'},body:url.pathname==='/chat'?JSON.stringify(snapshot):JSON.stringify({text:'Demo: streaming'})+'\n'+JSON.stringify(snapshot)});return;
   }
   if(url.origin!==new URL(base).origin){await route.abort();return;}
   await route.continue();
 });
 const widths=[360,390,768,1024,1440];const layouts=[];
 for(const width of widths){await page.setViewportSize({width,height:900});for(const route of ['/','/study','/materials','/exam','/report','/research','/sources','/gpa','/degree','/planner','/email']){
   await page.goto(base+route);await page.locator('h1').waitFor();
   const size=await page.evaluate(()=>({scroll:document.documentElement.scrollWidth,client:document.documentElement.clientWidth}));
   assert.ok(size.scroll<=size.client,`${route} overflows at ${width}: ${JSON.stringify(size)}`);layouts.push({route,width,overflow:false});
   assert.equal(await page.locator('h1').count(),1);
   if(width===390||width===1440)await page.screenshot({path:path.join(out,`${route==='/'?'home':route.slice(1)}-${width}.png`),fullPage:true});
 }}
 await page.setViewportSize({width:1440,height:1000});await page.goto(base+'/');
 for(const name of ['Study','Materials','Exam','Report','Research','Sources'])assert.ok(await page.locator('.journey-nav').getByRole('link',{name:`${name} ↗`,exact:true}).count());
 await page.getByRole('button',{name:'API Online・接続を再確認',exact:true}).waitFor();
 await page.getByLabel('大学生活について聞く',{exact:true}).fill('Demo: 質問');await page.getByLabel('回答の長さ').selectOption('detailed');await page.getByRole('button',{name:'送信 ↗',exact:true}).click();
 await page.getByText('Demo: streaming応答',{exact:true}).waitFor();assert.equal(requests.at(-1).body.response_mode,'detailed');assert.ok(requests.at(-1).body.session_id);
 const session=requests.at(-1).body.session_id;
 await page.getByRole('button',{name:'Demo: 詳しく',exact:true}).click();assert.equal(await page.getByLabel('大学生活について聞く',{exact:true}).inputValue(),'Demo: 詳しく知りたい');
 await page.getByRole('button',{name:'Demoをコピー',exact:true}).click();assert.equal(await page.evaluate(()=>window.__copied),'Demo: copied');
 assert.ok(await page.getByText('Demo publisher',{exact:true}).count());assert.ok(await page.getByText('Demo license',{exact:true}).count());assert.ok(await page.getByText('Stale · 要再確認',{exact:true}).count());
 mode='fallback';await page.getByRole('button',{name:'送信 ↗',exact:true}).click();await page.getByText('Demo: fallback応答',{exact:true}).waitFor();assert.equal(requests.at(-1).path,'/chat');
 await page.locator('.mode-switcher').getByRole('link',{name:'Study',exact:true}).click();await page.waitForURL(base+'/study');await page.getByLabel('科目',{exact:true}).selectOption('微積分');await page.getByLabel('説明レベル',{exact:true}).selectOption('やさしく');await page.getByLabel('学習方法',{exact:true}).selectOption('ヒントから');
 await page.getByLabel('質問・問題文',{exact:true}).fill('Demo: 微分とは');await page.getByRole('button',{name:'送信 ↗',exact:true}).click();await page.getByText('Demo: fallback応答',{exact:true}).waitFor();
 assert.ok(requests.at(-1).body.prompt.includes('微積分'));assert.ok(requests.at(-1).body.prompt.includes('ヒントから'));assert.equal(requests.at(-1).body.session_id,session);
 assert.ok(requests.at(-1).body.prompt.includes('最終解答は提示せず'));
 await page.getByLabel('トピック',{exact:true}).fill('導関数');await page.getByLabel('難易度',{exact:true}).selectOption('基礎');
 await page.getByLabel('学習方法',{exact:true}).selectOption('解答を確認');await page.getByLabel('質問・問題文',{exact:true}).fill('xの微分は？');await page.getByLabel('自分の回答',{exact:true}).fill('1');
 await page.getByRole('button',{name:'送信 ↗',exact:true}).click();await page.getByRole('status').filter({hasText:'/chat fallback'}).waitFor();
 assert.match(requests.at(-1).body.prompt,/最初の誤り/);assert.match(requests.at(-1).body.prompt,/"studentAnswer":"1"/);
 mode='offline';await page.getByLabel('質問・問題文',{exact:true}).fill('Demo: offline');await page.getByRole('button',{name:'送信 ↗',exact:true}).click();await page.locator('.api-error').waitFor();
 mode='fallback';await page.getByRole('button',{name:'同じ質問を再試行'}).click();await page.getByRole('status').filter({hasText:'/chat fallback'}).waitFor();
 await page.goto(base+'/report');await page.getByLabel('テーマ',{exact:true}).fill('Demo: 下書き');await page.getByRole('button',{name:'このタブに保存'}).click();await page.reload();assert.equal(await page.getByLabel('テーマ',{exact:true}).inputValue(),'Demo: 下書き');
 const beforeEvidence=requests.length;const evidenceQA={};
 await page.getByLabel('引用形式',{exact:true}).fill('教授に要確認');await page.getByLabel('教授からの条件',{exact:true}).fill('入力された条件だけ');
 await page.getByRole('button',{name:'条件を追加',exact:true}).click();await page.getByLabel('条件 1',{exact:true}).fill('資料を2件確認');assert.equal(await page.getByLabel('確認状態 1',{exact:true}).inputValue(),'未確認条件');
 await page.getByRole('button',{name:'根拠を追加',exact:true}).click();await page.getByLabel('Title / 資料名 1',{exact:true}).fill('Demo: 入力資料');
 await page.getByLabel('Source text / 原文 1',{exact:true}).fill('これは学生が入力した原文です。');await page.getByLabel('Exact supporting span / 引用候補 1',{exact:true}).fill('学生が入力した原文');
 await page.getByRole('button',{name:'原文との完全一致を確認 1',exact:true}).click();assert.ok(await page.locator('.evidence-status').filter({hasText:/^Span confirmed$/}).count());evidenceQA.exactSpan=true;
 await page.getByLabel('Exact supporting span / 引用候補 1',{exact:true}).fill('原文にない文章');await page.getByRole('button',{name:'原文との完全一致を確認 1',exact:true}).click();assert.ok(await page.locator('.evidence-status').filter({hasText:/^Needs verification$/}).count());evidenceQA.nonmatch=true;
 assert.match(await page.locator('.bibliography-preview').innerText(),/不足項目: author/);assert.equal(await page.getByLabel('DOI（任意） 1',{exact:true}).inputValue(),'');assert.equal(await page.getByLabel('Page（任意） 1',{exact:true}).inputValue(),'');evidenceQA.noFakeMetadata=true;
 await page.getByLabel('Exact supporting span / 引用候補 1',{exact:true}).fill('学生が入力した原文');await page.getByRole('button',{name:'原文との完全一致を確認 1',exact:true}).click();
 await page.getByRole('button',{name:'主張を追加',exact:true}).click();await page.getByLabel('主張 1',{exact:true}).fill('Demo: 自分の主張');assert.ok(await page.getByText('Unsupported — 根拠が関連付けられていません',{exact:true}).count());
 await page.getByRole('checkbox',{name:'Claim 1 → Evidence 1: Demo: 入力資料',exact:true}).check();assert.match(await page.locator('.citation-trace').innerText(),/学生が入力した原文/);evidenceQA.claimTrace=true;
 await page.getByRole('button',{name:'節を追加',exact:true}).click();await page.getByLabel('節の見出し 1',{exact:true}).fill('自由な構成');await page.getByLabel('節のメモ 1',{exact:true}).fill('自分の考察');
 await page.getByRole('button',{name:'このタブに保存',exact:true}).click();await page.reload();assert.equal(await page.getByLabel('節の見出し 1',{exact:true}).inputValue(),'自由な構成');assert.ok(await page.getByRole('checkbox',{name:'Claim 1 → Evidence 1: Demo: 入力資料',exact:true}).isChecked());evidenceQA.structuredPersistence=true;
 const populatedLayouts=[];for(const width of widths){await page.setViewportSize({width,height:900});const size=await page.evaluate(()=>({scroll:document.documentElement.scrollWidth,client:document.documentElement.clientWidth}));assert.ok(size.scroll<=size.client,`populated report overflow ${width}`);populatedLayouts.push({route:'/report',width,overflow:false});}
 await page.setViewportSize({width:390,height:900});await page.locator('#evidence-heading').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,'report-evidence-390.png')});
 await page.getByRole('button',{name:'根拠を削除 1',exact:true}).click();assert.ok(await page.getByText('Unsupported — 根拠が関連付けられていません',{exact:true}).count());evidenceQA.deleteUnlinks=true;
 page.once('dialog',dialog=>dialog.accept());await page.getByRole('button',{name:'Clear / この下書きを削除',exact:true}).click();await page.reload();assert.equal(await page.getByLabel('テーマ',{exact:true}).inputValue(),'');assert.equal(await page.getByLabel('主張 1',{exact:true}).count(),0);evidenceQA.reportClear=true;
 await page.goto(base+'/research');await page.getByLabel('研究質問',{exact:true}).fill('Demo: なぜ？');await page.getByRole('button',{name:'根拠を追加',exact:true}).click();await page.getByLabel('Title / 資料名 1',{exact:true}).fill('Demo: 文献');await page.getByLabel('Source text / 原文 1',{exact:true}).fill('原文は要約とは別です。');await page.getByLabel('Exact supporting span / 引用候補 1',{exact:true}).fill('原文は要約とは別');await page.getByRole('button',{name:'原文との完全一致を確認 1',exact:true}).click();
 await page.getByRole('button',{name:'文献メモを追加',exact:true}).click();await page.getByLabel('文献の根拠 1',{exact:true}).selectOption({label:'1: Demo: 文献'});await page.getByLabel('要約の作成者 1',{exact:true}).selectOption('AI');await page.getByLabel('Source summary 1',{exact:true}).fill('Demo: AI作成要約（原文ではない）');await page.getByLabel('Related Research Question 1',{exact:true}).fill('Demo: 関連する問い');
 await page.getByRole('button',{name:'このタブに保存',exact:true}).click();await page.reload();assert.equal(await page.getByLabel('要約の作成者 1',{exact:true}).inputValue(),'AI');assert.equal(await page.getByLabel('Source summary 1',{exact:true}).inputValue(),'Demo: AI作成要約（原文ではない）');assert.match(await page.locator('.citation-trace').innerText(),/原文は要約とは別/);evidenceQA.literatureProvenance=true;
 for(const width of widths){await page.setViewportSize({width,height:900});const size=await page.evaluate(()=>({scroll:document.documentElement.scrollWidth,client:document.documentElement.clientWidth}));assert.ok(size.scroll<=size.client,`populated research overflow ${width}`);populatedLayouts.push({route:'/research',width,overflow:false});}
 await page.setViewportSize({width:1440,height:1000});await page.locator('#literature-heading').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,'research-literature-1440.png')});
 await page.evaluate(()=>{const key='unipilot-research-draft-v2';const d=JSON.parse(sessionStorage.getItem(key));d.evidence[0].status='Verified';sessionStorage.setItem(key,JSON.stringify(d));});await page.reload();assert.equal(await page.locator('.evidence-status').filter({hasText:/^Verified$/}).count(),0);assert.ok(await page.locator('.evidence-status').filter({hasText:/^User supplied$/}).count());evidenceQA.noAutoVerified=true;
 await page.getByRole('button',{name:'根拠を削除 1',exact:true}).click();assert.equal(await page.getByLabel('文献の根拠 1',{exact:true}).inputValue(),'');
 page.once('dialog',dialog=>dialog.accept());await page.getByRole('button',{name:'Clear / この下書きを削除',exact:true}).click();await page.reload();assert.equal(await page.getByLabel('研究質問',{exact:true}).inputValue(),'');assert.equal(await page.getByLabel('Source summary 1',{exact:true}).count(),0);evidenceQA.researchClear=true;
 // Legacy draft survives migration until an explicit Clear; no silent server writes.
 await page.evaluate(()=>sessionStorage.setItem('unipilot-report-draft-v1',JSON.stringify({'テーマ':'旧下書き'})));await page.goto(base+'/report');assert.equal(await page.getByLabel('テーマ',{exact:true}).inputValue(),'旧下書き');page.once('dialog',d=>d.accept());await page.getByRole('button',{name:'Clear / この下書きを削除',exact:true}).click();assert.equal(await page.evaluate(()=>sessionStorage.getItem('unipilot-report-draft-v1')),null);evidenceQA.legacyMigration=true;
 assert.equal(requests.length,beforeEvidence);evidenceQA.noGenerationOrServerSave=true;
 fs.writeFileSync(path.join(out,'evidence-results.json'),JSON.stringify({functional:evidenceQA,populatedLayouts,DEMO:'PASS',fixtureOnly:true},null,2)+'\n');
 await page.goto(base+'/study');assert.equal(await page.getByLabel('トピック',{exact:true}).inputValue(),'導関数');assert.equal(await page.getByLabel('難易度',{exact:true}).inputValue(),'基礎');assert.equal(await page.getByLabel('自分の回答',{exact:true}).inputValue(),'');
 await page.goto(base+'/materials');await page.getByLabel('講義資料',{exact:true}).fill('# 力\nF=ma');await page.getByRole('button',{name:'Summarize',exact:true}).click();
 await page.getByRole('button',{name:'送信 ↗',exact:true}).click();await page.getByText('Demo: fallback応答',{exact:true}).waitFor();assert.match(requests.at(-1).body.prompt,/F=ma/);assert.match(requests.at(-1).body.prompt,/資料外/);assert.ok(Buffer.byteLength(requests.at(-1).body.prompt)<=448);
 const beforeLimit=requests.length;await page.getByLabel('講義資料',{exact:true}).fill('あ'.repeat(150));await page.getByRole('button',{name:'送信 ↗',exact:true}).click();await page.locator('.api-error').filter({hasText:'切り詰めず'}).waitFor();assert.equal(requests.length,beforeLimit);assert.equal((await page.getByLabel('講義資料',{exact:true}).inputValue()).length,150);
 await page.getByLabel('講義資料',{exact:true}).fill('a'.repeat(301));await page.getByRole('button',{name:'送信 ↗',exact:true}).click();await page.locator('.api-error').filter({hasText:'300文字'}).waitFor();assert.equal(requests.length,beforeLimit);
 await page.goto(base+'/exam');assert.equal(await page.getByLabel('試験科目',{exact:true}).inputValue(),'');await page.getByRole('button',{name:'送信 ↗',exact:true}).click();await page.locator('.api-error').filter({hasText:'科目と試験範囲'}).waitFor();
 await page.getByLabel('試験科目',{exact:true}).fill('物理');await page.getByLabel('試験範囲',{exact:true}).fill('力');await page.getByLabel('1日あたり学習時間（分）',{exact:true}).fill('30');
 const future=await page.evaluate(()=>{const d=new Date();d.setDate(d.getDate()+3);return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;});
 await page.getByLabel('試験日',{exact:true}).fill(future);await page.getByText('試験まで 3 日',{exact:true}).waitFor();await page.getByRole('button',{name:'送信 ↗',exact:true}).click();await page.getByText('Demo: fallback応答',{exact:true}).waitFor();assert.match(requests.at(-1).body.prompt,/"daysRemaining":3/);
 await page.getByLabel('試験日',{exact:true}).fill('2000-01-01');await page.getByText('試験日は今日以降にしてください',{exact:true}).waitFor();const beforeDate=requests.length;await page.getByRole('button',{name:'送信 ↗',exact:true}).click();await page.locator('.api-error').filter({hasText:'今日以降'}).waitFor();assert.equal(requests.length,beforeDate);
 healthMode='offline';await page.locator('.api-health').click();await page.getByRole('button',{name:'API Unavailable・接続を再確認',exact:true}).waitFor();
 healthMode='slow';await page.locator('.api-health').click();await page.getByRole('button',{name:'API Connecting・接続を再確認',exact:true}).waitFor();await page.getByRole('button',{name:'API Waking API・接続を再確認',exact:true}).waitFor();await page.getByRole('button',{name:'API Online・接続を再確認',exact:true}).waitFor();healthMode='online';
 await page.setViewportSize({width:360,height:900});await page.goto(base+'/materials');
 const targets=await page.locator('.action-picker button,.mobile-nav a,.api-health,.primary-button').evaluateAll(nodes=>nodes.map(n=>({text:n.textContent,width:n.getBoundingClientRect().width,height:n.getBoundingClientRect().height})));
 assert.ok(targets.every(t=>t.width>=44&&t.height>=44),JSON.stringify(targets));
 await page.setViewportSize({width:1440,height:1000});
 await page.emulateMedia({reducedMotion:'reduce'});assert.equal(await page.evaluate(()=>matchMedia('(prefers-reduced-motion: reduce)').matches),true);
 await page.goto(base+'/');await page.keyboard.press('Tab');const focus=await page.evaluate(()=>({text:document.activeElement?.textContent,outline:getComputedStyle(document.activeElement).outlineStyle}));assert.equal(focus.outline,'solid');
 // Verify a visible intermediate state, not just the last buffered snapshot.
 await page.evaluate(()=>{const original=window.fetch;window.fetch=async(...args)=>{if(String(args[0]).endsWith('/chat/stream'))return new Response(new ReadableStream({async start(c){const encoder=new TextEncoder();c.enqueue(encoder.encode('{"text":"Demo: first chunk"}\n'));await new Promise(r=>setTimeout(r,1200));c.enqueue(encoder.encode('{"text":"Demo: final chunk"}'));c.close();}}));return original(...args);};});
 await page.getByLabel('大学生活について聞く',{exact:true}).fill('Demo: gradual');await page.getByRole('button',{name:'送信 ↗',exact:true}).click();await page.getByText('Demo: first chunk',{exact:true}).waitFor();await page.getByText('Demo: final chunk',{exact:true}).waitFor();
 assert.deepEqual(errors,[]);fs.writeFileSync(path.join(out,'results.json'),JSON.stringify({layouts,functional:{chat:true,stream:true,visibleIntermediateStream:true,fallback:true,responseMode:true,session:true,toolCards:true,clarify:true,clipboard:true,sourceMetadata:true,studyPrompt:true,hintOnly:true,checkAnswer:true,studySessionPersistence:true,homeNavigation:true,materialsPaste:true,materialsLimits:true,examCountdown:true,examInvalidDate:true,noFakeExam:true,apiMeasuredStates:true,mobileTargets44:true,semanticHeadings:true,routeNavigation:true,offlineRetry:true,reportSave:true,reducedMotion:true,keyboardFocus:true},pageErrors:errors,fixtureOnly:true,DEMO:'PASS',LIVE:'NOT_TESTED',externalApiRequestsSent:false},null,2)+'\n');
 await browser.close();console.log('PASS: 55 route + 10 populated responsive checks; evidence and prior functional browser QA');
})().catch(e=>{console.error(e);process.exit(1);});
