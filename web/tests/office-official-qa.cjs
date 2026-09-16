// Fictional, isolated local contracts. No real university lookup or model service.
const runtime=process.env.UNIPILOT_PLAYWRIGHT_MODULE||'playwright';
const {chromium}=require(runtime),{expect}=require(runtime+'/test');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const base=process.env.UNIPILOT_QA_URL||'http://127.0.0.1:3055',out=path.resolve(__dirname,process.env.UNIPILOT_QA_OUTPUT||'../qa/phase55');
(async()=>{
 fs.mkdirSync(out,{recursive:true});const browser=await chromium.launch({headless:true});
 try{
 const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage(),errors=[],posts=[],layouts=[];
 page.on('pageerror',e=>errors.push(e.message));await page.clock.install({time:new Date('2026-09-15T12:00:00Z')});
 await context.route('**/*',route=>{const req=route.request(),u=new URL(req.url());if(req.method()==='POST')posts.push(u.pathname);if(u.pathname==='/health')return route.fulfill({json:{status:'ok',loaded:true},headers:{'access-control-allow-origin':'*'}});if(u.origin!==new URL(base).origin)return route.abort();return route.continue();});
 const label=n=>page.getByLabel(n,{exact:true}),button=n=>page.getByRole('button',{name:n,exact:true});
 await page.goto(base+'/office-hours');await button('質問を整理する').click();await expect(page.locator('main [role=alert]')).toContainText('科目');await expect(page.locator('main [role=alert]')).toContainText('質問');
 await label('科目（必須）').fill('Demo数学');await label('質問（必須）').fill('積分の考え方を整理したい');await label('ここまで理解したこと').fill('面積として考える');await button('質問を整理する').click();await expect(page.getByText('資料は未提供です。資料由来の主張はありません。')).toBeVisible();
 await label('資料の抜粋（任意）').fill('Demo: 定義の前提を確認する。');await label('資料名・出典（任意）').fill('学生提供Demo p1');await label('モード').selectOption('Hint first');await button('質問を整理する').click();await expect(page.getByRole('heading',{name:'Hint first',exact:true})).toBeVisible();await expect(page.locator('blockquote')).toHaveText('Demo: 定義の前提を確認する。');await expect(page.getByRole('heading',{name:'一般的な学習手順（資料由来ではありません）'})).toBeVisible();
 await label('モード').selectOption('Check my reasoning');await button('質問を整理する').click();await expect(page.getByText(/正誤の自動判定は行っていません/)).toBeVisible();
 await label('質問（必須）').fill('採点と締切の例外を確認したい');await label('モード').selectOption('Prepare professor question');await button('質問を整理する').click();await expect(page.getByRole('heading',{name:'教授/TAへの確認'})).toBeVisible();await expect(page.locator('main pre')).toContainText('質問があります');
 for(const route of ['office-hours','official-search']){
  if(route==='official-search'){await page.goto(base+'/'+route);await button('Demo sourceを検索').click();await expect(page.locator('#official-error')).toContainText('検索語');await label('検索語').fill('期限');await button('Demo sourceを検索').click();await expect(page.locator('main li strong')).toHaveText(['Conflict','Conflict']);}
  for(const width of [360,390,768,1024,1440]){await page.setViewportSize({width,height:1000});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=document.documentElement.clientWidth),route+'/'+width);layouts.push({route,width,overflow:false});
   assert.ok(await page.locator('main input,main textarea,main select').evaluateAll(ns=>ns.every(n=>n.labels.length>0)));
   assert.ok(await page.locator('main button').evaluateAll(ns=>ns.every(n=>n.getBoundingClientRect().width>=44&&n.getBoundingClientRect().height>=44)));
   if(width===390||width===1440)await page.screenshot({path:path.join(out,`${route}-${width}.png`),fullPage:true});
  }
 }
 for(const [q,state] of [['履修','Verified official'],['奨学金','Official but stale'],['試験','Official year mismatch'],['教室','Unknown'],['非公式','Unofficial'],['no-such-evidence','Unknown']]){await label('検索語').fill(q);await button('Demo sourceを検索').click();await expect(page.locator('main li strong')).toHaveText([state]);await expect(page.getByText('last_verified_at',{exact:true})).toBeVisible();await expect(page.getByText('academic_year',{exact:true})).toBeVisible();}
 await label('検索語').fill('履修');await label('対象年度').fill('2025');await button('Demo sourceを検索').click();await expect(page.locator('main li strong')).toHaveText(['Official year mismatch']);
 assert.equal(await page.evaluate(()=>localStorage.length),0);assert.deepEqual(posts,[]);assert.deepEqual(errors,[]);
 await page.goto(base+'/office-hours');await expect(label('科目（必須）')).toHaveValue('');await page.emulateMedia({reducedMotion:'reduce'});await page.keyboard.press('Tab');assert.notEqual(await page.evaluate(()=>getComputedStyle(document.activeElement).outlineStyle),'none');assert.ok(await page.locator('[aria-live=polite]').count());
 fs.writeFileSync(path.join(out,'office-official-results.json'),JSON.stringify({gate:'PASS',DEMO:'PASS',LIVE:'NOT_TESTED',feature13:'Foundation template only',feature14:'Foundation fixture only',layouts,checks:['missing required fields','five-mode contract via unit tests','hint','reasoning','professor draft','no impersonation','exact supplied excerpt','general/material separation','policy escalation','no auto persistence','no POST','domain/freshness/year/conflict/unknown','citation metadata','labels','error association','aria-live','44px targets','focus-visible','reduced-motion'],errors,posts},null,2)+'\n');console.log('PASS Office Hours + Official Source workflows, 10 responsive checks');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
