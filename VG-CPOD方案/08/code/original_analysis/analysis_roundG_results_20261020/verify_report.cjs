const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
const { chromium } = require('C:/Users/llenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async()=>{
  const root=path.resolve(__dirname,'..');
  const name='VG-CPOD_v1_G组结果分析与下一轮方向_20261020';
  const text=fs.readFileSync(path.join(root,name+'.md'),'utf8');
  const qa=path.join(__dirname,'qa');fs.mkdirSync(qa,{recursive:true});
  const browser=await chromium.launch({channel:'msedge',headless:true});
  try {
    const page=await browser.newPage({viewport:{width:1440,height:1100},deviceScaleFactor:1});
    await page.goto(pathToFileURL(path.join(root,name+'.html')).href);
    await page.evaluate(()=>document.fonts.ready);
    const report=await page.evaluate(()=>({
      displayMath:document.querySelectorAll('math[display="block"]').length,
      mathCount:document.querySelectorAll('math').length,
      unrenderedMath:document.querySelectorAll('span.math').length,
      horizontalOverflow:document.documentElement.scrollWidth>innerWidth,
      wideTables:[...document.querySelectorAll('table')].filter(x=>x.getBoundingClientRect().right>innerWidth).length,
      tables:document.querySelectorAll('table').length,
      images:[...document.images].map(x=>({loaded:x.complete&&x.naturalWidth>0,width:x.naturalWidth}))
    }));
    report.expectedDisplayMath=(text.match(/\$\$/g)||[]).length/2;
    await page.screenshot({path:path.join(qa,'01_results.png')});
    await page.locator('h2').filter({hasText:'3. seed3353'}).evaluate(x=>x.scrollIntoView({block:'start'}));
    await page.screenshot({path:path.join(qa,'02_gate_analysis.png')});
    await page.locator('h2').filter({hasText:'5. 约'}).evaluate(x=>x.scrollIntoView({block:'start'}));
    await page.screenshot({path:path.join(qa,'03_formula.png')});
    await page.locator('h3').filter({hasText:'5.1 固定均值'}).evaluate(x=>x.scrollIntoView({block:'start'}));
    await page.screenshot({path:path.join(qa,'05_static_equivalence.png')});
    await page.locator('h2').filter({hasText:'6. 下一轮 H1'}).evaluate(x=>x.scrollIntoView({block:'start'}));
    await page.screenshot({path:path.join(qa,'04_next_round.png')});
    fs.writeFileSync(path.join(qa,'layout.json'),JSON.stringify(report,null,2));
    console.log(JSON.stringify(report,null,2));
    if(report.displayMath!==report.expectedDisplayMath||report.unrenderedMath||report.horizontalOverflow||report.wideTables||report.images.some(x=>!x.loaded)) {
      throw new Error('Layout or formula display needs correction');
    }
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
