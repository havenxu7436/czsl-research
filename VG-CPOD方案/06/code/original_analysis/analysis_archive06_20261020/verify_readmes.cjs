const fs = require('fs');
const path = require('path');
const {pathToFileURL} = require('url');
const {chromium} = require('C:/Users/llenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const stage = path.resolve(__dirname,'../../CZSL/VG-CPOD方案/06');
const qa = path.join(stage,'analysis/repository_qa');
(async()=>{
  const browser = await chromium.launch({channel:'msedge',headless:true});
  const rows = [];
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1100},deviceScaleFactor:1});
    for (const item of JSON.parse(fs.readFileSync(path.join(qa,'render.json'),'utf8'))) {
      await page.goto(pathToFileURL(path.join(qa,item.html)).href);
      await page.evaluate(()=>document.fonts.ready);
      const state = await page.evaluate(()=>({
        displayMath:document.querySelectorAll('math[display="block"]').length,
        unrenderedMath:document.querySelectorAll('span.math').length,
        horizontalOverflow:document.documentElement.scrollWidth>innerWidth,
        wideTables:[...document.querySelectorAll('table')].filter(x=>x.getBoundingClientRect().right>innerWidth).length,
        wideMath:[...document.querySelectorAll('math')].filter(x=>x.scrollWidth>x.clientWidth+1).length,
        tables:document.querySelectorAll('table').length,
        images:[...document.images].map(x=>({loaded:x.complete&&x.naturalWidth>0}))
      }));
      rows.push({document:item.document,expectedDisplayMath:item.display_formulas,...state});
      if(state.displayMath!==item.display_formulas || state.unrenderedMath || state.horizontalOverflow || state.wideTables || state.wideMath || state.images.some(x=>!x.loaded)) throw new Error(JSON.stringify(rows.at(-1)));
      if(item.html==='main.html') {
        await page.screenshot({path:path.join(qa,'01_main.png')});
        await page.locator('h2').filter({hasText:'2. 公式'}).evaluate(x=>x.scrollIntoView({block:'start'}));
        await page.screenshot({path:path.join(qa,'02_cp_formula.png')});
        await page.locator('h2').filter({hasText:'3. 新增公式'}).evaluate(x=>x.scrollIntoView({block:'start'}));
        await page.screenshot({path:path.join(qa,'03_selection_formula.png')});
        await page.locator('h2').filter({hasText:'4. 沿用'}).evaluate(x=>x.scrollIntoView({block:'start'}));
        await page.screenshot({path:path.join(qa,'04_loss_formula.png')});
      }
      if(item.html==='analysis.html') {
        await page.screenshot({path:path.join(qa,'05_analysis.png')});
        await page.locator('h2').filter({hasText:'5. β'}).evaluate(x=>x.scrollIntoView({block:'start'}));
        await page.screenshot({path:path.join(qa,'06_beta_formula.png')});
      }
    }
    fs.writeFileSync(path.join(qa,'layout.json'),JSON.stringify({passed:true,readmes:rows},null,2));
    console.log(JSON.stringify(rows,null,2));
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
