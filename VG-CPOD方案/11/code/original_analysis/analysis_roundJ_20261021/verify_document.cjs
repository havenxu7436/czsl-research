const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
const { chromium } = require('C:/Users/llenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const root = path.resolve(__dirname, '..');
  const name = 'VG-CPOD_v1_J组实验实施方案_20261021';
  const md = fs.readFileSync(path.join(root, name+'.md'), 'utf8');
  const qa = path.join(__dirname,'qa');
  fs.mkdirSync(qa,{recursive:true});
  const browser = await chromium.launch({channel:'msedge',headless:true});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1100},deviceScaleFactor:1});
    await page.goto(pathToFileURL(path.join(root,name+'.html')).href);
    await page.evaluate(()=>document.fonts.ready);
    const result = await page.evaluate(()=>({
      displayMath:document.querySelectorAll('math[display="block"]').length,
      mathCount:document.querySelectorAll('math').length,
      unrenderedMath:document.querySelectorAll('span.math').length,
      horizontalOverflow:document.documentElement.scrollWidth>innerWidth,
      wideTables:[...document.querySelectorAll('table')].filter(x=>x.getBoundingClientRect().right>innerWidth).length,
      tables:document.querySelectorAll('table').length,codeBlocks:document.querySelectorAll('pre').length
    }));
    result.expectedDisplayMath = (md.match(/\$\$/g)||[]).length/2;
    await page.screenshot({path:path.join(qa,'01_overview.png')});
    await page.locator('h3').filter({hasText:'2.1 固定 TRAIN'}).evaluate(x=>x.scrollIntoView({block:'start'}));
    await page.screenshot({path:path.join(qa,'02_centered_formula.png')});
    await page.locator('h3').filter({hasText:'2.2 拆分静态'}).evaluate(x=>x.scrollIntoView({block:'start'}));
    await page.screenshot({path:path.join(qa,'03_beta_formula.png')});
    await page.locator('h3').filter({hasText:'2.3 联合概率'}).evaluate(x=>x.scrollIntoView({block:'start'}));
    await page.screenshot({path:path.join(qa,'06_loss_formula.png')});
    await page.locator('h3').filter({hasText:'2.4 TRAIN'}).evaluate(x=>x.scrollIntoView({block:'start'}));
    await page.screenshot({path:path.join(qa,'05_diagnostic_formula.png')});
    await page.locator('h3').filter({hasText:'5.1 三张 GPU'}).evaluate(x=>x.scrollIntoView({block:'start'}));
    await page.screenshot({path:path.join(qa,'04_commands.png')});
    fs.writeFileSync(path.join(qa,'layout.json'),JSON.stringify(result,null,2));
    console.log(JSON.stringify(result,null,2));
    if(result.displayMath!==result.expectedDisplayMath||result.unrenderedMath||result.horizontalOverflow||result.wideTables) {
      throw new Error('Document layout requires correction');
    }
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
