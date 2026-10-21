const fs=require('fs');
const path=require('path');
const {pathToFileURL}=require('url');
const {chromium}=require('C:/Users/llenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async()=>{
  const root=path.resolve(__dirname,'..');
  const name='VG-CPOD_v1_J组结果分析与下一轮方向_20261021';
  const md=fs.readFileSync(path.join(root,name+'.md'),'utf8');
  const qa=path.join(__dirname,'qa');fs.mkdirSync(qa,{recursive:true});
  const browser=await chromium.launch({channel:'msedge',headless:true});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1100},deviceScaleFactor:1});
    await page.goto(pathToFileURL(path.join(root,name+'.html')).href);
    await page.evaluate(()=>document.fonts.ready);
    const result=await page.evaluate(()=>({
      displayMath:document.querySelectorAll('math[display="block"]').length,
      mathCount:document.querySelectorAll('math').length,
      unrenderedMath:document.querySelectorAll('span.math').length,
      horizontalOverflow:document.documentElement.scrollWidth>innerWidth,
      wideTables:[...document.querySelectorAll('table')].filter(x=>x.getBoundingClientRect().right>innerWidth).length,
      tables:document.querySelectorAll('table').length,
      images:[...document.images].map(x=>({loaded:x.complete&&x.naturalWidth>0,width:x.naturalWidth}))
    }));
    result.expectedDisplayMath=(md.match(/\$\$/g)||[]).length/2;
    await page.screenshot({path:path.join(qa,'01_results.png')});
    for(const [selector,title,file] of [
      ['h2','2. 剩余差距','02_ranking.png'],
      ['h3','3.2 两路修正','03_components.png'],
      ['h3','4.2 checkpoint','04_adam_formula.png'],
      ['h2','5. 下一轮具体','05_plan.png']]){
      await page.locator(selector).filter({hasText:title}).evaluate(x=>x.scrollIntoView({block:'start'}));
      await page.screenshot({path:path.join(qa,file)});
    }
    fs.writeFileSync(path.join(qa,'layout.json'),JSON.stringify(result,null,2));
    console.log(JSON.stringify(result,null,2));
    if(result.displayMath!==result.expectedDisplayMath||result.unrenderedMath||result.horizontalOverflow||result.wideTables||result.images.some(x=>!x.loaded))throw new Error('Layout requires correction');
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
