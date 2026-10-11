// Render a local document artifact; does not operate a user browser session.
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const root = path.resolve(__dirname, '..');
const title = '截至20261010_知识恢复构建处理_来源与作用完整档案';
const source = path.join(root, 'output', title + '.md');
const preview = path.join(root, 'output', title + '_公式渲染预览.html');
const outputDir = path.join(__dirname, 'math_render_review');
fs.mkdirSync(outputDir, {recursive:true});
const katexDir = path.join(root, 'experiment_history_20261010/math_render_deps/package/dist');
const katex = require(path.join(katexDir, 'katex.js'));
const deps = 'C:/Users/llenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {chromium} = require(path.join(deps, 'playwright'));

(async () => {
  const {marked} = await import(pathToFileURL(path.join(deps, 'marked/lib/marked.esm.js')).href);
  const markdown = fs.readFileSync(source, 'utf8');
  const math = [];
  const protectedMd = markdown.replace(/\$\$([\s\S]*?)\$\$|\$([^\n$]+)\$/g, (match, block, inline, offset) => {
    const tex = (block === undefined ? inline : block).trim();
    const displayMode = block !== undefined;
    const line = markdown.slice(0, offset).split('\n').length;
    const id = math.length;
    const rendered = katex.renderToString(tex, {displayMode, throwOnError:true, strict:'error', trust:false, output:'htmlAndMathml'});
    math.push({id,line,displayMode,tex,rendered});
    return `MATHITEMTOKEN${String(id).padStart(4,'0')}END`;
  });
  let css = fs.readFileSync(path.join(katexDir, 'katex.min.css'), 'utf8');
  css = css.replace(/url\(([^)]+)\)/g, (full,asset) => {
    const font = path.resolve(katexDir, asset.replace(/^['"]|['"]$/g,''));
    return `url("data:font/${path.extname(font).slice(1)};base64,${fs.readFileSync(font).toString('base64')}")`;
  });
  const style = `body{font:16px/1.75 "Microsoft YaHei",sans-serif;color:#222;background:#fff;margin:0;padding:24px}main{max-width:1080px;margin:auto}h1{font-size:26px}h2{font-size:22px;border-bottom:1px solid #ddd;margin-top:32px}h3{font-size:18px}p{margin:14px 0}table{border-collapse:collapse;display:block;overflow-x:auto;font-size:13px}th,td{border:1px solid #ddd;padding:6px 10px}code{background:#f2f3f4;font-size:.9em;overflow-wrap:anywhere}pre{overflow-x:auto}a{color:#145dad;overflow-wrap:anywhere}.math-display{padding:12px 0;margin:14px 0}.katex{font-size:1.1em}.katex-display{margin:0;text-align:left}.katex-display>.katex{text-align:left}.math-inline{white-space:nowrap}.gallery-item{padding:18px 0;border-bottom:1px solid #ddd}.label{font:13px "Microsoft YaHei",sans-serif;color:#666;margin-bottom:10px}`;
  let full = marked.parse(protectedMd).replace(/<p>(MATHITEMTOKEN\d{4}END)<\/p>/g, '$1');
  for (const item of math) {
    const token = `MATHITEMTOKEN${String(item.id).padStart(4,'0')}END`;
    const tag = item.displayMode ? 'div' : 'span';
    full = full.replaceAll(token, `<${tag} class="math-${item.displayMode?'display':'inline'}" data-math-id="${item.id}" data-source-line="${item.line}">${item.rendered}</${tag}>`);
  }
  full = full.replace(/href="\/C:\/([^"#]*)"/g, (_,target) => `href="${pathToFileURL('C:/'+target).href}"`);
  const html = body => `<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>${title}</title><style>${css}\n${style}</style><main>${body}</main></html>`;
  const gallery = math.filter(x=>x.displayMode).map(x=>`<section class="gallery-item"><div class="label">公式 ${x.id+1} · Markdown 第 ${x.line} 行</div><div class="math-display" data-math-id="${x.id}" data-source-line="${x.line}">${x.rendered}</div></section>`).join('');
  const galleryPath = path.join(outputDir, 'formula_gallery.html');
  fs.writeFileSync(preview, html(full));
  fs.writeFileSync(galleryPath, html(gallery));
  const browser = await chromium.launch({headless:true, executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
  try {
    const page = await browser.newPage();
    await page.goto(pathToFileURL(galleryPath).href);
    await page.evaluate(()=>document.fonts.ready);
    const checks = [];
    for (const width of [1180,720,620]) {
      await page.setViewportSize({width,height:1000});
      const measures = await page.locator('.math-display').evaluateAll(items=>items.map(item=>{
        const formula = item.querySelector('.katex-display>.katex>.katex-html');
        const bases = Array.from(formula.querySelectorAll(':scope > .base')).map(x=>x.getBoundingClientRect());
        const naturalWidth = bases.length ? Math.max(...bases.map(x=>x.right))-Math.min(...bases.map(x=>x.left)) : formula.scrollWidth;
        return {id:Number(item.dataset.mathId),line:Number(item.dataset.sourceLine),available:item.clientWidth,width:Math.ceil(naturalWidth),height:Math.ceil(formula.getBoundingClientRect().height),overflow:naturalWidth>item.clientWidth+1};
      }));
      checks.push({viewport:width,overflow:measures.filter(x=>x.overflow),measures});
    }
    await page.setViewportSize({width:1180,height:1400});
    await page.setViewportSize({width:1180,height:await page.evaluate(()=>Math.ceil(document.documentElement.scrollHeight))});
    const cards = page.locator('.gallery-item');
    const shots = [];
    for (let start=0;start<await cards.count();start+=6) {
      const end = Math.min(await cards.count()-1,start+5);
      const first = await cards.nth(start).boundingBox();
      const last = await cards.nth(end).boundingBox();
      const output = path.join(outputDir, `formulas_${String(start+1).padStart(2,'0')}-${String(end+1).padStart(2,'0')}.png`);
      await page.screenshot({path:output,clip:{x:0,y:first.y,width:1180,height:Math.ceil(last.y+last.height-first.y)}});
      shots.push(output);
    }
    await page.goto(pathToFileURL(preview).href);
    await page.evaluate(()=>document.fonts.ready);
    if (await page.locator('.katex-error').count()) throw new Error('KaTeX error in document preview');
    if (await page.locator('.math-inline,.math-display').count() !== math.length) throw new Error('Unexpected math extraction count');
    const report = {katex_version:katex.version,display_formulas:math.filter(x=>x.displayMode).length,inline_formulas:math.filter(x=>!x.displayMode).length,parse_errors:0,checks,screenshots:shots,full_preview:preview,preview_fonts_embedded:true};
    fs.writeFileSync(path.join(outputDir,'math_audit.json'),JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify({display_formulas:report.display_formulas,inline_formulas:report.inline_formulas,parse_errors:0,overflows:checks.map(x=>({viewport:x.viewport,count:x.overflow.length,lines:x.overflow.map(y=>y.line)})),screenshots:shots.length}));
    if (checks.some(x=>x.overflow.length)) throw new Error('Formula overflow at a tested width');
  } finally {await browser.close();}
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
