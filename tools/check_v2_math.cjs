// Read-only math rendering check using the user's installed VS Code KaTeX.
// No UI control, no network requests, no app or security setting changes.
const fs = require('node:fs');
const vm = require('node:vm');
const crypto = require('node:crypto');
const path = require('node:path');
const options = {};
for (let i=2;i<process.argv.length;i+=2) options[process.argv[i].replace(/^--/,'')] = process.argv[i+1];
for (const key of ['renderer','desktop','github','out']) if (!options[key]) throw new Error('Missing --'+key);
const source = fs.readFileSync(options.renderer,'utf8');
// The installed notebook bundle exposes its KaTeX UMD factory as rn().
// The only top-level ES export is the notebook activation callback; never call it.
if (!source.startsWith('var tn=') || !source.includes('var rn=tn(') || !source.endsWith('export{xs as activate};\n') && !source.endsWith('export{xs as activate};'))
  throw new Error('Installed renderer layout changed; inspect instead of guessing');
const bootstrap = source.indexOf('var un=import.meta.url');
if (bootstrap < 0) throw new Error('Notebook bootstrap marker changed');
const katex = vm.runInNewContext(source.slice(0,bootstrap)+';rn()',{}, {timeout:10000});
function parts(text,kind) {
  const display = [...text.matchAll(kind==='desktop'?/^\$\$\r?\n([\s\S]*?)\r?\n\$\$$/gm:/```math\r?\n([\s\S]*?)\r?\n```/g)].map(m=>m[1]);
  const stripped = text.replace(kind==='desktop'?/^\$\$\r?\n[\s\S]*?\r?\n\$\$$/gm:/```math\r?\n[\s\S]*?\r?\n```/g,'');
  const inline = [...stripped.matchAll(kind==='desktop'?/(?<!\$)\$(?!\$)([^\n]*?)\$(?!\$)/g:/\$`([^\n]*?)`\$/g)].map(m=>m[1]);
  return {inline,display};
}
const results = {};
for (const kind of ['desktop','github']) {
  const text = fs.readFileSync(options[kind],'utf8');
  const values = parts(text,kind);
  const errors = [];
  for (const scope of ['inline','display']) values[scope].forEach((tex,index)=>{
    try { katex.renderToString(tex,{throwOnError:true,displayMode:scope==='display',strict:'error'}); }
    catch (error) { errors.push({scope,index,error:error.message}); }
  });
  results[kind] = {inline:values.inline.length,display:values.display.length,errors,
                  document_sha256:crypto.createHash('sha256').update(text).digest('hex')};
}
const result = {passed:Object.values(results).every(v=>v.errors.length===0 && v.inline===28 && v.display===17),
  engine:'KaTeX bundled with installed VS Code markdown-math',engine_version:katex.version,
  renderer_sha256:crypto.createHash('sha256').update(source).digest('hex'),results,
  actual_UI_visual_check:'Incomplete: user stopped Computer Use with Escape. No further UI actions taken.',
  scope:'Actual installed KaTeX parser/renderToString only; not a screenshot/layout/GUI completion claim'};
fs.mkdirSync(path.dirname(options.out),{recursive:true});
fs.writeFileSync(options.out,JSON.stringify(result,null,2)+'\n');
process.stdout.write(JSON.stringify(result)+'\n');
process.exitCode = result.passed ? 0 : 1;
