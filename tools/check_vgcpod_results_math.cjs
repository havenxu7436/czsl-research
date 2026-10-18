// Parse the published report using a locally installed VS Code KaTeX bundle.
const fs = require('fs');
const vm = require('vm');
const path = require('path');
const crypto = require('crypto');
const args = process.argv.slice(2);
if (args.length !== 2 || args[0] !== '--renderer') throw new Error('Use --renderer PATH_TO_NOTEBOOK_KATEX_JS');
const root = path.resolve(__dirname, '..');
const base = path.join(root, 'VG-CPOD方案/v1/results/ut-zap50k-3seeds-20261018');
const source = fs.readFileSync(args[1], 'utf8');
const at = source.indexOf('var un=import.meta.url');
if (at < 0 || !source.startsWith('var tn=')) throw new Error('Unexpected KaTeX bundle');
const katex = vm.runInNewContext(source.slice(0, at) + ';rn()', {}, {timeout:10000});
const report = fs.readFileSync(path.join(base, 'analysis/分析报告.md'), 'utf8');
const math = [...report.matchAll(/```math\r?\n([\s\S]*?)\r?\n```/g)].map(m => m[1]);
const errors = [];
for (let i = 0; i < math.length; i++) {
  try { katex.renderToString(math[i], {throwOnError:true, displayMode:true, strict:'error'}); }
  catch (e) { errors.push({index:i, error:e.message}); }
}
const result = {passed:math.length === 3 && errors.length === 0, engine:'KaTeX', version:katex.version,
  display:math.length, errors, document_sha256:crypto.createHash('sha256').update(report).digest('hex'),
  scope:'Published GitHub math blocks parsed locally with strict rendering; layout screenshots preserved separately.'};
fs.writeFileSync(path.join(base, 'analysis/qa/repository_math_check.json'), JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result));
if (!result.passed) process.exitCode = 1;
