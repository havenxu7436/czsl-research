// Local HTML layout and Markdown link verification. No model inference.
const fs = require('fs');
const path = require('path');
const assert = require('assert');
const {pathToFileURL} = require('url');
const {chromium} = require('C:/Users/llenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const root = path.resolve(__dirname, '..');
  const base = 'VG-CPOD_v1_实验结果分析_20261018';
  const md = fs.readFileSync(path.join(root, base + '.md'), 'utf8');
  const links = [...md.matchAll(/!?\[[^\]]*\]\(([^)]+)\)/g)].map(m => m[1]);
  for (const link of links) assert(fs.existsSync(path.resolve(root, link)), link);
  assert.equal((md.match(/^\$\$$/gm) || []).length, 6);
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  const page = await browser.newPage({viewport: {width: 1600, height: 1080}, deviceScaleFactor: 1});
  await page.goto(pathToFileURL(path.join(root, base + '.html')).href);
  await page.evaluate(() => document.fonts.ready);
  const state = await page.evaluate(() => ({
    title: document.title,
    headings: document.querySelectorAll('h1').length,
    math: document.querySelectorAll('math').length,
    images: [...document.images].map(i => ({loaded: i.complete && i.naturalWidth > 0, embedded: i.src.startsWith('data:')})),
    horizontalOverflow: document.documentElement.scrollWidth > innerWidth,
    tables: document.querySelectorAll('table').length,
    tableWidths: [...document.querySelectorAll('table')].map(t => ({width: t.getBoundingClientRect().width, overflow: t.scrollWidth > t.clientWidth + 1})),
    formulas: [...document.querySelectorAll('math')].map(m => ({width: m.getBoundingClientRect().width, height: m.getBoundingClientRect().height}))
  }));
  assert.equal(state.headings, 1);
  assert.equal(state.math, 3);
  assert.equal(state.images.length, 7);
  assert(state.images.every(i => i.loaded && i.embedded));
  assert(!state.horizontalOverflow);
  assert(state.tableWidths.every(t => t.width > 1000 && !t.overflow));
  assert(state.formulas.every(m => m.width > 0 && m.height > 15));
  const qa = path.join(__dirname, 'qa');
  fs.mkdirSync(qa, {recursive: true});
  await page.screenshot({path: path.join(qa, 'report_top.png')});
  await page.locator('table').nth(1).screenshot({path: path.join(qa, 'main_table.png')});
  for (let i = 0; i < 3; i++) await page.locator('math').nth(i).screenshot({path: path.join(qa, 'formula_' + (i + 1) + '.png')});
  fs.writeFileSync(path.join(qa, 'layout_check.json'), JSON.stringify({...state, linksVerified: links.length}, null, 2));
  await browser.close();
  console.log(JSON.stringify({...state, linksVerified: links.length}));
})().catch(e => {console.error(e); process.exit(1);});
