// Update log
// - [2026-11-20] type=feat (AutoBotHtsH: one text size for the whole app (user 2026-11-20: 10 pt everywhere, important items bold, the
//   size set in points). Every fixed "font-size: Npx" becomes the size variable --fs (set from the 글자 크기 setting, default 9pt - was 10pt):
//   chart text (price-chart axis / legend / tags, recharts) var(--fs) - 2px so the plots keep their room, small marks (NXT tag, link
//   dot, badges, pills) var(--fs) - 3px, everything else var(--fs). Used by scope-bot-css.cjs for the AutoBotH stylesheet (its source
//   stays the same as AutoBotH's) and run once in place on the HtsH stylesheet: node scripts/font-vars.cjs src/renderer/src/styles.css)

'use strict';

const fs = require('fs');
const postcss = require('postcss');

const CHART = /\.(pc-|ind-|recharts|tl-axis|chart-side|ob-d)|\btext\b/;
const MARK = /\.(nxt-tag|nxt-mark|link-dot|badge|flag|pill|code-tag|tag\b|ob-p small|dd-caption)/;

function sizeFor(selector) {
  if (MARK.test(selector)) return 'calc(var(--fs) - 3px)';
  if (CHART.test(selector)) return 'calc(var(--fs) - 2px)';
  return 'var(--fs)';
}

/** rewrites the font-size declarations of a postcss root; returns how many changed */
function fontVars(root) {
  let n = 0;
  root.walkDecls('font-size', (d) => {
    if (!/^\s*[\d.]+px\s*$/.test(d.value)) return;
    const rule = d.parent;
    const sel = rule && rule.type === 'rule' ? rule.selector : '';
    d.value = sizeFor(sel);
    n++;
  });
  return n;
}

module.exports = { fontVars, sizeFor };

if (require.main === module) {
  const file = process.argv[2];
  if (!file) throw new Error('usage: node scripts/font-vars.cjs <css file>');
  const root = postcss.parse(fs.readFileSync(file, 'utf8'), { from: file });
  const n = fontVars(root);
  fs.writeFileSync(file, root.toString(), 'utf8');
  console.log(`[font-vars] ${file}: ${n} font sizes -> var(--fs)`);
}
