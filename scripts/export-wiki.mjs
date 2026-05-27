#!/usr/bin/env node
/**
 * Export frontend wiki content → GitHub wiki markdown files.
 * Run from repo root: node scripts/export-wiki.mjs
 *
 * Produces: docs/github-wiki/*.md  (ready to push to {repo}.wiki.git)
 */

import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const PAGES_DIR = join(ROOT, 'frontend/src/content/wiki/pages');
const OUT_DIR = join(ROOT, 'docs/github-wiki');

mkdirSync(OUT_DIR, { recursive: true });

// Page slug → GitHub wiki filename mapping
const PAGE_MAP = [
  ['home',                 'Home'],
  ['installation-linux',   'Installation-Linux'],
  ['installation-windows', 'Installation-Windows'],
  ['installation-macos',   'Installation-macOS'],
  ['setup-wizard',         'Setup-Wizard'],
  ['infrastructure',       'Infrastructure'],
  ['data-collection',      'Data-Collection'],
  ['labeling',             'Labeling'],
  ['training',             'Training'],
  ['inference',            'Inference-and-Analysis'],
  ['fine-tuning',          'Fine-Tuning'],
  ['architecture',         'Architecture'],
  ['api-reference',        'API-Reference'],
  ['troubleshooting',      'Troubleshooting'],
];

/**
 * Extract the English template literal content from a wiki TS file.
 * Works by finding `const en = \`` and reading until the closing `\`;`
 */
function extractEnContent(src) {
  // Find start of the en template literal
  const startMarker = 'const en = `';
  const startIdx = src.indexOf(startMarker);
  if (startIdx === -1) throw new Error('Could not find `const en = \\``');

  let i = startIdx + startMarker.length;
  let content = '';

  while (i < src.length) {
    // Escaped backtick inside template literal → emit literal backtick
    if (src[i] === '\\' && src[i + 1] === '`') {
      content += '`';
      i += 2;
      continue;
    }
    // Unescaped backtick → end of template literal
    if (src[i] === '`') {
      break;
    }
    // Escaped backslash-n → newline (for \\n in source)
    // (template literals already preserve real newlines, so just pass through)
    content += src[i];
    i++;
  }

  return content.trim();
}

let exported = 0;
for (const [slug, wikiName] of PAGE_MAP) {
  const srcPath = join(PAGES_DIR, `${slug}.ts`);
  const src = readFileSync(srcPath, 'utf8');

  try {
    const content = extractEnContent(src);
    const outPath = join(OUT_DIR, `${wikiName}.md`);
    writeFileSync(outPath, content + '\n', 'utf8');
    console.log(`✓ ${wikiName}.md`);
    exported++;
  } catch (e) {
    console.error(`✗ ${slug}: ${e.message}`);
  }
}

console.log(`\nExported ${exported}/${PAGE_MAP.length} pages → docs/github-wiki/`);
