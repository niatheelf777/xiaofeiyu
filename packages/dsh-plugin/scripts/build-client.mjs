#!/usr/bin/env node
/**
 * Deterministic zero-dependency bundler assembling `client/styles.css` + `client/client-src.js`
 * into `lib/client.js` for DSH `dsh-client-modules` (`window.__ModuleLoader__.load`).
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const PKG_ROOT = path.resolve(SCRIPT_DIR, '..');
const CSS_PATH = path.join(PKG_ROOT, 'client', 'styles.css');
const SRC_PATH = path.join(PKG_ROOT, 'client', 'client-src.js');
const OUT_PATH = path.join(PKG_ROOT, 'lib', 'client.js');

export function buildClientBundleText() {
  const css = fs.readFileSync(CSS_PATH, 'utf8').replace(/\r\n/g, '\n').trim();
  const src = fs.readFileSync(SRC_PATH, 'utf8').replace(/\r\n/g, '\n');
  if (!src.includes('__XFY_CLIENT_CSS__')) {
    throw new Error('client-src.js is missing __XFY_CLIENT_CSS__ placeholder');
  }
  return src.replace('__XFY_CLIENT_CSS__', JSON.stringify(css));
}

export function writeClientBundle() {
  const rendered = buildClientBundleText();
  fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
  fs.writeFileSync(OUT_PATH, rendered, 'utf8');
  return { outPath: OUT_PATH, bytes: Buffer.byteLength(rendered, 'utf8') };
}

const isMain = process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url;
if (isMain) {
  const res = writeClientBundle();
  console.log(JSON.stringify(res));
}
