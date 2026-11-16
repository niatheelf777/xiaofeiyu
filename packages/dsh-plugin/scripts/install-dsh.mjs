#!/usr/bin/env node
/**
 * Reversible installer/uninstaller for `@xiaofeiyu/dsh-plugin` into a DeepSeek Harness (`DSH_HOME`) profile.
 * Zero external dependencies; never touches `.credentials.yaml`.
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { writeClientBundle } from './build-client.mjs';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
export const PLUGIN_ROOT = path.resolve(SCRIPT_DIR, '..');
const PACKAGE_NAME = '@xiaofeiyu/dsh-plugin';

const PROFILE_TEMPLATES = {
  web: ['@deepseek-ai/dsh-base', '@deepseek-ai/dsh-web-app'],
  desktop: ['@deepseek-ai/dsh-base', '@deepseek-ai/dsh-web-app'],
  acp: ['@deepseek-ai/dsh-base', '@deepseek-ai/dsh-acp-app'],
  headless: ['@deepseek-ai/dsh-base', '@deepseek-ai/dsh-headless'],
  sdk: ['@deepseek-ai/dsh-base', '@deepseek-ai/dsh-sdk-app'],
};

const DEFAULT_PATCH_TEMPLATE = `# DSH profile patch layer\n[]\n`;

export function parseCliArgs(argv) {
  let mode = 'dry-run';
  let dshHome = '';
  let profile = 'web';
  let bridgePort = 18964;
  let runtimeMode = 'unified';
  let upstreamUrl = '';

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--dry-run') mode = 'dry-run';
    else if (arg === '--apply') mode = 'apply';
    else if (arg === '--uninstall') mode = 'uninstall';
    else if (arg === '--dsh-home' && argv[i + 1]) dshHome = argv[++i];
    else if (arg === '--profile' && argv[i + 1]) profile = argv[++i];
    else if (arg === '--bridge-port' && argv[i + 1]) bridgePort = Number(argv[++i]);
    else if (arg === '--runtime-mode' && argv[i + 1]) runtimeMode = argv[++i];
    else if (arg === '--upstream-url' && argv[i + 1]) upstreamUrl = argv[++i];
  }

  const resolvedHome = path.resolve(
    dshHome || (process.env.DSH_HOME && process.env.DSH_HOME.trim()) || path.join(os.homedir(), '.dsh')
  );
  if (!profile || profile.includes('/') || profile.includes('\\') || profile === '.' || profile === '..') {
    throw new Error(`Invalid profile name: ${JSON.stringify(profile)}`);
  }
  return {
    mode,
    dshHome: resolvedHome,
    profile,
    bridgePort: Number.isFinite(bridgePort) && bridgePort >= 0 ? bridgePort : 18964,
    runtimeMode: runtimeMode === 'translator' ? 'translator' : 'unified',
    upstreamUrl: upstreamUrl.trim(),
    pluginRoot: PLUGIN_ROOT,
  };
}

function readOptionalUtf8(filePath) {
  try {
    return fs.readFileSync(filePath, 'utf8');
  } catch (err) {
    if (err && err.code === 'ENOENT') return null;
    throw err;
  }
}

function createDirLink(targetDir, linkPath) {
  fs.mkdirSync(path.dirname(linkPath), { recursive: true });
  try {
    const st = fs.lstatSync(linkPath);
    if (st.isSymbolicLink() || st.isDirectory()) {
      fs.rmSync(linkPath, { recursive: true, force: true });
    }
  } catch (err) {
    if (!err || err.code !== 'ENOENT') throw err;
  }
  fs.symlinkSync(targetDir, linkPath, process.platform === 'win32' ? 'junction' : 'dir');
}

function removeEmptyParents(dirPath, stopAt) {
  let cur = path.resolve(dirPath);
  const root = path.resolve(stopAt);
  while (cur.startsWith(root) && cur !== root) {
    try {
      if (fs.readdirSync(cur).length > 0) break;
      fs.rmdirSync(cur);
      cur = path.dirname(cur);
    } catch {
      break;
    }
  }
}

export function runInstaller(options) {
  const {
    mode,
    dshHome,
    profile,
    bridgePort = 18964,
    runtimeMode = 'unified',
    upstreamUrl = '',
    pluginRoot = PLUGIN_ROOT,
  } = options;
  const profileDir = path.join(dshHome, 'profiles', profile);
  const manifestPath = path.join(profileDir, 'package.json');
  const patchPath = path.join(profileDir, 'cordis.patch.yml');
  const nodeModulesLink = path.join(profileDir, 'node_modules', '@xiaofeiyu', 'dsh-plugin');
  const presetDestDir = path.join(dshHome, '.agent-presets', 'xiaofeiyu');
  const presetYmlDest = path.join(presetDestDir, 'preset.yml');
  const agentCordisDest = path.join(presetDestDir, 'agent.cordis.yml');
  const backupDir = path.join(dshHome, '.xiaofeiyu-backup');
  const backupFile = path.join(backupDir, `${profile}.json`);

  const actions = [];

  if (mode === 'uninstall') {
    const rawBackup = readOptionalUtf8(backupFile);
    if (rawBackup !== null) {
      const backup = JSON.parse(rawBackup);
      for (const [relPath, prevContent] of Object.entries(backup.files || {})) {
        const absPath = path.join(dshHome, relPath);
        if (prevContent === null) {
          if (fs.existsSync(absPath)) {
            fs.rmSync(absPath, { recursive: true, force: true });
            removeEmptyParents(path.dirname(absPath), dshHome);
          }
        } else {
          fs.mkdirSync(path.dirname(absPath), { recursive: true });
          fs.writeFileSync(absPath, prevContent, 'utf8');
        }
      }
      try {
        fs.lstatSync(nodeModulesLink);
        fs.rmSync(nodeModulesLink, { recursive: true, force: true });
        removeEmptyParents(path.dirname(nodeModulesLink), profileDir);
      } catch {
        // already absent
      }
      fs.rmSync(backupFile, { force: true });
      removeEmptyParents(backupDir, dshHome);
      removeEmptyParents(profileDir, dshHome);
      actions.push({ op: 'restore_from_backup', backupFile });
      return { mode, dshHome, profile, profileDir, actions };
    }
    return { mode, dshHome, profile, profileDir, actions };
  }

  const origManifest = readOptionalUtf8(manifestPath);
  const origPatch = readOptionalUtf8(patchPath);
  const manifest = origManifest
    ? JSON.parse(origManifest)
    : {
        name: `dsh-profile-${profile}`,
        private: true,
        dependencies: {},
        dsh: { profile: { bundles: [...(PROFILE_TEMPLATES[profile] || ['@deepseek-ai/dsh-base'])] } },
      };

  manifest.dependencies = manifest.dependencies || {};
  manifest.dependencies[PACKAGE_NAME] = `file:${pluginRoot.replace(/\\/g, '/')}`;
  manifest.dsh = manifest.dsh || {};
  manifest.dsh.profile = manifest.dsh.profile || {};
  const bundles = Array.isArray(manifest.dsh.profile.bundles)
    ? [...manifest.dsh.profile.bundles]
    : [...(PROFILE_TEMPLATES[profile] || ['@deepseek-ai/dsh-base'])];
  if (!bundles.includes(PACKAGE_NAME)) bundles.push(PACKAGE_NAME);
  manifest.dsh.profile.bundles = bundles;
  const nextManifestText = JSON.stringify(manifest, null, 2) + '\n';

  let nextPatchText = origPatch ?? DEFAULT_PATCH_TEMPLATE;
  if (upstreamUrl || bridgePort !== 18964 || runtimeMode !== 'unified') {
    const lines = ['- id: xiaofeiyu', '  config:', `    runtimeMode: ${runtimeMode}`, `    bridgePort: ${bridgePort}`];
    if (upstreamUrl) {
      lines.push(
        runtimeMode === 'translator'
          ? `    translatorProxyUrl: ${upstreamUrl}`
          : `    upstreamBaseUrl: ${upstreamUrl}`
      );
    }
    if (bridgePort !== 18964) {
      lines.push(
        '',
        '- id: llm-pi-ai',
        '  config:',
        '    providers:',
        '      xiaofeiyu:',
        `        baseURL: http://127.0.0.1:${bridgePort}/v1`
      );
    }
    nextPatchText = `${lines.join('\n')}\n`;
  }

  const srcPresetYml = fs.readFileSync(path.join(pluginRoot, 'presets', 'xiaofeiyu', 'preset.yml'), 'utf8');
  const srcAgentCordis = fs.readFileSync(path.join(pluginRoot, 'presets', 'xiaofeiyu', 'agent.cordis.yml'), 'utf8');

  actions.push(
    { op: origManifest === null ? 'create_manifest' : 'update_manifest', path: manifestPath, bundles },
    { op: origPatch === null ? 'create_patch' : 'update_patch', path: patchPath },
    { op: 'link_plugin_package', link: nodeModulesLink, target: pluginRoot },
    { op: 'sync_agent_preset', dir: presetDestDir }
  );

  if (mode === 'apply') {
    if (pluginRoot === PLUGIN_ROOT && !fs.existsSync(path.join(pluginRoot, 'lib', 'client.js'))) {
      writeClientBundle();
    }
    if (!fs.existsSync(backupFile)) {
      fs.mkdirSync(backupDir, { recursive: true });
      const tracked = [
        path.relative(dshHome, manifestPath),
        path.relative(dshHome, patchPath),
        path.relative(dshHome, presetYmlDest),
        path.relative(dshHome, agentCordisDest),
      ];
      const filesSnapshot = {};
      for (const rel of tracked) filesSnapshot[rel] = readOptionalUtf8(path.join(dshHome, rel));
      fs.writeFileSync(
        backupFile,
        JSON.stringify({ createdAt: new Date().toISOString(), profile, pluginRoot, files: filesSnapshot }, null, 2) + '\n',
        'utf8'
      );
    }

    fs.mkdirSync(profileDir, { recursive: true });
    fs.writeFileSync(manifestPath, nextManifestText, 'utf8');
    fs.writeFileSync(patchPath, nextPatchText, 'utf8');
    createDirLink(pluginRoot, nodeModulesLink);
    fs.mkdirSync(presetDestDir, { recursive: true });
    fs.writeFileSync(presetYmlDest, srcPresetYml, 'utf8');
    fs.writeFileSync(agentCordisDest, srcAgentCordis, 'utf8');
  }

  return { mode, dshHome, profile, profileDir, bundles, bridgeUrl: `http://127.0.0.1:${bridgePort}/v1`, actions };
}

const isMain = process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url;
if (isMain) {
  console.log(JSON.stringify(runInstaller(parseCliArgs(process.argv.slice(2))), null, 2));
}
