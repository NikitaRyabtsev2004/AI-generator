#!/usr/bin/env node
/* eslint-disable no-console */

const fs = require('fs/promises');
const path = require('path');

const argv = new Set(process.argv.slice(2));
const isDryRun = argv.has('--dry-run');
const isConfirmed = argv.has('--yes');

const projectRoot = path.resolve(__dirname, '..');
const serverRoot = path.join(projectRoot, 'server');
const targets = [
  path.join(serverRoot, 'data'),
  path.join(serverRoot, 'artifacts'),
];

function isPathInside(parentPath, childPath) {
  const parent = path.resolve(parentPath);
  const child = path.resolve(childPath);
  const relative = path.relative(parent, child);
  return relative && !relative.startsWith('..') && !path.isAbsolute(relative);
}

function assertSafeTarget(targetPath) {
  const resolved = path.resolve(targetPath);
  if (!isPathInside(projectRoot, resolved)) {
    throw new Error(`Unsafe target outside project root: ${resolved}`);
  }
  if (!isPathInside(serverRoot, resolved)) {
    throw new Error(`Unsafe target outside server directory: ${resolved}`);
  }
  if (!targets.map((entry) => path.resolve(entry)).includes(resolved)) {
    throw new Error(`Target is not in the allow-list: ${resolved}`);
  }
}

async function removePath(targetPath) {
  assertSafeTarget(targetPath);
  await fs.rm(targetPath, {
    recursive: true,
    force: true,
    maxRetries: 3,
    retryDelay: 200,
  });
}

async function ensurePath(targetPath) {
  await fs.mkdir(targetPath, { recursive: true });
}

function printUsage() {
  console.log('Reset AI Generator server data');
  console.log('');
  console.log('Usage:');
  console.log('  node tools/reset-studio-data.js --yes');
  console.log('  node tools/reset-studio-data.js --yes --dry-run');
  console.log('');
  console.log('What will be removed:');
  targets.forEach((entry) => {
    console.log(`  - ${entry}`);
  });
}

async function main() {
  if (!isConfirmed) {
    printUsage();
    console.log('');
    console.log('Nothing was deleted. Add --yes to confirm.');
    process.exitCode = 1;
    return;
  }

  console.log(`Project root: ${projectRoot}`);
  console.log(`Mode: ${isDryRun ? 'dry-run' : 'execute'}`);

  for (const targetPath of targets) {
    console.log(`${isDryRun ? '[dry-run] would remove' : 'removing'}: ${targetPath}`);
    if (!isDryRun) {
      await removePath(targetPath);
    }
  }

  for (const targetPath of targets) {
    console.log(`${isDryRun ? '[dry-run] would recreate' : 'recreating'}: ${targetPath}`);
    if (!isDryRun) {
      await ensurePath(targetPath);
    }
  }

  console.log(isDryRun ? 'Dry-run finished.' : 'Reset finished successfully.');
}

main().catch((error) => {
  console.error(`Reset failed: ${error.message}`);
  process.exitCode = 1;
});

