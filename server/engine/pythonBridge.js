const { spawn } = require('node:child_process');
const fs = require('fs/promises');
const os = require('os');
const path = require('path');

const { log, logError } = require('../lib/logger');

const PYTHON_DIR = path.join(__dirname, '..', 'python');
const VENV_DIR = path.join(PYTHON_DIR, '.venv');
const GPU_VENV_DIR = path.join(PYTHON_DIR, '.venv2');
const REQUIREMENTS_PATH = path.join(PYTHON_DIR, 'requirements.txt');
const GPU_REQUIREMENTS_PATH = path.join(PYTHON_DIR, 'requirements.windows-gpu.txt');
const BACKEND_SCRIPT_PATH = path.join(PYTHON_DIR, 'keras_llm_backend.py');
const DEFAULT_TIMEOUT_MS = 120000;
const TENSORFLOW_PROBE_TIMEOUT_MS = 180000;
const VENV_CREATE_TIMEOUT_MS = 180000;
const PIP_INSTALL_TIMEOUT_MS = 45 * 60 * 1000;

let resolvedPythonRuntime = null;
let resolvingPythonRuntimePromise = null;
let resolvedPythonRuntimeNoTensorflow = null;
let resolvingPythonRuntimeNoTensorflowPromise = null;
const pythonBridgeHealth = {
  runtimeResolved: false,
  runtime: '',
  resolvedAt: '',
  lastResolveError: '',
  lastResolveErrorAt: '',
  activeProcesses: 0,
  lastCommandStartedAt: '',
  lastCommandFinishedAt: '',
  lastCommandError: '',
};

function setRuntimeResolved(candidate) {
  pythonBridgeHealth.runtimeResolved = Boolean(candidate);
  pythonBridgeHealth.runtime = candidate
    ? `${candidate.command} ${(candidate.argsPrefix || []).join(' ')}`.trim()
    : '';
  pythonBridgeHealth.resolvedAt = new Date().toISOString();
  pythonBridgeHealth.lastResolveError = '';
  pythonBridgeHealth.lastResolveErrorAt = '';
}

function setRuntimeResolveError(error) {
  pythonBridgeHealth.runtimeResolved = false;
  pythonBridgeHealth.runtime = '';
  pythonBridgeHealth.lastResolveError = String(error?.message || error || 'Unknown Python resolve error');
  pythonBridgeHealth.lastResolveErrorAt = new Date().toISOString();
}

function getPythonBridgeHealth() {
  return {
    ...pythonBridgeHealth,
  };
}

function parseCommandString(raw) {
  if (typeof raw !== 'string' || !raw.trim()) {
    return null;
  }

  const matches = raw.match(/"[^"]+"|'[^']+'|\S+/g) || [];
  if (!matches.length) {
    return null;
  }

  const normalized = matches.map((token) => token.replace(/^['"]|['"]$/g, ''));
  return {
    command: normalized[0],
    argsPrefix: normalized.slice(1),
  };
}

function getSystemPythonCandidates() {
  if (process.platform === 'win32') {
    return [
      { command: 'py', argsPrefix: ['-3.11'] },
      { command: 'py', argsPrefix: ['-3.10'] },
      { command: 'py', argsPrefix: ['-3.9'] },
      { command: 'python', argsPrefix: [] },
      { command: 'python3', argsPrefix: [] },
    ];
  }

  return [
    { command: 'python3', argsPrefix: [] },
    { command: 'python', argsPrefix: [] },
  ];
}

async function getBundledVenvPythonCandidates() {
  const pythonDir = path.join(__dirname, '..', 'python');
  const out = [];

  if (process.platform === 'win32') {
    const preferredVenvDirs = ['.venv2', '.venv'];
    for (const venvDir of preferredVenvDirs) {
      const exe = path.join(pythonDir, venvDir, 'Scripts', 'python.exe');
      try {
        await fs.access(exe);
        out.push({ command: exe, argsPrefix: [] });
      } catch (_error) {
        // Try next venv.
      }
    }
    return out;
  }

  for (const name of ['python3', 'python']) {
    const binPath = path.join(pythonDir, '.venv', 'bin', name);
    try {
      await fs.access(binPath);
      out.push({ command: binPath, argsPrefix: [] });
      break;
    } catch (_error) {
      // Try next binary name.
    }
  }

  return out;
}

async function buildPythonCandidates() {
  const candidates = [];
  const envCandidate = parseCommandString(process.env.AI_GENERATOR_PYTHON || '');
  if (envCandidate) {
    candidates.push(envCandidate);
  }
  candidates.push(...(await getBundledVenvPythonCandidates()));
  candidates.push(...getSystemPythonCandidates());
  return candidates;
}

function runProbe(candidate, timeoutMs = 15000) {
  return new Promise((resolve) => {
    const args = [...candidate.argsPrefix, '-c', 'import sys;print(sys.version)'];
    const child = spawn(candidate.command, args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
    });

    let stdout = '';
    let stderr = '';
    let settled = false;
    const timeoutId = setTimeout(() => {
      if (settled) {
        return;
      }
      settled = true;
      child.kill('SIGTERM');
      resolve({
        ok: false,
        stdout,
        stderr: `${stderr}\nPython probe timed out.`,
      });
    }, timeoutMs);

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });
    child.on('error', (error) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timeoutId);
      resolve({
        ok: false,
        stdout,
        stderr: `${stderr}\n${error.message}`,
      });
    });
    child.on('exit', (code) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timeoutId);
      resolve({
        ok: code === 0,
        stdout,
        stderr,
      });
    });
  });
}

function runTensorflowImportProbe(candidate, timeoutMs = TENSORFLOW_PROBE_TIMEOUT_MS) {
  return new Promise((resolve) => {
    const args = [
      ...candidate.argsPrefix,
      '-c',
      'import tensorflow as tf; print(tf.__version__)',
    ];
    const child = spawn(candidate.command, args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
    });

    let stdout = '';
    let stderr = '';
    let settled = false;
    const timeoutId = setTimeout(() => {
      if (settled) {
        return;
      }
      settled = true;
      child.kill('SIGTERM');
      resolve({
        ok: false,
        stdout,
        stderr: `${stderr}\nTensorFlow probe timed out.`,
      });
    }, timeoutMs);

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });
    child.on('error', (error) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timeoutId);
      resolve({
        ok: false,
        stdout,
        stderr: `${stderr}\n${error.message}`,
      });
    });
    child.on('exit', (code) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timeoutId);
      resolve({
        ok: code === 0,
        stdout,
        stderr,
      });
    });
  });
}

async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch (_error) {
    return false;
  }
}

function runSpawnCapture(command, args, options = {}) {
  const { cwd, timeoutMs = 60000, env } = options;
  return new Promise((resolve) => {
    const child = spawn(command, args, {
      cwd,
      env: env || process.env,
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
    });

    let stdout = '';
    let stderr = '';
    let settled = false;
    const timeoutId = setTimeout(() => {
      if (settled) {
        return;
      }
      settled = true;
      child.kill('SIGTERM');
      resolve({
        code: -1,
        stdout,
        stderr: `${stderr}\nCommand timed out after ${timeoutMs} ms.`,
      });
    }, timeoutMs);

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });
    child.on('error', (error) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timeoutId);
      resolve({
        code: -1,
        stdout,
        stderr: `${stderr}\n${error.message}`,
      });
    });
    child.on('exit', (code) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timeoutId);
      resolve({
        code: code === null ? -1 : code,
        stdout,
        stderr,
      });
    });
  });
}

async function getBundledVenvPythonExecutable() {
  if (process.platform === 'win32') {
    const candidates = [
      path.join(GPU_VENV_DIR, 'Scripts', 'python.exe'),
      path.join(VENV_DIR, 'Scripts', 'python.exe'),
    ];
    for (const exe of candidates) {
      if (await pathExists(exe)) {
        return exe;
      }
    }
    return null;
  }

  for (const name of ['python3', 'python']) {
    const binPath = path.join(VENV_DIR, 'bin', name);
    if (await pathExists(binPath)) {
      return binPath;
    }
  }

  return null;
}

async function getVenvPythonExecutableByName(venvName) {
  if (process.platform === 'win32') {
    const exe = path.join(PYTHON_DIR, venvName, 'Scripts', 'python.exe');
    return (await pathExists(exe)) ? exe : null;
  }

  for (const name of ['python3', 'python']) {
    const binPath = path.join(PYTHON_DIR, venvName, 'bin', name);
    if (await pathExists(binPath)) {
      return binPath;
    }
  }

  return null;
}

async function removeVenvDirIfPresent(venvName = '.venv') {
  const targetDir = path.join(PYTHON_DIR, venvName);
  try {
    await fs.rm(targetDir, { recursive: true, force: true });
  } catch (_error) {
    // Ignore.
  }
}

async function createBundledVenvBestEffort(options = {}) {
  const {
    venvName = '.venv',
    windowsPythonTags = ['3.12', '3.11', '3.10', '3.9'],
  } = options;

  if (process.platform === 'win32') {
    for (const tag of windowsPythonTags) {
      await removeVenvDirIfPresent(venvName);
      const result = await runSpawnCapture('py', [`-${tag}`, '-m', 'venv', venvName], {
        cwd: PYTHON_DIR,
        timeoutMs: VENV_CREATE_TIMEOUT_MS,
      });
      if (result.code === 0 && (await getVenvPythonExecutableByName(venvName))) {
        return true;
      }
    }

    await removeVenvDirIfPresent(venvName);
    let result = await runSpawnCapture('py', ['-3', '-m', 'venv', venvName], {
      cwd: PYTHON_DIR,
      timeoutMs: VENV_CREATE_TIMEOUT_MS,
    });
    if (result.code === 0 && (await getVenvPythonExecutableByName(venvName))) {
      return true;
    }

    await removeVenvDirIfPresent(venvName);
    result = await runSpawnCapture('python', ['-m', 'venv', venvName], {
      cwd: PYTHON_DIR,
      timeoutMs: VENV_CREATE_TIMEOUT_MS,
    });
    return result.code === 0 && Boolean(await getVenvPythonExecutableByName(venvName));
  }

  const attempts = [['python3'], ['python']];
  for (const argsPrefix of attempts) {
    await removeVenvDirIfPresent(venvName);
    const result = await runSpawnCapture(argsPrefix[0], ['-m', 'venv', venvName], {
      cwd: PYTHON_DIR,
      timeoutMs: VENV_CREATE_TIMEOUT_MS,
    });
    if (result.code === 0 && (await getVenvPythonExecutableByName(venvName))) {
      return true;
    }
  }

  return false;
}

async function pipInstallRequirements(venvPython, requirementsPath = REQUIREMENTS_PATH) {
  const upgrade = await runSpawnCapture(
    venvPython,
    ['-m', 'pip', 'install', '--upgrade', 'pip'],
    {
      cwd: PYTHON_DIR,
      timeoutMs: PIP_INSTALL_TIMEOUT_MS,
    }
  );
  if (upgrade.code !== 0) {
    log('warn', 'pip upgrade failed during auto-setup.', {
      stderr: String(upgrade.stderr || '').slice(-4000),
    });
  }

  const install = await runSpawnCapture(
    venvPython,
    ['-m', 'pip', 'install', '-r', requirementsPath],
    {
      cwd: PYTHON_DIR,
      timeoutMs: PIP_INSTALL_TIMEOUT_MS,
    }
  );

  if (install.code !== 0) {
    throw new Error(
      `pip install failed: ${String(install.stderr || install.stdout || '').trim().slice(-2000)}`
    );
  }
}

async function ensureBundledVenvHasTensorflow() {
  const isWindows = process.platform === 'win32';
  const preferredVenvName = isWindows ? '.venv2' : '.venv';
  const requirementsPath = isWindows ? GPU_REQUIREMENTS_PATH : REQUIREMENTS_PATH;
  let venvPython = await getVenvPythonExecutableByName(preferredVenvName);

  if (!venvPython) {
    log(
      'info',
      `Создаётся локальное окружение Python (server/python/${preferredVenvName}) для TensorFlow — первый запуск может занять несколько минут.`
    );
    const created = await createBundledVenvBestEffort({
      venvName: preferredVenvName,
      windowsPythonTags: isWindows && preferredVenvName === '.venv2' ? ['3.10'] : undefined,
    });
    if (!created) {
      return false;
    }
    venvPython = await getVenvPythonExecutableByName(preferredVenvName);
    if (!venvPython) {
      return false;
    }
  }

  const candidate = { command: venvPython, argsPrefix: [] };
  let tfProbe = await runTensorflowImportProbe(candidate);
  if (tfProbe.ok) {
    return true;
  }

  log(
    'info',
    `Устанавливаются зависимости Python (TensorFlow и др.) в server/python/${preferredVenvName} — это может занять несколько минут.`
  );

  try {
    await pipInstallRequirements(venvPython, requirementsPath);
  } catch (error) {
    logError('Автоустановка pip-зависимостей не удалась.', error);
    return false;
  }

  tfProbe = await runTensorflowImportProbe(candidate);
  return tfProbe.ok;
}

const TF_SETUP_HINT =
  'Установите зависимости Python: в каталоге server/python выполните setup-venv2.ps1 (Windows GPU, рекомендуется) или setup-venv.ps1 (обычный режим). Либо укажите переменную окружения AI_GENERATOR_PYTHON на python с установленным пакетом tensorflow.';

async function resolvePythonRuntime() {
  if (resolvedPythonRuntime) {
    return resolvedPythonRuntime;
  }

  if (!resolvingPythonRuntimePromise) {
    resolvingPythonRuntimePromise = (async () => {
      try {
        const skipAuto = String(process.env.AI_GENERATOR_PYTHON_SKIP_AUTO_VENV || '').trim() === '1';
        const hasExplicitRuntime = Boolean(parseCommandString(process.env.AI_GENERATOR_PYTHON || ''));

        if (process.platform === 'win32' && !skipAuto && !hasExplicitRuntime) {
          const gpuVenvPython = path.join(GPU_VENV_DIR, 'Scripts', 'python.exe');
          if (!(await pathExists(gpuVenvPython))) {
            await ensureBundledVenvHasTensorflow();
          }
        }

        const maxPasses = 2;
        for (let pass = 0; pass < maxPasses; pass += 1) {
          const candidates = await buildPythonCandidates();
          let anyPythonOk = false;

          for (const candidate of candidates) {
            const probe = await runProbe(candidate);
            if (!probe.ok) {
              continue;
            }

            anyPythonOk = true;
            const tfProbe = await runTensorflowImportProbe(candidate);
            if (tfProbe.ok) {
              resolvedPythonRuntime = candidate;
              setRuntimeResolved(candidate);
              return candidate;
            }
          }

          if (!anyPythonOk) {
            throw new Error(
              'Python interpreter was not found. Install Python 3.9+ and set AI_GENERATOR_PYTHON if needed.'
            );
          }

          if (pass === 0 && !skipAuto) {
            const ok = await ensureBundledVenvHasTensorflow();
            if (!ok) {
              throw new Error(
                `Обучение требует пакет tensorflow в среде Python. ${TF_SETUP_HINT}`
              );
            }
            continue;
          }

          throw new Error(`Обучение требует пакет tensorflow в среде Python. ${TF_SETUP_HINT}`);
        }

        throw new Error(`Обучение требует пакет tensorflow в среде Python. ${TF_SETUP_HINT}`);
      } catch (error) {
        setRuntimeResolveError(error);
        throw error;
      } finally {
        resolvingPythonRuntimePromise = null;
      }
    })();
  }

  return resolvingPythonRuntimePromise;
}

function runModuleImportProbe(candidate, moduleName, timeoutMs = 60000) {
  const normalizedModule = String(moduleName || '').trim();
  if (!normalizedModule) {
    return Promise.resolve({ ok: false, stdout: '', stderr: 'Module name is empty.' });
  }

  return new Promise((resolve) => {
    const args = [
      ...candidate.argsPrefix,
      '-c',
      `import ${normalizedModule}; print(${normalizedModule}.__name__)`,
    ];
    const child = spawn(candidate.command, args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
    });

    let stdout = '';
    let stderr = '';
    let settled = false;
    const timeoutId = setTimeout(() => {
      if (settled) {
        return;
      }
      settled = true;
      child.kill('SIGTERM');
      resolve({
        ok: false,
        stdout,
        stderr: `${stderr}\nModule probe timed out.`,
      });
    }, timeoutMs);

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });
    child.on('error', (error) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timeoutId);
      resolve({
        ok: false,
        stdout,
        stderr: `${stderr}\n${error.message}`,
      });
    });
    child.on('exit', (code) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timeoutId);
      resolve({
        ok: code === 0,
        stdout,
        stderr,
      });
    });
  });
}

async function resolvePythonRuntimeNoTensorflow() {
  if (resolvedPythonRuntimeNoTensorflow) {
    return resolvedPythonRuntimeNoTensorflow;
  }

  if (!resolvingPythonRuntimeNoTensorflowPromise) {
    resolvingPythonRuntimeNoTensorflowPromise = (async () => {
      try {
        const candidates = await buildPythonCandidates();
        let fallbackCandidate = null;
        for (const candidate of candidates) {
          const probe = await runProbe(candidate);
          if (!probe.ok) {
            continue;
          }
          fallbackCandidate = fallbackCandidate || candidate;

          const pyarrowProbe = await runModuleImportProbe(candidate, 'pyarrow');
          if (pyarrowProbe.ok) {
            resolvedPythonRuntimeNoTensorflow = candidate;
            setRuntimeResolved(candidate);
            return candidate;
          }
        }

        if (fallbackCandidate) {
          resolvedPythonRuntimeNoTensorflow = fallbackCandidate;
          setRuntimeResolved(fallbackCandidate);
          return fallbackCandidate;
        }

        throw new Error(
          'Python interpreter was not found. Install Python 3.9+ and set AI_GENERATOR_PYTHON if needed.'
        );
      } catch (error) {
        setRuntimeResolveError(error);
        throw error;
      } finally {
        resolvingPythonRuntimeNoTensorflowPromise = null;
      }
    })();
  }

  return resolvingPythonRuntimeNoTensorflowPromise;
}

async function createTempConfigFile(prefix, payload) {
  const targetDir = await fs.mkdtemp(path.join(os.tmpdir(), `${prefix}-`));
  const filePath = path.join(targetDir, 'config.json');
  await fs.writeFile(filePath, JSON.stringify(payload, null, 2), 'utf8');
  return {
    dir: targetDir,
    filePath,
  };
}

async function cleanupTempConfig(targetDir) {
  if (!targetDir) {
    return;
  }
  try {
    await fs.rm(targetDir, { recursive: true, force: true });
  } catch (_error) {
    // Ignore cleanup failures.
  }
}

async function startPythonBackendProcess(commandName, configPath, options = {}) {
  const runtime = options.requireTensorflow === false
    ? await resolvePythonRuntimeNoTensorflow()
    : await resolvePythonRuntime();
  const args = [...runtime.argsPrefix, BACKEND_SCRIPT_PATH, commandName, '--config', configPath];
  const child = spawn(runtime.command, args, {
    cwd: options.cwd || process.cwd(),
    env: {
      ...process.env,
      ...(options.env || {}),
      PYTHONUTF8: '1',
      PYTHONIOENCODING: 'utf-8',
      TF_CPP_MIN_LOG_LEVEL: process.env.TF_CPP_MIN_LOG_LEVEL || '2',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  });

  pythonBridgeHealth.activeProcesses += 1;
  pythonBridgeHealth.lastCommandStartedAt = new Date().toISOString();
  pythonBridgeHealth.lastCommandError = '';
  let finished = false;

  const finalize = (error = null) => {
    if (finished) {
      return;
    }
    finished = true;
    pythonBridgeHealth.activeProcesses = Math.max(0, pythonBridgeHealth.activeProcesses - 1);
    pythonBridgeHealth.lastCommandFinishedAt = new Date().toISOString();
    if (error) {
      pythonBridgeHealth.lastCommandError = String(error?.message || error || 'Python backend command failed');
    }
  };

  child.once('error', (error) => finalize(error));
  child.once('exit', () => finalize(null));
  return child;
}

function parseLastJsonLine(stdout = '') {
  const lines = String(stdout || '')
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter(Boolean);

  for (let index = lines.length - 1; index >= 0; index -= 1) {
    try {
      return JSON.parse(lines[index]);
    } catch (_error) {
      // Keep searching.
    }
  }

  return null;
}

async function runPythonBackendJson(commandName, payload, options = {}) {
  const temp = await createTempConfigFile(`ai-generator-${commandName}`, payload || {});
  const timeoutMs = Math.max(1000, Number(options.timeoutMs) || DEFAULT_TIMEOUT_MS);

  try {
    const child = await startPythonBackendProcess(commandName, temp.filePath, options);
    const result = await new Promise((resolve, reject) => {
      let stdout = '';
      let stderr = '';
      let settled = false;

      const timeoutId = setTimeout(() => {
        if (settled) {
          return;
        }
        settled = true;
        child.kill('SIGTERM');
        reject(new Error(`Python backend timed out after ${timeoutMs} ms.`));
      }, timeoutMs);

      child.stdout.on('data', (chunk) => {
        stdout += chunk.toString();
      });
      child.stderr.on('data', (chunk) => {
        stderr += chunk.toString();
      });
      child.on('error', (error) => {
        if (settled) {
          return;
        }
        settled = true;
        clearTimeout(timeoutId);
        reject(error);
      });
      child.on('exit', (code) => {
        if (settled) {
          return;
        }
        settled = true;
        clearTimeout(timeoutId);
        resolve({ code, stdout, stderr });
      });
    });

    const payloadJson = parseLastJsonLine(result.stdout);
    if (result.code !== 0) {
      const stderr = String(result.stderr || '').trim();
      const fallback = payloadJson?.error?.message || payloadJson?.error || stderr || 'Python backend command failed.';
      pythonBridgeHealth.lastCommandError = String(fallback);
      throw new Error(String(fallback));
    }

    if (!payloadJson) {
      throw new Error('Python backend returned empty or non-JSON output.');
    }

    return payloadJson;
  } finally {
    await cleanupTempConfig(temp.dir);
  }
}

module.exports = {
  BACKEND_SCRIPT_PATH,
  cleanupTempConfig,
  createTempConfigFile,
  getPythonBridgeHealth,
  resolvePythonRuntime,
  runPythonBackendJson,
  startPythonBackendProcess,
};
