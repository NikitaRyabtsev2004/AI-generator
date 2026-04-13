const os = require('os');
const { monitorEventLoopDelay } = require('node:perf_hooks');
const { getPythonBridgeHealth } = require('../engine/pythonBridge');

function toMb(value) {
  return Number((Number(value || 0) / (1024 * 1024)).toFixed(2));
}

function createServerStatusCollector({ getState, modelEngine, getRealtimeClients, getOverloadSnapshot }) {
  const startedAt = Date.now();
  const eventLoopDelay = monitorEventLoopDelay({ resolution: 20 });
  eventLoopDelay.enable();

  function buildStatus() {
    const state = getState();
    const memory = process.memoryUsage();
    const overload = typeof getOverloadSnapshot === 'function' ? getOverloadSnapshot() : {};
    const operationalStatus = typeof modelEngine?.getOperationalStatus === 'function'
      ? modelEngine.getOperationalStatus()
      : {};

    return {
      serverTime: new Date().toISOString(),
      startedAt: new Date(startedAt).toISOString(),
      uptimeSec: Math.floor(process.uptime()),
      node: {
        pid: process.pid,
        version: process.version,
        platform: process.platform,
        arch: process.arch,
      },
      resources: {
        cpuCount: os.cpus()?.length || 0,
        loadAverage: os.loadavg ? os.loadavg().map((value) => Number(value.toFixed(3))) : [],
        memory: {
          rssMb: toMb(memory.rss),
          heapTotalMb: toMb(memory.heapTotal),
          heapUsedMb: toMb(memory.heapUsed),
          externalMb: toMb(memory.external),
        },
        eventLoopLagMs: {
          min: Number((eventLoopDelay.min / 1e6).toFixed(3)),
          mean: Number((eventLoopDelay.mean / 1e6).toFixed(3)),
          max: Number((eventLoopDelay.max / 1e6).toFixed(3)),
          p95: Number((eventLoopDelay.percentile(95) / 1e6).toFixed(3)),
        },
      },
      channels: {
        realtimeClients: Number(getRealtimeClients?.() || 0),
        overload,
      },
      model: {
        lifecycle: state.model?.lifecycle || 'unknown',
        status: state.model?.status || 'unknown',
        trainedEpochs: Number(state.model?.trainedEpochs || 0),
      },
      training: {
        status: state.training?.status || 'unknown',
        phase: state.training?.phase || 'unknown',
        message: state.training?.message || '',
        queueRunnerStatus: state.trainingQueues?.runner?.status || 'idle',
        queueRunnerActive: Boolean(state.trainingQueues?.runner?.active),
      },
      runtime: operationalStatus,
      pythonBridge: getPythonBridgeHealth(),
    };
  }

  function dispose() {
    eventLoopDelay.disable();
  }

  return {
    buildStatus,
    dispose,
  };
}

module.exports = {
  createServerStatusCollector,
};
