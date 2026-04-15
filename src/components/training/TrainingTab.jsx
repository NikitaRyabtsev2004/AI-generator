import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Button,
  FormControlLabel,
  IconButton,
  LinearProgress,
  Radio,
  RadioGroup,
  Slider,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import DeleteOutlineRoundedIcon from '@mui/icons-material/DeleteOutlineRounded';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import AddLinkRoundedIcon from '@mui/icons-material/AddLinkRounded';
import UploadFileRoundedIcon from '@mui/icons-material/UploadFileRounded';
import CachedRoundedIcon from '@mui/icons-material/CachedRounded';
import PlayArrowRoundedIcon from '@mui/icons-material/PlayArrowRounded';
import PauseRoundedIcon from '@mui/icons-material/PauseRounded';
import UndoRoundedIcon from '@mui/icons-material/UndoRounded';
import DeleteForeverRoundedIcon from '@mui/icons-material/DeleteForeverRounded';
import GlassPanel from '../shared/GlassPanel';
import MetricCard from '../shared/MetricCard';
import StatusPill from '../shared/StatusPill';
import TrainingChart from './TrainingChart';
import { SETTING_GROUPS, STATUS_FLOW, TOGGLE_CONTROLS } from '../../constants/modelConfig';
import {
  formatDateTime,
  formatDecimal,
  formatNumber,
  previewText,
  valueWithUnit,
} from '../../utils/text';
import '../../styles/training-tab.css';

const SUPPORTED_SOURCE_FILE_ACCEPT = '.txt,.csv,.json,.jsonl,.ndjson,.parquet,text/plain,text/csv,application/json,application/x-ndjson,application/parquet,application/vnd.apache.parquet';

function formatSettingValue(control, value) {
  if (control.precision) {
    return valueWithUnit(value, control.unit, control.precision);
  }

  return valueWithUnit(value, control.unit);
}

function clampControlValue(control, value) {
  const numericValue = Number(value);
  if (Number.isNaN(numericValue)) {
    return control.min;
  }

  const clamped = Math.min(control.max, Math.max(control.min, numericValue));
  if (control.precision) {
    return Number(clamped.toFixed(control.precision));
  }

  return clamped;
}

function estimateParameterCount(settings = {}) {
  const training = settings.training || {};
  const sequenceLength = Math.max(Number(training.sequenceLength) || 0, 1);
  const embeddingSize = Math.max(Number(training.embeddingSize) || 0, 1);
  const transformerLayers = Math.max(Number(training.transformerLayers) || 0, 1);
  const feedForwardSize = Math.max(Number(training.feedForwardSize) || 0, embeddingSize);
  const vocabularyLimit = Math.max(Number(training.vocabularyLimit) || 0, 256);

  const tokenEmbeddings = vocabularyLimit * embeddingSize;
  const positionalEmbeddings = sequenceLength * embeddingSize;
  const attentionWeightsPerLayer = 4 * embeddingSize * embeddingSize;
  const feedForwardWeightsPerLayer = 2 * embeddingSize * feedForwardSize;
  const layerNormAndBiasPerLayer = (embeddingSize * 6) + feedForwardSize;
  const transformerWeights = transformerLayers * (
    attentionWeightsPerLayer +
    feedForwardWeightsPerLayer +
    layerNormAndBiasPerLayer
  );
  const outputProjection = vocabularyLimit * embeddingSize;

  return Math.round(
    tokenEmbeddings +
    positionalEmbeddings +
    transformerWeights +
    outputProjection
  );
}

const TARGET_MODEL_PARAMETER_COUNT = 500_000_000;

function buildControlMap() {
  const map = new Map();
  SETTING_GROUPS.forEach((group) => {
    group.controls.forEach((control) => {
      map.set(`${control.section}.${control.key}`, control);
    });
  });
  return map;
}

const CONTROL_MAP = buildControlMap();

function getControl(section, key) {
  return CONTROL_MAP.get(`${section}.${key}`) || null;
}

function snapToControl(control, value, fallback = null) {
  if (!control) {
    return Number(value) || 0;
  }
  const baseValue = Number.isFinite(Number(value)) ? Number(value) : (fallback ?? control.min);
  const bounded = Math.max(control.min, Math.min(control.max, baseValue));
  const step = Number(control.step) || 1;
  const snapped = control.min + (Math.round((bounded - control.min) / step) * step);
  if (control.precision) {
    return Number(snapped.toFixed(control.precision));
  }
  return snapped;
}

function resolveAttentionHeads(embeddingSize, requestedHeads, maxHeads) {
  const allowed = [];
  for (let value = 1; value <= maxHeads; value += 1) {
    if (embeddingSize % value === 0) {
      allowed.push(value);
    }
  }
  if (!allowed.length) {
    return 1;
  }
  const requested = Math.max(1, Math.floor(Number(requestedHeads) || 1));
  const lowerOrEqual = allowed.filter((value) => value <= requested);
  if (lowerOrEqual.length) {
    return lowerOrEqual[lowerOrEqual.length - 1];
  }
  return allowed[0];
}

function estimateMaxSafeBatchSize(sequenceLength, embeddingSize, attentionHeads, transformerLayers, feedForwardSize = null) {
  const seq = Math.max(16, Number(sequenceLength) || 16);
  const emb = Math.max(32, Number(embeddingSize) || 32);
  const heads = Math.max(1, Number(attentionHeads) || 1);
  const layers = Math.max(1, Number(transformerLayers) || 1);

  let base;
  if (seq <= 128) base = 64;
  else if (seq <= 256) base = 24;
  else if (seq <= 384) base = 12;
  else if (seq <= 512) base = 8;
  else if (seq <= 768) base = 4;
  else if (seq <= 1024) base = 2;
  else base = 1;

  const widthPenalty = Math.sqrt(Math.max(1, emb / 512));
  const headPenalty = Math.sqrt(Math.max(1, heads / 8));
  const depthPenalty = Math.sqrt(Math.max(1, layers / 12));
  const feedForwardPenalty = feedForwardSize
    ? Math.sqrt(Math.max(1, Number(feedForwardSize) / Math.max(emb * 4, 1)))
    : 1;
  const corrected = Math.floor(base / (widthPenalty * headPenalty * depthPenalty * feedForwardPenalty));
  return Math.max(1, corrected);
}

function estimateAttentionTensorBytes(training = {}) {
  const batchSize = Math.max(1, Number(training.batchSize) || 1);
  const attentionHeads = Math.max(1, Number(training.attentionHeads) || 1);
  const sequenceLength = Math.max(16, Number(training.sequenceLength) || 16);
  return batchSize * attentionHeads * sequenceLength * sequenceLength * 4;
}

function estimateFfnTensorBytes(training = {}) {
  const batchSize = Math.max(1, Number(training.batchSize) || 1);
  const sequenceLength = Math.max(16, Number(training.sequenceLength) || 16);
  const feedForwardSize = Math.max(64, Number(training.feedForwardSize) || 64);
  return batchSize * sequenceLength * feedForwardSize * 4;
}

function estimateMaxSafeFeedForwardSize(sequenceLength, batchSize, transformerLayers, embeddingSize, controlMax = 32768) {
  const seq = Math.max(16, Number(sequenceLength) || 16);
  const batch = Math.max(1, Number(batchSize) || 1);
  const layers = Math.max(1, Number(transformerLayers) || 1);
  const emb = Math.max(32, Number(embeddingSize) || 32);

  const ratioCap = emb * 8;
  const activationBudgetBytes = 96 * 1024 * 1024;
  const denominator = Math.max(1, batch * seq * 4);
  const activationCap = Math.floor((activationBudgetBytes / denominator) / Math.max(1, Math.sqrt(layers / 12)));
  return Math.max(emb, Math.min(controlMax, ratioCap, activationCap));
}

function applyModelConstraints(settings = {}) {
  const nextSettings = structuredClone(settings || {});
  nextSettings.training = { ...(nextSettings.training || {}) };
  nextSettings.generation = { ...(nextSettings.generation || {}) };
  const adjustments = [];

  const sequenceControl = getControl('training', 'sequenceLength');
  const embeddingControl = getControl('training', 'embeddingSize');
  const headsControl = getControl('training', 'attentionHeads');
  const layersControl = getControl('training', 'transformerLayers');
  const feedForwardControl = getControl('training', 'feedForwardSize');
  const batchControl = getControl('training', 'batchSize');
  const chunkSizeControl = getControl('training', 'chunkSize');
  const chunkOverlapControl = getControl('training', 'chunkOverlap');
  const maxGeneratedTokensControl = getControl('generation', 'maxGeneratedTokens');

  const prevSequenceLength = Number(nextSettings.training.sequenceLength);
  nextSettings.training.sequenceLength = snapToControl(sequenceControl, prevSequenceLength, 128);

  const prevEmbeddingSize = Number(nextSettings.training.embeddingSize);
  nextSettings.training.embeddingSize = snapToControl(embeddingControl, prevEmbeddingSize, 128);

  const prevAttentionHeads = Number(nextSettings.training.attentionHeads);
  nextSettings.training.attentionHeads = resolveAttentionHeads(
    nextSettings.training.embeddingSize,
    prevAttentionHeads,
    headsControl?.max || 64
  );

  const prevLayers = Number(nextSettings.training.transformerLayers);
  nextSettings.training.transformerLayers = snapToControl(layersControl, prevLayers, 4);

  const feedForwardFloor = Math.max(
    nextSettings.training.embeddingSize,
    feedForwardControl?.min || 64
  );

  const prevChunkSize = Number(nextSettings.training.chunkSize);
  nextSettings.training.chunkSize = snapToControl(chunkSizeControl, prevChunkSize, 180);

  const prevChunkOverlap = Number(nextSettings.training.chunkOverlap);
  const overlapCeiling = Math.max(
    chunkOverlapControl?.min || 0,
    Math.min(chunkOverlapControl?.max || 1024, nextSettings.training.chunkSize - 8)
  );
  nextSettings.training.chunkOverlap = Math.min(
    overlapCeiling,
    snapToControl(chunkOverlapControl, prevChunkOverlap, 36)
  );

  const dynamicBatchMaxByCore = estimateMaxSafeBatchSize(
    nextSettings.training.sequenceLength,
    nextSettings.training.embeddingSize,
    nextSettings.training.attentionHeads,
    nextSettings.training.transformerLayers
  );
  const prevBatchSize = Number(nextSettings.training.batchSize);
  nextSettings.training.batchSize = Math.max(
    1,
    Math.min(dynamicBatchMaxByCore, snapToControl(batchControl, prevBatchSize, 8))
  );

  const prevFeedForward = Number(nextSettings.training.feedForwardSize);
  const dynamicFeedForwardMax = Math.max(
    feedForwardFloor,
    estimateMaxSafeFeedForwardSize(
      nextSettings.training.sequenceLength,
      nextSettings.training.batchSize,
      nextSettings.training.transformerLayers,
      nextSettings.training.embeddingSize,
      feedForwardControl?.max || 32768
    )
  );
  nextSettings.training.feedForwardSize = Math.max(
    feedForwardFloor,
    snapToControl(
      { ...feedForwardControl, min: feedForwardFloor, max: dynamicFeedForwardMax },
      prevFeedForward,
      nextSettings.training.embeddingSize * 4
    )
  );

  const dynamicBatchMax = estimateMaxSafeBatchSize(
    nextSettings.training.sequenceLength,
    nextSettings.training.embeddingSize,
    nextSettings.training.attentionHeads,
    nextSettings.training.transformerLayers,
    nextSettings.training.feedForwardSize
  );
  nextSettings.training.batchSize = Math.min(
    nextSettings.training.batchSize,
    Math.max(1, dynamicBatchMax)
  );

  const prevMaxGeneratedTokens = Number(nextSettings.generation.maxGeneratedTokens);
  const generatedLimit = Math.max(
    maxGeneratedTokensControl?.min || 16,
    Math.min(maxGeneratedTokensControl?.max || 4096, nextSettings.training.sequenceLength)
  );
  nextSettings.generation.maxGeneratedTokens = Math.min(
    generatedLimit,
    snapToControl(maxGeneratedTokensControl, prevMaxGeneratedTokens, 320)
  );

  if (prevAttentionHeads !== nextSettings.training.attentionHeads) {
    adjustments.push('`attentionHeads` автоматически подстроен под `embeddingSize` (должен делиться без остатка).');
  }
  if (prevBatchSize !== nextSettings.training.batchSize) {
    adjustments.push('`batchSize` ограничен динамически по длине контекста и размеру модели для снижения риска OOM.');
  }
  if (prevFeedForward !== nextSettings.training.feedForwardSize) {
    adjustments.push('`feedForwardSize` ограничен динамически по архитектуре и памяти (FFN/GELU).');
  }
  if (prevChunkOverlap !== nextSettings.training.chunkOverlap) {
    adjustments.push('`chunkOverlap` ограничен так, чтобы оставаться меньше `chunkSize`.');
  }
  if (prevMaxGeneratedTokens !== nextSettings.generation.maxGeneratedTokens) {
    adjustments.push('`maxGeneratedTokens` ограничен текущей длиной контекста модели.');
  }

  const parameterCount = estimateParameterCount(nextSettings);
  const attentionTensorBytes = estimateAttentionTensorBytes(nextSettings.training);
  const ffnTensorBytes = estimateFfnTensorBytes(nextSettings.training);
  const ratioToTarget = parameterCount / TARGET_MODEL_PARAMETER_COUNT;
  const nearTarget = ratioToTarget >= 0.8 && ratioToTarget <= 1.2;
  const attentionTensorGb = attentionTensorBytes / (1024 ** 3);
  const ffnTensorGb = ffnTensorBytes / (1024 ** 3);
  const highOomRisk = attentionTensorGb >= 1.1 || ffnTensorGb >= 0.2;

  return {
    settings: nextSettings,
    adjustments,
    dynamicLimits: {
      batchSizeMax: dynamicBatchMax,
      feedForwardSizeMax: dynamicFeedForwardMax,
      generatedTokensMax: generatedLimit,
      chunkOverlapMax: overlapCeiling,
    },
    diagnostics: {
      parameterCount,
      ratioToTarget,
      nearTarget,
      attentionTensorBytes,
      attentionTensorGb,
      ffnTensorBytes,
      ffnTensorGb,
      highOomRisk,
    },
  };
}

function formatRangeValue(control, value) {
  if (control.precision) {
    return Number(value).toFixed(control.precision);
  }
  return formatNumber(value);
}

function normalizeComparable(value) {
  if (Array.isArray(value)) {
    return value.map((item) => normalizeComparable(item));
  }

  if (value && typeof value === 'object') {
    return Object.keys(value)
      .sort()
      .reduce((accumulator, key) => {
        accumulator[key] = normalizeComparable(value[key]);
        return accumulator;
      }, {});
  }

  return value;
}

function stableStringify(value) {
  return JSON.stringify(normalizeComparable(value));
}

function modelFactRows(snapshot) {
  const { model, runtime, knowledge } = snapshot;

  return [
    ['Движок', model.engine],
    ['Жизненный цикл', model.lifecycle],
    ['Статус', model.status],
    ['Источников', formatNumber(model.sourceCount)],
    ['Чатов', formatNumber(model.chatCount)],
    ['Диалоговых пар', formatNumber(model.replyPairCount)],
    ['Фрагментов знаний', formatNumber(model.chunkCount)],
    ['Обучающих окон', formatNumber(model.trainingItemCount)],
    ['Батчей на эпоху', formatNumber(model.batchesPerEpoch)],
    ['Токенов корпуса', formatNumber(model.tokenCount)],
    ['Размер словаря', formatNumber(model.vocabularySize)],
    ['Параметров модели', formatNumber(model.parameterCount)],
    ['Обучено эпох', formatNumber(model.trainedEpochs)],
    ['План эпох', formatNumber(model.targetEpochs)],
    ['Последний loss', formatDecimal(model.lastLoss, 4)],
    ['Средний loss', formatDecimal(model.averageLoss, 4)],
    ['Validation loss', formatDecimal(model.validationLoss, 4)],
    ['Лучший validation loss', formatDecimal(model.bestValidationLoss, 4)],
    ['Perplexity', formatDecimal(model.perplexity, 4)],
    ['Средняя самооценка', formatDecimal(model.averageSelfScore, 3)],
    ['Положительных оценок', formatNumber(model.positiveFeedbackCount)],
    ['Отрицательных оценок', formatNumber(model.negativeFeedbackCount)],
    ['Последнее обучение', formatDateTime(model.lastTrainingAt)],
    ['Последняя генерация', formatDateTime(model.lastGenerationAt)],
    ['Стратегия контекста', runtime?.contextStrategy || 'n/a'],
    ['Генератор', runtime?.generatorBackend || 'neural'],
    ['Режим выполнения', snapshot.settings?.training?.executionMode || 'compatibility'],
    ['Backend обучения', model.computeBackendLabel || model.computeBackend || 'cpu'],
    ['Train-примеров в LM', formatNumber(knowledge?.languageModel?.trainingSampleCount || 0)],
    ['Validation-примеров', formatNumber(knowledge?.languageModel?.validationSampleCount || 0)],
  ];
}

function formatBytes(value) {
  const size = Math.max(Number(value) || 0, 0);
  if (!size) {
    return '0 B';
  }

  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let normalized = size;
  let unitIndex = 0;
  while (normalized >= 1024 && unitIndex < units.length - 1) {
    normalized /= 1024;
    unitIndex += 1;
  }

  const precision = normalized >= 100 || unitIndex === 0 ? 0 : 1;
  return `${normalized.toFixed(precision)} ${units[unitIndex]}`;
}

function formatDurationClock(totalSeconds) {
  const seconds = Math.max(0, Math.round(Number(totalSeconds) || 0));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainingSeconds = seconds % 60;
  return [
    String(hours).padStart(2, '0'),
    String(minutes).padStart(2, '0'),
    String(remainingSeconds).padStart(2, '0'),
  ].join(':');
}

function formatPercent(value) {
  if (!Number.isFinite(Number(value))) {
    return '0.00';
  }
  return Number(value).toFixed(2);
}

function formatQueueSourceStats(source = {}) {
  const typeLabel = String(source.type || 'file').toUpperCase();
  const tokenCount = Math.max(Number(source.stats?.tokenCount) || 0, 0);
  const charCount = Math.max(Number(source.stats?.charCount) || 0, 0);
  const rowCount = Math.max(Number(source.stats?.rowCount) || 0, 0);
  const columnCount = Math.max(Number(source.stats?.columnCount) || 0, 0);
  const contentSize = Math.max(Number(source.contentSize) || 0, 0);
  const datasetMeta = [];

  if (rowCount > 0) {
    datasetMeta.push(`${formatNumber(rowCount)} записей`);
  }
  if (columnCount > 0) {
    datasetMeta.push(`${formatNumber(columnCount)} колонок`);
  }

  if (tokenCount > 0 || charCount > 0) {
    return `${typeLabel} | ${formatNumber(tokenCount)} токенов | ${formatNumber(charCount)} символов${datasetMeta.length ? ` | ${datasetMeta.join(', ')}` : ''}`;
  }

  if (datasetMeta.length) {
    return `${typeLabel} | ${datasetMeta.join(', ')} | токены будут подсчитаны при обучении`;
  }

  if (contentSize > 0) {
    return `${typeLabel} | ${formatBytes(contentSize)} | токены будут подсчитаны при обучении`;
  }

  return `${typeLabel} | токены будут подсчитаны при обучении`;
}

function formatSourceStats(source = {}) {
  const typeLabel = String(source.type || 'file').toUpperCase();
  const tokenCount = Math.max(Number(source.stats?.tokenCount) || 0, 0);
  const charCount = Math.max(Number(source.stats?.charCount) || 0, 0);
  const rowCount = Math.max(Number(source.stats?.rowCount) || 0, 0);
  const columnCount = Math.max(Number(source.stats?.columnCount) || 0, 0);
  const datasetMeta = [];
  if (rowCount > 0) {
    datasetMeta.push(`${formatNumber(rowCount)} записей`);
  }
  if (columnCount > 0) {
    datasetMeta.push(`${formatNumber(columnCount)} колонок`);
  }

  return `${typeLabel} | ${formatNumber(tokenCount)} токенов | ${formatNumber(charCount)} символов${datasetMeta.length ? ` | ${datasetMeta.join(', ')}` : ''}`;
}

export default function TrainingTab({
  snapshot,
  busy,
  error,
  onNoticesChange,
  pendingAction,
  realtimeConnected,
  serverLogs,
  serverStatus,
  uploadProgress,
  processingProgress,
  onSaveSettings,
  onSaveRuntimeConfig,
  onUploadFiles,
  onCreateTrainingQueue,
  onUploadQueueFiles,
  onRemoveTrainingQueueSource,
  onDeleteTrainingQueue,
  onAddUrlSource,
  onRemoveSource,
  onCreateModel,
  onCreateNamedModel,
  onSelectModel,
  onDeleteLibraryModel,
  onTrainModel,
  onPauseModel,
  onRollbackModel,
  onResetModel,
}) {
  const snapshotConstrainedSettings = useMemo(
    () => applyModelConstraints(structuredClone(snapshot.settings)).settings,
    [snapshot.settings]
  );
  const snapshotSettingsKey = useMemo(
    () => stableStringify(snapshotConstrainedSettings),
    [snapshotConstrainedSettings]
  );
  const runtimeConfig = useMemo(() => snapshot.runtime?.config || {}, [snapshot.runtime?.config]);
  const runtimeSnapshotKey = useMemo(() => stableStringify(runtimeConfig), [runtimeConfig]);
  const [settingsDraft, setSettingsDraft] = useState(
    () => snapshotConstrainedSettings
  );
  const [runtimeDraft, setRuntimeDraft] = useState(() => structuredClone(runtimeConfig));
  const [urlInput, setUrlInput] = useState('');
  const [queueNameInput, setQueueNameInput] = useState('');
  const [modelNameInput, setModelNameInput] = useState('');
  const lastSnapshotKeyRef = useRef(snapshotSettingsKey);
  const currentDraftKeyRef = useRef(stableStringify(snapshotConstrainedSettings));
  const lastRuntimeSnapshotKeyRef = useRef(runtimeSnapshotKey);
  const currentRuntimeDraftKeyRef = useRef(stableStringify(runtimeConfig));

  useEffect(() => {
    currentDraftKeyRef.current = stableStringify(settingsDraft);
  }, [settingsDraft]);

  useEffect(() => {
    currentRuntimeDraftKeyRef.current = stableStringify(runtimeDraft);
  }, [runtimeDraft]);

  useEffect(() => {
    const previousSnapshotKey = lastSnapshotKeyRef.current;
    const currentDraftKey = currentDraftKeyRef.current;

    if (currentDraftKey === previousSnapshotKey || currentDraftKey === snapshotSettingsKey) {
      setSettingsDraft(snapshotConstrainedSettings);
      currentDraftKeyRef.current = snapshotSettingsKey;
    }

    lastSnapshotKeyRef.current = snapshotSettingsKey;
  }, [snapshotConstrainedSettings, snapshotSettingsKey]);

  useEffect(() => {
    const previousSnapshotKey = lastRuntimeSnapshotKeyRef.current;
    const currentDraftKey = currentRuntimeDraftKeyRef.current;

    if (currentDraftKey === previousSnapshotKey || currentDraftKey === runtimeSnapshotKey) {
      setRuntimeDraft(structuredClone(runtimeConfig));
      currentRuntimeDraftKeyRef.current = runtimeSnapshotKey;
    }

    lastRuntimeSnapshotKeyRef.current = runtimeSnapshotKey;
  }, [runtimeConfig, runtimeSnapshotKey]);

  const factRows = useMemo(() => modelFactRows(snapshot), [snapshot]);
  const statusEntries = snapshot.training?.recentStatuses || [];
  const trainingProgress = snapshot.training?.progress || {};
  const trainingEtaSeconds = Number.isFinite(Number(trainingProgress.etaSeconds))
    ? Math.max(0, Math.round(Number(trainingProgress.etaSeconds)))
    : null;
  const trainingAvgBatchMs = Number.isFinite(Number(trainingProgress.avgBatchTimeMs))
    ? Math.max(0, Number(trainingProgress.avgBatchTimeMs))
    : null;
  const trainingThroughputBps = Number.isFinite(Number(trainingProgress.throughputBatchesPerSec))
    ? Math.max(0, Number(trainingProgress.throughputBatchesPerSec))
    : null;
  const trainingPercentLabel = formatPercent(trainingProgress.percent);
  const trainingEpochNow = Math.max(0, Number(trainingProgress.currentEpoch) || 0);
  const trainingEpochTotal = Math.max(0, Number(trainingProgress.totalEpochs) || 0);
  const trainingBatchNow = Math.max(0, Number(trainingProgress.currentBatch) || 0);
  const trainingBatchTotal = Math.max(0, Number(trainingProgress.totalBatches) || 0);
  const modelConstraintReport = useMemo(
    () => applyModelConstraints(settingsDraft),
    [settingsDraft]
  );
  const hasUnsavedSettings = useMemo(
    () => stableStringify(settingsDraft) !== snapshotSettingsKey,
    [settingsDraft, snapshotSettingsKey]
  );
  const estimatedParameterCount = modelConstraintReport.diagnostics.parameterCount;
  const parameterDelta = estimatedParameterCount - TARGET_MODEL_PARAMETER_COUNT;
  const parameterDeltaSign = parameterDelta >= 0 ? '+' : '-';
  const parameterDeltaAbs = Math.abs(parameterDelta);
  const hasUnsavedRuntimeSettings = useMemo(
    () => stableStringify(runtimeDraft) !== runtimeSnapshotKey,
    [runtimeDraft, runtimeSnapshotKey]
  );
  const trainingQueues = snapshot.trainingQueues?.items || [];
  const modelRegistry = snapshot.modelRegistry?.items || [];
  const activeModelId = snapshot.modelRegistry?.activeModelId || null;
  const queueRunner = snapshot.trainingQueues?.runner || null;
  const queueRunnerAlertVisible = Boolean(
    queueRunner?.active ||
    ['error', 'paused', 'interrupted', 'rolled_back'].includes(queueRunner?.status)
  );
  const isQueueRunnerActive = Boolean(queueRunner?.active);
  const pendingQueueItems = trainingQueues.filter((queue) => queue.sourceCount > 0 && queue.status !== 'completed');
  const hasQueuedTraining = pendingQueueItems.length > 0;
  const canTrain = snapshot.sources.length > 0 || hasQueuedTraining;
  const isTraining = snapshot.training.status === 'training';
  const isSavingCheckpoint = snapshot.model.status === 'saving_checkpoint';
  const trainingLocked = isTraining || isSavingCheckpoint || isQueueRunnerActive;
  const isPaused = snapshot.training.status === 'paused';
  const isUploadingFiles = pendingAction === 'uploadFiles';
  const isUploadingQueueFiles = pendingAction === 'uploadQueueFiles';
  const normalizedUploadProgress = typeof uploadProgress === 'number'
    ? Math.min(100, Math.max(0, uploadProgress))
    : 0;
  const normalizedProcessingProgress = typeof processingProgress === 'number'
    ? Math.min(100, Math.max(0, processingProgress))
    : null;
  const isServerProcessing = normalizedProcessingProgress !== null && normalizedUploadProgress >= 100;
  const uploadIndicatorPercent = isServerProcessing
    ? normalizedProcessingProgress
    : normalizedUploadProgress;
  const hasModel = Boolean(
    snapshot.model.exists ||
    snapshot.model.trainedEpochs ||
    snapshot.knowledge?.languageModel?.checkpointReady
  );
  const trainActionLabel = isPaused ? 'Продолжить обучение' : 'Начать обучение';
  const primaryTrainActionLabel = hasQueuedTraining
    ? (isPaused ? 'Продолжить автоочередь' : 'Запустить автоочередь')
    : trainActionLabel;
  const canRollbackTraining = (isTraining || isSavingCheckpoint || isQueueRunnerActive) && snapshot.model.trainedEpochs > 0;
  const rollbackActionLabel = 'Остановить и откатить';
  const deleteActionLabel = trainingLocked ? 'Остановить и удалить' : 'Удалить модель';
  const updateSettingsDraft = (updater) => {
    setSettingsDraft((current) => {
      const nextDraft = typeof updater === 'function' ? updater(current) : updater;
      return applyModelConstraints(nextDraft).settings;
    });
  };

  const syncDraftsFromSnapshot = (nextSnapshot) => {
    if (!nextSnapshot) {
      return;
    }

    const nextSettings = applyModelConstraints(structuredClone(nextSnapshot.settings || {})).settings;
    const nextRuntimeConfig = structuredClone(nextSnapshot.runtime?.config || {});
    setSettingsDraft(nextSettings);
    setRuntimeDraft(nextRuntimeConfig);
    const nextSettingsKey = stableStringify(nextSettings);
    const nextRuntimeKey = stableStringify(nextRuntimeConfig);
    lastSnapshotKeyRef.current = nextSettingsKey;
    currentDraftKeyRef.current = nextSettingsKey;
    lastRuntimeSnapshotKeyRef.current = nextRuntimeKey;
    currentRuntimeDraftKeyRef.current = nextRuntimeKey;
  };

  const handleSaveSettingsClick = async () => {
    const constrainedDraft = modelConstraintReport.settings;
    if (stableStringify(constrainedDraft) !== stableStringify(settingsDraft)) {
      setSettingsDraft(constrainedDraft);
    }
    const nextSnapshot = await onSaveSettings(constrainedDraft);
    syncDraftsFromSnapshot(nextSnapshot);
    return nextSnapshot;
  };

  const handleSaveRuntimeClick = async () => {
    const nextSnapshot = await onSaveRuntimeConfig(runtimeDraft);
    syncDraftsFromSnapshot(nextSnapshot);
    return nextSnapshot;
  };

  const handleTrainClick = async () => {
    const constrainedDraft = modelConstraintReport.settings;
    if (stableStringify(constrainedDraft) !== stableStringify(settingsDraft)) {
      setSettingsDraft(constrainedDraft);
    }

    if (hasUnsavedSettings) {
      const savedSnapshot = await handleSaveSettingsClick();
      if (!savedSnapshot) {
        return;
      }
    }

    if (hasUnsavedRuntimeSettings) {
      const savedSnapshot = await handleSaveRuntimeClick();
      if (!savedSnapshot) {
        return;
      }
    }

    await onTrainModel();
  };

  const floatingNotices = useMemo(() => {
    const notices = [];

    if (error) {
      notices.push({ id: 'training-error', severity: 'error', message: error });
    }
    if (isPaused) {
      notices.push({ id: 'training-paused', severity: 'success', message: 'Обучение на паузе. Можно продолжить с чекпоинта.' });
    }
    if (hasUnsavedSettings || hasUnsavedRuntimeSettings) {
      notices.push({ id: 'training-unsaved', severity: 'warning', message: 'Есть несохранённые изменения настроек.' });
    }
    if (modelConstraintReport.adjustments.length) {
      notices.push({ id: 'training-adjustments', severity: 'info', message: 'Параметры авто-скорректированы под ограничения модели и памяти.' });
    }
    if (modelConstraintReport.diagnostics.highOomRisk) {
      notices.push({ id: 'training-oom-risk', severity: 'warning', message: `Высокий риск OOM: attention ~${formatBytes(modelConstraintReport.diagnostics.attentionTensorBytes)}, FFN ~${formatBytes(modelConstraintReport.diagnostics.ffnTensorBytes)}.` });
    }
    if (!canTrain) {
      notices.push({ id: 'training-no-data', severity: 'info', message: 'Добавьте хотя бы один источник (TXT/CSV/JSON/JSONL/PARQUET или URL).' });
    }
    if (hasQueuedTraining) {
      notices.push({ id: 'training-queue-ready', severity: 'info', message: `В очереди обучения: ${pendingQueueItems.length}.` });
    }
    if (queueRunnerAlertVisible) {
      notices.push({
        id: 'training-queue-runner',
        severity: queueRunner?.status === 'error' ? 'error' : 'info',
        message: `Автоочередь: ${queueRunner?.status || 'idle'}. Выполнено ${queueRunner?.completedQueueIds?.length || 0} из ${queueRunner?.totalQueues || 0}.`,
      });
    }
    if (snapshot.model.computeBackendWarning) {
      notices.push({ id: 'training-backend-warning', severity: 'warning', message: snapshot.model.computeBackendWarning });
    }

    const priority = { error: 0, warning: 1, success: 2, info: 3 };
    return notices
      .sort((left, right) => (priority[left.severity] ?? 10) - (priority[right.severity] ?? 10))
      .slice(0, 6);
  }, [
    canTrain,
    error,
    hasQueuedTraining,
    hasUnsavedRuntimeSettings,
    hasUnsavedSettings,
    isPaused,
    modelConstraintReport.adjustments.length,
    modelConstraintReport.diagnostics.attentionTensorBytes,
    modelConstraintReport.diagnostics.ffnTensorBytes,
    modelConstraintReport.diagnostics.highOomRisk,
    pendingQueueItems.length,
    queueRunner?.completedQueueIds?.length,
    queueRunner?.status,
    queueRunner?.totalQueues,
    queueRunnerAlertVisible,
    snapshot.model.computeBackendWarning,
  ]);

  useEffect(() => {
    onNoticesChange?.(floatingNotices);
    return () => {
      onNoticesChange?.([]);
    };
  }, [floatingNotices, onNoticesChange]);

  return (
    <div className="training-tab">
      <div className="training-grid">
        <GlassPanel className="panel--fit panel--settings">
          <div className="panel-heading panel-heading--full">
            <div className="panel-heading__copy">
              <Typography variant="h3">Настройки модели и обучения</Typography>
            </div>
          </div>

          <div className="settings-actions-grid">
            <Button
              className="action-button"
              variant="outlined"
              startIcon={<CachedRoundedIcon />}
              onClick={async () => {
                const nextSnapshot = modelNameInput.trim()
                  ? await onCreateNamedModel(modelNameInput.trim())
                  : await onCreateModel();
                if (nextSnapshot) {
                  setModelNameInput('');
                }
              }}
              disabled={busy || trainingLocked}
            >
              Новая модель
            </Button>
            <Button
              className="action-button"
              variant="contained"
              startIcon={<PlayArrowRoundedIcon />}
              onClick={handleTrainClick}
              disabled={busy || trainingLocked || !canTrain}
            >
              {primaryTrainActionLabel}
            </Button>
            <Button
              className="action-button"
              variant="outlined"
              startIcon={<PauseRoundedIcon />}
              onClick={() => onPauseModel()}
              disabled={busy || (!isTraining && !isQueueRunnerActive)}
            >
              Пауза
            </Button>
            <Button
              className="action-button"
              variant="outlined"
              color="warning"
              startIcon={<UndoRoundedIcon />}
              onClick={() => onRollbackModel()}
              disabled={busy || !canRollbackTraining}
            >
              {rollbackActionLabel}
            </Button>
            <Button
              className="action-button"
              color="error"
              variant="outlined"
              startIcon={<DeleteForeverRoundedIcon />}
              onClick={() => onResetModel()}
              disabled={busy || !hasModel}
            >
              {deleteActionLabel}
            </Button>
          </div>

          {isPaused ? (
            <Alert severity="success">
              Обучение поставлено на паузу. Можно продолжить с последнего чекпоинта.
            </Alert>
          ) : null}
          {trainingLocked ? (
            <Alert severity="info">
              Во время обучения редактирование корпуса и параметров заблокировано.
            </Alert>
          ) : null}
          {hasUnsavedSettings ? (
            <Alert severity="warning">
              Есть несохраненные изменения настроек.
            </Alert>
          ) : null}
          {modelConstraintReport.adjustments.length ? (
            <Alert severity="info">
              {`Автокоррекция параметров: ${modelConstraintReport.adjustments.join(' ')}`}
            </Alert>
          ) : null}
          {modelConstraintReport.diagnostics.highOomRisk ? (
            <Alert severity="warning">
              {`Высокий риск OOM: attention ~${formatBytes(modelConstraintReport.diagnostics.attentionTensorBytes)}, FFN/GELU ~${formatBytes(modelConstraintReport.diagnostics.ffnTensorBytes)} на шаге. Уменьшите batch size, sequence length или feed-forward size.`}
            </Alert>
          ) : null}
          {!canTrain ? (
            <Alert severity="info">
              Добавьте хотя бы один `TXT`, `CSV`, `JSON`-файл или URL.
            </Alert>
          ) : null}
          {hasQueuedTraining ? (
            <Alert severity="info">
              {`Готово очередей к автодообучению: ${pendingQueueItems.length}. Они будут применены по одной, каждая только после успешного сохранения чекпоинта предыдущей.`}
            </Alert>
          ) : null}
          {hasQueuedTraining && snapshot.sources.length > 0 ? (
            <Alert severity="warning">
              Ручная очередь источников и автоочереди разделены. При запуске автоочереди текущие загруженные файлы из ручного списка не будут подмешаны в нее автоматически.
            </Alert>
          ) : null}
          {queueRunnerAlertVisible ? (
            <Alert severity={queueRunner.status === 'error' ? 'error' : 'info'}>
              {`Автоочередь: ${queueRunner.status}. Выполнено ${queueRunner.completedQueueIds?.length || 0} из ${queueRunner.totalQueues || 0}.`}
            </Alert>
          ) : null}
          {snapshot.model.computeBackendWarning ? (
            <Alert severity="warning">
              {snapshot.model.computeBackendWarning}
            </Alert>
          ) : null}

          <div className="settings-scroll">
            <div className="settings-group">
              <Typography variant="subtitle1" className="settings-group__title">
                Библиотека моделей
              </Typography>
              <Typography variant="body2" className="muted-text">
                Здесь хранится список локальных моделей. При создании новой текущая модель сохраняется в библиотеке и ее можно выбрать обратно.
              </Typography>

              <div className="queue-builder">
                <TextField
                  className="text-input"
                  label="Название новой модели"
                  value={modelNameInput}
                  disabled={trainingLocked}
                  onChange={(event) => setModelNameInput(event.target.value)}
                  placeholder="Например, Русская LLM v2"
                />
                <Button
                  variant="outlined"
                  className="action-button"
                  onClick={async () => {
                    const nextSnapshot = modelNameInput.trim()
                      ? await onCreateNamedModel(modelNameInput.trim())
                      : await onCreateModel();
                    if (nextSnapshot) {
                      setModelNameInput('');
                    }
                  }}
                  disabled={busy || trainingLocked}
                >
                  Создать и переключить
                </Button>
              </div>

              <div className="settings-estimate-card">
                <div>
                  <Typography variant="subtitle2">Оценка параметров</Typography>
                  <Typography variant="caption" className="muted-text">
                    По текущим настройкам модель будет приблизительно этого масштаба. Значение удобно для проверки крупных конфигов.
                  </Typography>
                </div>
                <div className="settings-estimate-card__value">
                  <strong>{formatNumber(estimatedParameterCount)}</strong>
                  <span>params</span>
                </div>
              </div>
              <div className="settings-guidance-grid">
                <div className="settings-guidance-chip">
                  <Typography variant="caption" className="muted-text">Цель</Typography>
                  <Typography variant="body2">~{formatNumber(TARGET_MODEL_PARAMETER_COUNT)} params</Typography>
                </div>
                <div className="settings-guidance-chip">
                  <Typography variant="caption" className="muted-text">Отклонение</Typography>
                  <Typography variant="body2">{`${parameterDeltaSign}${formatNumber(parameterDeltaAbs)}`}</Typography>
                </div>
                <div className="settings-guidance-chip">
                  <Typography variant="caption" className="muted-text">Риск тензоров/OOM</Typography>
                  <Typography variant="body2">
                    {`${Math.max(
                      modelConstraintReport.diagnostics.attentionTensorGb,
                      modelConstraintReport.diagnostics.ffnTensorGb
                    ).toFixed(2)} GB`}
                  </Typography>
                </div>
              </div>
              {!modelConstraintReport.diagnostics.nearTarget ? (
                <Typography variant="caption" className="muted-text">
                  Для модели около 500M параметров поднимайте в первую очередь `embeddingSize` и `transformerLayers`, но компенсируйте это уменьшением `batchSize` и/или `sequenceLength`.
                </Typography>
              ) : null}

              <div className="model-library-list">
                {modelRegistry.length ? modelRegistry.map((entry) => (
                  <div
                    key={entry.id}
                    className={`model-library-card ${entry.id === activeModelId ? 'model-library-card--active' : ''}`}
                  >
                    <div className="model-library-card__head">
                      <div>
                        <Typography variant="subtitle2">{entry.name}</Typography>
                        <Typography variant="caption" className="muted-text">
                          {`${entry.summary?.backend || 'neural'} | ${formatNumber(entry.summary?.trainedEpochs || 0)} эпох | ${formatNumber(entry.summary?.parameterCount || 0)} параметров`}
                        </Typography>
                      </div>
                      <div className="training-queue-card__actions">
                        <StatusPill
                          label={entry.id === activeModelId ? 'active' : (entry.summary?.lifecycle || 'idle')}
                          active={entry.id === activeModelId}
                          tone={entry.id === activeModelId ? 'accent' : 'neutral'}
                        />
                        <Button
                          size="small"
                          variant="outlined"
                          className="action-button action-button--compact"
                          onClick={() => onSelectModel(entry.id)}
                          disabled={busy || trainingLocked || entry.id === activeModelId}
                        >
                          Выбрать
                        </Button>
                        <IconButton
                          onClick={() => onDeleteLibraryModel(entry.id)}
                          disabled={busy || trainingLocked || entry.id === activeModelId}
                        >
                          <DeleteOutlineRoundedIcon />
                        </IconButton>
                      </div>
                    </div>
                  </div>
                )) : (
                  <Typography variant="body2" className="muted-text">
                    В библиотеке пока нет сохраненных моделей.
                  </Typography>
                )}
              </div>
            </div>

            <div className="settings-group">
              <Typography variant="subtitle1" className="settings-group__title">
                Генератор и веб-поиск
              </Typography>
              <Typography variant="body2" className="muted-text">
                Генерация идет только через вашу локальную обучаемую Transformer-модель. Здесь настраивается веб-поиск и системное поведение генератора.
              </Typography>

              <div className="toggle-row">
                <div className="toggle-control">
                  <Typography variant="subtitle1">Веб-поиск</Typography>
                  <RadioGroup
                    row
                    value={runtimeDraft?.generation?.webSearchEnabled ? 'on' : 'off'}
                    onChange={(event) => {
                      const enabled = event.target.value === 'on';
                      setRuntimeDraft((current) => ({
                        ...current,
                        generation: {
                          ...current.generation,
                          webSearchEnabled: enabled,
                        },
                      }));
                    }}
                  >
                    <FormControlLabel value="off" control={<Radio />} disabled={trainingLocked} label="Выключен" />
                    <FormControlLabel value="on" control={<Radio />} disabled={trainingLocked} label="Включен" />
                  </RadioGroup>
                </div>
              </div>

              <div className="settings-control-list">
                <div className="settings-control">
                  <div className="settings-control__header">
                    <Typography variant="subtitle2">Предпочтительные домены для веб-поиска</Typography>
                  </div>
                  <TextField
                    className="text-input"
                    value={runtimeDraft?.generation?.webSearchPreferredDomains || ''}
                    disabled={trainingLocked}
                    onChange={(event) => {
                      const nextValue = event.target.value;
                      setRuntimeDraft((current) => ({
                        ...current,
                        generation: {
                          ...current.generation,
                          webSearchPreferredDomains: nextValue,
                        },
                      }));
                    }}
                    placeholder="docs.python.org, wikipedia.org, developer.mozilla.org"
                    helperText="Домены через запятую. Они будут подниматься выше в выдаче."
                  />
                </div>

                <div className="settings-number-grid">
                  <TextField
                    className="text-input"
                    type="number"
                    label="Результатов в поиске"
                    value={runtimeDraft?.generation?.webSearchMaxResults ?? 6}
                    disabled={trainingLocked}
                    inputProps={{ min: 1, max: 24, step: 1 }}
                    onChange={(event) => {
                      const nextValue = Math.min(24, Math.max(1, Number(event.target.value) || 1));
                      setRuntimeDraft((current) => ({
                        ...current,
                        generation: {
                          ...current.generation,
                          webSearchMaxResults: nextValue,
                        },
                      }));
                    }}
                  />
                  <TextField
                    className="text-input"
                    type="number"
                    label="Страниц для чтения"
                    value={runtimeDraft?.generation?.webSearchFetchPages ?? 3}
                    disabled={trainingLocked}
                    inputProps={{ min: 1, max: 12, step: 1 }}
                    onChange={(event) => {
                      const nextValue = Math.min(12, Math.max(1, Number(event.target.value) || 1));
                      setRuntimeDraft((current) => ({
                        ...current,
                        generation: {
                          ...current.generation,
                          webSearchFetchPages: nextValue,
                        },
                      }));
                    }}
                  />
                  <TextField
                    className="text-input"
                    type="number"
                    label="Timeout web-поиска"
                    value={runtimeDraft?.generation?.webSearchTimeoutMs ?? 12000}
                    disabled={trainingLocked}
                    inputProps={{ min: 2000, max: 60000, step: 1000 }}
                    onChange={(event) => {
                      const nextValue = Math.min(60000, Math.max(2000, Number(event.target.value) || 2000));
                      setRuntimeDraft((current) => ({
                        ...current,
                        generation: {
                          ...current.generation,
                          webSearchTimeoutMs: nextValue,
                        },
                      }));
                    }}
                  />
                </div>

                <div className="settings-control">
                  <div className="settings-control__header">
                    <Typography variant="subtitle2">Системный промпт</Typography>
                  </div>
                  <TextField
                    className="text-input"
                    multiline
                    minRows={4}
                    value={runtimeDraft?.generation?.systemPrompt || ''}
                    disabled={trainingLocked}
                    onChange={(event) => {
                      const nextValue = event.target.value;
                      setRuntimeDraft((current) => ({
                        ...current,
                        generation: {
                          ...current.generation,
                          systemPrompt: nextValue,
                        },
                      }));
                    }}
                  />
                </div>
              </div>

              <div className="settings-footer settings-footer--inline">
                <Button
                  variant="contained"
                  className="action-button"
                  onClick={handleSaveRuntimeClick}
                  disabled={busy || trainingLocked || !hasUnsavedRuntimeSettings}
                >
                  Сохранить runtime
                </Button>
              </div>
            </div>

            {TOGGLE_CONTROLS.length ? (
              <div className="toggle-row">
                {TOGGLE_CONTROLS.map((control) => {
                  const value = settingsDraft?.[control.section]?.[control.key];
                  return (
                    <div className="toggle-control" key={control.key}>
                      <Typography variant="subtitle1">{control.label}</Typography>
                      <Typography variant="body2" className="muted-text">
                        {control.description}
                      </Typography>
                      <RadioGroup
                        value={value}
                        onChange={(event) => {
                          const nextValue = event.target.value;
                          updateSettingsDraft((current) => ({
                            ...current,
                            [control.section]: {
                              ...current[control.section],
                              [control.key]: nextValue,
                            },
                          }));
                        }}
                      >
                        {control.options.map((option) => (
                          <FormControlLabel
                            key={option.value}
                            value={option.value}
                            control={<Radio />}
                            disabled={trainingLocked}
                            label={(
                              <div>
                                <Typography variant="subtitle2">{option.label}</Typography>
                                <Typography variant="caption" className="muted-text">
                                  {option.hint}
                                </Typography>
                              </div>
                            )}
                          />
                        ))}
                      </RadioGroup>
                    </div>
                  );
                })}
              </div>
            ) : null}

            <div className="settings-groups">
              {SETTING_GROUPS.map((group) => (
                <div className="settings-group" key={group.title}>
                  <Typography variant="subtitle1" className="settings-group__title">
                    {group.title}
                  </Typography>
                  <Typography variant="body2" className="muted-text">
                    {group.description}
                  </Typography>

                  <div className="settings-control-list">
                    {group.controls.map((control) => {
                      const value = settingsDraft?.[control.section]?.[control.key];
                      const embeddingSizeForHeads = Math.max(1, Number(settingsDraft?.training?.embeddingSize) || 1);
                      const maxAllowedHeads = (() => {
                        if (control.key !== 'attentionHeads') {
                          return control.max;
                        }
                        let candidate = 1;
                        for (let head = 1; head <= control.max; head += 1) {
                          if (embeddingSizeForHeads % head === 0) {
                            candidate = head;
                          }
                        }
                        return candidate;
                      })();
                      const dynamicMax = control.key === 'batchSize'
                        ? Math.max(control.min, modelConstraintReport.dynamicLimits.batchSizeMax)
                        : control.key === 'feedForwardSize'
                          ? Math.max(control.min, modelConstraintReport.dynamicLimits.feedForwardSizeMax)
                        : control.key === 'chunkOverlap'
                          ? Math.max(control.min, modelConstraintReport.dynamicLimits.chunkOverlapMax)
                          : control.key === 'maxGeneratedTokens'
                            ? Math.max(control.min, modelConstraintReport.dynamicLimits.generatedTokensMax)
                            : control.key === 'attentionHeads'
                              ? Math.max(1, maxAllowedHeads)
                              : control.max;
                      const dynamicMin = control.min;
                      const hasDynamicRange = dynamicMax < control.max;
                      const sliderMarks = hasDynamicRange
                        ? [
                            { value: control.min, label: `${formatRangeValue(control, control.min)}` },
                            { value: dynamicMax, label: `лимит ${formatRangeValue(control, dynamicMax)}` },
                          ]
                        : [
                            { value: control.min, label: `${formatRangeValue(control, control.min)}` },
                            { value: control.max, label: `${formatRangeValue(control, control.max)}` },
                          ];
                      const isOutsideDynamic = Number(value) > dynamicMax || Number(value) < dynamicMin;
                      return (
                        <div className="settings-control" key={control.key}>
                          <div className="settings-control__header">
                            <div className="settings-control__label-wrap">
                              <Typography variant="subtitle2">{control.label}</Typography>
                              <Tooltip title={control.hint}>
                                <InfoOutlinedIcon className="info-icon" />
                              </Tooltip>
                            </div>
                            <div className="settings-control__tools">
                              <span className="settings-control__value">
                                {formatSettingValue(control, value)}
                              </span>
                              <TextField
                                className="settings-control__input"
                                type="number"
                                size="small"
                                value={value}
                                disabled={trainingLocked}
                                inputProps={{
                                  min: dynamicMin,
                                  max: dynamicMax,
                                  step: control.step,
                                }}
                                error={isOutsideDynamic}
                                helperText={hasDynamicRange
                                  ? `Сейчас доступно: ${formatRangeValue(control, dynamicMin)}-${formatRangeValue(control, dynamicMax)}`
                                  : ' '}
                                onChange={(event) => {
                                  const nextValue = Number(event.target.value);
                                  if (Number.isNaN(nextValue)) {
                                    return;
                                  }

                                  updateSettingsDraft((current) => ({
                                    ...current,
                                    [control.section]: {
                                      ...current[control.section],
                                      [control.key]: clampControlValue(
                                        { ...control, min: dynamicMin, max: dynamicMax },
                                        nextValue
                                      ),
                                    },
                                  }));
                                }}
                              />
                            </div>
                          </div>
                          <Slider
                            value={clampControlValue({ ...control, min: dynamicMin, max: dynamicMax }, value)}
                            min={dynamicMin}
                            max={dynamicMax}
                            step={control.step}
                            disabled={trainingLocked}
                            marks={sliderMarks}
                            valueLabelDisplay="auto"
                            valueLabelFormat={(nextValue) => formatSettingValue(control, nextValue)}
                            onChange={(_event, nextValue) =>
                              updateSettingsDraft((current) => ({
                                ...current,
                                [control.section]: {
                                  ...current[control.section],
                                  [control.key]: clampControlValue(
                                    { ...control, min: dynamicMin, max: dynamicMax },
                                    nextValue
                                  ),
                                },
                              }))
                            }
                          />
                          {hasDynamicRange ? (
                            <Typography style={{marginLeft: '30px'}} variant="caption" className="muted-text settings-control__range-hint">
                              {`Полный диапазон: ${formatRangeValue(control, control.min)}-${formatRangeValue(control, control.max)}. Активный лимит сейчас: ${formatRangeValue(control, dynamicMin)}-${formatRangeValue(control, dynamicMax)}.`}
                            </Typography>
                          ) : null}
                          {control.key === 'batchSize' && snapshot.model.trainingItemCount > 0 ? (
                            <Typography variant="caption" className="muted-text">
                              {`При ${formatNumber(snapshot.model.trainingItemCount)} обучающих окнах это примерно ${formatNumber(
                                Math.max(
                                  Math.ceil(
                                    snapshot.model.trainingItemCount /
                                      Math.max(1, Number(value) || 1)
                                  ),
                                  1
                                )
                              )} батч(ей) на эпоху.`}
                            </Typography>
                          ) : null}
                          {control.key === 'batchSize' ? (
                            <Typography style={{marginLeft: '30px'}} variant="caption" className="muted-text">
                              {`Динамический лимит batch size для текущей архитектуры: ${formatNumber(modelConstraintReport.dynamicLimits.batchSizeMax)}.`}
                            </Typography>
                          ) : null}
                          {control.key === 'feedForwardSize' ? (
                            <Typography variant="caption" className="muted-text">
                              {`Динамический лимит feed-forward для текущей архитектуры: ${formatNumber(modelConstraintReport.dynamicLimits.feedForwardSizeMax)}.`}
                            </Typography>
                          ) : null}
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="settings-footer">
            <Button
              variant="contained"
              className="action-button"
              onClick={handleSaveSettingsClick}
              disabled={busy || trainingLocked || !hasUnsavedSettings}
            >
              Сохранить настройки сервера
            </Button>
          </div>
        </GlassPanel>

        <GlassPanel className="panel--fit panel--sources">
          <div className="panel-heading panel-heading--full">
            <div className="panel-heading__copy">
              <Typography variant="h3">Источники для обучения</Typography>
              <Typography variant="body2" className="muted-text">
                Тексты хранятся в SQLite, артефакты модели на диске.
              </Typography>
            </div>
          </div>

          <div className="upload-actions">
            <TextField
              className="text-input"
              label="URL страницы"
              value={urlInput}
              disabled={trainingLocked}
              onChange={(event) => setUrlInput(event.target.value)}
              placeholder="https://example.com/page"
            />

            <Button
              variant="contained"
              component="label"
              startIcon={<UploadFileRoundedIcon />}
              className="action-button"
              disabled={busy || trainingLocked}
            >
              Загрузить файлы
              <input
                hidden
                type="file"
                accept={SUPPORTED_SOURCE_FILE_ACCEPT}
                multiple
                onChange={async (event) => {
                  const files = Array.from(event.target.files || []);
                  event.target.value = '';
                  if (files.length) {
                    await onUploadFiles(files);
                  }
                }}
              />
            </Button>

            <Button
              variant="outlined"
              startIcon={<AddLinkRoundedIcon />}
              className="action-button"
              onClick={async () => {
                if (!urlInput.trim()) {
                  return;
                }

                await onAddUrlSource(urlInput.trim());
                setUrlInput('');
              }}
              disabled={busy || trainingLocked || !urlInput.trim()}
            >
              Добавить URL
            </Button>

            <Typography variant="caption" className="muted-text">
              Поддерживаемые форматы: TXT, CSV, JSON, JSONL/NDJSON, PARQUET.
            </Typography>

            {isUploadingFiles ? (
              <div className="upload-progress-wrap">
                <Typography variant="caption" className="muted-text">
                  {isServerProcessing
                    ? `Обработка данных на сервере: ${uploadIndicatorPercent.toFixed(2)}%`
                    : normalizedUploadProgress < 100
                      ? `Загрузка файла: ${normalizedUploadProgress.toFixed(2)}%`
                      : 'Файл загружен, запуск обработки данных...'}
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={uploadIndicatorPercent}
                  sx={{ mt: 0.5, borderRadius: 99 }}
                />
              </div>
            ) : null}
          </div>

          <div className="source-list">
            {snapshot.sources.map((source) => (
              <div className="source-card" key={source.id}>
                <div className="source-card__header">
                  <div>
                    <Typography variant="subtitle2">{source.label}</Typography>
                    <Typography variant="caption" className="muted-text">
                      {formatSourceStats(source)}
                    </Typography>
                  </div>
                  <IconButton onClick={() => onRemoveSource(source.id)} disabled={busy || trainingLocked}>
                    <DeleteOutlineRoundedIcon />
                  </IconButton>
                </div>
                <Typography variant="body2" className="muted-text">
                  {previewText(source.preview, 190)}
                </Typography>
              </div>
            ))}
          </div>

          <div className="training-queues-section">
            <div className="source-section__head">
              <Typography variant="subtitle2">Очереди автообучения</Typography>
              <Typography variant="caption" className="muted-text">
                {`${trainingQueues.length} очередей`}
              </Typography>
            </div>

            <div className="queue-builder">
              <TextField
                className="text-input"
                label="Название очереди"
                value={queueNameInput}
                disabled={trainingLocked}
                onChange={(event) => setQueueNameInput(event.target.value)}
                placeholder="Например, Диалоги недели 1"
              />
              <Button
                variant="outlined"
                className="action-button"
                onClick={async () => {
                  await onCreateTrainingQueue(queueNameInput.trim());
                  setQueueNameInput('');
                }}
                disabled={busy || trainingLocked}
              >
                Создать очередь
              </Button>
            </div>

            {isUploadingQueueFiles ? (
              <div className="upload-progress-wrap">
                <Typography variant="caption" className="muted-text">
                  {isServerProcessing
                    ? `Обработка файлов очереди: ${uploadIndicatorPercent.toFixed(2)}%`
                    : normalizedUploadProgress < 100
                      ? `Загрузка файлов очереди: ${normalizedUploadProgress.toFixed(2)}%`
                      : 'Файлы очереди загружены, запускается обработка...'}
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={uploadIndicatorPercent}
                  sx={{ mt: 0.5, borderRadius: 99 }}
                />
              </div>
            ) : null}

            <div className="training-queue-list">
              {trainingQueues.length ? trainingQueues.map((queue, index) => (
                <div className="training-queue-card" key={queue.id}>
                  <div className="training-queue-card__head">
                    <div>
                      <Typography variant="subtitle2">{`${index + 1}. ${queue.name}`}</Typography>
                      <Typography variant="caption" className="muted-text">
                        {`${queue.sourceCount} файлов`}
                      </Typography>
                    </div>
                    <div className="training-queue-card__actions">
                      <StatusPill
                        label={queue.status}
                        active={queue.status === 'running'}
                        tone={queue.status === 'completed' ? 'accent' : 'neutral'}
                      />
                      <IconButton
                        onClick={() => onDeleteTrainingQueue(queue.id)}
                        disabled={busy || trainingLocked}
                      >
                        <DeleteForeverRoundedIcon />
                      </IconButton>
                    </div>
                  </div>

                  {queue.lastError ? (
                    <Typography variant="caption" className="muted-text training-queue-card__error">
                      {queue.lastError}
                    </Typography>
                  ) : null}

                  <div className="training-queue-card__controls">
                    <Button
                      variant="contained"
                      component="label"
                      startIcon={<UploadFileRoundedIcon />}
                      className="action-button"
                      disabled={busy || trainingLocked}
                    >
                      Добавить файлы
                      <input
                        hidden
                        type="file"
                        accept={SUPPORTED_SOURCE_FILE_ACCEPT}
                        multiple
                        onChange={async (event) => {
                          const files = Array.from(event.target.files || []);
                          event.target.value = '';
                          if (files.length) {
                            await onUploadQueueFiles(queue.id, files);
                          }
                        }}
                      />
                    </Button>
                  </div>

                  <div className="training-queue-source-list">
                    {queue.sources.length ? queue.sources.map((source) => (
                      <div className="training-queue-source" key={source.id}>
                        <div>
                          <Typography variant="body2">{source.label}</Typography>
                          <Typography variant="caption" className="muted-text">
                            {formatQueueSourceStats(source)}
                          </Typography>
                        </div>
                        <IconButton
                          onClick={() => onRemoveTrainingQueueSource(queue.id, source.id)}
                          disabled={busy || trainingLocked}
                        >
                          <DeleteOutlineRoundedIcon />
                        </IconButton>
                      </div>
                    )) : (
                      <Typography variant="body2" className="muted-text">
                        В этой очереди пока нет файлов.
                      </Typography>
                    )}
                  </div>
                </div>
              )) : (
                <Typography variant="body2" className="muted-text">
                  Очереди пока не созданы.
                </Typography>
              )}
            </div>
          </div>
        </GlassPanel>

        <GlassPanel className="panel--fit panel--state">
          <div className="panel-heading panel-heading--full">
            <div className="panel-heading__copy">
              <Typography variant="h3">Состояние модели</Typography>
            </div>
          </div>

          <div className="state-scroll">

            <Alert severity={snapshot.model.lifecycle === 'error' ? 'error' : 'info'}>
              {snapshot.training.message}
            </Alert>
            {isTraining && trainingEtaSeconds !== null ? (
              <Alert severity="info" className="training-live-alert">
                <span className="training-live-alert__line">
                  {`Эпоха ${trainingEpochNow}/${trainingEpochTotal} | Батч ${trainingBatchNow}/${trainingBatchTotal} | Прогресс ${trainingPercentLabel}%`}
                </span>
                <span className="training-live-alert__line">
                  {`ETA ${formatDurationClock(trainingEtaSeconds)} | Окончание ${formatDateTime(trainingProgress.etaAt)} | Скорость ${trainingThroughputBps ? formatDecimal(trainingThroughputBps, 3) : '—'} батч/с | Шаг ~${trainingAvgBatchMs ? formatDecimal(trainingAvgBatchMs, 1) : '—'} мс`}
                </span>
              </Alert>
            ) : null}

            {serverStatus ? (
              <Alert severity="info">
                {`Node uptime: ${formatNumber(serverStatus.uptimeSec || 0)}s | Event-loop p95: ${formatDecimal(serverStatus.resources?.eventLoopLagMs?.p95, 2)}ms | Heavy queue: ${formatNumber(serverStatus.channels?.overload?.queuedCount || 0)} | Python active: ${formatNumber(serverStatus.pythonBridge?.activeProcesses || 0)}`}
              </Alert>
            ) : null}

            <div className="metric-grid">
              <MetricCard label="Параметры" value={formatNumber(snapshot.model.parameterCount)} hint="в модели" />
              <MetricCard label="Словарь" value={formatNumber(snapshot.model.vocabularySize)} hint="токенов" />
              <MetricCard label="Эпохи" value={formatNumber(snapshot.model.trainedEpochs)} hint="завершено" />
              <MetricCard label="Loss" value={formatDecimal(snapshot.model.lastLoss, 4)} hint="последний батч" />
              <MetricCard label="Val loss" value={formatDecimal(snapshot.model.validationLoss, 4)} hint="последняя проверка" />
              <MetricCard label="Оценки" value={formatNumber(snapshot.runtime?.ratedMessages || 0)} hint="ответов" />
            </div>

            <div className="status-flow">
              {STATUS_FLOW.map((status) => (
                <StatusPill
                  key={status}
                  label={status}
                  active={snapshot.model.lifecycle === status || snapshot.training.status === status}
                  tone={snapshot.model.lifecycle === status ? 'accent' : 'default'}
                />
              ))}
            </div>

            <div className="status-stream">
              <div className="status-stream__head">
                <Typography variant="subtitle2">Живые статусы</Typography>
                <StatusPill
                  label={realtimeConnected ? 'Realtime online' : 'Realtime offline'}
                  active={realtimeConnected}
                  tone={realtimeConnected ? 'accent' : 'neutral'}
                />
              </div>
              <div className="status-stream__list">
                {statusEntries.length ? (
                  statusEntries.slice(0, 10).map((entry) => (
                    <div className="status-stream__card" key={entry.id}>
                      <div className="status-stream__card-head">
                        <StatusPill label={entry.status} active tone="accent" />
                        <Typography variant="caption" className="muted-text">
                          {formatDateTime(entry.createdAt)}
                        </Typography>
                      </div>
                      <Typography variant="subtitle2">{entry.phase}</Typography>
                      <Typography variant="body2" className="muted-text">
                        {entry.message}
                      </Typography>
                    </div>
                  ))
                ) : (
                  <Typography variant="body2" className="muted-text">
                    Статусы пока не поступали.
                  </Typography>
                )}
              </div>
            </div>

            <div className="model-facts">
              {factRows.map(([label, value]) => (
                <div className="fact-row" key={label}>
                  <span className="fact-row__label">{label}</span>
                  <span className="fact-row__value">{value}</span>
                </div>
              ))}
            </div>

            <div className="server-log-panel">
              <div className="server-log-panel__head">
                <Typography variant="subtitle2">Журнал backend</Typography>
                <Typography variant="caption" className="muted-text">
                  {serverLogs.length} записей
                </Typography>
              </div>
              <div className="server-log-list">
                {serverLogs.length ? (
                  serverLogs.map((entry) => (
                    <div className={`server-log-entry server-log-entry--${entry.level || 'info'}`} key={entry.id}>
                      <div className="server-log-entry__head">
                        <span className="server-log-entry__level">{entry.level || 'info'}</span>
                        <Typography variant="caption" className="muted-text">
                          {formatDateTime(entry.createdAt)}
                        </Typography>
                      </div>
                      <Typography variant="body2">{entry.message}</Typography>
                      {entry.details?.path ? (
                        <Typography variant="caption" className="muted-text">
                          {`${entry.details.method || 'GET'} ${entry.details.path}`}
                        </Typography>
                      ) : null}
                    </div>
                  ))
                ) : (
                  <Typography variant="body2" className="muted-text">
                    Журнал сервера пока пуст.
                  </Typography>
                )}
              </div>
            </div>

            {snapshot.model.artifactFiles?.length ? (
              <div className="top-terms">
                <Typography variant="subtitle2">Файлы модели и хранилища</Typography>
                <div className="artifact-file-list">
                  {snapshot.model.artifactFiles.map((filePath) => (
                    <Typography
                      key={filePath}
                      variant="caption"
                      className="artifact-file-path"
                      title={filePath}
                    >
                      {filePath}
                    </Typography>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        </GlassPanel>
      </div>

      <div className="training-bottom-grid">
        <GlassPanel className="panel--fit panel--chart">
          <div className="panel-heading panel-heading--full">
            <div className="panel-heading__copy">
              <Typography variant="h3">График обучения</Typography>
              <Typography variant="body2" className="muted-text">
                Динамика `loss` по батчам.
              </Typography>
            </div>
          </div>
          <TrainingChart history={snapshot.training.history} />
        </GlassPanel>
      </div>
    </div>
  );
}
