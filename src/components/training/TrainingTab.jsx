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

function formatQueueSourceStats(source = {}) {
  const typeLabel = String(source.type || 'file').toUpperCase();
  const tokenCount = Math.max(Number(source.stats?.tokenCount) || 0, 0);
  const charCount = Math.max(Number(source.stats?.charCount) || 0, 0);
  const contentSize = Math.max(Number(source.contentSize) || 0, 0);

  if (tokenCount > 0 || charCount > 0) {
    return `${typeLabel} | ${formatNumber(tokenCount)} токенов | ${formatNumber(charCount)} символов`;
  }

  if (contentSize > 0) {
    return `${typeLabel} | ${formatBytes(contentSize)} | токены будут подсчитаны при обучении`;
  }

  return `${typeLabel} | токены будут подсчитаны при обучении`;
}

export default function TrainingTab({
  snapshot,
  busy,
  error,
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
  const snapshotSettingsKey = useMemo(() => JSON.stringify(snapshot.settings), [snapshot.settings]);
  const runtimeConfig = useMemo(() => snapshot.runtime?.config || {}, [snapshot.runtime?.config]);
  const runtimeSnapshotKey = useMemo(() => JSON.stringify(runtimeConfig), [runtimeConfig]);
  const [settingsDraft, setSettingsDraft] = useState(() => structuredClone(snapshot.settings));
  const [runtimeDraft, setRuntimeDraft] = useState(() => structuredClone(runtimeConfig));
  const [urlInput, setUrlInput] = useState('');
  const [queueNameInput, setQueueNameInput] = useState('');
  const [modelNameInput, setModelNameInput] = useState('');
  const lastSnapshotKeyRef = useRef(snapshotSettingsKey);
  const currentDraftKeyRef = useRef(JSON.stringify(snapshot.settings));
  const lastRuntimeSnapshotKeyRef = useRef(runtimeSnapshotKey);
  const currentRuntimeDraftKeyRef = useRef(JSON.stringify(runtimeConfig));

  useEffect(() => {
    currentDraftKeyRef.current = JSON.stringify(settingsDraft);
  }, [settingsDraft]);

  useEffect(() => {
    currentRuntimeDraftKeyRef.current = JSON.stringify(runtimeDraft);
  }, [runtimeDraft]);

  useEffect(() => {
    const previousSnapshotKey = lastSnapshotKeyRef.current;
    const currentDraftKey = currentDraftKeyRef.current;

    if (currentDraftKey === previousSnapshotKey || currentDraftKey === snapshotSettingsKey) {
      setSettingsDraft(structuredClone(snapshot.settings));
      currentDraftKeyRef.current = snapshotSettingsKey;
    }

    lastSnapshotKeyRef.current = snapshotSettingsKey;
  }, [snapshot.settings, snapshotSettingsKey]);

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
  const hasUnsavedSettings = useMemo(
    () => JSON.stringify(settingsDraft) !== snapshotSettingsKey,
    [settingsDraft, snapshotSettingsKey]
  );
  const hasUnsavedRuntimeSettings = useMemo(
    () => JSON.stringify(runtimeDraft) !== runtimeSnapshotKey,
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

  const handleTrainClick = async () => {
    if (hasUnsavedSettings) {
      const savedSnapshot = await onSaveSettings(settingsDraft);
      if (!savedSnapshot) {
        return;
      }
    }

    if (hasUnsavedRuntimeSettings) {
      const savedSnapshot = await onSaveRuntimeConfig(runtimeDraft);
      if (!savedSnapshot) {
        return;
      }
    }

    await onTrainModel();
  };

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

          {error ? <Alert severity="error">{error}</Alert> : null}
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
                  onClick={() => onSaveRuntimeConfig(runtimeDraft)}
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
                          setSettingsDraft((current) => ({
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
                      return (
                        <div className="settings-control" key={control.key}>
                          <div className="settings-control__header">
                            <div className="settings-control__label-wrap">
                              <Typography variant="subtitle2">{control.label}</Typography>
                              <Tooltip title={control.hint}>
                                <InfoOutlinedIcon className="info-icon" />
                              </Tooltip>
                            </div>
                            <span className="settings-control__value">
                              {formatSettingValue(control, value)}
                            </span>
                          </div>
                          <Slider
                            value={value}
                            min={control.min}
                            max={control.max}
                            step={control.step}
                            disabled={trainingLocked}
                            valueLabelDisplay="auto"
                            valueLabelFormat={(nextValue) => formatSettingValue(control, nextValue)}
                            onChange={(_event, nextValue) =>
                              setSettingsDraft((current) => ({
                                ...current,
                                [control.section]: {
                                  ...current[control.section],
                                  [control.key]: nextValue,
                                },
                              }))
                            }
                          />
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
              onClick={() => onSaveSettings(settingsDraft)}
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
                    ? `Обработка данных на сервере: ${uploadIndicatorPercent.toFixed(1)}%`
                    : normalizedUploadProgress < 100
                      ? `Загрузка файла: ${normalizedUploadProgress.toFixed(1)}%`
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
                      {source.type.toUpperCase()} | {formatNumber(source.stats.tokenCount)} токенов | {formatNumber(source.stats.charCount)} символов
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
                    ? `Обработка файлов очереди: ${uploadIndicatorPercent.toFixed(1)}%`
                    : normalizedUploadProgress < 100
                      ? `Загрузка файлов очереди: ${normalizedUploadProgress.toFixed(1)}%`
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
