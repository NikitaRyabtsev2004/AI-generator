import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Button,
  IconButton,
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
import DeleteForeverRoundedIcon from '@mui/icons-material/DeleteForeverRounded';
import GlassPanel from '../shared/GlassPanel';
import MetricCard from '../shared/MetricCard';
import StatusPill from '../shared/StatusPill';
import TrainingChart from './TrainingChart';
import { SETTING_GROUPS, STATUS_FLOW } from '../../constants/modelConfig';
import {
  formatDateTime,
  formatDecimal,
  formatNumber,
  previewText,
  valueWithUnit,
} from '../../utils/text';
import '../../styles/training-tab.css';

function formatSettingValue(control, value) {
  if (control.precision) {
    return valueWithUnit(value, control.unit, control.precision);
  }

  return valueWithUnit(value, control.unit);
}

function modelFactRows(snapshot) {
  const { model, runtime } = snapshot;

  return [
    ['Движок', model.engine],
    ['Жизненный цикл', model.lifecycle],
    ['Статус', model.status],
    ['Источников', formatNumber(model.sourceCount)],
    ['Чатов', formatNumber(model.chatCount)],
    ['Диалоговых пар', formatNumber(model.replyPairCount)],
    ['Knowledge-чанков', formatNumber(model.chunkCount)],
    ['Sequence-примеров', formatNumber(model.trainingItemCount)],
    ['Батчей на эпоху', formatNumber(model.batchesPerEpoch)],
    ['Токенов корпуса', formatNumber(model.tokenCount)],
    ['Размер словаря', formatNumber(model.vocabularySize)],
    ['Параметров модели', formatNumber(model.parameterCount)],
    ['Обучено эпох', formatNumber(model.trainedEpochs)],
    ['Последний loss', formatDecimal(model.lastLoss, 4)],
    ['Средний loss', formatDecimal(model.averageLoss, 4)],
    ['Perplexity', formatDecimal(model.perplexity, 4)],
    ['Средний self-score', formatDecimal(model.averageSelfScore, 3)],
    ['Положительных оценок', formatNumber(model.positiveFeedbackCount)],
    ['Отрицательных оценок', formatNumber(model.negativeFeedbackCount)],
    ['Последнее обучение', formatDateTime(model.lastTrainingAt)],
    ['Последняя генерация', formatDateTime(model.lastGenerationAt)],
    ['Стратегия контекста', runtime?.contextStrategy || 'n/a'],
    ['Генератор', runtime?.generatorBackend || 'neural'],
  ];
}

export default function TrainingTab({
  snapshot,
  busy,
  error,
  onSaveSettings,
  onUploadFiles,
  onAddUrlSource,
  onRemoveSource,
  onCreateModel,
  onTrainModel,
  onPauseModel,
  onResetModel,
}) {
  const snapshotSettingsKey = useMemo(() => JSON.stringify(snapshot.settings), [snapshot.settings]);
  const [settingsDraft, setSettingsDraft] = useState(() => structuredClone(snapshot.settings));
  const [urlInput, setUrlInput] = useState('');
  const lastSnapshotKeyRef = useRef(snapshotSettingsKey);
  const currentDraftKeyRef = useRef(JSON.stringify(snapshot.settings));

  useEffect(() => {
    currentDraftKeyRef.current = JSON.stringify(settingsDraft);
  }, [settingsDraft]);

  useEffect(() => {
    const previousSnapshotKey = lastSnapshotKeyRef.current;
    const currentDraftKey = currentDraftKeyRef.current;

    if (currentDraftKey === previousSnapshotKey || currentDraftKey === snapshotSettingsKey) {
      setSettingsDraft(structuredClone(snapshot.settings));
      currentDraftKeyRef.current = snapshotSettingsKey;
    }

    lastSnapshotKeyRef.current = snapshotSettingsKey;
  }, [snapshot.settings, snapshotSettingsKey]);

  const factRows = useMemo(() => modelFactRows(snapshot), [snapshot]);
  const hasUnsavedSettings = useMemo(
    () => JSON.stringify(settingsDraft) !== snapshotSettingsKey,
    [settingsDraft, snapshotSettingsKey]
  );
  const canTrain = snapshot.sources.length > 0;
  const trainingLocked = snapshot.training.status === 'training' || snapshot.model.status === 'saving_checkpoint';

  return (
    <div className="training-tab">
      <div className="training-grid">
        <GlassPanel className="panel--wide">
          <div className="panel-heading">
            <div>
              <Typography variant="h3">Настройки модели и обучения</Typography>
              <Typography variant="body2" className="muted-text">
                Здесь настраивается уже не псевдо-обучение, а реальная серверная sequence-модель:
                архитектура, training loop, retrieval-контекст и генерация ответа.
              </Typography>
            </div>
            <div className="panel-actions">
              <Button
                className="action-button"
                variant="outlined"
                startIcon={<CachedRoundedIcon />}
                onClick={() => onCreateModel()}
                disabled={busy || trainingLocked}
              >
                Новая модель
              </Button>
              <Button
                className="action-button"
                variant="contained"
                startIcon={<PlayArrowRoundedIcon />}
                onClick={() => onTrainModel()}
                disabled={busy || trainingLocked || !canTrain}
              >
                Запустить обучение
              </Button>
              <Button
                className="action-button"
                variant="outlined"
                startIcon={<PauseRoundedIcon />}
                onClick={() => onPauseModel()}
                disabled={snapshot.training.status !== 'training'}
              >
                Пауза
              </Button>
              <Button
                className="action-button"
                color="error"
                variant="outlined"
                startIcon={<DeleteForeverRoundedIcon />}
                onClick={() => onResetModel()}
                disabled={busy || trainingLocked}
              >
                Сбросить модель
              </Button>
            </div>
          </div>

          {error ? <Alert severity="error">{error}</Alert> : null}
          {trainingLocked ? (
            <Alert severity="info">
              Во время обучения редактирование корпуса и параметров заблокировано, чтобы чекпоинт не устарел прямо в процессе.
            </Alert>
          ) : null}
          {hasUnsavedSettings ? (
            <Alert severity="warning">
              Есть несохраненные изменения. Они не применятся к серверу, пока вы явно не нажмете сохранение.
            </Alert>
          ) : null}
          {!canTrain ? (
            <Alert severity="info">
              Добавьте хотя бы один `txt`-файл или URL, чтобы у модели был корпус для обучения.
            </Alert>
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
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
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

        <GlassPanel>
          <div className="panel-heading">
            <div>
              <Typography variant="h3">Источники для обучения</Typography>
              <Typography variant="body2" className="muted-text">
                Тексты лежат в SQLite, а чекпоинт модели, токенизатор и retrieval-артефакты
                сохраняются на диск отдельными файлами.
              </Typography>
            </div>
          </div>

          <div className="upload-actions">
            <Button
              variant="contained"
              component="label"
              startIcon={<UploadFileRoundedIcon />}
              className="action-button"
              disabled={busy || trainingLocked}
            >
              Загрузить txt
              <input
                hidden
                type="file"
                accept=".txt,text/plain"
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

            <div className="url-add-form">
              <TextField
                className="text-input"
                label="URL страницы"
                value={urlInput}
                disabled={trainingLocked}
                onChange={(event) => setUrlInput(event.target.value)}
                placeholder="https://example.com/page"
              />
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
            </div>
          </div>

          <div className="source-list">
            {snapshot.sources.map((source) => (
              <div className="source-card" key={source.id}>
                <div className="source-card__header">
                  <div>
                    <Typography variant="subtitle2">{source.label}</Typography>
                    <Typography variant="caption" className="muted-text">
                      {source.type.toUpperCase()} | {formatNumber(source.stats.tokenCount)} токенов |{' '}
                      {formatNumber(source.stats.charCount)} символов
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
        </GlassPanel>

        <GlassPanel>
          <div className="panel-heading">
            <div>
              <Typography variant="h3">Состояние модели</Typography>
              <Typography variant="body2" className="muted-text">
                Метрики ниже показывают реальный размер sequence-модели, обратную связь от пользователей
                и текущий статус обучения на сервере.
              </Typography>
            </div>
          </div>

          <div className="metric-grid">
            <MetricCard label="Параметры" value={formatNumber(snapshot.model.parameterCount)} hint="в модели" />
            <MetricCard label="Словарь" value={formatNumber(snapshot.model.vocabularySize)} hint="токенов" />
            <MetricCard label="Эпохи" value={formatNumber(snapshot.model.trainedEpochs)} hint="завершено" />
            <MetricCard label="Loss" value={formatDecimal(snapshot.model.lastLoss, 4)} hint="последний батч" />
            <MetricCard label="Perplexity" value={formatDecimal(snapshot.model.perplexity, 4)} hint="по модели" />
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

          <Alert severity={snapshot.model.lifecycle === 'error' ? 'error' : 'info'}>
            {snapshot.training.message}
          </Alert>

          <div className="model-facts">
            {factRows.map(([label, value]) => (
              <div className="fact-row" key={label}>
                <span className="fact-row__label">{label}</span>
                <span className="fact-row__value">{value}</span>
              </div>
            ))}
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
        </GlassPanel>
      </div>

      <div className="training-bottom-grid">
        <GlassPanel className="panel--wide">
          <div className="panel-heading">
            <div>
              <Typography variant="h3">График обучения</Typography>
              <Typography variant="body2" className="muted-text">
                Лента batch-loss обновляется во время реального server-side training loop и помогает
                понять, идет ли модель к стабильному обучению.
              </Typography>
            </div>
          </div>
          <TrainingChart history={snapshot.training.history} />
        </GlassPanel>

        <GlassPanel>
          <div className="panel-heading">
            <div>
              <Typography variant="h3">Лента статусов</Typography>
              <Typography variant="body2" className="muted-text">
                Здесь видно, на какой стадии находится модель: подготовка корпуса, fit, сохранение
                чекпоинта или обучение от обратной связи.
              </Typography>
            </div>
          </div>

          <div className="timeline-list">
            {snapshot.training.recentStatuses?.map((entry) => (
              <div className="timeline-item" key={entry.id}>
                <div className="timeline-item__head">
                  <StatusPill label={entry.status} active tone="accent" />
                  <span className="timeline-item__time">{formatDateTime(entry.createdAt)}</span>
                </div>
                <Typography variant="subtitle2">{entry.phase}</Typography>
                <Typography variant="body2" className="muted-text">
                  {entry.message}
                </Typography>
              </div>
            ))}
          </div>
        </GlassPanel>
      </div>
    </div>
  );
}
