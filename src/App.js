import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Badge,
  Button,
  Box,
  CircularProgress,
  Container,
  Dialog,
  DialogContent,
  IconButton,
  Tab,
  Tabs,
  Tooltip,
  Typography,
} from '@mui/material';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import ForumIcon from '@mui/icons-material/Forum';
import MemoryIcon from '@mui/icons-material/Memory';
import HubRoundedIcon from '@mui/icons-material/HubRounded';
import HistoryRoundedIcon from '@mui/icons-material/HistoryRounded';
import FileDownloadRoundedIcon from '@mui/icons-material/FileDownloadRounded';
import FileUploadRoundedIcon from '@mui/icons-material/FileUploadRounded';
import ChatTab from './components/chat/ChatTab';
import GlassPanel from './components/shared/GlassPanel';
import StatusPill from './components/shared/StatusPill';
import TrainingTab from './components/training/TrainingTab';
import { useStudioApp } from './hooks/useStudioApp';
import { formatNumber } from './utils/text';
import './styles/global.css';
import './styles/app-shell.css';

function resolveStatusTheme(snapshot) {
  if (!snapshot) {
    return 'not_created';
  }

  const trainingStatus = String(snapshot.training?.status || '').toLowerCase();
  if (['training', 'paused', 'error'].includes(trainingStatus)) {
    return trainingStatus;
  }

  const lifecycle = String(snapshot.model?.lifecycle || '').toLowerCase();
  if (['training', 'paused', 'generating_reply', 'syncing_knowledge', 'learning_from_feedback', 'error'].includes(lifecycle)) {
    return lifecycle;
  }

  if (!snapshot.model?.exists) {
    return 'not_created';
  }

  if (lifecycle === 'ready_for_training' || !Number(snapshot.model?.trainedEpochs || 0)) {
    return 'ready_for_training';
  }

  return 'trained';
}

function buildProcessBanner(snapshot) {
  if (!snapshot) {
    return null;
  }

  const trainingStatus = String(snapshot.training?.status || '').toLowerCase();
  const lifecycle = String(snapshot.model?.lifecycle || '').toLowerCase();
  const message = snapshot.training?.message || '';

  if (trainingStatus === 'training' || lifecycle === 'training') {
    return {
      id: 'process-training',
      tone: 'training',
      title: 'Идет обучение модели',
      message: message || 'Процесс обучения запущен в воркере.',
    };
  }

  if (lifecycle === 'generating_reply') {
    return {
      id: 'process-generating-reply',
      tone: 'generating_reply',
      title: 'Генерация ответа',
      message: message || 'Сервер формирует ответ по текущему контексту.',
    };
  }

  if (lifecycle === 'syncing_knowledge') {
    return {
      id: 'process-syncing',
      tone: 'syncing_knowledge',
      title: 'Синхронизация знаний',
      message: message || 'Индекс и артефакты знаний обновляются.',
    };
  }

  if (lifecycle === 'learning_from_feedback') {
    return {
      id: 'process-feedback',
      tone: 'learning_from_feedback',
      title: 'Анализ обратной связи',
      message: message || 'Система обновляет внутренние сигналы качества.',
    };
  }

  return null;
}

function LoadingState() {
  return (
    <div className="loading-state">
      <CircularProgress />
      <Typography variant="body1">Сервер и клиент синхронизируются...</Typography>
    </div>
  );
}

function App() {
  const [tab, setTab] = useState(0);
  const [statusDialogOpen, setStatusDialogOpen] = useState(false);
  const [baseStatusTheme, setBaseStatusTheme] = useState('not_created');
  const [overlayStatusTheme, setOverlayStatusTheme] = useState('not_created');
  const [overlayActive, setOverlayActive] = useState(false);
  const [trainingNotices, setTrainingNotices] = useState([]);
  const [chatNotices, setChatNotices] = useState([]);
  const importInputRef = useRef(null);
  const studio = useStudioApp();
  const resolvedStatusTheme = useMemo(
    () => resolveStatusTheme(studio.snapshot),
    [studio.snapshot]
  );

  useEffect(() => {
    if (resolvedStatusTheme === baseStatusTheme) {
      setOverlayStatusTheme(resolvedStatusTheme);
      setOverlayActive(false);
      return;
    }

    setOverlayStatusTheme(resolvedStatusTheme);
    setOverlayActive(false);

    const revealTimeoutId = setTimeout(() => {
      setOverlayActive(true);
    }, 24);
    const settleTimeoutId = setTimeout(() => {
      setBaseStatusTheme(resolvedStatusTheme);
      setOverlayStatusTheme(resolvedStatusTheme);
      setOverlayActive(false);
    }, 1200);

    return () => {
      clearTimeout(revealTimeoutId);
      clearTimeout(settleTimeoutId);
    };
  }, [baseStatusTheme, resolvedStatusTheme]);

  const summaryItems = useMemo(() => {
    if (!studio.snapshot) {
      return [];
    }

    return [
      `${studio.snapshot.sources.length} источников`,
      `${formatNumber(studio.snapshot.model.chunkCount)} фрагментов`,
      `${formatNumber(studio.snapshot.model.trainedEpochs)} эпох`,
      `${studio.snapshot.chats.length} чатов`,
    ];
  }, [studio.snapshot]);

  const processBanner = useMemo(
    () => buildProcessBanner(studio.snapshot),
    [studio.snapshot]
  );

  const toastNotices = useMemo(() => {
    const scopedNotices = tab === 0 ? trainingNotices : chatNotices;
    const notices = [];

    if (studio.error) {
      notices.push({
        id: 'global-error',
        severity: 'error',
        message: studio.error,
      });
    }

    scopedNotices.forEach((notice) => {
      if (!notice?.id || !notice?.message) {
        return;
      }
      notices.push(notice);
    });

    const deduped = [];
    const seen = new Set();
    notices.forEach((notice) => {
      if (seen.has(notice.id)) {
        return;
      }
      seen.add(notice.id);
      deduped.push(notice);
    });

    const priority = { error: 0, warning: 1, success: 2, info: 3 };
    return deduped
      .sort((left, right) => (priority[left.severity] ?? 10) - (priority[right.severity] ?? 10))
      .slice(0, 5);
  }, [chatNotices, studio.error, tab, trainingNotices]);

  const backgroundImg = 'b-4.gif'

  if (studio.loading || !studio.snapshot) {
    return (
      <Box
        className={`app-shell app-shell--loading app-shell--status-${resolvedStatusTheme}`}
        style={{ '--app-bg-image': `url(${process.env.PUBLIC_URL}/${backgroundImg})` }}
      >
        <div
          aria-hidden="true"
          className={`app-shell__status-layer app-shell__status-layer--current app-shell__status-layer--${baseStatusTheme}`}
        />
        <div
          aria-hidden="true"
          className={`app-shell__status-layer app-shell__status-layer--overlay app-shell__status-layer--${overlayStatusTheme} ${overlayActive ? 'app-shell__status-layer--active' : ''}`.trim()}
        />
        <LoadingState />
      </Box>
    );
  }

  const statusEntries = studio.snapshot.training?.recentStatuses || [];
  const statusCount = statusEntries.length;
  const modelActionLocked =
    studio.busy ||
    studio.snapshot.training.status === 'training' ||
    studio.snapshot.model.status === 'saving_checkpoint';
  const isExportingModel = studio.pendingAction === 'exportModel';
  const isImportingModel = studio.pendingAction === 'importModel';

  return (
    <Box
      className={`app-shell app-shell--status-${resolvedStatusTheme}`}
      style={{ '--app-bg-image': `url(${process.env.PUBLIC_URL}/${backgroundImg})` }}
    >
      <div
        aria-hidden="true"
        className={`app-shell__status-layer app-shell__status-layer--current app-shell__status-layer--${baseStatusTheme}`}
      />
      <div
        aria-hidden="true"
        className={`app-shell__status-layer app-shell__status-layer--overlay app-shell__status-layer--${overlayStatusTheme} ${overlayActive ? 'app-shell__status-layer--active' : ''}`.trim()}
      />
      {processBanner ? (
        <div className={`app-process-banner app-process-banner--${processBanner.tone}`}>
          <span className="app-process-banner__title">{processBanner.title}</span>
          <span className="app-process-banner__message">{processBanner.message}</span>
        </div>
      ) : null}
      <div className="app-toast-layer" aria-live="polite" role="status">
        {toastNotices.map((notice) => (
          <Alert
            key={notice.id}
            severity={notice.severity || 'info'}
            className="app-toast"
            variant="filled"
          >
            {notice.message}
          </Alert>
        ))}
      </div>
      <Container maxWidth={false} className="app-shell__container">
        <GlassPanel className="hero-panel">
          <div className="hero-panel__topline">
            <div className="hero-panel__title-wrap">
              <div className="hero-panel__badge">
                <AutoAwesomeIcon fontSize="medium" />
                <span className="hero-panel__badge-text">
                    <strong className="hero-panel__brand">AI Generator</strong>
                  <strong className="hero-panel__brand-accent">Studio</strong>
                </span>
              </div>
            </div>

            <div className="hero-panel__status-wrap">
              <div style={{display: 'flex', flexDirection: 'row', gap: '0.5em'}}>
                <Button
                    variant="contained"
                    className="hero-model-button"
                    startIcon={isExportingModel ? <CircularProgress size={16} color="inherit" /> : <FileDownloadRoundedIcon fontSize="small" />}
                    onClick={() => studio.actions.exportModel()}
                    disabled={modelActionLocked}
                >
                  Экспорт модели
                </Button>
                <Button
                    variant="contained"
                    className="hero-model-button hero-model-button--import"
                    startIcon={isImportingModel ? <CircularProgress size={16} color="inherit" /> : <FileUploadRoundedIcon fontSize="small" />}
                    onClick={() => importInputRef.current?.click()}
                    disabled={modelActionLocked}
                >
                  Импорт модели
                </Button>
                <input
                    ref={importInputRef}
                    type="file"
                    accept=".json,.aistudio,.aistudio.json,application/json"
                    style={{ display: 'none' }}
                    onChange={(event) => {
                      const selectedFile = event.target.files?.[0] || null;
                      if (selectedFile) {
                        studio.actions.importModel(selectedFile);
                      }
                      event.target.value = '';
                    }}
                />
                <Tooltip title={`Статусы обучения: ${statusCount}`}>
                  <IconButton
                      className="hero-status-button"
                      onClick={() => setStatusDialogOpen(true)}
                      size="small"
                  >
                    <Badge badgeContent={statusCount} color="default" max={999}>
                      <HistoryRoundedIcon fontSize="small" />
                    </Badge>
                  </IconButton>
                </Tooltip>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.5em' }}>
                <StatusPill label={studio.realtimeConnected ? 'Realtime online' : 'Realtime reconnecting'} tone="neutral" />
                <StatusPill label={studio.snapshot.model.engine} active tone="accent" />
                <StatusPill label={`Статус: ${studio.snapshot.model.lifecycle}`} tone="neutral" />
                {summaryItems.map((item) => (
                    <StatusPill key={item} label={item} tone="neutral" />
                ))}
              </div>
            </div>
          </div>

          <div className="hero-panel__tabs">
            <Tabs value={tab} onChange={(_event, nextTab) => setTab(nextTab)}>
              <Tab icon={<MemoryIcon />} iconPosition="start" label="Обучение модели" />
              <Tab icon={<ForumIcon />} iconPosition="start" label="Чаты и память" />
              <Tab icon={<HubRoundedIcon />} iconPosition="start" label="О проекте" disabled />
            </Tabs>
          </div>
        </GlassPanel>

        <div className={`app-shell__content ${tab === 0 ? 'app-shell__content--training' : 'app-shell__content--chat'}`}>
          {tab === 0 ? (
            <TrainingTab
              snapshot={studio.snapshot}
              busy={studio.busy}
              error={studio.error}
              onNoticesChange={setTrainingNotices}
              pendingAction={studio.pendingAction}
              realtimeConnected={studio.realtimeConnected}
              serverLogs={studio.serverLogs}
              serverStatus={studio.serverStatus}
              uploadProgress={studio.uploadProgress}
              processingProgress={studio.processingProgress}
              onSaveSettings={studio.actions.saveSettings}
              onSaveRuntimeConfig={studio.actions.saveRuntimeConfig}
              onUploadFiles={studio.actions.uploadFiles}
              onCreateTrainingQueue={studio.actions.createTrainingQueue}
              onUploadQueueFiles={studio.actions.uploadQueueFiles}
              onRemoveTrainingQueueSource={studio.actions.removeTrainingQueueSource}
              onDeleteTrainingQueue={studio.actions.deleteTrainingQueue}
              onAddUrlSource={studio.actions.addUrlSource}
              onRemoveSource={studio.actions.removeSource}
              onCreateModel={studio.actions.createModel}
              onCreateNamedModel={studio.actions.createNamedModel}
              onSelectModel={studio.actions.selectModel}
              onDeleteLibraryModel={studio.actions.deleteLibraryModel}
              onTrainModel={studio.actions.trainModel}
              onPauseModel={studio.actions.pauseModel}
              onRollbackModel={studio.actions.rollbackTraining}
              onResetModel={studio.actions.resetModel}
            />
          ) : (
            <ChatTab
              snapshot={studio.snapshot}
              selectedChatId={studio.selectedChatId}
              setSelectedChatId={studio.setSelectedChatId}
              busy={studio.busy}
              pendingAction={studio.pendingAction}
              pendingReply={studio.pendingReply}
              pendingRatings={studio.pendingRatings}
              error={studio.error}
              onNoticesChange={setChatNotices}
              onCreateChat={studio.actions.createChat}
              onDeleteChat={studio.actions.deleteChat}
              onSendMessage={studio.actions.sendMessage}
              onRateMessage={studio.actions.rateMessage}
            />
          )}
        </div>

        <Dialog
          open={statusDialogOpen}
          onClose={() => setStatusDialogOpen(false)}
          PaperProps={{ className: 'status-dialog-paper' }}
        >
          <DialogContent className="status-dialog-content">
            <GlassPanel className="status-glass-panel" innerClassName="status-glass-inner">
              <div className="status-glass-head">
                <Typography variant="h3">Статусы обучения</Typography>
                <StatusPill label={`${statusCount}`} active tone="accent" />
              </div>

              <div className="status-glass-list">
                {statusEntries.length ? (
                  statusEntries.map((entry) => (
                    <div key={entry.id} className="status-glass-card">
                      <div className="status-glass-card__head">
                        <StatusPill label={entry.status} active tone="accent" />
                        <span className="status-dialog-time">{new Date(entry.createdAt).toLocaleString()}</span>
                      </div>
                      <Typography variant="subtitle2">{entry.phase}</Typography>
                      <Typography variant="body2" className="muted-text">
                        {entry.message}
                      </Typography>
                    </div>
                  ))
                ) : (
                  <Typography variant="body2" className="muted-text">
                    Пока нет записей статусов.
                  </Typography>
                )}
              </div>
            </GlassPanel>
          </DialogContent>
        </Dialog>
      </Container>
    </Box>
  );
}

export default App;
