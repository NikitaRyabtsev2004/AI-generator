import { useMemo, useRef, useState } from 'react';
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
  const importInputRef = useRef(null);
  const studio = useStudioApp();

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

  const backgroundImg = 'b-1.gif'

  if (studio.loading || !studio.snapshot) {
    return (
      <Box
        className="app-shell app-shell--loading"
        style={{ '--app-bg-image': `url(${process.env.PUBLIC_URL}/${backgroundImg})` }}
      >
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
      className="app-shell"
      style={{ '--app-bg-image': `url(${process.env.PUBLIC_URL}/${backgroundImg})` }}
    >
      <Container maxWidth="xl" className="app-shell__container">
        <GlassPanel className="hero-panel">
          <div className="hero-panel__topline">
            <div className="hero-panel__title-wrap">
              <div className="hero-panel__badge">
                <AutoAwesomeIcon fontSize="medium" />
                <span className="hero-panel__badge-text">
                  <span>
                    <strong style={{fontSize: '30px'}}>AI Generator</strong>
                  </span>
                  <strong style={{textDecoration: 'underline'}}>Studio</strong>
                </span>
              </div>
            </div>

            <div className="hero-panel__status-wrap">
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
              <StatusPill label={studio.snapshot.model.engine} active tone="accent" />
              <StatusPill label={`Статус: ${studio.snapshot.model.lifecycle}`} tone="neutral" />
              {summaryItems.map((item) => (
                <StatusPill key={item} label={item} tone="neutral" />
              ))}
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

        {studio.error ? (
          <Alert severity="error" className="top-alert">
            {studio.error}
          </Alert>
        ) : null}

        <div className={`app-shell__content ${tab === 0 ? 'app-shell__content--training' : 'app-shell__content--chat'}`}>
          {tab === 0 ? (
            <TrainingTab
              snapshot={studio.snapshot}
              busy={studio.busy}
              error={studio.error}
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
