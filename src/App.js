import { useMemo, useState } from 'react';
import {
  Alert,
  Box,
  CircularProgress,
  Container,
  Tab,
  Tabs,
  Typography,
} from '@mui/material';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import ForumIcon from '@mui/icons-material/Forum';
import MemoryIcon from '@mui/icons-material/Memory';
import HubRoundedIcon from '@mui/icons-material/HubRounded';
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

  if (studio.loading || !studio.snapshot) {
    return (
      <Box
        className="app-shell app-shell--loading"
        style={{ '--app-bg-image': `url(${process.env.PUBLIC_URL}/b-1.gif)` }}
      >
        <LoadingState />
      </Box>
    );
  }

  return (
    <Box
      className="app-shell"
      style={{ '--app-bg-image': `url(${process.env.PUBLIC_URL}/b-1.gif)` }}
    >
      <Container maxWidth="xl" className="app-shell__container">
        <GlassPanel className="hero-panel">
          <div className="hero-panel__topline">
            <div className="hero-panel__title-wrap">
              <div className="hero-panel__badge">
                <AutoAwesomeIcon fontSize="small" />
                <span>Liquid Glass AI Studio</span>
              </div>
              <Typography variant="h2" className="hero-panel__title">
                Серверная AI-студия с обучением, контекстными чатами и памятью диалогов
              </Typography>
              <Typography variant="body1" className="hero-panel__description">
                Теперь обучение и ответы живут на Node.js-сервере, а интерфейс управляет корпусом,
                чатами, оценками и статусом модели. Если раньше модель выглядела игрушечной, теперь
                это уже полноценный локальный проект с сохранением артефактов и памяти диалогов.
              </Typography>
            </div>

            <div className="hero-panel__status-wrap">
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
              <Tab icon={<ForumIcon />} iconPosition="start" label="Чаты и контекст" />
              <Tab icon={<HubRoundedIcon />} iconPosition="start" label="О модели" disabled />
            </Tabs>
          </div>
        </GlassPanel>

        {studio.error ? (
          <Alert severity="error" className="top-alert">
            {studio.error}
          </Alert>
        ) : null}

        {tab === 0 ? (
          <TrainingTab
            snapshot={studio.snapshot}
            busy={studio.busy}
            error={studio.error}
            onSaveSettings={studio.actions.saveSettings}
            onUploadFiles={studio.actions.uploadFiles}
            onAddUrlSource={studio.actions.addUrlSource}
            onRemoveSource={studio.actions.removeSource}
            onCreateModel={studio.actions.createModel}
            onTrainModel={studio.actions.trainModel}
            onPauseModel={studio.actions.pauseModel}
            onResetModel={studio.actions.resetModel}
          />
        ) : (
          <ChatTab
            snapshot={studio.snapshot}
            selectedChatId={studio.selectedChatId}
            setSelectedChatId={studio.setSelectedChatId}
            busy={studio.busy}
            error={studio.error}
            onCreateChat={studio.actions.createChat}
            onDeleteChat={studio.actions.deleteChat}
            onSendMessage={studio.actions.sendMessage}
            onRateMessage={studio.actions.rateMessage}
          />
        )}
      </Container>
    </Box>
  );
}

export default App;
