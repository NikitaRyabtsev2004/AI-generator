import { useMemo, useState } from 'react';
import {
  Alert,
  Button,
  IconButton,
  TextField,
  Typography,
} from '@mui/material';
import AddCommentRoundedIcon from '@mui/icons-material/AddCommentRounded';
import DeleteOutlineRoundedIcon from '@mui/icons-material/DeleteOutlineRounded';
import SendRoundedIcon from '@mui/icons-material/SendRounded';
import ThumbDownAltOutlinedIcon from '@mui/icons-material/ThumbDownAltOutlined';
import ThumbUpAltOutlinedIcon from '@mui/icons-material/ThumbUpAltOutlined';
import GlassPanel from '../shared/GlassPanel';
import StatusPill from '../shared/StatusPill';
import { formatDateTime, formatNumber, previewText } from '../../utils/text';
import '../../styles/chat-tab.css';

function MessageBubble({ message, busy, onRateMessage }) {
  const isRateable = message.role === 'assistant' && message.metadata?.type !== 'system';
  const currentRating = Number(message.metadata?.userRating || 0);
  const selfScore = Number(message.metadata?.selfScore || 0);

  return (
    <div className={`message-bubble message-bubble--${message.role}`}>
      <div className="message-bubble__meta">
        <span>{message.role === 'user' ? 'Вы' : 'Модель'}</span>
        <span>{formatDateTime(message.createdAt)}</span>
      </div>

      <Typography variant="body1" className="message-bubble__content">
        {message.content}
      </Typography>

      {message.metadata ? (
        <div className="message-bubble__badges">
          {message.metadata.mode ? <StatusPill label={message.metadata.mode} tone="neutral" /> : null}
          {message.metadata.chunkMatches ? (
            <StatusPill label={`фрагменты: ${message.metadata.chunkMatches}`} tone="neutral" />
          ) : null}
          {message.metadata.usedContextMessages ? (
            <StatusPill label={`контекст: ${message.metadata.usedContextMessages}`} tone="neutral" />
          ) : null}
          {message.metadata.selfScore !== undefined ? (
            <StatusPill label={`self-score: ${selfScore.toFixed(3)}`} tone="neutral" />
          ) : null}
        </div>
      ) : null}

      {isRateable ? (
        <div className="message-bubble__feedback">
          <span className="message-bubble__feedback-label">Оценить ответ</span>
          <div className="message-bubble__feedback-actions">
            <IconButton
              size="small"
              className={`feedback-button ${currentRating === 1 ? 'feedback-button--active' : ''}`.trim()}
              disabled={busy}
              onClick={() => onRateMessage(message.id, 1)}
            >
              <ThumbUpAltOutlinedIcon fontSize="small" />
            </IconButton>
            <IconButton
              size="small"
              className={`feedback-button feedback-button--negative ${currentRating === -1 ? 'feedback-button--active' : ''}`.trim()}
              disabled={busy}
              onClick={() => onRateMessage(message.id, -1)}
            >
              <ThumbDownAltOutlinedIcon fontSize="small" />
            </IconButton>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default function ChatTab({
  snapshot,
  selectedChatId,
  setSelectedChatId,
  busy,
  error,
  onCreateChat,
  onDeleteChat,
  onSendMessage,
  onRateMessage,
}) {
  const [draft, setDraft] = useState('');

  const activeChat = snapshot.activeChat;
  const canChat = Boolean(snapshot.model.exists && snapshot.model.trainedEpochs);
  const trainingLocked = snapshot.training.status === 'training' || snapshot.model.status === 'saving_checkpoint';
  const interactionLocked = busy || trainingLocked;

  const contextInfo = useMemo(
    () => [
      `Чатов сохранено: ${formatNumber(snapshot.chats.length)}`,
      `Контекст: ${snapshot.runtime?.contextStrategy === 'whole_chat_weighted' ? 'релевантный + недавний' : 'настраиваемый'}`,
      `Генератор: ${snapshot.runtime?.generatorBackend || 'neural'}`,
      `Оцененных ответов: ${formatNumber(snapshot.runtime?.ratedMessages || 0)}`,
    ],
    [
      snapshot.chats.length,
      snapshot.runtime?.contextStrategy,
      snapshot.runtime?.generatorBackend,
      snapshot.runtime?.ratedMessages,
    ]
  );

  return (
    <div className="chat-tab">
      <div className="chat-layout">
        <GlassPanel className="chat-sidebar-panel">
          <div className="panel-heading">
            <div>
              <Typography variant="h3">Сохраненные чаты</Typography>
              <Typography variant="body2" className="muted-text">
                Каждый чат хранится на сервере. Память контекста, оценки ответов и обучающие пары
                остаются в проекте между перезапусками.
              </Typography>
            </div>
            <Button
              variant="contained"
              className="action-button"
              startIcon={<AddCommentRoundedIcon />}
              onClick={() => onCreateChat()}
              disabled={interactionLocked}
            >
              Новый чат
            </Button>
          </div>

          <div className="chat-list">
            {snapshot.chats.map((chat) => (
              <div
                key={chat.id}
                className={`chat-list-item ${selectedChatId === chat.id ? 'chat-list-item--active' : ''}`.trim()}
              >
                <button
                  type="button"
                  className="chat-list-item__select"
                  onClick={() => setSelectedChatId(chat.id)}
                >
                  <div className="chat-list-item__head">
                    <strong>{chat.title}</strong>
                  </div>
                  <span className="chat-list-item__preview">{previewText(chat.lastMessagePreview, 72)}</span>
                </button>
                <div className="chat-list-item__actions">
                  <IconButton
                    size="small"
                    onClick={() => onDeleteChat(chat.id)}
                    disabled={interactionLocked || snapshot.chats.length === 1}
                  >
                    <DeleteOutlineRoundedIcon fontSize="small" />
                  </IconButton>
                </div>
              </div>
            ))}
          </div>
        </GlassPanel>

        <GlassPanel className="chat-main-panel">
          <div className="panel-heading">
            <div>
              <Typography variant="h3">{activeChat?.title || 'Чат'}</Typography>
              <Typography variant="body2" className="muted-text">
                Ответ строится из трех слоев: контекст текущего чата, релевантные фрагменты корпуса
                и реальная sequence-модель. Лайк усиливает этот ответ в будущих обучающих прогонах,
                дизлайк подавляет его.
              </Typography>
            </div>
            <div className="chat-context-pills">
              {contextInfo.map((item) => (
                <StatusPill key={item} label={item} tone="neutral" />
              ))}
            </div>
          </div>

          {error ? <Alert severity="error">{error}</Alert> : null}
          {trainingLocked ? (
            <Alert severity="info">
              Во время обучения чат переведен в режим просмотра. Дождитесь завершения или поставьте обучение на паузу.
            </Alert>
          ) : null}
          {!canChat ? (
            <Alert severity="info">
              Сначала обучите модель на вкладке обучения, иначе ответы будут недоступны.
            </Alert>
          ) : null}

          <div className="message-list">
            {activeChat?.messages?.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                busy={interactionLocked}
                onRateMessage={onRateMessage}
              />
            ))}
          </div>

          <div className="chat-input-row">
            <TextField
              className="text-input"
              multiline
              minRows={3}
              maxRows={8}
              value={draft}
              disabled={interactionLocked || !canChat}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="Напишите сообщение. Сервер сам соберет релевантный контекст текущего чата и базы знаний."
            />
            <Button
              variant="contained"
              className="action-button chat-send-button"
              startIcon={<SendRoundedIcon />}
              onClick={async () => {
                if (!draft.trim() || !activeChat) {
                  return;
                }

                const nextValue = draft;
                setDraft('');
                await onSendMessage(activeChat.id, nextValue);
              }}
              disabled={interactionLocked || !draft.trim() || !canChat}
            >
              Отправить
            </Button>
          </div>
        </GlassPanel>
      </div>
    </div>
  );
}
