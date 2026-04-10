import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Button,
  CircularProgress,
  IconButton,
  TextField,
  Typography,
} from '@mui/material';
import AddCommentRoundedIcon from '@mui/icons-material/AddCommentRounded';
import DeleteOutlineRoundedIcon from '@mui/icons-material/DeleteOutlineRounded';
import SendRoundedIcon from '@mui/icons-material/SendRounded';
import ThumbDownAltRoundedIcon from '@mui/icons-material/ThumbDownAltRounded';
import ThumbDownAltOutlinedIcon from '@mui/icons-material/ThumbDownAltOutlined';
import ThumbUpAltRoundedIcon from '@mui/icons-material/ThumbUpAltRounded';
import ThumbUpAltOutlinedIcon from '@mui/icons-material/ThumbUpAltOutlined';
import GlassPanel from '../shared/GlassPanel';
import StatusPill from '../shared/StatusPill';
import { formatDateTime, formatNumber, previewText } from '../../utils/text';
import '../../styles/chat-tab.css';

function mapModeLabel(mode) {
  switch (mode) {
    case 'reply_memory':
      return 'Память ответов';
    case 'knowledge':
      return 'Извлечение знаний';
    case 'neural':
      return 'Нейросеть';
    case 'hybrid':
      return 'Гибридный ответ';
    case 'fallback':
      return 'Недостаточно данных';
    case 'system':
      return 'Системное сообщение';
    default:
      return mode || 'Ответ';
  }
}

function MessageBubble({ message, busy, optimisticRating = 0, onRateMessage }) {
  const isRateable = message.role === 'assistant' && !['system', 'pending'].includes(message.metadata?.type);
  const currentRating = Number(optimisticRating || message.metadata?.userRating || 0);
  const selfScore = Number(message.metadata?.selfScore || 0);
  const isPendingReply = message.metadata?.type === 'pending';

  return (
    <div className={`message-bubble message-bubble--${message.role}`}>
      <div className="message-bubble__meta">
        <span>{message.role === 'user' ? 'Вы' : 'Модель'}</span>
        <span>{formatDateTime(message.createdAt)}</span>
      </div>

      {isPendingReply ? (
        <div className="message-bubble__loading">
          <CircularProgress size={16} thickness={5} />
          <span>{message.metadata?.loadingLabel || 'Формирую ответ...'}</span>
        </div>
      ) : (
        <Typography variant="body1" className="message-bubble__content">
          {message.content}
        </Typography>
      )}

      {message.metadata ? (
        <div className="message-bubble__badges">
          {message.metadata.mode ? (
            <StatusPill label={mapModeLabel(message.metadata.mode)} tone="neutral" />
          ) : null}
          {message.metadata.chunkMatches ? (
            <StatusPill label={`Фрагменты: ${message.metadata.chunkMatches}`} tone="neutral" />
          ) : null}
          {message.metadata.usedContextMessages ? (
            <StatusPill label={`Контекст: ${message.metadata.usedContextMessages}`} tone="neutral" />
          ) : null}
          {message.metadata.selfScore !== undefined ? (
            <StatusPill label={`Самооценка: ${selfScore.toFixed(3)}`} tone="neutral" />
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
              {currentRating === 1 ? (
                <ThumbUpAltRoundedIcon fontSize="small" />
              ) : (
                <ThumbUpAltOutlinedIcon fontSize="small" />
              )}
            </IconButton>
            <IconButton
              size="small"
              className={`feedback-button feedback-button--negative ${currentRating === -1 ? 'feedback-button--active' : ''}`.trim()}
              disabled={busy}
              onClick={() => onRateMessage(message.id, -1)}
            >
              {currentRating === -1 ? (
                <ThumbDownAltRoundedIcon fontSize="small" />
              ) : (
                <ThumbDownAltOutlinedIcon fontSize="small" />
              )}
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
  pendingAction,
  pendingReply,
  pendingRatings,
  error,
  onCreateChat,
  onDeleteChat,
  onSendMessage,
  onRateMessage,
}) {
  const [draft, setDraft] = useState('');
  const messageListRef = useRef(null);

  const activeChat = snapshot.activeChat;
  const canChat = Boolean(snapshot.model.exists && snapshot.model.trainedEpochs);
  const trainingLocked = snapshot.training.status === 'training' || snapshot.model.status === 'saving_checkpoint';
  const interactionLocked = busy || trainingLocked;
  const replyPending = pendingAction === 'sendMessage' && pendingReply?.chatId === activeChat?.id;

  const visibleMessages = useMemo(() => {
    const baseMessages = [...(activeChat?.messages || [])];
    if (!replyPending || !pendingReply) {
      return baseMessages;
    }

    const userAlreadySynced = baseMessages.some((message) => (
      message.role === 'user' &&
      message.content === pendingReply.content &&
      message.createdAt >= pendingReply.createdAt
    ));

    if (!userAlreadySynced) {
      baseMessages.push({
        id: `pending-user-${pendingReply.createdAt}`,
        role: 'user',
        content: pendingReply.content,
        createdAt: pendingReply.createdAt,
      });
    }

    baseMessages.push({
      id: `pending-assistant-${pendingReply.createdAt}`,
      role: 'assistant',
      content: '',
      createdAt: pendingReply.createdAt,
      metadata: {
        type: 'pending',
        loadingLabel: 'Нейросеть формирует ответ по чату и базе знаний. Это может занять время.',
      },
    });

    return baseMessages;
  }, [activeChat?.messages, pendingReply, replyPending]);

  useEffect(() => {
    const node = messageListRef.current;
    if (!node) {
      return;
    }

    node.scrollTop = node.scrollHeight;
  }, [visibleMessages]);

  const contextInfo = useMemo(
    () => [
      `Сохранено чатов: ${formatNumber(snapshot.chats.length)}`,
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
                Каждый чат хранится на сервере. История переписки, оценки ответов и обучающие пары не теряются между перезапусками.
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
                Ответ собирается из памяти текущего чата, релевантных фрагментов корпуса, памяти удачных ответов и нейросетевой генерации. Лайк усиливает ответ в памяти, дизлайк подавляет его в будущих диалогах.
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
          {replyPending ? (
            <Alert severity="info">
              Генерация ответа выполняется на сервере. Сообщение ожидания показано прямо в ленте чата.
            </Alert>
          ) : null}
          {!canChat ? (
            <Alert severity="info">
              Сначала обучите модель на вкладке обучения, иначе ответы будут недоступны.
            </Alert>
          ) : null}

          <div className="message-list" ref={messageListRef}>
            {visibleMessages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                busy={interactionLocked}
                optimisticRating={pendingRatings?.[message.id]}
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
              placeholder="Введите сообщение. Сервер сам соберет релевантный контекст из чата и базы знаний."
            />
            <Button
              variant="contained"
              className="action-button chat-send-button"
              startIcon={replyPending ? <CircularProgress size={18} color="inherit" /> : <SendRoundedIcon />}
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
