import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
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
import CheckRoundedIcon from '@mui/icons-material/CheckRounded';
import ContentCopyRoundedIcon from '@mui/icons-material/ContentCopyRounded';
import OpenInNewRoundedIcon from '@mui/icons-material/OpenInNewRounded';
import MenuRoundedIcon from '@mui/icons-material/MenuRounded';
import SendRoundedIcon from '@mui/icons-material/SendRounded';
import ThumbDownAltRoundedIcon from '@mui/icons-material/ThumbDownAltRounded';
import ThumbDownAltOutlinedIcon from '@mui/icons-material/ThumbDownAltOutlined';
import ThumbUpAltRoundedIcon from '@mui/icons-material/ThumbUpAltRounded';
import ThumbUpAltOutlinedIcon from '@mui/icons-material/ThumbUpAltOutlined';
import GlassPanel from '../shared/GlassPanel';
import StatusPill from '../shared/StatusPill';
import { highlightCode, parseMessageContent, splitTextParagraphs } from '../../utils/chatContent';
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

const TYPING_PROGRESS_STORAGE_KEY = 'chat.typing-progress.v1';
const MAX_TYPING_RECORDS = 400;

function createContentSignature(text) {
  const value = String(text || '');
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = ((hash << 5) - hash) + value.charCodeAt(index);
    hash |= 0;
  }
  return `${value.length}:${hash}`;
}

function readTypingProgressStore() {
  if (typeof window === 'undefined') {
    return {};
  }

  try {
    const raw = window.localStorage.getItem(TYPING_PROGRESS_STORAGE_KEY);
    if (!raw) {
      return {};
    }
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch (_error) {
    return {};
  }
}

function saveTypingProgressStore(store) {
  if (typeof window === 'undefined') {
    return store;
  }

  const entries = Object.entries(store || {})
    .filter(([id, value]) => Boolean(id) && value && typeof value === 'object')
    .sort((left, right) => (Number(right[1].updatedAt || 0) - Number(left[1].updatedAt || 0)))
    .slice(0, MAX_TYPING_RECORDS);

  const normalized = Object.fromEntries(entries);
  try {
    window.localStorage.setItem(TYPING_PROGRESS_STORAGE_KEY, JSON.stringify(normalized));
  } catch (_error) {
    // Ignore storage quota errors: visual effect must not break chat flow.
  }
  return normalized;
}

function upsertTypingProgress(store, messageId, patch) {
  if (!messageId) {
    return store || {};
  }

  const nextStore = {
    ...(store || {}),
    [messageId]: {
      ...((store || {})[messageId] || {}),
      ...patch,
      updatedAt: Date.now(),
    },
  };
  return saveTypingProgressStore(nextStore);
}

function isAssistantReplyMessage(message) {
  return message?.role === 'assistant' && !['system', 'pending'].includes(message?.metadata?.type);
}

function getNextTypingIndex(text, currentIndex) {
  const value = String(text || '');
  if (!value.length) {
    return 0;
  }

  let nextIndex = Math.max(0, Number(currentIndex) || 0);
  if (nextIndex >= value.length) {
    return value.length;
  }

  while (nextIndex < value.length && /\s/u.test(value[nextIndex])) {
    nextIndex += 1;
  }
  while (nextIndex < value.length && !/\s/u.test(value[nextIndex])) {
    nextIndex += 1;
  }
  while (nextIndex < value.length && /\s/u.test(value[nextIndex])) {
    nextIndex += 1;
  }

  return Math.max(currentIndex + 1, Math.min(nextIndex, value.length));
}

function renderInlineContent(text) {
  const segments = String(text || '').split(/(`[^`\n]+`)/gu);
  return segments.map((segment, index) => {
    if (/^`[^`\n]+`$/u.test(segment)) {
      return (
        <code key={`inline-${index}`} className="message-inline-code">
          {segment.slice(1, -1)}
        </code>
      );
    }

    const lines = segment.split('\n');
    return (
      <span key={`text-${index}`}>
        {lines.map((line, lineIndex) => (
          <span key={`line-${lineIndex}`}>
            {line}
            {lineIndex < lines.length - 1 ? <br /> : null}
          </span>
        ))}
      </span>
    );
  });
}

function MessageTextBlock({ content }) {
  const paragraphs = useMemo(() => splitTextParagraphs(content), [content]);

  return (
    <>
      {paragraphs.map((paragraph, index) => (
        <p key={`${paragraph.slice(0, 24)}-${index}`} className="message-text-block">
          {renderInlineContent(paragraph)}
        </p>
      ))}
    </>
  );
}

function CodeBlock({ code, language }) {
  const [copied, setCopied] = useState(false);
  const highlightedLines = useMemo(() => highlightCode(code, language), [code, language]);

  const handleCopy = async () => {
    try {
      await window.navigator.clipboard.writeText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch (_error) {
      setCopied(false);
    }
  };

  return (
    <div className="message-code-block">
      <div className="message-code-block__header">
        <div className="message-code-block__meta">
          <StatusPill label={language || 'text'} tone="neutral" />
          <span className="message-code-block__line-count">
            {highlightedLines.length} lines
          </span>
        </div>
        <IconButton size="small" className="message-code-block__copy" onClick={handleCopy}>
          {copied ? <CheckRoundedIcon fontSize="small" /> : <ContentCopyRoundedIcon fontSize="small" />}
        </IconButton>
      </div>
      <div className="message-code-block__body">
        <div className="message-code-block__gutter">
          {highlightedLines.map((_tokens, index) => (
            <span key={`line-number-${index + 1}`}>{index + 1}</span>
          ))}
        </div>
        <pre className="message-code-block__pre">
          <code>
            {highlightedLines.map((tokens, lineIndex) => (
              <div key={`code-line-${lineIndex}`} className="message-code-block__line">
                {tokens.length ? tokens.map((token, tokenIndex) => (
                  <span
                    key={`token-${lineIndex}-${tokenIndex}`}
                    className={`token token--${token.type}`.trim()}
                  >
                    {token.text}
                  </span>
                )) : ' '}
              </div>
            ))}
          </code>
        </pre>
      </div>
    </div>
  );
}

function MessageContent({ content }) {
  const parts = useMemo(() => parseMessageContent(content), [content]);

  return (
    <>
      {parts.map((part, index) => (
        part.type === 'code' ? (
          <CodeBlock
            key={`code-${index}`}
            code={part.content}
            language={part.language}
          />
        ) : (
          <MessageTextBlock
            key={`text-${index}`}
            content={part.content}
          />
        )
      ))}
    </>
  );
}

function MessageReferences({ references = [] }) {
  if (!references.length) {
    return null;
  }

  return (
    <div className="message-references">
      <div className="message-references__header">
        <span>Sources</span>
        <StatusPill label={`${references.length}`} tone="neutral" />
      </div>
      <div className="message-references__list">
        {references.map((reference, index) => {
          const card = (
            <>
              <div className="message-reference-card__head">
                <StatusPill
                  label={reference.type === 'web' ? 'Web' : 'Knowledge'}
                  tone={reference.type === 'web' ? 'accent' : 'neutral'}
                />
                {reference.host ? (
                  <span className="message-reference-card__host">{reference.host}</span>
                ) : null}
              </div>
              <strong className="message-reference-card__title">{reference.title}</strong>
              {reference.excerpt ? (
                <span className="message-reference-card__excerpt">{reference.excerpt}</span>
              ) : null}
            </>
          );

          return reference.url ? (
            <a
              key={`${reference.url}-${index}`}
              className="message-reference-card message-reference-card--link"
              href={reference.url}
              target="_blank"
              rel="noreferrer"
            >
              {card}
              <OpenInNewRoundedIcon fontSize="small" className="message-reference-card__icon" />
            </a>
          ) : (
            <div key={`${reference.title}-${index}`} className="message-reference-card">
              {card}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function supportsTypingAnimation(message) {
  return !message?.metadata?.hasCodeBlocks && !/```/u.test(String(message?.content || ''));
}

function MessageBubble({
  message,
  busy,
  optimisticRating = 0,
  onRateMessage,
  displayContent = null,
  isTyping = false,
}) {
  const isRateable = message.role === 'assistant' && !['system', 'pending'].includes(message.metadata?.type);
  const currentRating = Number(optimisticRating || message.metadata?.userRating || 0);
  const selfScore = Number(message.metadata?.selfScore || 0);
  const isPendingReply = message.metadata?.type === 'pending';

  return (
    <div className={`message-bubble message-bubble--${message.role}`}>
      <GlassPanel filterStyle={{borderRadius:"28px"}}>
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
          <div className="message-bubble__content">
            <MessageContent content={displayContent ?? message.content} />
            {isTyping ? <span className="message-bubble__typing-cursor" aria-hidden>|</span> : null}
          </div>
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

        <MessageReferences references={message.metadata?.references || []} />

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
      </GlassPanel>
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
  onNoticesChange,
  onCreateChat,
  onDeleteChat,
  onSendMessage,
  onRateMessage,
}) {
  const [draft, setDraft] = useState('');
  const [mobileChatsOpen, setMobileChatsOpen] = useState(false);
  const messageListRef = useRef(null);
  const previousMessageIdsRef = useRef([]);
  const followTypingRef = useRef(false);
  const initializedChatIdRef = useRef(null);
  const typingStoreRef = useRef(readTypingProgressStore());
  const typingSaveThrottleRef = useRef(0);
  const typingVisibleLengthRef = useRef(0);
  const [typingMessageId, setTypingMessageId] = useState(null);
  const [typingVisibleLength, setTypingVisibleLength] = useState(0);

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

  const assistantReplyMessages = useMemo(
    () => visibleMessages.filter((message) => isAssistantReplyMessage(message) && Boolean(message.content)),
    [visibleMessages]
  );

  const scrollToBottom = useCallback(() => {
    const node = messageListRef.current;
    if (!node) {
      return;
    }
    node.scrollTop = node.scrollHeight;
  }, []);

  const handleMessageListScroll = useCallback((event) => {
    const node = event.currentTarget;
    const distanceToBottom = node.scrollHeight - node.scrollTop - node.clientHeight;
    followTypingRef.current = distanceToBottom <= 36;
  }, []);

  const startTypingForMessage = useCallback((message) => {
    if (!isAssistantReplyMessage(message) || !message.content || !supportsTypingAnimation(message)) {
      return false;
    }

    const signature = createContentSignature(message.content);
    const stored = typingStoreRef.current?.[message.id];
    let nextVisibleLength = 0;

    if (stored) {
      const storedLength = Math.max(0, Number(stored.visibleLength) || 0);
      if (stored.contentSignature === signature && stored.completed) {
        return false;
      }
      nextVisibleLength = Math.min(storedLength, message.content.length);
    }

    if (nextVisibleLength >= message.content.length) {
      typingStoreRef.current = upsertTypingProgress(typingStoreRef.current, message.id, {
        contentSignature: signature,
        visibleLength: message.content.length,
        completed: true,
      });
      return false;
    }

    typingStoreRef.current = upsertTypingProgress(typingStoreRef.current, message.id, {
      contentSignature: signature,
      visibleLength: nextVisibleLength,
      completed: false,
    });

    typingVisibleLengthRef.current = nextVisibleLength;
    setTypingVisibleLength(nextVisibleLength);
    setTypingMessageId(message.id);
    return true;
  }, []);

  useEffect(() => {
    typingVisibleLengthRef.current = typingVisibleLength;
  }, [typingVisibleLength]);

  useEffect(() => {
    const currentChatId = activeChat?.id || null;
    if (initializedChatIdRef.current === currentChatId) {
      return;
    }
    initializedChatIdRef.current = currentChatId;

    const store = readTypingProgressStore();
    const normalizedStore = { ...store };
    const persistedMessages = (activeChat?.messages || []).filter((message) => (
      isAssistantReplyMessage(message) && Boolean(message.content)
    ));
    const latestAssistantMessageId = persistedMessages[persistedMessages.length - 1]?.id || null;

    persistedMessages.forEach((message) => {
      const signature = createContentSignature(message.content);
      const stored = normalizedStore[message.id];
      const allowResume = message.id === latestAssistantMessageId;

      if (!stored) {
        normalizedStore[message.id] = {
          contentSignature: signature,
          visibleLength: message.content.length,
          completed: true,
          updatedAt: Date.now(),
        };
        return;
      }

      const storedLength = Math.max(0, Number(stored.visibleLength) || 0);
      const keepCompleted = !allowResume || (stored.contentSignature === signature && Boolean(stored.completed));
      normalizedStore[message.id] = {
        contentSignature: signature,
        visibleLength: keepCompleted ? message.content.length : Math.min(storedLength, message.content.length),
        completed: keepCompleted,
        updatedAt: Date.now(),
      };
    });

    typingStoreRef.current = saveTypingProgressStore(normalizedStore);
    previousMessageIdsRef.current = visibleMessages.map((message) => message.id);
    followTypingRef.current = true;
    typingVisibleLengthRef.current = 0;
    setTypingVisibleLength(0);
    setTypingMessageId(null);
    requestAnimationFrame(() => {
      scrollToBottom();
    });
  }, [activeChat?.id, activeChat?.messages, scrollToBottom, visibleMessages]);

  useEffect(() => {
    const currentIds = visibleMessages.map((message) => message.id);
    const previousIds = previousMessageIdsRef.current;

    if (!previousIds.length) {
      previousMessageIdsRef.current = currentIds;
      return;
    }

    const previousSet = new Set(previousIds);
    const newMessages = visibleMessages.filter((message) => !previousSet.has(message.id));
    if (newMessages.length) {
      followTypingRef.current = true;
      requestAnimationFrame(() => {
        scrollToBottom();
      });
    }

    const newAssistantMessage = [...newMessages]
      .reverse()
      .find((message) => isAssistantReplyMessage(message) && Boolean(message.content));

    if (newAssistantMessage) {
      startTypingForMessage(newAssistantMessage);
    }

    previousMessageIdsRef.current = currentIds;
  }, [scrollToBottom, startTypingForMessage, visibleMessages]);

  useEffect(() => {
    if (typingMessageId) {
      return;
    }

    const resumable = assistantReplyMessages[assistantReplyMessages.length - 1];
    if (!resumable) {
      return;
    }

    const stored = typingStoreRef.current?.[resumable.id];
    if (!stored) {
      return;
    }

    const visibleLength = Math.max(0, Number(stored.visibleLength) || 0);
    if (!stored.completed && visibleLength < resumable.content.length) {
      followTypingRef.current = true;
      startTypingForMessage(resumable);
    }
  }, [assistantReplyMessages, startTypingForMessage, typingMessageId]);

  const typingTargetMessage = useMemo(
    () => visibleMessages.find((message) => message.id === typingMessageId) || null,
    [typingMessageId, visibleMessages]
  );

  useEffect(() => {
    if (!typingTargetMessage || !isAssistantReplyMessage(typingTargetMessage)) {
      return undefined;
    }

    const fullText = typingTargetMessage.content || '';
    const signature = createContentSignature(fullText);
    if (!fullText.length) {
      typingStoreRef.current = upsertTypingProgress(typingStoreRef.current, typingTargetMessage.id, {
        contentSignature: signature,
        visibleLength: 0,
        completed: true,
      });
      setTypingMessageId(null);
      return undefined;
    }

    const tick = () => {
      const currentLength = Math.min(typingVisibleLengthRef.current, fullText.length);
      if (currentLength >= fullText.length) {
        typingStoreRef.current = upsertTypingProgress(typingStoreRef.current, typingTargetMessage.id, {
          contentSignature: signature,
          visibleLength: fullText.length,
          completed: true,
        });
        setTypingMessageId((currentId) => (currentId === typingTargetMessage.id ? null : currentId));
        return;
      }

      const nextLength = getNextTypingIndex(fullText, currentLength);
      typingVisibleLengthRef.current = nextLength;
      setTypingVisibleLength(nextLength);

      const completed = nextLength >= fullText.length;
      const now = Date.now();
      if (completed || now - typingSaveThrottleRef.current > 140) {
        typingStoreRef.current = upsertTypingProgress(typingStoreRef.current, typingTargetMessage.id, {
          contentSignature: signature,
          visibleLength: nextLength,
          completed,
        });
        typingSaveThrottleRef.current = now;
      }

      if (followTypingRef.current) {
        requestAnimationFrame(() => {
          scrollToBottom();
        });
      }

      if (completed) {
        setTypingMessageId((currentId) => (currentId === typingTargetMessage.id ? null : currentId));
      }
    };

    const intervalId = setInterval(tick, 38);
    return () => clearInterval(intervalId);
  }, [scrollToBottom, typingTargetMessage]);

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

  const floatingNotices = useMemo(() => {
    const notices = [];

    if (error) {
      notices.push({ id: 'chat-error', severity: 'error', message: error });
    }
    if (trainingLocked) {
      notices.push({ id: 'chat-locked', severity: 'info', message: 'Чат в режиме просмотра: пока идёт обучение.' });
    }
    if (replyPending) {
      notices.push({ id: 'chat-reply-pending', severity: 'info', message: 'Сервер формирует ответ...' });
    }
    if (activeChat?.truncated) {
      notices.push({
        id: 'chat-truncated',
        severity: 'info',
        message: `Показаны последние ${formatNumber(activeChat.returnedMessageCount || 0)} сообщений из ${formatNumber(activeChat.totalMessageCount || 0)}.`,
      });
    }
    if (!canChat) {
      notices.push({ id: 'chat-not-trained', severity: 'warning', message: 'Сначала обучите модель, затем чат станет доступен.' });
    }

    const priority = { error: 0, warning: 1, success: 2, info: 3 };
    return notices
      .sort((left, right) => (priority[left.severity] ?? 10) - (priority[right.severity] ?? 10))
      .slice(0, 5);
  }, [
    activeChat?.returnedMessageCount,
    activeChat?.totalMessageCount,
    activeChat?.truncated,
    canChat,
    error,
    replyPending,
    trainingLocked,
  ]);

  useEffect(() => {
    onNoticesChange?.(floatingNotices);
    return () => {
      onNoticesChange?.([]);
    };
  }, [floatingNotices, onNoticesChange]);

  const handleCreateChat = useCallback(async () => {
    await onCreateChat();
    setMobileChatsOpen(false);
  }, [onCreateChat]);

  const handleSelectChat = useCallback((chatId) => {
    setSelectedChatId(chatId);
    setMobileChatsOpen(false);
  }, [setSelectedChatId]);

  return (
    <div className="chat-tab">
      <div className="chat-layout">
        <div
          className={`chat-sidebar-panel chat-sidebar-panel--plain ${mobileChatsOpen ? 'chat-sidebar-panel--open' : ''}`.trim()}
        >
          <div className="chat-sidebar-panel__inner">
            <div className="panel-heading">
              <div>
                <Typography variant="h3">Сохраненные чаты</Typography>
                <Typography variant="body2" className="muted-text">
                  История переписки, оценки ответов и обучающие пары, хранится на сервере.
                </Typography>
              </div>
              <GlassPanel className="chat-sidebar-action-shell" innerClassName="chat-sidebar-action-shell__inner">
                <Button
                  variant="contained"
                  className="action-button chat-sidebar-action-button"
                  startIcon={<AddCommentRoundedIcon />}
                  onClick={handleCreateChat}
                  disabled={interactionLocked}
                  fullWidth
                >
                  Новый чат
                </Button>
              </GlassPanel>
            </div>

            <div className="chat-list">
              {snapshot.chats.map((chat) => (
                <GlassPanel
                  key={chat.id}
                  className={`chat-list-item-panel ${selectedChatId === chat.id ? 'chat-list-item-panel--active' : ''}`.trim()}
                  innerClassName="chat-list-item-panel__inner"
                >
                  <div
                    className={`chat-list-item ${selectedChatId === chat.id ? 'chat-list-item--active' : ''}`.trim()}
                  >
                    <button
                      type="button"
                      className="chat-list-item__select"
                      onClick={() => handleSelectChat(chat.id)}
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
                </GlassPanel>
              ))}
            </div>
          </div>
        </div>

        <button
          type="button"
          className={`chat-sidebar-backdrop ${mobileChatsOpen ? 'chat-sidebar-backdrop--visible' : ''}`.trim()}
          aria-label="Закрыть список чатов"
          onClick={() => setMobileChatsOpen(false)}
        />

        <div className="chat-main-panel chat-main-panel--plain">
          <div className="chat-main-panel__inner">
            <div className="panel-heading">
              <div className="chat-main-panel__heading-primary">
                <GlassPanel className="chat-mobile-menu-shell" innerClassName="chat-mobile-menu-shell__inner">
                  <Button
                    variant="contained"
                    className="action-button chat-mobile-menu-button"
                    startIcon={<MenuRoundedIcon />}
                    onClick={() => setMobileChatsOpen(true)}
                  >
                    Чаты
                  </Button>
                </GlassPanel>
                <div style={{marginLeft: '30px'}}>
                  <Typography variant="h3">{activeChat?.title || 'Чат'}</Typography>
                </div>
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
              Генерация ответа выполняется на сервере.
            </Alert>
          ) : null}
          {activeChat?.truncated ? (
            <Alert severity="info">
              {`Показаны последние ${formatNumber(activeChat.returnedMessageCount || 0)} из ${formatNumber(activeChat.totalMessageCount || 0)} сообщений, чтобы чат не перегружал память.`}
            </Alert>
          ) : null}
          {!canChat ? (
            <Alert severity="info">
              Сначала обучите модель на вкладке обучения, иначе ответы будут недоступны.
            </Alert>
          ) : null}

          <div className="message-list" ref={messageListRef} onScroll={handleMessageListScroll}>
            {visibleMessages.map((message) => {
              const isTyping = (
                message.id === typingMessageId &&
                isAssistantReplyMessage(message) &&
                supportsTypingAnimation(message)
              );
              const displayContent = isTyping
                ? message.content.slice(0, Math.min(typingVisibleLength, message.content.length))
                : message.content;

              return (
                <MessageBubble
                  key={message.id}
                  message={message}
                  busy={interactionLocked}
                  optimisticRating={pendingRatings?.[message.id]}
                  onRateMessage={onRateMessage}
                  displayContent={displayContent}
                  isTyping={isTyping}
                />
              );
            })}
          </div>

          <div className="chat-input-row">
            <GlassPanel className="chat-input-shell" innerClassName="chat-input-shell__inner">
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
            </GlassPanel>
            <GlassPanel className="chat-send-shell" innerClassName="chat-send-shell__inner">
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
                fullWidth
              >
                Отправить
              </Button>
            </GlassPanel>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
