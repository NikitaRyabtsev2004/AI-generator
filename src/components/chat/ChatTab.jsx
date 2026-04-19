import {useCallback, useEffect, useMemo, useRef, useState} from 'react';
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
import EditRoundedIcon from '@mui/icons-material/EditRounded';
import OpenInNewRoundedIcon from '@mui/icons-material/OpenInNewRounded';
import MenuRoundedIcon from '@mui/icons-material/MenuRounded';
import SendRoundedIcon from '@mui/icons-material/SendRounded';
import ShareRoundedIcon from '@mui/icons-material/ShareRounded';
import StopRoundedIcon from '@mui/icons-material/StopRounded';
import ThumbDownAltRoundedIcon from '@mui/icons-material/ThumbDownAltRounded';
import ThumbDownAltOutlinedIcon from '@mui/icons-material/ThumbDownAltOutlined';
import ThumbUpAltRoundedIcon from '@mui/icons-material/ThumbUpAltRounded';
import ThumbUpAltOutlinedIcon from '@mui/icons-material/ThumbUpAltOutlined';
import GlassPanel from '../shared/GlassPanel';
import StatusPill from '../shared/StatusPill';
import {highlightCode, parseMessageContent, splitTextParagraphs} from '../../utils/chatContent';
import {formatDateTime, formatNumber, previewText} from '../../utils/text';
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
        case 'api_error':
            return 'Ошибка API';
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

function isNonRateableAssistantMessage(message) {
    if (!message || message.role !== 'assistant') {
        return true;
    }

    const metadata = message.metadata || {};
    const type = String(metadata.type || '').toLowerCase();
    const mode = String(metadata.mode || '').toLowerCase();
    if (['system', 'pending', 'error', 'error_info'].includes(type)) {
        return true;
    }
    if (['api_error', 'error', 'system_error'].includes(mode)) {
        return true;
    }

    return Boolean(metadata.generationError || metadata.apiError);
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
    const segments = String(text || '').split(/(`[^`\n]+`|\[[^\]]{1,160}\]\(https?:\/\/[^\s)]+\))/gu);
    return segments.map((segment, index) => {
        if (/^`[^`\n]+`$/u.test(segment)) {
            return (
                <code key={`inline-${index}`} className="message-inline-code">
                    {segment.slice(1, -1)}
                </code>
            );
        }

        const markdownLinkMatch = segment.match(/^\[([^\]]{1,160})\]\((https?:\/\/[^\s)]+)\)$/u);
        if (markdownLinkMatch) {
            return (
                <a
                    key={`link-${index}`}
                    href={markdownLinkMatch[2]}
                    target="_blank"
                    rel="noreferrer"
                    className="message-inline-link"
                >
                    {markdownLinkMatch[1]}
                </a>
            );
        }

        const lines = segment.split('\n');
        return (
            <span key={`text-${index}`}>
        {lines.map((line, lineIndex) => (
            <span key={`line-${lineIndex}`}>
            {line}
                {lineIndex < lines.length - 1 ? <br/> : null}
          </span>
        ))}
      </span>
        );
    });
}

function splitTableCells(line = '') {
    return String(line || '')
        .trim()
        .replace(/^\|/u, '')
        .replace(/\|$/u, '')
        .split('|')
        .map((cell) => cell.trim());
}

function isMarkdownTableDivider(line = '') {
    const cells = splitTableCells(line);
    if (!cells.length) {
        return false;
    }
    return cells.every((cell) => /^:?-{3,}:?$/u.test(cell));
}

function parseTableAlignments(line = '') {
    const cells = splitTableCells(line);
    return cells.map((cell) => {
        const starts = cell.startsWith(':');
        const ends = cell.endsWith(':');
        if (starts && ends) {
            return 'center';
        }
        if (ends) {
            return 'right';
        }
        return 'left';
    });
}

function parseMarkdownTextBlocks(content = '') {
    return splitTextParagraphs(content).map((paragraph) => {
        const lines = String(paragraph || '')
            .split('\n')
            .map((line) => line.replace(/\s+$/u, ''))
            .filter(Boolean);

        if (!lines.length) {
            return null;
        }

        if (lines.length === 1 && /^#{1,3}\s+/u.test(lines[0])) {
            const match = lines[0].match(/^(#{1,3})\s+([\s\S]+)$/u);
            return {
                type: 'heading',
                level: Math.min(match?.[1]?.length || 1, 3),
                content: match?.[2] || lines[0],
            };
        }

        if (
            lines.length >= 2 &&
            /\|/u.test(lines[0]) &&
            /\|/u.test(lines[1]) &&
            isMarkdownTableDivider(lines[1])
        ) {
            const headers = splitTableCells(lines[0]);
            const alignments = parseTableAlignments(lines[1]);
            const rows = lines
                .slice(2)
                .filter((line) => /\|/u.test(line))
                .map((line) => splitTableCells(line));

            return {
                type: 'table',
                headers,
                rows,
                alignments,
            };
        }

        if (lines.every((line) => /^\s*[-*]\s+/u.test(line))) {
            return {
                type: 'ul',
                items: lines.map((line) => line.replace(/^\s*[-*]\s+/u, '')),
            };
        }

        if (lines.every((line) => /^\s*\d+\.\s+/u.test(line))) {
            return {
                type: 'ol',
                items: lines.map((line) => line.replace(/^\s*\d+\.\s+/u, '')),
            };
        }

        if (lines.every((line) => /^\s*>\s?/u.test(line))) {
            return {
                type: 'quote',
                content: lines.map((line) => line.replace(/^\s*>\s?/u, '')).join('\n'),
            };
        }

        return {
            type: 'paragraph',
            content: paragraph,
        };
    }).filter(Boolean);
}

function MessageTextBlock({content}) {
    const blocks = useMemo(() => parseMarkdownTextBlocks(content), [content]);

    return (
        <>
            {blocks.map((block, index) => {
                if (block.type === 'heading') {
                    const HeadingTag = block.level === 1 ? 'h3' : block.level === 2 ? 'h4' : 'h5';
                    return (
                        <HeadingTag
                            key={`${block.content.slice(0, 24)}-${index}`}
                            className={`message-heading-block message-heading-block--${block.level}`}
                        >
                            {renderInlineContent(block.content)}
                        </HeadingTag>
                    );
                }

                if (block.type === 'ul' || block.type === 'ol') {
                    const ListTag = block.type === 'ul' ? 'ul' : 'ol';
                    return (
                        <ListTag key={`list-${index}`} className="message-list-block">
                            {block.items.map((item, itemIndex) => (
                                <li key={`item-${itemIndex}`}>{renderInlineContent(item)}</li>
                            ))}
                        </ListTag>
                    );
                }

                if (block.type === 'quote') {
                    return (
                        <blockquote key={`quote-${index}`} className="message-quote-block">
                            {renderInlineContent(block.content)}
                        </blockquote>
                    );
                }

                if (block.type === 'table') {
                    const columnCount = Math.max(
                        block.headers?.length || 0,
                        ...((block.rows || []).map((row) => row.length))
                    );

                    return (
                        <div key={`table-${index}`} className="message-table-wrap">
                            <table className="message-table">
                                <thead>
                                <tr>
                                    {Array.from({length: columnCount}).map((_, columnIndex) => (
                                        <th
                                            key={`th-${columnIndex}`}
                                            style={{textAlign: block.alignments?.[columnIndex] || 'left'}}
                                        >
                                            {renderInlineContent(block.headers?.[columnIndex] || '')}
                                        </th>
                                    ))}
                                </tr>
                                </thead>
                                <tbody>
                                {(block.rows || []).map((row, rowIndex) => (
                                    <tr key={`tr-${rowIndex}`}>
                                        {Array.from({length: columnCount}).map((_, columnIndex) => (
                                            <td
                                                key={`td-${rowIndex}-${columnIndex}`}
                                                style={{textAlign: block.alignments?.[columnIndex] || 'left'}}
                                            >
                                                {renderInlineContent(row?.[columnIndex] || '')}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                                </tbody>
                            </table>
                        </div>
                    );
                }

                return (
                    <p key={`${block.content.slice(0, 24)}-${index}`} className="message-text-block">
                        {renderInlineContent(block.content)}
                    </p>
                );
            })}
        </>
    );
}

function CodeBlock({code, language}) {
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
                    <StatusPill label={language || 'text'} tone="neutral"/>
                    <span className="message-code-block__line-count">
            {highlightedLines.length} lines
          </span>
                </div>
                <IconButton size="small" className="message-code-block__copy" onClick={handleCopy}>
                    {copied ? <CheckRoundedIcon fontSize="small"/> : <ContentCopyRoundedIcon fontSize="small"/>}
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

function MessageContent({content}) {
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

function MessageReferences({references = []}) {
    if (!references.length) {
        return null;
    }

    return (
        <div className="message-references">
            <div className="message-references__header">
                <span>Sources</span>
                <StatusPill label={`${references.length}`} tone="neutral"/>
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
                            <OpenInNewRoundedIcon fontSize="small" className="message-reference-card__icon"/>
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
                           onCopyMessage,
                           onEditMessage,
                           displayContent = null,
                           isTyping = false,
                       }) {
    const isRateable = !isNonRateableAssistantMessage(message);
    const isEditableUserMessage = message.role === 'user' && typeof onEditMessage === 'function';
    const currentRating = Number(optimisticRating || message.metadata?.userRating || 0);
    const selfScore = Number(message.metadata?.selfScore || 0);
    const isPendingReply = message.metadata?.type === 'pending';

    return (
        <div className={`message-bubble message-bubble--${message.role}`}>
            <GlassPanel filterStyle={{borderRadius: '28px'}}>
                <div className="message-bubble__meta">
                    <div className="message-bubble__meta-copy">
                        <span>{message.role === 'user' ? 'Вы' : 'Модель'}</span>
                        <span>{formatDateTime(message.createdAt)}</span>
                    </div>
                    <div className="message-bubble__meta-actions">
                        <IconButton
                            size="small"
                            className="message-bubble__meta-action"
                            onClick={() => onCopyMessage?.(message, message.content)}
                        >
                            <ContentCopyRoundedIcon fontSize="small"/>
                        </IconButton>
                        {isEditableUserMessage ? (
                            <IconButton
                                size="small"
                                className="message-bubble__meta-action"
                                onClick={() => onEditMessage(message)}
                                disabled={busy}
                            >
                                <EditRoundedIcon fontSize="small"/>
                            </IconButton>
                        ) : null}
                    </div>
                </div>

                {isPendingReply ? (
                    <div className="message-bubble__loading">
                        <CircularProgress size={16} thickness={5}/>
                        <span>{message.metadata?.loadingLabel || 'Модель формирует ответ...'}</span>
                    </div>
                ) : (
                    <div className="message-bubble__content">
                        <MessageContent content={displayContent ?? message.content}/>
                        {isTyping ? <span className="message-bubble__typing-cursor" aria-hidden>|</span> : null}
                    </div>
                )}

                {message.metadata ? (
                    <div className="message-bubble__badges">
                        {message.metadata.mode ? (
                            <StatusPill label={mapModeLabel(message.metadata.mode)} tone="neutral"/>
                        ) : null}
                        {message.metadata.chunkMatches ? (
                            <StatusPill label={`Фрагменты: ${message.metadata.chunkMatches}`} tone="neutral"/>
                        ) : null}
                        {message.metadata.usedContextMessages ? (
                            <StatusPill label={`Контекст: ${message.metadata.usedContextMessages}`} tone="neutral"/>
                        ) : null}
                        {message.metadata.selfScore !== undefined && !isNonRateableAssistantMessage(message) ? (
                            <StatusPill label={`Самооценка: ${selfScore.toFixed(3)}`} tone="neutral"/>
                        ) : null}
                    </div>
                ) : null}

                <MessageReferences references={message.metadata?.references || []}/>

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
                                    <ThumbUpAltRoundedIcon fontSize="small"/>
                                ) : (
                                    <ThumbUpAltOutlinedIcon fontSize="small"/>
                                )}
                            </IconButton>
                            <IconButton
                                size="small"
                                className={`feedback-button feedback-button--negative ${currentRating === -1 ? 'feedback-button--active' : ''}`.trim()}
                                disabled={busy}
                                onClick={() => onRateMessage(message.id, -1)}
                            >
                                {currentRating === -1 ? (
                                    <ThumbDownAltRoundedIcon fontSize="small"/>
                                ) : (
                                    <ThumbDownAltOutlinedIcon fontSize="small"/>
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
                                    mobileChatsOpen: mobileChatsOpenProp,
                                    onOpenMobileChatsMenu,
                                    onCloseMobileChatsMenu,
                                    onCreateChat,
                                    onNavigateChat,
                                    onCreateShareLink,
                                    onDeleteChat,
                                    onStopReply,
                                    onSendMessage,
                                    onEditMessage,
                                    onRateMessage,
                                    routeChatMissing = false,
                                }) {
    const [draft, setDraft] = useState('');
    const [editingMessageId, setEditingMessageId] = useState(null);
    const [shareNotice, setShareNotice] = useState(null);
    const [copyNotice, setCopyNotice] = useState(null);
    const [stoppedTypingMap, setStoppedTypingMap] = useState({});
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
    const isApiModel = String(snapshot.model?.kind || 'local').toLowerCase() === 'api';
    const apiModelReady = Boolean(
        snapshot.model?.exists &&
        String(snapshot.model?.externalEndpoint || '').trim() &&
        String(snapshot.model?.externalModelName || '').trim()
    );
    const canChat = isApiModel
        ? apiModelReady
        : Boolean(snapshot.model.exists && snapshot.model.trainedEpochs);
    const trainingLocked = snapshot.training.status === 'training' || snapshot.model.status === 'saving_checkpoint';
    const interactionLocked = busy || trainingLocked;
    const replyPending = pendingAction === 'sendMessage' && pendingReply?.chatId === activeChat?.id;
    const isRegeneratingReply = pendingAction === 'updateChatMessage';
    const mobileChatsOpen = Boolean(mobileChatsOpenProp);

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

        const assistantPendingAlreadySynced = baseMessages.some((message) => (
            message.role === 'assistant' &&
            message?.metadata?.type === 'pending'
        ));

        if (!assistantPendingAlreadySynced) {
            baseMessages.push({
                id: `pending-assistant-${pendingReply.createdAt}`,
                role: 'assistant',
                content: '',
                createdAt: pendingReply.createdAt,
                metadata: {
                    type: 'pending',
                    loadingLabel: 'Модель формирует ответ по чату и базе знаний. Это может занять время.',
                },
            });
        }

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
        const normalizedStore = {...store};
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

    // eslint-disable-next-line no-unused-vars
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
            notices.push({id: 'chat-error', severity: 'error', message: error});
        }
        if (trainingLocked) {
            notices.push({
                id: 'chat-locked',
                severity: 'info',
                message: 'Во время обучения чат временно работает в режиме просмотра.'
            });
        }
        if (replyPending || isRegeneratingReply) {
            notices.push({
                id: 'chat-reply-pending',
                severity: 'info',
                message: editingMessageId
                    ? 'Идет пересборка ответа после редактирования сообщения.'
                    : 'Модель формирует ответ...',
            });
        }
        if (activeChat?.truncated) {
            notices.push({
                id: 'chat-truncated',
                severity: 'info',
                message: `Показаны последние ${formatNumber(activeChat.returnedMessageCount || 0)} сообщений из ${formatNumber(activeChat.totalMessageCount || 0)}.`,
            });
        }
        if (routeChatMissing) {
            notices.push({
                id: 'chat-route-missing',
                severity: 'warning',
                message: 'Этот чат больше не найден или недоступен. Возможно, ссылка устарела или чат был удален.',
            });
        }
        if (shareNotice?.message) {
            notices.push({
                id: shareNotice.id || 'chat-share-notice',
                severity: shareNotice.severity || 'success',
                message: shareNotice.message,
            });
        }
        if (copyNotice?.message) {
            notices.push({
                id: copyNotice.id || 'chat-copy-notice',
                severity: copyNotice.severity || 'success',
                message: copyNotice.message,
            });
        }
        if (editingMessageId) {
            notices.push({
                id: 'chat-editing-message',
                severity: 'info',
                message: 'Вы редактируете сообщение. Новый ответ заменит следующий ответ модели в этой ветке.',
            });
        }
        if (!canChat) {
            notices.push({
                id: 'chat-not-trained',
                severity: 'warning',
                message: 'Сначала подготовьте модель, чтобы чат начал отвечать.'
            });
        }

        const priority = {error: 0, warning: 1, success: 2, info: 3};
        return notices
            .sort((left, right) => (priority[left.severity] ?? 10) - (priority[right.severity] ?? 10))
            .slice(0, 5);
    }, [
        activeChat?.returnedMessageCount,
        activeChat?.totalMessageCount,
        activeChat?.truncated,
        canChat,
        copyNotice,
        editingMessageId,
        error,
        isRegeneratingReply,
        replyPending,
        routeChatMissing,
        shareNotice,
        trainingLocked,
    ]);

    useEffect(() => {
        onNoticesChange?.(floatingNotices);
        return () => {
            onNoticesChange?.([]);
        };
    }, [floatingNotices, onNoticesChange]);

    const handleCreateChat = useCallback(async () => {
        const nextSnapshot = await onCreateChat();
        const nextChatId = nextSnapshot?.activeChat?.id || nextSnapshot?.chats?.[0]?.id || null;
        if (nextChatId) {
            onNavigateChat?.(nextChatId);
        }
        onCloseMobileChatsMenu?.();
    }, [onCloseMobileChatsMenu, onCreateChat, onNavigateChat]);

    const handleShareChat = useCallback(async (chat) => {
        if (!chat || typeof onCreateShareLink !== 'function') {
            return;
        }

        try {
            const share = await onCreateShareLink(chat.id);
            const shareUrl = share?.url || (share?.path ? `${window.location.origin}${share.path}` : '');
            if (!shareUrl) {
                throw new Error('Ссылка приглашения не была создана.');
            }

            await window.navigator.clipboard.writeText(shareUrl);
            setShareNotice({
                id: `chat-share-${chat.id}`,
                severity: 'success',
                message: 'Ссылка на чат скопирована. По ней откроется режим только для чтения.',
            });
            window.setTimeout(() => {
                setShareNotice((current) => (current?.id === `chat-share-${chat.id}` ? null : current));
            }, 3200);
        } catch (shareError) {
            setShareNotice({
                id: `chat-share-error-${chat?.id || 'unknown'}`,
                severity: 'error',
                message: shareError.message || 'Не удалось создать ссылку на чат.',
            });
        }
    }, [onCreateShareLink]);

    const handleSelectChat = useCallback((chatId) => {
        setSelectedChatId(chatId);
        onNavigateChat?.(chatId);
        onCloseMobileChatsMenu?.();
    }, [onCloseMobileChatsMenu, onNavigateChat, setSelectedChatId]);

    const handleCopyMessage = useCallback(async (_message, contentToCopy) => {
        try {
            await window.navigator.clipboard.writeText(String(contentToCopy || ''));
            setCopyNotice({
                id: 'chat-copy-success',
                severity: 'success',
                message: 'Сообщение скопировано.',
            });
            window.setTimeout(() => {
                setCopyNotice((current) => (current?.id === 'chat-copy-success' ? null : current));
            }, 2200);
        } catch (_error) {
            setCopyNotice({
                id: 'chat-copy-error',
                severity: 'error',
                message: 'Не удалось скопировать сообщение.',
            });
        }
    }, []);

    const handleStartEditing = useCallback((message) => {
        if (!message?.id) {
            return;
        }

        setEditingMessageId(message.id);
        setDraft(String(message.content || ''));
    }, []);

    const handleCancelEditing = useCallback(() => {
        setEditingMessageId(null);
        setDraft('');
    }, []);

    const handleStopOutput = useCallback(async () => {
        if ((replyPending || isRegeneratingReply) && activeChat?.id && typeof onStopReply === 'function') {
            await onStopReply(activeChat.id);
            return;
        }

        if (!typingMessageId) {
            return;
        }

        const visibleLength = Math.max(1, Number(typingVisibleLengthRef.current || typingVisibleLength || 0));
        setStoppedTypingMap((current) => ({
            ...current,
            [typingMessageId]: visibleLength,
        }));
        typingStoreRef.current = upsertTypingProgress(typingStoreRef.current, typingMessageId, {
            visibleLength,
            completed: true,
        });
        setTypingMessageId(null);
    }, [activeChat?.id, isRegeneratingReply, onStopReply, replyPending, typingMessageId, typingVisibleLength]);

    const handleSubmitMessage = useCallback(async () => {
        if (!draft.trim() || !activeChat) {
            return;
        }

        const nextValue = draft;
        setDraft('');

        if (editingMessageId && typeof onEditMessage === 'function') {
            const nextSnapshot = await onEditMessage(editingMessageId, nextValue);
            if (nextSnapshot) {
                setEditingMessageId(null);
            } else {
                setDraft(nextValue);
            }
            return;
        }

        const nextSnapshot = await onSendMessage(activeChat.id, nextValue);
        if (!nextSnapshot) {
            setDraft(nextValue);
        }
    }, [activeChat, draft, editingMessageId, onEditMessage, onSendMessage]);

    return (
        <div className="chat-tab">
            <div className="chat-layout">
                <div
                    className={`chat-sidebar-panel chat-sidebar-panel--plain ${mobileChatsOpen ? 'chat-sidebar-panel--open' : ''}`.trim()}
                >
                    <div className="chat-sidebar-panel__inner">
                        <div className="panel-heading">
                            <div>
                                <Typography sx={{textAlign: 'center', marginBottom: '10px'}}
                                            variant="h3">{'Сохраненные чаты'}</Typography>
                            </div>
                            <GlassPanel
                                innerStyle={{padding: 0}}
                                style={{overflow: 'visible !important'}}
                                filterStyle={{borderRadius: '24px'}}
                                className="chat-sidebar-action-shell"
                                innerClassName="chat-sidebar-action-shell__inner"
                            >
                                <Button
                                    variant="contained"
                                    className="action-button chat-sidebar-action-button"
                                    color="inherit"
                                    startIcon={<AddCommentRoundedIcon/>}
                                    onClick={handleCreateChat}
                                    disabled={interactionLocked}
                                    fullWidth
                                >
                                    {'Новый чат'}
                                </Button>
                            </GlassPanel>
                        </div>

                        <div className="chat-list">
                            {snapshot.chats.map((chat) => (
                                <GlassPanel
                                    style={{overflow: 'visible', background: selectedChatId === chat.id ? 'rgb(255 255 255 / 15%)' : ''}}
                                    filterStyle={{borderRadius: '24px'}}
                                    key={chat.id}
                                    className={`chat-list-item-panel ${selectedChatId === chat.id ? 'chat-list-item-panel--active' : ''}`.trim()}
                                    innerClassName="chat-list-item-panel__inner"
                                >
                                    <div
                                        className={`chat-list-item ${selectedChatId === chat.id ? 'chat-list-item--active' : ''}`.trim()}>
                                        <button
                                            type="button"
                                            className="chat-list-item__select"
                                            onClick={() => handleSelectChat(chat.id)}
                                        >
                                            <div className="chat-list-item__head">
                                                <strong>{previewText(chat.title, 30)}</strong>
                                            </div>
                                            <span
                                                className="chat-list-item__preview">{previewText(chat.lastMessagePreview, 34)}</span>
                                        </button>
                                        <div className="chat-list-item__actions">
                                            <IconButton
                                                size="small"
                                                onClick={() => handleShareChat(chat)}
                                                disabled={interactionLocked}
                                            >
                                                <ShareRoundedIcon fontSize="small"/>
                                            </IconButton>
                                            <IconButton
                                                size="small"
                                                onClick={() => onDeleteChat(chat.id)}
                                                disabled={interactionLocked || snapshot.chats.length === 1}
                                            >
                                                <DeleteOutlineRoundedIcon fontSize="small"/>
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
                    aria-label={'Закрыть список чатов'}
                    onClick={() => onCloseMobileChatsMenu?.()}
                />

                <div className="chat-main-panel main-panel--plain">
                    <div className="chat-main-panel__inner">
                        <div className="panel-heading">
                            <div className="chat-main-panel__heading-primary">
                                <GlassPanel className="chat-mobile-menu-shell"
                                            innerClassName="chat-mobile-menu-shell__inner">
                                    <Button
                                        variant="contained"
                                        className="action-button chat-mobile-menu-button"
                                        startIcon={<MenuRoundedIcon/>}
                                        onClick={() => onOpenMobileChatsMenu?.()}
                                    >
                                        {'Чаты'}
                                    </Button>
                                </GlassPanel>
                                <div style={{marginLeft: '30px'}}>
                                    <Typography variant="h3">{activeChat?.title || 'Чат'}</Typography>
                                </div>
                            </div>
                        </div>

                        {error ? <Alert severity="error">{error}</Alert> : null}
                        {trainingLocked ? (
                            <Alert severity="info">
                                {'Во время обучения чат переведен в режим просмотра. Дождитесь завершения или поставьте обучение на паузу.'}
                            </Alert>
                        ) : null}
                        {(replyPending || isRegeneratingReply) ? (
                            <Alert severity="info">
                                {editingMessageId
                                    ? 'Идет пересборка ответа после редактирования сообщения.'
                                    : 'Модель формирует ответ по запросу.'}
                            </Alert>
                        ) : null}
                        {activeChat?.truncated ? (
                            <Alert severity="info">
                                {`Показаны последние ${formatNumber(activeChat.returnedMessageCount || 0)} из ${formatNumber(activeChat.totalMessageCount || 0)} сообщений, чтобы чат не перегружал память.`}
                            </Alert>
                        ) : null}
                        {!canChat ? (
                            <Alert severity="info">
                                {'Сначала подготовьте модель на вкладке обучения, иначе ответы будут недоступны.'}
                            </Alert>
                        ) : null}

                        <div className="message-list" ref={messageListRef} onScroll={handleMessageListScroll}>
                            {visibleMessages.map((message) => {
                                const isTyping = (
                                    message.id === typingMessageId &&
                                    isAssistantReplyMessage(message) &&
                                    supportsTypingAnimation(message)
                                );
                                const stoppedVisibleLength = Number(stoppedTypingMap[message.id] || 0);
                                const displayContent = isTyping
                                    ? message.content.slice(0, Math.min(typingVisibleLength, message.content.length))
                                    : stoppedVisibleLength > 0
                                        ? message.content.slice(0, Math.min(stoppedVisibleLength, message.content.length))
                                        : message.content;

                                return (
                                    <MessageBubble
                                        key={message.id}
                                        message={message}
                                        busy={interactionLocked}
                                        optimisticRating={pendingRatings?.[message.id]}
                                        onRateMessage={onRateMessage}
                                        onCopyMessage={handleCopyMessage}
                                        onEditMessage={message.role === 'user' ? handleStartEditing : null}
                                        displayContent={displayContent}
                                        isTyping={isTyping}
                                    />
                                );
                            })}
                        </div>

                        {editingMessageId ? (
                            <div className="chat-editing-bar">
                                <span className="chat-editing-bar__label">{'Редактирование сообщения'}</span>
                                <Button variant="text" size="small" onClick={handleCancelEditing}>{'Отменить'}</Button>
                            </div>
                        ) : null}

                        <div className="chat-input-row">
                            <GlassPanel
                                innerStyle={{padding: 0}}
                                style={{overflow: 'visible !important'}}
                                filterStyle={{borderRadius: '24px'}}
                                className="chat-input-shell"
                                innerClassName="chat-input-shell__inner"
                            >
                                <TextField
                                    className="text-input"
                                    multiline
                                    minRows={1}
                                    maxRows={6}
                                    value={draft}
                                    disabled={interactionLocked || !canChat}
                                    onChange={(event) => setDraft(event.target.value)}
                                    placeholder={editingMessageId
                                        ? 'Измените текст сообщения и отправьте, чтобы заменить следующий ответ модели.'
                                        : 'Введите сообщение. Модель учтет историю диалога, память чата и базу знаний.'}
                                />
                            </GlassPanel>
                            <div
                                className="chat-send-shell"
                            >
                                {(replyPending || isRegeneratingReply || typingMessageId) ? (
                                    <Button
                                        variant="contained"
                                        className="action-button chat-send-button chat-send-button--stop"
                                        color="inherit"
                                        startIcon={<StopRoundedIcon/>}
                                        onClick={handleStopOutput}
                                        fullWidth
                                    >
                                        {'Стоп'}
                                    </Button>
                                ) : (
                                    <Button
                                        variant="contained"
                                        className="action-button chat-send-button"
                                        color="inherit"
                                        startIcon={<SendRoundedIcon/>}
                                        onClick={handleSubmitMessage}
                                        disabled={interactionLocked || !draft.trim() || !canChat}
                                        fullWidth
                                    >
                                        {editingMessageId ? 'Сохранить' : 'Отправить'}
                                    </Button>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
