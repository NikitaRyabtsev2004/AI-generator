import { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Button,
  CircularProgress,
  IconButton,
  Typography,
} from '@mui/material';
import CheckRoundedIcon from '@mui/icons-material/CheckRounded';
import ContentCopyRoundedIcon from '@mui/icons-material/ContentCopyRounded';
import KeyboardBackspaceRoundedIcon from '@mui/icons-material/KeyboardBackspaceRounded';
import OpenInNewRoundedIcon from '@mui/icons-material/OpenInNewRounded';
import GlassPanel from '../shared/GlassPanel';
import StatusPill from '../shared/StatusPill';
import { fetchSharedChat } from '../../api/studioApi';
import { highlightCode, parseMessageContent, splitTextParagraphs } from '../../utils/chatContent';
import { formatDateTime } from '../../utils/text';

function renderInlineContent(text) {
  const segments = String(text || '').split(/(`[^`\n]+`|\[[^\]]{1,160}\]\(https?:\/\/[^\s)]+\))/gu);
  return segments.map((segment, index) => {
    if (/^`[^`\n]+`$/u.test(segment)) {
      return <code key={`inline-${index}`} className="message-inline-code">{segment.slice(1, -1)}</code>;
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
            {lineIndex < lines.length - 1 ? <br /> : null}
          </span>
        ))}
      </span>
    );
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

    return {
      type: 'paragraph',
      content: paragraph,
    };
  }).filter(Boolean);
}

function MessageTextBlock({ content }) {
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

        return (
          <p key={`${block.content.slice(0, 24)}-${index}`} className="message-text-block">
            {renderInlineContent(block.content)}
          </p>
        );
      })}
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
          <span className="message-code-block__line-count">{highlightedLines.length} lines</span>
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
        part.type === 'code'
          ? <CodeBlock key={`code-${index}`} code={part.content} language={part.language} />
          : <MessageTextBlock key={`text-${index}`} content={part.content} />
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
        <span>Источники</span>
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

function SharedMessageBubble({ message }) {
  const metadata = message.metadata || {};

  return (
    <div className={`message-bubble message-bubble--${message.role}`}>
      <GlassPanel filterStyle={{ borderRadius: '28px' }}>
        <div className="message-bubble__meta">
          <span>{message.role === 'user' ? 'Автор чата' : 'Модель'}</span>
          <span>{formatDateTime(message.createdAt)}</span>
        </div>
        <div className="message-bubble__content">
          <MessageContent content={message.content} />
        </div>
        {metadata ? (
          <div className="message-bubble__badges">
            {metadata.mode ? <StatusPill label={metadata.mode} tone="neutral" /> : null}
            {metadata.selfScore !== undefined ? (
              <StatusPill label={`Самооценка: ${Number(metadata.selfScore || 0).toFixed(3)}`} tone="neutral" />
            ) : null}
          </div>
        ) : null}
        <MessageReferences references={metadata.references || []} />
      </GlassPanel>
    </div>
  );
}

export default function SharedChatView({
  token,
  onReturnHome,
}) {
  const [state, setState] = useState({
    loading: true,
    error: '',
    chat: null,
  });

  useEffect(() => {
    let cancelled = false;
    setState({
      loading: true,
      error: '',
      chat: null,
    });

    fetchSharedChat(token)
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setState({
          loading: false,
          error: payload?.chat ? '' : 'Чат по этой ссылке недоступен.',
          chat: payload?.chat || null,
        });
      })
      .catch((error) => {
        if (cancelled) {
          return;
        }
        setState({
          loading: false,
          error: error.message || 'Не удалось открыть приглашенный чат.',
          chat: null,
        });
      });

    return () => {
      cancelled = true;
    };
  }, [token]);

  return (
    <div className="chat-tab chat-tab--shared">
      <div className="chat-layout chat-layout--shared">
        <div className="chat-main-panel main-panel--plain chat-main-panel--shared">
          <div className="chat-main-panel__inner">
            <div className="panel-heading panel-heading--shared-chat">
              <div className="chat-main-panel__heading-primary chat-main-panel__heading-primary--shared">
                <Button
                  variant="contained"
                  className="hero-model-button"
                  startIcon={<KeyboardBackspaceRoundedIcon />}
                  onClick={onReturnHome}
                >
                  Вернуться в свое окружение
                </Button>
                <div className="shared-chat-heading__copy">
                  <Typography variant="h3">
                    {state.chat?.title || 'Приглашенный чат'}
                  </Typography>
                  <Typography variant="body2" className="muted-text">
                    Режим только для чтения по пригласительной ссылке.
                  </Typography>
                </div>
              </div>
            </div>

            {state.loading ? (
              <div className="shared-chat-view__loading">
                <CircularProgress />
              </div>
            ) : null}

            {!state.loading && state.error ? (
              <Alert severity="warning">{state.error}</Alert>
            ) : null}

            {!state.loading && state.chat ? (
              <div className="message-list message-list--shared">
                {state.chat.messages.map((message) => (
                  <SharedMessageBubble key={message.id} message={message} />
                ))}
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}
