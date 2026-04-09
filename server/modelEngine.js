const crypto = require('crypto');
const path = require('path');
const { Worker } = require('node:worker_threads');
const {
  MODEL_STATUS_FLOW,
  createDefaultKnowledgeState,
  createDefaultModelState,
  createDefaultTrainingState,
} = require('./config');
const {
  disposeRuntime,
  generateText,
  loadRuntime,
  tokenizeForModel,
} = require('./neuralModel');
const { getRuntimeConfig } = require('./runtimeConfig');
const {
  cleanText,
  computeStats,
  createId,
  inferChatTitle,
  previewText,
  splitIntoSentences,
  tokenizeWords,
} = require('./text');

const RETRAINING_KEYS = [
  'sequenceLength',
  'embeddingSize',
  'hiddenSize',
  'recurrentLayers',
  'dropout',
  'learningRate',
  'chunkSize',
  'chunkOverlap',
  'vocabularyLimit',
  'topKChunks',
  'minSimilarity',
];

const TRAINING_LOCKED_STATUSES = new Set(['training', 'syncing_knowledge']);

function nowIso() {
  return new Date().toISOString();
}

function getGeneratorBackend(runtimeConfig) {
  return runtimeConfig.generation?.backend === 'ollama' ? 'ollama' : 'neural';
}

function structuredTrainingText(parts) {
  return cleanText(parts.filter(Boolean).join(' '));
}

function createEmptyChat() {
  const timestamp = nowIso();
  return {
    id: createId('chat'),
    title: 'Новый чат',
    createdAt: timestamp,
    updatedAt: timestamp,
    messages: [
      {
        id: createId('msg'),
        role: 'assistant',
        content:
          'Я готов работать с моделью и базой знаний. После обучения смогу отвечать с учетом корпуса, контекста чата и оценок качества.',
        createdAt: timestamp,
        metadata: {
          type: 'system',
        },
      },
    ],
  };
}

function pushStatus(state, status, phase, message) {
  const entry = {
    id: createId('status'),
    status,
    phase,
    message,
    createdAt: nowIso(),
  };

  state.training.status = status;
  state.training.phase = phase;
  state.training.message = message;
  state.training.updatedAt = entry.createdAt;
  state.training.recentStatuses = [entry, ...(state.training.recentStatuses || [])].slice(0, 32);
}

function ensureChatAvailability(state) {
  if (!Array.isArray(state.chats)) {
    state.chats = [];
  }

  if (!state.chats.length) {
    state.chats.push(createEmptyChat());
  }
}

function summarizeChat(chat) {
  return {
    id: chat.id,
    title: chat.title,
    createdAt: chat.createdAt,
    updatedAt: chat.updatedAt,
    messageCount: chat.messages.length,
    lastMessagePreview: previewText(chat.messages[chat.messages.length - 1]?.content || '', 90),
  };
}

function summarizeSource(source) {
  return {
    id: source.id,
    type: source.type,
    label: source.label,
    url: source.url || null,
    addedAt: source.addedAt,
    stats: source.stats,
    preview: previewText(source.content, 220),
  };
}

function addWeightedTokens(targetMap, tokens, weight = 1) {
  tokens.forEach((token) => {
    targetMap[token] = (targetMap[token] || 0) + weight;
  });
}

function createTermMap(tokens, allowedTerms = null) {
  const map = {};
  tokens.forEach((token) => {
    if (allowedTerms && !allowedTerms.has(token)) {
      return;
    }
    map[token] = (map[token] || 0) + 1;
  });
  return map;
}

function normalizeSpeakerLabel(value) {
  const normalized = cleanText(value).toLowerCase();
  if (['user', 'human', 'client', 'пользователь', 'клиент', 'юзер'].includes(normalized)) {
    return 'user';
  }

  if (['assistant', 'bot', 'ассистент', 'бот', 'ai'].includes(normalized)) {
    return 'assistant';
  }

  return null;
}

function extractDialogueTurns(text) {
  const cleanValue = cleanText(text);
  const speakerPattern = /(Пользователь|Клиент|Юзер|User|Human|Client|Ассистент|Бот|Assistant|Bot|AI)\s*:\s*/giu;
  const matches = Array.from(cleanValue.matchAll(speakerPattern));
  const turns = [];

  matches.forEach((match, index) => {
    const role = normalizeSpeakerLabel(match[1]);
    if (!role) {
      return;
    }

    const contentStart = (match.index || 0) + match[0].length;
    const contentEnd = index + 1 < matches.length ? matches[index + 1].index : cleanValue.length;
    const content = cleanText(cleanValue.slice(contentStart, contentEnd));
    if (!content) {
      return;
    }

    turns.push({ role, content });
  });

  return turns;
}

function extractDialoguePairsFromTurns(turns, ownerId, title, origin = 'source', score = 0.75) {
  const pairs = [];

  for (let index = 0; index < turns.length - 1; index += 1) {
    const userTurn = turns[index];
    const assistantTurn = turns[index + 1];
    if (userTurn.role !== 'user' || assistantTurn.role !== 'assistant') {
      continue;
    }

    pairs.push({
      id: createId('pair'),
      ownerId,
      title,
      origin,
      score,
      promptText: cleanText(userTurn.content),
      responseText: cleanText(assistantTurn.content),
      combinedText: cleanText(`Пользователь: ${userTurn.content}\nАссистент: ${assistantTurn.content}`),
    });
  }

  return pairs;
}

function extractDialoguePairsFromSource(source) {
  return extractDialoguePairsFromTurns(
    extractDialogueTurns(source.content),
    source.id,
    source.label,
    'source',
    0.9
  );
}

function startsWithSpeakerLabel(text) {
  return /^\s*(пользователь|user|assistant|ассистент|бот|bot|client|клиент)\s*:/iu.test(text);
}

function sanitizeReplyText(text) {
  return cleanText(text)
    .replace(/^\s*[-вЂ“вЂ”]\s*/u, '')
    .replace(/^\s*(пользователь|user|assistant|ассистент|бот|bot|client|клиент)\s*:\s*/iu, '')
    .trim();
}

function isEchoReply(userText, replyText) {
  const userTokens = new Set(tokenizeWords(userText));
  const replyTokens = tokenizeWords(replyText);
  if (!userTokens.size || !replyTokens.length) {
    return false;
  }

  let shared = 0;
  replyTokens.forEach((token) => {
    if (userTokens.has(token)) {
      shared += 1;
    }
  });

  return shared / Math.max(replyTokens.length, userTokens.size) >= 0.82;
}

function createChunksForDocument(documentEntry, chunkSize, chunkOverlap, allowedTerms) {
  const sentences = splitIntoSentences(documentEntry.text);
  const sentenceWords = sentences.map((sentence) => tokenizeWords(sentence));
  const chunks = [];

  if (!sentences.length) {
    return chunks;
  }

  let start = 0;
  while (start < sentences.length) {
    let end = start;
    let wordCount = 0;

    while (end < sentences.length && wordCount < chunkSize) {
      wordCount += sentenceWords[end].length;
      end += 1;
    }

    const selectedSentences = sentences.slice(start, end);
    const chunkText = cleanText(selectedSentences.join(' '));
    if (!chunkText || startsWithSpeakerLabel(chunkText)) {
      if (end >= sentences.length) {
        break;
      }
      start = end;
      continue;
    }

    const chunkTokens = tokenizeWords(chunkText);
    const termMap = createTermMap(chunkTokens, allowedTerms);

    chunks.push({
      id: createId('chunk'),
      sourceId: documentEntry.sourceId,
      sourceType: documentEntry.sourceType,
      sourceKind: documentEntry.sourceKind,
      label: documentEntry.label,
      text: chunkText,
      sentences: selectedSentences,
      tokenCount: chunkTokens.length,
      documentLength: chunkTokens.length,
      termMap,
    });

    if (end >= sentences.length) {
      break;
    }

    let overlapWords = 0;
    let overlapSentenceCount = 0;
    for (let index = end - 1; index > start && overlapWords < chunkOverlap; index -= 1) {
      overlapWords += sentenceWords[index].length;
      overlapSentenceCount += 1;
    }

    start = Math.max(start + 1, end - overlapSentenceCount);
  }

  return chunks;
}

function buildSourceDocuments(source) {
  return [
    {
      sourceId: source.id,
      sourceType: source.type,
      sourceKind: 'knowledge',
      label: source.label,
      text: source.content,
    },
  ];
}

function createCorpusSignature(state, artifacts) {
  const signaturePayload = JSON.stringify({
    sourceIds: state.sources.map((source) => source.id),
    sourceSizes: state.sources.map((source) => source.stats?.tokenCount || 0),
    replyPairCount: artifacts.replyMemories.length,
    chunkCount: artifacts.chunks.length,
    tokenCount: artifacts.tokenCount,
    positiveFeedbackCount: artifacts.positiveFeedbackCount,
    negativeFeedbackCount: artifacts.negativeFeedbackCount,
    settings: {
      sequenceLength: state.settings.training.sequenceLength,
      embeddingSize: state.settings.training.embeddingSize,
      hiddenSize: state.settings.training.hiddenSize,
      recurrentLayers: state.settings.training.recurrentLayers,
      dropout: state.settings.training.dropout,
      learningRate: state.settings.training.learningRate,
      chunkSize: state.settings.training.chunkSize,
      chunkOverlap: state.settings.training.chunkOverlap,
      vocabularyLimit: state.settings.training.vocabularyLimit,
      topKChunks: state.settings.training.topKChunks,
      minSimilarity: state.settings.training.minSimilarity,
    },
    runtime: getRuntimeConfig(),
  });

  return crypto.createHash('sha1').update(signaturePayload).digest('hex');
}

function buildBm25Index(entries) {
  const documentFrequency = {};
  let totalLength = 0;

  entries.forEach((entry) => {
    totalLength += entry.documentLength || 0;
    Object.keys(entry.termMap).forEach((term) => {
      documentFrequency[term] = (documentFrequency[term] || 0) + 1;
    });
  });

  const documentCount = entries.length;
  const averageDocumentLength = documentCount ? totalLength / documentCount : 0;
  const idf = {};

  Object.entries(documentFrequency).forEach(([term, frequency]) => {
    idf[term] = Math.log(1 + (documentCount - frequency + 0.5) / (frequency + 0.5));
  });

  return {
    documentCount,
    averageDocumentLength,
    documentFrequency,
    idf,
  };
}

function computeBm25Score(queryMap, entry, bm25, runtimeConfig) {
  const k1 = runtimeConfig.retrieval.bm25k1;
  const b = runtimeConfig.retrieval.bm25b;
  const avgLength = Math.max(bm25.averageDocumentLength || 0, 1);
  let score = 0;

  Object.entries(queryMap).forEach(([term, queryWeight]) => {
    const tf = entry.termMap[term] || 0;
    if (!tf) {
      return;
    }

    const idf = bm25.idf[term] || 0;
    const denominator = tf + k1 * (1 - b + b * ((entry.documentLength || 0) / avgLength));
    score += queryWeight * idf * ((tf * (k1 + 1)) / denominator);
  });

  return score;
}

function scoreSentenceAgainstQuery(sentence, queryMap) {
  const tokens = tokenizeWords(sentence);
  let score = 0;
  tokens.forEach((token) => {
    score += queryMap[token] || 0;
  });
  if (sentence.trim().endsWith('?')) {
    score *= 0.7;
  }
  return score;
}

function isAllowedKnowledgeSentence(sentence) {
  const cleanSentence = sanitizeReplyText(sentence);
  const tokens = tokenizeWords(cleanSentence);

  if (!cleanSentence || startsWithSpeakerLabel(cleanSentence)) {
    return false;
  }

  if (tokens.length < 6) {
    return false;
  }

  if (/\?\s*$/u.test(cleanSentence)) {
    return false;
  }

  if (/^(ответить|reply|главная|меню|голосовать|результаты|сезон\s+\d+)/iu.test(cleanSentence)) {
    return false;
  }

  return true;
}

function selectKnowledgeSentences(chunks, queryMap, maxSentences) {
  const candidates = [];
  const used = new Set();

  chunks.forEach((chunk) => {
    chunk.sentences.forEach((sentence) => {
      const cleanSentence = sanitizeReplyText(sentence);
      if (!isAllowedKnowledgeSentence(cleanSentence)) {
        return;
      }

      const normalized = cleanSentence.toLowerCase();
      if (used.has(normalized)) {
        return;
      }

      const score = scoreSentenceAgainstQuery(cleanSentence, queryMap) + (chunk.score || 0);
      if (score <= 0) {
        return;
      }

      used.add(normalized);
      candidates.push({ sentence: cleanSentence, score });
    });
  });

  return candidates
    .sort((left, right) => right.score - left.score)
    .slice(0, maxSentences)
    .map((candidate) => candidate.sentence);
}

function isPositiveMessageFeedback(metadata, runtimeConfig) {
  if (!metadata) {
    return false;
  }

  if (metadata.userRating === 1) {
    return true;
  }

  return metadata.selfScore >= runtimeConfig.feedback.autoAcceptThreshold && metadata.userRating !== -1;
}

function isNegativeMessageFeedback(metadata) {
  return metadata?.userRating === -1;
}

function extractDialoguePairsFromChat(chat, runtimeConfig) {
  const pairs = [];

  for (let index = 1; index < chat.messages.length; index += 1) {
    const currentMessage = chat.messages[index];
    const previousMessage = chat.messages[index - 1];
    if (previousMessage.role !== 'user' || currentMessage.role !== 'assistant') {
      continue;
    }

    if (currentMessage.metadata?.type === 'system') {
      continue;
    }

    if (!isPositiveMessageFeedback(currentMessage.metadata, runtimeConfig)) {
      continue;
    }

    pairs.push({
      id: createId('pair'),
      ownerId: chat.id,
      title: chat.title,
      origin: 'chat_feedback',
      score: currentMessage.metadata?.userRating === 1 ? 1 : currentMessage.metadata?.selfScore || 0.7,
      promptText: cleanText(previousMessage.content),
      responseText: cleanText(currentMessage.content),
      combinedText: cleanText(`Пользователь: ${previousMessage.content}\nАссистент: ${currentMessage.content}`),
    });
  }

  return pairs;
}

function collectBlockedReplies(state) {
  const blocked = new Set();
  state.chats.forEach((chat) => {
    chat.messages.forEach((message) => {
      if (message.role !== 'assistant') {
        return;
      }
      if (isNegativeMessageFeedback(message.metadata)) {
        blocked.add(cleanText(message.content).toLowerCase());
      }
    });
  });
  return blocked;
}

function buildTrainingTexts(state, replyMemories, runtimeConfig) {
  const texts = [];

  state.sources.forEach((source) => {
    const cleanSourceText = cleanText(source.content);
    if (cleanSourceText) {
      texts.push(structuredTrainingText(['__bos__', cleanSourceText, '__eos__']));
    }
  });

  state.sources.forEach((source) => {
    extractDialoguePairsFromSource(source).forEach((pair) => {
      texts.push(
        structuredTrainingText([
          '__bos__',
          '__usr__',
          pair.promptText,
          '__sep__',
          '__asst__',
          pair.responseText,
          '__eos__',
        ])
      );
    });
  });

  replyMemories.forEach((pair) => {
    const weight = pair.origin === 'chat_feedback'
      ? Math.max(1, runtimeConfig.feedback.positiveReplayWeight)
      : 1;
    for (let repeatIndex = 0; repeatIndex < weight; repeatIndex += 1) {
      texts.push(
        structuredTrainingText([
          '__bos__',
          '__usr__',
          pair.promptText,
          '__sep__',
          '__asst__',
          pair.responseText,
          '__eos__',
        ])
      );
    }
  });

  return texts.filter(Boolean);
}

function buildKnowledgeArtifacts(state) {
  const runtimeConfig = getRuntimeConfig();
  const sourceReplyMemories = state.sources.flatMap((source) => extractDialoguePairsFromSource(source));
  const chatReplyMemories = state.chats.flatMap((chat) => extractDialoguePairsFromChat(chat, runtimeConfig));
  const blockedReplies = collectBlockedReplies(state);

  const replyMemories = [...sourceReplyMemories, ...chatReplyMemories]
    .map((pair) => ({
      ...pair,
      responseText: sanitizeReplyText(pair.responseText),
    }))
    .filter((pair) => pair.promptText && pair.responseText)
    .filter((pair) => !blockedReplies.has(pair.responseText.toLowerCase()))
    .filter((pair) => !isEchoReply(pair.promptText, pair.responseText));

  const documents = state.sources.flatMap((source) => buildSourceDocuments(source));
  const rawVocabulary = {};
  let tokenCount = 0;
  let sentenceCount = 0;

  documents.forEach((documentEntry) => {
    const tokens = tokenizeWords(documentEntry.text);
    const sentences = splitIntoSentences(documentEntry.text);
    tokenCount += tokens.length;
    sentenceCount += sentences.length;
    tokens.forEach((token) => {
      rawVocabulary[token] = (rawVocabulary[token] || 0) + 1;
    });
  });

  replyMemories.forEach((memory) => {
    tokenizeWords(memory.promptText).forEach((token) => {
      rawVocabulary[token] = (rawVocabulary[token] || 0) + 1;
    });
    tokenizeWords(memory.responseText).forEach((token) => {
      rawVocabulary[token] = (rawVocabulary[token] || 0) + 1;
    });
  });

  const vocabularyEntries = Object.entries(rawVocabulary)
    .sort((left, right) => right[1] - left[1])
    .slice(0, state.settings.training.vocabularyLimit);
  const allowedTerms = new Set(vocabularyEntries.map(([term]) => term));
  const vocabulary = Object.fromEntries(vocabularyEntries);

  const chunks = documents.flatMap((documentEntry) =>
    createChunksForDocument(
      documentEntry,
      state.settings.training.chunkSize,
      state.settings.training.chunkOverlap,
      allowedTerms
    )
  );

  const knowledgeBm25 = buildBm25Index(chunks);
  const indexedReplyMemories = replyMemories.map((memory) => {
    const promptTokens = tokenizeWords(memory.promptText);
    return {
      ...memory,
      termMap: createTermMap(promptTokens, allowedTerms),
      documentLength: promptTokens.length,
    };
  });
  const replyBm25 = buildBm25Index(indexedReplyMemories);

  const topTerms = vocabularyEntries.slice(0, 12).map(([term, count]) => ({ term, count }));
  const positiveFeedbackCount = state.chats
    .flatMap((chat) => chat.messages)
    .filter((message) => message.metadata?.userRating === 1)
    .length;
  const negativeFeedbackCount = state.chats
    .flatMap((chat) => chat.messages)
    .filter((message) => message.metadata?.userRating === -1)
    .length;
  const trainingTexts = buildTrainingTexts(state, indexedReplyMemories, runtimeConfig);
  const tokenizerTokenCount = trainingTexts.reduce(
    (sum, text) => sum + tokenizeForModel(text).length,
    0
  );

  const artifacts = {
    chunks,
    replyMemories: indexedReplyMemories,
    blockedReplies,
    vocabulary,
    bm25: {
      knowledge: knowledgeBm25,
      reply: replyBm25,
    },
    topTerms,
    tokenCount,
    sentenceCount,
    replyPairCount: indexedReplyMemories.length,
    trainingTexts,
    tokenizerTokenCount,
    positiveFeedbackCount,
    negativeFeedbackCount,
  };

  return {
    ...artifacts,
    corpusSignature: createCorpusSignature(state, artifacts),
  };
}

function selectContextMessages(chat, userMessage, runtimeConfig) {
  const messages = chat.messages
    .filter((message) => message.metadata?.type !== 'system')
    .filter((message) => cleanText(message.content))
    .map((message, index) => ({
      ...message,
      index,
    }));

  const recentMessages = messages.slice(-10);
  const queryTokens = new Set(tokenizeWords(userMessage));
  const scored = messages
    .filter((message) => message.content !== userMessage)
    .map((message) => {
      const tokens = tokenizeWords(message.content);
      let lexicalScore = 0;
      tokens.forEach((token) => {
        if (queryTokens.has(token)) {
          lexicalScore += 1;
        }
      });

      return {
        message,
        score: lexicalScore + (message.role === 'assistant' ? 0.15 : 0.35),
      };
    })
    .sort((left, right) => right.score - left.score)
    .slice(0, 8)
    .map((entry) => entry.message);

  const merged = new Map();
  [...recentMessages, ...scored].forEach((message) => {
    merged.set(message.id, message);
  });

  return [...merged.values()]
    .sort((left, right) => left.createdAt.localeCompare(right.createdAt))
    .slice(-runtimeConfig.context.maxMessages);
}

function buildWeightedQueryMap(userMessage, chat, runtimeConfig) {
  const queryMap = {};
  addWeightedTokens(queryMap, tokenizeWords(userMessage), 3);

  const contextMessages = selectContextMessages(chat, userMessage, runtimeConfig);
  const reversedContext = [...contextMessages].reverse();
  reversedContext.forEach((message, index) => {
    const roleWeight = message.role === 'assistant'
      ? runtimeConfig.context.assistantWeight
      : runtimeConfig.context.userWeight;
    const weight = roleWeight * Math.pow(runtimeConfig.context.decay, index);
    addWeightedTokens(queryMap, tokenizeWords(message.content), weight);
  });

  return {
    queryMap,
    contextMessages,
  };
}

function buildPromptForGeneration(userMessage, contextMessages, chunkCandidates, replyCandidates) {
  const promptParts = ['__bos__', '__ctx__'];

  chunkCandidates.slice(0, 4).forEach((chunk) => {
    promptParts.push(chunk.text);
    promptParts.push('__sep__');
  });

  replyCandidates.slice(0, 2).forEach((pair) => {
    promptParts.push('__usr__');
    promptParts.push(pair.promptText);
    promptParts.push('__sep__');
    promptParts.push('__asst__');
    promptParts.push(pair.responseText);
    promptParts.push('__sep__');
  });

  contextMessages.slice(-8).forEach((message) => {
    promptParts.push(message.role === 'assistant' ? '__asst__' : '__usr__');
    promptParts.push(message.content);
    promptParts.push('__sep__');
  });

  promptParts.push('__usr__');
  promptParts.push(userMessage);
  promptParts.push('__sep__');
  promptParts.push('__asst__');

  return structuredTrainingText(promptParts);
}

function computeGroundedOverlap(replyTokens, candidateTexts) {
  if (!replyTokens.length || !candidateTexts.length) {
    return 0;
  }

  const groundedTokens = new Set(candidateTexts.flatMap((text) => tokenizeWords(text)));
  if (!groundedTokens.size) {
    return 0;
  }

  let shared = 0;
  replyTokens.forEach((token) => {
    if (groundedTokens.has(token)) {
      shared += 1;
    }
  });

  return shared / replyTokens.length;
}

function computeReplySimilarity(leftText, rightText) {
  const leftTokens = new Set(tokenizeWords(leftText));
  const rightTokens = new Set(tokenizeWords(rightText));

  if (!leftTokens.size || !rightTokens.size) {
    return 0;
  }

  let shared = 0;
  leftTokens.forEach((token) => {
    if (rightTokens.has(token)) {
      shared += 1;
    }
  });

  return (2 * shared) / (leftTokens.size + rightTokens.size);
}

function computeRecentReplyPenalty(replyText, recentAssistantReplies = []) {
  const maxSimilarity = recentAssistantReplies.reduce((maxValue, recentReply) => (
    Math.max(maxValue, computeReplySimilarity(replyText, recentReply))
  ), 0);

  if (maxSimilarity >= 0.96) {
    return 0.42;
  }

  if (maxSimilarity >= 0.88) {
    return 0.24;
  }

  if (maxSimilarity >= 0.78) {
    return 0.12;
  }

  return 0;
}

function hasLoopingPattern(tokens) {
  if (tokens.length < 6) {
    return false;
  }

  const lastThree = tokens.slice(-3).join(' ');
  const previousThree = tokens.slice(-6, -3).join(' ');
  return lastThree === previousThree;
}

function computeSelfScore({
  userMessage,
  replyText,
  chunkCandidates,
  replyCandidates,
  contextMessages,
  recentAssistantReplies = [],
}) {
  const cleanReply = sanitizeReplyText(replyText);
  const replyTokens = tokenizeWords(cleanReply);
  if (!cleanReply || !replyTokens.length) {
    return 0;
  }

  const groundedTexts = [
    ...chunkCandidates.map((chunk) => chunk.text),
    ...replyCandidates.map((pair) => pair.responseText),
    ...contextMessages.map((message) => message.content),
  ];

  const groundedOverlap = computeGroundedOverlap(replyTokens, groundedTexts);
  const lengthScore = Math.min(replyTokens.length / 18, 1);
  const echoPenalty = isEchoReply(userMessage, cleanReply) ? 0.55 : 0;
  const loopPenalty = hasLoopingPattern(replyTokens) ? 0.4 : 0;
  const questionPenalty = cleanReply.endsWith('?') ? 0.12 : 0;
  const repeatPenalty = computeRecentReplyPenalty(cleanReply, recentAssistantReplies);

  return Math.max(
    0,
    Math.min(
      1,
      0.28 + groundedOverlap * 0.5 + lengthScore * 0.22
      - echoPenalty - loopPenalty - questionPenalty - repeatPenalty
    )
  );
}

function chooseBestReplyCandidate({
  userMessage,
  neuralReply,
  bestReply,
  knowledgeText,
  chunkCandidates,
  replyCandidates,
  contextMessages,
  recentAssistantReplies = [],
}) {
  const candidates = [];

  if (bestReply) {
    const content = sanitizeReplyText(bestReply.responseText);
    candidates.push({
      mode: 'reply_memory',
      content,
      score: computeSelfScore({
        userMessage,
        replyText: content,
        chunkCandidates,
        replyCandidates,
        contextMessages,
        recentAssistantReplies,
      }) + bestReply.score * 0.08,
    });
  }

  if (knowledgeText) {
    candidates.push({
      mode: 'knowledge',
      content: cleanText(knowledgeText),
      score: computeSelfScore({
        userMessage,
        replyText: knowledgeText,
        chunkCandidates,
        replyCandidates,
        contextMessages,
        recentAssistantReplies,
      }),
    });
  }

  if (neuralReply) {
    candidates.push({
      mode: 'neural',
      content: neuralReply,
      score: computeSelfScore({
        userMessage,
        replyText: neuralReply,
        chunkCandidates,
        replyCandidates,
        contextMessages,
        recentAssistantReplies,
      }) + 0.04,
    });
  }

  const filtered = candidates
    .filter((candidate) => candidate.content)
    .filter((candidate) => !isEchoReply(userMessage, candidate.content))
    .filter((candidate) => computeRecentReplyPenalty(candidate.content, recentAssistantReplies) < 0.42)
    .sort((left, right) => right.score - left.score);

  return filtered[0] || null;
}

function findMessageById(state, messageId) {
  for (const chat of state.chats) {
    for (let index = 0; index < chat.messages.length; index += 1) {
      if (chat.messages[index].id === messageId) {
        return {
          chat,
          message: chat.messages[index],
          messageIndex: index,
        };
      }
    }
  }

  return null;
}

function updateModelMetricsFromArtifacts(state, artifacts) {
  state.model.engine = createDefaultModelState().engine;
  state.knowledge.chunks = artifacts.chunks;
  state.knowledge.replyMemories = artifacts.replyMemories;
  state.knowledge.vocabulary = artifacts.vocabulary;
  state.knowledge.bm25 = artifacts.bm25;
  state.knowledge.blockedReplies = Array.from(artifacts.blockedReplies || []);

  state.model.sourceCount = state.sources.length;
  state.model.chatCount = state.chats.length;
  state.model.replyPairCount = artifacts.replyPairCount;
  state.model.chunkCount = artifacts.chunks.length;
  state.model.tokenCount = artifacts.tokenCount;
  state.model.sentenceCount = artifacts.sentenceCount;
  state.model.vocabularySize = Object.keys(artifacts.vocabulary).length;
  state.model.topTerms = artifacts.topTerms;
  state.model.positiveFeedbackCount = artifacts.positiveFeedbackCount;
  state.model.negativeFeedbackCount = artifacts.negativeFeedbackCount;
}

function resetTrainedModelState(state) {
  state.model.trainedEpochs = 0;
  state.model.targetEpochs = 0;
  state.model.batchesProcessed = 0;
  state.model.lastLoss = null;
  state.model.averageLoss = null;
  state.model.perplexity = null;
  state.model.lastTrainingAt = null;
  state.model.corpusSignature = null;
  state.model.parameterCount = 0;
  state.model.averageSelfScore = null;
  state.training.history = [];
  state.training.progress = {
    currentEpoch: 0,
    totalEpochs: 0,
    currentBatch: 0,
    totalBatches: 0,
    percent: 0,
  };
  state.knowledge.languageModel = createDefaultKnowledgeState().languageModel;
}

function isTrainingLocked(state) {
  return (
    TRAINING_LOCKED_STATUSES.has(state.model.lifecycle) ||
    TRAINING_LOCKED_STATUSES.has(state.training.status) ||
    state.model.status === 'saving_checkpoint'
  );
}

function assertTrainingUnlocked(state, actionLabel) {
  if (!isTrainingLocked(state)) {
    return;
  }

  throw new Error(
    `${actionLabel} недоступно, пока идет обучение или сохранение чекпоинта. Дождитесь завершения или поставьте обучение на паузу.`
  );
}

function hasRetrainingImpact(previousSettings, nextSettings) {
  return RETRAINING_KEYS.some((key) => previousSettings.training?.[key] !== nextSettings.training?.[key]);
}

function buildDashboard(state, chatId = null) {
  ensureChatAvailability(state);
  const runtimeConfig = getRuntimeConfig();
  const selectedChat = state.chats.find((chat) => chat.id === chatId) || state.chats[0] || null;
  const ratedMessages = state.chats
    .flatMap((chat) => chat.messages)
    .filter((message) => message.role === 'assistant' && message.metadata?.userRating)
    .length;

  return {
    settings: state.settings,
    model: state.model,
    training: {
      ...state.training,
      availableStatuses: MODEL_STATUS_FLOW,
    },
    runtime: {
      contextStrategy: runtimeConfig.context.strategy,
      generatorBackend: getGeneratorBackend(runtimeConfig),
      storage: state.model.storage,
      ratedMessages,
    },
    sources: state.sources.map(summarizeSource),
    chats: state.chats.map(summarizeChat),
    activeChat: selectedChat,
  };
}

function createModelEngine({ getState, updateState }) {
  let activeTrainingPromise = null;
  let activeTrainingWorker = null;
  let neuralRuntime = null;

  async function disposeNeuralRuntime() {
    disposeRuntime(neuralRuntime);
    neuralRuntime = null;
  }

  async function ensureRuntimeLoaded(state) {
    if (neuralRuntime?.model) {
      return neuralRuntime;
    }

    if (!state.knowledge.languageModel?.checkpointReady) {
      return null;
    }

    neuralRuntime = await loadRuntime({
      storage: state.model.storage,
      settings: state.settings,
    });

    return neuralRuntime;
  }

  async function rebuildKnowledge(state) {
    const artifacts = buildKnowledgeArtifacts(state);
    updateModelMetricsFromArtifacts(state, artifacts);
    return artifacts;
  }

  async function ensureModelExists() {
    await updateState(async (state) => {
      ensureChatAvailability(state);
      if (!state.model.exists) {
        state.model = createDefaultModelState();
        state.training = createDefaultTrainingState();
        state.knowledge = createDefaultKnowledgeState();
      }

      state.model.exists = true;
      state.model.lifecycle = 'ready_for_training';
      state.model.status = 'ready';
      state.model.configSnapshot = structuredClone(state.settings);
      pushStatus(state, 'idle', 'model_ready', 'Серверная нейросеть подготовлена к обучению.');
      await rebuildKnowledge(state);
    });
  }

  function buildTrainingManifest(state, artifacts) {
    return {
      version: 1,
      corpusSignature: artifacts.corpusSignature,
      modelSettings: {
        sequenceLength: state.settings.training.sequenceLength,
        embeddingSize: state.settings.training.embeddingSize,
        hiddenSize: state.settings.training.hiddenSize,
        recurrentLayers: state.settings.training.recurrentLayers,
        dropout: state.settings.training.dropout,
        learningRate: state.settings.training.learningRate,
      },
      vocabularySize: 0,
      parameterCount: 0,
      trainedEpochs: state.model.trainedEpochs,
      trainingSequenceCount: state.model.trainingItemCount,
      corpusTokenCount: artifacts.tokenizerTokenCount,
      feedbackExampleCount: artifacts.positiveFeedbackCount,
      savedAt: nowIso(),
    };
  }

  async function runTrainingJob() {
    if (activeTrainingPromise) {
      return activeTrainingPromise;
    }

    activeTrainingPromise = (async () => {
      const initialState = getState();
      const initialArtifacts = buildKnowledgeArtifacts(initialState);
      const trainingTexts = initialArtifacts.trainingTexts;

      if (!trainingTexts.length) {
        throw new Error('Для обучения не хватает данных. Добавьте источники или диалоговые пары.');
      }

      const sameArchitecture =
        initialState.knowledge.languageModel?.checkpointReady &&
        initialState.model.corpusSignature === initialArtifacts.corpusSignature &&
        initialState.model.configSnapshot?.training?.sequenceLength === initialState.settings.training.sequenceLength &&
        initialState.model.configSnapshot?.training?.embeddingSize === initialState.settings.training.embeddingSize &&
        initialState.model.configSnapshot?.training?.hiddenSize === initialState.settings.training.hiddenSize &&
        initialState.model.configSnapshot?.training?.recurrentLayers === initialState.settings.training.recurrentLayers;

      await disposeNeuralRuntime();

      await updateState(async (state) => {
        state.model.exists = true;
        state.model.lifecycle = 'training';
        state.model.status = 'training';
        state.model.configSnapshot = structuredClone(state.settings);
        state.training = {
          ...state.training,
          ...createDefaultTrainingState(),
          status: 'training',
          phase: 'building_tokenizer',
          message: 'Сервер собирает токенизатор, обучающие последовательности и готовит background worker.',
          startedAt: nowIso(),
          updatedAt: nowIso(),
          requestedStop: false,
        };
        pushStatus(state, 'training', 'building_tokenizer', 'Подготовка корпуса и запуск отдельного процесса обучения.');
      });

      const denominatorBase = Math.max(initialState.settings.training.epochs, 1);
      const workerPath = path.join(__dirname, 'trainingWorker.js');
      const manifest = buildTrainingManifest(initialState, initialArtifacts);

      await new Promise((resolve, reject) => {
        const worker = new Worker(workerPath, {
          workerData: {
            settings: initialState.settings,
            trainingTexts,
            storage: initialState.model.storage,
            resumeFromCheckpoint: Boolean(sameArchitecture),
            manifest,
            tokenizerTokenCount: initialArtifacts.tokenizerTokenCount,
            positiveFeedbackCount: initialArtifacts.positiveFeedbackCount,
          },
        });

        activeTrainingWorker = worker;
        let finished = false;
        let workerQueue = Promise.resolve();
        let knownBatchesPerEpoch = 1;

        const finalizeFailure = (error) => {
          if (finished) {
            return;
          }

          finished = true;
          activeTrainingWorker = null;
          reject(error);
        };

        const finalizeSuccess = () => {
          if (finished) {
            return;
          }

          finished = true;
          activeTrainingWorker = null;
          resolve();
        };

        const enqueueWorkerUpdate = (task) => {
          workerQueue = workerQueue.then(task).catch(finalizeFailure);
        };

        worker.on('message', (message) => {
          if (!message?.type || finished) {
            return;
          }

          if (message.type === 'error') {
            finalizeFailure(new Error(message.error?.message || 'Training worker failed.'));
            return;
          }

          if (message.type === 'prepared') {
            enqueueWorkerUpdate(async () => {
              knownBatchesPerEpoch = Math.max(message.batchesPerEpoch || 1, 1);
              await updateState(async (state) => {
                state.model.trainingItemCount = message.datasetSampleCount;
                state.model.batchesPerEpoch = message.batchesPerEpoch;
                state.model.parameterCount = message.parameterCount;
                state.model.targetEpochs = state.settings.training.epochs;
                state.training.phase = 'training_batches';
                state.training.message = `Подготовлено ${message.datasetSampleCount} sequence-window примеров, ${message.batchesPerEpoch} батчей на эпоху и ${message.tokenizerVocabularySize} токенов словаря.`;
                pushStatus(
                  state,
                  'training',
                  'training_batches',
                  `Нейросеть обучается в отдельном worker: ${message.parameterCount} параметров, ${message.tokenizerVocabularySize} токенов, ${message.datasetSampleCount} примеров.`
                );
              });
            });
            return;
          }

          if (message.type === 'batch') {
            enqueueWorkerUpdate(async () => {
              const totalBatches = Math.max(knownBatchesPerEpoch * denominatorBase, 1);
              await updateState(async (state) => {
                state.training.status = 'training';
                state.training.phase = 'training_batches';
                state.training.message = `Обучение идет: эпоха ${message.epoch}/${state.settings.training.epochs}, батч ${message.batch}/${message.batchesThisEpoch}.`;
                state.training.progress = {
                  currentEpoch: message.epoch,
                  totalEpochs: state.settings.training.epochs,
                  currentBatch: message.batch,
                  totalBatches: message.batchesThisEpoch,
                  percent: Number(((message.processedBatches / totalBatches) * 100).toFixed(1)),
                };
                state.training.history = [...state.training.history, message.historyEntry].slice(-state.settings.training.maxHistoryPoints);
                state.training.updatedAt = nowIso();
                state.model.lastLoss = message.lastLoss;
                state.model.averageLoss = message.averageLoss;
                state.model.perplexity = message.averageLoss ? Number(Math.exp(message.averageLoss).toFixed(4)) : null;
                state.model.batchesProcessed = message.processedBatches;
              }, { persist: false });
            });
            return;
          }

          if (message.type === 'checkpointing') {
            enqueueWorkerUpdate(async () => {
              await updateState(async (state) => {
                state.model.lifecycle = 'syncing_knowledge';
                state.model.status = 'saving_checkpoint';
                state.training.phase = 'saving_checkpoint';
                state.training.message = 'Обучение завершено, сервер сохраняет веса модели и токенизатор.';
                pushStatus(
                  state,
                  'syncing_knowledge',
                  'saving_checkpoint',
                  'Сохранение нейросетевого чекпоинта и артефактов retriever.'
                );
              }, { persist: false });
            });
            return;
          }

          if (message.type === 'done') {
            enqueueWorkerUpdate(async () => {
              await disposeNeuralRuntime();
              await updateState(async (state) => {
                const currentArtifacts = await rebuildKnowledge(state);
                const corpusChanged = currentArtifacts.corpusSignature !== initialArtifacts.corpusSignature;

                state.model.exists = true;
                state.model.lastLoss = message.trainingResult.lastLoss;
                state.model.averageLoss = message.trainingResult.averageLoss;
                state.model.perplexity = message.trainingResult.perplexity;
                state.model.lastTrainingAt = nowIso();
                state.model.batchesProcessed = message.trainingResult.processedBatches;
                state.model.targetEpochs = state.settings.training.epochs;
                state.model.parameterCount = message.parameterCount;

                if (corpusChanged) {
                  resetTrainedModelState(state);
                  state.model.exists = true;
                  state.model.lifecycle = 'ready_for_training';
                  state.model.status = 'ready';
                  state.training.status = 'idle';
                  state.training.phase = 'training_invalidated';
                  state.training.message = 'Во время обучения изменился корпус или настройки. Чекпоинт не принят, модель нужно обучить заново.';
                  pushStatus(state, 'idle', 'training_invalidated', state.training.message);
                  return;
                }

                state.model.trainedEpochs = message.trainingResult.completedEpochs;
                state.model.corpusSignature = currentArtifacts.corpusSignature;
                state.knowledge.languageModel = message.languageModel;
                state.model.lifecycle = message.trainingResult.stopRequested ? 'paused' : 'trained';
                state.model.status = 'ready';
                state.training.status = message.trainingResult.stopRequested ? 'paused' : 'completed';
                state.training.phase = message.trainingResult.stopRequested ? 'paused_by_user' : 'ready_for_chat';
                state.training.message = message.trainingResult.stopRequested
                  ? 'Обучение остановлено пользователем. Последний чекпоинт сохранен.'
                  : 'Обучение завершено. Нейросеть, токенизатор и индекс знаний сохранены на сервере.';
                state.training.progress = {
                  currentEpoch: message.trainingResult.completedEpochs,
                  totalEpochs: state.settings.training.epochs,
                  currentBatch: 0,
                  totalBatches: knownBatchesPerEpoch,
                  percent: message.trainingResult.stopRequested ? state.training.progress.percent : 100,
                };
                pushStatus(
                  state,
                  message.trainingResult.stopRequested ? 'paused' : 'completed',
                  message.trainingResult.stopRequested ? 'paused_by_user' : 'ready_for_chat',
                  state.training.message
                );
              });
              finalizeSuccess();
            });
          }
        });

        worker.on('error', finalizeFailure);
        worker.on('exit', (code) => {
          if (finished || code === 0) {
            return;
          }

          finalizeFailure(new Error(`Training worker stopped with code ${code}.`));
        });
      });
    })().finally(() => {
      activeTrainingPromise = null;
      activeTrainingWorker = null;
    });

    return activeTrainingPromise;
  }

  async function renderReply({ state, chat, userMessage }) {
    const runtimeConfig = getRuntimeConfig();
    const { queryMap, contextMessages } = buildWeightedQueryMap(userMessage, chat, runtimeConfig);
    const recentAssistantReplies = contextMessages
      .filter((message) => message.role === 'assistant')
      .slice(-4)
      .map((message) => sanitizeReplyText(message.content))
      .filter(Boolean);
    const blockedReplies = new Set((state.knowledge.blockedReplies || []).map((value) => value.toLowerCase()));

    const replyCandidates = (state.knowledge.replyMemories || [])
      .map((memory) => ({
        ...memory,
        score:
          computeBm25Score(queryMap, memory, state.knowledge.bm25.reply, runtimeConfig) *
          runtimeConfig.retrieval.assistantReplyBoost *
          (1 + (memory.score || 0) * 0.15),
      }))
      .filter((memory) => memory.score >= state.settings.training.minSimilarity)
      .filter((memory) => !blockedReplies.has(memory.responseText.toLowerCase()))
      .filter((memory) => !isEchoReply(userMessage, memory.responseText))
      .filter((memory) => computeRecentReplyPenalty(memory.responseText, recentAssistantReplies) < 0.42)
      .sort((left, right) => right.score - left.score)
      .slice(0, runtimeConfig.retrieval.topReplyPairs);

    const chunkCandidates = (state.knowledge.chunks || [])
      .map((chunk) => ({
        ...chunk,
        score:
          computeBm25Score(queryMap, chunk, state.knowledge.bm25.knowledge, runtimeConfig) *
          runtimeConfig.retrieval.knowledgeChunkBoost,
      }))
      .filter((chunk) => chunk.score > 0)
      .sort((left, right) => right.score - left.score)
      .slice(0, state.settings.training.topKChunks);

    const bestReply = replyCandidates[0] || null;
    const knowledgeText = selectKnowledgeSentences(
      chunkCandidates,
      queryMap,
      state.settings.generation.maxReplySentences
    ).join(' ');

    await ensureRuntimeLoaded(state);
    let neuralReply = '';

    if (neuralRuntime?.model) {
      const promptText = buildPromptForGeneration(
        userMessage,
        contextMessages,
        chunkCandidates,
        replyCandidates
      );
      const generated = await generateText({
        runtime: neuralRuntime,
        promptText,
        settings: state.settings,
      });
      neuralReply = sanitizeReplyText(generated.text);
    }

    const candidate = chooseBestReplyCandidate({
      userMessage,
      neuralReply,
      bestReply,
      knowledgeText,
      chunkCandidates,
      replyCandidates,
      contextMessages,
      recentAssistantReplies,
    });

    if (!candidate) {
      return {
        content: 'Пока в базе знаний не нашлось достаточно сильного ответа по этому запросу. Добавьте больше данных, оцените хорошие ответы и повторите обучение.',
        metadata: {
          mode: 'fallback',
          usedContextMessages: contextMessages.length,
          matchedSourceLabels: chunkCandidates.map((chunk) => chunk.label).slice(0, 4),
          matchedMemoryScore: bestReply?.score || 0,
          chunkMatches: chunkCandidates.length,
          selfScore: 0,
          userRating: 0,
        },
      };
    }

    const selfScore = computeSelfScore({
      userMessage,
      replyText: candidate.content,
      chunkCandidates,
      replyCandidates,
      contextMessages,
      recentAssistantReplies,
    });

    return {
      content: previewText(candidate.content, state.settings.generation.maxReplyCharacters),
      metadata: {
        mode: candidate.mode,
        usedContextMessages: contextMessages.length,
        matchedSourceLabels: chunkCandidates.map((chunk) => chunk.label).slice(0, 4),
        matchedMemoryScore: bestReply?.score || 0,
        chunkMatches: chunkCandidates.length,
        selfScore: Number(selfScore.toFixed(3)),
        userRating: 0,
      },
    };
  }

  return {
    async hydrateState() {
      return updateState(async (state) => {
        ensureChatAvailability(state);
        const artifacts = await rebuildKnowledge(state);
        await ensureRuntimeLoaded(state);

        if (state.training.status === 'training') {
          state.training.status = 'idle';
          state.training.phase = 'server_restarted';
          state.training.message = 'Сервер был перезапущен во время обучения. Запустите обучение снова, чтобы продолжить от последнего чекпоинта.';
          state.model.lifecycle = state.model.trainedEpochs ? 'trained' : 'ready_for_training';
          state.model.status = 'ready';
          pushStatus(state, 'idle', 'server_restarted', state.training.message);
        }

        if (state.model.trainedEpochs > 0) {
          const signatureMismatch =
            !state.model.corpusSignature || state.model.corpusSignature !== artifacts.corpusSignature;

          if (signatureMismatch || !neuralRuntime?.model) {
            await disposeNeuralRuntime();
            resetTrainedModelState(state);
            state.model.lifecycle = state.model.exists ? 'ready_for_training' : 'not_created';
            state.model.status = state.model.exists ? 'ready' : 'idle';
            pushStatus(
              state,
              'idle',
              'model_rehydrated_retrain_required',
              'Сервер нашел изменения корпуса или не смог восстановить чекпоинт. Для корректной работы модель нужно обучить заново.'
            );
          } else {
            state.model.parameterCount = neuralRuntime.parameterCount || state.model.parameterCount;
          }
        }
      });
    },

    async getDashboard(chatId) {
      return buildDashboard(getState(), chatId);
    },

    async createChat() {
      return updateState(async (state) => {
        assertTrainingUnlocked(state, 'Создание чата');
        ensureChatAvailability(state);
        state.chats.unshift(createEmptyChat());
      }).then((state) => buildDashboard(state, state.chats[0]?.id));
    },

    async deleteChat(chatId) {
      return updateState(async (state) => {
        assertTrainingUnlocked(state, 'Удаление чата');
        state.chats = state.chats.filter((chat) => chat.id !== chatId);
        ensureChatAvailability(state);
      }).then((state) => buildDashboard(state, state.chats[0]?.id));
    },

    async addSource({ type, label, content, url = null }) {
      return updateState(async (state) => {
        assertTrainingUnlocked(state, 'Изменение источников');
        const text = cleanText(content);
        if (!text) {
          throw new Error('РСЃС‚РѕС‡РЅРёРє РЅРµ СЃРѕРґРµСЂР¶РёС‚ С‚РµРєСЃС‚Р°.');
        }

        state.sources.unshift({
          id: createId('source'),
          type,
          label,
          url,
          content: text,
          stats: computeStats(text),
          addedAt: nowIso(),
        });

        if (state.model.exists) {
          await disposeNeuralRuntime();
          resetTrainedModelState(state);
          state.model.lifecycle = 'ready_for_training';
          state.model.status = 'ready';
        }

        pushStatus(state, 'idle', 'source_added', `РСЃС‚РѕС‡РЅРёРє ${label} РґРѕР±Р°РІР»РµРЅ РІ РєРѕСЂРїСѓСЃ.`);
        await rebuildKnowledge(state);
      }).then((state) => buildDashboard(state));
    },

    async removeSource(sourceId) {
      return updateState(async (state) => {
        assertTrainingUnlocked(state, 'Изменение источников');
        state.sources = state.sources.filter((source) => source.id !== sourceId);

        if (state.model.exists) {
          await disposeNeuralRuntime();
          resetTrainedModelState(state);
          state.model.lifecycle = 'ready_for_training';
          state.model.status = 'ready';
        }

        pushStatus(state, 'idle', 'source_removed', 'РСЃС‚РѕС‡РЅРёРє СѓРґР°Р»РµРЅ РёР· РєРѕСЂРїСѓСЃР°.');
        await rebuildKnowledge(state);
      }).then((state) => buildDashboard(state));
    },

    async updateSettings(partialSettings) {
      return updateState(async (state) => {
        assertTrainingUnlocked(state, 'Изменение настроек');
        const previousSettings = structuredClone(state.settings);
        state.settings = {
          training: {
            ...state.settings.training,
            ...(partialSettings.training || {}),
          },
          generation: {
            ...state.settings.generation,
            ...(partialSettings.generation || {}),
          },
        };

        if (hasRetrainingImpact(previousSettings, state.settings)) {
          await disposeNeuralRuntime();
          resetTrainedModelState(state);
          state.model.lifecycle = state.model.exists ? 'ready_for_training' : 'not_created';
          state.model.status = state.model.exists ? 'ready' : 'idle';
          pushStatus(
            state,
            'idle',
            'settings_updated_retrain_required',
            'РР·РјРµРЅРµРЅС‹ Р°СЂС…РёС‚РµРєС‚СѓСЂР° РёР»Рё РєРѕСЂРїСѓСЃРЅС‹Рµ РїР°СЂР°РјРµС‚СЂС‹. Р”Р»СЏ РЅРѕРІРѕР№ РєРѕРЅС„РёРіСѓСЂР°С†РёРё С‚СЂРµР±СѓРµС‚СЃСЏ РїРѕРІС‚РѕСЂРЅРѕРµ РѕР±СѓС‡РµРЅРёРµ.'
          );
        } else {
          pushStatus(state, 'idle', 'settings_updated', 'РќР°СЃС‚СЂРѕР№РєРё СЃРµСЂРІРµСЂР° РѕР±РЅРѕРІР»РµРЅС‹.');
        }

        await rebuildKnowledge(state);
      }).then((state) => buildDashboard(state));
    },

    async createFreshModel() {
      await updateState(async (state) => {
        assertTrainingUnlocked(state, 'Создание модели');
        await disposeNeuralRuntime();
        state.model = createDefaultModelState();
        state.training = createDefaultTrainingState();
        state.knowledge = createDefaultKnowledgeState();
        state.model.exists = true;
        state.model.lifecycle = 'ready_for_training';
        state.model.status = 'ready';
        state.model.configSnapshot = structuredClone(state.settings);
        pushStatus(state, 'idle', 'fresh_model_created', 'РЎРѕР·РґР°РЅР° РЅРѕРІР°СЏ СЃРµСЂРІРµСЂРЅР°СЏ РЅРµР№СЂРѕСЃРµС‚СЊ.');
        await rebuildKnowledge(state);
      });
      return buildDashboard(getState());
    },

    async pauseTraining() {
      await updateState(async (state) => {
        state.training.requestedStop = true;
        pushStatus(state, 'training', 'pause_requested', 'РћСЃС‚Р°РЅРѕРІРєР° Р±СѓРґРµС‚ РІС‹РїРѕР»РЅРµРЅР° РїРѕСЃР»Рµ С‚РµРєСѓС‰РµРіРѕ Р±Р°С‚С‡Р°.');
      }, { persist: false });
      activeTrainingWorker?.postMessage({ type: 'stop' });
      return buildDashboard(getState());
    },

    async resetModel() {
      await updateState(async (state) => {
        assertTrainingUnlocked(state, 'Сброс модели');
        await disposeNeuralRuntime();
        state.model = createDefaultModelState();
        state.training = createDefaultTrainingState();
        state.knowledge = createDefaultKnowledgeState();
        state.model.exists = true;
        state.model.lifecycle = 'ready_for_training';
        state.model.status = 'ready';
        pushStatus(state, 'idle', 'model_reset', 'РњРѕРґРµР»СЊ СЃР±СЂРѕС€РµРЅР°. РСЃС‚РѕС‡РЅРёРєРё Рё С‡Р°С‚С‹ СЃРѕС…СЂР°РЅРµРЅС‹.');
        await rebuildKnowledge(state);
      });
      return buildDashboard(getState());
    },

    async trainModel() {
      await ensureModelExists();
      runTrainingJob().catch(async (error) => {
        await updateState(async (state) => {
          state.model.lifecycle = 'error';
          state.model.status = 'error';
          state.training.status = 'error';
          state.training.phase = 'training_failed';
          pushStatus(state, 'error', 'training_failed', error.message);
        });
      });
      return buildDashboard(getState());
    },

    async sendMessage(chatId, content) {
      const normalizedContent = cleanText(content);
      if (!normalizedContent) {
        throw new Error('РЎРѕРѕР±С‰РµРЅРёРµ РїСѓСЃС‚РѕРµ.');
      }

      const snapshot = await updateState(async (state) => {
        assertTrainingUnlocked(state, 'Отправка сообщений');
        ensureChatAvailability(state);
        const chat = state.chats.find((entry) => entry.id === chatId) || state.chats[0];

        const userMessage = {
          id: createId('msg'),
          role: 'user',
          content: normalizedContent,
          createdAt: nowIso(),
        };

        chat.messages.push(userMessage);

        if (!state.model.exists || !state.model.trainedEpochs) {
          chat.messages.push({
            id: createId('msg'),
            role: 'assistant',
            content: 'Модель пока не обучена. Сначала добавьте данные и запустите обучение.',
            createdAt: nowIso(),
            metadata: {
              mode: 'system',
              selfScore: 0,
              userRating: 0,
            },
          });
        } else {
          state.model.lifecycle = 'generating_reply';
          state.model.status = 'busy';
          pushStatus(state, 'syncing_knowledge', 'generating_reply', 'Сервер формирует ответ по обученной нейросети и индексу знаний.');

          const reply = await renderReply({
            state,
            chat,
            userMessage: normalizedContent,
          });

          chat.messages.push({
            id: createId('msg'),
            role: 'assistant',
            content: reply.content,
            createdAt: nowIso(),
            metadata: reply.metadata,
          });

          const assistantMessages = chat.messages
            .filter((message) => message.role === 'assistant' && message.metadata?.selfScore !== undefined)
            .map((message) => Number(message.metadata.selfScore || 0));
          state.model.averageSelfScore = assistantMessages.length
            ? Number((assistantMessages.reduce((sum, value) => sum + value, 0) / assistantMessages.length).toFixed(3))
            : state.model.averageSelfScore;
          state.model.lifecycle = 'trained';
          state.model.status = 'ready';
          state.model.lastGenerationAt = nowIso();
        }

        chat.title = inferChatTitle(chat.messages);
        chat.updatedAt = nowIso();
      });

      return buildDashboard(snapshot, chatId);
    },

    async rateMessage(messageId, score) {
      const normalizedScore = score > 0 ? 1 : -1;

      return updateState(async (state) => {
        assertTrainingUnlocked(state, 'Оценка ответов');
        const located = findMessageById(state, messageId);
        if (!located) {
          throw new Error('РЎРѕРѕР±С‰РµРЅРёРµ РґР»СЏ РѕС†РµРЅРєРё РЅРµ РЅР°Р№РґРµРЅРѕ.');
        }

        if (located.message.role !== 'assistant' || located.message.metadata?.type === 'system') {
          throw new Error('РћС†РµРЅРёРІР°С‚СЊ РјРѕР¶РЅРѕ С‚РѕР»СЊРєРѕ РѕС‚РІРµС‚С‹ РјРѕРґРµР»Рё.');
        }

        located.message.metadata = {
          ...(located.message.metadata || {}),
          userRating: normalizedScore,
          ratedAt: nowIso(),
        };

        if (normalizedScore === 1) {
          state.model.positiveFeedbackCount += 1;
          pushStatus(
            state,
            'learning_from_feedback',
            'feedback_recorded',
            'РџРѕР»РѕР¶РёС‚РµР»СЊРЅР°СЏ РѕС†РµРЅРєР° СЃРѕС…СЂР°РЅРµРЅР°. Р­С‚РѕС‚ РѕС‚РІРµС‚ РІРѕР№РґРµС‚ РІ РїР°РјСЏС‚СЊ РјРѕРґРµР»Рё Рё РІ СЃР»РµРґСѓСЋС‰РёР№ РѕР±СѓС‡Р°СЋС‰РёР№ РїСЂРѕРіРѕРЅ.'
          );
        } else {
          state.model.negativeFeedbackCount += 1;
          pushStatus(
            state,
            'learning_from_feedback',
            'feedback_recorded',
            'РћС‚СЂРёС†Р°С‚РµР»СЊРЅР°СЏ РѕС†РµРЅРєР° СЃРѕС…СЂР°РЅРµРЅР°. Р­С‚РѕС‚ РѕС‚РІРµС‚ Р±СѓРґРµС‚ РїРѕРґР°РІР»СЏС‚СЊСЃСЏ РІ Р±СѓРґСѓС‰РёС… РѕС‚РІРµС‚Р°С… Рё РёСЃРєР»СЋС‡РµРЅ РёР· РѕР±СѓС‡Р°СЋС‰РёС… РїСЂРёРјРµСЂРѕРІ.'
          );
        }

        await rebuildKnowledge(state);
        state.model.lifecycle = state.model.trainedEpochs ? 'trained' : 'ready_for_training';
        state.model.status = 'ready';
      }).then((state) => buildDashboard(state));
    },
  };
}

module.exports = {
  createModelEngine,
};
