# Серверная часть (`server`)

Node.js backend, который:

- отдает API для клиента;
- управляет жизненным циклом моделей;
- запускает Python backend для обучения/генерации;
- хранит состояние и артефакты;
- обслуживает публичные read-only страницы shared-чатов.

## Ключевые модули

## Точка входа

- `server/index.js` — bootstrap приложения, middleware, роуты, SSE, статическая раздача build.

## Роуты (`server/routes`)

- `systemRoutes.js` — health/status/dashboard/logs/events.
- `sourcesRoutes.js` — загрузка и управление источниками.
- `trainingRoutes.js` — запуск/пауза/сброс/откат обучения.
- `modelsRoutes.js` — создание/выбор/удаление моделей, import/export.
- `chatsRoutes.js` — сообщения чата, шаринг, stop/retry/edit-flow.
- `publicRoutes.js` — API выдачи shared-чата по токену.
- `publicPageRoutes.js` — HTML-страница read-only чата по `/shared/chat/:token`.

## Движок и интеграции (`server/engine`)

- `modelEngine.js` — основная бизнес-логика студии.
- `trainingWorker.js` — вынесенные тяжелые задачи обучения.
- `pythonBridge.js` — запуск Python-команд, контроль процесса, abort-сигналы.
- `neuralModel.js` — работа с локальной моделью через Python runtime.

## Сервисы (`server/services`)

- `uploadSourceService.js` — прием и первичная обработка входных файлов.
- `apiModelClient.js` — запросы к внешним API-моделям.
- `chatShareService.js` — создание/валидация share-токенов чатов.

## Хранение (`server/storage`)

- `store.js` — SQLite-слой, state management и миграции.
- `runtimeConfig.js` — чтение/запись runtime-конфига.
- `modelLibraryStorage.js` — библиотека моделей пользователя.
- `trainingQueueStorage.js` — очередь источников на дообучение.

## Вспомогательные библиотеки (`server/lib`)

- `logger.js` — структурированные логи.
- `modelSettings.js` — валидация конфигов модели/обучения.
- `webSearch.js` и `content.js` — режим веб-поиска и извлечение контента.
- `text.js` — нормализация и служебные преобразования текста.

## Данные и артефакты

- `server/data` — runtime-БД, логи, служебные файлы.
- `server/artifacts` — веса, токенизаторы, индексы, манифесты.

Эти каталоги считаются рабочими и обычно не редактируются вручную.

## Запуск сервера

```bash
npm run server:dev
```

Для production (после `npm run build`):

```bash
npm run server:start
```

## Отладка типичных проблем

1. `Cannot GET /shared/chat/...` — проверь, что сервер поднят с актуальным `publicPageRoutes.js`.
2. `No module named tensorflow` — подготовь `server/python/.venv` через `setup-venv.ps1`.
3. API-модель не отвечает — проверь `endpoint`, `model id`, `api key`, а также доступность провайдера.
4. Долгие операции зависают — смотри `/api/status`, очередь и серверные логи.
