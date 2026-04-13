# Python backend (`server/python`) — что за что отвечает

Эта папка содержит реализацию обучения и инференса на TensorFlow + Keras.

## Основная точка входа

### `server/python/keras_llm_backend.py`
- CLI backend, который вызывает Node через `pythonBridge`.
- Команды:
  - `train` — обучение модели.
  - `load_runtime` — подготовка runtime для генерации.
  - `generate` — генерация ответа.
- Читает JSON-конфиг, запускает пайплайн, пишет чекпоинты и отдает статусы.

## Архитектура модели

### `server/python/llm_model.py`
- GPT-подобный autoregressive Transformer:
  - multi-head causal self-attention,
  - positional encoding,
  - layer norm,
  - feed-forward блоки,
  - конфигурируемая глубина/размеры.

## Токенизация

### `server/python/llm_tokenizer.py`
- Кастомный subword-токенизатор.
- Обучение словаря на пользовательских данных.
- Сохранение/загрузка токенизатора.
- Обработка `PAD/UNK/BOS/EOS`, паддинг и обрезка.

## Data pipeline

### `server/python/llm_data.py`
- Подготовка данных и `tf.data.Dataset`.
- Загрузка и нормализация текстов (TXT/CSV/JSON).
- Токенизация, батчинг, паддинг, train/val-разделение.

## Зависимости и окружение

### `server/python/requirements.txt`
- Python-зависимости backend (включая TensorFlow).

### `server/python/setup-venv.ps1`
- Windows-скрипт для быстрой подготовки окружения:
  - создает `.venv`,
  - устанавливает зависимости,
  - подготавливает backend к запуску из Node.

### `server/python/setup-venv2.ps1`
- Альтернативный Windows-скрипт для GPU-режима TensorFlow:
  - создает `.venv2` на Python 3.10,
  - ставит стек `tensorflow==2.10.1` из `requirements.windows-gpu.txt`,
  - подсказывает команду для `AI_GENERATOR_PYTHON`.

## Локальное окружение

### `server/python/.venv/`
- Генерируемая папка виртуального окружения.
- Не является исходным кодом проекта.
