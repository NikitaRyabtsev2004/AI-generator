# Python backend (`server/python`)

Python-часть отвечает за обучение и инференс локальной модели на TensorFlow/Keras.

## Основные файлы

- `keras_llm_backend.py` — CLI-точка входа для Node bridge (`train`, `load_runtime`, `generate`).
- `llm_model.py` — GPT-подобная Transformer-архитектура (attention, FFN, layer norm, positional encoding).
- `llm_tokenizer.py` — subword-токенизация, обучение и сохранение словаря.
- `llm_data.py` — подготовка датасета и `tf.data` pipeline.
- `requirements.txt` — зависимости Python runtime.
- `setup-venv.ps1` — автоматическая настройка виртуального окружения (Windows).
- `setup-venv2.ps1` — альтернативный скрипт для отдельного GPU-окружения.

## Что реализовано в контуре обучения

- Кастомный training loop (без `model.fit`).
- Teacher forcing.
- Cross-entropy loss + метрики (`loss`, `perplexity`).
- Checkpoints и артефакты для продолжения обучения.
- Поддержка mixed precision (по возможностям окружения).
- Streaming-подход к данным через `tf.data`.

## Поддерживаемые источники данных

Через Node-пайплайн в Python поступают нормализованные тренировочные тексты, полученные из:

- `txt`
- `csv`
- `json`
- `jsonl`
- `parquet`

## Подготовка окружения (Windows)

```powershell
cd server/python
./setup-venv.ps1
```

После этого Node backend сможет автоматически использовать созданный `.venv`.

## Ручной запуск для диагностики

Пример проверки доступности backend-команды:

```powershell
cd server/python
.\.venv\Scripts\python.exe .\keras_llm_backend.py --help
```

## Частые ошибки

1. `No module named tensorflow`
   причина: окружение не создано или не установлены зависимости.
   решение: выполнить `setup-venv.ps1`.

2. Ошибка legacy Keras optimizer
   причина: смешение Keras 3 и legacy API.
   решение: использовать совместимые оптимизаторы в коде или включить совместимый стек `tf_keras` при необходимости.

3. Слишком быстрый train на большом parquet
   причина: в корпус попал только небольшой фрагмент данных.
   решение: проверять лог извлечения и статистику окон/токенов перед запуском обучения.
