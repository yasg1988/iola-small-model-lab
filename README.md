# Лаборатория маленьких локальных моделей IOLA

![Исследовательская витрина малых ИИ-моделей Йошкар-Олы](assets/research-showcase.png)

## Wiki

| Раздел | Страницы |
| --- | --- |
| Архитектурные стратегии | [conditional-memory](https://github.com/yasg1988/iola-small-model-lab/wiki/conditional-memory) · [sparse-escalation](https://github.com/yasg1988/iola-small-model-lab/wiki/sparse-escalation) · [early-exit](https://github.com/yasg1988/iola-small-model-lab/wiki/early-exit) |
| Проверка и консенсус | [strict-skill](https://github.com/yasg1988/iola-small-model-lab/wiki/strict-skill) · [verify](https://github.com/yasg1988/iola-small-model-lab/wiki/verify) · [council](https://github.com/yasg1988/iola-small-model-lab/wiki/council) |
| Гибридные стратегии | [skill-router](https://github.com/yasg1988/iola-small-model-lab/wiki/skill-router) · [memory-verified](https://github.com/yasg1988/iola-small-model-lab/wiki/memory-verified) · [escalation-ladder](https://github.com/yasg1988/iola-small-model-lab/wiki/escalation-ladder) |
| Подбор моделей | [итог исследования](docs/model-selection/research-conclusion-2026-05-27.md) · [профиль ПК](docs/model-selection/local-pc-profile-2026-05-27.md) · [план мастера iola](docs/model-selection/iola-cli-local-model-wizard-plan.md) · [MoE/Sparse](docs/model-selection/moe-sparse-models-research.md) |
| Обучение router-модели | [контракт tool-call](docs/training/router-tool-contract.md) · [training pack Gemma 3 1B](docs/training/gemma3-router-training-pack.md) · [HF dataset](https://huggingface.co/datasets/LMSerg/iola-gemma3-router-sft) · [LoRA job](docs/training/gemma3-router-lora-job.md) |
| Бенчмарки | [стратегии](benchmarks/2026-05-27-strategy-baseline.md) · [Ollama-модели](benchmarks/2026-05-27-ollama-node26.md) · [router baseline](benchmarks/2026-05-27-router-eval-baseline.md) |

Исследовательская лаборатория маленьких локальных моделей на открытых данных Йошкар-Олы.

Цель: сравнить, где маленькой модели достаточно прямого ответа, а где нужен внешний факт из API/MCP, строгая проверка или эскалация.

## Панель результатов

Графики ниже показывают не "качество модели", а качество стратегии обработки вопроса. Обычные вопросы - это прямые запросы к факту. Вопросы с ловушками - это запросы, где пользователь уже называет неверное значение и просит подтвердить его.

Главный вывод: стратегия может быть идеальной на обычных вопросах и полностью проваливаться на ловушках, если в ней нет явного шага проверки.

### Стратегии

Проверено 9 стратегий на 200 вопросах: 100 обычных и 100 с ложными предпосылками.

| Стратегия | Блок | Обычные вопросы | Вопросы с ловушками | Итог |
| --- | --- | ---: | ---: | --- |
| [`conditional-memory`](benchmarks/strategies/conditional-memory.md) | Архитектура | 100/100 | 0/100 | Быстрый доступ к фактам, но без опровержения ловушек опасен |
| [`sparse-escalation`](benchmarks/strategies/sparse-escalation.md) | Архитектура | 100/100 | 100/100 | Надежная минимальная эскалация |
| [`early-exit`](benchmarks/strategies/early-exit.md) | Архитектура | 100/100 | 0/100 | Быстрый режим годится только для безопасных вопросов |
| [`strict-skill`](benchmarks/strategies/strict-skill.md) | Проверка | 100/100 | 100/100 | Лучший простой базовый режим |
| [`verify`](benchmarks/strategies/verify.md) | Проверка | 100/100 | 100/100 | Надежная проверка ответа |
| [`council`](benchmarks/strategies/council.md) | Проверка | 100/100 | 100/100 | Консилиум полезен для спорных вопросов |
| [`skill-router`](benchmarks/strategies/skill-router.md) | Гибрид | 100/100 | 100/100 | Практичный режим для CLI |
| [`memory-verified`](benchmarks/strategies/memory-verified.md) | Гибрид | 100/100 | 100/100 | Память плюс проверка закрывает ловушки |
| [`escalation-ladder`](benchmarks/strategies/escalation-ladder.md) | Гибрид | 100/100 | 100/100 | Лучший продуктовый сценарий |

Что означает график:

- `██████████ 100%` - стратегия дала правильный ответ на все вопросы датасета.
- `__________ 0%` - стратегия отвечала, но не выполнила строгое условие проверки.
- Для вопросов с ловушками правильный ответ должен не только содержать факт, но и явно не подтверждать неверное утверждение пользователя.

```text
Обычные вопросы
conditional-memory  ██████████ 100%
sparse-escalation   ██████████ 100%
early-exit          ██████████ 100%
strict-skill        ██████████ 100%
verify              ██████████ 100%
council             ██████████ 100%
skill-router        ██████████ 100%
memory-verified     ██████████ 100%
escalation-ladder   ██████████ 100%

Вопросы с ловушками
conditional-memory  __________   0%
sparse-escalation   ██████████ 100%
early-exit          __________   0%
strict-skill        ██████████ 100%
verify              ██████████ 100%
council             ██████████ 100%
skill-router        ██████████ 100%
memory-verified     ██████████ 100%
escalation-ladder   ██████████ 100%
```

Почему `conditional-memory` и `early-exit` падают на ловушках:

- `conditional-memory` достает правильный факт, но без проверяющего не обязана опровергать ложное значение пользователя.
- `early-exit` слишком рано завершает обработку, когда поиск уверен в факте, но не проверяет саму формулировку вопроса.
- Все стратегии с проверкой, маршрутизацией или эскалацией проходят ловушки, потому что отдельный шаг сравнивает факт с утверждением пользователя.

### Wiki по стратегиям

- [conditional-memory](https://github.com/yasg1988/iola-small-model-lab/wiki/conditional-memory)
- [sparse-escalation](https://github.com/yasg1988/iola-small-model-lab/wiki/sparse-escalation)
- [early-exit](https://github.com/yasg1988/iola-small-model-lab/wiki/early-exit)
- [strict-skill](https://github.com/yasg1988/iola-small-model-lab/wiki/strict-skill)
- [verify](https://github.com/yasg1988/iola-small-model-lab/wiki/verify)
- [council](https://github.com/yasg1988/iola-small-model-lab/wiki/council)
- [skill-router](https://github.com/yasg1988/iola-small-model-lab/wiki/skill-router)
- [memory-verified](https://github.com/yasg1988/iola-small-model-lab/wiki/memory-verified)
- [escalation-ladder](https://github.com/yasg1988/iola-small-model-lab/wiki/escalation-ladder)

### Реальные модели Ollama

Проверены 4 маленькие модели в 3 режимах промпта. Лучший результат на обычных вопросах дала `gemma3:1b` в режиме `mcp-context`: `9/10`. На вопросах с ловушками одних промптов недостаточно: максимум `1/10`.

| Модель | Лучший режим на обычных вопросах | Результат | Средняя задержка |
| --- | --- | ---: | ---: |
| `gemma3:1b` | `mcp-context` | 9/10 | 1.75 с |
| `llama3.2:1b` | `mcp-context` | 7/10 | 4.55 с |
| `qwen3:0.6b` | `mcp-context` | 7/10 | 5.08 с |
| `qwen3:1.7b` | `mcp-context` / `verified-context` | 6/10 | 12.80-13.50 с |

```text
Лучший результат моделей на обычных вопросах
gemma3:1b      █████████_ 90%
llama3.2:1b    ███████___ 70%
qwen3:0.6b     ███████___ 70%
qwen3:1.7b     ██████____ 60%
```

### Предел локального ПК

На текущем RedmiBook 15 с 7.79 ГБ ОЗУ практический вывод такой:

| Класс модели | Статус |
| --- | --- |
| 1B-2B | рабочий интерактивный уровень |
| 4B | запускается, нужен отдельный prompt-тест |
| 7B | работает, но медленно |
| 12B | предельный нерабочий уровень для практического CLI |
| 14B+ | не тестировать автоматически на этом ПК |

Подробно: [профиль локального ПК и пределы моделей](docs/model-selection/local-pc-profile-2026-05-27.md).

План внедрения в будущий мастер `iola`: [подбор локальной модели](docs/model-selection/iola-cli-local-model-wizard-plan.md).

MoE/Sparse-кандидаты и раннеры: [исследование MoE/Sparse](docs/model-selection/moe-sparse-models-research.md).

Итог исследовательского этапа: сырые локальные модели и крупные MoE/Sparse-кандидаты не дали надежного продукта на слабом ПК; следующий этап строится вокруг RAG/GraphRAG, FastAPI/MCP и LoRA-адаптеров маленькой модели. Подробно: [итог исследования](docs/model-selection/research-conclusion-2026-05-27.md).

## Что внутри

- `datasets/simple-facts.jsonl` - 100 обычных вопросов по школам и детским садам.
- `datasets/adversarial-facts.jsonl` - 100 вопросов с ложными предпосылками.
- `datasets/router-eval-v1.jsonl` - стартовый eval-набор для проверки JSON/tool-call router-модели.
- `datasets/router-train-v1.jsonl` - SFT training pack для `gemma3:1b`: маршрутизация по учреждениям, ловушки, clarify/refuse, исторические direct-answer и RAG-source режимы.
- `datasets/router-train-v2.jsonl` - V2 training pack без запоминания ИНН: модель возвращает `entity_number/entity_name`, а ИНН резолвит API.
- `datasets/router-train-v3.jsonl` - V3 training pack с усилением alias, search, clarify/refuse, RAG и officials.
- `datasets/router-train-v4.jsonl` - V4 training pack с точечным усилением провалов V3.
- `datasets/router-train-v5.jsonl` - V5 training pack с точечным усилением провалов V4.
- `datasets/city-history-stable-facts.jsonl` - стабильные исторические факты Йошкар-Олы для SFT и RAG-source режимов.
- `scripts/run-evaluation.js` - детерминированные базовые стратегии без LLM.
- `scripts/run-ollama-evaluation.js` - реальные прогоны через локальный Ollama API.
- `scripts/generate-router-training.js` - сборка router training pack без пересечения с eval-набором.
- `scripts/validate-router-datasets.js` - проверка строгого JSON-контракта, разрешенных tools/layers/fields и train/eval overlap.
- `training/train_gemma3_router_lora.py` - HF Jobs/UV скрипт для LoRA-обучения `google/gemma-3-1b-it`.
- `results/latest-summary.json` - последняя сводка базового прогона.
- `results/ollama-runs/latest-summary.json` - последняя Ollama-сводка, не коммитится.

## Требования

- Node.js 26+
- Ollama
- Переменная `OLLAMA_MODELS`, если модели лежат не в стандартной папке. На текущем стенде используется:

```powershell
$env:OLLAMA_MODELS = 'D:\ollama\models'
```

## Команды

```powershell
npm test
npm run models
npm run pull -- --models qwen3:0.6b,qwen3:1.7b
npm run generate
npm run run:baseline -- --all
npm run summary
npm run run:ollama -- --models qwen3:0.6b,gemma3:1b --modes mcp-context,verified-context --dataset simple-facts --limit 5
npm run summary:ollama
npm run generate:router
npm run validate:router
```

## Версия

Текущая публичная версия: `v0.1.17`.

История изменений: [CHANGELOG.md](CHANGELOG.md).

Полный небольшой стенд:

```powershell
npm run run:ollama -- --models qwen3:0.6b,qwen3:1.7b,gemma3:1b,llama3.2:1b --modes direct,mcp-context,verified-context --dataset all --limit 10
npm run summary:ollama
```

## Интерпретация

- `direct` проверяет, пытается ли модель отвечать из памяти.
- `mcp-context` дает модели точный факт и просит отвечать только по нему.
- `verified-context` жестче требует опровергать ложные значения.
- Метрика вопросов с ловушками строгая: ответ должен не только содержать правильное значение, но и явно не подтверждать ложную предпосылку.

## Бенчмарки

- [Бенчмарк стратегий на открытых данных Йошкар-Олы от 2026-05-27](benchmarks/2026-05-27-strategy-baseline.md)
- [Бенчмарк Ollama на Node 26 от 2026-05-27](benchmarks/2026-05-27-ollama-node26.md)
- [Router eval baseline от 2026-05-27](benchmarks/2026-05-27-router-eval-baseline.md)

## Итог исследования

Текущий исследовательский этап закрыт: установка сырых локальных моделей и проверка крупных MoE/Sparse-кандидатов не дали надежного продуктового результата на слабом ПК. Следующий этап лаборатории: RAG/GraphRAG на стороне FastAPI/MCP, Qdrant как внутренний индекс и LoRA-адаптеры маленькой модели для маршрутизации, уточнений, отказов и строгих tool-call.

Подробно: [итог исследовательского этапа](docs/model-selection/research-conclusion-2026-05-27.md).
