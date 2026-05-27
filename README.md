# Лаборатория маленьких локальных моделей IOLA

![Исследовательская витрина малых ИИ-моделей Йошкар-Олы](assets/research-showcase.png)

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

- [conditional-memory](docs/wiki/conditional-memory.md)
- [sparse-escalation](docs/wiki/sparse-escalation.md)
- [early-exit](docs/wiki/early-exit.md)
- [strict-skill](docs/wiki/strict-skill.md)
- [verify](docs/wiki/verify.md)
- [council](docs/wiki/council.md)
- [skill-router](docs/wiki/skill-router.md)
- [memory-verified](docs/wiki/memory-verified.md)
- [escalation-ladder](docs/wiki/escalation-ladder.md)

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

## Что внутри

- `datasets/simple-facts.jsonl` - 100 обычных вопросов по школам и детским садам.
- `datasets/adversarial-facts.jsonl` - 100 вопросов с ложными предпосылками.
- `scripts/run-evaluation.js` - детерминированные базовые стратегии без LLM.
- `scripts/run-ollama-evaluation.js` - реальные прогоны через локальный Ollama API.
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
```

## Версия

Текущая публичная версия: `v0.1.1`.

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
