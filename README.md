# Лаборатория маленьких локальных моделей IOLA

![Исследовательская витрина малых ИИ-моделей Йошкар-Олы](assets/research-showcase.png)

Исследовательская лаборатория маленьких локальных моделей на открытых данных Йошкар-Олы.

Цель: сравнить, где маленькой модели достаточно прямого ответа, а где нужен внешний факт из API/MCP, строгая проверка или эскалация.

## Панель результатов

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

Текущая публичная версия: `v0.1.0`.

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
