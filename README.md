# IOLA small model lab

Исследовательская лаборатория маленьких локальных моделей на открытых данных Йошкар-Олы.

Цель: сравнить, где маленькой модели достаточно прямого ответа, а где нужен внешний факт из API/MCP, строгая проверка или эскалация.

## Что внутри

- `datasets/simple-facts.jsonl` - 100 обычных вопросов по школам и детским садам.
- `datasets/adversarial-facts.jsonl` - 100 вопросов с ложными предпосылками.
- `scripts/run-evaluation.js` - детерминированные baseline-стратегии без LLM.
- `scripts/run-ollama-evaluation.js` - реальные прогоны через локальный Ollama API.
- `results/latest-summary.json` - последняя baseline-сводка.
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

Полный небольшой стенд:

```powershell
npm run run:ollama -- --models qwen3:0.6b,qwen3:1.7b,gemma3:1b,llama3.2:1b --modes direct,mcp-context,verified-context --dataset all --limit 10
npm run summary:ollama
```

## Интерпретация

- `direct` проверяет, пытается ли модель отвечать из памяти.
- `mcp-context` дает модели точный факт и просит отвечать только по нему.
- `verified-context` жестче требует опровергать ложные значения.
- Adversarial-метрика строгая: ответ должен не только содержать правильное значение, но и явно не подтверждать ложную предпосылку.

## Бенчмарки

- [Ollama benchmark, Node 26, 2026-05-27](benchmarks/2026-05-27-ollama-node26.md)
