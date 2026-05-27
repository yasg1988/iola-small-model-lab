# История изменений

## v0.1.0 - 2026-05-27

Первый публичный релиз лаборатории.

Добавлено:

- CLI `iola-model-lab` для генерации датасетов, запуска baseline-стратегий и Ollama-прогонов.
- Датасеты `simple-facts` и `adversarial-facts` по открытым данным Йошкар-Олы.
- 9 отдельных витрин стратегий:
  - `conditional-memory`
  - `sparse-escalation`
  - `early-exit`
  - `strict-skill`
  - `verify`
  - `council`
  - `skill-router`
  - `memory-verified`
  - `escalation-ladder`
- Бенчмарк реальных Ollama-моделей:
  - `gemma3:1b`
  - `llama3.2:1b`
  - `qwen3:0.6b`
  - `qwen3:1.7b`
- README-витрина с изображением, сводными таблицами и графиками.
