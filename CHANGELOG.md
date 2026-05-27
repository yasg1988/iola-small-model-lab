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

## v0.1.1 - 2026-05-27

Добавлено:

- Расширенные пояснения к графикам в README.
- Wiki-страницы стратегий в `docs/wiki/`.
- Фиксация параметров локального ПК и практического предела моделей.
- План внедрения подбора локальной модели в мастер `iola`.
- Исследовательская заметка по MoE/Sparse-моделям и раннерам.

Уточнено:

- Для RedmiBook 15 с 7.79 ГБ ОЗУ класс 12B считается предельным нерабочим для практического CLI.
- MoE/Sparse-модели нельзя выбирать только по активным параметрам: общий размер модели и поведение раннера остаются критичными.

## v0.1.2 - 2026-05-27

Добавлено:

- Страницы стратегий опубликованы в GitHub Wiki.
- README переключен с локальных `docs/wiki`-ссылок на настоящие wiki-ссылки.

## v0.1.3 - 2026-05-27

Добавлено:

- Wiki-меню в README сразу под изображением витрины.

## v0.1.4 - 2026-05-27

Уточнено:

- Зафиксирован практический вывод локальной проверки Qwen 35B-A3B и Gemma 26B-A4B: большие модели с малым числом активных параметров не становятся рабочими на слабом ПК без учета общего размера, поддержки раннера, памяти, шаблона и фактической скорости.

## v0.1.5 - 2026-05-27

Добавлено:

- Датасет `city-history-stable-facts.jsonl` со стабильными историческими фактами о Йошкар-Оле для прямого обучения модели и отдельными примерами, где требуется RAG-подтверждение источником.

## v0.1.6 - 2026-05-27

Добавлено:

- Итоговый документ исследовательского этапа: сырые локальные модели и крупные MoE/Sparse-кандидаты закрыты как недостаточно надежная продуктовая гипотеза для слабого ПК; следующий этап зафиксирован как RAG/GraphRAG + FastAPI/MCP + LoRA-адаптеры маленькой модели.

## v0.1.7 - 2026-05-27

Добавлено:

- Контракт `router-tool-contract.md` для строгого JSON/tool-call выхода маленькой router-модели.
- Стартовый eval-набор `router-eval-v1.jsonl` на 30 вопросов для проверки layer, field, inn, clarify/refuse, RAG/source-required, officials и adversarial-сценариев.

## v0.1.8 - 2026-05-27

Добавлено:

- Скрипт `run-router-eval.js` для Ollama-прогона router-моделей в JSON-mode.
- Router baseline по `gemma3:1b`, `qwen3:1.7b` и `batiai/gemma4-e2b:q4`; Gemma 4 E2B Q4 показала лучший сырой router-сигнал, но требует больше ресурсов.

## v0.1.9 - 2026-05-27

Добавлено:

- Training pack `router-train-v1.jsonl` на 810 SFT-примеров для первого LoRA-прогона `gemma3:1b`.
- Раздельные train-файлы: `router-train-entities`, `router-train-safety`, `router-train-history`.
- Генератор `generate-router-training.js`, который исключает точные пересечения с `router-eval-v1`.
- Валидатор `validate-router-datasets.js` для строгого JSON/tool-call контракта, разрешенных tools/layers/fields и train/eval overlap.

## v0.1.10 - 2026-05-27

Добавлено:

- Hugging Face dataset `LMSerg/iola-gemma3-router-sft` для обучения первого LoRA-адаптера `gemma3:1b`.
- Dataset card для Hugging Face с описанием train/eval файлов и JSON/tool-call контракта.
- Ссылка на HF dataset в README и документации training pack.

## v0.1.11 - 2026-05-27

Добавлено:

- HF Jobs/UV training script `training/train_gemma3_router_lora.py` для LoRA-обучения `google/gemma-3-1b-it`.
- Документация запуска `gemma3-router-lora-job.md` с командой, лимитом времени, мониторингом и критерием приемки.
- Фиксация output model repo `LMSerg/iola-gemma3-router-gemma3-1b-lora`.

## v0.1.12 - 2026-05-27

Исправлено:

- Первый HF Jobs запуск LoRA упал на backward из-за `gradient_checkpointing` с LoRA. Для короткого T4-прогона checkpointing отключен в training script.

## v0.1.13 - 2026-05-27

Исправлено:

- Второй HF Jobs запуск технически завершился, но не улучшил exact-match: модель повторяла JSON и не выучила стабильную маршрутизацию ИНН.
- Training script теперь маскирует user prompt и считает loss только по assistant JSON.
- Eval parser считает первый сбалансированный JSON-объект, а не весь хвост генерации.

## v0.1.14 - 2026-05-27

Добавлено:

- V2 router contract `resolve_entity_field`: модель возвращает `layer + entity_number/entity_name + field`, а ИНН резолвит API.
- V2 train/eval файлы: `router-train-v2.jsonl`, `router-eval-v2.jsonl` и раздельные компоненты.

Зафиксировано:

- Третий HF Jobs запуск `6a1757885c8d10ffa1104b6f` дал `json_ok 30/30`, но `exact_ok 1/30` на v1-контракте с ИНН. Гипотеза "зашить ИНН в 1B-модель" признана слабой.

## v0.1.15 - 2026-05-27

Добавлено:

- V3 router train/eval с усиленными примерами для alias, `search_entities`, `clarify`, `refuse`, `rag_search` и `officials`.
- V3 eval переводит alias-запросы на `entity_name`, чтобы API выполнял fuzzy resolve, а модель не угадывала номер учреждения.

Исправлено:

- HF eval теперь учитывает `answer_contains` для `direct_answer` и допускает краткий `refuse` по `reason`.

## v0.1.16 - 2026-05-27

Добавлено:

- V4 router train/eval с точечным усилением провалов V3: ordinal school #1, missing entity clarify, private-data refuse, officials и source-required RAG.

Зафиксировано:

- V3 HF Jobs запуск `6a1760965c8d10ffa1104c1c` дал `json_ok 30/30`, `exact_ok 21/30`.
