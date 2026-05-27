# LoRA job для Gemma 3 1B router

Дата: 2026-05-27.

Цель: обучить первый LoRA-адаптер для `google/gemma-3-1b-it` на датасете `LMSerg/iola-gemma3-router-sft`.

## Входы

- Base model: `google/gemma-3-1b-it`
- Train dataset: `LMSerg/iola-gemma3-router-sft`, файл `data/router-train-v1.jsonl`
- Eval dataset: `LMSerg/iola-gemma3-router-sft`, файл `data/router-eval-v1.jsonl`
- Output model repo: `LMSerg/iola-gemma3-router-gemma3-1b-lora`

## Почему HF Jobs

На локальном ПК доступен только CPU PyTorch, без CUDA. Для LoRA-прогона нужен GPU. Минимальный практичный вариант для первого короткого запуска - Hugging Face Jobs `t4-small`.

По официальному прайсу Hugging Face Jobs `t4-small` стоит `$0.40/час`. Команда ниже ставит лимит `--timeout 1h`, то есть верхняя граница затрат около `$0.40`, если аккаунт допускает платные Jobs.

Источник цены: https://huggingface.co/docs/hub/en/jobs-pricing

## Команда запуска

```powershell
hf jobs uv run `
  --flavor t4-small `
  --timeout 1h `
  --secrets HF_TOKEN `
  --detach `
  training/train_gemma3_router_lora.py `
  --max-steps 180 `
  --output-repo LMSerg/iola-gemma3-router-gemma3-1b-lora
```

## Мониторинг

```powershell
hf jobs ps
hf jobs logs <JOB_ID>
hf jobs inspect <JOB_ID>
```

Если нужно остановить:

```powershell
hf jobs cancel <JOB_ID>
```

## Критерий результата

После завершения job в model repo должны появиться:

- LoRA adapter files;
- tokenizer files;
- `eval/eval_metrics.json`;
- `eval/eval_predictions.jsonl`.

Первый целевой минимум:

- `json_ok`: 30/30;
- `exact_ok`: заметно выше сырого baseline `2/30`.

Если exact-match недостаточен, следующий прогон делать не увеличением модели, а улучшением train-набора и loss-маски под assistant-ответ.

## Журнал запусков

| Job | Статус | Вывод |
| --- | --- | --- |
| `6a1753683a4b8cae6044cb3d` | Error | Первый запуск дошел до загрузки модели и train-набора, но упал на backward из-за `gradient_checkpointing` с LoRA: `element 0 of tensors does not require grad`. Для следующего запуска checkpointing отключен. |
| `6a1753cb3a4b8cae6044cb43` | Completed | Технически успешно обучил и загрузил адаптер, но eval слабый: строгий JSON `0/30`, first-balanced JSON `29/30`, exact `2/30`. Причина: модель повторяет JSON и часто подставляет один и тот же ИНН. Следующий запуск обучает loss только на assistant JSON. |
| `6a1757885c8d10ffa1104b6f` | Completed | Assistant-only loss исправил формат: `json_ok 30/30`, но exact всего `1/30`. Вывод: не заставлять 1B-модель хранить ИНН; перейти на V2 tool `resolve_entity_field` с `entity_number/entity_name`. |
| `6a175ce65c8d10ffa1104be5` | Completed | V2 улучшил результат до `json_ok 30/30`, `exact_ok 10/30`. Номерные школы и сады стали работать, но нужны дополнительные примеры для алиасов, `search_entities`, `clarify`, `refuse`, `rag_search` и `officials`. |
| `6a1760965c8d10ffa1104c1c` | Completed | V3 улучшил результат до `json_ok 30/30`, `exact_ok 21/30`. Остались провалы в ordinal school #1, missing entity clarify, private-data refuse, RAG query exact и officials. |
| `6a1764135c8d10ffa1104c46` | Completed | V4 улучшил результат до `json_ok 30/30`, `exact_ok 24/30`. Остались 6 ошибок: ordinal school #1, salary/home-address refuse, source-required RAG and etymology direct answer. |
| `6a1767a53a4b8cae6044cbd2` | Completed | V5 улучшил результат до `json_ok 30/30`, `exact_ok 29/30`. Единственная ошибка: salary/private-data refuse для директора школы №7. |
