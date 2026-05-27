# History adapter training pack

Дата: 2026-05-28.

Цель: отдельный LoRA-адаптер для стабильных исторических фактов Йошкар-Олы на базе `google/gemma-3-1b-it`.

Этот адаптер отличается от router-адаптера:

- на стабильные исторические вопросы он отвечает обычным текстом;
- если пользователь просит источник, подтверждение или справку с источником, он возвращает JSON `rag_search`;
- если вопрос неоднозначный, например "когда переименовали город?", он уточняет, о каком переименовании речь.

## Файлы

| Файл | Строк | Назначение |
| --- | ---: | --- |
| `datasets/history-train-v1.jsonl` | 42 | Train SFT-набор для исторического адаптера. |
| `datasets/history-eval-v1.jsonl` | 16 | Eval-набор без точного пересечения с train. |

## Команды

```powershell
npm run generate:history
npm run validate:history
npm test
```

## HF dataset

Планируемый dataset repo:

`LMSerg/iola-history-sft`

## Output model

Планируемый adapter repo:

`LMSerg/iola-history-1b-2026-05-28`

## Критерий приемки

- `json_or_text_ok`: 16/16;
- `exact_ok`: не ниже 15/16;
- source-required вопросы должны уходить в `rag_search`, а не отвечать текстом из памяти.
