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
| `datasets/history-train-v2.jsonl` | 56 | V2-набор с усилением ошибок первого прогона. |
| `datasets/history-eval-v2.jsonl` | 16 | V2 eval на той же поверхности. |
| `datasets/history-train-v3.jsonl` | 65 | V3-набор с усилением точной даты 1919 и RAG по переименованиям. |
| `datasets/history-eval-v3.jsonl` | 16 | V3 eval на той же поверхности. |

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

## Журнал запусков

| Job | Статус | Вывод |
| --- | --- | --- |
| `6a1771225c8d10ffa1104d2d` | Completed | V1 дал `json_or_text_ok 15/16`, `exact_ok 11/16`. Ошибки: точная дата 1919, военная история, source-required RAG и ambiguous rename. |
| `6a1773da5c8d10ffa1104d4f` | Completed | V2 дал `json_or_text_ok 16/16`, `exact_ok 14/16`. Остались ошибки: точная дата 17 февраля 1919 и полный RAG-запрос по переименованиям. |
| `6a17769a5c8d10ffa1104d8f` | Completed | V3 дал `json_or_text_ok 16/16`, `exact_ok 15/16`. Итоговый кандидат опубликован как `LMSerg/iola-history-1b-2026-05-28`. |

## Критерий приемки

- `json_or_text_ok`: 16/16;
- `exact_ok`: не ниже 15/16;
- source-required вопросы должны уходить в `rag_search`, а не отвечать текстом из памяти.
