---
license: gemma
base_model: google/gemma-3-1b-it
datasets:
  - LMSerg/iola-history-sft
language:
  - ru
tags:
  - lora
  - peft
  - history
  - rag
  - yoshkar-ola
  - iola
  - gemma3
model-index:
  - name: IOLA History 1B 2026-05-28
    results:
      - task:
          type: text-generation
          name: Stable city history
        dataset:
          name: IOLA History SFT eval v3
          type: LMSerg/iola-history-sft
          split: data/history-eval-v3.jsonl
        metrics:
          - type: json_or_text_accuracy
            value: 1.0
            name: JSON or text validity
          - type: exact_accuracy
            value: 0.9375
            name: Exact history eval accuracy
---

# IOLA History 1B 2026-05-28

LoRA-адаптер для `google/gemma-3-1b-it`.

Назначение: стабильные исторические факты Йошкар-Олы. Адаптер отвечает обычным текстом на неизменяемые исторические вопросы и возвращает `rag_search`, когда пользователь просит источник.

## Результат

Лучший прогон: V3.

| Метрика | Значение |
| --- | ---: |
| JSON/text validity | 16/16 |
| Exact history eval | 15/16 |
| Exact accuracy | 93.75% |

Единственный провал V3: краткая историческая справка не перечислила оба старых названия (`Царевококшайск`, `Краснококшайск`), хотя сохранила год основания и перевод названия.

## Поведение

Прямой вопрос:

```text
Йошкар-Ола основана в 1584 году...
```

Запрос источника:

```json
{
  "action": "tool_call",
  "tool": "rag_search",
  "args": {
    "query": "Йошкар-Ола основана 1584 источник",
    "collections": ["city_history", "official_documents"]
  }
}
```

## Источники

- Dataset: `https://huggingface.co/datasets/LMSerg/iola-history-sft`
- Lab repo: `https://github.com/yasg1988/iola-small-model-lab`
- Technical best source repo: `https://huggingface.co/LMSerg/iola-history-1b-2026-05-28-v3`
