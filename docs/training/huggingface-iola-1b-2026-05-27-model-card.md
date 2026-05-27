---
license: gemma
base_model: google/gemma-3-1b-it
datasets:
  - LMSerg/iola-gemma3-router-sft
language:
  - ru
tags:
  - lora
  - peft
  - tool-calling
  - rag
  - yoshkar-ola
  - iola
  - gemma3
model-index:
  - name: IOLA 1B 2026-05-27
    results:
      - task:
          type: text-generation
          name: Router tool calling
        dataset:
          name: IOLA Gemma 3 Router SFT eval v5
          type: LMSerg/iola-gemma3-router-sft
          split: data/router-eval-v5.jsonl
        metrics:
          - type: json_accuracy
            value: 1.0
            name: JSON accuracy
          - type: exact_accuracy
            value: 0.9666666667
            name: Exact router accuracy
---

# IOLA 1B 2026-05-27

LoRA-адаптер для `google/gemma-3-1b-it`.

Назначение: router-модель для CLI/FastAPI/MCP/RAG контура Йошкар-Олы. Модель не должна отвечать по изменяемым городским данным из памяти; она возвращает строгий JSON/tool-call.

## Результат

Лучший прогон: V5.

| Метрика | Значение |
| --- | ---: |
| JSON parse | 30/30 |
| Exact router match | 29/30 |
| Exact accuracy | 96.67% |

Единственный провал V5: salary/private-data вопрос про зарплату директора школы №7. В продукте этот класс должен дополнительно закрываться deterministic API/CLI guard до или после модели.

## Пример выхода

```json
{
  "action": "tool_call",
  "tool": "resolve_entity_field",
  "args": {
    "layer": "schools",
    "entity_number": 2,
    "field": "phone"
  }
}
```

## Архитектурное решение

V1 пытался заставить 1B-модель запоминать ИНН учреждений. Это оказалось ненадежно.

Текущий контракт использует `resolve_entity_field`: модель возвращает слой, номер или имя учреждения и поле; API резолвит ИНН через справочник.

## Источники

- Dataset: `https://huggingface.co/datasets/LMSerg/iola-gemma3-router-sft`
- Lab repo: `https://github.com/yasg1988/iola-small-model-lab`
- Technical best source repo: `https://huggingface.co/LMSerg/iola-gemma3-router-v5-lora`
