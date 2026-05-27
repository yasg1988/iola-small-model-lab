---
license: mit
language:
  - ru
task_categories:
  - text-generation
tags:
  - sft
  - tool-calling
  - rag
  - yoshkar-ola
  - gemma3
pretty_name: IOLA Gemma 3 Router SFT
size_categories:
  - n<1K
---

# IOLA Gemma 3 Router SFT

SFT-набор для первого LoRA-прогона маленькой router-модели на базе `gemma3:1b`.

Цель датасета - научить модель не отвечать по изменяемым городским данным из памяти, а возвращать строгий JSON/tool-call к источнику истины: FastAPI/MCP, RAG/Qdrant или слой персоналий.

## Files

| File | Rows | Purpose |
| --- | ---: | --- |
| `data/router-train-v1.jsonl` | 810 | Combined SFT train pack. |
| `data/router-train-entities.jsonl` | 782 | Schools/kindergartens entity routing and adversarial false-premise examples. |
| `data/router-train-safety.jsonl` | 8 | Clarify, refuse, entity search, officials, RAG examples. |
| `data/router-train-history.jsonl` | 20 | Stable city history direct-answer and source-required examples. |
| `data/router-eval-v1.jsonl` | 30 | Held-out eval set, not included in train. |

## Format

Each row is JSONL with two chat messages:

```json
{
  "id": "router-train-entity-simple-002-01",
  "dataset": "router-train-entities",
  "source_id": "simple-002",
  "type": "entity_field",
  "messages": [
    {
      "role": "user",
      "content": "дай телефон школы № 2"
    },
    {
      "role": "assistant",
      "content": "{\"action\":\"tool_call\",\"tool\":\"get_entity_field\",\"args\":{\"layer\":\"schools\",\"inn\":\"1215067590\",\"field\":\"phone\"}}"
    }
  ],
  "tags": ["entity_field", "schools", "phone"]
}
```

For schools and kindergartens, assistant messages do not contain final facts such as phone numbers, addresses, websites, or head names. They contain only routing decisions.

## Contract

Allowed actions:

- `tool_call`
- `clarify`
- `refuse`
- `direct_answer`

Allowed tools:

- `get_entity_field`
- `search_entities`
- `rag_search`
- `get_current_official`
- `get_official_by_date`

The full contract and validators are maintained in the GitHub repository:

`https://github.com/yasg1988/iola-small-model-lab`

## Validation

The source repository validates:

- strict assistant JSON;
- allowed actions, tools, layers and fields;
- `get_entity_field` arguments;
- no exact train/eval question overlap.

Commands:

```powershell
npm run generate:router
npm run validate:router
npm test
```
