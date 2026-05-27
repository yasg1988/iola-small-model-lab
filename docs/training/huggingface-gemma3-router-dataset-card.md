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
| `data/router-train-v2.jsonl` | 810 | V2 train pack: model returns `entity_number/entity_name`; API resolves INN. |
| `data/router-eval-v2.jsonl` | 30 | V2 held-out eval set. |
| `data/router-train-v3.jsonl` | 829 | V3 train pack with extra alias/search/clarify/refuse/RAG/officials examples. |
| `data/router-eval-v3.jsonl` | 30 | V3 held-out eval; alias queries expect `entity_name`. |
| `data/router-train-v4.jsonl` | 856 | V4 train pack with targeted fixes for V3 failures. |
| `data/router-eval-v4.jsonl` | 30 | V4 held-out eval on the V3 contract surface. |
| `data/router-train-v5.jsonl` | 890 | V5 train pack with targeted fixes for V4 failures. |
| `data/router-eval-v5.jsonl` | 30 | V5 held-out eval on the V4 contract surface. |
| `data/router-train-v6.jsonl` | 910 | V6 train pack with salary/private-data refusal reinforcement. |
| `data/router-eval-v6.jsonl` | 30 | V6 held-out eval on the V5 contract surface. |

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

V2 avoids forcing the 1B model to memorize arbitrary INNs:

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
