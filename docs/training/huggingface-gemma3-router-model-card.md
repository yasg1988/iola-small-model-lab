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
  - gemma3
---

# IOLA Gemma 3 Router LoRA

LoRA-адаптер для `google/gemma-3-1b-it`, обучаемый на датасете `LMSerg/iola-gemma3-router-sft`.

Цель модели - возвращать строгий JSON/tool-call для CLI и MCP/RAG-слоя Йошкар-Олы, а не отвечать по изменяемым городским данным из памяти.

## Expected output

```json
{
  "action": "tool_call",
  "tool": "get_entity_field",
  "args": {
    "layer": "schools",
    "inn": "1215067590",
    "field": "phone"
  }
}
```

## Training

Training script:

`training/train_gemma3_router_lora.py`

Source repository:

`https://github.com/yasg1988/iola-small-model-lab`

Dataset:

`https://huggingface.co/datasets/LMSerg/iola-gemma3-router-sft`

## Evaluation

After the first training job completes, this repo should contain:

- adapter files;
- tokenizer files;
- `eval/eval_metrics.json`;
- `eval/eval_predictions.jsonl`.

Acceptance target for the first adapter:

- JSON parse: `30/30`;
- strict exact match on `router-eval-v1`: substantially above raw `gemma3:1b` baseline `2/30`.
