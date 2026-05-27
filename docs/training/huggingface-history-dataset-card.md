---
license: mit
language:
  - ru
task_categories:
  - text-generation
tags:
  - sft
  - history
  - rag
  - yoshkar-ola
  - gemma3
pretty_name: IOLA History SFT
size_categories:
  - n<1K
---

# IOLA History SFT

SFT-набор для отдельного исторического LoRA-адаптера IOLA на базе `google/gemma-3-1b-it`.

## Files

| File | Rows | Purpose |
| --- | ---: | --- |
| `data/history-train-v1.jsonl` | 42 | Stable history SFT train rows. |
| `data/history-eval-v1.jsonl` | 16 | Held-out eval rows without exact train overlap. |
| `data/history-train-v2.jsonl` | 56 | V2 train rows with targeted fixes after first eval. |
| `data/history-eval-v2.jsonl` | 16 | V2 held-out eval rows. |
| `data/history-train-v3.jsonl` | 65 | V3 train rows with targeted fixes after second eval. |
| `data/history-eval-v3.jsonl` | 16 | V3 held-out eval rows. |

## Behavior

The adapter should:

- answer stable historical facts as natural text;
- return `rag_search` JSON when the user asks for a source;
- ask for clarification for ambiguous rename questions.

Source repository:

`https://github.com/yasg1988/iola-small-model-lab`
