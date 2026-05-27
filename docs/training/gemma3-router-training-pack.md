# Training pack для Gemma 3 1B router

Дата: 2026-05-27.

Базовая модель для первого прогона: `gemma3:1b`.

Цель обучения: не зашить изменяемые городские факты в веса модели, а научить маленькую модель строго выбирать действие и инструмент:

- `get_entity_field` для школ и детских садов по `layer + inn + field`;
- `search_entities`, если объект задан нестрого;
- `rag_search`, если нужен источник, справка или документ;
- `get_current_official` / `get_official_by_date` для персоналий;
- `clarify` и `refuse`, когда данных недостаточно или поле не публичное;
- `direct_answer` только для стабильных исторических фактов.

## Файлы

| Файл | Строк | Назначение |
| --- | ---: | --- |
| `datasets/router-train-entities.jsonl` | 782 | Учреждения, поля, ИНН, вопросы с ложными предпосылками. |
| `datasets/router-train-safety.jsonl` | 8 | Уточнения, отказы, поиск учреждений, персоналии и RAG. |
| `datasets/router-train-history.jsonl` | 20 | Стабильные исторические факты и source-required сценарии. |
| `datasets/router-train-v1.jsonl` | 810 | Объединенный SFT pack для обучения. |
| `datasets/router-eval-v1.jsonl` | 30 | Отложенный eval-набор, не входит в train. |
| `datasets/router-train-v2.jsonl` | 810 | V2 pack: модель возвращает `entity_number`, а не запоминает ИНН. |
| `datasets/router-eval-v2.jsonl` | 30 | V2 eval с `resolve_entity_field`. |

## Формат

Каждая строка - JSONL SFT-запись:

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

Важно: для школ и детских садов в assistant-ответе нет значений телефонов, адресов, ФИО руководителей или сайтов. Модель учится только маршрутизировать запрос к источнику истины.

## V2 contract

Первый LoRA-прогон по `router-train-v1.jsonl` показал, что 1B-модель стабильно учит JSON-формат, но плохо запоминает произвольные ИНН: третий запуск дал `json_ok 30/30`, но `exact_ok 1/30`.

Поэтому добавлен V2-набор: модель возвращает номер или название учреждения, а API уже резолвит ИНН в справочнике.

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

Это ближе к продуктовой архитектуре: веса модели не являются источником справочника, а только выбирают слой, поле и понятный идентификатор объекта.

## Генерация и проверка

```powershell
npm run generate:router
npm run validate:router
npm test
```

Валидатор проверяет:

- assistant output является одним JSON-объектом;
- `action`, `tool`, `layer`, `field` входят в разрешенный контракт;
- `get_entity_field` содержит слой, ИНН и поле;
- `rag_search` содержит query;
- `clarify`, `refuse`, `direct_answer` имеют нужные текстовые поля;
- train не содержит точных вопросов из `router-eval-v1`.

## Первый прогон обучения

Hugging Face dataset:

`https://huggingface.co/datasets/LMSerg/iola-gemma3-router-sft`

Следующий шаг - обучить LoRA-адаптер под `gemma3:1b` на `data/router-train-v1.jsonl`.

После обучения проверяем не красивость ответа, а контракт:

```powershell
npm run run:router -- --model gemma3:1b --limit 30
```

Критерий приемки первого адаптера:

- JSON parse: 30/30;
- strict exact match по `router-eval-v1`: существенно выше сырого baseline `2/30`;
- для школ и детских садов модель не отвечает фактом из памяти, а возвращает tool-call.
