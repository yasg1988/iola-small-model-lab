# memory-verified

## Что это

`memory-verified` - стратегия, где внешний поиск факта обязательно дополняется проверкой ответа.

## На какие исследования опираемся

- RAG: [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- Chain-of-Verification: [Reduce Hallucination via Chain-of-Verification](https://arxiv.org/abs/2309.11495)
- Self-RAG: [Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)

## Почему взяли

`conditional-memory` быстро находит факт, но сама по себе не решает проблему ложной предпосылки. Нужна связка "память + проверка".

## Гипотеза

Добавление проверяющего сохранит скорость retrieval-подхода и уберет главный риск ложного подтверждения.

## Что проверяем

- Работает ли внешняя память на обычных вопросах.
- Закрывает ли проверяющий adversarial-вопросы.
- Подходит ли схема для маленьких моделей.

## Результат

| Датасет | Верно | Вывод |
| --- | ---: | --- |
| `simple-facts` | 100/100 | работает |
| `adversarial-facts` | 100/100 | работает |

## Решение

Использовать как надежный режим, когда нужен баланс скорости и безопасности.
