# MoE/Sparse-модели и раннеры для следующего этапа

## Главная мысль

MoE/Sparse-модель экономит вычисления на токен, потому что активирует только часть экспертов. Но локальный запуск на слабом ПК ограничен не только активными параметрами. Нужно учитывать:

- общий размер модели;
- размер квантованного файла;
- RAM/VRAM;
- поддержку MoE в раннере;
- возможность держать экспертов на CPU/GPU;
- скорость на конкретном устройстве.

## Кандидаты

| Кандидат | Тип | Почему интересен | Риск для текущего ПК |
| --- | --- | --- | --- |
| Mixtral 8x7B | MoE, 46.7B всего, около 12.9B активных | классический открытый SMoE, хорошая база для понимания MoE | общий размер слишком велик для 8 ГБ RAM |
| Qwen3-30B-A3B | MoE, около 30B всего, около 3B активных | активных параметров мало, хорошая гипотеза для sparse-инференса | общий размер 30B может быть слишком тяжелым |
| Qwen3-235B-A22B | MoE, 235B всего, около 22B активных | эталон крупного Qwen MoE | не подходит для этого ПК |
| DeepSeek-MoE/V3-подобные | MoE | интересны как архитектурный ориентир | не для локального RedmiBook 15 |

## Раннеры

| Раннер | Зачем смотреть |
| --- | --- |
| Ollama | простой UX, уже используется, но скрывает часть низкоуровневых настроек |
| llama.cpp | важен для GGUF, Qwen/Qwen3MoE и точного контроля параметров запуска |
| Hugging Face Transformers | нужен для моделей, которых нет в Ollama, но на Windows+CPU может быть тяжелым |
| LM Studio | удобен для ручной проверки GGUF-моделей |

## План проверки без риска

1. Не скачивать сразу 30B+.
2. Сначала собрать список GGUF-квантов и их размеров.
3. Проверить, поддерживает ли выбранный раннер конкретную MoE-архитектуру.
4. Запускать только короткий health-prompt.
5. Для текущего ПК не выходить за класс, где файл модели заметно больше 7-8 ГБ, без отдельного подтверждения пользователя.

## Источники

- Mistral AI: [Mixtral of Experts](https://mistral.ai/en/news/mixtral-of-experts)
- Mistral Docs: [Mixtral 8x7B model card](https://docs.mistral.ai/models/model-cards/mixtral-8x7b-0-1)
- arXiv: [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- Qwen: [Open Foundation Models](https://qwen.moe/)
- Qwen Docs: [Key concepts](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html)
- Qwen Docs: [llama.cpp локальный запуск](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html)
- NVIDIA NeMo: [Qwen3 MoE coverage](https://docs.nvidia.com/nemo/automodel/nightly/model-coverage/llm/qwen/qwen3-moe.html)
