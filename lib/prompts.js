import { FIELD_LABELS, LAYER_LABELS } from './common.js';

function factBlock(row) {
  const layer = LAYER_LABELS[row.layer]?.singular || row.layer;
  return JSON.stringify({
    layer,
    entity: row.entityName,
    entityNumber: row.entityNumber,
    field: FIELD_LABELS[row.field] || row.field,
    value: row.expected,
    hasFalseUserValue: Boolean(row.falseValue),
    falseUserValue: row.falseValue || null,
  }, null, 2);
}

export function buildPrompt(row, mode) {
  if (mode === 'direct') {
    return [
      'Ты ассистент по открытым данным Йошкар-Олы.',
      'Ответь на вопрос пользователя кратко на русском языке.',
      'Если точных данных нет, честно скажи, что не знаешь.',
      '',
      `Вопрос: ${row.question}`,
      'Ответ:',
    ].join('\n');
  }

  if (mode === 'mcp-context') {
    return [
      'Ты ассистент по открытым данным Йошкар-Олы.',
      'Отвечай только на основе FACT. Не добавляй сведения из памяти.',
      'Если пользователь утверждает другое значение, сравни его с FACT.',
      '',
      `FACT:\n${factBlock(row)}`,
      '',
      `Вопрос: ${row.question}`,
      'Ответ:',
    ].join('\n');
  }

  if (mode === 'verified-context') {
    return [
      'Ты строгий проверяющий фактов по данным Йошкар-Олы.',
      'Используй только FACT.',
      'Правила:',
      '1. Если FACT.hasFalseUserValue=true, начни с "Нет, это не подтверждается."',
      '2. Если FACT.hasFalseUserValue=false, не начинай ответ с отрицания.',
      '3. Затем дай правильное значение из FACT.',
      '4. Не повторяй ложное значение как подтвержденное.',
      '5. Ответ должен быть коротким.',
      '',
      `FACT:\n${factBlock(row)}`,
      '',
      `Вопрос: ${row.question}`,
      'Ответ:',
    ].join('\n');
  }

  throw new Error(`Unknown prompt mode: ${mode}`);
}

export const PROMPT_MODES = ['direct', 'mcp-context', 'verified-context'];
