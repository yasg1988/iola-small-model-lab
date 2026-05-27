import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ensureDir, readJsonl, writeJsonl } from '../lib/common.js';
import { generate } from '../lib/ollama.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const DATASET_DIR = path.join(ROOT, 'datasets');
const RESULT_DIR = path.join(ROOT, 'results', 'router-runs');

const DEFAULT_ROUTER_MODELS = ['gemma3:1b', 'qwen3:1.7b'];

function argValue(name) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : null;
}

function csvArg(name, fallback) {
  const value = argValue(name);
  if (!value) return fallback;
  return value.split(',').map((item) => item.trim()).filter(Boolean);
}

function buildPrompt(row) {
  return `Ты router-модель для городского CLI. /no_think
Верни строго один JSON-объект без markdown и без пояснений.

Разрешенные actions: tool_call, clarify, refuse, direct_answer.
Разрешенные tools: get_entity_field, search_entities, rag_search, get_current_official, get_official_by_date.
Разрешенные layers: schools, kindergartens, city_history, officials, documents.
Разрешенные fields: name, inn, address, email, website, phone, head, license_status.

Если нужен факт по школе или детскому саду, не отвечай сам: вызови get_entity_field или search_entities.
Если пользователь просит источник по истории, вызови rag_search.
Если запрос неоднозначный, верни clarify.
Если поле непубличное, верни refuse.
Если пользователь просит подтвердить неверное значение, вызови точный tool и передай ложное значение в args.must_refute_user_value.

Примеры:
Вопрос: дай телефон школы № 2
Ответ: {"action":"tool_call","tool":"get_entity_field","args":{"layer":"schools","inn":"1215067590","field":"phone"}}

Вопрос: дай телефон школы
Ответ: {"action":"clarify","question":"Уточните номер или название школы."}

Вопрос: сколько получает директор школы № 7?
Ответ: {"action":"refuse","reason":"field_not_public"}

Вопрос: когда основана Йошкар-Ола?
Ответ: {"action":"direct_answer","answer":"Йошкар-Ола основана в 1584 году."}

Вопрос: ${row.question}
Ответ:`;
}

function extractJson(text) {
  const raw = String(text || '').trim();
  const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/iu);
  const candidate = fenced ? fenced[1].trim() : raw;
  try {
    return { ok: true, value: JSON.parse(candidate), raw };
  } catch {
    const start = candidate.indexOf('{');
    const end = candidate.lastIndexOf('}');
    if (start >= 0 && end > start) {
      const slice = candidate.slice(start, end + 1);
      try {
        return { ok: true, value: JSON.parse(slice), raw };
      } catch {
        return { ok: false, value: null, raw };
      }
    }
    return { ok: false, value: null, raw };
  }
}

function getPath(value, dottedPath) {
  return dottedPath.split('.').reduce((current, part) => current?.[part], value);
}

function includesAll(value, expectedParts) {
  const text = String(value || '').toLowerCase();
  return expectedParts.every((part) => text.includes(String(part).toLowerCase()));
}

function equalScalar(actual, expected) {
  return String(actual ?? '') === String(expected ?? '');
}

function evaluateExpected(actual, expected) {
  if (!actual || typeof actual !== 'object') {
    return {
      action: false,
      tool: false,
      args: false,
      exact: false,
    };
  }

  const action = actual.action === expected.action;
  const tool = expected.tool ? actual.tool === expected.tool : true;

  let args = true;
  if (expected.args) {
    for (const [key, expectedValue] of Object.entries(expected.args)) {
      const actualValue = actual.args?.[key];
      if (Array.isArray(expectedValue)) {
        args &&= Array.isArray(actualValue)
          && expectedValue.length === actualValue.length
          && expectedValue.every((item, index) => equalScalar(actualValue[index], item));
      } else if (expectedValue && typeof expectedValue === 'object') {
        for (const [nestedKey, nestedExpected] of Object.entries(expectedValue)) {
          args &&= equalScalar(actualValue?.[nestedKey], nestedExpected);
        }
      } else {
        args &&= equalScalar(actualValue, expectedValue);
      }
    }
  }

  const answerContains = expected.answer_contains
    ? includesAll(actual.answer, expected.answer_contains)
    : true;

  return {
    action,
    tool,
    args,
    layer: expected.args?.layer ? getPath(actual, 'args.layer') === expected.args.layer : true,
    inn: expected.args?.inn ? getPath(actual, 'args.inn') === expected.args.inn : true,
    field: expected.args?.field ? getPath(actual, 'args.field') === expected.args.field : true,
    answerContains,
    exact: action && tool && args && answerContains,
  };
}

function summarize(rows) {
  const total = rows.length;
  const count = (name) => rows.filter((row) => row.metrics[name]).length;
  return {
    total,
    validJson: count('validJson'),
    exact: count('exact'),
    action: count('action'),
    tool: count('tool'),
    args: count('args'),
    layer: count('layer'),
    inn: count('inn'),
    field: count('field'),
    jsonRate: Number((count('validJson') / total).toFixed(4)),
    exactRate: Number((count('exact') / total).toFixed(4)),
    avgLatencyMs: Number((rows.reduce((sum, row) => sum + row.latencyMs, 0) / total).toFixed(2)),
  };
}

const models = csvArg('--models', DEFAULT_ROUTER_MODELS);
const limit = Number(argValue('--limit') || 0);
const datasetFile = `${argValue('--dataset') || 'router-eval-v1'}.jsonl`.replace(/\.jsonl\.jsonl$/u, '.jsonl');
const allRows = await readJsonl(path.join(DATASET_DIR, datasetFile));
const rows = limit > 0 ? allRows.slice(0, limit) : allRows;

const runId = new Date().toISOString().replace(/[:.]/gu, '-');
const runDir = path.join(RESULT_DIR, runId);
await ensureDir(runDir);

const summary = {
  runId,
  createdAt: new Date().toISOString(),
  dataset: datasetFile.replace(/\.jsonl$/u, ''),
  models,
  limit: limit || null,
  results: {},
};

for (const model of models) {
  const evaluatedRows = [];
  for (const row of rows) {
    const prompt = buildPrompt(row);
    const started = performance.now();
    let generated;
    try {
      generated = await generate(model, prompt, {
        format: 'json',
        temperature: 0,
        num_ctx: 2048,
        num_predict: 220,
      });
    } catch (error) {
      const latencyMs = Number((performance.now() - started).toFixed(2));
      evaluatedRows.push({
        id: row.id,
        model,
        question: row.question,
        expected: row.expected,
        error: error.message,
        latencyMs,
        metrics: {
          validJson: false,
          action: false,
          tool: false,
          args: false,
          exact: false,
        },
      });
      process.stdout.write('E');
      continue;
    }

    const latencyMs = Number((performance.now() - started).toFixed(2));
    const parsed = extractJson(generated.answer);
    const expectedMetrics = parsed.ok
      ? evaluateExpected(parsed.value, row.expected)
      : { action: false, tool: false, args: false, exact: false };
    evaluatedRows.push({
      id: row.id,
      model,
      type: row.type,
      question: row.question,
      expected: row.expected,
      rawAnswer: generated.answer.trim(),
      parsed: parsed.value,
      latencyMs,
      evalCount: generated.evalCount,
      evalDurationMs: generated.evalDurationMs,
      totalDurationMs: generated.totalDurationMs,
      metrics: {
        validJson: parsed.ok,
        ...expectedMetrics,
      },
    });
    process.stdout.write(parsed.ok ? (expectedMetrics.exact ? '.' : 'x') : '!');
  }
  process.stdout.write(` ${model}\n`);
  const outputFile = path.join(runDir, `${model.replace(/[:/\\]/gu, '_')}.jsonl`);
  await writeJsonl(outputFile, evaluatedRows);
  summary.results[model] = {
    model,
    file: path.relative(ROOT, outputFile).replace(/\\/gu, '/'),
    ...summarize(evaluatedRows),
  };
}

await fs.writeFile(path.join(runDir, 'summary.json'), JSON.stringify(summary, null, 2), 'utf8');
await fs.writeFile(path.join(RESULT_DIR, 'latest-summary.json'), JSON.stringify(summary, null, 2), 'utf8');

console.log(`Saved router results to ${path.relative(process.cwd(), runDir)}`);
