import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { containsNormalized, ensureDir, readJsonl, writeJsonl } from '../lib/common.js';
import { DEFAULT_MODELS, generate } from '../lib/ollama.js';
import { buildPrompt, PROMPT_MODES } from '../lib/prompts.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const DATASET_DIR = path.join(ROOT, 'datasets');
const RESULT_DIR = path.join(ROOT, 'results', 'ollama-runs');

function argValue(name) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : null;
}

function csvArg(name, fallback) {
  const value = argValue(name);
  if (!value) return fallback;
  return value.split(',').map((item) => item.trim()).filter(Boolean);
}

function selectedDatasets() {
  const value = argValue('--dataset');
  if (!value || value === 'all') return ['simple-facts.jsonl', 'adversarial-facts.jsonl'];
  return [`${value}.jsonl`.replace(/\.jsonl\.jsonl$/u, '.jsonl')];
}

function evaluate(row, answer) {
  const hasExpected = containsNormalized(answer, row.expected);
  const affirmsFalse = row.falseValue
    ? containsNormalized(answer, row.falseValue) && !/не подтверж|нет|на самом деле|по данным|правильн|факт/iu.test(answer)
    : false;
  const refuted = row.shouldRefute
    ? /не подтверж|нет|на самом деле|по данным|правильн|факт/iu.test(answer) && hasExpected
    : true;
  return {
    correct: hasExpected && !affirmsFalse && refuted,
    hasExpected,
    affirmsFalse,
    refuted,
  };
}

function summarize(rows) {
  const total = rows.length;
  const count = (name) => rows.filter((row) => row.metrics[name]).length;
  return {
    total,
    correct: count('correct'),
    accuracy: Number((count('correct') / total).toFixed(4)),
    hasExpected: count('hasExpected'),
    affirmsFalse: count('affirmsFalse'),
    refuted: count('refuted'),
    avgLatencyMs: Number((rows.reduce((sum, row) => sum + row.latencyMs, 0) / total).toFixed(2)),
    avgEvalDurationMs: Number((rows.reduce((sum, row) => sum + row.evalDurationMs, 0) / total).toFixed(2)),
  };
}

const models = csvArg('--models', DEFAULT_MODELS);
const modes = csvArg('--modes', PROMPT_MODES);
const limit = Number(argValue('--limit') || 0);

const runId = new Date().toISOString().replace(/[:.]/gu, '-');
const runDir = path.join(RESULT_DIR, runId);
await ensureDir(runDir);

const summary = {
  runId,
  createdAt: new Date().toISOString(),
  models,
  modes,
  limit: limit || null,
  results: {},
};

for (const datasetFile of selectedDatasets()) {
  const datasetName = datasetFile.replace(/\.jsonl$/u, '');
  const allRows = await readJsonl(path.join(DATASET_DIR, datasetFile));
  const datasetRows = limit > 0 ? allRows.slice(0, limit) : allRows;

  for (const model of models) {
    for (const mode of modes) {
      const evaluatedRows = [];
      for (const row of datasetRows) {
        const prompt = buildPrompt(row, mode);
        const started = performance.now();
        const generated = await generate(model, prompt);
        const latencyMs = Number((performance.now() - started).toFixed(2));
        evaluatedRows.push({
          id: row.id,
          dataset: datasetName,
          model,
          mode,
          question: row.question,
          expected: row.expected,
          falseValue: row.falseValue,
          answer: generated.answer.trim(),
          latencyMs,
          evalCount: generated.evalCount,
          evalDurationMs: generated.evalDurationMs,
          totalDurationMs: generated.totalDurationMs,
          metrics: evaluate(row, generated.answer),
        });
        process.stdout.write('.');
      }
      process.stdout.write(` ${model} ${mode} ${datasetName}\n`);
      const outputFile = path.join(runDir, `${model.replace(/[:/\\]/gu, '_')}-${mode}-${datasetName}.jsonl`);
      await writeJsonl(outputFile, evaluatedRows);
      summary.results[`${model}:${mode}:${datasetName}`] = {
        model,
        mode,
        dataset: datasetName,
        file: path.relative(ROOT, outputFile).replace(/\\/gu, '/'),
        ...summarize(evaluatedRows),
      };
    }
  }
}

await fs.writeFile(path.join(runDir, 'summary.json'), JSON.stringify(summary, null, 2), 'utf8');
await fs.writeFile(path.join(RESULT_DIR, 'latest-summary.json'), JSON.stringify(summary, null, 2), 'utf8');

console.log(`Saved Ollama results to ${path.relative(process.cwd(), runDir)}`);
