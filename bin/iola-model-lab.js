#!/usr/bin/env node
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { DEFAULT_MODELS, listModels, pullModel } from '../lib/ollama.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

const COMMANDS = new Map([
  ['generate', { script: 'scripts/generate-datasets.js', description: 'Создать JSONL-датасеты из публичного API.' }],
  ['run-baseline', { script: 'scripts/run-evaluation.js', description: 'Запустить детерминированные базовые стратегии.' }],
  ['run-ollama', { script: 'scripts/run-ollama-evaluation.js', description: 'Запустить оценку реальных моделей Ollama.' }],
  ['summary', { script: 'scripts/summarize-results.js', description: 'Показать последнюю сводку базового прогона.' }],
  ['summary-ollama', { script: 'scripts/summarize-ollama-results.js', description: 'Показать последнюю сводку Ollama-прогона.' }],
]);

function help() {
  console.log(`iola-model-lab

Использование:
  iola-model-lab <command> [options]

Команды:
  models          Показать локально доступные модели Ollama
  pull            Скачать модели Ollama по умолчанию или из --models
  generate        Создать датасеты
  run-baseline    Запустить детерминированные базовые концепции
  run-ollama      Запустить реальные модели Ollama
  summary         Показать последнюю сводку базового прогона
  summary-ollama  Показать последнюю сводку Ollama-прогона

Примеры:
  iola-model-lab pull --models qwen3:0.6b
  iola-model-lab run-ollama --models qwen3:0.6b --modes mcp-context,verified-context --dataset simple-facts --limit 5
  iola-model-lab run-baseline --all
`);
}

function argValue(args, name) {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : null;
}

function selectedModels(args) {
  const value = argValue(args, '--models');
  if (!value) return DEFAULT_MODELS;
  return value.split(',').map((item) => item.trim()).filter(Boolean);
}

async function runNodeScript(script, args) {
  const child = spawn(process.execPath, [path.join(ROOT, script), ...args], {
    cwd: ROOT,
    stdio: 'inherit',
    env: process.env,
  });
  const code = await new Promise((resolve) => child.on('close', resolve));
  if (code !== 0) process.exit(code ?? 1);
}

async function listLocalModels() {
  const models = await listModels();
  if (!models.length) {
    console.log('Ollama доступен, но модели не установлены.');
    return;
  }
  for (const model of models) {
    const sizeGb = model.size ? `${(model.size / 1024 / 1024 / 1024).toFixed(2)} GB` : '';
    console.log(`${model.name || model.model}\t${sizeGb}`);
  }
}

async function pullModels(args) {
  for (const model of selectedModels(args)) {
    console.log(`Скачиваю ${model}...`);
    await pullModel(model);
    console.log(`Скачано ${model}`);
  }
}

async function main() {
  const [command, ...args] = process.argv.slice(2);
  if (!command || command === 'help' || command === '--help' || command === '-h') {
    help();
    return;
  }

  if (command === 'models') {
    await listLocalModels();
    return;
  }

  if (command === 'pull') {
    await pullModels(args);
    return;
  }

  const entry = COMMANDS.get(command);
  if (!entry) {
    console.error(`Неизвестная команда: ${command}`);
    help();
    process.exit(1);
  }
  await runNodeScript(entry.script, args);
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
