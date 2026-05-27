#!/usr/bin/env node
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { DEFAULT_MODELS, listModels, pullModel } from '../lib/ollama.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

const COMMANDS = new Map([
  ['generate', { script: 'scripts/generate-datasets.js', description: 'Generate JSONL datasets from the public API.' }],
  ['run-baseline', { script: 'scripts/run-evaluation.js', description: 'Run deterministic strategy baselines.' }],
  ['run-ollama', { script: 'scripts/run-ollama-evaluation.js', description: 'Run real Ollama model evaluation.' }],
  ['summary', { script: 'scripts/summarize-results.js', description: 'Print the latest baseline summary table.' }],
  ['summary-ollama', { script: 'scripts/summarize-ollama-results.js', description: 'Print the latest Ollama summary table.' }],
]);

function help() {
  console.log(`iola-model-lab

Usage:
  iola-model-lab <command> [options]

Commands:
  models          List locally available Ollama models
  pull            Pull default or selected Ollama models
  generate        Generate datasets
  run-baseline    Run deterministic baseline concepts
  run-ollama      Run real Ollama models
  summary         Print latest baseline summary
  summary-ollama  Print latest Ollama summary

Examples:
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
    console.log('Ollama is reachable, but no models are installed.');
    return;
  }
  for (const model of models) {
    const sizeGb = model.size ? `${(model.size / 1024 / 1024 / 1024).toFixed(2)} GB` : '';
    console.log(`${model.name || model.model}\t${sizeGb}`);
  }
}

async function pullModels(args) {
  for (const model of selectedModels(args)) {
    console.log(`Pulling ${model}...`);
    await pullModel(model);
    console.log(`Pulled ${model}`);
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
    console.error(`Unknown command: ${command}`);
    help();
    process.exit(1);
  }
  await runNodeScript(entry.script, args);
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
