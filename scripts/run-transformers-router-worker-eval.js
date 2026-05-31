import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";

const ROOT = process.cwd();
const DEFAULT_PYTHON = "C:\\Users\\YAkunin\\.iola\\model-runtime\\Scripts\\python.exe";
const DEFAULT_BASE_DIR = "D:\\new_adm_iola\\tmp\\iola-ollama-build\\base-gemma-3-1b-it";
const DEFAULT_ADAPTER_DIR = "D:\\new_adm_iola\\tmp\\iola-router-v9-lora";

function argValue(name, fallback = "") {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

function hasArg(name) {
  return process.argv.includes(name);
}

async function readJsonl(filePath) {
  const raw = await fs.readFile(filePath, "utf8");
  return raw.split(/\r?\n/u).filter(Boolean).map((line) => JSON.parse(line));
}

function scalarEqual(actual, expected) {
  return String(actual ?? "") === String(expected ?? "");
}

function includesAll(value, parts) {
  const text = String(value || "").toLocaleLowerCase("ru-RU");
  return parts.every((part) => text.includes(String(part).toLocaleLowerCase("ru-RU")));
}

function evaluate(actual, expected) {
  if (!actual || typeof actual !== "object") {
    return { action: false, tool: false, args: false, answerContains: false, exact: false };
  }

  const action = actual.action === expected.action;
  const tool = expected.tool ? actual.tool === expected.tool : true;
  let args = true;

  if (expected.args) {
    for (const [key, expectedValue] of Object.entries(expected.args)) {
      const actualValue = actual.args?.[key];
      if (Array.isArray(expectedValue)) {
        args &&= Array.isArray(actualValue)
          && actualValue.length === expectedValue.length
          && expectedValue.every((item, index) => scalarEqual(actualValue[index], item));
      } else {
        args &&= scalarEqual(actualValue, expectedValue);
      }
    }
  }

  if (expected.reason) {
    args &&= scalarEqual(actual.reason, expected.reason);
  }

  const answerContains = expected.answer_contains ? includesAll(actual.answer, expected.answer_contains) : true;
  return { action, tool, args, answerContains, exact: action && tool && args && answerContains };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function requestJson(url, init = {}) {
  const response = await fetch(url, {
    ...init,
    headers: { "content-type": "application/json", ...(init.headers || {}) },
  });
  const text = await response.text();
  let payload = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    payload = { raw: text };
  }
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }
  return payload;
}

async function waitForWorker(baseUrl, child, timeoutMs) {
  const started = performance.now();
  let lastError = "";
  while (performance.now() - started < timeoutMs) {
    if (child.exitCode !== null) {
      throw new Error(`Worker exited with code ${child.exitCode}`);
    }
    try {
      return await requestJson(`${baseUrl}/health`);
    } catch (error) {
      lastError = error.message;
      await sleep(1000);
    }
  }
  throw new Error(`Worker did not become ready in ${timeoutMs}ms. Last error: ${lastError}`);
}

function stopWorker(child) {
  if (child.exitCode === null) {
    child.kill("SIGTERM");
  }
}

const python = argValue("--python", process.env.IOLA_PYTHON || DEFAULT_PYTHON);
const baseDir = argValue("--base-dir", DEFAULT_BASE_DIR);
const adapterDir = argValue("--adapter-dir", DEFAULT_ADAPTER_DIR);
const port = Number(argValue("--port", "8765"));
const threads = Number(argValue("--threads", "4"));
const maxNewTokens = Number(argValue("--max-new-tokens", "128"));
const dtype = argValue("--dtype", "bfloat16");
const keepAlive = hasArg("--keep-alive");
const limit = Number(argValue("--limit", "0"));
const datasetName = argValue("--dataset", "router-eval-v9").replace(/\.jsonl$/u, "");
const datasetPath = path.join(ROOT, "datasets", `${datasetName}.jsonl`);
const workerUrl = `http://127.0.0.1:${port}`;

const workerArgs = [
  "scripts/transformers_router_worker.py",
  "--base-dir", baseDir,
  "--adapter-dir", adapterDir,
  "--host", "127.0.0.1",
  "--port", String(port),
  "--threads", String(threads),
  "--max-new-tokens", String(maxNewTokens),
  "--dtype", dtype,
];

const child = spawn(python, workerArgs, {
  cwd: ROOT,
  env: { ...process.env, PYTHONIOENCODING: "utf-8", TOKENIZERS_PARALLELISM: "false" },
  stdio: ["ignore", "pipe", "pipe"],
});

child.stdout.on("data", (chunk) => process.stdout.write(`[worker] ${chunk}`));
child.stderr.on("data", (chunk) => process.stderr.write(`[worker] ${chunk}`));

let health;
try {
  health = await waitForWorker(workerUrl, child, 240_000);
  const rows = (await readJsonl(datasetPath)).slice(0, limit || undefined);
  const runId = `${new Date().toISOString().replace(/[:.]/gu, "-")}-${process.pid}`;
  const outDir = path.join(ROOT, "results", "transformers-router-worker", runId);
  await fs.mkdir(outDir, { recursive: true });

  const evaluated = [];
  for (const row of rows) {
    const started = performance.now();
    const generated = await requestJson(`${workerUrl}/generate`, {
      method: "POST",
      body: JSON.stringify({ question: row.question, maxNewTokens }),
    });
    const requestLatencyMs = Number((performance.now() - started).toFixed(2));
    const metrics = generated.parsed
      ? { validJson: true, ...evaluate(generated.parsed, row.expected) }
      : { validJson: false, action: false, tool: false, args: false, answerContains: false, exact: false };
    evaluated.push({
      id: row.id,
      type: row.type,
      question: row.question,
      expected: row.expected,
      rawAnswer: String(generated.answer || "").trim(),
      parsed: generated.parsed || null,
      latencyMs: generated.latencyMs,
      requestLatencyMs,
      generatedTokens: generated.generatedTokens || 0,
      metrics,
    });
    process.stdout.write(metrics.exact ? "." : metrics.validJson ? "x" : "!");
  }
  process.stdout.write("\n");

  const count = (name) => evaluated.filter((row) => row.metrics[name]).length;
  const summary = {
    runId,
    runtime: "transformers-worker",
    baseDir,
    adapterDir,
    dataset: datasetName,
    total: evaluated.length,
    validJson: count("validJson"),
    exact: count("exact"),
    exactRate: Number((count("exact") / evaluated.length).toFixed(4)),
    jsonRate: Number((count("validJson") / evaluated.length).toFixed(4)),
    avgLatencyMs: Number((evaluated.reduce((sum, row) => sum + row.latencyMs, 0) / evaluated.length).toFixed(2)),
    avgRequestLatencyMs: Number((evaluated.reduce((sum, row) => sum + row.requestLatencyMs, 0) / evaluated.length).toFixed(2)),
    avgGeneratedTokens: Number((evaluated.reduce((sum, row) => sum + row.generatedTokens, 0) / evaluated.length).toFixed(2)),
    loadSeconds: health.loadSeconds,
    threads,
    maxNewTokens,
    dtype,
  };

  await fs.writeFile(path.join(outDir, "predictions.jsonl"), `${evaluated.map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");
  await fs.writeFile(path.join(outDir, "summary.json"), JSON.stringify(summary, null, 2), "utf8");
  await fs.writeFile(path.join(ROOT, "results", "transformers-router-worker", "latest-summary.json"), JSON.stringify(summary, null, 2), "utf8");
  console.log(JSON.stringify(summary, null, 2));

  if (keepAlive) {
    console.log(`Worker is still running at ${workerUrl}`);
  } else {
    stopWorker(child);
  }
} catch (error) {
  stopWorker(child);
  console.error(error);
  process.exitCode = 1;
}
