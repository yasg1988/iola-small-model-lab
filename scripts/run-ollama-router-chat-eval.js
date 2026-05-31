import fs from "node:fs/promises";
import path from "node:path";

const ROOT = process.cwd();
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || "http://127.0.0.1:11434";
const ROUTER_SYSTEM_PROMPT = `You are the IOLA CLI router for public open data of Yoshkar-Ola.
Return exactly one JSON object. Do not use markdown. Do not add explanations.
Allowed actions: tool_call, clarify, refuse, direct_answer.
Allowed tools: resolve_entity_field, search_entities, rag_search, get_current_official, get_official_by_date.
For schools and kindergartens, do not answer mutable facts from memory. Use resolve_entity_field or search_entities.
Allowed organization layers: schools, kindergartens.
Allowed organization fields: name, inn, address, email, website, phone, head, license_status.
If the user asks for a source about city history, use rag_search. If no source is requested and the fact is stable, direct_answer is allowed.
If the request is ambiguous, return clarify. If the requested field is not public, return refuse with reason field_not_public.
If the user asks to confirm a possibly false value, call the exact tool and pass that value as args.must_refute_user_value.`;

function argValue(name, fallback = "") {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

async function readJsonl(filePath) {
  const raw = await fs.readFile(filePath, "utf8");
  return raw.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}

function extractJson(text) {
  const raw = String(text || "").trim();
  try {
    return { ok: true, value: JSON.parse(raw), raw };
  } catch {
    const start = raw.indexOf("{");
    const end = raw.lastIndexOf("}");
    if (start >= 0 && end > start) {
      try {
        return { ok: true, value: JSON.parse(raw.slice(start, end + 1)), raw };
      } catch {
        return { ok: false, value: null, raw };
      }
    }
    return { ok: false, value: null, raw };
  }
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

async function callOllama(model, question, options) {
  const userContent = options.qwenNoThink ? `${question}\n/no_think` : question;
  const messages = options.system
    ? [
        { role: "system", content: ROUTER_SYSTEM_PROMPT },
        { role: "user", content: userContent },
      ]
    : [{ role: "user", content: userContent }];
  const started = performance.now();
  const response = await fetch(`${OLLAMA_BASE_URL}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model,
      stream: false,
      think: options.qwenNoThink ? false : undefined,
      messages,
      options: {
        temperature: 0,
        num_predict: options.numPredict,
      },
    }),
  });
  const latencyMs = Number((performance.now() - started).toFixed(2));
  if (!response.ok) {
    throw new Error(`Ollama chat failed: ${response.status} ${response.statusText}\n${await response.text()}`);
  }
  const payload = await response.json();
  return {
    answer: payload.message?.content || payload.message?.thinking || "",
    thinking: payload.message?.thinking || "",
    latencyMs,
    evalCount: payload.eval_count || 0,
    evalDurationMs: payload.eval_duration ? Math.round(payload.eval_duration / 1_000_000) : 0,
    totalDurationMs: payload.total_duration ? Math.round(payload.total_duration / 1_000_000) : 0,
  };
}

const model = argValue("--model", "iola-router:1b-v9-q4-rc");
const datasetName = argValue("--dataset", "router-eval-v9").replace(/\.jsonl$/u, "");
const numPredict = Number(argValue("--num-predict", "180"));
const limit = Number(argValue("--limit", "0"));
const useSystem = !process.argv.includes("--no-system");
const qwenNoThink = process.argv.includes("--qwen-no-think");
const datasetPath = path.join(ROOT, "datasets", `${datasetName}.jsonl`);
const rows = (await readJsonl(datasetPath)).slice(0, limit || undefined);
const runId = new Date().toISOString().replace(/[:.]/gu, "-");
const outDir = path.join(ROOT, "results", "ollama-router-chat", runId);
await fs.mkdir(outDir, { recursive: true });

const evaluated = [];
for (const row of rows) {
  const generated = await callOllama(model, row.question, { system: useSystem, qwenNoThink, numPredict });
  const parsed = extractJson(generated.answer);
  const metrics = parsed.ok
    ? { validJson: true, ...evaluate(parsed.value, row.expected) }
    : { validJson: false, action: false, tool: false, args: false, answerContains: false, exact: false };
  evaluated.push({
    id: row.id,
    type: row.type,
    question: row.question,
    expected: row.expected,
    rawAnswer: generated.answer.trim(),
    rawThinking: generated.thinking.trim(),
    parsed: parsed.value,
    latencyMs: generated.latencyMs,
    evalCount: generated.evalCount,
    evalDurationMs: generated.evalDurationMs,
    totalDurationMs: generated.totalDurationMs,
    metrics,
  });
  process.stdout.write(metrics.exact ? "." : metrics.validJson ? "x" : "!");
}
process.stdout.write("\n");

const count = (name) => evaluated.filter((row) => row.metrics[name]).length;
const summary = {
  runId,
  model,
  dataset: datasetName,
  systemPrompt: useSystem,
  qwenNoThink,
  numPredict,
  limit,
  total: evaluated.length,
  validJson: count("validJson"),
  exact: count("exact"),
  exactRate: Number((count("exact") / evaluated.length).toFixed(4)),
  jsonRate: Number((count("validJson") / evaluated.length).toFixed(4)),
  avgLatencyMs: Number((evaluated.reduce((sum, row) => sum + row.latencyMs, 0) / evaluated.length).toFixed(2)),
  avgEvalDurationMs: Number((evaluated.reduce((sum, row) => sum + row.evalDurationMs, 0) / evaluated.length).toFixed(2)),
};

await fs.writeFile(path.join(outDir, "predictions.jsonl"), `${evaluated.map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");
await fs.writeFile(path.join(outDir, "summary.json"), JSON.stringify(summary, null, 2), "utf8");
await fs.writeFile(path.join(ROOT, "results", "ollama-router-chat", "latest-summary.json"), JSON.stringify(summary, null, 2), "utf8");

console.log(JSON.stringify(summary, null, 2));
