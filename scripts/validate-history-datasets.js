import { readFile } from "node:fs/promises";
import path from "node:path";

const ROOT = process.cwd();
const DATASETS_DIR = path.join(ROOT, "datasets");

async function readJsonl(fileName) {
  const raw = await readFile(path.join(DATASETS_DIR, fileName), "utf8");
  return raw
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        throw new Error(`${fileName}:${index + 1}: ${error.message}`);
      }
    });
}

function fail(context, message) {
  throw new Error(`${context}: ${message}`);
}

function isJsonToolCall(value) {
  try {
    const parsed = JSON.parse(value);
    return parsed.action === "tool_call" && parsed.tool === "rag_search";
  } catch {
    return false;
  }
}

const files = [
  ["history-train-v1.jsonl", "history-eval-v1.jsonl"],
  ["history-train-v2.jsonl", "history-eval-v2.jsonl"],
];

for (const [trainFile, evalFile] of files) {
  const train = await readJsonl(trainFile);
  const evalRows = await readJsonl(evalFile);
  const trainQuestions = new Set(
    train.map((item) => item.messages[0].content.replace(/\s+/g, " ").trim().toLowerCase()),
  );
  const ids = new Set();

  for (const [index, item] of train.entries()) {
    const context = `${trainFile}:${index + 1}:${item.id}`;
    if (ids.has(item.id)) {
      fail(context, "duplicate id");
    }
    ids.add(item.id);

    if (!Array.isArray(item.messages) || item.messages.length !== 2) {
      fail(context, "messages must contain user and assistant");
    }
    if (!item.messages[0].content || !item.messages[1].content) {
      fail(context, "messages must be non-empty");
    }
    if (item.type === "source_required" && !isJsonToolCall(item.messages[1].content)) {
      fail(context, "source_required answer must be rag_search JSON");
    }
    if (item.type === "direct_answer" && item.messages[1].content.trim().startsWith("{")) {
      fail(context, "direct_answer must be natural text");
    }
  }

  for (const [index, item] of evalRows.entries()) {
    const context = `${evalFile}:${index + 1}:${item.id}`;
    if (!item.question || !item.expected) {
      fail(context, "question and expected are required");
    }
    if (trainQuestions.has(item.question.replace(/\s+/g, " ").trim().toLowerCase())) {
      fail(context, "eval question overlaps train");
    }
    if (item.type === "source_required" && item.expected.tool !== "rag_search") {
      fail(context, "source_required expected must be rag_search");
    }
  }

  console.log(`${trainFile}: ${train.length} rows`);
  console.log(`${evalFile}: ${evalRows.length} rows`);
}

console.log("History datasets are valid");
