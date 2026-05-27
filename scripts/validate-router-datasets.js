import { readFile } from "node:fs/promises";
import path from "node:path";

const ROOT = process.cwd();
const DATASETS_DIR = path.join(ROOT, "datasets");

const TRAIN_FILES = [
  "router-train-entities.jsonl",
  "router-train-safety.jsonl",
  "router-train-history.jsonl",
  "router-train-v1.jsonl",
  "router-train-entities-v2.jsonl",
  "router-train-safety-v2.jsonl",
  "router-train-history-v2.jsonl",
  "router-train-v2.jsonl",
  "router-train-v3.jsonl",
  "router-train-v4.jsonl",
  "router-train-v5.jsonl",
];

const ALLOWED_ACTIONS = new Set(["tool_call", "clarify", "refuse", "direct_answer"]);
const ALLOWED_TOOLS = new Set([
  "get_entity_field",
  "resolve_entity_field",
  "search_entities",
  "rag_search",
  "get_current_official",
  "get_official_by_date",
]);
const ALLOWED_LAYERS = new Set([
  "schools",
  "kindergartens",
  "city_history",
  "officials",
  "documents",
]);
const ALLOWED_ENTITY_FIELDS = new Set([
  "name",
  "inn",
  "address",
  "email",
  "website",
  "phone",
  "head",
  "license_status",
]);

function normalizeQuestion(value) {
  return String(value ?? "")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}

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

function validateMessages(row, context) {
  if (!Array.isArray(row.messages) || row.messages.length !== 2) {
    fail(context, "messages must contain user and assistant records");
  }

  const [user, assistant] = row.messages;
  if (user.role !== "user" || typeof user.content !== "string" || !user.content.trim()) {
    fail(context, "first message must be non-empty user content");
  }

  if (
    assistant.role !== "assistant" ||
    typeof assistant.content !== "string" ||
    !assistant.content.trim()
  ) {
    fail(context, "second message must be non-empty assistant content");
  }

  try {
    return JSON.parse(assistant.content);
  } catch (error) {
    fail(context, `assistant content must be strict JSON: ${error.message}`);
  }
}

function validatePayload(payload, context) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    fail(context, "assistant JSON must be an object");
  }

  if (!ALLOWED_ACTIONS.has(payload.action)) {
    fail(context, `unknown action ${payload.action}`);
  }

  if (payload.action === "tool_call") {
    if (!ALLOWED_TOOLS.has(payload.tool)) {
      fail(context, `unknown tool ${payload.tool}`);
    }

    if (!payload.args || typeof payload.args !== "object" || Array.isArray(payload.args)) {
      fail(context, "tool_call requires args object");
    }

    if (payload.args.layer && !ALLOWED_LAYERS.has(payload.args.layer)) {
      fail(context, `unknown layer ${payload.args.layer}`);
    }

    if (payload.tool === "get_entity_field") {
      if (!["schools", "kindergartens"].includes(payload.args.layer)) {
        fail(context, "get_entity_field requires schools or kindergartens layer");
      }

      if (!/^\d{10}$/.test(String(payload.args.inn ?? ""))) {
        fail(context, "get_entity_field requires 10-digit inn");
      }

      if (!ALLOWED_ENTITY_FIELDS.has(payload.args.field)) {
        fail(context, `unknown entity field ${payload.args.field}`);
      }
    }

    if (payload.tool === "resolve_entity_field") {
      if (!["schools", "kindergartens"].includes(payload.args.layer)) {
        fail(context, "resolve_entity_field requires schools or kindergartens layer");
      }

      if (
        typeof payload.args.entity_number !== "number" &&
        (typeof payload.args.entity_name !== "string" || !payload.args.entity_name.trim())
      ) {
        fail(context, "resolve_entity_field requires entity_number or entity_name");
      }

      if (!ALLOWED_ENTITY_FIELDS.has(payload.args.field)) {
        fail(context, `unknown entity field ${payload.args.field}`);
      }
    }

    if (payload.tool === "search_entities") {
      if (!["schools", "kindergartens"].includes(payload.args.layer)) {
        fail(context, "search_entities requires schools or kindergartens layer");
      }

      if (typeof payload.args.query !== "string" || !payload.args.query.trim()) {
        fail(context, "search_entities requires query");
      }
    }

    if (payload.tool === "rag_search") {
      if (typeof payload.args.query !== "string" || !payload.args.query.trim()) {
        fail(context, "rag_search requires query");
      }

      if (
        payload.args.collections &&
        (!Array.isArray(payload.args.collections) ||
          payload.args.collections.some((collection) => typeof collection !== "string"))
      ) {
        fail(context, "rag_search collections must be string array");
      }
    }

    if (["get_current_official", "get_official_by_date"].includes(payload.tool)) {
      if (payload.args.layer && payload.args.layer !== "officials") {
        fail(context, `${payload.tool} requires officials layer when layer is provided`);
      }

      if (
        typeof payload.args.position !== "string" &&
        typeof payload.args.office_query !== "string"
      ) {
        fail(context, `${payload.tool} requires position or office_query`);
      }
    }

    return;
  }

  if (payload.action === "clarify") {
    if (typeof payload.question !== "string" || !payload.question.trim()) {
      fail(context, "clarify requires question");
    }
    return;
  }

  if (payload.action === "refuse") {
    if (typeof payload.reason !== "string" || !payload.reason.trim()) {
      fail(context, "refuse requires reason");
    }
    return;
  }

  if (payload.action === "direct_answer") {
    if (typeof payload.answer !== "string" || !payload.answer.trim()) {
      fail(context, "direct_answer requires answer");
    }
  }
}

function validateNoEvalOverlap(trainRows, evalRows, fileName) {
  const evalQuestions = new Set(evalRows.map((row) => normalizeQuestion(row.question)));

  for (const row of trainRows) {
    const question = normalizeQuestion(row.messages?.[0]?.content);
    if (evalQuestions.has(question)) {
      fail(fileName, `train/eval question overlap: ${row.id}`);
    }
  }
}

function validateUniqueIds(rows, fileName) {
  const ids = new Set();
  for (const row of rows) {
    if (typeof row.id !== "string" || !row.id.trim()) {
      fail(fileName, "row id is required");
    }
    if (ids.has(row.id)) {
      fail(fileName, `duplicate id ${row.id}`);
    }
    ids.add(row.id);
  }
}

const evalRows = await readJsonl("router-eval-v1.jsonl");
const allTrainRowsByFile = new Map();

for (const fileName of TRAIN_FILES) {
  const rows = await readJsonl(fileName);
  allTrainRowsByFile.set(fileName, rows);
  validateUniqueIds(rows, fileName);
  validateNoEvalOverlap(rows, evalRows, fileName);

  rows.forEach((row, index) => {
    const context = `${fileName}:${index + 1}:${row.id ?? "no-id"}`;
    const payload = validateMessages(row, context);
    validatePayload(payload, context);
  });
}

const combined = allTrainRowsByFile.get("router-train-v1.jsonl");
const componentIds = new Set(
  ["router-train-entities.jsonl", "router-train-safety.jsonl", "router-train-history.jsonl"]
    .flatMap((fileName) => allTrainRowsByFile.get(fileName))
    .map((row) => row.id),
);

if (combined.length !== componentIds.size) {
  fail("router-train-v1.jsonl", "combined row count does not match component files");
}

const combinedV2 = allTrainRowsByFile.get("router-train-v2.jsonl");
const componentV2Ids = new Set(
  [
    "router-train-entities-v2.jsonl",
    "router-train-safety-v2.jsonl",
    "router-train-history-v2.jsonl",
  ]
    .flatMap((fileName) => allTrainRowsByFile.get(fileName))
    .map((row) => row.id),
);

if (combinedV2.length !== componentV2Ids.size) {
  fail("router-train-v2.jsonl", "combined row count does not match component files");
}

for (const row of combinedV2) {
  if (!componentV2Ids.has(row.id)) {
    fail("router-train-v2.jsonl", `combined contains unknown component id ${row.id}`);
  }
}

for (const row of combined) {
  if (!componentIds.has(row.id)) {
    fail("router-train-v1.jsonl", `combined contains unknown component id ${row.id}`);
  }
}

console.log("Router datasets are valid");
for (const [fileName, rows] of allTrainRowsByFile.entries()) {
  console.log(`${fileName}: ${rows.length} rows`);
}
