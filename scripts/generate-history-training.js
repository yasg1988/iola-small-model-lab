import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const ROOT = process.cwd();
const DATASETS_DIR = path.join(ROOT, "datasets");

function assistantJson(payload) {
  return JSON.stringify(payload);
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

async function writeJsonl(fileName, rows) {
  await mkdir(DATASETS_DIR, { recursive: true });
  const body = rows.map((row) => JSON.stringify(row)).join("\n");
  await writeFile(path.join(DATASETS_DIR, fileName), `${body}\n`, "utf8");
}

function row({ id, dataset, sourceId, type, topic, question, answer, tags = [] }) {
  return {
    id,
    dataset,
    source_id: sourceId,
    type,
    topic,
    messages: [
      { role: "user", content: question },
      { role: "assistant", content: answer },
    ],
    tags: [...new Set(tags)],
  };
}

function evalRow({ id, type, topic, question, expected, tags = [] }) {
  return {
    id,
    dataset: "history-eval-v1",
    type,
    topic,
    question,
    expected,
    tags: [...new Set(tags)],
  };
}

const baseRows = await readJsonl("city-history-stable-facts.jsonl");
const train = [];
const evals = [];

for (const source of baseRows) {
  const [user, assistant] = source.messages;
  const answer =
    source.type === "source_required" ? assistant.content : assistant.content;

  train.push(
    row({
      id: `history-train-base-${source.id}`,
      dataset: "history-train-v1",
      sourceId: source.id,
      type: source.type,
      topic: source.topic,
      question: user.content,
      answer,
      tags: ["history", source.type, source.topic],
    }),
  );
}

const direct = [
  [
    "foundation-001",
    "назови год основания Йошкар-Олы",
    "в каком году была основана Йошкар-Ола?",
    "Йошкар-Ола ведет историю с 1584 года. Город был основан как Царев город на Кокшаге, позднее известный как Царевококшайск.",
    ["1584", "Царев"],
  ],
  [
    "foundation-002",
    "с какого года существует город Йошкар-Ола?",
    "когда появился город Йошкар-Ола?",
    "Город существует с 1584 года. Первоначально это был Царев город на Кокшаге, затем Царевококшайск.",
    ["1584", "Царев"],
  ],
  [
    "old-names-001",
    "перечисли прежние названия Йошкар-Олы",
    "какие исторические названия были у Йошкар-Олы?",
    "Прежние названия Йошкар-Олы: Царевококшайск до 1919 года и Краснококшайск с 1919 года до переименования в Йошкар-Олу в 1928 году.",
    ["Царевококшайск", "Краснококшайск", "1928"],
  ],
  [
    "old-names-002",
    "как назывался город до Йошкар-Олы?",
    "какое название было у города перед Йошкар-Олой?",
    "До современного названия город назывался Краснококшайск, а ранее - Царевококшайск. Название Йошкар-Ола закрепилось в 1928 году.",
    ["Царевококшайск", "Краснококшайск", "1928"],
  ],
  [
    "etymology-001",
    "как переводится название Йошкар-Ола?",
    "что означает название города Йошкар-Ола?",
    'Йошкар-Ола переводится с марийского как "Красный город": "йошкар" означает красный, а "ола" - город.',
    ["Красный город", "йошкар", "ола"],
  ],
  [
    "etymology-002",
    "что значит Йошкар и Ола отдельно?",
    "что означают части названия Йошкар и Ола?",
    'В названии Йошкар-Ола слово "йошкар" означает "красный", а "ола" означает "город"; вместе это "Красный город".',
    ["красный", "город"],
  ],
  [
    "river-001",
    "на какой реке расположен город?",
    "какая река протекает через Йошкар-Олу?",
    "Йошкар-Ола расположена на реке Малая Кокшага.",
    ["Малая Кокшага"],
  ],
  [
    "region-001",
    "столицей чего является Йошкар-Ола?",
    "какого региона столица Йошкар-Ола?",
    'Йошкар-Ола является столицей Республики Марий Эл и административным центром городского округа "Город Йошкар-Ола".',
    ["Республики Марий Эл"],
  ],
  [
    "rename-1919-001",
    "когда Царевококшайск стал Краснококшайском?",
    "в каком году Царевококшайск переименовали в Краснококшайск?",
    "Царевококшайск был переименован в Краснококшайск 17 февраля 1919 года.",
    ["17 февраля 1919"],
  ],
  [
    "rename-1928-001",
    "когда появилось название Йошкар-Ола?",
    "когда Краснококшайск получил название Йошкар-Ола?",
    "Современное название Йошкар-Ола город получил в 1928 году.",
    ["1928"],
  ],
  [
    "war-001",
    "что изменилось в городе во время Великой Отечественной войны?",
    "как война повлияла на развитие Йошкар-Олы?",
    "В годы Великой Отечественной войны в Йошкар-Олу эвакуировали ряд заводов. Это ускорило промышленное и социально-экономическое развитие города.",
    ["эвакуировали", "завод"],
  ],
  [
    "summary-001",
    "дай короткую историческую справку о городе",
    "кратко расскажи историю Йошкар-Олы",
    'Йошкар-Ола основана в 1584 году как Царев город на Кокшаге. Город назывался Царевококшайск, затем Краснококшайск, а в 1928 году получил название Йошкар-Ола, что означает "Красный город".',
    ["1584", "Царевококшайск", "Краснококшайск", "Красный город"],
  ],
];

for (const [id, question, evalQuestion, answer, contains] of direct) {
  train.push(
    row({
      id: `history-train-direct-${id}`,
      dataset: "history-train-v1",
      sourceId: "manual-history",
      type: "direct_answer",
      topic: id.split("-").slice(0, -1).join("_"),
      question,
      answer,
      tags: ["history", "direct_answer"],
    }),
  );

  evals.push(
    evalRow({
      id: `history-eval-${id}`,
      type: "direct_answer",
      topic: id.split("-").slice(0, -1).join("_"),
      question: evalQuestion,
      expected: { type: "text_contains", contains },
      tags: ["history", "direct_answer"],
    }),
  );
}

const sourceRequired = [
  [
    "source-foundation-001",
    "когда основана Йошкар-Ола? нужен источник",
    "подбери источник по году основания Йошкар-Олы",
    {
      action: "tool_call",
      tool: "rag_search",
      args: {
        query: "Йошкар-Ола основана 1584 источник",
        collections: ["city_history", "official_documents"],
      },
    },
  ],
  [
    "source-etymology-001",
    "подтверди источником перевод Йошкар-Олы",
    "найди источник о переводе названия Йошкар-Ола",
    {
      action: "tool_call",
      tool: "rag_search",
      args: {
        query: "Йошкар-Ола йошкар красный ола город источник",
        collections: ["city_history", "official_documents"],
      },
    },
  ],
  [
    "source-renames-001",
    "дай источник по переименованиям Царевококшайск Краснококшайск Йошкар-Ола",
    "найди источник о переименованиях Йошкар-Олы",
    {
      action: "tool_call",
      tool: "rag_search",
      args: {
        query: "Царевококшайск Краснококшайск Йошкар-Ола переименование 1919 1928 источник",
        collections: ["city_history", "official_documents"],
      },
    },
  ],
];

for (const [id, question, evalQuestion, answer] of sourceRequired) {
  train.push(
    row({
      id: `history-train-${id}`,
      dataset: "history-train-v1",
      sourceId: "manual-history",
      type: "source_required",
      topic: id,
      question,
      answer: assistantJson(answer),
      tags: ["history", "source_required", "rag_search"],
    }),
  );

  evals.push(
    evalRow({
      id: `history-eval-${id}`,
      type: "source_required",
      topic: id,
      question: evalQuestion,
      expected: answer,
      tags: ["history", "source_required", "rag_search"],
    }),
  );
}

const clarifyAnswer =
  "Уточните, какое переименование вас интересует: из Царевококшайска в Краснококшайск в 1919 году или из Краснококшайска в Йошкар-Олу в 1928 году.";
const clarifyQuestions = [
  "когда переименовали город?",
  "в каком году город получил новое название?",
  "когда сменилось название города?",
];

clarifyQuestions.forEach((question, index) => {
  train.push(
    row({
      id: `history-train-clarify-${index + 1}`,
      dataset: "history-train-v1",
      sourceId: "manual-history",
      type: "clarify",
      topic: "ambiguous_rename",
      question,
      answer: clarifyAnswer,
      tags: ["history", "clarify"],
    }),
  );
});

evals.push(
  evalRow({
    id: "history-eval-clarify-rename-001",
    type: "clarify",
    topic: "ambiguous_rename",
    question: "когда было переименование?",
    expected: { type: "text_contains", contains: ["1919", "1928"] },
    tags: ["history", "clarify"],
  }),
);

await writeJsonl("history-train-v1.jsonl", train);
await writeJsonl("history-eval-v1.jsonl", evals);

const trainV2 = [...train];
const evalV2 = evals.map((item) => ({ ...item, dataset: "history-eval-v2" }));

function pushV2({ id, type, topic, question, answer, tags = [] }) {
  trainV2.push(
    row({
      id: `history-train-v2-${id}`,
      dataset: "history-train-v2",
      sourceId: "manual-history-v2",
      type,
      topic,
      question,
      answer,
      tags: ["history", type, ...tags],
    }),
  );
}

[
  [
    "rename-1919-1",
    "укажи точную дату переименования Царевококшайска в Краснококшайск",
    "Царевококшайск был переименован в Краснококшайск 17 февраля 1919 года.",
  ],
  [
    "rename-1919-2",
    "когда именно Царевококшайск стал Краснококшайском",
    "Точная дата переименования Царевококшайска в Краснококшайск - 17 февраля 1919 года.",
  ],
].forEach(([id, question, answer]) =>
  pushV2({
    id,
    type: "direct_answer",
    topic: "rename_1919",
    question,
    answer,
    tags: ["old_names"],
  }),
);

[
  [
    "war-1",
    "какую роль сыграла война в развитии Йошкар-Олы",
    "В годы Великой Отечественной войны в Йошкар-Олу эвакуировали ряд заводов. Это стало важным толчком для промышленного и социально-экономического развития города.",
  ],
  [
    "war-2",
    "почему Великая Отечественная война важна для истории Йошкар-Олы",
    "Великая Отечественная война важна для истории Йошкар-Олы тем, что в город были эвакуированы заводы, что ускорило промышленное развитие.",
  ],
  [
    "war-3",
    "что происходило с промышленностью Йошкар-Олы во время войны",
    "Во время Великой Отечественной войны в Йошкар-Олу эвакуировали предприятия и заводы, что усилило промышленное развитие города.",
  ],
].forEach(([id, question, answer]) =>
  pushV2({
    id,
    type: "direct_answer",
    topic: "war_industry",
    question,
    answer,
    tags: ["war_industry"],
  }),
);

[
  [
    "source-foundation-1",
    "нужен источник по основанию Йошкар-Олы",
    {
      action: "tool_call",
      tool: "rag_search",
      args: {
        query: "Йошкар-Ола основана 1584 источник",
        collections: ["city_history", "official_documents"],
      },
    },
  ],
  [
    "source-foundation-2",
    "подтверди источником что Йошкар-Олу основали в 1584",
    {
      action: "tool_call",
      tool: "rag_search",
      args: {
        query: "Йошкар-Ола основана 1584 источник",
        collections: ["city_history", "official_documents"],
      },
    },
  ],
  [
    "source-foundation-3",
    "подбери источник по году основания города",
    {
      action: "tool_call",
      tool: "rag_search",
      args: {
        query: "Йошкар-Ола основана 1584 источник",
        collections: ["city_history", "official_documents"],
      },
    },
  ],
  [
    "source-etymology-1",
    "нужен источник о переводе названия Йошкар-Ола",
    {
      action: "tool_call",
      tool: "rag_search",
      args: {
        query: "Йошкар-Ола йошкар красный ола город источник",
        collections: ["city_history", "official_documents"],
      },
    },
  ],
  [
    "source-etymology-2",
    "найди источник что Йошкар-Ола означает Красный город",
    {
      action: "tool_call",
      tool: "rag_search",
      args: {
        query: "Йошкар-Ола йошкар красный ола город источник",
        collections: ["city_history", "official_documents"],
      },
    },
  ],
].forEach(([id, question, answer]) =>
  pushV2({
    id,
    type: "source_required",
    topic: "source_required",
    question,
    answer: assistantJson(answer),
    tags: ["rag_search"],
  }),
);

[
  "когда было переименование города?",
  "какое переименование города ты имеешь в виду?",
  "когда город сменил название?",
  "когда город переименовали без уточнения",
].forEach((question, index) =>
  pushV2({
    id: `clarify-rename-${index + 1}`,
    type: "clarify",
    topic: "ambiguous_rename",
    question,
    answer: clarifyAnswer,
    tags: ["clarify"],
  }),
);

await writeJsonl("history-train-v2.jsonl", trainV2);
await writeJsonl("history-eval-v2.jsonl", evalV2);

console.log(`Generated history-train-v1.jsonl: ${train.length} rows`);
console.log(`Generated history-eval-v1.jsonl: ${evals.length} rows`);
console.log(`Generated history-train-v2.jsonl: ${trainV2.length} rows`);
console.log(`Generated history-eval-v2.jsonl: ${evalV2.length} rows`);
