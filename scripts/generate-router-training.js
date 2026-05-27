import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const ROOT = process.cwd();
const DATASETS_DIR = path.join(ROOT, "datasets");

const OUTPUTS = {
  entities: "router-train-entities.jsonl",
  safety: "router-train-safety.jsonl",
  history: "router-train-history.jsonl",
  combined: "router-train-v1.jsonl",
  entitiesV2: "router-train-entities-v2.jsonl",
  safetyV2: "router-train-safety-v2.jsonl",
  historyV2: "router-train-history-v2.jsonl",
  combinedV2: "router-train-v2.jsonl",
  evalV2: "router-eval-v2.jsonl",
};

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

async function writeJsonl(fileName, rows) {
  await mkdir(DATASETS_DIR, { recursive: true });
  const body = rows.map((row) => JSON.stringify(row)).join("\n");
  await writeFile(path.join(DATASETS_DIR, fileName), `${body}\n`, "utf8");
}

function assistantJson(payload) {
  return JSON.stringify(payload);
}

function sftRow({ id, dataset, sourceId, type, question, answer, tags = [] }) {
  return {
    id,
    dataset,
    source_id: sourceId,
    type,
    messages: [
      { role: "user", content: question },
      { role: "assistant", content: assistantJson(answer) },
    ],
    tags: [...new Set(tags)],
  };
}

function entityToolCall(row, extraArgs = {}) {
  return {
    action: "tool_call",
    tool: "get_entity_field",
    args: {
      layer: row.layer,
      inn: row.expectedInn,
      field: row.field,
      ...extraArgs,
    },
  };
}

function entityToolCallV2(row, extraArgs = {}) {
  return {
    action: "tool_call",
    tool: "resolve_entity_field",
    args: {
      layer: row.layer,
      entity_number: row.entityNumber,
      field: row.field,
      ...extraArgs,
    },
  };
}

function layerWords(layer) {
  if (layer === "schools") {
    return {
      nominative: "школа",
      accusative: "школу",
      genitive: "школы",
      plural: "школ",
      short: "шк",
    };
  }

  return {
    nominative: "детский сад",
    accusative: "детский сад",
      genitive: "детского сада",
    plural: "детских садов",
    short: "дс",
  };
}

function fieldQuestionTemplates(field, words) {
  const templates = {
    head: [
      `кто руководитель ${words.genitive} № {number}?`,
      `нужен руководитель ${words.genitive} номер {number}`,
      `кто сейчас главный в ${words.accusative} № {number}?`,
    ],
    phone: [
      `дай телефон ${words.genitive} № {number}`,
      `как позвонить в ${words.accusative} номер {number}?`,
      `нужен номер телефона ${words.genitive} № {number}`,
    ],
    website: [
      `какой сайт у ${words.genitive} № {number}?`,
      `ссылка на сайт ${words.genitive} номер {number}`,
      `официальный сайт ${words.genitive} № {number}`,
    ],
    license_status: [
      `проверь статус лицензии ${words.genitive} № {number}`,
      `лицензия ${words.genitive} номер {number} действует?`,
      `какой статус лицензии у ${words.genitive} № {number}?`,
    ],
    address: [
      `где находится ${words.nominative} № {number}?`,
      `дай адрес ${words.genitive} номер {number}`,
      `куда ехать в ${words.accusative} № {number}?`,
    ],
    email: [
      `какая почта у ${words.genitive} № {number}?`,
      `email ${words.genitive} номер {number}`,
      `куда писать в ${words.accusative} № {number}?`,
    ],
    inn: [
      `какой инн у ${words.genitive} № {number}?`,
      `дай ИНН ${words.genitive} номер {number}`,
      `ИНН ${words.short} № {number}`,
    ],
  };

  return templates[field] ?? [`дай поле ${field} для ${words.genitive} № {number}`];
}

function renderTemplate(template, row) {
  return template.replaceAll("{number}", String(row.entityNumber));
}

function buildEntityTraining(simpleRows, adversarialRows, evalQuestions, { version = 1 } = {}) {
  const rows = [];
  const seenQuestions = new Set(evalQuestions);
  const toolCall = version === 2 ? entityToolCallV2 : entityToolCall;
  const suffix = version === 2 ? "-v2" : "";

  function push(row) {
    const key = normalizeQuestion(row.messages[0].content);
    if (seenQuestions.has(key)) {
      return;
    }
    seenQuestions.add(key);
    rows.push(row);
  }

  for (const source of simpleRows) {
    const words = layerWords(source.layer);
    const templates = fieldQuestionTemplates(source.field, words);
    const questions = [
      source.question,
      ...templates.map((template) => renderTemplate(template, source)),
      `найди поле ${source.field} для организации ${source.entityName}`,
    ];

    questions.forEach((question, index) => {
      push(
        sftRow({
          id: `router-train-entity-${source.id}-${String(index + 1).padStart(2, "0")}`,
          dataset: `router-train-entities${suffix}`,
          sourceId: source.id,
          type: "entity_field",
          question,
          answer: toolCall(source),
          tags: ["entity_field", source.layer, source.field, ...source.tags],
        }),
      );
    });
  }

  for (const source of adversarialRows) {
    const words = layerWords(source.layer);
    const questions = [
      source.question,
      `пользователь утверждает: ${source.falseValue}. Проверь ${source.field} для ${words.genitive} № ${source.entityNumber}`,
      `не подтверждай из памяти: ${source.falseValue}. Нужно проверить ${words.accusative} № ${source.entityNumber}`,
    ];

    questions.forEach((question, index) => {
      push(
        sftRow({
          id: `router-train-adversarial-${source.id}-${String(index + 1).padStart(2, "0")}`,
          dataset: `router-train-entities${suffix}`,
          sourceId: source.id,
          type: "adversarial_fact",
          question,
          answer: toolCall(source, {
            must_refute_user_value: source.falseValue,
          }),
          tags: ["adversarial", source.layer, source.field, ...source.tags],
        }),
      );
    });
  }

  return rows;
}

function buildEvalV2(evalRows, sourceRows) {
  const byInn = new Map(sourceRows.map((row) => [row.expectedInn, row]));

  return evalRows.map((row) => {
    const next = JSON.parse(JSON.stringify(row));
    next.dataset = "router-eval-v2";

    if (next.expected?.tool === "get_entity_field" && next.expected?.args?.inn) {
      const source = byInn.get(next.expected.args.inn);
      if (!source) {
        throw new Error(`No entity source for eval ${next.id} inn ${next.expected.args.inn}`);
      }

      next.expected = {
        action: "tool_call",
        tool: "resolve_entity_field",
        args: {
          layer: next.expected.args.layer,
          entity_number: source.entityNumber,
          field: next.expected.args.field,
        },
      };

      if (row.expected.must_refute_user_value) {
        next.expected.args.must_refute_user_value = row.expected.must_refute_user_value;
      }
    }

    if (next.expected?.tool === "search_entities" && next.expected?.args?.filters) {
      const filters = next.expected.args.filters;
      next.expected.args.query =
        filters.address_contains ?? filters.name_contains ?? JSON.stringify(filters);
      delete next.expected.args.filters;
    }

    return next;
  });
}

function buildSafetyTraining(evalQuestions) {
  const rows = [];
  const seenQuestions = new Set(evalQuestions);

  function push(row) {
    const key = normalizeQuestion(row.messages[0].content);
    if (seenQuestions.has(key)) {
      return;
    }
    seenQuestions.add(key);
    rows.push(row);
  }

  const manualRows = [
    {
      id: "router-train-safety-001",
      type: "clarify",
      question: "дай телефон школы",
      answer: {
        action: "clarify",
        question: "Уточните номер или название школы.",
      },
      tags: ["clarify", "schools"],
    },
    {
      id: "router-train-safety-002",
      type: "clarify",
      question: "какой адрес у детского сада?",
      answer: {
        action: "clarify",
        question: "Уточните номер или название детского сада.",
      },
      tags: ["clarify", "kindergartens"],
    },
    {
      id: "router-train-safety-003",
      type: "clarify",
      question: "кто руководитель учреждения?",
      answer: {
        action: "clarify",
        question: "Уточните тип учреждения и его номер или название.",
      },
      tags: ["clarify", "ambiguous"],
    },
    {
      id: "router-train-safety-004",
      type: "refuse",
      question: "дай паспортные данные директора школы",
      answer: {
        action: "refuse",
        reason: "field_not_public",
        message: "В публичном наборе нет данных для ответа на этот вопрос.",
      },
      tags: ["refuse", "private_data"],
    },
    {
      id: "router-train-safety-005",
      type: "refuse",
      question: "найди домашний адрес заведующей детского сада",
      answer: {
        action: "refuse",
        reason: "field_not_public",
        message: "В публичном наборе нет данных для ответа на этот вопрос.",
      },
      tags: ["refuse", "private_data"],
    },
    {
      id: "router-train-safety-006",
      type: "search_entities",
      question: "найди школу на улице Волкова",
      answer: {
        action: "tool_call",
        tool: "search_entities",
        args: {
          layer: "schools",
          query: "улица Волкова",
        },
      },
      tags: ["search_entities", "schools", "address"],
    },
    {
      id: "router-train-safety-007",
      type: "search_entities",
      question: "найди детский сад сказка",
      answer: {
        action: "tool_call",
        tool: "search_entities",
        args: {
          layer: "kindergartens",
          query: "сказка",
        },
      },
      tags: ["search_entities", "kindergartens", "name"],
    },
    {
      id: "router-train-safety-008",
      type: "officials",
      question: "кто сейчас мэр Йошкар-Олы?",
      answer: {
        action: "tool_call",
        tool: "get_current_official",
        args: {
          layer: "officials",
          position: "мэр Йошкар-Олы",
        },
      },
      tags: ["officials", "current"],
    },
    {
      id: "router-train-safety-009",
      type: "officials",
      question: "кто был главой города в 2021 году?",
      answer: {
        action: "tool_call",
        tool: "get_official_by_date",
        args: {
          layer: "officials",
          position: "глава города Йошкар-Олы",
          date: "2021",
        },
      },
      tags: ["officials", "date"],
    },
    {
      id: "router-train-safety-010",
      type: "rag_search",
      question: "подготовь справку по документам о школах города",
      answer: {
        action: "tool_call",
        tool: "rag_search",
        args: {
          query: "документы школы Йошкар-Ола справка",
          collections: ["documents", "schools"],
        },
      },
      tags: ["rag_search", "documents"],
    },
  ];

  for (const row of manualRows) {
    push(
      sftRow({
        ...row,
        dataset: "router-train-safety",
        sourceId: "manual",
      }),
    );
  }

  return rows;
}

function buildHistoryTraining(historyRows, evalQuestions) {
  const rows = [];
  const seenQuestions = new Set(evalQuestions);

  function push(row) {
    const key = normalizeQuestion(row.messages[0].content);
    if (seenQuestions.has(key)) {
      return;
    }
    seenQuestions.add(key);
    rows.push(row);
  }

  for (const source of historyRows) {
    const question = source.messages[0].content;
    const assistant = source.messages[1].content;
    let answer;

    if (source.type === "source_required") {
      answer = JSON.parse(assistant);
    } else if (source.type === "clarify") {
      answer = {
        action: "clarify",
        question: assistant,
      };
    } else {
      answer = {
        action: "direct_answer",
        answer: assistant,
      };
    }

    push(
      sftRow({
        id: `router-train-history-${source.id}`,
        dataset: "router-train-history",
        sourceId: source.id,
        type: source.type,
        question,
        answer,
        tags: ["city_history", source.type, source.topic],
      }),
    );
  }

  return rows;
}

function sortRows(rows) {
  return [...rows].sort((left, right) => left.id.localeCompare(right.id, "ru"));
}

const simpleRows = await readJsonl("simple-facts.jsonl");
const adversarialRows = await readJsonl("adversarial-facts.jsonl");
const historyRows = await readJsonl("city-history-stable-facts.jsonl");
const evalRows = await readJsonl("router-eval-v1.jsonl");
const evalQuestions = new Set(evalRows.map((row) => normalizeQuestion(row.question)));

const entities = sortRows(buildEntityTraining(simpleRows, adversarialRows, evalQuestions));
const safety = sortRows(buildSafetyTraining(evalQuestions));
const history = sortRows(buildHistoryTraining(historyRows, evalQuestions));
const combined = sortRows([...entities, ...safety, ...history]);
const evalV2 = sortRows(buildEvalV2(evalRows, [...simpleRows, ...adversarialRows]));
const evalV2Questions = new Set(evalV2.map((row) => normalizeQuestion(row.question)));
const entitiesV2 = sortRows(
  buildEntityTraining(simpleRows, adversarialRows, evalV2Questions, { version: 2 }),
);
const safetyV2 = safety.map((row) => ({ ...row, dataset: "router-train-safety-v2" }));
const historyV2 = history.map((row) => ({ ...row, dataset: "router-train-history-v2" }));
const combinedV2 = sortRows([...entitiesV2, ...safetyV2, ...historyV2]);

await writeJsonl(OUTPUTS.entities, entities);
await writeJsonl(OUTPUTS.safety, safety);
await writeJsonl(OUTPUTS.history, history);
await writeJsonl(OUTPUTS.combined, combined);
await writeJsonl(OUTPUTS.entitiesV2, entitiesV2);
await writeJsonl(OUTPUTS.safetyV2, safetyV2);
await writeJsonl(OUTPUTS.historyV2, historyV2);
await writeJsonl(OUTPUTS.combinedV2, combinedV2);
await writeJsonl(OUTPUTS.evalV2, evalV2);

console.log(`Generated ${OUTPUTS.entities}: ${entities.length} rows`);
console.log(`Generated ${OUTPUTS.safety}: ${safety.length} rows`);
console.log(`Generated ${OUTPUTS.history}: ${history.length} rows`);
console.log(`Generated ${OUTPUTS.combined}: ${combined.length} rows`);
console.log(`Generated ${OUTPUTS.entitiesV2}: ${entitiesV2.length} rows`);
console.log(`Generated ${OUTPUTS.safetyV2}: ${safetyV2.length} rows`);
console.log(`Generated ${OUTPUTS.historyV2}: ${historyV2.length} rows`);
console.log(`Generated ${OUTPUTS.combinedV2}: ${combinedV2.length} rows`);
console.log(`Generated ${OUTPUTS.evalV2}: ${evalV2.length} rows`);
