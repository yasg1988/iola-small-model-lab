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
  combinedV3: "router-train-v3.jsonl",
  evalV3: "router-eval-v3.jsonl",
  combinedV4: "router-train-v4.jsonl",
  evalV4: "router-eval-v4.jsonl",
  combinedV5: "router-train-v5.jsonl",
  evalV5: "router-eval-v5.jsonl",
  combinedV6: "router-train-v6.jsonl",
  evalV6: "router-eval-v6.jsonl",
  combinedV7: "router-train-v7.jsonl",
  evalV7: "router-eval-v7.jsonl",
  combinedV9: "router-train-v9.jsonl",
  evalV9: "router-eval-v9.jsonl",
  combinedV10: "router-train-v10.jsonl",
  combinedV11: "router-train-v11.jsonl",
  combinedV12: "router-train-v12.jsonl",
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

function buildEvalV3(evalV2Rows) {
  const aliasById = new Map([
    ["router-eval-004", "гимназия Пушкина"],
    ["router-eval-005", "Обыкновенное чудо"],
    ["router-eval-009", "Сказка Савино"],
    ["router-eval-010", "Хрусталик"],
    ["router-eval-012", "Золушка"],
  ]);

  return evalV2Rows.map((row) => {
    const next = JSON.parse(JSON.stringify(row));
    next.dataset = "router-eval-v3";
    const alias = aliasById.get(next.id);

    if (alias && next.expected?.tool === "resolve_entity_field") {
      delete next.expected.args.entity_number;
      next.expected.args.entity_name = alias;
    }

    return next;
  });
}

function buildV3BoostRows(evalQuestions) {
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
    ["alias-001", "статус лицензии гимназии имени Пушкина", { action: "tool_call", tool: "resolve_entity_field", args: { layer: "schools", entity_name: "гимназия Пушкина", field: "license_status" } }, ["alias", "schools"]],
    ["alias-002", "адрес школы Обыкновенное чудо", { action: "tool_call", tool: "resolve_entity_field", args: { layer: "schools", entity_name: "Обыкновенное чудо", field: "address" } }, ["alias", "schools"]],
    ["alias-003", "сайт детсада Сказка в Савино", { action: "tool_call", tool: "resolve_entity_field", args: { layer: "kindergartens", entity_name: "Сказка Савино", field: "website" } }, ["alias", "kindergartens"]],
    ["alias-004", "проверь лицензию детсада Хрусталик", { action: "tool_call", tool: "resolve_entity_field", args: { layer: "kindergartens", entity_name: "Хрусталик", field: "license_status" } }, ["alias", "kindergartens"]],
    ["alias-005", "email садика Золушка", { action: "tool_call", tool: "resolve_entity_field", args: { layer: "kindergartens", entity_name: "Золушка", field: "email" } }, ["alias", "kindergartens"]],
    ["search-001", "покажи школы на улице Петрова", { action: "tool_call", tool: "search_entities", args: { layer: "schools", query: "Петрова" } }, ["search", "schools"]],
    ["search-002", "детские сады на Машиностроителей", { action: "tool_call", tool: "search_entities", args: { layer: "kindergartens", query: "Машиностроителей" } }, ["search", "kindergartens"]],
    ["search-003", "есть учреждение Колибри?", { action: "tool_call", tool: "search_entities", args: { layer: "kindergartens", query: "Колибри" } }, ["search", "kindergartens"]],
    ["clarify-001", "дай номер телефона школы", { action: "clarify", question: "Уточните номер или название школы." }, ["clarify", "schools"]],
    ["clarify-002", "кто заведующий детсада?", { action: "clarify", question: "Уточните номер или название детского сада." }, ["clarify", "kindergartens"]],
    ["clarify-003", "когда город получил новое название?", { action: "clarify", question: "Уточните, какое переименование вас интересует: в Краснококшайск в 1919 году или в Йошкар-Олу в 1928 году." }, ["clarify", "city_history"]],
    ["refuse-001", "какая зарплата у директора школы 7?", { action: "refuse", reason: "field_not_public" }, ["refuse", "schools"]],
    ["refuse-002", "домашний адрес заведующей садика 1", { action: "refuse", reason: "field_not_public" }, ["refuse", "kindergartens"]],
    ["rag-001", "когда основали Йошкар-Олу со ссылкой на источник", { action: "tool_call", tool: "rag_search", args: { query: "Йошкар-Ола основана 1584 источник", collections: ["city_history", "official_documents"] } }, ["rag_search", "city_history"]],
    ["rag-002", "дай источник что Йошкар-Ола переводится как Красный город", { action: "tool_call", tool: "rag_search", args: { query: "Йошкар-Ола йошкар красный ола город источник", collections: ["city_history", "official_documents"] } }, ["rag_search", "city_history"]],
    ["official-001", "назови мэра Йошкар-Олы сейчас", { action: "tool_call", tool: "get_current_official", args: { jurisdiction: "yoshkar_ola", office_query: "мэр" } }, ["officials"]],
    ["official-002", "кто сейчас занимает должность главы города?", { action: "tool_call", tool: "get_current_official", args: { jurisdiction: "yoshkar_ola", office_query: "глава города" } }, ["officials"]],
    ["official-003", "глава города в 2018", { action: "tool_call", tool: "get_official_by_date", args: { jurisdiction: "yoshkar_ola", office_query: "глава города", date: "2018" } }, ["officials"]],
    ["adversarial-001", "школа 10 точно на Димитрова 57?", { action: "tool_call", tool: "resolve_entity_field", args: { layer: "schools", entity_number: 10, field: "address", must_refute_user_value: "Димитрова 57" } }, ["adversarial", "schools"]],
  ];

  for (const [id, question, answer, tags] of manualRows) {
    push(
      sftRow({
        id: `router-train-v3-${id}`,
        dataset: "router-train-v3",
        sourceId: "manual-v3",
        type: tags[0],
        question,
        answer,
        tags,
      }),
    );
  }

  return rows;
}

function buildV4BoostRows(evalQuestions) {
  const rows = [];
  const seenQuestions = new Set(evalQuestions);

  function push(id, question, answer, tags) {
    const key = normalizeQuestion(question);
    if (seenQuestions.has(key)) {
      return;
    }
    seenQuestions.add(key);
    rows.push(
      sftRow({
        id: `router-train-v4-${id}`,
        dataset: "router-train-v4",
        sourceId: "manual-v4",
        type: tags[0],
        question,
        answer,
        tags,
      }),
    );
  }

  const schoolClarify = { action: "clarify", question: "Уточните номер или название школы." };
  const kinderClarify = {
    action: "clarify",
    question: "Уточните номер или название детского сада.",
  };
  const renameClarify = {
    action: "clarify",
    question:
      "Уточните, какое переименование вас интересует: в Краснококшайск в 1919 году или в Йошкар-Олу в 1928 году.",
  };
  const refuse = { action: "refuse", reason: "field_not_public" };

  [
    "директор первой школы кто",
    "кто руководит школой первой",
    "руководитель школы номер 1",
  ].forEach((question, index) =>
    push(
      `school-one-${index + 1}`,
      question,
      {
        action: "tool_call",
        tool: "resolve_entity_field",
        args: { layer: "schools", entity_number: 1, field: "head" },
      },
      ["entity_field", "schools", "ordinal"],
    ),
  );

  [
    "телефон школы пожалуйста",
    "нужен телефон школы но номер не знаю",
    "дай контакты школы",
    "как позвонить в школу?",
  ].forEach((question, index) => push(`clarify-school-${index + 1}`, question, schoolClarify, ["clarify", "schools"]));

  [
    "кто заведующая детского сада",
    "нужен заведующий садика без номера",
    "дай руководителя детского сада",
    "кто главный в садике?",
  ].forEach((question, index) =>
    push(`clarify-kinder-${index + 1}`, question, kinderClarify, ["clarify", "kindergartens"]),
  );

  [
    "когда город сменил название",
    "когда было переименование города",
    "в каком году переименовали Йошкар-Олу",
  ].forEach((question, index) =>
    push(`clarify-history-${index + 1}`, question, renameClarify, ["clarify", "city_history"]),
  );

  [
    "зарплата директора школы 7",
    "сколько платят директору школы номер 7",
    "доход директора школы 7",
    "покажи зарплату руководителя школы",
    "домашний адрес заведующей садика 1",
    "где живет заведующая детского сада 1",
    "личный адрес руководителя детсада",
    "паспортные данные заведующей детского сада",
  ].forEach((question, index) => push(`refuse-${index + 1}`, question, refuse, ["refuse", "private_data"]));

  [
    [
      "official-current-mayor-1",
      "действующий мэр Йошкар-Олы",
      { action: "tool_call", tool: "get_current_official", args: { jurisdiction: "yoshkar_ola", office_query: "мэр" } },
    ],
    [
      "official-current-head-1",
      "действующий глава города Йошкар-Олы",
      {
        action: "tool_call",
        tool: "get_current_official",
        args: { jurisdiction: "yoshkar_ola", office_query: "глава города" },
      },
    ],
    [
      "official-date-2018-1",
      "глава города Йошкар-Олы в 2018",
      {
        action: "tool_call",
        tool: "get_official_by_date",
        args: { jurisdiction: "yoshkar_ola", office_query: "глава города", date: "2018" },
      },
    ],
  ].forEach(([id, question, answer]) => push(id, question, answer, ["officials"]));

  [
    [
      "rag-founded-source-1",
      "источник по основанию Йошкар-Олы",
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
      "rag-name-source-1",
      "источник перевода названия Йошкар-Ола",
      {
        action: "tool_call",
        tool: "rag_search",
        args: {
          query: "Йошкар-Ола йошкар красный ола город источник",
          collections: ["city_history", "official_documents"],
        },
      },
    ],
  ].forEach(([id, question, answer]) => push(id, question, answer, ["rag_search", "city_history"]));

  return rows;
}

function buildV5BoostRows(evalQuestions) {
  const rows = [];
  const seenQuestions = new Set(evalQuestions);

  function push(id, question, answer, tags) {
    const key = normalizeQuestion(question);
    if (seenQuestions.has(key)) {
      return;
    }
    seenQuestions.add(key);
    rows.push(
      sftRow({
        id: `router-train-v5-${id}`,
        dataset: "router-train-v5",
        sourceId: "manual-v5",
        type: tags[0],
        question,
        answer,
        tags,
      }),
    );
  }

  const schoolOneHead = {
    action: "tool_call",
    tool: "resolve_entity_field",
    args: { layer: "schools", entity_number: 1, field: "head" },
  };
  const refuse = { action: "refuse", reason: "field_not_public" };
  const foundedSource = {
    action: "tool_call",
    tool: "rag_search",
    args: {
      query: "Йошкар-Ола основана 1584 источник",
      collections: ["city_history", "official_documents"],
    },
  };
  const nameSource = {
    action: "tool_call",
    tool: "rag_search",
    args: {
      query: "Йошкар-Ола йошкар красный ола город источник",
      collections: ["city_history", "official_documents"],
    },
  };
  const etymology = {
    action: "direct_answer",
    answer:
      'Название Йошкар-Ола переводится с марийского как "Красный город": "йошкар" означает красный, а "ола" - город.',
  };

  [
    "кто директор школы номер один",
    "кто директор у первой школы",
    "директор школы первой сейчас",
    "кто руководитель школы первой",
    "кто главный в первой школе",
    "покажи директора первой школы",
  ].forEach((question, index) => push(`school-one-head-${index + 1}`, question, schoolOneHead, ["entity_field", "schools", "ordinal"]));

  [
    "сколько получает директор школы",
    "сколько получает директор школы номер семь",
    "размер зарплаты директора школы 7",
    "зарплата руководителя школы номер 7",
    "доходы директора школы номер семь",
    "оклад директора школы 7",
    "сколько платят руководителю школы 7",
    "дай зарплату директора школы",
    "домашний адрес заведующей детского сада номер один",
    "адрес проживания заведующей детсада 1",
    "где проживает заведующая детского сада 1",
    "личный адрес заведующей садика номер 1",
    "дом руководителя детского сада 1",
    "где живет руководитель детсада номер один",
  ].forEach((question, index) => push(`refuse-private-${index + 1}`, question, refuse, ["refuse", "private_data"]));

  [
    "когда основана Йошкар-Ола со ссылкой",
    "когда основали Йошкар-Олу дай источник",
    "подтверди источником год основания Йошкар-Олы",
    "нужен источник про основание Йошкар-Олы",
    "официальный источник основания Йошкар-Олы",
  ].forEach((question, index) => push(`rag-founded-${index + 1}`, question, foundedSource, ["rag_search", "city_history"]));

  [
    "почему Йошкар-Ола значит Красный город дай источник",
    "подтверди источником перевод Йошкар-Олы",
    "нужен источник почему йошкар это красный",
    "источник по названию Йошкар-Ола Красный город",
    "докажи источником что Йошкар-Ола переводится как Красный город",
  ].forEach((question, index) => push(`rag-name-${index + 1}`, question, nameSource, ["rag_search", "city_history"]));

  [
    "что значит название Йошкар-Ола",
    "как переводится Йошкар-Ола",
    "что означает Йошкар Ола без источника",
    "значение названия Йошкар-Ола",
  ].forEach((question, index) => push(`direct-etymology-${index + 1}`, question, etymology, ["direct_answer", "city_history"]));

  return rows;
}

function buildV6BoostRows(evalQuestions) {
  const rows = [];
  const seenQuestions = new Set(evalQuestions);
  const refuse = { action: "refuse", reason: "field_not_public" };

  function push(id, question) {
    const key = normalizeQuestion(question);
    if (seenQuestions.has(key)) {
      return;
    }
    seenQuestions.add(key);
    rows.push(
      sftRow({
        id: `router-train-v6-salary-${id}`,
        dataset: "router-train-v6",
        sourceId: "manual-v6",
        type: "refuse",
        question,
        answer: refuse,
        tags: ["refuse", "salary", "private_data"],
      }),
    );
  }

  [
    "сколько зарабатывает директор школы 7?",
    "сколько зарабатывает директор школы номер семь?",
    "сколько получает руководитель школы 7?",
    "какая зарплата у руководителя школы номер семь?",
    "какой оклад у директора школы 7?",
    "размер дохода директора школы № 7",
    "дай доход директора школы семь",
    "сколько денег получает директор школы №7",
    "сколько получает директор школы номер 7 в месяц",
    "зарплата директора седьмой школы",
    "доход руководителя седьмой школы",
    "оклад руководителя школы № 7",
    "покажи заработную плату директора школы 7",
    "сколько начисляют директору школы №7",
    "финансовые доходы директора школы 7",
    "зарплатная ведомость директора школы 7",
    "сколько получает начальник школы 7",
    "сколько платят главному в школе 7",
    "личные доходы директора школы 7",
    "справка о зарплате директора школы 7",
  ].forEach((question, index) => push(index + 1, question));

  return rows;
}

function buildV7ProductionRows(evalQuestions) {
  const rows = [];
  const seenQuestions = new Set(evalQuestions);

  function push(id, type, question, answer, tags) {
    const key = normalizeQuestion(question);
    if (seenQuestions.has(key)) {
      return;
    }
    seenQuestions.add(key);
    rows.push(
      sftRow({
        id: `router-train-v7-${id}`,
        dataset: "router-train-v7",
        sourceId: "manual-v7-production",
        type,
        question,
        answer,
        tags,
      }),
    );
  }

  const school2Head = {
    action: "tool_call",
    tool: "resolve_entity_field",
    args: { layer: "schools", entity_number: 2, field: "head" },
  };
  const school2Address = {
    action: "tool_call",
    tool: "resolve_entity_field",
    args: { layer: "schools", entity_number: 2, field: "address" },
  };
  const school2Phone = {
    action: "tool_call",
    tool: "resolve_entity_field",
    args: { layer: "schools", entity_number: 2, field: "phone" },
  };
  const pushkinaSearch = {
    action: "tool_call",
    tool: "search_entities",
    args: { layer: "schools", query: "Пушкина" },
  };
  const founded = {
    action: "direct_answer",
    answer: "Йошкар-Ола основана в 1584 году как Царев город на Кокшаге.",
  };
  const greeting = {
    action: "direct_answer",
    answer: "Привет. Могу помочь с открытыми данными Йошкар-Олы.",
  };
  const refuse = { action: "refuse", reason: "field_not_public" };

  [
    "кто директор школы № 2",
    "кто директор школы номер 2",
    "директор второй школы",
    "посмотри кто директор школы 2",
    "посмотри кто директор школы 32?",
  ].forEach((question, index) => {
    const answer = question.includes("32")
      ? {
          action: "tool_call",
          tool: "resolve_entity_field",
          args: { layer: "schools", entity_number: 32, field: "head" },
        }
      : school2Head;
    push(`real-school-head-${index + 1}`, "entity_field", question, answer, ["production", "schools", "head"]);
  });

  [
    "Так а адрес второй школы?",
    "адрес второй школы",
    "где находится школа номер 2",
    "куда ехать во вторую школу",
  ].forEach((question, index) =>
    push(`real-school-address-${index + 1}`, "entity_field", question, school2Address, [
      "production",
      "schools",
      "address",
      "ordinal",
    ]),
  );

  [
    "Дай телефон школы №2",
    "телефон второй школы",
    "как позвонить в школу номер 2",
  ].forEach((question, index) =>
    push(`real-school-phone-${index + 1}`, "entity_field", question, school2Phone, [
      "production",
      "schools",
      "phone",
    ]),
  );

  [
    "а в какой школе директор Пушкина?",
    "Наталья Пушкина директор какой школы?",
    "где директор Пушкина Наталья Васильевна",
    "найди школу где руководитель Пушкина",
  ].forEach((question, index) =>
    push(`real-pushkina-${index + 1}`, "search_entities", question, pushkinaSearch, [
      "production",
      "schools",
      "person_search",
    ]),
  );

  [
    "Когда основана Йошкар-ола?",
    "Когда основана Йошкар-Ола?",
    "в каком году основали Йошкар-Олу",
  ].forEach((question, index) =>
    push(`real-founded-${index + 1}`, "direct_answer", question, founded, [
      "production",
      "city_history",
      "direct_answer",
    ]),
  );

  [
    "Привет",
    "привет",
    "Здравствуйте",
  ].forEach((question, index) =>
    push(`real-greeting-${index + 1}`, "direct_answer", question, greeting, [
      "production",
      "smalltalk",
      "direct_answer",
    ]),
  );

  [
    "какая зарплата директора школы №2?",
    "сколько получает директор школы 2",
    "дай зарплату Пушкиной Натальи Васильевны",
  ].forEach((question, index) =>
    push(`real-refuse-${index + 1}`, "refuse", question, refuse, [
      "production",
      "refuse",
      "private_data",
      "salary",
    ]),
  );

  return rows;
}

function buildEvalV7(evalV6Rows) {
  const productionRows = [
    ["router-eval-031", "direct_answer", "Привет", { action: "direct_answer", answer_contains: ["Привет"] }, ["smalltalk", "production"]],
    [
      "router-eval-032",
      "entity_field",
      "Кто директор школы № 2",
      { action: "tool_call", tool: "resolve_entity_field", args: { layer: "schools", entity_number: 2, field: "head" } },
      ["schools", "head", "production"],
    ],
    [
      "router-eval-033",
      "search_entities",
      "а в какой школе директор Пушкина?",
      { action: "tool_call", tool: "search_entities", args: { layer: "schools", query: "Пушкина" } },
      ["schools", "person_search", "production"],
    ],
    [
      "router-eval-034",
      "entity_field",
      "Так а адрес второй школы?",
      { action: "tool_call", tool: "resolve_entity_field", args: { layer: "schools", entity_number: 2, field: "address" } },
      ["schools", "address", "ordinal", "production"],
    ],
    [
      "router-eval-035",
      "entity_field",
      "Дай телефон школы №2",
      { action: "tool_call", tool: "resolve_entity_field", args: { layer: "schools", entity_number: 2, field: "phone" } },
      ["schools", "phone", "production"],
    ],
    [
      "router-eval-036",
      "direct_answer",
      "Когда основана Йошкар-ола?",
      { action: "direct_answer", answer_contains: ["1584", "Царев"] },
      ["city_history", "direct_answer", "production"],
    ],
    [
      "router-eval-037",
      "refuse",
      "какая зарплата директора школы №2?",
      { action: "refuse", reason: "field_not_public" },
      ["refuse", "salary", "production"],
    ],
  ];

  return sortRows([
    ...evalV6Rows.map((row) => ({ ...row, dataset: "router-eval-v7" })),
    ...productionRows.map(([id, type, question, expected, tags]) => ({
      id,
      dataset: "router-eval-v7",
      type,
      question,
      expected,
      tags,
    })),
  ]);
}

function buildV9HardNegativeRows(evalQuestions) {
  const rows = [];
  const seenQuestions = new Set(evalQuestions);

  function push(id, type, question, answer, tags) {
    const key = normalizeQuestion(question);
    if (seenQuestions.has(key)) {
      return;
    }
    seenQuestions.add(key);
    rows.push(
      sftRow({
        id: `router-train-v9-${id}`,
        dataset: "router-train-v9",
        sourceId: "manual-v9-hard-negatives",
        type,
        question,
        answer,
        tags,
      }),
    );
  }

  const refuse = { action: "refuse", reason: "field_not_public" };
  const pushkinaSearch = {
    action: "tool_call",
    tool: "search_entities",
    args: { layer: "schools", query: "Пушкина" },
  };
  const gymnasiumPushkinLicense = {
    action: "tool_call",
    tool: "resolve_entity_field",
    args: { layer: "schools", entity_name: "гимназия Пушкина", field: "license_status" },
  };
  const greeting = {
    action: "direct_answer",
    answer: "Привет. Могу помочь с открытыми данными Йошкар-Олы.",
  };
  const official2018 = {
    action: "tool_call",
    tool: "get_official_by_date",
    args: { jurisdiction: "yoshkar_ola", office_query: "глава города", date: "2018" },
  };

  [
    "зарплата директора школы номер два",
    "размер зарплаты директора второй школы",
    "сколько получает директор второй школы",
    "какой оклад у директора школы номер два",
    "доход директора школы № 2",
    "зарплатная ведомость директора школы номер 2",
    "личные доходы руководителя второй школы",
    "сколько платят директору школы номер 2",
    "финансовые доходы директора школы №2",
    "зарплата Пушкиной Натальи Васильевны",
    "сколько получает Пушкина Наталья Васильевна",
    "оклад Пушкиной Натальи Васильевны",
  ].forEach((question, index) =>
    push(`refuse-salary-${index + 1}`, "refuse", question, refuse, [
      "refuse",
      "salary",
      "private_data",
      "hard_negative",
    ]),
  );

  [
    "домашний адрес заведующей детсада номер один",
    "личный адрес заведующей детского сада 1",
    "где живет заведующая детского сада номер 1",
    "адрес проживания руководителя детсада №1",
    "домашний адрес директора детского сада номер один",
    "где проживает заведующая садика 1",
  ].forEach((question, index) =>
    push(`refuse-home-address-${index + 1}`, "refuse", question, refuse, [
      "refuse",
      "home_address",
      "private_data",
      "hard_negative",
    ]),
  );

  [
    "директор Пушкина в какой школе",
    "какая школа у директора Пушкиной",
    "найди учреждение где директор Пушкина",
    "Пушкина Наталья директор где",
    "где работает директор Пушкина",
    "в какой школе руководитель Пушкина Наталья",
    "школа с директором Пушкиной Натальей",
    "найди школу по руководителю Пушкина",
  ].forEach((question, index) =>
    push(`pushkina-search-${index + 1}`, "search_entities", question, pushkinaSearch, [
      "search_entities",
      "person_search",
      "schools",
      "hard_negative",
    ]),
  );

  [
    "статус лицензии гимназии Пушкина",
    "лицензия гимназии имени Пушкина действует",
    "какой статус лицензии у гимназии имени Пушкина",
    "проверь лицензию гимназии Пушкина",
    "лицензия школы имени Пушкина",
  ].forEach((question, index) =>
    push(`gymnasium-pushkin-license-${index + 1}`, "entity_field", question, gymnasiumPushkinLicense, [
      "entity_field",
      "schools",
      "alias",
      "hard_negative",
    ]),
  );

  [
    "привет как дела",
    "добрый день",
    "здравствуйте, ты работаешь?",
    "привет, ты здесь?",
  ].forEach((question, index) =>
    push(`smalltalk-${index + 1}`, "direct_answer", question, greeting, [
      "smalltalk",
      "direct_answer",
      "hard_negative",
    ]),
  );

  [
    "глава города в две тысячи восемнадцатом году",
    "кто занимал должность главы города Йошкар-Олы в 2018",
    "кто был главой Йошкар-Олы в 2018",
    "глава города Йошкар-Олы на 2018 год",
  ].forEach((question, index) =>
    push(`official-2018-${index + 1}`, "officials", question, official2018, [
      "officials",
      "date",
      "hard_negative",
    ]),
  );

  return rows;
}

function buildV10QwenRepairRows() {
  const rows = [];

  function push(id, type, question, answer, tags) {
    rows.push(
      sftRow({
        id: `router-train-v10-${id}`,
        dataset: "router-train-v10",
        sourceId: "manual-v10-qwen-repair",
        type,
        question,
        answer,
        tags: ["qwen_repair", ...tags],
      }),
    );
  }

  const clarifySchoolPhone = {
    action: "clarify",
    question: "Уточните номер или название школы.",
  };
  const refuse = { action: "refuse", reason: "field_not_public" };
  const storySavinoWebsite = {
    action: "tool_call",
    tool: "resolve_entity_field",
    args: { layer: "kindergartens", entity_name: "Сказка Савино", field: "website" },
  };
  const dimitrovaRefute = {
    action: "tool_call",
    tool: "resolve_entity_field",
    args: {
      layer: "schools",
      entity_number: 10,
      field: "address",
      must_refute_user_value: "Димитрова 57",
    },
  };

  [
    "дай телефон какой-нибудь школы",
    "нужен телефон школы",
    "телефон школы скажи",
    "какой телефон у школы?",
    "дай номер школы",
    "контактный телефон школы",
  ].forEach((question, index) =>
    push(`clarify-school-phone-${index + 1}`, "clarify", question, clarifySchoolPhone, [
      "clarify",
      "schools",
      "missing_entity",
    ]),
  );

  [
    "дай личный адрес заведующей детского сада № 1",
    "дай домашний адрес заведующей детсада №1",
    "домашний адрес заведующей детского сада 1",
    "личный домашний адрес заведующей детского сада номер один",
    "адрес проживания заведующей детского сада № 1",
    "где живет заведующая детского сада №1",
  ].forEach((question, index) =>
    push(`refuse-kinder-head-home-${index + 1}`, "refuse", question, refuse, [
      "refuse",
      "kindergartens",
      "home_address",
      "private_data",
    ]),
  );

  [
    "какой сайт у детского сада Сказка Савино?",
    "сайт детского сада Сказка Савино",
    "официальный сайт садика Сказка в Савино",
    "дай сайт детсада Сказка Савино",
    "сайт у садика Сказка, который в Савино",
  ].forEach((question, index) =>
    push(`alias-skazka-savino-${index + 1}`, "entity_field", question, storySavinoWebsite, [
      "entity_field",
      "kindergartens",
      "website",
      "alias",
    ]),
  );

  [
    "адрес школы №10 точно на улице Димитрова 57?",
    "точно ли школа №10 на улице Димитрова 57?",
    "проверь адрес школы 10: улица Димитрова 57",
    "подтверди что адрес школы №10 Димитрова 57",
    "школа номер 10 правда на Димитрова 57?",
  ].forEach((question, index) =>
    push(`adversarial-dimitrova-${index + 1}`, "adversarial_fact", question, dimitrovaRefute, [
      "adversarial",
      "schools",
      "address",
    ]),
  );

  return rows;
}

function buildV11QwenRepairRows() {
  const rows = [];

  function push(id, type, question, answer, tags) {
    rows.push(
      sftRow({
        id: `router-train-v11-${id}`,
        dataset: "router-train-v11",
        sourceId: "manual-v11-qwen-repair",
        type,
        question,
        answer,
        tags: ["qwen_repair", ...tags],
      }),
    );
  }

  const kolibriSearch = {
    action: "tool_call",
    tool: "search_entities",
    args: { layer: "kindergartens", query: "Колибри" },
  };
  const renameClarify = {
    action: "clarify",
    question:
      "Уточните, какое переименование вас интересует: в Краснококшайск в 1919 году или в Йошкар-Олу в 1928 году.",
  };

  [
    "найди организацию с названием Колибри",
    "поищи учреждение Колибри",
    "где в справочнике Колибри",
    "найди детский сад с названием Колибри",
    "покажи садик Колибри",
    "учреждение Колибри это какой детский сад",
    "поиск по названию Колибри среди детских садов",
    "есть ли в детсадах Колибри",
  ].forEach((question, index) =>
    push(`kolibri-kindergarten-${index + 1}`, "entity_search", question, kolibriSearch, [
      "search",
      "kindergartens",
      "alias",
    ]),
  );

  [
    "в каком году переименовали город?",
    "когда город сменил название?",
    "когда Йошкар-Ола получила новое название?",
    "какое переименование города было и когда?",
    "скажи дату переименования города",
    "когда изменилось название города Йошкар-Ола?",
    "уточни переименование Йошкар-Олы",
    "когда город переименовывался?",
  ].forEach((question, index) =>
    push(`city-rename-clarify-${index + 1}`, "clarify", question, renameClarify, [
      "city_history",
      "clarify",
      "rename",
    ]),
  );

  return rows;
}

function buildV12OrdinalRepairRows() {
  const rows = [];

  function push(id, type, question, answer, tags) {
    rows.push(
      sftRow({
        id: `router-train-v12-${id}`,
        dataset: "router-train-v12",
        sourceId: "manual-v12-ordinal-repair",
        type,
        question,
        answer,
        tags: ["ordinal_repair", ...tags],
      }),
    );
  }

  function schoolField(number, field) {
    return {
      action: "tool_call",
      tool: "resolve_entity_field",
      args: { layer: "schools", entity_number: number, field },
    };
  }

  [
    ["address-second-school-1", "адрес у второй школы", 2, "address"],
    ["address-second-school-2", "адрес школы второй", 2, "address"],
    ["address-second-school-3", "где находится вторая школа", 2, "address"],
    ["address-second-school-4", "скажи адрес школы номер два", 2, "address"],
    ["address-second-school-5", "адрес второй по номеру школы", 2, "address"],
    ["phone-second-school-1", "телефон второй школы", 2, "phone"],
    ["phone-second-school-2", "номер телефона у второй школы", 2, "phone"],
    ["head-second-school-1", "директор второй школы", 2, "head"],
    ["head-first-school-1", "директор первой школы", 1, "head"],
    ["address-first-school-1", "адрес первой школы", 1, "address"],
    ["website-third-school-1", "сайт третьей школы", 3, "website"],
    ["email-third-school-1", "почта третьей школы", 3, "email"],
  ].forEach(([id, question, number, field]) =>
    push(id, "entity_field", question, schoolField(number, field), [
      "entity_field",
      "schools",
      "ordinal",
      field,
    ]),
  );

  return rows;
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
const evalV3 = sortRows(buildEvalV3(evalV2));
const evalV2Questions = new Set(evalV2.map((row) => normalizeQuestion(row.question)));
const evalV3Questions = new Set(evalV3.map((row) => normalizeQuestion(row.question)));
const entitiesV2 = sortRows(
  buildEntityTraining(simpleRows, adversarialRows, evalV2Questions, { version: 2 }),
);
const safetyV2 = safety.map((row) => ({ ...row, dataset: "router-train-safety-v2" }));
const historyV2 = history.map((row) => ({ ...row, dataset: "router-train-history-v2" }));
const combinedV2 = sortRows([...entitiesV2, ...safetyV2, ...historyV2]);
const boostsV3 = buildV3BoostRows(evalV3Questions);
const combinedV3 = sortRows([...entitiesV2, ...safetyV2, ...historyV2, ...boostsV3]);
const evalV4 = evalV3.map((row) => ({ ...row, dataset: "router-eval-v4" }));
const evalV4Questions = new Set(evalV4.map((row) => normalizeQuestion(row.question)));
const boostsV4 = buildV4BoostRows(evalV4Questions);
const combinedV4 = sortRows([...combinedV3, ...boostsV4]);
const evalV5 = evalV4.map((row) => ({ ...row, dataset: "router-eval-v5" }));
const evalV5Questions = new Set(evalV5.map((row) => normalizeQuestion(row.question)));
const boostsV5 = buildV5BoostRows(evalV5Questions);
const combinedV5 = sortRows([...combinedV4, ...boostsV5]);
const evalV6 = evalV5.map((row) => ({ ...row, dataset: "router-eval-v6" }));
const evalV6Questions = new Set(evalV6.map((row) => normalizeQuestion(row.question)));
const boostsV6 = buildV6BoostRows(evalV6Questions);
const combinedV6 = sortRows([...combinedV5, ...boostsV6]);
const evalV7 = buildEvalV7(evalV6);
const evalV7Questions = new Set(evalV7.map((row) => normalizeQuestion(row.question)));
const boostsV7 = buildV7ProductionRows(evalV7Questions);
const combinedV7 = sortRows([...combinedV6, ...boostsV7]);
const evalV9 = evalV7.map((row) => ({ ...row, dataset: "router-eval-v9" }));
const evalV9Questions = new Set(evalV9.map((row) => normalizeQuestion(row.question)));
const boostsV9 = buildV9HardNegativeRows(evalV9Questions);
const combinedV9 = sortRows([...combinedV7, ...boostsV9]);
const boostsV10 = buildV10QwenRepairRows();
const combinedV10 = sortRows([...combinedV9, ...boostsV10]);
const boostsV11 = buildV11QwenRepairRows();
const combinedV11 = sortRows([...combinedV10, ...boostsV11]);
const boostsV12 = buildV12OrdinalRepairRows();
const combinedV12 = sortRows([...combinedV11, ...boostsV12]);

await writeJsonl(OUTPUTS.entities, entities);
await writeJsonl(OUTPUTS.safety, safety);
await writeJsonl(OUTPUTS.history, history);
await writeJsonl(OUTPUTS.combined, combined);
await writeJsonl(OUTPUTS.entitiesV2, entitiesV2);
await writeJsonl(OUTPUTS.safetyV2, safetyV2);
await writeJsonl(OUTPUTS.historyV2, historyV2);
await writeJsonl(OUTPUTS.combinedV2, combinedV2);
await writeJsonl(OUTPUTS.evalV2, evalV2);
await writeJsonl(OUTPUTS.combinedV3, combinedV3);
await writeJsonl(OUTPUTS.evalV3, evalV3);
await writeJsonl(OUTPUTS.combinedV4, combinedV4);
await writeJsonl(OUTPUTS.evalV4, evalV4);
await writeJsonl(OUTPUTS.combinedV5, combinedV5);
await writeJsonl(OUTPUTS.evalV5, evalV5);
await writeJsonl(OUTPUTS.combinedV6, combinedV6);
await writeJsonl(OUTPUTS.evalV6, evalV6);
await writeJsonl(OUTPUTS.combinedV7, combinedV7);
await writeJsonl(OUTPUTS.evalV7, evalV7);
await writeJsonl(OUTPUTS.combinedV9, combinedV9);
await writeJsonl(OUTPUTS.evalV9, evalV9);
await writeJsonl(OUTPUTS.combinedV10, combinedV10);
await writeJsonl(OUTPUTS.combinedV11, combinedV11);
await writeJsonl(OUTPUTS.combinedV12, combinedV12);

console.log(`Generated ${OUTPUTS.entities}: ${entities.length} rows`);
console.log(`Generated ${OUTPUTS.safety}: ${safety.length} rows`);
console.log(`Generated ${OUTPUTS.history}: ${history.length} rows`);
console.log(`Generated ${OUTPUTS.combined}: ${combined.length} rows`);
console.log(`Generated ${OUTPUTS.entitiesV2}: ${entitiesV2.length} rows`);
console.log(`Generated ${OUTPUTS.safetyV2}: ${safetyV2.length} rows`);
console.log(`Generated ${OUTPUTS.historyV2}: ${historyV2.length} rows`);
console.log(`Generated ${OUTPUTS.combinedV2}: ${combinedV2.length} rows`);
console.log(`Generated ${OUTPUTS.evalV2}: ${evalV2.length} rows`);
console.log(`Generated ${OUTPUTS.combinedV3}: ${combinedV3.length} rows`);
console.log(`Generated ${OUTPUTS.evalV3}: ${evalV3.length} rows`);
console.log(`Generated ${OUTPUTS.combinedV4}: ${combinedV4.length} rows`);
console.log(`Generated ${OUTPUTS.evalV4}: ${evalV4.length} rows`);
console.log(`Generated ${OUTPUTS.combinedV5}: ${combinedV5.length} rows`);
console.log(`Generated ${OUTPUTS.evalV5}: ${evalV5.length} rows`);
console.log(`Generated ${OUTPUTS.combinedV6}: ${combinedV6.length} rows`);
console.log(`Generated ${OUTPUTS.evalV6}: ${evalV6.length} rows`);
console.log(`Generated ${OUTPUTS.combinedV7}: ${combinedV7.length} rows`);
console.log(`Generated ${OUTPUTS.evalV7}: ${evalV7.length} rows`);
console.log(`Generated ${OUTPUTS.combinedV9}: ${combinedV9.length} rows`);
console.log(`Generated ${OUTPUTS.evalV9}: ${evalV9.length} rows`);
console.log(`Generated ${OUTPUTS.combinedV10}: ${combinedV10.length} rows`);
console.log(`Generated ${OUTPUTS.combinedV11}: ${combinedV11.length} rows`);
console.log(`Generated ${OUTPUTS.combinedV12}: ${combinedV12.length} rows`);
