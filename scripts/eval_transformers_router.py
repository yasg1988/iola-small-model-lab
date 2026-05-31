import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """You are the IOLA CLI router for public open data of Yoshkar-Ola.
Return exactly one JSON object. Do not use markdown. Do not add explanations.
Allowed actions: tool_call, clarify, refuse, direct_answer.
Allowed tools: resolve_entity_field, search_entities, rag_search, get_current_official, get_official_by_date.
For schools and kindergartens, do not answer mutable facts from memory. Use resolve_entity_field or search_entities.
Allowed organization layers: schools, kindergartens.
Allowed organization fields: name, inn, address, email, website, phone, head, license_status.
If the user asks for a source about city history, use rag_search. If no source is requested and the fact is stable, direct_answer is allowed.
If the request is ambiguous, return clarify. If the requested field is not public, return refuse with reason field_not_public.
If the user asks to confirm a possibly false value, call the exact tool and pass that value as args.must_refute_user_value."""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset", default="datasets/router-eval-v9.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


def scalar_equal(actual, expected):
    return str(actual if actual is not None else "") == str(expected if expected is not None else "")


def includes_all(value, parts):
    text = str(value or "").lower()
    return all(str(part).lower() in text for part in parts)


def exact_match(actual, expected):
    if not isinstance(actual, dict):
        return False
    if actual.get("action") != expected.get("action"):
        return False
    if expected.get("tool") and actual.get("tool") != expected.get("tool"):
        return False
    if expected.get("reason") and actual.get("reason") != expected.get("reason"):
        return False
    if expected.get("answer_contains") and not includes_all(actual.get("answer"), expected["answer_contains"]):
        return False
    for key, expected_value in (expected.get("args") or {}).items():
        actual_value = (actual.get("args") or {}).get(key)
        if isinstance(expected_value, list):
            if not isinstance(actual_value, list) or len(actual_value) != len(expected_value):
                return False
            if any(not scalar_equal(left, right) for left, right in zip(actual_value, expected_value)):
                return False
        elif not scalar_equal(actual_value, expected_value):
            return False
    return True


def main():
    args = parse_args()
    rows = read_jsonl(args.dataset)
    if args.limit:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    json_ok = 0
    exact_ok = 0
    latencies = []
    for row in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt")
        started = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=180,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        latencies.append((time.perf_counter() - started) * 1000)
        answer_ids = generated[0][encoded["input_ids"].shape[1] :]
        raw = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        parsed = extract_json(raw)
        is_json = parsed is not None
        is_exact = is_json and exact_match(parsed, row["expected"])
        json_ok += int(is_json)
        exact_ok += int(is_exact)
        print(json.dumps({
            "id": row["id"],
            "question": row["question"],
            "raw": raw,
            "parsed": parsed,
            "json_ok": is_json,
            "exact_ok": is_exact,
        }, ensure_ascii=False))

    print(json.dumps({
        "total": len(rows),
        "json_ok": json_ok,
        "exact_ok": exact_ok,
        "json_accuracy": json_ok / max(1, len(rows)),
        "exact_accuracy": exact_ok / max(1, len(rows)),
        "avg_latency_ms": sum(latencies) / max(1, len(latencies)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
