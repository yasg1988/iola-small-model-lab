# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "accelerate>=1.8.0",
#   "datasets>=3.6.0",
#   "huggingface_hub>=0.34.0,<1.0",
#   "peft>=0.15.0",
#   "torch>=2.6.0",
#   "transformers>=4.53.0,<5.0",
# ]
# ///

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

DEFAULT_ROUTER_SYSTEM_PROMPT = """You are the IOLA CLI router for public open data of Yoshkar-Ola.
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
    parser.add_argument("--base-model", default="google/gemma-3-1b-it")
    parser.add_argument("--dataset", default="LMSerg/iola-gemma3-router-sft")
    parser.add_argument("--train-file", default="data/router-train-v1.jsonl")
    parser.add_argument("--eval-file", default="data/router-eval-v1.jsonl")
    parser.add_argument("--output-repo", default="LMSerg/iola-gemma3-router-gemma3-1b-lora")
    parser.add_argument("--output-dir", default="outputs/gemma3-router-lora")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-prompt", default=DEFAULT_ROUTER_SYSTEM_PROMPT)
    parser.add_argument("--append-no-think", action="store_true")
    return parser.parse_args()


def format_chat(tokenizer, messages, add_generation_prompt=False):
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    text = ""
    for message in messages:
        text += f"{message['role']}: {message['content']}\n"
    if add_generation_prompt:
        text += "assistant: "
    return text


def normalize_json(value):
    if isinstance(value, str):
        value = json.loads(value)
    return value


def normalize_expected(row):
    return normalize_json(row["expected"])


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def with_system_prompt(messages, system_prompt):
    if not system_prompt:
        return messages

    if messages and messages[0].get("role") == "system":
        return messages

    return [{"role": "system", "content": system_prompt}, *messages]


def with_no_think(messages):
    updated = []
    last_user_index = -1
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            last_user_index = index
        updated.append(dict(message))

    if last_user_index >= 0:
        content = str(updated[last_user_index].get("content", "")).rstrip()
        if "/no_think" not in content:
            updated[last_user_index]["content"] = f"{content}\n/no_think"

    return updated


def is_expected_match(parsed, expected):
    if parsed.get("action") != expected.get("action"):
        return False

    if expected.get("action") == "direct_answer" and "answer_contains" in expected:
        answer = str(parsed.get("answer", ""))
        return all(fragment in answer for fragment in expected["answer_contains"])

    if expected.get("action") == "refuse":
        return parsed.get("reason") == expected.get("reason")

    return compact_json(parsed) == compact_json(expected)


def extract_json_object(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])

    raise ValueError("unterminated JSON object")


@dataclass
class CausalCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features):
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []

        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    args = parse_args()
    set_seed(args.seed)

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for gated Gemma access and Hub upload")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token=token,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = load_dataset(
        args.dataset,
        data_files={"train": args.train_file},
        split="train",
        token=token,
    )

    def tokenize_row(row):
        prompt_messages = row["messages"][:1]
        full_messages = row["messages"]
        if args.append_no_think:
            prompt_messages = with_no_think(prompt_messages)
            full_messages = with_no_think(full_messages)
        prompt_messages = with_system_prompt(prompt_messages, args.system_prompt)
        full_messages = with_system_prompt(full_messages, args.system_prompt)
        prompt_text = format_chat(tokenizer, prompt_messages, add_generation_prompt=True)
        text = format_chat(tokenizer, full_messages, add_generation_prompt=False)
        if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
            text += tokenizer.eos_token

        prompt_ids = tokenizer(
            prompt_text,
            max_length=args.max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]
        encoded = tokenizer(
            text,
            max_length=args.max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )
        prompt_len = min(len(prompt_ids), len(encoded["input_ids"]))
        encoded["labels"] = [-100] * prompt_len + encoded["input_ids"][prompt_len:]
        return encoded

    train_dataset = train_dataset.map(
        tokenize_row,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train rows",
    )

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=max(60, args.max_steps),
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        bf16=False,
        gradient_checkpointing=False,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=CausalCollator(tokenizer),
    )
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    eval_rows = load_dataset(
        args.dataset,
        data_files={"eval": args.eval_file},
        split="eval",
        token=token,
    )

    model.eval()
    predictions = []
    json_ok = 0
    exact_ok = 0

    for row in eval_rows:
        prompt = format_chat(
            tokenizer,
            with_system_prompt(
                with_no_think([{"role": "user", "content": row["question"]}])
                if args.append_no_think
                else [{"role": "user", "content": row["question"]}],
                args.system_prompt,
            ),
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=180,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        answer_ids = generated[0][encoded["input_ids"].shape[1] :]
        raw_output = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        expected = normalize_expected(row)
        parsed = None
        is_json_ok = False
        is_exact_ok = False

        try:
            parsed = extract_json_object(raw_output)
            is_json_ok = True
            is_exact_ok = is_expected_match(parsed, expected)
        except Exception as error:
            parsed = {"error": str(error)}

        json_ok += int(is_json_ok)
        exact_ok += int(is_exact_ok)
        predictions.append(
            {
                "id": row["id"],
                "question": row["question"],
                "raw_output": raw_output,
                "parsed": parsed,
                "expected": expected,
                "json_ok": is_json_ok,
                "exact_ok": is_exact_ok,
            }
        )

    metrics = {
        "base_model": args.base_model,
        "dataset": args.dataset,
        "train_file": args.train_file,
        "eval_file": args.eval_file,
        "max_steps": args.max_steps,
        "system_prompt": args.system_prompt,
        "append_no_think": args.append_no_think,
        "eval_total": len(predictions),
        "json_ok": json_ok,
        "exact_ok": exact_ok,
        "json_accuracy": json_ok / max(1, len(predictions)),
        "exact_accuracy": exact_ok / max(1, len(predictions)),
    }

    predictions_path = output_dir / "eval_predictions.jsonl"
    metrics_path = output_dir / "eval_metrics.json"
    predictions_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n",
        encoding="utf8",
    )
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    api = HfApi(token=token)
    api.create_repo(args.output_repo, repo_type="model", exist_ok=True)
    model.push_to_hub(args.output_repo, token=token)
    tokenizer.push_to_hub(args.output_repo, token=token)
    api.upload_file(
        path_or_fileobj=str(metrics_path),
        path_in_repo="eval/eval_metrics.json",
        repo_id=args.output_repo,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(predictions_path),
        path_in_repo="eval/eval_predictions.jsonl",
        repo_id=args.output_repo,
        repo_type="model",
    )


if __name__ == "__main__":
    main()
