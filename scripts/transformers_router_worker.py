import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from peft import PeftModel
from transformers import StoppingCriteria, StoppingCriteriaList
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


class RouterState:
    def __init__(self, model, tokenizer, load_seconds):
        self.model = model
        self.tokenizer = tokenizer
        self.load_seconds = load_seconds
        self.lock = threading.Lock()


class CompleteJsonStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_tokens):
        self.tokenizer = tokenizer
        self.prompt_tokens = prompt_tokens

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[0][self.prompt_tokens :]
        if generated.numel() < 2:
            return False
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not text.startswith("{"):
            return False

        depth = 0
        in_string = False
        escaped = False
        for char in text:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return True
        return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    return parser.parse_args()


def extract_json(text):
    raw = str(text or "").strip()
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except Exception:
                return None
        return None


def load_model(base_dir, adapter_dir, dtype):
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(base_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return RouterState(model, tokenizer, time.perf_counter() - started)


def make_prompt(tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(state, question, max_new_tokens):
    prompt = make_prompt(state.tokenizer, question)
    encoded = state.tokenizer(prompt, return_tensors="pt")
    prompt_tokens = encoded["input_ids"].shape[1]
    stopping_criteria = StoppingCriteriaList([CompleteJsonStoppingCriteria(state.tokenizer, prompt_tokens)])
    started = time.perf_counter()
    with state.lock:
        with torch.inference_mode():
            generated = state.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=state.tokenizer.pad_token_id,
                eos_token_id=state.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )
    latency_ms = (time.perf_counter() - started) * 1000
    answer_ids = generated[0][prompt_tokens:]
    answer = state.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    return {
        "answer": answer,
        "parsed": extract_json(answer),
        "latencyMs": round(latency_ms, 2),
        "generatedTokens": int(answer_ids.numel()),
    }


def build_handler(state, default_max_new_tokens):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def write_json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf8")
            self.send_response(status)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.write_json(200, {
                    "ok": True,
                    "runtime": "transformers",
                    "loadSeconds": round(state.load_seconds, 2),
                })
                return
            self.write_json(404, {"ok": False, "error": "not_found"})

        def do_POST(self):
            if self.path != "/generate":
                self.write_json(404, {"ok": False, "error": "not_found"})
                return

            try:
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf8"))
                question = str(payload.get("question", "")).strip()
                max_new_tokens = int(payload.get("maxNewTokens", default_max_new_tokens))
                if not question:
                    self.write_json(400, {"ok": False, "error": "question_required"})
                    return
                result = generate(state, question, max_new_tokens)
                self.write_json(200, {"ok": True, **result})
            except Exception as exc:
                self.write_json(500, {"ok": False, "error": str(exc)})

    return Handler


def main():
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    state = load_model(args.base_dir, args.adapter_dir, args.dtype)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(state, args.max_new_tokens))
    print(json.dumps({
        "event": "ready",
        "host": args.host,
        "port": args.port,
        "loadSeconds": round(state.load_seconds, 2),
        "threads": args.threads,
        "dtype": args.dtype,
    }), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
