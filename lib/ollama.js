export const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || 'http://127.0.0.1:11434';

export const DEFAULT_MODELS = [
  'gemma3:1b',
  'llama3.2:1b',
  'qwen3:0.6b',
  'qwen3:1.7b',
];

export async function listModels() {
  const response = await fetch(`${OLLAMA_BASE_URL}/api/tags`);
  if (!response.ok) throw new Error(`Ollama tags failed: ${response.status} ${response.statusText}`);
  const payload = await response.json();
  return payload.models || [];
}

export async function pullModel(model) {
  const response = await fetch(`${OLLAMA_BASE_URL}/api/pull`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model, stream: false }),
  });
  if (!response.ok) throw new Error(`Ollama pull ${model} failed: ${response.status} ${response.statusText}`);
  return response.json();
}

export async function generate(model, prompt, options = {}) {
  const response = await fetch(`${OLLAMA_BASE_URL}/api/generate`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model,
      prompt,
      stream: false,
      options: {
        temperature: 0,
        top_p: 0.9,
        num_ctx: 2048,
        num_predict: 180,
        ...options,
      },
    }),
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Ollama generate ${model} failed: ${response.status} ${response.statusText}\n${body}`);
  }
  const payload = await response.json();
  return {
    answer: payload.response || '',
    evalCount: payload.eval_count || 0,
    evalDurationMs: payload.eval_duration ? Math.round(payload.eval_duration / 1_000_000) : 0,
    totalDurationMs: payload.total_duration ? Math.round(payload.total_duration / 1_000_000) : 0,
  };
}
