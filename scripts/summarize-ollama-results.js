import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const SUMMARY_FILE = path.join(ROOT, 'results', 'ollama-runs', 'latest-summary.json');

const summary = JSON.parse(await fs.readFile(SUMMARY_FILE, 'utf8'));
const rows = Object.values(summary.results)
  .sort((a, b) => a.dataset.localeCompare(b.dataset)
    || b.accuracy - a.accuracy
    || a.model.localeCompare(b.model)
    || a.mode.localeCompare(b.mode));

console.log(`Run: ${summary.runId}`);
console.log(`Limit: ${summary.limit ?? 'all'}`);
console.log('');
console.log('| Dataset | Model | Mode | Correct | Accuracy | Has expected | False affirmed | Avg ms |');
console.log('| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |');
for (const row of rows) {
  console.log(`| ${row.dataset} | ${row.model} | ${row.mode} | ${row.correct}/${row.total} | ${(row.accuracy * 100).toFixed(1)}% | ${row.hasExpected}/${row.total} | ${row.affirmsFalse} | ${Math.round(row.avgLatencyMs)} |`);
}
