# HaluEval — Hallucination Evaluation (Protocols A & B)

Reproducible evaluation of small open-source LLMs on [HaluEval](https://github.com/RUCAIBox/HaluEval):

| Model | Ollama tag |
|-------|------------|
| Llama 3.2 1B | `llama3.2:1b` |
| Qwen3 1.7B | `qwen3:1.7b` |
| Gemma 3 1B | `gemma3:1b` |

**Protocol A** — paired discrimination: the model sees context + one candidate answer and answers Yes/No (hallucinated?). Positive class = hallucinated.

**Protocol B** — free generation with Ollama, scored by task-aware `is_correct` / `word_overlap`, then logistic regression + a small Keras NN on overlap features.

```
halu_eval_project/
├── notebook/halu_eval.ipynb      # full A+B pipeline
├── scripts/run_remaining.py      # CLI for Protocol A + joint outputs
├── backend/app.py                # chat API
├── frontend/                     # React chat UI
├── data/                         # HaluEval JSONL (gitignored)
└── results/                      # generated metrics, figures (gitignored)
```

## 0. Environment

```bash
cd halu_eval_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

TensorFlow is required for the Protocol B neural net. If your venv lacks it, use a conda env that has `tensorflow`, or:

```bash
pip install tensorflow
```

## 1. Data

```bash
mkdir -p data
for f in qa dialogue summarization general; do
  curl -L -o data/${f}_data.json \
    https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/${f}_data.json
done
```

Optional: truncate each file to 500 lines to match `N_PER_TASK = 500` in the notebook:

```bash
for f in qa dialogue summarization general; do
  head -500 data/${f}_data.json > data/${f}_tmp.json && mv data/${f}_tmp.json data/${f}_data.json
done
```

A smaller subset (e.g. 20–50 lines per task) is enough to smoke-test the pipeline.

## 2. Ollama

```bash
ollama serve
```

In another terminal:

```bash
ollama pull llama3.2:1b
ollama pull qwen3:1.7b
ollama pull gemma3:1b
```

Confirm tags:

```bash
curl -s http://localhost:11434/api/tags | python3 -m json.tool
```

**Qwen3 note.** Qwen3 is a thinking model. All Ollama calls in this project pass top-level `"think": false` so the answer lands in `message.content`. Without that, `num_predict` is spent on thinking and `content` is empty (Protocol A/B scores look fake-zero). The runner also falls back to `message.thinking` if content is empty.

## 3. Full evaluation (recommended)

### Option A — Jupyter notebook

```bash
jupyter notebook notebook/halu_eval.ipynb
```

Run cells top to bottom. Flip these flags when you want Ollama calls:

| Flag | Cell | Effect |
|------|------|--------|
| `RUN_GENERATION = True` | Ollama helpers | Protocol B free generation (resumable) |
| `RUN_PROTOCOL_A = True` | Protocol A | Yes/No discrimination (resumable) |

Order that matters:

1. Imports + config (`MODELS`, `N_PER_TASK`, `RANDOM_SEED`)
2. Migrate legacy Llama files (once)
3. Check Ollama / warmup
4. Load HaluEval samples (shared across models)
5. Wall-time estimate — drop models/tasks if over deadline
6. Protocol B generation (`RUN_GENERATION`)
7. Protocol A generation (`RUN_PROTOCOL_A`)
8. Protocol B scoring (`is_correct` unchanged)
9. Classifiers + Step 2 metrics / probabilities
10. Step 3 `combined_results.json`
11. Step 4 `protocol_comparison.csv`
12. Step 6 figures + captions
13. Comparison bar chart

Progress is written after every Ollama call. Re-run the same cell to resume.

### Option B — CLI for Protocol A + joint artifacts

From the repo root (with `ollama serve` already running):

```bash
python scripts/run_remaining.py
# or only one model:
python scripts/run_remaining.py --models llama3.2:1b
# skip Ollama and only rebuild JSON/CSV/figures from existing runs:
python scripts/run_remaining.py --skip-protocol-a
```

## 4. Output files

Per model (`:` → `_` in filenames):

| File | Contents |
|------|----------|
| `protocol_a_progress_<model>.json` | Resumable Protocol A judgements |
| `protocol_a_run_<model>.csv` | Per-judgement gold / pred / correct |
| `protocol_a_metrics_<model>.csv` | Accuracy, precision, recall, F1, TP/FP/TN/FN |
| `ollama_progress_<model>.json` | Resumable Protocol B generations |
| `ollama_run_<model>.csv` | Free-generation answers |
| `task_metrics_<model>.csv` | Protocol B accuracy by task |
| `protocol_b_metrics_<model>.csv` | Detector precision/recall/F1 |
| `protocol_b_classifier_proba_<model>.csv` | LR `predict_proba` + NN sigmoid |

Joint:

| File | Contents |
|------|----------|
| `combined_results.json` | Keyed by `task\|sample_id\|model` |
| `protocol_comparison.csv` | `a_minus_b`, `sign_flip` per task |
| `task_comparison.png` | Protocol B accuracy bars |
| `precision_recall_scatter.png` | A vs B-LR precision/recall |
| `figure_captions.md` | One-paragraph captions |

Missing runs print `MISSING` and are never fabricated.

## 5. Chat frontend (optional)

```bash
uvicorn backend.app:app --reload --port 8001
cd frontend && npm install && npm run dev
```

Ask Llama with optional HaluEval samples and a reference answer for word-overlap scoring.

## 6. Scoring definitions (do not change lightly)

- **Protocol A positive class** = hallucinated (`Yes`).
- **Protocol B `is_correct`**: QA requires the normalized ground-truth phrase in the answer; other tasks require `word_overlap(answer, truth) > word_overlap(answer, hallucinated)`.
- **Protocol B detectors** invert `correct` so positive = hallucinated, matching Protocol A.

## 7. Wall time

Rough cost per remaining Ollama call ≈ `SLEEP_SEC + GEN_SEC_ESTIMATE` (defaults 1.5 + 2.0 s).

- Protocol B: one call per sample per model.
- Protocol A: up to two calls per sample per model (truth + hallucinated candidates; general usually one).

The notebook prints an estimate before Protocol B generation. If it exceeds your deadline, drop a model or task rather than fabricating numbers.
