# Measuring Hallucination Rates in Open-Source Language Models Using HaluEval

**Team:** Sushmita Hari, Connor Fahs, William McDonald, Sriram Rella

---

## Overview

Large language models can produce answers that sound correct but are false, unsupported, or not grounded in the information they were given. These errors are commonly called hallucinations. This project studies how often selected open-source language models hallucinate and whether hallucination behavior changes across model family, model size, task type, and evaluation method. The main goal is to build a reproducible evaluation pipeline that compares several open-source models under the same conditions, rather than judging one model in isolation.

---

## Motivation & Background

Hallucinations are currently evaluated through human review, benchmark datasets, automated judging, and repeated sampling. Each approach has significant limitations:

- **Human review** is useful for catching errors, but is slow, expensive, and subjective.
- **Benchmark datasets** offer consistent comparisons but may not fully reflect real-world use.
- **Automated judges** are faster, but can make mistakes or favor certain response styles.
- **Repeated sampling** reveals inconsistency, but a model can still be consistently wrong.

Because of these limits, a single hallucination score is not enough to explain how or why models fail. A multi-faceted, comparative pipeline is therefore needed.

---

## Approach & Models

We compare multiple open-source language model families using the same datasets, prompts, scoring rules, and analysis pipeline. Models are drawn from the Qwen, Llama, Gemma, Phi, and Mistral families. Smaller models that can run locally through Ollama or Hugging Face Transformers are prioritized:

- Qwen 2.5 0.5B
- TinyLlama 1.1B
- Llama 3.2 1B
- Qwen 2.5 1.5B
- Phi-3.5 Mini
- Gemma 3 1B

If hardware allows, larger or quantized model variants will be tested to examine whether scale reduces hallucination. The main baseline is direct evaluation on HaluEval, comparing model outputs against human-labeled ground truth. We also test simple prompt variations — direct answering vs. asking the model to first judge whether a response is supported or hallucinated.

---

## Datasets

- **HaluEval** (primary) — human-annotated hallucination examples across question answering, dialogue, and summarization tasks.
- **Secondary dataset** (e.g. TruthfulQA or a public Kaggle benchmark) — used to test whether findings generalize beyond HaluEval.
- **Synthetic test set** — a small set of factual questions with known ground-truth answers to verify the evaluation pipeline.

---

## Evaluation Criteria

### Quantitative Metrics

- Hallucination rate, accuracy, precision, recall, F1, and error rate by task type
- Comparisons across model family and size to assess whether larger models hallucinate less

### Qualitative Analysis

Incorrect outputs are categorized as:

- Minor factual error
- Unsupported claim
- Fabricated answer
- Confidently wrong

Consistent labeling rules and equal sample sizes are used to control for bias.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Benchmark performance may not reflect real-world model behavior | Clearly limit conclusions to the datasets and tasks tested |
| Models may be too large or slow to run on available hardware | Use quantized versions, smaller parameter sizes, or a fixed subset of the benchmark |
| Hallucination can be difficult to define consistently | Use HaluEval's existing labels as ground truth and apply the same scoring process to every model |
| Automated evaluation misses subtle errors | Include a manual review step for error analysis |

---

## Implementation

The project is implemented in Python using the following tools:

- **Hugging Face Transformers** and/or **Ollama** — model inference
- **pandas** — organizing outputs
- **scikit-learn** — evaluation metrics
- **matplotlib** — visualizing results

All code is placed in this open-source repository with scripts for loading datasets, running models, saving outputs, computing metrics, and generating result tables.

---

## Success Criteria

Success is defined by producing a working, reproducible pipeline and a clear empirical comparison of hallucination behavior across multiple open-source models. At minimum, the project reports hallucination performance by model family, model size, and task type. A stronger outcome includes comparisons across multiple datasets, prompt formats, and hallucination severity categories. The final result should help future researchers understand which open-source models are more reliable, where they fail, and what evaluation practices are most useful for studying hallucination.
