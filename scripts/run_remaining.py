#!/usr/bin/env python3
"""Protocol A + Steps 3–6. Run from repo root: python scripts/run_remaining.py"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OLLAMA_URL = "http://localhost:11434"
MODELS = ["llama3.2:1b", "qwen3:1.7b", "gemma3:1b"]
N_PER_TASK = 500
MAX_CTX_CHARS = 2000
SLEEP_SEC = 0.4
RANDOM_SEED = 42
TASKS = ["qa", "dialogue", "summarization", "general"]


def model_slug(model):
    return model.replace(":", "_").replace("/", "_")


def progress_path(model):
    return RESULTS_DIR / f"ollama_progress_{model_slug(model)}.json"


def run_csv_path(model):
    return RESULTS_DIR / f"ollama_run_{model_slug(model)}.csv"


def metrics_csv_path(model):
    return RESULTS_DIR / f"task_metrics_{model_slug(model)}.csv"


def protocol_a_progress_path(model):
    return RESULTS_DIR / f"protocol_a_progress_{model_slug(model)}.json"


def protocol_a_run_path(model):
    return RESULTS_DIR / f"protocol_a_run_{model_slug(model)}.csv"


def protocol_a_metrics_path(model):
    return RESULTS_DIR / f"protocol_a_metrics_{model_slug(model)}.csv"


def load_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def to_sample(task, row, idx):
    if task == "qa":
        return dict(
            task=task,
            sample_id=idx,
            question=row["question"],
            context=row.get("knowledge", ""),
            ground_truth=row["right_answer"],
            hallucinated=row["hallucinated_answer"],
        )
    if task == "dialogue":
        return dict(
            task=task,
            sample_id=idx,
            question=row.get("dialogue_history", ""),
            context=row.get("knowledge", ""),
            ground_truth=row["right_response"],
            hallucinated=row["hallucinated_response"],
        )
    if task == "summarization":
        return dict(
            task=task,
            sample_id=idx,
            question="Summarize this document.",
            context=row.get("document", ""),
            ground_truth=row["right_summary"],
            hallucinated=row["hallucinated_summary"],
        )
    label = str(row.get("hallucination_label", row.get("hallucination", ""))).lower()
    resp = row.get("chatgpt_response", "")
    return dict(
        task=task,
        sample_id=idx,
        question=row["user_query"],
        context="",
        ground_truth=resp if label in ("no", "n") else "",
        hallucinated=resp if label in ("yes", "y") else "",
    )


def load_samples():
    samples = []
    for task in TASKS:
        path = DATA_DIR / f"{task}_data.json"
        rows = load_jsonl(path)
        n = min(N_PER_TASK, len(rows))
        rng = random.Random(RANDOM_SEED + TASKS.index(task))
        picked = rng.sample(rows, n)
        for i, row in enumerate(picked):
            samples.append(to_sample(task, row, i))
    return samples


def trim(text, n=MAX_CTX_CHARS):
    text = str(text or "")
    return text if len(text) <= n else text[:n] + "..."


def ollama_available(retries=8):
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            r.raise_for_status()
            tags = {m["name"] for m in r.json().get("models", [])}
            if tags:
                return tags
        except Exception as e:
            last_err = e
        time.sleep(1)
    print(f"ollama tags unavailable after {retries} tries: {last_err}")
    return set()


def ensure_ollama():
    if ollama_available(retries=2):
        return True
    import subprocess

    print("restarting ollama serve…")
    subprocess.Popen(
        ["/opt/homebrew/bin/ollama", "serve"],
        stdout=open("/tmp/ollama_serve.log", "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return bool(ollama_available(retries=15))


def ask_ollama(model, prompt, retries=6, num_predict=16, temperature=0.0):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        # Disable thinking for Qwen3-style models so answers land in `content`
        # (otherwise num_predict is spent on thinking and content is empty).
        "think": False,
        "options": {"num_predict": num_predict, "temperature": temperature},
    }
    for attempt in range(retries):
        try:
            if attempt and not ollama_available(retries=1):
                ensure_ollama()
            r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=180)
            r.raise_for_status()
            msg = r.json().get("message", {})
            content = (msg.get("content") or "").strip()
            if content:
                return content
            # Qwen3 (and similar) may put the reply in `thinking` with empty content
            thinking = (msg.get("thinking") or "").strip()
            if thinking:
                return thinking
            return ""
        except (requests.ConnectionError, requests.Timeout) as e:
            print(f"ollama call failed ({e}); retry {attempt + 1}/{retries}")
            ensure_ollama()
            time.sleep(2 * (attempt + 1))
    raise ConnectionError(f"ollama unreachable after {retries} tries")


def build_protocol_a_prompt(sample, candidate):
    return (
        "Decide whether the candidate answer is a hallucination "
        "(fabricated or unsupported by the context). "
        "Reply with only Yes or No.\n\n"
        f"Context:\n{trim(sample['context'])}\n\n"
        f"Question / dialogue / task:\n{trim(sample['question'], 800)}\n\n"
        f"Candidate answer:\n{trim(candidate, 800)}\n\n"
        "Is the candidate hallucinated? Yes or No:"
    )


def parse_yes_no(text):
    t = str(text or "").strip().lower()
    t = re.sub(r"[^a-z\s]", " ", t)
    tokens = t.split()
    if not tokens:
        return None
    if tokens[0] in ("yes", "y"):
        return 1
    if tokens[0] in ("no", "n"):
        return 0
    if "yes" in tokens and "no" not in tokens:
        return 1
    if "no" in tokens and "yes" not in tokens:
        return 0
    return None


def judgment_items(samples):
    items = []
    for s in samples:
        if str(s.get("ground_truth") or "").strip():
            items.append({**s, "candidate_type": "truth", "candidate": s["ground_truth"], "gold_hallucinated": 0})
        if str(s.get("hallucinated") or "").strip():
            items.append({**s, "candidate_type": "hallucinated", "candidate": s["hallucinated"], "gold_hallucinated": 1})
    return items


def judgment_key(row):
    return f"{row['task']}|{row['sample_id']}|{row['candidate_type']}"


def detection_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n": int(len(y_true)),
    }


def run_protocol_a(models, samples):
    items = judgment_items(samples)
    available = ollama_available()
    print(f"Protocol A judgements per model: {len(items)}")
    print(f"Installed Ollama tags: {sorted(available)}")

    for model in models:
        if model not in available:
            print(f"MISSING Protocol A for {model}: not installed (ollama pull {model})")
            continue
        path = protocol_a_progress_path(model)
        results = json.loads(path.read_text()) if path.exists() else []
        done = {judgment_key(r) for r in results}
        remaining = [it for it in items if judgment_key(it) not in done]
        print(f"{model}: {len(done)} done, {len(remaining)} remaining")
        for it in remaining:
            prompt = build_protocol_a_prompt(it, it["candidate"])
            raw = ask_ollama(model, prompt, num_predict=128, temperature=0.0)
            pred = parse_yes_no(raw)
            if pred is None:
                pred = 1 if "yes" in raw.lower() else 0
            row = {
                "task": it["task"],
                "sample_id": it["sample_id"],
                "model": model,
                "candidate_type": it["candidate_type"],
                "candidate": it["candidate"],
                "gold_hallucinated": it["gold_hallucinated"],
                "prompt_sent": prompt,
                "judgment_raw": raw,
                "pred_hallucinated": int(pred),
            }
            row["correct"] = int(row["pred_hallucinated"] == row["gold_hallucinated"])
            results.append(row)
            path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            time.sleep(SLEEP_SEC)
        df = pd.DataFrame(results)
        df.to_csv(protocol_a_run_path(model), index=False)
        rows = []
        for task in sorted(df["task"].unique().tolist()) + ["overall"]:
            sub = df if task == "overall" else df[df["task"] == task]
            m = detection_metrics(sub["gold_hallucinated"].values, sub["pred_hallucinated"].values)
            rows.append({"model": model, "task": task, **m})
        metrics = pd.DataFrame(rows)
        metrics.to_csv(protocol_a_metrics_path(model), index=False)
        print(f"saved {protocol_a_run_path(model).name} ({len(df)} rows)")
        print(metrics.round(3).to_string(index=False))


def norm(s):
    return re.sub(r"[^\w\s]", "", str(s).lower()).strip()


def word_overlap(a, b):
    wa, wb = set(norm(a).split()), set(norm(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def is_correct(row):
    truth, hall, ans, task = row["ground_truth"], row["hallucinated"], row["llama_answer"], row["task"]
    t, a = norm(truth), norm(ans)
    if not t:
        return False
    if task == "qa":
        if t in a:
            return True
        tw = set(t.split())
        if tw and tw <= set(a.split()):
            return True
    ot = word_overlap(ans, truth)
    oh = word_overlap(ans, hall) if hall else 0.0
    return ot > oh


def score_dataframe(df):
    df = df.copy()
    df["overlap_truth"] = df.apply(lambda r: word_overlap(r["llama_answer"], r["ground_truth"]), axis=1)
    df["overlap_hall"] = df.apply(lambda r: word_overlap(r["llama_answer"], r["hallucinated"]), axis=1)
    df["correct"] = df.apply(is_correct, axis=1)
    return df


def load_protocol_b_scored():
    scored = {}
    for model in MODELS:
        path = run_csv_path(model)
        if not path.exists():
            print(f"MISSING Protocol B run for {model}")
            continue
        df = pd.read_csv(path)
        if "model" not in df.columns:
            df["model"] = model
        scored[model] = score_dataframe(df)
    return scored


def build_combined_results(scored_b):
    combined = {}
    coverage = []
    for model in MODELS:
        a_path = protocol_a_run_path(model)
        a_df = pd.read_csv(a_path) if a_path.exists() else None
        b_df = scored_b.get(model)
        sample_ids = set()
        if a_df is not None:
            sample_ids |= set(zip(a_df["task"], a_df["sample_id"]))
        if b_df is not None:
            sample_ids |= set(zip(b_df["task"], b_df["sample_id"]))
        n_a = n_b = n_both = 0
        for task, sid in sorted(sample_ids):
            key = f"{task}|{sid}|{model}"
            entry = {"task": task, "sample_id": int(sid), "model": model}
            if a_df is not None:
                sub = a_df[(a_df["task"] == task) & (a_df["sample_id"] == sid)]
                if len(sub):
                    judgements = []
                    for _, r in sub.iterrows():
                        judgements.append({
                            "candidate_type": r["candidate_type"],
                            "gold_hallucinated": int(r["gold_hallucinated"]),
                            "pred_hallucinated": int(r["pred_hallucinated"]),
                            "correct": int(r["correct"]),
                            "judgment_raw": r["judgment_raw"],
                        })
                    entry["protocol_a"] = {
                        "judgements": judgements,
                        "accuracy": float(sub["correct"].mean()),
                    }
                    n_a += 1
            if b_df is not None:
                sub = b_df[(b_df["task"] == task) & (b_df["sample_id"] == sid)]
                if len(sub):
                    r = sub.iloc[0]
                    entry["protocol_b"] = {
                        "llama_answer": r["llama_answer"],
                        "overlap_truth": float(r["overlap_truth"]),
                        "overlap_hall": float(r["overlap_hall"]),
                        "correct": bool(r["correct"]),
                        "hallucinated_label": int(not bool(r["correct"])),
                    }
                    n_b += 1
            if "protocol_a" in entry and "protocol_b" in entry:
                n_both += 1
            combined[key] = entry
        coverage.append({
            "model": model,
            "protocol_a_items": n_a if a_df is not None else "MISSING",
            "protocol_b_items": n_b if b_df is not None else "MISSING",
            "both": n_both,
            "combined_keys": len(sample_ids),
        })
        if a_df is None:
            print(f"MISSING Protocol A for {model} in combined_results")
        if b_df is None:
            print(f"MISSING Protocol B for {model} in combined_results")
    out = RESULTS_DIR / "combined_results.json"
    out.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    cov = pd.DataFrame(coverage)
    print("Coverage summary:")
    print(cov.to_string(index=False))
    print(f"wrote {out} ({len(combined)} keys)")
    return combined, cov


def build_protocol_comparison(scored_b):
    rows = []
    for model in MODELS:
        a_path = protocol_a_metrics_path(model)
        if not a_path.exists():
            print(f"MISSING Protocol A metrics for {model}")
            continue
        a_m = pd.read_csv(a_path)
        a_m = a_m[a_m["task"] != "overall"].set_index("task")["accuracy"]
        if model not in scored_b:
            print(f"MISSING Protocol B for comparison: {model}")
            continue
        b_df = scored_b[model]
        b_acc = b_df.groupby("task")["correct"].mean()
        for task in TASKS:
            if task not in a_m.index or task not in b_acc.index:
                continue
            a = float(a_m.loc[task])
            b = float(b_acc.loc[task])
            diff = a - b
            rows.append({
                "model": model,
                "task": task,
                "protocol_a_accuracy": round(a, 4),
                "protocol_b_accuracy": round(b, 4),
                "a_minus_b": round(diff, 4),
                "sign_flip": bool(diff < 0),
            })
    if not rows:
        print("MISSING protocol_comparison.csv — need both Protocol A and B metrics")
        return None
    cmp_df = pd.DataFrame(rows)
    out = RESULTS_DIR / "protocol_comparison.csv"
    cmp_df.to_csv(out, index=False)
    print(cmp_df.to_string(index=False))
    flips = cmp_df[cmp_df["sign_flip"]]
    print(f"sign flips (B > A): {len(flips)} / {len(cmp_df)}")
    print(f"wrote {out}")
    return cmp_df


def make_figures(scored_b):
    captions = {}

    if scored_b:
        plot_rows = []
        for model, df in scored_b.items():
            for task, g in df.groupby("task"):
                plot_rows.append({"model": model, "task": task, "accuracy": g["correct"].mean()})
        plot_df = pd.DataFrame(plot_rows)
        pivot = plot_df.pivot(index="task", columns="model", values="accuracy").reindex(TASKS)
        ax = pivot.plot(kind="bar", figsize=(10, 5))
        ax.set_ylim(0, 1)
        ax.set_ylabel("Protocol B accuracy")
        ax.set_title(f"Protocol B accuracy by task (seed={RANDOM_SEED})")
        ax.legend(title="model")
        plt.xticks(rotation=20)
        plt.tight_layout()
        path = RESULTS_DIR / "task_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        nums = []
        for model in pivot.columns:
            for task in TASKS:
                if task in pivot.index and pd.notna(pivot.loc[task, model]):
                    nums.append(f"{model}/{task}={pivot.loc[task, model]:.3f}")
        captions["task_comparison.png"] = (
            "Figure: Protocol B task comparison. Grouped bars show free-generation accuracy "
            f"(seed={RANDOM_SEED}) for each HaluEval task on the x-axis and each evaluated "
            "model as a series. Accuracy is the fraction of items where is_correct is True "
            "(QA: ground-truth phrase contained in the answer; other tasks: word_overlap with "
            "truth exceeds overlap with the hallucinated reference). "
            f"Values: {'; '.join(nums)}. "
            "Higher bars indicate easier tasks under free generation; compare across models "
            "only where a Protocol B run exists (MISSING models are omitted)."
        )
        print(f"wrote {path}")

    scatter_rows = []
    for model in MODELS:
        a_path = protocol_a_metrics_path(model)
        b_path = RESULTS_DIR / f"protocol_b_metrics_{model_slug(model)}.csv"
        if a_path.exists():
            a = pd.read_csv(a_path)
            a = a[a["task"] != "overall"]
            for _, r in a.iterrows():
                scatter_rows.append({
                    "model": model,
                    "task": r["task"],
                    "protocol": "A",
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "f1": r["f1"],
                })
        if b_path.exists():
            b = pd.read_csv(b_path)
            b = b[(b["task"] != "overall") & (b["detector"] == "logistic_regression")]
            for _, r in b.iterrows():
                scatter_rows.append({
                    "model": model,
                    "task": r["task"],
                    "protocol": "B-LR",
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "f1": r["f1"],
                })
    if scatter_rows:
        sdf = pd.DataFrame(scatter_rows)
        fig, ax = plt.subplots(figsize=(7, 6))
        markers = {"A": "o", "B-LR": "s"}
        for protocol, g in sdf.groupby("protocol"):
            ax.scatter(
                g["recall"], g["precision"],
                marker=markers.get(protocol, "x"),
                s=70, label=protocol, alpha=0.85,
            )
            for _, r in g.iterrows():
                ax.annotate(
                    f"{r['model'].split(':')[0][:5]}/{r['task'][:3]}",
                    (r["recall"], r["precision"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points",
                )
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Recall (positive = hallucinated)")
        ax.set_ylabel("Precision (positive = hallucinated)")
        ax.set_title("Precision–recall by task (Protocols A and B)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = RESULTS_DIR / "precision_recall_scatter.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        support = "; ".join(
            f"{r.protocol}/{r.model}/{r.task}: P={r.precision:.2f} R={r.recall:.2f} F1={r.f1:.2f}"
            for r in sdf.itertuples()
        )
        captions["precision_recall_scatter.png"] = (
            "Figure: Precision–recall scatter. Each point is one (model, task) under Protocol A "
            "(circles: Yes/No discrimination, positive class = hallucinated) or Protocol B "
            "logistic regression on the held-out test split (squares: detector trained on "
            "overlap_truth/overlap_hall features, same positive class). The x-axis is recall "
            "and the y-axis is precision; the top-right corner is best. "
            f"Supporting numbers: {support}."
        )
        print(f"wrote {path}")
    else:
        print("MISSING precision_recall_scatter.png — no A/B metrics yet")

    cap_path = RESULTS_DIR / "figure_captions.md"
    lines = ["# Figure captions\n"]
    for name, text in captions.items():
        lines.append(f"## `{name}`\n\n{text}\n")
    cap_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {cap_path}")
    return captions


def build_protocol_b_metrics_if_needed(scored_b):
    try:
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.losses import BinaryCrossentropy
        from tensorflow.keras.metrics import BinaryAccuracy
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
    except Exception as e:
        print(f"SKIP Protocol B classifier metrics (tensorflow unavailable): {e}")
        return

    for model, df in scored_b.items():
        out = RESULTS_DIR / f"protocol_b_metrics_{model_slug(model)}.csv"
        if out.exists():
            continue
        df = df.reset_index(drop=True)
        X = df[["overlap_truth", "overlap_hall"]].values
        y_hall = (~df["correct"].astype(bool)).astype(int).values
        idx = np.arange(len(df))
        if len(np.unique(y_hall)) < 2:
            print(f"SKIP classifiers for {model}: only one class")
            continue
        X_tr, X_te, y_tr, y_te, i_tr, i_te = train_test_split(
            X, y_hall, idx, test_size=0.25, random_state=RANDOM_SEED, stratify=y_hall
        )
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        grid = GridSearchCV(LogisticRegression(max_iter=1000), {"C": [0.1, 1, 10]}, cv=3)
        grid.fit(X_tr_s, y_tr)
        lr_proba = grid.predict_proba(X_te_s)[:, 1]
        lr_pred = (lr_proba >= 0.5).astype(int)
        nn = Sequential([
            Dense(8, activation="relu", input_shape=(2,)),
            Dense(1, activation="sigmoid"),
        ])
        nn.compile(optimizer=Adam(0.01), loss=BinaryCrossentropy(), metrics=[BinaryAccuracy()])
        nn.fit(X_tr_s, y_tr, epochs=40, batch_size=16, verbose=0)
        nn_proba = nn.predict(X_te_s, verbose=0).ravel()
        nn_pred = (nn_proba >= 0.5).astype(int)
        proba_df = pd.DataFrame({
            "task": df.loc[i_te, "task"].values,
            "sample_id": df.loc[i_te, "sample_id"].values,
            "y_true_hallucinated": y_te,
            "lr_proba_hall": lr_proba,
            "lr_pred_hall": lr_pred,
            "nn_proba_hall": nn_proba,
            "nn_pred_hall": nn_pred,
        })
        proba_df.to_csv(RESULTS_DIR / f"protocol_b_classifier_proba_{model_slug(model)}.csv", index=False)
        rows = []
        for detector, col in [("logistic_regression", "lr_pred_hall"), ("neural_network", "nn_pred_hall")]:
            for task in sorted(proba_df["task"].unique().tolist()) + ["overall"]:
                sub = proba_df if task == "overall" else proba_df[proba_df["task"] == task]
                m = detection_metrics(sub["y_true_hallucinated"].values, sub[col].values)
                rows.append({"model": model, "detector": detector, "task": task, **m})
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"wrote {out.name}")


def build_prompt(sample):
    task, ctx, q = sample["task"], trim(sample["context"]), sample["question"]
    if task == "summarization":
        return f"Summarize in 2-3 sentences.\n\nDocument:\n{ctx}"
    if task == "dialogue":
        return f"Knowledge:\n{ctx}\n\nDialogue:\n{trim(q, 800)}\n\nNext response (short):"
    if task == "general":
        return f"{q}\n\nShort answer:"
    return f"Context:\n{ctx}\n\nQuestion: {q}\n\nShort answer:"


def sample_key(s):
    return f"{s['task']}|{s['sample_id']}"


def run_protocol_b(models, samples):
    available = ollama_available()
    print(f"Protocol B samples per model: {len(samples)}")
    print(f"Installed Ollama tags: {sorted(available)}")
    for model in models:
        if model not in available:
            print(f"MISSING Protocol B for {model}: not installed")
            continue
        path = progress_path(model)
        results = json.loads(path.read_text()) if path.exists() else []
        done = {sample_key(r) for r in results}
        remaining = [s for s in samples if sample_key(s) not in done]
        print(f"{model}: {len(done)} done, {len(remaining)} remaining")
        for s in remaining:
            if not ensure_ollama():
                raise ConnectionError("ollama down")
            prompt = build_prompt(s)
            answer = ask_ollama(model, prompt, num_predict=160, temperature=0.1)
            row = {**s, "model": model, "prompt_sent": prompt, "llama_answer": answer}
            results.append(row)
            path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            time.sleep(SLEEP_SEC)
        df = pd.DataFrame(results)
        df.to_csv(run_csv_path(model), index=False)
        print(f"saved {run_csv_path(model).name} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-protocol-a", action="store_true")
    parser.add_argument("--skip-protocol-b", action="store_true")
    parser.add_argument("--models", nargs="*", default=MODELS)
    args = parser.parse_args()

    ensure_ollama()
    samples = load_samples()
    print(f"Loaded {len(samples)} samples across tasks")

    if not args.skip_protocol_a:
        run_protocol_a(args.models, samples)
    if not args.skip_protocol_b:
        run_protocol_b(args.models, samples)

    scored_b = load_protocol_b_scored()
    build_protocol_b_metrics_if_needed(scored_b)
    build_combined_results(scored_b)
    build_protocol_comparison(scored_b)
    make_figures(scored_b)


if __name__ == "__main__":
    main()
