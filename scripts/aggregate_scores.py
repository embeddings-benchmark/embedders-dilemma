#!/usr/bin/env python3
"""Aggregate embedding and LLM results into a single tidy dataframe.

Output: scores.csv with columns:
    model, model_type, task, task_category, score

Only the 38 canonical tasks (from embedding_results) are included.
Retrieval tasks use recall@1 as the common metric.
"""

import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Canonical task set (driven by embedding results)
# Populated at runtime from embedding_results/ — no hardcoding needed.

# LLM filename stem → canonical task name
# For tasks where prepending "LLM" is not enough.
LLM_TO_CANONICAL = {
    "SICK-R":                     "LLMSICKR",
    "STS22.v2":                   "LLMSTS22v2",
    "RTE3":                       "LLMRTE3PC",
    "SprintDuplicateQuestions":   "LLMSprintDuplicateQuestionsPC",
    "TwitterURLCorpus-VN":        "LLMTwitterURLCorpusPC",
    "BiorxivClusteringP2P":       "LLMBiorxivClusteringP2PV2",
    "MedrxivClusteringP2P":       "LLMMedrxivClusteringP2PV2",
    "MedrxivClusteringS2S":       "LLMMedrxivClusteringS2SV2",
    "StackExchangeClustering":    "LLMStackExchangeClusteringV2",
    "StackExchangeClusteringP2P": "LLMStackExchangeClusteringP2PV2",
    "TwentyNewsgroupsClustering": "LLMTwentyNewsgroupsClusteringV2",
}

# Tasks dropped from the final MTEB(LLM) selection (still present in older
# embedding_results/ dirs, so we filter them out explicitly):
#   - BigPatentClustering: dropped from clustering (team decision)
#   - TempReasonL1 / SpartQA / WinoGrande / LegalBenchCorporateLobbying:
#     replaced by the new small-corpus retrieval set (no TempReason).
DROP_TASKS = {
    "LLMBigPatentClustering",
    "LLMTempReasonL1",
    "LLMSpartQA",
    "LLMWinoGrande",
    "LLMLegalBenchCorporateLobbying",
}

# Task categories
CATEGORY_OVERRIDES = {
    # STS tasks whose names don't contain "STS"
    "LLMBIOSSES":                     "STS",
    "LLMSICKR":                       "STS",
    "LLMSTS22v2":                     "STS",
    # The 6 final retrieval tasks (small-corpus, no TempReason)
    "LLMAILAStatutes":                "Retrieval",
    "LLMFQuADRetrieval":              "Retrieval",
    "LLMHC3FinanceRetrieval":         "Retrieval",
    "LLMLegalBenchConsumerContractsQA": "Retrieval",
    "LLMPublicHealthQA":              "Retrieval",
    "LLMTwitterHjerneRetrieval":      "Retrieval",
}

def task_category(task: str) -> str:
    if task in CATEGORY_OVERRIDES:
        return CATEGORY_OVERRIDES[task]
    if "Clustering"     in task: return "Clustering"
    if "Classification" in task: return "Classification"
    if "STS"            in task: return "STS"
    if "Retrieval"      in task: return "Retrieval"
    if "PC"             in task: return "PairClassification"
    if "RTE"            in task: return "PairClassification"
    return "Other"

# Retrieval: use recall@1 instead of main_score for embeddings
RETRIEVAL_TASKS = {
    "LLMAILAStatutes",
    "LLMFQuADRetrieval",
    "LLMHC3FinanceRetrieval",
    "LLMLegalBenchConsumerContractsQA",
    "LLMPublicHealthQA",
    "LLMTwitterHjerneRetrieval",
}

# Pair classification: use similarity_accuracy for both sides
# Embedding main_score uses AP (continuous cosine); LLM main_score uses accuracy.
# similarity_accuracy exists in both and is the same metric.
PAIR_CLS_TASKS = {"LLMSprintDuplicateQuestionsPC", "LLMTwitterURLCorpusPC",
                  "LLMLegalBenchPC", "LLMRTE3PC"}

def canonicalize(stem: str) -> str | None:
    """LLM filename stem → canonical name, or None to skip."""
    if stem in LLM_TO_CANONICAL:
        return LLM_TO_CANONICAL[stem]
    if stem.startswith("LLM"):
        return stem
    return "LLM" + stem

def extract_score(scores: dict, task: str, model_type: str) -> float | None:
    split = list(scores.keys())[0]
    s = scores[split]
    if isinstance(s, list):
        s = s[0]
    if task in RETRIEVAL_TASKS:
        if model_type == "embedding":
            return s.get("recall_at_1")
        else:
            return s.get("main_score")  # already recall@1 for LLMs
    if task in PAIR_CLS_TASKS:
        return s.get("similarity_accuracy") or s.get("max_accuracy")
    return s.get("main_score")

# Loaders

def load_embeddings(emb_dir: Path) -> list[dict]:
    rows = []
    for model_dir in sorted(emb_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        subdirs = [d for d in model_dir.iterdir() if d.is_dir()]
        if not subdirs:
            continue
        result_dir = subdirs[0]
        for jf in sorted(result_dir.glob("*.json")):
            if jf.stem == "model_meta":
                continue
            task = jf.stem
            if task in DROP_TASKS:
                continue
            try:
                d = json.loads(jf.read_text())
                score = extract_score(d["scores"], task, "embedding")
                if score is not None:
                    rows.append(dict(
                        model=model_dir.name,
                        model_type="embedding",
                        task=task,
                        task_category=task_category(task),
                        score=round(score, 6),
                    ))
            except Exception as e:
                print(f"  [warn] {model_dir.name}/{jf.name}: {e}")
    return rows


def load_llms(llm_dir: Path, canonical_tasks: set[str]) -> list[dict]:
    rows = []
    for model_dir in sorted(llm_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        # Some model dirs contain two result trees that differ by a single
        # underscore: ".../no_model_name__available/..." (complete / latest)
        # and ".../no_model_name_available/..." (stale partial re-run, e.g.
        # minimax-m2.7 whose clustering scores differ).  Sorting puts the
        # "__available" path first; we keep the first file seen per task and
        # skip the rest, so each task is counted once from the canonical run.
        seen_tasks: set[str] = set()
        for jf in sorted(model_dir.rglob("*.json")):
            if jf.stem == "model_meta" or jf.stem.endswith("_samples"):
                continue
            task = canonicalize(jf.stem)
            if task in DROP_TASKS:
                continue  # dropped from final selection (BigPatent, TempReason, etc.)
            if task not in canonical_tasks:
                continue  # skip RAG, HUME, long-context, etc.
            if task in seen_tasks:
                continue  # duplicate copy of an already-counted task
            seen_tasks.add(task)
            try:
                d = json.loads(jf.read_text())
                score = extract_score(d["scores"], task, "llm")
                if score is not None:
                    rows.append(dict(
                        model=model_dir.name,
                        model_type="llm",
                        task=task,
                        task_category=task_category(task),
                        score=round(score, 6),
                    ))
            except Exception as e:
                print(f"  [warn] {model_dir.name}/{jf.name}: {e}")
    return rows


# Main

def main():
    emb_rows = load_embeddings(ROOT / "embedding_results")
    canonical_tasks = {r["task"] for r in emb_rows}
    llm_rows = load_llms(ROOT / "llm_results", canonical_tasks)

    df = pd.DataFrame(emb_rows + llm_rows)

    emb_models = sorted(df[df.model_type == "embedding"]["model"].unique())
    llm_models = sorted(df[df.model_type == "llm"]["model"].unique())
    rer_models = sorted(df[df.model_type == "reranker"]["model"].unique())

    print(f"Embedding models ({len(emb_models)}):")
    for m in emb_models:
        n = df[(df.model == m) & (df.model_type == "embedding")]["task"].nunique()
        print(f"  {m:55s} {n:2d}/37 tasks")

    print(f"\nLLM models ({len(llm_models)}):")
    for m in llm_models:
        n = df[(df.model == m) & (df.model_type == "llm")]["task"].nunique()
        print(f"  {m:55s} {n:2d}/37 tasks")

    if rer_models:
        print(f"\nReranker pipelines ({len(rer_models)}):")
        for m in rer_models:
            n = df[(df.model == m) & (df.model_type == "reranker")]["task"].nunique()
            print(f"  {m:75s} {n:2d}/6 retrieval tasks")

    print(f"\nTask coverage by category:")
    print(df.groupby(["task_category", "model_type"])["task"].nunique().to_string())

    out = ROOT / "data" / "scores.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
