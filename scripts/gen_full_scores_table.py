"""Generate the full per-task scores table (all models × all 37 tasks)."""

import pandas as pd

scores = pd.read_csv("data/scores.csv")
cost   = pd.read_csv("data/cost_summary.csv")

# ── keep only the 29 paper models ─────────────────────────────────────────────
paper_models = cost["model"].tolist()
scores = scores[scores["model"].isin(paper_models)].copy()

model_score = dict(zip(cost["model"], cost["avg_score"]))
model_short = dict(zip(cost["model"], cost["short_name"]))
model_type  = dict(zip(cost["model"], cost["type"]))

# LLMs first (by score desc), then embeddings (by score desc)
llms  = cost[cost["type"]=="LLM"].sort_values("avg_score", ascending=False)["model"].tolist()
embs  = cost[cost["type"]=="Embedding"].sort_values("avg_score", ascending=False)["model"].tolist()
models_ordered = llms + embs

# ── task display names & ordering ──────────────────────────────────────────────
task_info = {
    # Classification
    "LLMAmazonCounterfactualClassification": ("AmazonCF",       "Classification"),
    "LLMBanking77Classification":            ("Banking77",       "Classification"),
    "LLMImdbClassification":                 ("IMDB",            "Classification"),
    "LLMMTOPDomainClassification":           ("MTOPDomain",      "Classification"),
    "LLMMassiveIntentClassification":        ("MassiveIntent",   "Classification"),
    "LLMMassiveScenarioClassification":      ("MassiveScenario", "Classification"),
    "LLMToxicConversationsClassification":   ("ToxicConvs",      "Classification"),
    "LLMTweetSentimentExtractionClassification": ("TweetSent",  "Classification"),
    # STS
    "LLMBIOSSES":       ("BIOSSES",   "STS"),
    "LLMSICKR":         ("SICK-R",    "STS"),
    "LLMSTS12":         ("STS12",     "STS"),
    "LLMSTS13":         ("STS13",     "STS"),
    "LLMSTS14":         ("STS14",     "STS"),
    "LLMSTS15":         ("STS15",     "STS"),
    "LLMSTS16":         ("STS16",     "STS"),
    "LLMSTS17":         ("STS17",     "STS"),
    "LLMSTS22v2":       ("STS22v2",   "STS"),
    "LLMSTSBenchmark":  ("STSBench",  "STS"),
    # Clustering
    "LLMArxivClusteringP2P":          ("ArxivP2P",  "Clustering"),
    "LLMArxivClusteringS2S":          ("ArxivS2S",  "Clustering"),
    "LLMBiorxivClusteringP2PV2":      ("BioP2P",    "Clustering"),
    "LLMMedrxivClusteringP2PV2":      ("MedP2P",    "Clustering"),
    "LLMMedrxivClusteringS2SV2":      ("MedS2S",    "Clustering"),
    "LLMRedditClusteringP2P":         ("Reddit",    "Clustering"),
    "LLMStackExchangeClusteringP2PV2":("SE-P2P",    "Clustering"),
    "LLMStackExchangeClusteringV2":   ("SE-Cl",     "Clustering"),
    "LLMTwentyNewsgroupsClusteringV2":("20News",    "Clustering"),
    # PairClassification
    "LLMLegalBenchPC":              ("LegalPC",   "PairClassification"),
    "LLMRTE3PC":                    ("RTE3",      "PairClassification"),
    "LLMSprintDuplicateQuestionsPC":("SprintDup", "PairClassification"),
    "LLMTwitterURLCorpusPC":        ("TwtURL",    "PairClassification"),
    # Retrieval
    "LLMAILAStatutes":                    ("AILA",       "Retrieval"),
    "LLMFQuADRetrieval":                  ("FQuAD",      "Retrieval"),
    "LLMHC3FinanceRetrieval":             ("HC3Fin",     "Retrieval"),
    "LLMLegalBenchConsumerContractsQA":   ("ConsumerQA", "Retrieval"),
    "LLMPublicHealthQA":                  ("PubHealth",  "Retrieval"),
    "LLMTwitterHjerneRetrieval":          ("TwtHjerne",  "Retrieval"),
}

category_order = ["Classification", "STS", "Clustering", "PairClassification", "Retrieval"]
category_display = {
    "Classification":    r"Classification (Accuracy)",
    "STS":               r"STS (Spearman $\rho$)",
    "Clustering":        r"Clustering (V-measure)",
    "PairClassification":r"Pair Classification (AP / Accuracy)",
    "Retrieval":         r"Retrieval (Recall@1)",
}

# ordered task list
tasks_ordered = [t for cat in category_order
                   for t, (_, c) in task_info.items() if c == cat]

# ── pivot to model × task matrix ───────────────────────────────────────────────
pivot = scores.pivot(index="model", columns="task", values="score")

# keep only tasks actually present in the data (guards against stale dict entries)
tasks_ordered = [t for t in tasks_ordered if t in pivot.columns]

# ── per-task best score (for bolding) ─────────────────────────────────────────
best_per_task = pivot.max(axis=0)

# ── short name for LLMs ───────────────────────────────────────────────────────
llm_short = {
    "google__gemini-3.1-pro-preview":       r"Gemini 3.1 Pro",
    "google__gemini-3-flash-preview":       r"Gemini 3 Flash",
    "google__gemini-3.1-flash-lite-preview":r"Gemini 3.1 FLite",
}

def short(m):
    if m in llm_short:
        return llm_short[m]
    return model_short.get(m, m.split("__")[-1][:12])

# ── build LaTeX ────────────────────────────────────────────────────────────────
# One upright table per task category: models are ROWS (36) and the category's
# tasks are COLUMNS (4-10). Nothing is rotated, so every page stays US Letter
# (COLM requirement) and both model and task names read horizontally.
n_models = len(models_ordered)
n_tasks  = len(tasks_ordered)

llms = [m for m in models_ordered if model_type.get(m) == "LLM"]
embs = [m for m in models_ordered if model_type.get(m) != "LLM"]

cats_ordered = []
tasks_by_cat = {}
for k in tasks_ordered:
    _, cat = task_info[k]
    if cat not in tasks_by_cat:
        tasks_by_cat[cat] = []
        cats_ordered.append(cat)
    tasks_by_cat[cat].append(k)

CAT_LABEL = {"Classification": "cls", "STS": "sts", "Clustering": "clu",
             "PairClassification": "pair", "Retrieval": "ret"}


def cell(m, task_key):
    if task_key not in pivot.columns or m not in pivot.index:
        return "--"
    val = pivot.loc[m, task_key]
    if pd.isna(val):
        return "--"
    formatted = f"{val*100:.1f}"
    if abs(val - best_per_task[task_key]) < 1e-5:
        return r"\textbf{" + formatted + r"}"
    return formatted


def _raw_mean(m, keys):
    vals = [pivot.loc[m, k] for k in keys
            if k in pivot.columns and m in pivot.index and not pd.isna(pivot.loc[m, k])]
    return sum(vals) / len(vals) if vals else None


def cat_mean(m, keys, best):
    """Category mean: full precision then rounded - the same path
    Table~\\ref{tab:main_results} uses, so the two never disagree.
    Bold marks the single best mean in the category, matching how bold is
    used everywhere else in these tables (best in column)."""
    v = _raw_mean(m, keys)
    if v is None:
        return "--"
    out = f"{v*100:.1f}"
    return r"\textbf{" + out + r"}" if best is not None and abs(v - best) < 1e-9 else out


def build_cat(cat, is_first):
    keys = tasks_by_cat[cat]
    disp = category_display[cat]
    _means = [x for x in (_raw_mean(m, keys) for m in models_ordered) if x is not None]
    best_mean = max(_means) if _means else None
    out = []
    out.append(r"\begin{table*}[htbp]")
    out.append(r"\centering")
    out.append(r"\small")
    out.append(r"\setlength{\tabcolsep}{4pt}")
    out.append(r"\renewcommand{\arraystretch}{0.95}")
    # Shrink-only: \resizebox{\textwidth} would MAGNIFY a narrow table (and its
    # height with it), which overflows the page. Scale down only when needed.
    out.append(r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%")
    sep = r"@{\hspace{6pt}}!{\color{gray!35}\vrule width 0.3pt}@{\hspace{6pt}}"
    out.append(r"\begin{tabular}{@{}l" + "r" * len(keys) + sep + r"r@{}}")
    out.append(r"\toprule")
    out.append(" & ".join([r"\textbf{Model}"] +
                          [r"\textbf{" + task_info[k][0] + r"}" for k in keys] +
                          [r"\textbf{Mean}"]) + r" \\")
    out.append(r"\midrule")

    out.append(r"\multicolumn{" + str(len(keys) + 2) + r"}{l}{\emph{LLMs}} \\")
    for m in llms:
        out.append(" & ".join([short(m)] + [cell(m, k) for k in keys] +
                              [cat_mean(m, keys, best_mean)]) + r" \\")
    out.append(r"\midrule")
    out.append(r"\multicolumn{" + str(len(keys) + 2) + r"}{l}{\emph{Embedding models}} \\")
    for m in embs:
        out.append(" & ".join([short(m)] + [cell(m, k) for k in keys] +
                              [cat_mean(m, keys, best_mean)]) + r" \\")

    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"}")

    # Caption stays minimal: the shared reading instructions (ordering, bolding,
    # missing-value marker) live once in the appendix prose, not five times here.
    cap = r"\caption{\textbf{Per-task scores: " + disp + r".}}"
    lab = (r"\label{tab:full_per_task_all_models}" if is_first
           else r"\label{tab:full_per_task_" + CAT_LABEL.get(cat, cat.lower()) + r"}")
    out.append(cap)
    out.append(lab)
    out.append(r"\end{table*}")
    return out


lines = [
    f"% Full per-task scores: {n_models} models ({len(llms)} LLMs, {len(embs)} embeddings), {n_tasks} tasks",
    "% auto-generated by gen_full_scores_table.py",
    r"% One upright table per category; nothing rotated, so pages stay US Letter (COLM).",
]
for i, cat in enumerate(cats_ordered):
    lines += build_cat(cat, is_first=(i == 0))
    if i < len(cats_ordered) - 1:
        lines.append("")

latex = "\n".join(lines)

out_path = "69c70f50c5d1a4ea5eb6dfec/tables/full_per_task_all_models.tex"
with open(out_path, "w") as f:
    f.write(latex + "\n")

print(f"Written {out_path}")
print(f"Models: {len(models_ordered)}, Tasks: {len(tasks_ordered)}")
print("\nModel order (by avg score desc):")
for m in models_ordered:
    print(f"  {short(m):20s}  {model_score[m]:.4f}  {model_type.get(m)}")
