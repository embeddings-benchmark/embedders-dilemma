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
n_models = len(models_ordered)
n_tasks  = len(tasks_ordered)

# Column spec: LLMs first, then a thick separator |, then embeddings
n_llms = sum(1 for m in models_ordered if model_type.get(m) == "LLM")
sep = r"@{\hspace{5pt}}!{\color{gray!35}\vrule width 0.3pt}@{\hspace{5pt}}"
col_spec = "ll" + "r" * n_llms + sep + "r" * (len(models_ordered) - n_llms)

lines = []

lines.append(f"% Full per-task scores for all {n_models} models ({n_llms} LLMs, {n_models - n_llms} embeddings), {n_tasks} tasks – auto-generated by gen_full_scores_table.py")
lines.append(r"% Requires \usepackage{rotating} in the preamble.")
lines.append(r"\begin{sidewaystable*}")
lines.append(r"\centering")
lines.append(r"\tiny")
lines.append(r"\setlength{\tabcolsep}{2.5pt}")
lines.append(r"\renewcommand{\arraystretch}{0.95}")

# build the tabular
lines.append(r"\resizebox{\linewidth}{!}{%")
lines.append(r"\begin{tabular}{@{}" + col_spec + r"@{}}")
lines.append(r"\toprule")

# ── header row: model short names (rotated, scriptsize) ───────────────────────
header_parts = [r"\textbf{Task}", r"\textbf{Cat.}"]
for m in models_ordered:
    s = short(m)
    t = model_type.get(m, "Embedding")
    if t == "LLM":
        header_parts.append(r"\rotatebox{90}{\scriptsize\textbf{" + s + r"}}")
    else:
        header_parts.append(r"\rotatebox{90}{\scriptsize " + s + r"}")
lines.append(" & ".join(header_parts) + r" \\")
lines.append(r"\midrule")

# ── data rows grouped by category ─────────────────────────────────────────────
cat_task_ranges = {}  # for \cmidrule after category
prev_cat = None
row_idx = 0

for task_key in tasks_ordered:
    display_name, cat = task_info[task_key]
    cat_abbrev = cat[:3] if cat != "PairClassification" else "PairCls"

    # category section header
    if cat != prev_cat:
        if prev_cat is not None:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{" + str(n_models + 2) + r"}{l}{\emph{" + category_display[cat] + r"}} \\")
        prev_cat = cat
        row_idx = 0

    row_parts = [display_name, cat_abbrev]

    for m in models_ordered:
        if task_key not in pivot.columns or m not in pivot.index:
            row_parts.append("--")
            continue
        val = pivot.loc[m, task_key]
        if pd.isna(val):
            row_parts.append("--")
            continue

        # scores on the 0--100 scale (x100), matching the MTEB leaderboard
        formatted = f"{val*100:.1f}"
        # bold if best (within floating-point tolerance)
        if abs(val - best_per_task[task_key]) < 1e-5:
            row_parts.append(r"\textbf{" + formatted + r"}")
        else:
            row_parts.append(formatted)

    lines.append(" & ".join(row_parts) + r" \\")
    row_idx += 1

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"}")

# ── caption & label ────────────────────────────────────────────────────────────
n_emb = sum(1 for m in models_ordered if model_type.get(m) == "Embedding")
n_llm = sum(1 for m in models_ordered if model_type.get(m) == "LLM")

caption = (
    r"\caption{\textbf{Full per-task scores for all " + str(len(models_ordered)) + r" models.} "
    r"Models are ordered by overall score; LLM column headers and the best score per task are bold. "
    r"Metrics are listed by category, and `--' denotes a missing result.}"
)
lines.append(caption)
lines.append(r"\label{tab:full_per_task_all_models}")

lines.append(r"\end{sidewaystable*}")

latex = "\n".join(lines)

out_path = "tables/full_per_task_all_models.tex"
with open(out_path, "w") as f:
    f.write(latex + "\n")

print(f"Written {out_path}")
print(f"Models: {len(models_ordered)}, Tasks: {len(tasks_ordered)}")
print("\nModel order (by avg score desc):")
for m in models_ordered:
    print(f"  {short(m):20s}  {model_score[m]:.4f}  {model_type.get(m)}")
