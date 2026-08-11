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

llms = [m for m in models_ordered if model_type.get(m) == "LLM"]
embs = [m for m in models_ordered if model_type.get(m) != "LLM"]
half = (len(embs) + 1) // 2

# Portrait-friendly: split the 36-model matrix into three tables that each fit
# \textwidth upright, so no page or content rotation is needed (COLM requires
# every page to stay US Letter).
PARTS = [
    (llms,           "llms",   "the ten LLMs"),
    (embs[:half],    "embA",   f"embedding models 1--{half}"),
    (embs[half:],    "embB",   f"embedding models {half+1}--{len(embs)}"),
]


def cell(m, task_key):
    """One score cell, bolded when it is the best across ALL 36 models."""
    if task_key not in pivot.columns or m not in pivot.index:
        return "--"
    val = pivot.loc[m, task_key]
    if pd.isna(val):
        return "--"
    formatted = f"{val*100:.1f}"
    if abs(val - best_per_task[task_key]) < 1e-5:
        return r"\textbf{" + formatted + r"}"
    return formatted


def build_part(part_models, label_suffix, blurb, is_first):
    out = []
    out.append(r"\begin{table*}[htbp]")
    out.append(r"\centering")
    out.append(r"\scriptsize")
    out.append(r"\setlength{\tabcolsep}{3pt}")
    out.append(r"\renewcommand{\arraystretch}{0.95}")
    out.append(r"\resizebox{\textwidth}{!}{%")
    out.append(r"\begin{tabular}{@{}ll" + "r" * len(part_models) + r"@{}}")
    out.append(r"\toprule")

    header = [r"\textbf{Task}", r"\textbf{Cat.}"]
    for m in part_models:
        s_ = short(m)
        if model_type.get(m) == "LLM":
            header.append(r"\rotatebox{90}{\scriptsize\textbf{" + s_ + r"}}")
        else:
            header.append(r"\rotatebox{90}{\scriptsize " + s_ + r"}")
    out.append(" & ".join(header) + r" \\")
    out.append(r"\midrule")

    prev_cat = None
    for task_key in tasks_ordered:
        display_name, cat = task_info[task_key]
        cat_abbrev = cat[:3] if cat != "PairClassification" else "PairCls"
        if cat != prev_cat:
            if prev_cat is not None:
                out.append(r"\midrule")
            out.append(r"\multicolumn{" + str(len(part_models) + 2) +
                       r"}{l}{\emph{" + category_display[cat] + r"}} \\")
            prev_cat = cat
        row = [display_name, cat_abbrev] + [cell(m, task_key) for m in part_models]
        out.append(" & ".join(row) + r" \\")

    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"}")

    if is_first:
        cap = (r"\caption{\textbf{Full per-task scores: " + blurb + r".} "
               r"Scores for all " + str(n_models) + r" models are split across "
               r"Tables~\ref{tab:full_per_task_all_models}--\ref{tab:full_per_task_embB} "
               r"so each fits upright. Models are ordered by overall score. "
               r"A bold score is the best across all " + str(n_models) +
               r" models, not merely within this table; `--' denotes a missing result.}")
        lab = r"\label{tab:full_per_task_all_models}"
    else:
        cap = (r"\caption{\textbf{Full per-task scores: " + blurb + r".} "
               r"Continues Table~\ref{tab:full_per_task_all_models}; bold marks the best "
               r"score across all " + str(n_models) + r" models.}")
        lab = r"\label{tab:full_per_task_" + label_suffix + r"}"
    out.append(cap)
    out.append(lab)
    out.append(r"\end{table*}")
    return out


lines = [
    f"% Full per-task scores for all {n_models} models ({len(llms)} LLMs, {len(embs)} embeddings), {n_tasks} tasks",
    "% auto-generated by gen_full_scores_table.py",
    r"% Portrait tables (no landscape/sideways): COLM requires every page to remain US Letter.",
]
for i, (part_models, suffix, blurb) in enumerate(PARTS):
    lines += build_part(part_models, suffix, blurb, is_first=(i == 0))
    if i < len(PARTS) - 1:
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
