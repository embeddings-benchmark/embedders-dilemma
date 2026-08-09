#!/usr/bin/env python3
"""Regenerate the paper's data tables from a single source of truth.

Everything derives from data/scores.csv + the cost/token CSVs + the thinking
ablation JSONs, routed through scripts/plotting/registry.py, so the whole paper
stays consistent with the current model/task set (10 complete LLMs, 26 complete
embeddings, 37 canonical tasks; MACRO overall = mean of the 5 category means).

Tables written (12):
  main_results, category_scores, retrieval_tasks, per_task_scores,
  full_results_by_category, significance, models, full_models, llm_tokens,
  embedding_throughput, cost_sensitivity,
  ablation_nocot (reduced-thinking, new retrieval tasks)

NOT written (blocked on missing data — see print summary at the end):
  ablation_fewshot   few-shot classification runs are not in the repo
  token_budget       new retrieval tasks lack computed token counts
  datasets           new retrieval tasks lack sample-count/citation metadata
  full_per_task_all_models  regenerate via scripts/gen_full_scores_table.py

Usage:  python scripts/generate_tables.py
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "plotting"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
import registry  # noqa: E402
import aggregate_rerank_matrix as rr  # noqa: E402  (BEIR/BRIGHT reranker cells)

TABLES = ROOT / "tables"   # LaTeX output, one .tex file per paper table
DATA = ROOT / "data"

CAT_ABBR = {
    "Classification": "Cls", "Clustering": "Clust", "STS": "STS",
    "PairClassification": "PairCls", "Retrieval": "Retr",
}

# Embedding citations (bib keys), keyed by registry model id. LLMs cite "--"
# (the paper already uses "--" for the LLM reference column). Octen has no paper.
EMB_CITE = {
    "tencent__KaLM-Embedding-Gemma3-12B-2511": "hu2025kalmv2",
    "Qwen__Qwen3-Embedding-8B": "qwen2025qwen3embedding",
    "Qwen__Qwen3-Embedding-4B": "qwen2025qwen3embedding",
    "Qwen__Qwen3-Embedding-0.6B": "qwen2025qwen3embedding",
    "jinaai__jina-embeddings-v5-text-small": "sturua2024jinaembeddings3",
    "jinaai__jina-embeddings-v5-text-nano": "sturua2024jinaembeddings3",
    "nvidia__llama-embed-nemotron-8b": "lee2024nvembedimprovedtechniques",
    "Salesforce__SFR-Embedding-2_R": "meng2024sfremb2",
    "Salesforce__SFR-Embedding-Mistral": "meng2024sfremb2",
    "Alibaba-NLP__gte-Qwen2-7B-instruct": "li2023generaltextembeddingsmultistage",
    "Alibaba-NLP__gte-Qwen2-1.5B-instruct": "li2023generaltextembeddingsmultistage",
    "codefuse-ai__F2LLM-v2-14B": "zhang2026f2llmv2",
    "codefuse-ai__F2LLM-v2-8B": "zhang2026f2llmv2",
    "codefuse-ai__F2LLM-v2-4B": "zhang2026f2llmv2",
    "codefuse-ai__F2LLM-v2-1.7B": "zhang2026f2llmv2",
    "codefuse-ai__F2LLM-v2-0.6B": "zhang2026f2llmv2",
    "Linq-AI-Research__Linq-Embed-Mistral": "choi2024linq",
    "google__embeddinggemma-300m": "lee2025geminiembedding",
    "intfloat__multilingual-e5-large-instruct": "wang2024improvingtextembeddingslarge",
    "intfloat__multilingual-e5-large": "wang2024textembeddingsweaklysupervisedcontrastive",
    "intfloat__multilingual-e5-base": "wang2024textembeddingsweaklysupervisedcontrastive",
    "intfloat__multilingual-e5-small": "wang2024textembeddingsweaklysupervisedcontrastive",
    "BAAI__bge-m3": "chen2025m3embeddingmultilingualitymultifunctionalitymultigranularity",
    "Snowflake__snowflake-arctic-embed-l-v2.0": "yu2024arctic",
    "GritLM__GritLM-7B": "muennighoff2025generativerepresentationalinstructiontuning",
    "intfloat__e5-mistral-7b-instruct": "wang2024improvingtextembeddingslarge",
    "bflhc__Octen-Embedding-8B": None,
}


# ─────────────────────────────────────────────────────────────────────────────
# Shared data
# ─────────────────────────────────────────────────────────────────────────────
def load():
    df = registry.load_scores()
    cs = registry.category_scores(df)                      # complete models
    cost = pd.read_csv(DATA / "cost_summary.csv").set_index("model")
    tok = pd.read_csv(DATA / "llm_token_usage.csv").set_index("model")
    thr = pd.read_csv(DATA / "embedding_throughput.csv")
    thr["model"] = thr["model_id"].str.replace("/", "__", regex=False)
    params = dict(zip(thr["model"], thr["params"]))
    return df, cs, cost, tok, thr, params


def fmt_params(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    p = float(p)
    return f"{p/1e9:.1f}B" if p >= 1e9 else f"{p/1e6:.0f}M"


def fmt3(x, bold=False, lead_dot=False) -> str:
    # Scores are reported on the 0--100 scale (x100), matching the MTEB leaderboard.
    # `lead_dot` is retained for call-site compatibility but no longer meaningful.
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{x*100:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


def cost_str(model_type: str, c: float) -> str:
    return f"\\${c:,.2f}" if model_type == "llm" else f"\\${c:.3f}"


def ci_str(lo, hi) -> str:
    return (f"[{'+' if lo>=0 else '$-$'}{abs(lo)*100:.1f}, "
            f"{'+' if hi>=0 else '$-$'}{abs(hi)*100:.1f}]")


def write(name: str, body: str):
    TABLES.mkdir(exist_ok=True)
    (TABLES / f"{name}.tex").write_text(body)
    print(f"  wrote tables/{name}.tex")


# ─────────────────────────────────────────────────────────────────────────────
# Score tables
# ─────────────────────────────────────────────────────────────────────────────
def t_main_results(cs, params, cost):
    """Merged model overview + per-category results.

    Cost used to live in a separate model-overview table whose other columns
    (name, params, overall) all duplicated this one. Merging removes that
    duplication and, more usefully, puts cost on the same row as quality, which
    is the comparison the paper is about.
    """
    cats = registry.CATEGORIES
    counts = {c: len(v) for c, v in registry.canonical_tasks_by_category().items()}
    total = sum(counts.values())
    llms = cs[cs.model_type == "llm"].sort_values("Overall", ascending=False)
    embs = cs[cs.model_type == "embedding"].sort_values("Overall", ascending=False).head(10)
    shown = pd.concat([llms, embs])
    colmax = {c: shown[c].max() for c in cats + ["Overall"]}

    def money(m):
        v = cost.loc[m, "total_cost"] if m in cost.index else None
        if v is None:
            return "--"
        return f"\\${v:,.0f}" if v >= 1 else f"\\${v:.2f}"

    def row(m, r):
        cells = " & ".join(fmt3(r[c], bold=np.isclose(r[c], colmax[c])) for c in cats)
        ov = fmt3(r["Overall"], bold=np.isclose(r["Overall"], colmax["Overall"]))
        return (f"& {registry.display_name(m)} & {fmt_params(params.get(m))} & "
                f"{cells} & {ov} & {money(m)} \\\\")

    hdr = ("& \\textbf{Model} & \\textbf{Params} & "
           + " & ".join(f"\\textbf{{{CAT_ABBR[c]} ({counts[c]})}}" for c in cats)
           + f" & \\textbf{{Overall ({total})}} & \\textbf{{Cost}} \\\\")
    llm_rows = "\n".join(row(m, r) for m, r in llms.iterrows())
    emb_rows = "\n".join(row(m, r) for m, r in embs.iterrows())
    return f"""% Main results: per-category scores and cost (auto-generated)
\\begin{{table*}}[t]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{llcccccccr}}
\\toprule
{hdr}
\\midrule
\\multirow{{{len(llms)}}}{{*}}{{\\rotatebox{{90}}{{\\small LLM}}}}
{llm_rows}
\\midrule
\\multirow{{{len(embs)}}}{{*}}{{\\rotatebox{{90}}{{\\small Embedding}}}}
{emb_rows}
\\bottomrule
\\end{{tabular}}}}
\\caption{{\\textbf{{Model overview and per-category results.}}
All ten LLMs and the ten highest-scoring embedding models. Scores are category means on a 0--100 scale; Overall is their mean.
Cost is one \\mteblm{{}} pass using API rates for LLMs and H100 throughput at \\$2.49/hr for embeddings (\\S\\ref{{sec:cost}}). Bold = best shown; full results are in Appendix~\\ref{{app:models}}.}}
\\label{{tab:main_results}}
\\end{{table*}}
"""



def t_category_scores(cs):
    counts = {c: len(v) for c, v in registry.canonical_tasks_by_category().items()}
    llm = cs[cs.model_type == "llm"].sort_values("Overall", ascending=False)
    best_llm, second_llm = llm.index[0], llm.index[1]
    emb = cs[cs.model_type == "embedding"]
    rows = []
    for c in sorted(registry.CATEGORIES, key=lambda x: -(llm.loc[best_llm, x] - emb[x].max())):
        pro = llm.loc[best_llm, c]
        flash = llm.loc[second_llm, c]
        be_model = emb[c].idxmax()
        be = emb[c].max()
        delta = pro - be
        win = "pro" if delta > 0 else "emb"
        dstr = f"+{delta*100:.1f}" if delta >= 0 else f"$-${abs(delta)*100:.1f}"
        rows.append(
            f"{c.replace('PairClassification','PairCls')} ({counts[c]}) & "
            f"{fmt3(pro, bold=win=='pro')} & {fmt3(flash)} & "
            f"{fmt3(be, bold=win=='emb')} & {registry.display_name(be_model)} & {dstr} \\\\"
        )
    body = "\n".join(rows)
    return f"""% Category-level performance summary (best LLM vs best embedding) (auto-generated)
\\begin{{table}}[h]
\\centering
\\small
\\begin{{tabular}}{{lccccc}}
\\toprule
\\textbf{{Category}} & \\textbf{{{registry.display_name(best_llm)}}} & \\textbf{{{registry.display_name(second_llm)}}} & \\textbf{{Best Emb.}} & \\textbf{{Best Model}} & \\textbf{{$\\Delta$}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{\\textbf{{Category-level performance}} (best LLM vs.\\ best embedding per category).
$\\Delta$ = best LLM $-$ best embedding; bold = winner; task counts in parentheses.}}
\\label{{tab:category_scores}}
\\end{{table}}
"""


def _pivot_task(df):
    keep = set(registry.complete_models(df))
    canon = set(registry.canonical_tasks(df))
    d = df[df.model.isin(keep) & df.task.isin(canon)]
    return d.pivot_table(index="task", columns="model", values="score"), d


def t_retrieval_tasks(df, cs):
    piv, d = _pivot_task(df)
    tcat = registry.task_category_map(df)
    rtasks = [t for t in piv.index if tcat[t] == "Retrieval"]
    llm = cs[cs.model_type == "llm"].sort_values("Overall", ascending=False)
    best_llm, second_llm = llm.index[0], llm.index[1]
    emb_models = cs[cs.model_type == "embedding"].index
    best_emb_overall = cs.loc[emb_models, "Overall"].idxmax()

    rows = []
    for t in sorted(rtasks, key=lambda t: -(piv.loc[t, best_llm] - piv.loc[t, emb_models].max())):
        pro = piv.loc[t, best_llm]
        flash = piv.loc[t, second_llm]
        kalm = piv.loc[t, best_emb_overall]
        be = piv.loc[t, emb_models].max()
        delta = pro - be
        dstr = f"+{delta*100:.1f}" if delta >= 0 else f"$-${abs(delta)*100:.1f}"
        name = t[3:] if t.startswith("LLM") else t
        win_pro = delta > 0
        rows.append(
            f"{name} & {fmt3(pro, bold=win_pro)} & {fmt3(flash)} & {fmt3(kalm)} & "
            f"{fmt3(be, bold=not win_pro)} & {dstr} \\\\"
        )
    body = "\n".join(rows)
    return f"""% Retrieval task breakdown (main paper) (auto-generated)
\\begin{{table}}[t]
\\centering
\\small
\\resizebox{{\\linewidth}}{{!}}{{%
\\begin{{tabular}}{{lccccc}}
\\toprule
\\textbf{{Task}} & \\textbf{{{registry.display_name(best_llm)}}} & \\textbf{{{registry.display_name(second_llm)}}} & \\textbf{{{registry.display_name(best_emb_overall)}}} & \\textbf{{Best Emb.}} & \\textbf{{$\\Delta$}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
}}
\\caption{{\\textbf{{Retrieval task breakdown.}} Per-task scores for the best LLM, second LLM,
best embedding overall, and the best embedding on each task. $\\Delta$ = best LLM $-$ best embedding;
bold = task winner.}}
\\label{{tab:retrieval_tasks}}
\\end{{table}}
"""


def t_full_results_by_category(cs):
    cats = registry.CATEGORIES
    counts = {c: len(v) for c, v in registry.canonical_tasks_by_category().items()}
    ranked = cs.sort_values("Overall", ascending=False)
    colmax = {c: ranked[c].max() for c in cats + ["Overall"]}
    rows = []
    for i, (m, r) in enumerate(ranked.iterrows(), 1):
        typ = "LLM" if r.model_type == "llm" else "Emb"
        cells = " & ".join(fmt3(r[c], bold=np.isclose(r[c], colmax[c]), lead_dot=True) for c in cats)
        ov = fmt3(r["Overall"], bold=np.isclose(r["Overall"], colmax["Overall"]), lead_dot=True)
        rows.append(f"{i} & {registry.display_name(m)} & {typ} & {cells} & {ov} \\\\")
    body = "\n".join(rows)
    hdr = " & ".join(f"\\textbf{{{CAT_ABBR[c]} ({counts[c]})}}" for c in cats)
    return f"""% Full results: all complete models ranked by overall score (auto-generated)
\\begin{{table*}}[h]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{rlccccccc}}
\\toprule
\\textbf{{Rank}} & \\textbf{{Model}} & \\textbf{{Type}} & {hdr} & \\textbf{{Overall}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{\\textbf{{Complete per-category results}} for all {len(ranked)} complete models,
ranked by overall (macro) score. Bold = best in column.}}
\\label{{tab:full_results_by_category}}
\\end{{table*}}
"""


def t_per_task_scores(df, cs):
    piv, d = _pivot_task(df)
    tcat = registry.task_category_map(df)
    llm = cs[cs.model_type == "llm"].sort_values("Overall", ascending=False)
    emb = cs[cs.model_type == "embedding"].sort_values("Overall", ascending=False)
    best_nongemini = next((m for m in llm.index
                           if registry.LLM_META.get(m, {}).get("family") != "Gemini"), llm.index[1])
    cols = [llm.index[0], llm.index[1], best_nongemini, emb.index[0], emb.index[1]]
    emb_models = emb.index
    metric_label = {
        "Classification": "Accuracy", "STS": "Spearman $\\rho$", "Clustering": "V-measure",
        "PairClassification": "Avg. Precision", "Retrieval": "Recall@1",
    }
    lines = []
    for cat in registry.CATEGORIES:
        ctasks = sorted(t for t in piv.index if tcat[t] == cat)
        lines.append(f"\\multicolumn{{{len(cols)+3}}}{{l}}{{\\emph{{{cat} ({metric_label[cat]})}}}} \\\\")
        for t in ctasks:
            be = piv.loc[t, emb_models].max()
            vals = [piv.loc[t, m] for m in cols]
            rowmax = max(vals + [be])
            cellstr = " & ".join(fmt3(v, bold=np.isclose(v, rowmax), lead_dot=True) for v in vals)
            name = t[3:] if t.startswith("LLM") else t
            lines.append(f"{name} & {CAT_ABBR[cat]} & {cellstr} & {fmt3(be, bold=np.isclose(be, rowmax), lead_dot=True)} \\\\")
    body = "\n".join(lines)
    chead = " & ".join(f"\\textbf{{{registry.display_name(m)}}}" for m in cols)
    ncols = len(cols)
    return f"""% Per-task scores for representative models (auto-generated)
\\begin{{table*}}[h]
\\centering
\\scriptsize
\\setlength{{\\tabcolsep}}{{3pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{ll{'c'*ncols}r}}
\\toprule
\\textbf{{Task}} & \\textbf{{Cat.}} & {chead} & \\textbf{{Best Emb.}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
}}
\\caption{{\\textbf{{Per-task scores}} for representative models across all {len(piv.index)} \\mteblm{{}} tasks.
Bold = best in row (incl.\\ best embedding). Metric per category in italics.}}
\\label{{tab:per_task_scores}}
\\end{{table*}}
"""


def t_significance(df, cs):
    piv, d = _pivot_task(df)
    tcat = registry.task_category_map(df)
    llm = cs[cs.model_type == "llm"].sort_values("Overall", ascending=False)
    best_llm = llm.index[0]
    emb_models = cs[cs.model_type == "embedding"].index
    best_emb_overall = cs.loc[emb_models, "Overall"].idxmax()

    rng = np.random.default_rng(42)
    B = 10000

    def boot(tasks, other):
        pro = piv.loc[tasks, best_llm].values
        oth = piv.loc[tasks, other].values
        diff = pro - oth
        idx = rng.integers(0, len(diff), size=(B, len(diff)))
        means = diff[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        p = 2 * min((means <= 0).mean(), (means >= 0).mean())
        return diff.mean(), lo, hi, min(p, 1.0)

    def fmt_row(label, tasks, other):
        d0, lo, hi, p = boot(tasks, other)
        dstr = f"+{d0*100:.1f}" if d0 >= 0 else f"$-${abs(d0)*100:.1f}"
        pstr = "$<$0.01" if p < 0.01 else ("$<$0.05" if p < 0.05 else f"{p:.2f}")
        sig = "Yes" if p < 0.05 else "No"
        return f"\\quad {label} & {dstr} & {ci_str(lo, hi)} & {pstr} & {sig} \\\\"

    all_tasks = list(piv.index)
    over = fmt_row("All tasks", all_tasks, best_emb_overall)
    cat_rows = []
    for c in ["Retrieval", "Clustering", "STS", "PairClassification", "Classification"]:
        ctasks = [t for t in piv.index if tcat[t] == c]
        be = cs.loc[emb_models, c].idxmax()
        cat_rows.append(fmt_row(c.replace("PairClassification", "Pair Classification"), ctasks, be))
    cat_body = "\n".join(cat_rows)
    return f"""% Statistical significance (paired bootstrap) (auto-generated)
\\begin{{table}}[h]
\\centering
\\small
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Comparison}} & \\textbf{{$\\Delta$}} & \\textbf{{95\\% CI}} & \\textbf{{$p$}} & \\textbf{{Sig.}} \\\\
\\midrule
\\multicolumn{{5}}{{l}}{{\\emph{{Overall ({registry.display_name(best_llm)} vs.\\ {registry.display_name(best_emb_overall)}, \\mteblm)}}}} \\\\
{over}
\\midrule
\\multicolumn{{5}}{{l}}{{\\emph{{Per category ({registry.display_name(best_llm)} vs.\\ best embedding)}}}} \\\\
{cat_body}
\\bottomrule
\\end{{tabular}}
\\caption{{\\textbf{{Statistical significance.}} Paired bootstrap test (10{{,}}000 resamples, seed 42).
$\\Delta$ = {registry.display_name(best_llm)} $-$ best embedding; significance at $\\alpha$\\,=\\,0.05. Pair classification uses Pro for consistency, although Flash scores higher.}}
\\label{{tab:significance}}
\\end{{table}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Cost / token tables
# ─────────────────────────────────────────────────────────────────────────────
def t_models(cs, cost, params):
    llm = cs[cs.model_type == "llm"].sort_values("Overall", ascending=False)
    emb = cs[cs.model_type == "embedding"].sort_values("Overall", ascending=False).head(10)

    def cells(m, r):
        c = cost.loc[m, "total_cost"]
        return (f"{registry.display_name(m)} & {fmt_params(params.get(m))} & "
                f"{r['Overall']*100:.1f} & {cost_str(r.model_type, c)}")

    llm_list = [cells(m, r) for m, r in llm.iterrows()]
    emb_list = [cells(m, r) for m, r in emb.iterrows()]
    n = max(len(llm_list), len(emb_list))
    empty = " & & & "
    body = "\n".join(
        f"{llm_list[i] if i < len(llm_list) else empty} & "
        f"{emb_list[i] if i < len(emb_list) else empty} \\\\" for i in range(n))
    return f"""% Model overview (main paper, LLMs beside embeddings, auto-generated)
\\begin{{table}}[t]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lrrr|lrrr}}
\\toprule
\\multicolumn{{4}}{{c}}{{\\textbf{{LLMs}}}} & \\multicolumn{{4}}{{c}}{{\\textbf{{Embedding models (top 10)}}}} \\\\
\\cmidrule(lr){{1-4}} \\cmidrule(lr){{5-8}}
\\textbf{{Model}} & \\textbf{{Params}} & \\textbf{{Score}} & \\textbf{{Cost}} & \\textbf{{Model}} & \\textbf{{Params}} & \\textbf{{Score}} & \\textbf{{Cost}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}}}
\\caption{{\\textbf{{Model overview.}} Mean score and total \\mteblm{{}} cost for all LLMs and the ten highest-scoring embedding models. LLM costs use API token usage; embedding costs use H100 throughput at \\$2.49/hr.}}
\\label{{tab:models}}
\\end{{table}}
"""


def t_full_models(cs, cost, params):
    llm = cs[cs.model_type == "llm"].sort_values("Overall", ascending=False)
    emb = cs[cs.model_type == "embedding"].sort_values("Overall", ascending=False)
    ov_max = cs["Overall"].max()

    def cite(m):
        k = EMB_CITE.get(m)
        return f"\\citealt{{{k}}}" if k else "--"

    def row(m, r, with_cite):
        c = cost.loc[m, "total_cost"]
        sc = fmt3(r["Overall"], bold=np.isclose(r["Overall"], ov_max))
        base = f"{registry.display_name(m)} & {fmt_params(params.get(m))} & {sc} & {cost_str(r.model_type, c)}"
        return base + (f" & {cite(m)} \\\\" if with_cite else " & -- \\\\")

    llm_rows = "\n".join(row(m, r, False) for m, r in llm.iterrows())
    emb_rows = "\n".join(row(m, r, True) for m, r in emb.iterrows())
    return f"""% Appendix: full model listing with citations (auto-generated)
\\begin{{table}}[h]
\\centering
\\small
\\begin{{tabular}}{{lcrrl}}
\\toprule
\\textbf{{Model}} & \\textbf{{Params}} & \\textbf{{Score}} & \\textbf{{Cost}} & \\textbf{{Reference}} \\\\
\\midrule
\\multicolumn{{5}}{{l}}{{\\emph{{LLM Models}}}} \\\\
{llm_rows}
\\midrule
\\multicolumn{{5}}{{l}}{{\\emph{{Embedding Models (ranked by score)}}}} \\\\
{emb_rows}
\\bottomrule
\\end{{tabular}}
\\caption{{\\textbf{{Complete model listing}} with mean (macro) scores across {len(registry.canonical_tasks())} \\mteblm{{}}
tasks, total benchmark costs, and references. LLM costs reflect actual API usage; embedding costs
from H100 throughput benchmarking (\\$2.49/hr).}}
\\label{{tab:full_models}}
\\end{{table}}
"""


def t_llm_tokens(tok, cs):
    # One row per LLM (transposed vs the old 3-column layout, which cannot hold 10).
    order = cs[cs.model_type == "llm"].sort_values("Overall", ascending=False).index
    order = [m for m in order if m in tok.index]
    rows = []
    for m in order:
        r = tok.loc[m]
        nc_in = (r.input_tokens - r.cached_tokens) / 1e6
        ca = r.cached_tokens / 1e6
        out = r.output_tokens / 1e6
        th = r.thinking_tokens / 1e6
        rows.append(
            f"{registry.display_name(m)} & {nc_in:.1f} & {ca:.1f} & {out:.1f} & {th:.1f} & "
            f"\\${r.cost_input_usd:,.2f} & \\${r.cost_cached_usd:,.2f} & \\${r.cost_output_usd:,.2f} & "
            f"\\textbf{{\\${r.total_cost_usd:,.2f}}} \\\\"
        )
    body = "\n".join(rows)
    return f"""% Appendix: detailed LLM token usage and cost (auto-generated)
\\begin{{table}}[h]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{lrrrrrrrr}}
\\toprule
& \\multicolumn{{4}}{{c}}{{\\textbf{{Tokens (M)}}}} & \\multicolumn{{4}}{{c}}{{\\textbf{{Cost (USD)}}}} \\\\
\\cmidrule(lr){{2-5}} \\cmidrule(lr){{6-9}}
\\textbf{{Model}} & In$_{{nc}}$ & Cached & Out & Think & In$_{{nc}}$ & Cached & Out+Th & \\textbf{{Total}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{\\textbf{{Detailed LLM token usage and cost}} across all \\mteblm{{}} tasks.
In$_{{nc}}$ = non-cached input (cached billed at 10\\% of input rate); Think = reasoning tokens
(billed at output rate); Out+Th combines standard output and thinking cost.}}
\\label{{tab:llm_tokens}}
\\end{{table}}
"""


def t_embedding_throughput(thr, cs, params):
    d = thr[thr.status == "success"].copy()
    keep = set(registry.complete_models())
    d = d[d.model.isin(keep)]
    d = d.merge(cs[["Overall"]], left_on="model", right_index=True, how="left")
    d = d.sort_values("median_tok_per_sec", ascending=False)
    rows = []
    for _, r in d.iterrows():
        rows.append(
            f"{registry.display_name(r.model)} & {fmt_params(params.get(r.model))} & "
            f"{r.median_tok_per_sec:,.0f} & {r.cost_usd_per_mtok:.4f} & {r.Overall*100:.1f} \\\\"
        )
    body = "\n".join(rows)
    return f"""% Appendix: embedding throughput on H100 (auto-generated)
\\begin{{table}}[h]
\\centering
\\small
\\begin{{tabular}}{{lcrrr}}
\\toprule
\\textbf{{Model}} & \\textbf{{Params}} & \\textbf{{Tok/s}} & \\textbf{{\\$/MTok}} & \\textbf{{Score}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{\\textbf{{Embedding throughput}} on a single NVIDIA H100 80GB (median tokens/s over the
benchmark), per-MTok cost at \\$2.49/hr spot, and mean (macro) \\mteblm{{}} score.}}
\\label{{tab:embedding_throughput}}
\\end{{table}}
"""


def t_cost_sensitivity(cs, cost):
    emb_models = cs[cs.model_type == "embedding"].index
    best_emb = cs.loc[emb_models, "Overall"].idxmax()
    llm = cs[cs.model_type == "llm"].sort_values("Overall", ascending=False)
    best_llm = llm.index[0]
    emb_cost = cost.loc[best_emb, "total_cost"]     # at $2.49/hr
    llm_cost = cost.loc[best_llm, "total_cost"]
    scenarios = [
        ("H100 spot \\$2.49/hr (our setup)", 1.0),
        ("H100 on-demand \\$3.99/hr", 3.99 / 2.49),
        ("A100 spot \\$1.49/hr (est.\\ 1.5$\\times$ slower)", 1.49 / 2.49 * 1.5),
        ("L4 spot \\$0.49/hr (est.\\ 3$\\times$ slower)", 0.49 / 2.49 * 3.0),
    ]
    rows = []
    for label, mult in scenarios:
        ec = emb_cost * mult
        ratio = llm_cost / ec
        rows.append(f"{label} & \\${ec:.3f} & \\${llm_cost:,.2f} & {ratio:,.0f}$\\times$ \\\\")
    _pm = pd.read_csv(ROOT / "data" / "embedding_costs_per_model.csv")
    median_mtok = _pm[_pm.tokens > 0].tokens.median() / 1e6  # benchmark tokens, 37-task set
    api_ec = 0.10 * median_mtok  # $0.10/MTok x median benchmark token count
    rows.append(f"Commercial API \\$0.10/MTok & \\${api_ec:.3f} & \\${llm_cost:,.2f} & {llm_cost/api_ec:,.0f}$\\times$ \\\\")
    body = "\n".join(rows)
    ratios = [llm_cost / (emb_cost * m) for _, m in scenarios] + [llm_cost / api_ec]
    return f"""% Appendix: cost sensitivity analysis (auto-generated)
\\begin{{table}}[h]
\\centering
\\small
\\begin{{tabular}}{{lrrr}}
\\toprule
\\textbf{{Hardware / Pricing Scenario}} & \\textbf{{Emb.\\ Cost}} & \\textbf{{LLM Cost}} & \\textbf{{Ratio}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{\\textbf{{Cost sensitivity analysis.}} LLM-to-embedding cost ratio under alternative hardware
and pricing scenarios. Costs compare {registry.display_name(best_emb)} with {registry.display_name(best_llm)} at fixed API pricing; ratios range from {min(ratios):,.0f}--{max(ratios):,.0f}$\\times$.}}
\\label{{tab:cost_sensitivity}}
\\end{{table}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Ablation: reduced-thinking (new retrieval tasks)
# ─────────────────────────────────────────────────────────────────────────────
def _ablation_scores(variant):
    base = ROOT / "ablation_results" / "thinking" / variant
    out = {}
    if not base.exists():
        return out
    for mdir in sorted(base.iterdir()):
        if not mdir.is_dir():
            continue
        scores, think = {}, {}
        for jf in mdir.rglob("*.json"):
            if jf.stem == "model_meta":
                continue
            d = json.loads(jf.read_text())
            task = jf.stem[3:] if jf.stem.startswith("LLM") else jf.stem
            sd = d.get("scores", {}).get("test")
            if not sd:
                continue
            e = sd[0] if isinstance(sd, list) else sd
            scores[task] = e.get("main_score")
            think[task] = e.get("usage_stats", {}).get("thinking_tokens", 0)
        out[mdir.name] = {"score": scores, "think": think}
    return out


def _default_retrieval_thinking(model_key):
    """Thinking tokens per task in the DEFAULT run (from llm_results JSONs)."""
    from importlib import import_module
    sys.path.insert(0, str(ROOT / "scripts"))
    canonicalize = import_module("aggregate_scores").canonicalize
    mdir = ROOT / "llm_results" / model_key
    out = {}
    if not mdir.exists():
        return out
    seen = set()
    for jf in sorted(mdir.rglob("*.json")):
        if jf.stem in ("model_meta",) or jf.stem.endswith("_samples"):
            continue
        c = canonicalize(jf.stem)
        if c in seen:
            continue
        seen.add(c)
        try:
            d = json.loads(jf.read_text())
            sd = d.get("scores", {}).get("test")
            if not sd:
                continue
            e = sd[0] if isinstance(sd, list) else sd
            out[c] = e.get("usage_stats", {}).get("thinking_tokens", 0)
        except Exception:
            pass
    return out


def t_ablation_nocot(df, cs):
    """Reduced-thinking (reasoning_effort=low) vs default, on the new retrieval tasks."""
    low = _ablation_scores("low")
    if not low:
        raise RuntimeError("no ablation_results/thinking/low data")
    model_key = "google__gemini-3-flash-preview"
    if model_key not in low:
        model_key = sorted(low)[0]
    piv, d = _pivot_task(df)
    tcat = registry.task_category_map(df)
    default = _default_retrieval_thinking(model_key)
    emb_models = cs[cs.model_type == "embedding"].index
    lowd = low[model_key]

    def canon(t):
        for full in piv.index:
            if full[3:] == t or full.endswith(t):
                return full
        return None

    rows = []
    for t, low_score in sorted(lowd["score"].items()):
        full = canon(t)
        if full is None or tcat.get(full) != "Retrieval":
            continue
        base_score = piv.loc[full, model_key]
        delta = low_score - base_score
        dstr = f"+{delta*100:.1f}" if delta >= 0 else f"$-${abs(delta)*100:.1f}"
        be = piv.loc[full, emb_models].max()
        d_th = default.get(full)
        l_th = lowd["think"].get(t, 0)
        red = f"{100*(1-l_th/d_th):.0f}\\%" if d_th else "--"
        rows.append(f"{full[3:]} & {fmt3(base_score)} & {fmt3(low_score, bold=delta>0)} & {dstr} & {red} & {fmt3(be)} \\\\")
    body = "\n".join(rows)
    mname = registry.display_name(model_key)
    return f"""% Ablation: reduced thinking (reasoning_effort=low) (auto-generated)
\\begin{{table}}[t]
\\centering
\\small
\\resizebox{{\\linewidth}}{{!}}{{%
\\begin{{tabular}}{{lccccc}}
\\toprule
\\textbf{{Task}} & \\textbf{{{mname}}} & \\textbf{{{mname} (low)}} & \\textbf{{$\\Delta$}} & \\textbf{{Think $\\downarrow$}} & \\textbf{{Best Emb.}} \\\\
\\midrule
\\multicolumn{{6}}{{l}}{{\\emph{{Retrieval}}}} \\\\
{body}
\\bottomrule
\\end{{tabular}}
}}
\\caption{{\\textbf{{Reduced-thinking ablation}} ({mname} with \\texttt{{reasoning\\_effort=low}} vs.\\ default)
on the \\mteblm{{}} retrieval tasks. Think~$\\downarrow$ = reduction in thinking tokens vs.\\ default.
Reducing thinking by 54--94\\% improves all six retrieval scores in this ablation.}}
\\label{{tab:ablation_nocot}}
\\end{{table}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Dataset suite + per-task token budget (from data/token_counts_per_task.csv)
# ─────────────────────────────────────────────────────────────────────────────
# Per-task metadata keyed by the token-count CSV task name.
# lang, citation (bib key or None), n_classes ("--" for non-classification).
# FQuAD/HC3/PublicHealthQA bib keys were added to the .bib in this pass — verify.
TASK_META = {
    # Classification
    "ImdbClassification":       ("en", "maas2011learning", "2"),
    "Banking77Classification":  ("en", "casanueva2020banking77", "77"),
    "AmazonCounterfactualCls":  ("en, de, ja", "oneill2021counterfactual", "2"),
    "MTOPDomainCls":            ("en, de, fr", "li2021mtop", "11"),
    "MassiveIntentCls":         ("en, de, fr, ja", "fitzgerald2022massive", "60"),
    "MassiveScenarioCls":       ("en, de, fr, ja", "fitzgerald2022massive", "18"),
    "ToxicConversationsCls":    ("en", "borkan2019nuanced", "2"),
    "TweetSentimentCls":        ("en", "muennighoff2023mteb", "3"),
    # STS
    "STSBenchmark":  ("en", "cer2017semeval", "--"),
    "SICK-R":        ("en", "marelli2014sick", "--"),
    "STS12":         ("en", "agirre2012semeval", "--"),
    "STS13":         ("en", "agirre2012semeval", "--"),
    "STS14":         ("en", "agirre2012semeval", "--"),
    "STS15":         ("en", "agirre2016semeval", "--"),
    "STS16":         ("en", "agirre2016semeval", "--"),
    "BIOSSES":       ("en", "sougancioglu2017biosses", "--"),
    "STS17":         ("en, de, es, fr", "cer2017semeval", "--"),
    "STS22v2":       ("en, de, es, fr, ru, zh", "chen2022semeval", "--"),
    # Clustering
    "RedditClusteringP2P":     ("en", "muennighoff2023mteb", "--"),
    "TwentyNewsgroupsV2":      ("en", "muennighoff2023mteb", "--"),
    "StackExchangeClustP2PV2": ("en", "muennighoff2023mteb", "--"),
    "StackExchangeClustV2":    ("en", "muennighoff2023mteb", "--"),
    "ArxivClusteringP2P":      ("en", "muennighoff2023mteb", "--"),
    "ArxivClusteringS2S":      ("en", "muennighoff2023mteb", "--"),
    "BiorxivClusteringP2PV2":  ("en", "muennighoff2023mteb", "--"),
    "MedrxivClusteringP2PV2":  ("en", "muennighoff2023mteb", "--"),
    "MedrxivClusteringS2SV2":  ("en", "muennighoff2023mteb", "--"),
    # Pair Classification
    "SprintDuplicateQuestionsPC": ("en", "shah2018adversarial", "2"),
    "TwitterURLCorpusPC":         ("en", "lan2017continuously", "2"),
    "LegalBenchPC":               ("en", "guha2023legalbench", "2"),
    "RTE3PC":                     ("de, en, fr, it", "giampiccolo2007third", "2"),
    # Retrieval (current 6-task suite)
    "AILAStatutes":                  ("en", "enevoldsen2025mmtebmassivemultilingualtext", "--"),
    "FQuADRetrieval":                ("fr", "dhoffschmidt2020fquad", "--"),
    "HC3FinanceRetrieval":           ("en", "guo2023hc3", "--"),
    "LegalBenchConsumerContractsQA": ("en", "guha2023legalbench", "--"),
    "PublicHealthQA":                ("en", None, "--"),  # HF: xhluca/publichealth-qa
    "TwitterHjerneRetrieval":        ("da", "holm2024danoliterate", "--"),
}
CAT_METRIC = {"Classification": "Acc.", "STS": "Spearman", "Clustering": "V-meas.",
              "PairClassification": "Avg.~Prec.", "Retrieval": "Recall@1"}
CAT_FULL = {"Classification": "Classification", "STS": "Semantic Textual Similarity",
            "Clustering": "Clustering", "PairClassification": "Pair Classification",
            "Retrieval": "Retrieval"}


def _load_token_counts():
    p = DATA / "token_counts_per_task.csv"
    if not p.exists():
        raise RuntimeError("data/token_counts_per_task.csv missing — run count_tokens_per_task.py")
    return pd.read_csv(p)


def _cite(key):
    return f"\\citet{{{key}}}" if key else "HF dataset"


def _knum(n):
    n = int(n)
    return f"{n/1000:.0f}k" if n >= 1000 else str(n)


def t_datasets():
    tc = _load_token_counts()
    order = ["Classification", "STS", "Clustering", "PairClassification", "Retrieval"]
    lines = []
    for cat in order:
        sub = tc[tc.category == cat]
        lines.append(f"\\addlinespace[4pt]\n\\multicolumn{{6}}{{l}}{{\\textit{{{CAT_FULL[cat]} ({len(sub)} tasks)}}}} \\\\")
        lines.append("\\addlinespace[2pt]")
        for _, r in sub.iterrows():
            name = r["task"]
            lang, cite, ncls = TASK_META.get(name, ("en", None, "--"))
            if cat == "Retrieval":
                nstr = f"{_knum(r.test_n)}\\,Q / {_knum(r.train_n)}\\,C"
            else:
                nstr = _knum(r.test_n)
            disp = name.replace("Classification", "Cls").replace("Clustering", "Clust")
            lines.append(f"{disp} & {lang} & {nstr} & {ncls} & {CAT_METRIC[cat]} & {_cite(cite)} \\\\")
    body = "\n".join(lines)
    ntasks = len(tc)
    return f"""% Dataset overview (auto-generated from token_counts_per_task.csv + TASK_META)
\\begin{{table}}[h]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{3pt}}
\\resizebox{{\\linewidth}}{{!}}{{%
\\begin{{tabular}}{{llrcll}}
\\toprule
\\textbf{{Task}} & \\textbf{{Lang.}} & \\textbf{{N}} & \\textbf{{Cls.}} & \\textbf{{Metric}} & \\textbf{{Source}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
}}
\\caption{{\\textbf{{\\mteblm{{}} task suite ({ntasks} tasks).}}
N = held-out test samples (summed over languages for multilingual tasks);
Q = queries, C = corpus documents.
Multilingual tasks are evaluated per language and averaged. Held-out subsets
(seed 42) derived from MTEB and MMTEB tasks \\citep{{muennighoff2023mteb,enevoldsen2025mmtebmassivemultilingualtext}},
hosted at \\texttt{{mteb/llm-eval-*}}. Token counts: Table~\\ref{{tab:token_budget}}.}}
\\label{{tab:datasets}}
\\end{{table}}
"""


def t_token_budget():
    tc = _load_token_counts()

    def kk(n):
        n = int(round(n / 1000))
        return f"{n:,}k"

    lines = []
    # Classification: test + train (kNN reference corpus)
    lines.append("\\multicolumn{3}{l}{\\textit{Classification}} \\\\\n\\addlinespace[2pt]")
    lines.append("\\textbf{Task} & \\textbf{Test} & \\textbf{Train (kNN)\\textsuperscript{*}} \\\\\n\\midrule")
    for _, r in tc[tc.category == "Classification"].iterrows():
        disp = r["task"].replace("Classification", "Cls")
        lines.append(f"{disp} & {kk(r.test_tokens)} & {kk(r.train_tokens)} \\\\")
    # Non-retrieval single-column groups
    for cat in ["STS", "Clustering", "PairClassification"]:
        lines.append(f"\\addlinespace[4pt]\n\\multicolumn{{3}}{{l}}{{\\textit{{{cat}}}}} \\\\\n\\addlinespace[2pt]")
        for _, r in tc[tc.category == cat].iterrows():
            disp = r["task"].replace("Clustering", "Clust")
            lines.append(f"{disp} & \\multicolumn{{2}}{{r}}{{{kk(r.total_tokens)}}} \\\\")
    # Retrieval: queries + corpus
    lines.append("\\midrule\n\\textbf{Task} & \\textbf{Queries} & \\textbf{Corpus} \\\\\n\\midrule")
    lines.append("\\multicolumn{3}{l}{\\textit{Retrieval}} \\\\\n\\addlinespace[2pt]")
    for _, r in tc[tc.category == "Retrieval"].iterrows():
        lines.append(f"{r['task']} & {kk(r.query_tokens)} & {kk(r.corpus_tokens)} \\\\")
    body = "\n".join(lines)
    return f"""% Per-task token budget (GPT-4o tokenizer, raw text) (auto-generated)
\\begin{{table}}[h]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{5pt}}
\\begin{{tabular}}{{lrr}}
\\toprule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{\\textbf{{Token budget per task}} (GPT-4o tokenizer; raw text).
Actual counts vary by ${{\\pm}}$15--40\\% across model vocabularies.
\\textsuperscript{{*}}Train split is the kNN reference corpus processed only by
embedding models; LLMs process only the test split. Corpus-in-context formatting
adds ${{\\sim}}$20 tokens per document for LLM retrieval.}}
\\label{{tab:token_budget}}
\\end{{table}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Reranker matrix (BEIR / BRIGHT) — retrieve-then-rerank baseline (reviewer ask)
# ─────────────────────────────────────────────────────────────────────────────
def t_reranker_matrix():
    # Curated, well-covered columns: pure first stage, two strong CE rerankers,
    # a bi-encoder CE, and two LLM listwise rerankers.
    cols = ["pure", "bge-rerank-gemma", "Qwen3-RR-4B", "Qwen3-RR-8B",
            "llm-qwen3.6-27b", "llm-qwen3.6-35b-a3b"]
    col_hdr = ["Pure", "BGE-Gemma", "Qwen3-RR-4B", "Qwen3-RR-8B",
               "Qwen3.6-27B$^\\dagger$", "Qwen3.6-35B$^\\dagger$"]

    def panel(tasks):
        lines = []
        for fs_name, fs_slug in rr.FIRST_STAGES.items():
            vals = [rr._avg(fs_slug, rr.RERANKERS[c], tasks) for c in cols]
            present = [v for v in vals if v is not None]
            mx = max(present) if present else None
            cells = " & ".join(
                ("--" if v is None else (f"\\textbf{{{v:.1f}}}" if v == mx else f"{v:.1f}"))
                for v in vals)
            lines.append(f"{fs_name} & {cells} \\\\")
        return "\n".join(lines)

    bright = panel(rr.BRIGHT)
    beir = panel(rr.BEIR)
    chead = " & ".join(f"\\textbf{{{c}}}" for c in col_hdr)
    ncol = len(cols)
    return f"""% Retrieve-then-rerank matrix on BEIR + BRIGHT (auto-generated)
\\begin{{table*}}[t]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{l{'c'*ncol}}}
\\toprule
\\textbf{{First stage}} & {chead} \\\\
\\midrule
\\multicolumn{{{ncol+1}}}{{l}}{{\\emph{{BRIGHT (reasoning)}}}} \\\\
{bright}
\\midrule
\\multicolumn{{{ncol+1}}}{{l}}{{\\emph{{BEIR (semantic)}}}} \\\\
{beir}
\\bottomrule
\\end{{tabular}}
}}
\\caption{{\\textbf{{Retrieve-then-rerank matrix.}} Average nDCG@10 for first-stage retrievers
crossed with cross-encoder and LLM listwise ($\\dagger$) rerankers over 7 BRIGHT and 5 BEIR tasks. Bold = best per row.
An LLM reranker improves Qwen3-E-8B on BRIGHT ({rr._avg(rr.FIRST_STAGES['Qwen3-E-8B'], None, rr.BRIGHT):.1f}$\\to${rr._avg(rr.FIRST_STAGES['Qwen3-E-8B'], RR_27B, rr.BRIGHT):.1f}); on BEIR, the embedding alone scores highest ({rr._avg(rr.FIRST_STAGES['Qwen3-E-8B'], None, rr.BEIR):.1f}).}}
\\label{{tab:reranker_matrix}}
\\end{{table*}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Cross-family thinking-token tax (default vs reasoning-off) on retrieval
# ─────────────────────────────────────────────────────────────────────────────
_RETR_STEMS = ["LLMAILAStatutes", "LLMFQuADRetrieval", "LLMHC3FinanceRetrieval",
               "LLMLegalBenchConsumerContractsQA", "LLMPublicHealthQA",
               "LLMTwitterHjerneRetrieval"]


def _sum_gen_tokens(model_root: Path, stems) -> float:
    """Sum output+thinking tokens over the given retrieval task files (deduped)."""
    seen, total = set(), 0.0
    for jf in sorted(Path(model_root).rglob("*.json")):
        if jf.stem in ("model_meta",) or jf.stem.endswith("_samples") or jf.stem in seen:
            continue
        if jf.stem not in stems:
            continue
        seen.add(jf.stem)
        try:
            d = json.loads(jf.read_text())
            sd = d.get("scores", {}).get("test")
            if not sd:
                continue
            e = sd[0] if isinstance(sd, list) else sd
            u = e.get("usage_stats", {})
            total += (u.get("output_tokens", 0) or 0) + (u.get("thinking_tokens", 0) or 0)
        except Exception:
            pass
    return total


def t_ablation_thinking_families(cs):
    off = _ablation_scores("off")
    if not off:
        raise RuntimeError("no ablation_results/thinking/off data")
    rows = []
    families, n_better = set(), 0
    # complete LLMs that have an off-ablation, ordered by default retrieval score
    models = [m for m in cs[cs.model_type == "llm"].index if m in off]
    models.sort(key=lambda m: -cs.loc[m, "Retrieval"])
    for m in models:
        default = cs.loc[m, "Retrieval"]
        off_scores = [v for v in off[m]["score"].values() if v is not None]
        if not off_scores:
            continue
        off_avg = sum(off_scores) / len(off_scores)
        delta = off_avg - default
        dstr = f"+{delta*100:.1f}" if delta >= 0 else f"$-${abs(delta)*100:.1f}"
        gen_def = _sum_gen_tokens(ROOT / "llm_results" / m, _RETR_STEMS)
        gen_off = _sum_gen_tokens(ROOT / "ablation_results" / "thinking" / "off" / m, _RETR_STEMS)
        red = f"$-${100*(1-gen_off/gen_def):.0f}\\%" if gen_def else "--"
        better = off_avg >= default
        if better:
            n_better += 1
        families.add(registry.LLM_META.get(m, {}).get("family"))
        rows.append(f"{registry.display_name(m)} & {fmt3(default)} & "
                    f"{fmt3(off_avg, bold=better)} & {dstr} & {red} \\\\")
    body = "\n".join(rows)
    n_models = len(rows)
    n_families = len(families)
    return f"""% Cross-family thinking-token tax: default vs reasoning-off on retrieval (auto-generated)
\\begin{{table}}[t]
\\centering
\\small
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Model}} & \\textbf{{Default}} & \\textbf{{No-think}} & \\textbf{{$\\Delta$}} & \\textbf{{Gen tokens}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{\\textbf{{Cross-family reduced-reasoning ablation.}} Mean retrieval score with reasoning enabled vs.\\ disabled for {n_models} LLMs from {n_families} families. Gen tokens = reduction in output$+$thinking tokens.}}
\\label{{tab:ablation_thinking_families}}
\\end{{table}}
"""


# ─────────────────────────────────────────────────────────────────────────────
RR_27B = "llm-qwen3.6-27b"


def main():
    df, cs, cost, tok, thr, params = load()
    print("Regenerating paper tables from scores.csv + cost/token data ...\n")

    jobs = {
        "main_results": lambda: t_main_results(cs, params, cost),
        "category_scores": lambda: t_category_scores(cs),
        "retrieval_tasks": lambda: t_retrieval_tasks(df, cs),
        "per_task_scores": lambda: t_per_task_scores(df, cs),
        "full_results_by_category": lambda: t_full_results_by_category(cs),
        "significance": lambda: t_significance(df, cs),
        "models": lambda: t_models(cs, cost, params),
        "full_models": lambda: t_full_models(cs, cost, params),
        "llm_tokens": lambda: t_llm_tokens(tok, cs),
        "embedding_throughput": lambda: t_embedding_throughput(thr, cs, params),
        "cost_sensitivity": lambda: t_cost_sensitivity(cs, cost),
        "ablation_nocot": lambda: t_ablation_nocot(df, cs),
        "reranker_matrix": lambda: t_reranker_matrix(),
        "ablation_thinking_families": lambda: t_ablation_thinking_families(cs),
        "datasets": lambda: t_datasets(),
        "token_budget": lambda: t_token_budget(),
    }
    ok, failed = [], []
    for name, fn in jobs.items():
        try:
            write(name, fn())
            ok.append(name)
        except Exception as exc:
            failed.append((name, str(exc)))
            print(f"  [FAIL] {name}: {exc}")

    print(f"\nGenerated {len(ok)}/{len(jobs)} tables.")
    if failed:
        print("Failed:", ", ".join(n for n, _ in failed))
    print("\nBLOCKED (missing data, not generated): "
          "ablation_fewshot (no few-shot runs). "
          "full_per_task_all_models via scripts/gen_full_scores_table.py.")


if __name__ == "__main__":
    main()
