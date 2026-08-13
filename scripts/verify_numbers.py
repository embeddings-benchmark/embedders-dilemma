#!/usr/bin/env python3
"""Single source of truth for every headline number in the paper.

Run this to re-verify all figures/tables/prose numbers against the current data.
Everything is computed from data/scores.csv + the cost/token CSVs via the same
registry the figures/tables use, so the paper stays internally consistent.

Usage:  python scripts/verify_numbers.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "plotting"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
import registry
import aggregate_rerank_matrix as rr


def main():
    df = registry.load_scores()
    cs = registry.category_scores(df)
    cats = registry.CATEGORIES
    counts = {c: len(v) for c, v in registry.canonical_tasks_by_category().items()}
    cost = pd.read_csv(ROOT / "data" / "cost_summary.csv").set_index("model")

    llms = registry.complete_models(df, "llm")
    embs = registry.complete_models(df, "embedding")
    fams = sorted(set(registry.LLM_META[m]["family"] for m in llms))
    llm = cs[cs.model_type == "llm"].sort_values("Overall", ascending=False)
    emb = cs[cs.model_type == "embedding"].sort_values("Overall", ascending=False)
    best_llm, best_emb = llm.index[0], emb.index[0]

    print("## COUNTS")
    print(f"LLMs={len(llms)}  families={len(fams)} {fams}")
    print(f"embeddings={len(embs)}  total={len(llms)+len(embs)}")
    print(f"tasks={sum(counts.values())} {counts}")

    print("\n## OVERALL (macro %)")
    print(f"best LLM {registry.display_name(best_llm)}={cs.loc[best_llm,'Overall']*100:.1f}  "
          f"best emb {registry.display_name(best_emb)}={cs.loc[best_emb,'Overall']*100:.1f}")
    flash = llm.iloc[1]
    print(f"2nd LLM {registry.display_name(llm.index[1])}={flash.Overall*100:.1f} "
          f"({'ABOVE' if flash.Overall>cs.loc[best_emb,'Overall'] else 'BELOW'} best emb)")
    lite = [m for m in llm.index if "lite" in m][0]
    print(f"Flash-Lite={cs.loc[lite,'Overall']*100:.1f}, below {(emb['Overall']>cs.loc[lite,'Overall']).sum()}/{len(emb)} embeddings")

    # bootstrap
    keep = set(registry.complete_models(df))
    canon = set(registry.canonical_tasks(df))
    piv = df[df.model.isin(keep) & df.task.isin(canon)].pivot_table(index="task", columns="model", values="score")
    tcat = registry.task_category_map(df)
    rng = np.random.default_rng(42)

    def boot(tasks, a, b):
        d = piv.loc[tasks, a].values - piv.loc[tasks, b].values
        idx = rng.integers(0, len(d), size=(10000, len(d)))
        m = d[idx].mean(axis=1)
        return d.mean() * 100, *np.percentile(m, [2.5, 97.5]) * 100, 2 * min((m <= 0).mean(), (m >= 0).mean())

    print("\n## SIGNIFICANCE (best LLM vs best emb, bootstrap seed 42)")
    d0, lo, hi, p = boot(list(piv.index), best_llm, best_emb)
    print(f"Overall  d={d0:+.1f} CI[{lo:+.1f},{hi:+.1f}] p={p:.2f}")
    for c in cats:
        ct = [t for t in piv.index if tcat[t] == c]
        be = cs[cs.model_type == "embedding"][c].idxmax()
        d0, lo, hi, p = boot(ct, best_llm, be)
        print(f"  {c:18s}({counts[c]}) Pro={cs.loc[best_llm,c]*100:.1f} vs {registry.display_name(be):12s}={cs.loc[be,c]*100:.1f}  "
              f"d={d0:+.1f} p={p:.3f} {'SIG' if p < 0.05 else 'ns'}")

    print("\n## COST")
    pc, bc = cost.loc[best_llm, "total_cost"], cost.loc[best_emb, "total_cost"]
    cheap = cost[cost.type == "LLM"].total_cost.min()
    print(f"best LLM=${pc:.2f}  best emb=${bc:.3f}  ratio={pc/bc:.0f}x  cheapest LLM ${cheap:.2f}={cheap/bc:.0f}x  range {cheap/bc:.0f}-{pc/bc:.0f}x")

    print("\n## THINKING (% of total inference cost)")
    tok = pd.read_csv(ROOT / "data" / "llm_token_usage.csv")
    tok["tp"] = 100 * tok.thinking_tokens * tok.price_output_per_mtok / 1e6 / tok.total_cost_usd
    r = tok[tok.tp >= 1]
    print(f"reasoning models: {r.tp.min():.0f}-{r.tp.max():.0f}%  (Flash-Lite=0%)")

    print("\n## THROUGHPUT (1xH100 tok/s)")
    thr = pd.read_csv(ROOT / "data" / "embedding_throughput.csv")
    thr = thr[thr.status == "success"].copy()
    thr["model"] = thr.model_id.str.replace("/", "__", regex=False)
    thr = thr[thr.model.isin(keep)]
    lo_e, hi_e = thr.median_tok_per_sec.min(), thr.median_tok_per_sec.max()
    llm_thr = pd.read_csv(ROOT / "data" / "llm_throughput_h100.csv")
    lo_l, hi_l = llm_thr.tok_per_sec.min(), llm_thr.tok_per_sec.max()
    print(f"embeddings {lo_e:,.0f}-{hi_e:,.0f} ; open LLMs {lo_l:,.0f}-{hi_l:,.0f} "
          f"({len(llm_thr)} models) ; slowest emb={lo_e/hi_l:.1f}x fastest emb={hi_e/hi_l:.0f}x the fastest LLM")

    print("\n## RERANKER (avg nDCG@10)")
    for bench, tasks in [("BRIGHT", rr.BRIGHT), ("BEIR", rr.BEIR)]:
        q = rr.FIRST_STAGES["Qwen3-E-8B"]
        print(f"{bench}: Qwen3-E-8B pure={rr._avg(q,None,tasks):.1f} "
              f"+CE-4B={rr._avg(q,'Qwen__Qwen3-Reranker-4B',tasks):.1f} "
              f"+LLM-listwise={rr._avg(q,'llm-qwen3.6-27b',tasks):.1f}")

    print("\n## RERANKER COST (LLM listwise, Qwen3-E-8B first stage, from usage tokens x pricing)")
    import glob
    usage = ROOT / "pipeline_results" / "usage"
    agg = {}  # (model, bench) -> [in, out+think]
    for f in glob.glob(str(usage / "*Qwen3-Embedding-8B*" / "*.json")):
        d = json.loads(Path(f).read_text())
        m = d["model"]
        if m not in registry.LLM_PRICING:
            continue
        bench = "BRIGHT" if d["task"].startswith("BRIGHT") else "BEIR"
        a = agg.setdefault((m, bench), [0.0, 0.0])
        a[0] += d["input_tokens"]; a[1] += d["output_tokens"] + d["thinking_tokens"]
    for (m, bench), (ti, to) in sorted(agg.items()):
        p = registry.LLM_PRICING[m]
        cost = (ti * p["input"] + to * p["output"]) / 1e6
        print(f"  {m:16s} {bench:6s} = ${cost:5.1f}")


if __name__ == "__main__":
    main()
