"""Reconcile embedding token counts vs LLM input tokens to validate both.

For each task: LLM_input = raw_text + prompt_overhead
If overhead/request is reasonable (100-3000 tokens), both counts are correct.
"""
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Load embedding token counts
emb = pd.read_csv(ROOT / "embedding_token_counts.csv")
emb = emb[emb.task != "TOTAL"].set_index("task")

# Extract per-task LLM tokens from Flash Lite (no thinking = cleanest)
model = "google__gemini-3.1-flash-lite-preview"
llm_dir = ROOT / "llm_results" / model

# Get canonical task set
scores_raw = pd.read_csv(ROOT / "scores.csv")
canonical = set(scores_raw[scores_raw.model_type == "embedding"]["task"].unique())

import sys
sys.path.insert(0, str(ROOT))
from aggregate_scores import canonicalize

llm_per_task = {}
for jf in sorted(llm_dir.rglob("*.json")):
    if "model_meta" in jf.stem or "_samples" in jf.stem:
        continue
    canon = canonicalize(jf.stem)
    if canon not in canonical:
        continue
    d = json.loads(jf.read_text())
    for _split, sd in d.get("scores", {}).items():
        if not sd:
            continue
        entries = sd if isinstance(sd, list) else [sd]
        u0 = entries[0].get("usage_stats", {})
        # Detect per-language entries (different usage) vs duplicated
        if len(entries) > 1:
            u1 = entries[1].get("usage_stats", {})
            same = (u0.get("input_tokens") == u1.get("input_tokens"))
        else:
            same = True

        if same:
            to_sum = [entries[0]]
        else:
            to_sum = entries  # per-language: sum all

        for entry in to_sum:
            u = entry.get("usage_stats", {})
            inp = u.get("input_tokens", 0)
            out = u.get("output_tokens", 0)
            if canon in llm_per_task:
                llm_per_task[canon]["input"] += inp
                llm_per_task[canon]["output"] += out
            else:
                llm_per_task[canon] = {"input": inp, "output": out}

# Print reconciliation
hdr = "{:<50s} {:<6s} {:>10s} {:>7s} {:>12s} {:>12s} {:>10s}"
row_fmt = "{:<50s} {:<6s} {:>10,d} {:>7,d} {:>12,d} {:>12,d} {:>10.0f}{}"
print(hdr.format("Task", "Type", "EmbTok", "#Texts", "LLMInput", "Overhead", "OH/Req"))
print("-" * 117)

total_emb = 0
total_llm = 0
issues = []

for task in sorted(emb.index):
    e = emb.loc[task]
    cat = str(e["category"])[:5]
    etok = int(e["tokens"])
    ntexts = int(e["n_texts"])

    llm = llm_per_task.get(task, {})
    llm_inp = llm.get("input", 0)

    overhead = llm_inp - etok

    # Effective requests: STS/PairCls process pairs, so n_requests = n_texts/2
    if cat in ("PairC", "STS"):
        n_req = max(ntexts // 2, 1)
    elif cat == "Clust":
        # Clustering: LLM processes all docs in one call per clustering row
        # The number of rows is small (5 rows), not n_texts
        n_req = max(ntexts // 200, 1)  # ~200 docs per row
    elif cat == "Retri":
        # Retrieval: each query is one request (corpus is sent as context)
        # But for embedding, n_texts = queries + corpus
        n_req = max(1, ntexts // 2)  # rough
    else:
        n_req = max(ntexts, 1)

    oh_per_req = overhead / n_req

    flag = ""
    if llm_inp == 0:
        flag = " [NO LLM]"
    elif oh_per_req < 0:
        flag = " [NEGATIVE!]"
    elif cat == "Class" and oh_per_req > 5000:
        flag = " [HIGH?]"

    total_emb += etok
    total_llm += llm_inp

    print(row_fmt.format(task, cat, etok, ntexts, llm_inp, overhead, oh_per_req, flag))
    if flag:
        issues.append((task, flag, oh_per_req))

print("-" * 117)
print()
print("TOTALS:")
print(f"  Embedding raw text tokens:  {total_emb:>12,d}  ({total_emb/1e6:.1f}M)")
print(f"  LLM input tokens (FlashL):  {total_llm:>12,d}  ({total_llm/1e6:.1f}M)")
print(f"  Total overhead:             {total_llm - total_emb:>12,d}  ({(total_llm-total_emb)/1e6:.1f}M)")
print(f"  Overhead multiplier:        {total_llm / total_emb:.1f}x")
print()

# Expected overhead breakdown
n_cls = sum(1 for t in emb.index if emb.loc[t, "category"] == "Classification")
n_sts = sum(1 for t in emb.index if emb.loc[t, "category"] == "STS")
n_clust = sum(1 for t in emb.index if emb.loc[t, "category"] == "Clustering")
n_pair = sum(1 for t in emb.index if emb.loc[t, "category"] == "PairClassification")
n_ret = sum(1 for t in emb.index if emb.loc[t, "category"] == "Retrieval")
print(f"Task counts: {n_cls} cls, {n_sts} sts, {n_clust} clust, {n_pair} pair, {n_ret} ret = {n_cls + n_sts + n_clust + n_pair + n_ret} total")

if issues:
    print(f"\nFlagged tasks ({len(issues)}):")
    for t, f, oh in issues:
        print(f"  {t}: {f} (OH/req={oh:.0f})")
else:
    print("\nNo issues found - all overhead values look reasonable!")
