"""Estimate listwise-reranker INPUT tokens per task, at different max_doc_chars.

No LLM calls — just rebuilds the exact listwise prompt (instructions + query +
top-100 candidate docs truncated to max_doc_chars) and tokenizes it with a Qwen
tokenizer. Reports avg input tokens/query and total, for max_doc_chars=800
(current) vs no truncation (full docs).
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets import load_dataset
from transformers import AutoTokenizer

LISTWISE_INSTRUCTIONS = (
    "You are an expert at evaluating document relevance. Given a query and a list of "
    "candidate documents, rank them by relevance to the query. Output ONLY a JSON list "
    "of integer document IDs in order of decreasing relevance. Example output format: "
    "[12, 3, 47, 1, ...]. Include ALL document IDs from the input."
)

BRIGHT_SUBSET = {
    "BRIGHTBiology": "biology", "BRIGHTEarthScience": "earth_science",
    "BRIGHTEconomics": "economics", "BRIGHTPsychology": "psychology",
    "BRIGHTRobotics": "robotics", "BRIGHTStackoverflow": "stackoverflow",
    "BRIGHTSustainableLiving": "sustainable_living",
}

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-8B", trust_remote_code=True)


def build_user_msg(qtext, cand_texts, max_doc_chars):
    blocks = []
    for i, txt in enumerate(cand_texts):
        t = txt if max_doc_chars is None else txt[:max_doc_chars]
        if max_doc_chars is not None and len(txt) > max_doc_chars:
            t += "..."
        blocks.append(f"[{i}] {t}")
    docs_str = "\n\n".join(blocks)
    return (f"Query: {qtext}\n\nCandidate documents ({len(cand_texts)}):\n{docs_str}\n\n"
            "Output the JSON list of document IDs ranked by decreasing relevance:")


def n_tokens(msg):
    return len(tok(LISTWISE_INSTRUCTIONS, add_special_tokens=False)["input_ids"]) + \
           len(tok(msg, add_special_tokens=False)["input_ids"])


def run(first_stage_slug, task_name, top_k=100, max_queries=100):
    sub = BRIGHT_SUBSET[task_name]
    docs = load_dataset("xlangai/BRIGHT", "documents", split=sub)
    examples = load_dataset("xlangai/BRIGHT", "examples", split=sub)
    doc_text = {r["id"]: r["content"] for r in docs}
    queries = {str(e["id"]): e["query"] for e in examples}
    qids = list(queries.keys())[:max_queries]

    pred_files = glob.glob(f"pipeline_results/predictions/{first_stage_slug}/{task_name}_predictions.json")
    if not pred_files:
        return None
    preds = json.loads(Path(pred_files[0]).read_text())["default"]["test"]

    tot = {800: 0, None: 0}
    nq = 0
    for qid in qids:
        if qid not in preds:
            continue
        ranked = sorted(preds[qid].items(), key=lambda kv: -kv[1])[:top_k]
        cand = [doc_text[d] for d, _ in ranked if d in doc_text]
        if not cand:
            continue
        for mdc in (800, None):
            tot[mdc] += n_tokens(build_user_msg(queries[qid], cand, mdc))
        nq += 1
    if nq == 0:
        return None
    return nq, tot[800] / nq, tot[None] / nq, tot[800], tot[None]


def main():
    fs = sys.argv[1] if len(sys.argv) > 1 else "Qwen__Qwen3-Embedding-8B"
    print(f"First stage: {fs}  (input tokens/query, top-100)\n")
    print(f'{"task":24s} {"in@800":>9s} {"in@full":>9s} {"x":>5s}')
    print("-" * 52)
    s800 = sfull = 0
    for task in BRIGHT_SUBSET:
        r = run(fs, task)
        if r is None:
            print(f"{task:24s}  (no preds)")
            continue
        nq, a800, afull, t800, tfull = r
        print(f'{task:24s} {a800:9,.0f} {afull:9,.0f} {afull/a800:4.1f}x')
        s800 += t800
        sfull += tfull
    print("-" * 52)
    print(f'{"TOTAL (input tok)":24s} {s800:9,.0f} {sfull:9,.0f} {sfull/max(s800,1):4.1f}x')


if __name__ == "__main__":
    main()
