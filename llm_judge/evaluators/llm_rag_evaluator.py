import asyncio
import logging
import random
import re
import string
import os
import json
from typing import Any

from llm_judge.llm_client import send_request_multi

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You will be given a list of documents. You need to read carefully and understand all of them. "
    "Then you will be given a query, and your goal is to answer the query based on the documents you have read.\n\n"
    "Your final answer should be in a list, in the following format:\n"
    "Final Answer: ['answer1', 'answer2', ...]\n"
    "If there is only one answer, it should be in the format:\n"
    "Final Answer: ['answer']"
)

def _format_doc(doc_id: str, title: str, text: str, seq_id: int) -> str:
    title_part = f" | TITLE: {title}" if title.strip() else ""
    return f"ID: {seq_id}{title_part} | CONTENT: {text} | END ID: {seq_id}"

def _parse_answer(response: str) -> str:
    import ast
    
    # 1. Look for the list format: Final Answer: ['answer1', 'answer2']
    match = re.search(r"Final Answer:\s*(\[.*?\])", response, re.IGNORECASE | re.DOTALL)
    if match:
        try:
            # Safely evaluate the string as a Python list
            parsed_list = ast.literal_eval(match.group(1))
            if isinstance(parsed_list, list) and len(parsed_list) > 0:
                # We return it joined by commas to match the existing split(',') downstream multi-answer logic
                return ",".join(str(x) for x in parsed_list)
        except Exception:
            pass
            
    match = re.search(r"Final Answer:\s*(.+)", response, re.IGNORECASE)
    if not match:
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        ans = lines[-1].replace("FINAL ANSWER:", "").replace("Final Answer:", "").strip() if lines else response.strip()
    else:
        ans = match.group(1).strip()

    ans = re.sub(r"^\[|\]$", "", ans)
    ans = re.sub(r"^['\"]|['\"]$", "", ans)
    return ans


def _parse_answer_from_rlm(response: str) -> str:
    """Parse RLM output into an answer string.

    RLM returns raw FINAL_VAR output — could be a Python string, a list repr,
    or prose.  We normalise it into the comma-separated string format that
    ``_compute_metrics`` expects.
    """
    import ast
    import json

    if not response or not response.strip():
        return ""

    text = response.strip()

    # 1. Try JSON parse (handles '["Paris"]' or '"Paris"')
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return ",".join(str(x) for x in parsed)
        if isinstance(parsed, str):
            return parsed
    except Exception:
        pass

    # 2. Try ast.literal_eval (handles "['Paris', 'London']")
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return ",".join(str(x) for x in parsed)
        if isinstance(parsed, str):
            return parsed
    except Exception:
        pass

    # 3. Check for "Final Answer:" in case the RLM formatted it that way
    fa = _parse_answer(text)
    if fa:
        return fa

    # 4. Return the raw text as the answer
    return text


def normalize_answer(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text: str) -> str:
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s: str) -> list[str]:
    if not s:
        return []
    return normalize_answer(s).split()

def compute_em(gold_answers: list[str], pred_answer: str) -> float:
    return max([float(normalize_answer(ga) == normalize_answer(pred_answer)) for ga in gold_answers])

def compute_subspan_em(gold_answers: list[str], pred_answer: str) -> float:
    norm_pred = normalize_answer(pred_answer)
    return max([1.0 if normalize_answer(ga) in norm_pred else 0.0 for ga in gold_answers])

def compute_f1(gold_answers: list[str], pred_answer: str) -> float:
    import collections
    pred_toks = get_tokens(pred_answer)

    f1_scores = []
    for ga in gold_answers:
        gold_toks = get_tokens(ga)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())

        if num_same == 0:
            f1_scores.append(0.0)
            continue

        if not gold_toks or not pred_toks:
            f1 = float(gold_toks == pred_toks)
        else:
            precision = 1.0 * num_same / len(pred_toks)
            recall = 1.0 * num_same / len(gold_toks)
            f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)

    return max(f1_scores) if f1_scores else 0.0

# --- MULTI VALUE METRICS ---
def compute_em_multi_value(gold_answers: list[str], pred_answers: list[str]) -> float:
    norm_gold = set(normalize_answer(g) for g in gold_answers)
    norm_pred = set(normalize_answer(p) for p in pred_answers)
    return float(norm_gold == norm_pred)

def compute_coverage(gold_answers: list[str], pred_answers: list[str]) -> float:
    norm_gold = set(normalize_answer(g) for g in gold_answers)
    norm_pred = set(normalize_answer(p) for p in pred_answers)
    if not norm_gold:
        return 0.0
    return len(norm_pred.intersection(norm_gold)) / float(len(norm_gold))

def compute_multi_value_subspan_em(gold_answers: list[str], pred_answers: list[str]) -> float:
    import numpy as np
    try:
        import scipy.optimize
    except ImportError:
        logger.warning("scipy is required for multi-answer subspan EM.")
        return 0.0
    
    scores = np.zeros([len(gold_answers), len(pred_answers)])
    for gold_index, gold_item in enumerate(gold_answers):
        norm_gold = normalize_answer(gold_item)
        if not norm_gold:
            continue
        for pred_index, pred_item in enumerate(pred_answers):
            norm_pred = normalize_answer(pred_item)
            if norm_gold in norm_pred or norm_pred in norm_gold:
                scores[gold_index, pred_index] = 1.0
                
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-scores)
    aligned_scores = np.zeros(len(gold_answers))
    for r, c in zip(row_ind, col_ind):
        aligned_scores[r] = scores[r, c]
    return float(all(aligned_scores))


class LLMRAGEvaluator:
    def __init__(
        self,
        corpus: dict[str, dict],
        queries: dict[str, dict],
        qrels: dict[str, dict[str, int]],
        is_multi_answer: bool = False,
    ):
        self._corpus = corpus
        self._queries = queries
        self._qrels = qrels
        self.is_multi_answer = is_multi_answer

    async def _answer_query(
        self,
        query_id: str,
        query_text: str,
        candidate_ids: list[str],
    ) -> tuple[str, dict]:
        """Send one query to the LLM. Returns (answer, usage)."""
        doc_lines = []
        for seq_id, cid in enumerate(candidate_ids, start=0):
            doc = self._corpus.get(cid, {"title": "", "text": ""})
            doc_lines.append(_format_doc(cid, doc["title"], doc["text"], seq_id))

        docs_str = "\n\n".join(doc_lines)
        query_str = (
            f"Based on the documents above, can you answer the following query? "
            f"Print out the passage number and TITLE of the documents you use to answer. "
            f"Then format the answers into a list.\n"
            f"query: {query_text}"
        )

        response, usage = await send_request_multi(
            instructions=SYSTEM_PROMPT,
            context=docs_str,
            query=query_str,
        )
        answer = _parse_answer(response or "")
        return answer, usage, response or ""

    def _answer_query_rlm(
        self,
        query_id: str,
        query_text: str,
        candidate_ids: list[str],
    ) -> tuple[str, dict]:
        """
        RLM-native RAG: docs as context, query in root_prompt. Returns (answer, usage).

        Properly separates data (documents) from instructions (query + task)
        so the RLM can reason iteratively over the corpus to find and
        synthesise an answer.
        """
        from llm_judge.llm_client import send_request_rlm

        doc_entries = []
        for seq_id, cid in enumerate(candidate_ids, start=0):
            doc = self._corpus.get(cid, {"title": "", "text": ""})
            doc_entries.append({
                "id": seq_id,
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            })

        is_multi = self.is_multi_answer
        answer_fmt = (
            "Store ALL answers as a Python list of strings called `answers`, "
            "then call FINAL_VAR(answers)."
            if is_multi
            else "Store the answer as a Python string called `answer`, "
                 "then call FINAL_VAR(answer)."
        )

        root_prompt = (
            f"The `context` variable contains a list of {len(candidate_ids)} documents. "
            f"Each document is a dict with keys 'id' (int), 'title', and 'text'.\n\n"
            f"Answer the following query using the documents:\n"
            f"  \"{query_text}\"\n\n"
            f"Read the documents carefully and find the answer. "
            f"Cite the document IDs you used.\n\n"
            f"{answer_fmt}"
        )

        result, usage = send_request_rlm(root_prompt, doc_entries)

        answer = _parse_answer_from_rlm(result or "")
        logger.info(
            f"  [RLM] Query {query_id}: answer = {answer[:80]}..."
            if len(answer) > 80 else f"  [RLM] Query {query_id}: answer = {answer}"
        )
        return answer, usage, result or ""

    async def evaluate_async(self) -> dict[str, Any]:
        """Run all queries and compute RAG metrics."""
        from llm_judge.llm_client import settings

        total_usage = {
            "input_tokens": 0, "cached_tokens": 0,
            "output_tokens": 0, "thinking_tokens": 0,
            "total_tokens": 0, "cost": 0.0,
            "rlm_total_calls": 0, "rlm_wall_time_s": 0.0,
        }

        valid_queries = list(self._queries.items())
        all_answers = {}
        enriched_results = {}

        if settings.use_rlm:
            # RLM path: synchronous per-query (RLM is blocking)
            logger.info(f"[RLM] Answering {len(valid_queries)} RAG queries ...")
            for query_id, query_data in valid_queries:
                candidate_ids = list(self._corpus.keys())
                random.shuffle(candidate_ids)  # RLM: shuffle is fine (no prefix caching)
                ans, usage, raw = self._answer_query_rlm(
                    query_id, query_data["text"], candidate_ids
                )
                all_answers[query_id] = ans
                enriched_results[query_id] = (ans, raw)
                for k in total_usage:
                    total_usage[k] += usage.get(k, 0)
        else:
            # Standard LLM path: async parallel
            tasks = []
            query_ids_ordered = []

            for query_id, query_data in valid_queries:
                # Use sorted order — NOT shuffled — so the corpus string is
                # IDENTICAL across all queries, enabling Vertex AI implicit caching.
                candidate_ids = sorted(self._corpus.keys())
                tasks.append(self._answer_query(query_id, query_data["text"], candidate_ids))
                query_ids_ordered.append(query_id)

            logger.info(f"Answering {len(tasks)} queries ...")
            results_with_usage = await asyncio.gather(*tasks)

            for qid, (ans, usage, raw) in zip(query_ids_ordered, results_with_usage):
                all_answers[qid] = ans
                enriched_results[qid] = (ans, raw)
                for k in total_usage:
                    total_usage[k] += usage.get(k, 0)
                    
        try:
            import json
            import os
            from llm_judge.llm_client import settings

            sample_size = min(5, len(valid_queries))
            samples = []
            count = 0
            for qid, (ans, raw) in enriched_results.items():
                if count >= sample_size:
                    break
                qtext = next((data["text"] for q, data in valid_queries if q == qid), "Unknown Query")
                gold_answers = self._queries.get(qid, {}).get("gold_answers", [])

                samples.append({
                    "query_id": qid,
                    "query_text": qtext,
                    "model_raw_output": raw,
                    "predicted_answer": ans,
                    "gold_answers": gold_answers,
                    "is_correct": compute_f1(gold_answers, ans) > 0.5 # or compute_em
                })
                count += 1
            
            model_name = settings.model.replace("/", "__")
            output_dir = os.path.join("llm_results", model_name, "results", "no_model_name__available", "no_revision_available")
            os.makedirs(output_dir, exist_ok=True)

            try:
                task_name_guess = self.metadata.name
            except Exception:
                task_name_guess = "rag_task"

            sample_path = os.path.join(output_dir, f"{task_name_guess}_samples.json")
            with open(sample_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2)
            logger.info(f"Saved {sample_size} RAG response samples to {sample_path}")
        except Exception as e:
            logger.warning(f"Failed to save debug samples: {e}")

        metrics = self._compute_metrics(all_answers)
        metrics["usage_stats"] = total_usage
        return metrics

    def _compute_metrics(self, answers: dict[str, str]) -> dict[str, float]:
        import collections
        import numpy as np

        metrics_dict = collections.defaultdict(list)

        for qid, prediction in answers.items():
            gold_answers = self._queries[qid]["gold_answers"]

            if self.is_multi_answer:
                # Same as MultiValueRagEvaluation
                pred_answers = [p.strip() for p in prediction.split(",")] if prediction else []
                
                if not pred_answers:
                    metrics_dict['em'].append(0.0)
                    metrics_dict['subspan_em'].append(0.0)
                    metrics_dict['f1'].append(0.0)
                else:
                    metrics_dict['em'].append(compute_em_multi_value(gold_answers, pred_answers))
                    metrics_dict['coverage'].append(compute_coverage(gold_answers, pred_answers))
                    metrics_dict['subspan_em'].append(compute_multi_value_subspan_em(gold_answers, pred_answers))
            else:
                # Same as RagEvaluation
                if not prediction:
                    metrics_dict['em'].append(0.0)
                    metrics_dict['subspan_em'].append(0.0)
                    metrics_dict['f1'].append(0.0)
                else:
                    metrics_dict['em'].append(compute_em(gold_answers, prediction))
                    metrics_dict['subspan_em'].append(compute_subspan_em(gold_answers, prediction))
                    metrics_dict['f1'].append(compute_f1(gold_answers, prediction))

        final_metrics = {}
        for metric_name, metric_values in metrics_dict.items():
            final_metrics[metric_name] = float(np.mean(metric_values)) if metric_values else 0.0
        
        # MTEB fallback
        final_metrics["main_score"] = final_metrics["em"]

        logger.info(
            f"RAG Results: Exact Match={final_metrics.get('em', 0):.4f} "
            f"Subspan EM={final_metrics.get('subspan_em', 0):.4f}"
        )

        return final_metrics


class HybridRAGEvaluator(LLMRAGEvaluator):
    """
    Retrieve-then-Read Evaluator:
    1. Uses an MTEB Encoder (embedding model) to embed the full corpus and queries.
    2. Performs cosine-similarity retrieval to find the Top-K relevant docs for each query.
    3. Builds the LLM prompt using ONLY those Top-K docs instead of the entire corpus.
    4. Evaluates generation exactly like LLMRAGEvaluator.
    """
    def __init__(
        self,
        corpus: dict[str, dict],
        queries: dict[str, dict],
        qrels: dict[str, dict[str, int]],
        encoder: Any,
        top_k: int = 5,
        is_multi_answer: bool = False,
        encoder_kwargs: dict | None = None,
    ):
        super().__init__(corpus, queries, qrels, is_multi_answer)
        self.encoder = encoder
        self.top_k = top_k
        self.corpus_ids = list(self._corpus.keys())
        self.corpus_embeddings = None
        self.encoder_kwargs = encoder_kwargs or {}
        
    def _safe_encode(self, texts, is_corpus: bool):
        # MTEB's base encode and encode_queries usually expect flat strings.
        flat_texts = [f"{d.get('title', '')} {d.get('text', '')}".strip() if isinstance(d, dict) else d for d in texts]
        
        # MTEB's InstructWrapper (InstructSentenceTransformerModel) specifically expects a DataLoader-like
        # behavior where the input is an iterable of batches, and each batch is a dict with a "text" key
        # containing a list of strings: `[{"text": ["str1", "str2", ...]}]`
        batched_inputs = [{"text": flat_texts}]
        
        # MTEB's encode_corpus expects dictionaries
        mteb_dicts = texts if is_corpus and isinstance(texts[0], dict) else [{"text": t} for t in flat_texts]
        
        kwargs = dict(self.encoder_kwargs or {})
        kwargs.pop("task_name", None) # MTEB wrappers forward this to SentenceTransformer, triggering ValueError
        kwargs.pop("output_folder", None) # Just in case

        # 1. Native MTEB retrieval methods
        try:
            if is_corpus and hasattr(self.encoder, "encode_corpus"):
                out = self.encoder.encode_corpus(mteb_dicts, batch_size=32, **kwargs)
                return out if not hasattr(out, "cpu") else out.cpu().numpy()
            elif not is_corpus and hasattr(self.encoder, "encode_queries"):
                out = self.encoder.encode_queries(flat_texts, batch_size=32, **kwargs)
                return out if not hasattr(out, "cpu") else out.cpu().numpy()
        except (TypeError, ValueError):
            # If native fails because of kwargs or wrapper issues, fall through to safe base encode
            pass

        # 2. Base encode handling
        method = self.encoder.encode
        
        # We need to detect if this is an InstructWrapper or a base SentenceTransformer
        # InstructWrapper expects `[{"text": [...]}]` because it runs `[text for batch in inputs for text in batch["text"]]`
        # SentenceTransformer expects `["...", "..."]` directly.
        from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
        is_instruct_wrapper = isinstance(self.encoder, InstructSentenceTransformerModel) or self.encoder.__class__.__name__ == 'InstructGritLMModel'
        
        inputs_to_use = batched_inputs if is_instruct_wrapper else flat_texts

        try:
            out = method(inputs_to_use, batch_size=32, **kwargs)
            return out if not hasattr(out, "cpu") else out.cpu().numpy()
        except (TypeError, ValueError) as e:
            err_msg = str(e).lower()
            if "unexpected keyword argument" in err_msg or "additional keyword arguments" in err_msg:
                # InstructWrappers strictly require task_metadata, hf_split, hf_subset.
                # If we hit an unexpected kwarg error here, it's either an inner model complaining 
                # (in which case we shouldn't strip the outer wrapper's required kwargs), 
                # or an outer model that genuinely doesn't want them.
                if is_instruct_wrapper:
                    logger.debug("encode rejected kwargs. Instruct wrapper must keep metadata. Raising.")
                    raise e
                
                logger.debug("encode rejected kwargs. Running without them.")
                out = method(inputs_to_use, batch_size=32)
                return out if not hasattr(out, "cpu") else out.cpu().numpy()
            
            # If it failed another way, just blind fallback to the other input format as a last resort
            fallback_inputs = flat_texts if is_instruct_wrapper else batched_inputs
            try:
                out = method(fallback_inputs, batch_size=32, **kwargs)
            except Exception:
                out = method(fallback_inputs, batch_size=32)
            return out if not hasattr(out, "cpu") else out.cpu().numpy()

    def _prepare_retrieval(self):
        """Embeds the entire corpus once."""
        logger.info(f"Embedding {len(self.corpus_ids)} corpus documents for Hybrid RAG...")
        
        # Format corpus for the MTEB encode_corpus method (expects list of dicts)
        corpus_dicts = []
        for cid in self.corpus_ids:
            doc = self._corpus[cid]
            corpus_dicts.append({
                "id": cid,
                "title": doc.get("title", ""),
                "text": doc.get("text", "")
            })
            
        self.corpus_embeddings = self._safe_encode(corpus_dicts, is_corpus=True)
             
    async def evaluate_async(self) -> dict[str, Any]:
        """Run all queries: Retrieve Top-K, then Gen & Eval."""
        import numpy as np
        import asyncio
        from sklearn.metrics.pairwise import cosine_similarity
        from llm_judge.llm_client import settings
        
        # 1. Embed corpus (sync because encoders usually block)
        self._prepare_retrieval()

        # 2. Setup queries and similarities
        valid_queries = list(self._queries.items())
        qids = [qid for qid, _ in valid_queries]
        q_texts = [data["text"] for _, data in valid_queries]
        
        logger.info(f"Embedding {len(q_texts)} queries for Hybrid RAG similarity...")
        query_embeddings = self._safe_encode(q_texts, is_corpus=False)
        similarities = cosine_similarity(query_embeddings, self.corpus_embeddings)

        all_answers = {}
        enriched_results = {}

        total_usage = {
            "input_tokens": 0, "cached_tokens": 0,
            "output_tokens": 0, "thinking_tokens": 0,
            "total_tokens": 0, "cost": 0.0,
            "rlm_total_calls": 0, "rlm_wall_time_s": 0.0,
        }

        if settings.use_rlm:
            # RLM path: synchronous per-query
            logger.info(f"[RLM] Answering {len(qids)} Hybrid RAG queries using Top-{self.top_k}...")
            for i, qid in enumerate(qids):
                qdata = self._queries[qid]
                sim_scores = similarities[i]
                top_k_idx = np.argsort(sim_scores)[::-1][:np.min([self.top_k, len(self.corpus_ids)])]
                retrieved_cids = [self.corpus_ids[idx] for idx in top_k_idx]
                ans, usage, raw = self._answer_query_rlm(qid, qdata["text"], retrieved_cids)
                all_answers[qid] = ans
                enriched_results[qid] = (ans, raw)
                for k in total_usage:
                    total_usage[k] += usage.get(k, 0)
        else:
            # Standard LLM path: async parallel
            tasks = []
            query_ids_ordered = []
            for i, qid in enumerate(qids):
                qdata = self._queries[qid]
                sim_scores = similarities[i]
                top_k_idx = np.argsort(sim_scores)[::-1][:np.min([self.top_k, len(self.corpus_ids)])]
                retrieved_cids = [self.corpus_ids[idx] for idx in top_k_idx]
                tasks.append(self._answer_query(qid, qdata["text"], retrieved_cids))
                query_ids_ordered.append(qid)

            logger.info(f"Answering {len(tasks)} queries using Top-{self.top_k} RAG...")
            results_with_usage = await asyncio.gather(*tasks)

            for qid, (ans, usage, raw) in zip(query_ids_ordered, results_with_usage):
                all_answers[qid] = ans
                enriched_results[qid] = (ans, raw)
                for k in total_usage:
                    total_usage[k] += usage.get(k, 0)

        try:
            sample_size = min(5, len(qids))
            samples = []
            count = 0
            for qid, (ans, raw) in enriched_results.items():
                if count >= sample_size:
                    break
                qtext = self._queries[qid]["text"]
                gold_answers = self._queries[qid].get("gold_answers", [])
                samples.append({
                    "query_id": qid,
                    "query_text": qtext,
                    "model_raw_output": raw,
                    "predicted_answer": ans,
                    "gold_answers": gold_answers,
                    "is_correct": compute_f1(gold_answers, ans) > 0.5
                })
                count += 1

            model_name = settings.model.replace("/", "__")
            output_dir = os.path.join("llm_results", model_name, "results", "no_model_name__available", "no_revision_available")
            os.makedirs(output_dir, exist_ok=True)
            try:
                task_name_guess = self.metadata.name
            except Exception:
                task_name_guess = "hybrid_rag_task"
            sample_path = os.path.join(output_dir, f"{task_name_guess}_samples.json")
            with open(sample_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2)
            logger.info(f"Saved {sample_size} Hybrid RAG response samples to {sample_path}")
        except Exception as e:
            logger.warning(f"Failed to save Hybrid RAG samples: {e}")

        metrics = self._compute_metrics(all_answers)
        metrics["usage_stats"] = total_usage
        return metrics
