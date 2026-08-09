import asyncio
import logging
import random
import re
from typing import Any



from llm_judge.llm_client import send_request_multi

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You will be given a list of documents. You need to read carefully and understand all of them. "
    "Then you will be given a query, and your goal is to find all documents from the list that can help answer the query. "
    "Print out the ID and TITLE of each document.\n\n"
    "Your final answer should be a list of IDs, in the following format:\n"
    "Final Answer: [id1, id2, ...]\n"
    "If there is only one ID, it should be in the format:\n"
    "Final Answer: [id1]\n\n"
    "If there is no perfect answer output the closest one. Do not give an empty final answer."
)


def _format_doc(doc_id: str, title: str, text: str, seq_id: int) -> str:
    """Format a single document for the LLM prompt using the LOFT paper style."""
    title_part = f" | TITLE: {title}" if title.strip() else ""
    return (
        f"ID: {seq_id}{title_part} | CONTENT: {text} | END ID: {seq_id}"
    )


def _parse_ids(response: str, id_map: dict[int, str]) -> list[str]:
    """Extract relevant doc IDs from the LLM's response.

    Returns a list of original corpus _id strings (in the order mentioned).
    """
    # Find the Final Answer: [...] line
    match = re.search(r"Final Answer:\s*\[([^\]]+)\]", response, re.IGNORECASE)
    if not match:
        # Fallback: search for any integer in the response and map them
        nums = re.findall(r"\b(\d+)\b", response)
        found = []
        for n in nums:
            seq = int(n)
            if seq in id_map and id_map[seq] not in found:
                found.append(id_map[seq])
        return found

    raw = match.group(1).strip()
    
    found = []
    # Tokenize by comma, remove quotes and whitespace
    for token in raw.split(","):
        token = token.strip().strip("'\"")
        if token.isdigit():
            seq = int(token)
            if seq in id_map and id_map[seq] not in found:
                found.append(id_map[seq])
    return found


def _parse_ids_from_rlm(response: str, id_map: dict[int, str]) -> list[str]:
    """Parse RLM output into corpus IDs.

    RLM returns raw FINAL_VAR output — could be a Python list repr like
    ``[3, 7, 12]``, a comma-separated string, or prose containing integers.
    We try multiple strategies.
    """
    import json

    if not response or not response.strip():
        return []

    text = response.strip()

    # 1. Try JSON parse (handles "[3, 7, 12]")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            found = []
            for item in parsed:
                seq = int(item)
                if seq in id_map and id_map[seq] not in found:
                    found.append(id_map[seq])
            return found
    except Exception:
        pass

    # 2. Try extracting a list from within the text (handles prose + list)
    list_match = re.search(r"\[([^\]]+)\]", text)
    if list_match:
        found = []
        for token in list_match.group(1).split(","):
            token = token.strip().strip("'\"")
            if token.isdigit():
                seq = int(token)
                if seq in id_map and id_map[seq] not in found:
                    found.append(id_map[seq])
        if found:
            return found

    # 3. Also check for "Final Answer:" in case RLM produced that format
    found = _parse_ids(text, id_map)
    if found:
        return found

    # 4. Last resort: grab any integers
    nums = re.findall(r"\b(\d+)\b", text)
    found = []
    for n in nums:
        seq = int(n)
        if seq in id_map and id_map[seq] not in found:
            found.append(id_map[seq])
    return found


class LLMRetrievalEvaluator:
    def __init__(
        self,
        corpus: dict[str, dict],
        queries: dict[str, str],
        qrels: dict[str, dict[str, int]],
        num_negatives: int = 19,
    ):
        self._corpus = corpus
        self._queries = queries
        self._qrels = qrels
        self.num_negatives = num_negatives

    async def _score_query(
        self,
        query_id: str,
        query_text: str,
        candidate_ids: list[str],
    ) -> list[str]:
        """Send one query to the LLM. Returns list of retrieved corpus_ids in order."""
        id_map: dict[int, str] = {}
        doc_lines = []
        for seq_id, cid in enumerate(candidate_ids):
            doc = self._corpus.get(cid, {"title": "", "text": ""})
            doc_lines.append(_format_doc(cid, doc["title"], doc["text"], seq_id))
            id_map[seq_id] = cid

        docs_str = "\n\n".join(doc_lines)
        query_str = (
            "====== Now let's start! ======\n"
            "Which document is most relevant to answer the query? "
            "Print out the TITLE and ID of the document. Then format the IDs into a list.\n"
            "If there is no perfect answer output the closest one. Do not give an empty final answer.\n"
            f"query: {query_text}\n"
            "The following documents can help answer the query:"
        )

        response, usage = await send_request_multi(
            instructions=SYSTEM_PROMPT,
            context=docs_str,
            query=query_str,
        )

        retrieved_ids = _parse_ids(response or "", id_map)
        logger.info(
            f"  Query {query_id}: retrieved {len(retrieved_ids)}/{len(candidate_ids)} docs"
        )

        return retrieved_ids, usage, response or ""

    def _score_query_rlm_fresh(
        self,
        query_id: str,
        query_text: str,
        candidate_ids: list[str],
    ) -> tuple[list[str], dict, str]:
        """Thread-safe variant of _score_query_rlm — uses a fresh RLM instance per call."""
        from llm_judge.llm_client import send_request_rlm_fresh

        id_map: dict[int, str] = {}
        doc_entries = []
        for seq_id, cid in enumerate(candidate_ids):
            doc = self._corpus.get(cid, {"title": "", "text": ""})
            doc_entries.append({
                "id": seq_id,
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            })
            id_map[seq_id] = cid

        root_prompt = (
            f"The `context` variable contains a list of {len(candidate_ids)} documents. "
            f"Each document is a dict with keys 'id' (int), 'title', and 'text'.\n\n"
            f"Find ALL documents relevant to this query:\n"
            f"  \"{query_text}\"\n\n"
            f"Store the result as a Python list of integer IDs called `relevant_ids` "
            f"(in order of relevance, most relevant first), "
            f"then call FINAL_VAR(relevant_ids).\n\n"
            f"If no document is a perfect match, return the closest one. "
            f"Do not return an empty list."
        )

        result, usage = send_request_rlm_fresh(root_prompt, doc_entries)

        retrieved_ids = _parse_ids_from_rlm(result or "", id_map)
        logger.info(
            f"  [RLM-fresh] Query {query_id}: retrieved {len(retrieved_ids)}/{len(candidate_ids)} docs"
        )

        return retrieved_ids, usage, result or ""

    def _score_query_rlm(
        self,
        query_id: str,
        query_text: str,
        candidate_ids: list[str],
    ) -> list[str]:
        """RLM-native retrieval: docs as context, query in root_prompt.

        Properly separates data (documents) from instructions (query + task)
        so the RLM can reason over the corpus iteratively.
        """
        from llm_judge.llm_client import send_request_rlm

        id_map: dict[int, str] = {}
        doc_entries = []
        for seq_id, cid in enumerate(candidate_ids):
            doc = self._corpus.get(cid, {"title": "", "text": ""})
            doc_entries.append({
                "id": seq_id,
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            })
            id_map[seq_id] = cid

        root_prompt = (
            f"The `context` variable contains a list of {len(candidate_ids)} documents. "
            f"Each document is a dict with keys 'id' (int), 'title', and 'text'.\n\n"
            f"Find ALL documents relevant to this query:\n"
            f"  \"{query_text}\"\n\n"
            f"Store the result as a Python list of integer IDs called `relevant_ids` "
            f"(in order of relevance, most relevant first), "
            f"then call FINAL_VAR(relevant_ids).\n\n"
            f"If no document is a perfect match, return the closest one. "
            f"Do not return an empty list."
        )

        result, usage = send_request_rlm(root_prompt, doc_entries)

        retrieved_ids = _parse_ids_from_rlm(result or "", id_map)
        logger.info(
            f"  [RLM] Query {query_id}: retrieved {len(retrieved_ids)}/{len(candidate_ids)} docs"
        )

        return retrieved_ids, usage, result or ""

    async def evaluate_async(self) -> dict[str, Any]:
        """Run all queries and compute retrieval metrics."""
        from llm_judge.llm_client import settings

        total_usage = {
            "input_tokens": 0, "cached_tokens": 0,
            "output_tokens": 0, "thinking_tokens": 0,
            "total_tokens": 0, "cost": 0.0,
            "rlm_total_calls": 0, "rlm_wall_time_s": 0.0,
        }

        query_items = list(self._queries.items())
        all_results = {}
        enriched_results = {}

        if settings.use_rlm:
            import concurrent.futures
            _query_timeout = 600
            _zero_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                           "cost": 0.0, "rlm_total_calls": 0, "rlm_wall_time_s": 0.0}

            logger.info(
                f"[RLM] Scoring {len(query_items)} retrieval queries "
                f"(max_workers={settings.max_concurrency}, timeout={_query_timeout}s) ..."
            )

            prepared = []
            for query_id, query_text in query_items:
                candidate_ids = list(self._corpus.keys())
                random.shuffle(candidate_ids)
                prepared.append((query_id, query_text, candidate_ids))

            def _run_all_rlm_sync(items):
                with concurrent.futures.ThreadPoolExecutor(max_workers=settings.max_concurrency) as pool:
                    fs = {pool.submit(self._score_query_rlm_fresh, qid, qtext, cids): qid
                          for qid, qtext, cids in items}
                    done_count = 0
                    for f in concurrent.futures.as_completed(fs, timeout=_query_timeout * len(items)):
                        qid = fs[f]
                        done_count += 1
                        try:
                            res, usage, raw = f.result(timeout=_query_timeout)
                        except (concurrent.futures.TimeoutError, TimeoutError):
                            logger.error(f"[RLM] Query {qid} timed out after {_query_timeout}s, skipping")
                            res, usage, raw = [], _zero_usage.copy(), ""
                        except Exception as exc:
                            logger.error(f"[RLM] Query {qid} failed: {type(exc).__name__}: {exc}")
                            res, usage, raw = [], _zero_usage.copy(), ""

                        all_results[qid] = res
                        enriched_results[qid] = (res, raw)
                        for k in total_usage:
                            total_usage[k] += usage.get(k, 0)
                        logger.info(f"[RLM] {done_count}/{len(items)} ({qid}): retrieved {len(res)} docs")

                    for qid, _qtext, _cids in items:
                        if qid not in all_results:
                            logger.error(f"[RLM] Query {qid} never completed, skipping")
                            all_results[qid] = []
                            enriched_results[qid] = ([], "")

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _run_all_rlm_sync, prepared)
        else:
            tasks = []
            query_ids_ordered = []
            for query_id, query_text in query_items:
                candidate_ids = sorted(self._corpus.keys())
                tasks.append(self._score_query(query_id, query_text, candidate_ids))
                query_ids_ordered.append(query_id)

            logger.info(f"Scoring {len(tasks)} queries ...")
            results_with_usage = await asyncio.gather(*tasks)

            for qid, (res, usage, raw) in zip(query_ids_ordered, results_with_usage):
                all_results[qid] = res
                enriched_results[qid] = (res, raw)
                for k in total_usage:
                    total_usage[k] += usage.get(k, 0)
                    
        try:
            import json
            import os
            from llm_judge.llm_client import settings

            sample_size = min(5, len(query_items))
            samples = []
            count = 0
            for qid, (res, raw) in enriched_results.items():
                if count >= sample_size:
                    break
                qtext = next((text for q, text in query_items if q == qid), "Unknown Query")
                gold_ids = [doc_id for doc_id, rel in self._qrels.get(qid, {}).items() if rel > 0]
                
                samples.append({
                    "query_id": qid,
                    "query_text": qtext,
                    "model_raw_output": raw,
                    "parsed_ids": res,
                    "gold_ids": gold_ids,
                    "is_correct": bool(set(res).intersection(gold_ids))
                })
                count += 1
            
            model_name = settings.model.replace("/", "__")
            output_dir = os.path.join("llm_results", model_name, "results", "no_model_name__available", "no_revision_available")
            os.makedirs(output_dir, exist_ok=True)

            try:
                task_name_guess = self.metadata.name
            except Exception:
                task_name_guess = "retrieval_task"

            sample_path = os.path.join(output_dir, f"{task_name_guess}_samples.json")
            with open(sample_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2)
            logger.info(f"Saved {sample_size} retrieval response samples to {sample_path}")
        except Exception as e:
            logger.warning(f"Failed to save debug samples: {e}")

        metrics = self._compute_metrics(all_results)
        metrics["usage_stats"] = total_usage
        return metrics

    def _compute_metrics(self, results: dict[str, list[str]]) -> dict[str, float]:
        import collections
        import numpy as np

        def compute_recall_at_k(gold_ids: list[str], pred_ids: list[str], top_k: int, capped: bool = False) -> float:
            if not pred_ids:
                return 0.0
            pred_ids_k = set(pred_ids[:top_k])
            relevant_in_top_k = float(len(pred_ids_k.intersection(gold_ids)))
            if capped and len(gold_ids) > top_k:
                return relevant_in_top_k / top_k
            else:
                return relevant_in_top_k / (len(gold_ids) if len(gold_ids) > 0 else 1)

        def compute_mrecall_at_k(gold_ids: list[str], pred_ids: list[str], top_k: int) -> float:
            if not pred_ids:
                return 0.0
            pred_ids_k = set(pred_ids[:top_k])
            relevant_in_top_k = float(len(pred_ids_k.intersection(gold_ids)))
            return float(relevant_in_top_k == min(top_k, len(gold_ids)))

        metrics = collections.defaultdict(list)

        for qid, res in results.items():
            if qid not in self._qrels:
                continue
            
            gold_ids = [doc_id for doc_id, rel in self._qrels[qid].items() if rel > 0]
            pred_ids = res

            metrics["recall@1"].append(compute_recall_at_k(gold_ids, pred_ids, 1, False))
            metrics["recall@2"].append(compute_recall_at_k(gold_ids, pred_ids, 2, False))
            metrics["recall@3"].append(compute_recall_at_k(gold_ids, pred_ids, 3, False))
            metrics["recall@5"].append(compute_recall_at_k(gold_ids, pred_ids, 5, False))
            metrics["mrecall@1"].append(compute_mrecall_at_k(gold_ids, pred_ids, 1))
            metrics["mrecall@2"].append(compute_mrecall_at_k(gold_ids, pred_ids, 2))
            metrics["mrecall@3"].append(compute_mrecall_at_k(gold_ids, pred_ids, 3))
            metrics["mrecall@5"].append(compute_mrecall_at_k(gold_ids, pred_ids, 5))
            metrics["capped_recall@1"].append(compute_recall_at_k(gold_ids, pred_ids, 1, True))

        final_metrics = {}
        for metric_name, metric_values in metrics.items():
            final_metrics[metric_name] = float(np.mean(metric_values)) if metric_values else 0.0
        
        final_metrics["main_score"] = final_metrics["recall@1"]
        
        logger.info(
            f"Retrieval Results: Recall@1={final_metrics.get('recall@1', 0):.4f} "
            f"mRecall@1={final_metrics.get('mrecall@1', 0):.4f}"
        )
        return final_metrics
