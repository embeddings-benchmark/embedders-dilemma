"""BRIGHT retrieval tasks as standard MTEB `AbsTaskRetrieval` subclasses.

Replaces the old `retrieval_pipeline.py`. Each task is a single-stage retrieval task
from MTEB's perspective; running multi-stage pipelines (bi-encoder → cross-encoder →
LLM rerank) is done by composing them at the runner level via MTEB's native
`prediction_folder` + `task.convert_to_reranking(...)` mechanism.

See `scripts/experiments/run_pipeline.py` for the orchestration.

BRIGHT-specific behavior: each example carries an `excluded_ids` list (documents
that must not be ranked for that query, by paper convention). We mask them out of
the retrieval results *before* scoring — without altering the corpus, since one
query's excluded doc may be another query's gold doc.

Reference: https://arxiv.org/abs/2407.12883
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset, load_dataset
from mteb.abstasks.retrieval import (
    AbsTaskRetrieval,
    _filter_queries_without_positives,
)
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata
from mteb._evaluators import RetrievalEvaluator
from mteb._evaluators.retrieval_metrics import make_score_dict
from mteb.models import (
    CrossEncoderProtocol,
    EncoderProtocol,
    SearchCrossEncoderWrapper,
    SearchEncoderWrapper,
    SearchProtocol,
)

logger = logging.getLogger(__name__)


def _make_metadata(
    name: str,
    description: str,
    domains: list[str],
    prompt_domain: str,
) -> TaskMetadata:
    # Paper Table 8 (Su et al. 2024) — used uniformly for E5/GritLM/Qwen/SFR-family
    # instruction-tuned embedders on the StackExchange-style BRIGHT subsets.
    # mteb's instruct_wrapper reads `metadata.prompt['query']` and prepends each
    # model's own instruction prefix (e.g. "Instruct: " for Qwen3).
    prompt = {
        "query": f"Given a {prompt_domain} post, retrieve relevant passages that help answer the post",
    }
    return TaskMetadata(
        name=name,
        # `path` is informational only — we override load_data() and ignore the loader.
        dataset={"path": "xlangai/BRIGHT", "revision": "main"},
        description=description,
        reference="https://arxiv.org/abs/2407.12883",
        category="t2t",
        modalities=["text"],
        type="Retrieval",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-12-31"),
        domains=domains,
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        prompt=prompt,
    )


class _BRIGHTRetrievalBase(AbsTaskRetrieval):
    """Base class. Subclasses set `bright_subset`."""

    bright_subset: str = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._excluded_ids: dict[str, set[str]] = {}

    def load_data(self) -> None:
        if self.data_loaded:
            return

        docs = load_dataset("xlangai/BRIGHT", "documents", split=self.bright_subset)
        examples = load_dataset("xlangai/BRIGHT", "examples", split=self.bright_subset)

        corpus_ids = list(docs["id"])
        corpus_texts = list(docs["content"])
        corpus = Dataset.from_dict(
            {
                "id": corpus_ids,
                "title": [""] * len(corpus_ids),
                "text": corpus_texts,
            }
        )

        query_ids: list[str] = []
        query_texts: list[str] = []
        relevant_docs: dict[str, dict[str, int]] = {}
        excluded_ids: dict[str, set[str]] = {}
        for e in examples:
            qid = str(e["id"])
            query_ids.append(qid)
            query_texts.append(e["query"])
            relevant_docs[qid] = {gid: 1 for gid in e["gold_ids"]}
            excl = {
                x
                for x in (e.get("excluded_ids") or [])
                if x and x != "N/A"
            }
            excluded_ids[qid] = excl

        queries = Dataset.from_dict({"id": query_ids, "text": query_texts})

        self._excluded_ids = excluded_ids
        self.dataset["default"]["test"] = RetrievalSplitData(
            corpus=corpus,
            queries=queries,
            relevant_docs=relevant_docs,
            top_ranked=None,
        )
        self.data_loaded = True
        logger.info(
            "Loaded %s: %d docs, %d queries, %d gold links, %d excluded",
            self.metadata.name,
            len(corpus_ids),
            len(query_ids),
            sum(len(v) for v in relevant_docs.values()),
            sum(len(v) for v in excluded_ids.values()),
        )

    def _evaluate_subset(
        self,
        model,
        data_split: RetrievalSplitData,
        *,
        encode_kwargs: dict[str, Any],
        hf_split: str,
        hf_subset: str,
        prediction_folder=None,
        **kwargs,
    ):
        """Same flow as `AbsTaskRetrieval._evaluate_subset`, with BRIGHT excluded-id
        masking applied to the search results before scoring.
        """
        data_split["relevant_docs"], data_split["queries"] = (
            _filter_queries_without_positives(
                data_split["relevant_docs"], data_split["queries"]
            )
        )
        retriever = RetrievalEvaluator(
            corpus=data_split["corpus"],
            queries=data_split["queries"],
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            top_ranked=data_split["top_ranked"],
            top_k=self._top_k,
            **kwargs,
        )

        if isinstance(model, EncoderProtocol) and not isinstance(model, SearchProtocol):
            search_model = SearchEncoderWrapper(model)
        elif isinstance(model, CrossEncoderProtocol):
            search_model = SearchCrossEncoderWrapper(model)
        elif isinstance(model, SearchProtocol):
            search_model = model
        else:
            raise TypeError(
                f"Unsupported model type for BRIGHT retrieval: {type(model)}"
            )

        results = retriever(search_model, encode_kwargs=encode_kwargs)

        # BRIGHT excluded-id mask: applied per-query, after search, before scoring.
        # Done here (not in the corpus) because a doc excluded for query A may be
        # gold for query B.
        n_dropped = 0
        for qid, excl in self._excluded_ids.items():
            if not excl or qid not in results:
                continue
            before = len(results[qid])
            results[qid] = {d: s for d, s in results[qid].items() if d not in excl}
            n_dropped += before - len(results[qid])
        if n_dropped:
            logger.info(
                "%s: dropped %d excluded docs from results across queries",
                self.metadata.name,
                n_dropped,
            )

        if prediction_folder:
            self._save_task_predictions(
                results,
                model,
                prediction_folder,
                hf_subset=hf_subset,
                hf_split=hf_split,
            )

        (
            all_scores,
            ndcg,
            _map,
            recall,
            precision,
            naucs,
            mrr,
            naucs_mrr,
            cv_recall,
        ) = retriever.evaluate(
            data_split["relevant_docs"],
            results,
            self.k_values,
            ignore_identical_ids=self.ignore_identical_ids,
            skip_first_result=self.skip_first_result,
        )
        task_specific = self.task_specific_scores(
            all_scores,
            data_split["relevant_docs"],
            results,
            hf_split=hf_split,
            hf_subset=hf_subset,
        )
        return make_score_dict(
            ndcg,
            _map,
            recall,
            precision,
            mrr,
            naucs,
            naucs_mrr,
            cv_recall,
            task_specific,
            self._previous_results_model_meta,
        )


# ---------------------------------------------------------------------------
# BRIGHT STEM-7
# ---------------------------------------------------------------------------


class BRIGHTBiology(_BRIGHTRetrievalBase):
    bright_subset = "biology"
    metadata = _make_metadata(
        "BRIGHTBiology",
        "BRIGHT Biology: reasoning-intensive retrieval of biology Wikipedia passages.",
        ["Academic", "Written"],
        prompt_domain="Biology",
    )


class BRIGHTEarthScience(_BRIGHTRetrievalBase):
    bright_subset = "earth_science"
    metadata = _make_metadata(
        "BRIGHTEarthScience",
        "BRIGHT Earth Science: reasoning-intensive retrieval of earth science passages.",
        ["Academic", "Written"],
        prompt_domain="Earth Science",
    )


class BRIGHTEconomics(_BRIGHTRetrievalBase):
    bright_subset = "economics"
    metadata = _make_metadata(
        "BRIGHTEconomics",
        "BRIGHT Economics: reasoning-intensive retrieval of economics passages.",
        ["Academic", "Written"],
        prompt_domain="Economics",
    )


class BRIGHTPsychology(_BRIGHTRetrievalBase):
    bright_subset = "psychology"
    metadata = _make_metadata(
        "BRIGHTPsychology",
        "BRIGHT Psychology: reasoning-intensive retrieval of psychology passages.",
        ["Academic", "Written"],
        prompt_domain="Psychology",
    )


class BRIGHTRobotics(_BRIGHTRetrievalBase):
    bright_subset = "robotics"
    metadata = _make_metadata(
        "BRIGHTRobotics",
        "BRIGHT Robotics: reasoning-intensive retrieval over robotics/ROS docs and code.",
        ["Engineering", "Programming", "Written"],
        prompt_domain="Robotics",
    )


class BRIGHTStackoverflow(_BRIGHTRetrievalBase):
    bright_subset = "stackoverflow"
    metadata = _make_metadata(
        "BRIGHTStackoverflow",
        "BRIGHT StackOverflow: reasoning-intensive retrieval over Stack Overflow Q&A.",
        ["Programming", "Written"],
        prompt_domain="Stack Overflow",
    )


class BRIGHTSustainableLiving(_BRIGHTRetrievalBase):
    bright_subset = "sustainable_living"
    metadata = _make_metadata(
        "BRIGHTSustainableLiving",
        "BRIGHT Sustainable Living: reasoning-intensive retrieval over sustainability discussions.",
        ["Academic", "Written"],
        prompt_domain="Sustainable Living",
    )


BRIGHT_STEM7 = [
    BRIGHTBiology,
    BRIGHTEarthScience,
    BRIGHTEconomics,
    BRIGHTPsychology,
    BRIGHTRobotics,
    BRIGHTStackoverflow,
    BRIGHTSustainableLiving,
]

# Register custom BRIGHT tasks with mteb so instruction-tuned embedders that
# call `mteb.get_task(task_name=...)` to fetch task-specific prompts can find
# them (e.g. Qwen3-Embedding via mteb.models.instruct_wrapper).
def _register_with_mteb() -> None:
    from mteb.get_tasks import _TASKS_REGISTRY
    for cls in BRIGHT_STEM7:
        _TASKS_REGISTRY.setdefault(cls.metadata.name, cls)

_register_with_mteb()
