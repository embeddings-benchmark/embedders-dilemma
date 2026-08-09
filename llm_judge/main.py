from pathlib import Path
from typing import Any

import numpy as np
import mteb
from mteb.models.model_meta import ModelMeta

from llm_judge.settings import Settings


class _DummyEncoder:
    """Encoder stub for our LLM tasks, which override _evaluate_subset and never
    encode. mteb>=2.6 calls meta.load_model() before running, so the loader
    points back at this instance to avoid a HuggingFace lookup for the anonymous
    name. mteb_model_meta is a settable attribute because load_model() reassigns it.
    """

    def __init__(self) -> None:
        meta = ModelMeta._from_hub(None)
        meta.loader = lambda *args, **kwargs: self
        self.mteb_model_meta = meta

    def encode(self, inputs: Any, **kwargs: Any) -> np.ndarray:
        n = len(inputs) if hasattr(inputs, "__len__") else 1
        return np.zeros((n, 1), dtype=np.float32)

    def similarity(self, embeddings1: Any, embeddings2: Any) -> np.ndarray:
        return np.zeros((len(embeddings1), len(embeddings2)), dtype=np.float32)

    def similarity_pairwise(self, embeddings1: Any, embeddings2: Any) -> np.ndarray:
        return np.zeros(len(embeddings1), dtype=np.float32)

# Classification
from llm_judge.tasks.classification import (
    LLMAmazonCounterfactualClassification,
    LLMBanking77Classification,
    LLMImdbClassification,
    LLMMTOPDomainClassification,
    LLMMassiveIntentClassification,
    LLMMassiveScenarioClassification,
    LLMToxicConversationsClassification,
    LLMTweetSentimentExtractionClassification,
)
# STS
from llm_judge.tasks.sts import (
    LLMSTSBenchmark,
    LLMSICKR,
    LLMSTS12,
    LLMSTS13,
    LLMSTS14,
    LLMSTS15,
    LLMSTS16,
    LLMBIOSSES,
    LLMSTS17,
    LLMSTS22v2,
)
# Clustering
from llm_judge.tasks.clustering import (
    LLMRedditClusteringP2P,
    LLMTwentyNewsgroupsClusteringV2,
    LLMStackExchangeClusteringP2PV2,
    LLMStackExchangeClusteringV2,
    LLMArxivClusteringP2P,
    LLMArxivClusteringS2S,
    LLMBiorxivClusteringP2PV2,
    LLMMedrxivClusteringP2PV2,
    LLMMedrxivClusteringS2SV2,
)
# Pair Classification
from llm_judge.tasks.pair_cls import (
    LLMSprintDuplicateQuestionsPC,
    LLMTwitterURLCorpusPC,
    LLMLegalBenchPC,  # disabled in active task list (mteb 2.11+ needed for multilingual config)
)
# Retrieval — V2/V3 small-corpus MTEB datasets (uploaded as mteb/llm-eval-*).
# Track B (BRIGHT pipeline) is run via scripts/experiments/run_pipeline.py — not here.
# Additional LOFT retrieval, LOFT RAG, Hybrid RAG, and HUME variants are defined in
# llm_judge/tasks/{retrieval,rag,sts,reranking}.py; import and add them below for ablations.
from llm_judge.tasks.retrieval import (
    LLMAILAStatutes,
    LLMTwitterHjerneRetrieval,
    LLMFQuADRetrieval,
    LLMLegalBenchConsumerContractsQA,
    LLMPublicHealthQA,
    LLMHC3FinanceRetrieval,
)
import argparse

settings = Settings()

def main():
    parser = argparse.ArgumentParser(description="Run Loft LLM/Hybrid RAG Evaluation")
    parser.add_argument("--mode", type=str, choices=["pure_llm", "hybrid_rag"], default="pure_llm",
                        help="Whether to run Pure LLM RAG or Hybrid (Retrieve-then-Read) RAG.")
    parser.add_argument("--embedding_model", type=str, default="intfloat/e5-small-v2",
                        help="HuggingFace model ID for embedding corpus in Hybrid RAG mode.")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of documents to retrieve per query in Hybrid RAG mode.")
    args = parser.parse_args()

    if args.mode == "hybrid_rag":
        print(f"Loading embedding model: {args.embedding_model} for Hybrid RAG Top-{args.top_k} retrieval.")
        mteb_model = mteb.get_model(args.embedding_model)
        # We need to temporarily patch AbsTaskHybridRAG to use this specific top_k
        from llm_judge.tasks.rag import AbsTaskHybridRAG
        AbsTaskHybridRAG.top_k = args.top_k
        output_prefix = "embedding_results"
    else:
        mteb_model = _DummyEncoder()
        output_prefix = "rlm_results" if settings.use_rlm else "llm_results"

    task = [

        # # ---- Classification (8 Tasks + Multilingual) ----
        LLMBanking77Classification(),
        LLMImdbClassification(),
        LLMToxicConversationsClassification(),
        LLMTweetSentimentExtractionClassification(),
        
        # # # Multilingual:
        LLMAmazonCounterfactualClassification(),
        LLMMTOPDomainClassification(),
        LLMMassiveIntentClassification(),
        LLMMassiveScenarioClassification(),

        # ---- STS (10 tasks: 8 standard + 2 multilingual) ----
        LLMSTSBenchmark(),
        LLMSICKR(),
        LLMSTS12(),
        LLMSTS13(),
        LLMSTS14(),
        LLMSTS15(),
        LLMSTS16(),
        LLMBIOSSES(),

        LLMSTS17(),
        LLMSTS22v2(),

        # ---- Clustering (9 tasks) ----
        LLMRedditClusteringP2P(),
        LLMTwentyNewsgroupsClusteringV2(),
        LLMStackExchangeClusteringP2PV2(),
        LLMStackExchangeClusteringV2(),
        LLMArxivClusteringP2P(),
        LLMArxivClusteringS2S(),
        LLMBiorxivClusteringP2PV2(),
        LLMMedrxivClusteringP2PV2(),
        LLMMedrxivClusteringS2SV2(),

        # ---- Pair Classification (3 active; RTE3PC needs mteb 2.11+ for multilingual config) ----
        LLMSprintDuplicateQuestionsPC(),
        LLMTwitterURLCorpusPC(),
        LLMLegalBenchPC(),

        # ---- Retrieval (6 small-corpus CiC tasks; Track B BRIGHT pipeline runs via run_pipeline.py) ----
        LLMAILAStatutes(),                  #  82 docs /  50 q  (legal case→statute)
        LLMTwitterHjerneRetrieval(),        # 262 docs /  77 q  (Danish social Q&A)
        LLMFQuADRetrieval(),                # 269 docs / 100 q  (French Wikipedia QA)
        LLMLegalBenchConsumerContractsQA(), # 154 docs / 100 q  (consumer ToS clauses)
        LLMPublicHealthQA(),                # 172 docs / 100 q  (English health/COVID QA)
        LLMHC3FinanceRetrieval(),           # 415 docs / 100 q  (finance Q&A — non-saturated)
        # Audit-excluded / saturated / context-overflow — re-enable for appendix runs:
        # LLMTempReasonL1(),             # answer-pool corpus (date arithmetic)
        # LLMSpartQA(),                  # answer-pool corpus (MCQ spatial reasoning)
        # LLMWinoGrande(),               # answer-pool corpus (single-word coref)
        # LLMHumanEvalRetrieval(),       # saturated for both paradigms (Octen 0.998)
        # LLMFinanceBenchRetrieval(),    # saturated for top embedders (Octen 0.946)
        # LLMBuiltBenchRetrieval(),      # ~231K tokens exceeds 128K context of most open LLMs
        # LLMLegalBenchCorporateLobbying(),  # near-saturated; redundant with LBConsumer
    ]

    root_folder = Path(__file__).parent.parent / output_prefix
    
    if args.mode == "hybrid_rag":
        # Nest under embedding model -> hybrid_rag__LLM_model
        emb_folder_name = args.embedding_model.replace("/", "__")
        hybrid_folder_name = f"hybrid_rag__{settings.model.replace('/', '__')}"
        model_folder = root_folder / emb_folder_name / hybrid_folder_name
    else:
        model_folder = root_folder / settings.model.replace("/", "__")
        
    model_folder.mkdir(parents=True, exist_ok=True)
    
    cache = mteb.cache.ResultCache(model_folder)
    mteb.evaluate(
        model=mteb_model,
        tasks=task,
        cache=cache,
    )


if __name__ == "__main__":
    main()