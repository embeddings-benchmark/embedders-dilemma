"""TEMP: 6 retrieval tasks at a reasoning level for the .env model.
Writes to ablation_results/thinking/<level>/<model>/. level = off|low|default. Delete after."""
from pathlib import Path
import mteb
from llm_judge.main import _DummyEncoder
from llm_judge.settings import Settings
from llm_judge.tasks.retrieval import (
    LLMAILAStatutes, LLMTwitterHjerneRetrieval, LLMFQuADRetrieval,
    LLMLegalBenchConsumerContractsQA, LLMPublicHealthQA, LLMHC3FinanceRetrieval,
)

settings = Settings()
level = "off" if not settings.enable_thinking else (settings.reasoning_effort or "default")
tasks = [
    LLMAILAStatutes(), LLMTwitterHjerneRetrieval(), LLMFQuADRetrieval(),
    LLMLegalBenchConsumerContractsQA(), LLMPublicHealthQA(), LLMHC3FinanceRetrieval(),
]
mf = Path("ablation_results") / "thinking" / level / settings.model.replace("/", "__")
mf.mkdir(parents=True, exist_ok=True)
print(f"model={settings.model} effort={level} -> {mf}")
cache = mteb.cache.ResultCache(mf)
mteb.evaluate(model=_DummyEncoder(), tasks=tasks, cache=cache)
