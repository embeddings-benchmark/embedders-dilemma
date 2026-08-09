# MTEB(LLM) Task Suite (37 Tasks)

All tasks are held-out subsets (seed = 42) of MTEB/MMTEB tasks, hosted on Hugging Face under `mteb/llm-eval-*`.
Each task class in [`llm_judge/tasks/`](llm_judge/tasks/) pins its dataset to an exact revision, so results reproduce even if a dataset is later updated.
Both paradigms are evaluated on identical data: embedding models embed the same texts the LLMs read.

Category sizes: Classification 8, Clustering 9, STS 10, PairClassification 4, Retrieval 6.

## Classification (8)

| Task | Hugging Face dataset | Pinned revision |
|---|---|---|
| LLMAmazonCounterfactualClassification | [`mteb/llm-eval-amazon_counterfactual`](https://huggingface.co/datasets/mteb/llm-eval-amazon_counterfactual) | `8df4b672d551` |
| LLMBanking77Classification | [`mteb/llm-eval-banking77`](https://huggingface.co/datasets/mteb/llm-eval-banking77) | `2c82d499a4aa` |
| LLMImdbClassification | [`mteb/llm-eval-imdb`](https://huggingface.co/datasets/mteb/llm-eval-imdb) | `c7cd15a51954` |
| LLMMTOPDomainClassification | [`mteb/llm-eval-mtop_domain`](https://huggingface.co/datasets/mteb/llm-eval-mtop_domain) | `14315a9fd430` |
| LLMMassiveIntentClassification | [`mteb/llm-eval-massive_intent`](https://huggingface.co/datasets/mteb/llm-eval-massive_intent) | `69fe18bc7341` |
| LLMMassiveScenarioClassification | [`mteb/llm-eval-massive_scenario`](https://huggingface.co/datasets/mteb/llm-eval-massive_scenario) | `ba1521289b08` |
| LLMToxicConversationsClassification | [`mteb/llm-eval-toxic_conversations`](https://huggingface.co/datasets/mteb/llm-eval-toxic_conversations) | `34eeb6105ca2` |
| LLMTweetSentimentExtractionClassification | [`mteb/llm-eval-tweet_sentiment`](https://huggingface.co/datasets/mteb/llm-eval-tweet_sentiment) | `5e847f1f41ec` |

## Clustering (9)

| Task | Hugging Face dataset | Pinned revision |
|---|---|---|
| LLMArxivClusteringP2P | [`mteb/llm-eval-arxiv_clustering_p2p`](https://huggingface.co/datasets/mteb/llm-eval-arxiv_clustering_p2p) | `3ce50711e475` |
| LLMArxivClusteringS2S | [`mteb/llm-eval-arxiv_clustering_s2s`](https://huggingface.co/datasets/mteb/llm-eval-arxiv_clustering_s2s) | `28c899f0bb82` |
| LLMBiorxivClusteringP2PV2 | [`mteb/llm-eval-biorxiv_clustering_p2p_v2`](https://huggingface.co/datasets/mteb/llm-eval-biorxiv_clustering_p2p_v2) | `9e11c95384ef` |
| LLMMedrxivClusteringP2PV2 | [`mteb/llm-eval-medrxiv_clustering_p2p_v2`](https://huggingface.co/datasets/mteb/llm-eval-medrxiv_clustering_p2p_v2) | `63c8e6cfbcab` |
| LLMMedrxivClusteringS2SV2 | [`mteb/llm-eval-medrxiv_clustering_s2s_v2`](https://huggingface.co/datasets/mteb/llm-eval-medrxiv_clustering_s2s_v2) | `c565eea82f4a` |
| LLMRedditClusteringP2P | [`mteb/llm-eval-reddit_clustering_p2p`](https://huggingface.co/datasets/mteb/llm-eval-reddit_clustering_p2p) | `298518f04f9f` |
| LLMStackExchangeClusteringP2PV2 | [`mteb/llm-eval-stackexchange_clustering_p2p_v2`](https://huggingface.co/datasets/mteb/llm-eval-stackexchange_clustering_p2p_v2) | `5de039af3492` |
| LLMStackExchangeClusteringV2 | [`mteb/llm-eval-stackexchange_clustering_v2`](https://huggingface.co/datasets/mteb/llm-eval-stackexchange_clustering_v2) | `2cbee8fd7c71` |
| LLMTwentyNewsgroupsClusteringV2 | [`mteb/llm-eval-twenty_newsgroups_v2`](https://huggingface.co/datasets/mteb/llm-eval-twenty_newsgroups_v2) | `0187d654bb25` |

## STS (10)

| Task | Hugging Face dataset | Pinned revision |
|---|---|---|
| LLMBIOSSES | [`mteb/llm-eval-biosses`](https://huggingface.co/datasets/mteb/llm-eval-biosses) | `cf968edb41fa` |
| LLMSICKR | [`mteb/llm-eval-sickr`](https://huggingface.co/datasets/mteb/llm-eval-sickr) | `82eb9939fa17` |
| LLMSTS12 | [`mteb/llm-eval-sts12`](https://huggingface.co/datasets/mteb/llm-eval-sts12) | `1559a6e5259e` |
| LLMSTS13 | [`mteb/llm-eval-sts13`](https://huggingface.co/datasets/mteb/llm-eval-sts13) | `e0251ba4e151` |
| LLMSTS14 | [`mteb/llm-eval-sts14`](https://huggingface.co/datasets/mteb/llm-eval-sts14) | `6237db5b2c4b` |
| LLMSTS15 | [`mteb/llm-eval-sts15`](https://huggingface.co/datasets/mteb/llm-eval-sts15) | `6283e03ad9bd` |
| LLMSTS16 | [`mteb/llm-eval-sts16`](https://huggingface.co/datasets/mteb/llm-eval-sts16) | `69862fce58ef` |
| LLMSTS17 | [`mteb/llm-eval-sts17`](https://huggingface.co/datasets/mteb/llm-eval-sts17) | `fe4f4e1b9fda` |
| LLMSTS22v2 | [`mteb/llm-eval-sts22_v2`](https://huggingface.co/datasets/mteb/llm-eval-sts22_v2) | `7a79fd41b024` |
| LLMSTSBenchmark | [`mteb/llm-eval-stsbenchmark`](https://huggingface.co/datasets/mteb/llm-eval-stsbenchmark) | `86bbaf4470f5` |

## PairClassification (4)

| Task | Hugging Face dataset | Pinned revision |
|---|---|---|
| LLMLegalBenchPC | [`mteb/llm-eval-legal_bench_pc`](https://huggingface.co/datasets/mteb/llm-eval-legal_bench_pc) | `a0217dc60ec5` |
| LLMRTE3PC | [`mteb/llm-eval-rte3`](https://huggingface.co/datasets/mteb/llm-eval-rte3) | `5d745bd9e435` |
| LLMSprintDuplicateQuestionsPC | [`mteb/llm-eval-sprint_duplicate_questions`](https://huggingface.co/datasets/mteb/llm-eval-sprint_duplicate_questions) | `d1c6be04a5f8` |
| LLMTwitterURLCorpusPC | [`mteb/llm-eval-twitter_url_corpus`](https://huggingface.co/datasets/mteb/llm-eval-twitter_url_corpus) | `741fbe5cf43a` |

## Retrieval (6)

| Task | Hugging Face dataset | Pinned revision |
|---|---|---|
| LLMAILAStatutes | [`mteb/llm-eval-aila-statutes`](https://huggingface.co/datasets/mteb/llm-eval-aila-statutes) | `a2acf12d293e` |
| LLMFQuADRetrieval | [`mteb/llm-eval-fquad`](https://huggingface.co/datasets/mteb/llm-eval-fquad) | `dc5443dbfad5` |
| LLMHC3FinanceRetrieval | [`mteb/llm-eval-hc3-finance`](https://huggingface.co/datasets/mteb/llm-eval-hc3-finance) | `8733760a6f3e` |
| LLMLegalBenchConsumerContractsQA | [`mteb/llm-eval-legalbench-consumer-contracts`](https://huggingface.co/datasets/mteb/llm-eval-legalbench-consumer-contracts) | `642870c78f65` |
| LLMPublicHealthQA | [`mteb/llm-eval-public-health-qa`](https://huggingface.co/datasets/mteb/llm-eval-public-health-qa) | `b05938525381` |
| LLMTwitterHjerneRetrieval | [`mteb/llm-eval-twitter-hjerne`](https://huggingface.co/datasets/mteb/llm-eval-twitter-hjerne) | `31f9b918c30e` |
