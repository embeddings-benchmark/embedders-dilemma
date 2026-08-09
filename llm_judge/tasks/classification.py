from typing import Literal

from mteb.tasks import (
    # Base MTEB tasks
    AmazonCounterfactualClassification,
    Banking77Classification,
    ImdbClassification,
    MTOPDomainClassification,
    MassiveIntentClassification,
    MassiveScenarioClassification,
    ToxicConversationsClassification,
    TweetSentimentExtractionClassification,
)
from pydantic import BaseModel, Field

from llm_judge.evaluators.llm_classification_evaluator import LLMClassificationEvaluator

BANKING_LABELS = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay",
    "atm_support", "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance",
    "card_arrival", "card_delivery_estimate", "card_linking",
    "card_not_working", "card_payment_fee_charged", "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised", "change_pin", "compromised_card",
    "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised",
    "disposable_card_limits", "edit_personal_details", "exchange_charge",
    "exchange_rate", "exchange_via_app", "extra_charge_on_statement",
    "failed_transfer", "fiat_currency_support", "get_disposable_virtual_card",
    "get_physical_card", "getting_spare_card", "getting_virtual_card",
    "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card",
    "passcode_forgotten", "pending_card_payment", "pending_cash_withdrawal",
    "pending_top_up", "pending_transfer", "pin_blocked",
    "receiving_money", "Refund_not_showing_up", "request_refund",
    "reverted_card_payment?", "supported_cards_and_currencies",
    "terminate_account", "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed",
    "top_up_limits", "top_up_reverted", "topping_up_by_card",
    "transaction_charged_twice", "transfer_fee_charged", "transfer_into_account",
    "transfer_not_received_by_recipient", "transfer_timing", "unable_to_verify_identity",
    "verify_my_identity", "verify_source_of_funds", "verify_top_up",
    "virtual_card_not_working", "visa_or_mastercard", "why_verify_identity",
    "wrong_amount_of_cash_received", "wrong_exchange_rate_for_cash_withdrawal",
]

INTENT_LABELS = [
    "alarm_query", "alarm_remove", "alarm_set", "audio_volume_down",
    "audio_volume_mute", "audio_volume_other", "audio_volume_up",
    "calendar_query", "calendar_remove", "calendar_set", "cooking_query",
    "cooking_recipe", "datetime_convert", "datetime_query", "email_addcontact",
    "email_query", "email_querycontact", "email_sendemail", "general_greet",
    "general_joke", "general_quirky", "iot_cleaning", "iot_coffee",
    "iot_hue_lightchange", "iot_hue_lightdim", "iot_hue_lightoff",
    "iot_hue_lighton", "iot_hue_lightup", "iot_wemo_off", "iot_wemo_on",
    "lists_createoradd", "lists_query", "lists_remove", "music_dislikeness",
    "music_likeness", "music_query", "music_settings", "news_query",
    "play_audiobook", "play_game", "play_music", "play_podcasts", "play_radio",
    "qa_currency", "qa_definition", "qa_factoid", "qa_maths", "qa_stock",
    "recommendation_events", "recommendation_locations", "recommendation_movies",
    "social_post", "social_query", "takeaway_order", "takeaway_query",
    "transport_query", "transport_taxi", "transport_ticket", "transport_traffic",
    "weather_query",
]

SCENARIO_LABELS = [
    "alarm", "audio", "calendar", "cooking", "datetime", "email", "general",
    "iot", "lists", "music", "news", "play", "qa", "recommendation", "social",
    "takeaway", "transport", "weather",
]


class BaseResponse(BaseModel):
    reasoning: str | None = Field(
        None, description="The reasoning behind the classification."
    )
    output: str

    @property
    def idx2label(self) -> dict[int, str]:
        raise NotImplementedError

    @property
    def output_class(self) -> int:
        label2idx = {v: k for k, v in self.idx2label.items()}
        return label2idx[self.output]


class ClassificationEvaluator(BaseModel):
    instruction: str
    response_model: type[BaseModel]

    def get_params(self) -> dict[str, str]:
        return {}


# ---------------------------------------------------------------------------
# Base class for LLM Classification
from mteb.abstasks.classification import AbsTaskClassification

class AbsTaskLLMClassification(AbsTaskClassification):
    """Base class to handle LLM classification evaluation and usage stats."""
    
    def evaluate(self, model, split="test", **kwargs):
        # Prevent zero-shot LLMs from re-running the same test dataset 10 times!
        self.n_experiments = 1 
        self.is_cross_validation = False
        
        # Inject the task name so the evaluator knows what to name its sample file
        import llm_judge.evaluators.llm_classification_evaluator as eval_module
        eval_module.LLMClassificationEvaluator.CURRENT_TASK_NAME = self.metadata.name
        
        # MTEB Classification runs the sklearn model and computes metrics
        scores = super().evaluate(model, split, **kwargs)
        
        import llm_judge.evaluators.llm_classification_evaluator as eval_module
        usage = getattr(eval_module.LLMClassificationEvaluator, "GLOBAL_USAGE", None)
        if usage:
            for hf_subset in scores:
                for target_k, val in usage.items():
                    scores[hf_subset][target_k] = val
                scores[hf_subset]["usage_stats"] = usage
                
            # Reset global tracker for the next task
            eval_module.LLMClassificationEvaluator.GLOBAL_USAGE = None

        return scores

# ===================================================================
# Primary Benchmark Tasks (8 tasks)
# ===================================================================

# ---------------------------------------------------------------------------
# 1. ImdbClassification — 2 labels (Sentiment)
# ---------------------------------------------------------------------------
class LLMImdbClassification(AbsTaskLLMClassification, ImdbClassification):
    metadata = ImdbClassification.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-imdb", "revision": "c7cd15a51954e6862a5d29508c8d8db61cb8f1e8"},
        "eval_splits": ["test"],
    })
    train_split = "test"

    class ImdbResponse(BaseResponse):
        output: Literal["positive", "negative"] = Field(
            description="The sentiment of the movie review. One of: positive, negative."
        )

        @property
        def idx2label(self) -> dict[int, str]:
            return {0: "negative", 1: "positive"}

    evaluator_model = ClassificationEvaluator(
        instruction=(
            "Classify the sentiment expressed in the given movie review text as 'positive' or 'negative'. "
            "Output json with fields 'reasoning' and 'output'."
        ),
        response_model=ImdbResponse,
    )
    evaluator = LLMClassificationEvaluator


# ---------------------------------------------------------------------------
# 2. Banking77Classification — 77 labels (Intent Detection)
# ---------------------------------------------------------------------------
class LLMBanking77Classification(AbsTaskLLMClassification, Banking77Classification):
    metadata = Banking77Classification.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-banking77", "revision": "2c82d499a4aab26ba0d98a4d37a8c838871d1bb1"},
        "eval_splits": ["test"],
    })
    train_split = "test"

    class Banking77Response(BaseResponse):
        output: str = Field(description="The banking customer service intent category.")

        @property
        def idx2label(self) -> dict[int, str]:
            return {i: label for i, label in enumerate(BANKING_LABELS)}

    evaluator_model = ClassificationEvaluator(
        instruction=(
            "Given an online banking query, find the corresponding intent from the list of categories. "
            "Output the exact category name. Output json with fields 'reasoning' and 'output'.\n\n"
            "Categories:\n" + ", ".join(BANKING_LABELS)
        ),
        response_model=Banking77Response,
    )
    evaluator = LLMClassificationEvaluator


# ---------------------------------------------------------------------------
# 3. AmazonCounterfactualClassification — 2 labels (Reasoning)
# ---------------------------------------------------------------------------
class LLMAmazonCounterfactualClassification(AbsTaskLLMClassification, AmazonCounterfactualClassification):
    metadata = AmazonCounterfactualClassification.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-amazon_counterfactual", "revision": "8df4b672d55146368ad9d82a498ac0f16b8f177f"},
        "eval_splits": ["test"],
        "eval_langs": {lang: AmazonCounterfactualClassification.metadata.eval_langs[lang] for lang in ["en", "de", "ja"]},
    })
    train_split = "test"

    class CounterfactualResponse(BaseResponse):
        output: Literal["counterfactual", "not-counterfactual"] = Field(
            description="Indicates whether the review contains a counterfactual statement. "
            "Respond with 'counterfactual' or 'not-counterfactual'."
        )

        @property
        def idx2label(self) -> dict[int, str]:
            return {0: "not-counterfactual", 1: "counterfactual"}

    evaluator_model = ClassificationEvaluator(
        instruction=(
            "Classify an Amazon customer review as either 'counterfactual' or 'not-counterfactual'. "
            "A counterfactual statement expresses what has not happened but could, would, or might have under different conditions. "
            "Output json with fields 'reasoning' and 'output'."
        ),
        response_model=CounterfactualResponse,
    )
    evaluator = LLMClassificationEvaluator


# ---------------------------------------------------------------------------
# 4. MTOPDomainClassification — 11 labels (Multilingual Domain Detection)
# ---------------------------------------------------------------------------
class LLMMTOPDomainClassification(AbsTaskLLMClassification, MTOPDomainClassification):
    metadata = MTOPDomainClassification.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-mtop_domain", "revision": "14315a9fd4305bf99cf0e400e06fb54a9b815f9c"},
        "eval_splits": ["test"],
        "eval_langs": {lang: MTOPDomainClassification.metadata.eval_langs[lang] for lang in ["en", "de", "fr"]},
    })
    train_split = "test"

    class MTOPDomainResponse(BaseResponse):
        output: Literal[
            "messaging", "calling", "event", "timer", "music", "weather", 
            "alarm", "people", "reminder", "recipes", "news"
        ] = Field(description="The intent domain of the utterance.")

        @property
        def idx2label(self) -> dict[int, str]:
            return {
                0: "messaging",
                1: "calling",
                2: "event",
                3: "timer",
                4: "music",
                5: "weather",
                6: "alarm",
                7: "people",
                8: "reminder",
                9: "recipes",
                10: "news"
            }

    evaluator_model = ClassificationEvaluator(
        instruction=(
            "Classify the intent domain of the given task-oriented utterance. "
            "Respond with one of: 'messaging', 'calling', 'event', 'music', 'news', 'weather', "
            "'timer', 'alarm', 'people', 'recipes', 'reminder'. "
            "Output json with fields 'reasoning' and 'output'."
        ),
        response_model=MTOPDomainResponse,
    )
    evaluator = LLMClassificationEvaluator


# ---------------------------------------------------------------------------
# 5. MassiveIntentClassification — 60 labels (Multilingual Intent)
# ---------------------------------------------------------------------------
class LLMMassiveIntentClassification(AbsTaskLLMClassification, MassiveIntentClassification):
    fast_loading = False  # custom dataset has no "default" subset — must load per-language config
    metadata = MassiveIntentClassification.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-massive_intent", "revision": "69fe18bc73414a7fc85905fda463a60da8ac10ce"},
        "eval_splits": ["test"],
        "eval_langs": {lang: MassiveIntentClassification.metadata.eval_langs[lang] for lang in ["en", "de", "fr", "ja"]},
    })
    train_split = "test"

    class MassiveIntentResponse(BaseResponse):
        output: str = Field(description="The user intent for the given utterance.")

        @property
        def idx2label(self) -> dict[int, str]:
            return {i: label for i, label in enumerate(INTENT_LABELS)}

    evaluator_model = ClassificationEvaluator(
        instruction=(
            "Given a user utterance, find the user intent from the specialized MASSIVE intent list. "
            "Output json with fields 'reasoning' and 'output'.\n\n"
            "Intents:\n" + ", ".join(INTENT_LABELS)
        ),
        response_model=MassiveIntentResponse,
    )
    evaluator = LLMClassificationEvaluator


# ---------------------------------------------------------------------------
# 6. MassiveScenarioClassification — 18 labels (Multilingual Scenario)
# ---------------------------------------------------------------------------
class LLMMassiveScenarioClassification(AbsTaskLLMClassification, MassiveScenarioClassification):
    fast_loading = False  # custom dataset has no "default" subset — must load per-language config
    metadata = MassiveScenarioClassification.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-massive_scenario", "revision": "ba1521289b080d967f820b289dd285f22fb968a8"},
        "eval_splits": ["test"],
        "eval_langs": {lang: MassiveScenarioClassification.metadata.eval_langs[lang] for lang in ["en", "de", "fr", "ja"]},
    })
    train_split = "test"

    class MassiveScenarioResponse(BaseResponse):
        output: str = Field(description="The scenario category.")

        @property
        def idx2label(self) -> dict[int, str]:
            return {i: label for i, label in enumerate(SCENARIO_LABELS)}

    evaluator_model = ClassificationEvaluator(
        instruction=(
            "Identify the user scenario for the given utterance. "
            "Output json with fields 'reasoning' and 'output'.\n\n"
            "Scenarios:\n" + ", ".join(SCENARIO_LABELS)
        ),
        response_model=MassiveScenarioResponse,
    )
    evaluator = LLMClassificationEvaluator


# ---------------------------------------------------------------------------
# 7. ToxicConversationsClassification — 2 labels (Safety)
# ---------------------------------------------------------------------------
class LLMToxicConversationsClassification(AbsTaskLLMClassification, ToxicConversationsClassification):
    metadata = ToxicConversationsClassification.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-toxic_conversations", "revision": "34eeb6105ca217433c04b207ea66810a2ff42625"},
        "eval_splits": ["test"],
    })
    train_split = "test"

    class ToxicResponse(BaseResponse):
        output: Literal["toxic", "not toxic"] = Field(description="Toxic or not toxic.")

        @property
        def idx2label(self) -> dict[int, str]:
            return {0: "not toxic", 1: "toxic"}

    evaluator_model = ClassificationEvaluator(
        instruction=(
            "Classify the given comment as 'toxic' or 'not toxic'. "
            "Output json with fields 'reasoning' and 'output'."
        ),
        response_model=ToxicResponse,
    )
    evaluator = LLMClassificationEvaluator


# ---------------------------------------------------------------------------
# 8. TweetSentimentExtractionClassification — 3 labels (Social Media Sentiment)
# ---------------------------------------------------------------------------
class LLMTweetSentimentExtractionClassification(AbsTaskLLMClassification, TweetSentimentExtractionClassification):
    metadata = TweetSentimentExtractionClassification.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-tweet_sentiment", "revision": "5e847f1f41ec9089cfdb7ea7b2852a23bd5ea7c8"},
        "eval_splits": ["test"],
    })
    train_split = "test"

    class TweetSentimentResponse(BaseResponse):
        output: Literal["positive", "negative", "neutral"] = Field(description="Sentiment.")

        @property
        def idx2label(self) -> dict[int, str]:
            return {0: "negative", 1: "neutral", 2: "positive"}

    evaluator_model = ClassificationEvaluator(
        instruction=(
            "Classify the sentiment of the tweet as 'positive', 'negative', or 'neutral'. "
            "Output json with fields 'reasoning' and 'output'."
        ),
        response_model=TweetSentimentResponse,
    )
    evaluator = LLMClassificationEvaluator



