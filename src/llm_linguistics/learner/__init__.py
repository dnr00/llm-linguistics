"""Online linguistic adaptation store for OOV neologisms, pragmatic cues, and semantic patterns."""

from llm_linguistics.learner.online_fetcher import (
    html_or_xml_to_sentences,
    ingest_online_url,
)
from llm_linguistics.learner.schemas import (
    CorpusMiningResult,
    LearnedPragmaticCue,
    LearnedSemanticPattern,
    LearnedWordEntry,
    OnlineKnowledgeState,
    OnlineUrlIngestionResult,
    PragmaticCategory,
)
from llm_linguistics.learner.store import (
    OnlineLinguisticLearner,
    get_learner,
    reset_learner_for_testing,
)

__all__ = [
    "CorpusMiningResult",
    "LearnedPragmaticCue",
    "LearnedSemanticPattern",
    "LearnedWordEntry",
    "OnlineKnowledgeState",
    "OnlineLinguisticLearner",
    "OnlineUrlIngestionResult",
    "PragmaticCategory",
    "get_learner",
    "html_or_xml_to_sentences",
    "ingest_online_url",
    "reset_learner_for_testing",
]
