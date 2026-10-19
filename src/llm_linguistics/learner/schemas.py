"""Pydantic schemas for online linguistic learning and persistent state."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

UserWordTag = Literal["NNP", "NNG", "MAG", "IC"]

PragmaticCategory = Literal[
    "indirect_refusal_cue",
    "soft_urge_cue",
    "hedging",
    "apology_or_cushion",
    "direct_pressure_or_blame",
]


class LearnedWordEntry(BaseModel):
    """Domain neologism or compound noun learned from online corpus or feedback."""

    word: str = Field(min_length=2)
    tag: UserWordTag = "NNP"
    user_score: float = 10.0
    pos_score: float = 0.0
    frequency: int = 1
    source: Literal["corpus_mining", "explicit_feedback"] = "corpus_mining"


class LearnedPragmaticCue(BaseModel):
    """Pragmatic marker with Beta-Binomial posterior weight updated from online feedback."""

    phrase: str = Field(min_length=2)
    category: PragmaticCategory
    positive_count: int = Field(default=0, ge=0)
    negative_count: int = Field(default=0, ge=0)
    posterior_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Beta(pos+1, neg+1) posterior mean: (pos + 1) / (pos + neg + 2)",
    )

    @property
    def is_active(self) -> bool:
        """Return True if posterior evidence favors this cue."""
        return self.posterior_weight >= 0.55

    @property
    def is_suppressed(self) -> bool:
        """Return True if negative feedback suppresses even a static default cue."""
        return self.posterior_weight <= 0.35


class LearnedSemanticPattern(BaseModel):
    """Domain-specific deontic modality or exclusivity trigger learned online."""

    pattern: str = Field(min_length=2)
    modality: Literal["MUST", "MAY", "MUST_NOT", "NONE"] = "NONE"
    is_exclusive: bool = False


class OnlineKnowledgeState(BaseModel):
    """Persisted state of online-learned linguistic knowledge."""

    version: str = "0.2.0"
    user_words: dict[str, LearnedWordEntry] = Field(default_factory=dict)
    pragmatic_cues: dict[str, LearnedPragmaticCue] = Field(default_factory=dict)
    semantic_patterns: dict[str, LearnedSemanticPattern] = Field(default_factory=dict)


class CorpusMiningResult(BaseModel):
    """Summary of OOV domain words discovered and hot-loaded from a text batch."""

    newly_registered_words: list[LearnedWordEntry]
    skipped_known_or_low_score: int
    total_learned_words: int
