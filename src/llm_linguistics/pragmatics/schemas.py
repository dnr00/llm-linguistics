"""Pydantic schemas for Korean pragmatic discourse and speech-act analysis."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SpeechLevel = Literal[
    "formal_high",     # 하십시오체 (rank 5)
    "informal_high",   # 해요체 (rank 4)
    "neutral_mid",     # 하오/하게체 (rank 3)
    "formal_low",      # 해라체 (rank 2)
    "informal_low",    # 해체/반말 (rank 1)
    "fragment",        # 명사구/음슴체 종결 (rank 2)
]

PolitenessShift = Literal["initial", "maintained", "upgraded", "downgraded"]

SentenceType = Literal["declarative", "interrogative", "imperative", "proposative", "fragment"]

IllocutionaryForce = Literal[
    "indirect_refusal_or_deferral",
    "indirect_request_or_urge",
    "hedged_dissent_or_reservation",
    "direct_pressure_or_blame",
    "direct_directive",
    "proposal",
    "literal_declarative",
    "literal_interrogative",
    "literal_fragment",
]


class DialogueTurnInput(BaseModel):
    """Single dialogue turn input with speaker identifier and utterance text."""

    speaker: str = Field(min_length=1, description="Speaker identifier or name")
    text: str = Field(min_length=1, description="Utterance text in Korean")


class UtterancePragmatics(BaseModel):
    """Pragmatic and honorific analysis for a single Korean utterance."""

    turn_index: int
    speaker: str
    text: str
    speech_level: SpeechLevel
    politeness_rank: int = Field(ge=1, le=5)
    has_honorific_si: bool = Field(
        description="True if pre-final honorific '-시-' or lexical honorific verb is present"
    )
    politeness_shift_from_prev: PolitenessShift = Field(
        description="Politeness shift relative to the previous turn in the conversation"
    )
    speaker_politeness_shift: PolitenessShift = Field(
        description="Politeness shift relative to this same speaker's previous utterance"
    )
    sentence_type: SentenceType
    illocutionary_force: IllocutionaryForce
    detected_markers: dict[str, list[str]]
    final_ending_morphemes: list[str]
    pragmatic_note: str


class DiscourseAnalysisResult(BaseModel):
    """Full discourse-level pragmatic analysis across a sequence of Korean utterances."""

    turn_count: int
    utterances: list[UtterancePragmatics]
    asymmetry_detected: bool = Field(
        description=(
            "True if speakers use asymmetrically different politeness tiers (e.g. 존댓말 vs 반말)"
        )
    )
    politeness_drop_turns: list[int] = Field(
        description="Turn indices where a speaker downgraded their politeness tier"
    )
    indirect_speech_act_turns: list[int] = Field(
        description="Turn indices where surface sentence type diverges from illocutionary force"
    )
    summary_notes: list[str]
