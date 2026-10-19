"""Pydantic schemas for Korean morphological and syllable-level tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from llm_linguistics.engine import VowelPolarity

BatchimRule = Literal["any", "none", "all", "only_sonorant"]

JosaPair = Literal[
    "은/는",
    "이/가",
    "을/를",
    "과/와",
    "으로/로",
    "아/야",
    "이나/나",
    "이란/란",
    "이에요/예요",
    "이라서/라서",
]


class SyllableInfo(BaseModel):
    """Detailed decomposition of a single Hangul syllable in the input text."""

    char_index: int = Field(description="0-based index in the input string")
    syllable_index: int = Field(description="0-based index among Hangul syllables only")
    char: str
    choseong: str
    jungseong: str
    jongseong: str | None
    has_batchim: bool
    is_rieul_batchim: bool
    vowel_polarity: VowelPolarity


class MorphConstraintSpec(BaseModel):
    """Declarative constraints for deterministic Korean syllable and Jamo validation."""

    exact_syllable_count: int | None = Field(
        default=None,
        ge=0,
        description="Required number of Hangul syllables",
    )
    exact_char_count_no_space: int | None = Field(
        default=None,
        ge=0,
        description="Required number of non-whitespace characters",
    )
    max_char_count_with_space: int | None = Field(
        default=None,
        ge=1,
        description="Maximum total character length including whitespace",
    )
    batchim_rule: BatchimRule = Field(
        default="any",
        description=(
            "'none': forbid coda (받침 금지); "
            "'all': require coda on every Hangul syllable; "
            "'only_sonorant': allow only ㄴ/ㄹ/ㅁ/ㅇ codas"
        ),
    )
    expected_choseong_sequence: str | None = Field(
        default=None,
        description="Expected initial consonant sequence across Hangul syllables (e.g. 'ㄱㄴㄷ')",
    )
    enforce_vowel_harmony_per_word: bool = Field(
        default=False,
        description=(
            "If True, each whitespace-delimited word must not mix positive and negative vowels"
        ),
    )


class ConstraintCheckResult(BaseModel):
    """Deterministic evaluation result of Korean character/syllable constraints."""

    is_valid: bool
    char_count_with_space: int
    char_count_no_space: int
    syllable_count: int
    violations: list[str]
    syllables: list[SyllableInfo]


class JosaAttachmentResult(BaseModel):
    """Result of deterministic Korean allomorphic postposition (조사) attachment."""

    word: str
    josa_pair: JosaPair
    selected_josa: str
    combined: str
    last_syllable: SyllableInfo
    reason: str


class MorphemeToken(BaseModel):
    """Single morphological token produced by Kiwi."""

    form: str
    tag: str
    lemma: str
    start: int
    length: int


class RagNormalizedOutput(BaseModel):
    """Normalized morphological representation for Korean Hybrid RAG and BM25 indexing."""

    surface_text: str
    morphemes: list[MorphemeToken]
    content_lemmas: list[str] = Field(
        description=(
            "Canonical lemmas of content words (nouns, verbs, adjectives, roots, foreign/numbers)"
        )
    )
    decomposed_compounds: dict[str, list[str]] = Field(
        description="Mapping from surface compound/derivative word to split root morphemes"
    )
    normalized_query: str = Field(
        description="Space-joined canonical content lemmas suitable for lexical search indexing"
    )
