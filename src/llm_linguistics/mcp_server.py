"""FastMCP server exposing Korean morphology, pragmatics, formal semantics, and online learning."""

from __future__ import annotations

from typing import Literal

from fastmcp import FastMCP

from llm_linguistics.learner import (
    CorpusMiningResult,
    LearnedPragmaticCue,
    LearnedSemanticPattern,
    LearnedWordEntry,
    OnlineUrlIngestionResult,
    PragmaticCategory,
    get_learner,
    ingest_online_url,
)
from llm_linguistics.learner.schemas import UserWordTag
from llm_linguistics.morph import (
    ConstraintCheckResult,
    JosaAttachmentResult,
    JosaPair,
    MorphConstraintSpec,
    RagNormalizedOutput,
    attach_josa,
    normalize_for_rag,
    verify_constraints,
)
from llm_linguistics.morph.schemas import BatchimRule
from llm_linguistics.pragmatics import (
    DialogueTurnInput,
    DiscourseAnalysisResult,
    analyze_discourse,
)
from llm_linguistics.semantics import (
    AutoVerificationReport,
    ConsistencyReport,
    LogicalMarkerExtractionResult,
    PolicyRule,
    auto_verify_korean_sentences,
    extract_logical_markers,
    verify_rule_consistency,
)

mcp = FastMCP(
    name="llm-linguistics",
    instructions=(
        "Deterministic Korean linguistic middleware and online adaptation server for LLM agents: "
        "(1) morph_* tools for exact Hangul syllable/Jamo constraint verification, "
        "allomorphic Josa attachment, and irregular predicate/compound normalization for RAG; "
        "(2) pragmatics_analyze_discourse for 6-tier Korean speech-level tracking, "
        "honorific shifts, and Bayesian indirect speech-act detection; "
        "(3) semantics_* tools for scope/quantifier/negation extraction, auto-formalization, "
        "and Z3 theorem prover verification of policy/contract rule consistency; "
        "(4) learn_* tools for online URL/RSS corpus ingestion, OOV word mining, Bayesian "
        "pragmatic cue feedback updates, and semantic modality pattern registration."
    ),
)


@mcp.tool()
def morph_verify_constraints(
    text: str,
    constraints: MorphConstraintSpec | None = None,
    exact_syllable_count: int | None = None,
    exact_char_count_no_space: int | None = None,
    max_char_count_with_space: int | None = None,
    batchim_rule: BatchimRule = "any",
    expected_choseong_sequence: str | None = None,
    enforce_vowel_harmony_per_word: bool = False,
) -> ConstraintCheckResult:
    """Verify Korean syllable count, length, batchim rules, choseong, and vowel harmony."""
    spec = constraints or MorphConstraintSpec(
        exact_syllable_count=exact_syllable_count,
        exact_char_count_no_space=exact_char_count_no_space,
        max_char_count_with_space=max_char_count_with_space,
        batchim_rule=batchim_rule,
        expected_choseong_sequence=expected_choseong_sequence,
        enforce_vowel_harmony_per_word=enforce_vowel_harmony_per_word,
    )
    return verify_constraints(text=text, constraints=spec)


@mcp.tool()
def morph_attach_josa(
    word: str,
    josa_pair: JosaPair,
) -> JosaAttachmentResult:
    """Select and attach the phonologically conditioned Korean postposition (은/는, 으로/로)."""
    return attach_josa(word=word, josa_pair=josa_pair)


@mcp.tool()
def morph_normalize_for_rag(text: str) -> RagNormalizedOutput:
    """Restore irregular Korean predicates to canonical lemmas and split compounds for RAG."""
    return normalize_for_rag(text=text)


@mcp.tool()
def pragmatics_analyze_discourse(
    utterances: list[DialogueTurnInput],
) -> DiscourseAnalysisResult:
    """Analyze 6-tier Korean speech levels, honorifics (-시-), shifts, and indirect speech acts."""
    return analyze_discourse(utterances=utterances)


@mcp.tool()
def semantics_extract_logical_markers(
    sentences: list[str],
) -> LogicalMarkerExtractionResult:
    """Extract quantifiers, exclusive particles, double negations, modality, and numeric bounds."""
    return extract_logical_markers(sentences=sentences)


@mcp.tool()
def semantics_verify_rule_consistency(
    rules: list[PolicyRule],
) -> ConsistencyReport:
    """Prove consistency or find concrete counterexample witnesses across policy rules via Z3."""
    return verify_rule_consistency(rules=rules)


@mcp.tool()
def semantics_auto_verify_sentences(
    sentences: list[str],
) -> AutoVerificationReport:
    """Auto-formalize raw Korean policy sentences into PolicyRules and verify consistency via Z3."""
    return auto_verify_korean_sentences(sentences=sentences)


@mcp.tool()
def learn_domain_corpus(
    texts: list[str],
    min_cnt: int = 2,
    min_pos_score: float = 0.2,
) -> CorpusMiningResult:
    """Discover OOV multi-morpheme domain nouns from online texts and hot-load into Kiwi."""
    return get_learner().mine_corpus_oov_words(
        texts=texts,
        min_cnt=min_cnt,
        min_pos_score=min_pos_score,
    )


@mcp.tool()
def learn_from_online_url(
    url: str,
    min_cnt: int = 2,
    min_pos_score: float = 0.2,
    timeout_sec: float = 10.0,
) -> OnlineUrlIngestionResult:
    """Fetch a public HTTP/HTTPS page or RSS feed, extract Korean sentences, and mine OOV words."""
    return ingest_online_url(
        url=url,
        min_cnt=min_cnt,
        min_pos_score=min_pos_score,
        timeout_sec=timeout_sec,
    )


@mcp.tool()
def learn_domain_word(
    word: str,
    tag: UserWordTag = "NNP",
    user_score: float = 10.0,
) -> LearnedWordEntry:
    """Register a domain neologism or compound noun into Kiwi and persist to online state."""
    return get_learner().register_domain_word(word=word, tag=tag, user_score=user_score)


@mcp.tool()
def learn_pragmatic_feedback(
    phrase: str,
    category: PragmaticCategory,
    is_positive: bool = True,
) -> LearnedPragmaticCue:
    """Update Beta-Binomial posterior weight for a pragmatic marker from online feedback."""
    return get_learner().record_pragmatic_feedback(
        phrase=phrase,
        category=category,
        is_positive=is_positive,
    )


@mcp.tool()
def learn_semantic_pattern(
    pattern: str,
    modality: Literal["MUST", "MAY", "MUST_NOT", "NONE"] = "NONE",
    is_exclusive: bool = False,
) -> LearnedSemanticPattern:
    """Register a domain-specific deontic modality or exclusivity trigger phrase online."""
    return get_learner().register_semantic_pattern(
        pattern=pattern,
        modality=modality,
        is_exclusive=is_exclusive,
    )


def main() -> None:
    """Entrypoint for running the FastMCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
