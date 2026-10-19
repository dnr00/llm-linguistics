"""FastMCP server exposing Korean morphology, pragmatics, formal semantics, and online learning."""

from __future__ import annotations

import os
from typing import Literal

from fastmcp import FastMCP

from llm_linguistics.autopilot import (
    AutopilotRunReport,
    AutopilotState,
    get_autopilot_store,
    run_autopilot_cycle,
)
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
    ChannelCollectionResult,
    DialogueIngestionResult,
    DialogueTurnInput,
    DiscourseAnalysisResult,
    PragmaticCorpusStats,
    PrivacyMode,
    analyze_discourse,
    collect_slack_channel,
    collect_telegram_updates,
    collect_web_dialogues,
    get_corpus_engine,
    ingest_telegram_export,
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
        "Deterministic Korean linguistic middleware and self-evolving adaptation server: "
        "(1) morph_* tools for exact Hangul syllable/Jamo constraint verification, "
        "allomorphic Josa attachment, and irregular predicate/compound normalization for RAG; "
        "(2) pragmatics_* tools for 6-tier Korean speech-level tracking, honorific shifts, "
        "morphological N-gram log-odds classification, and privacy-separated corpus growth; "
        "(3) collect_* tools for incremental dialogue harvesting from Slack, Telegram, and Web; "
        "(4) semantics_* tools for scope/quantifier/negation extraction, auto-formalization, "
        "and Z3 theorem prover verification of policy/contract rule consistency; "
        "(5) learn_* tools for online URL/RSS corpus ingestion, OOV word mining, Bayesian "
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
    auto_harvest_reactions: bool = True,
    privacy_mode: PrivacyMode = "features_only",
) -> DiscourseAnalysisResult:
    """Analyze 6-tier Korean speech levels, honorifics (-시-), shifts, and indirect speech acts."""
    return analyze_discourse(
        utterances=utterances,
        auto_harvest_reactions=auto_harvest_reactions,
        privacy_mode=privacy_mode,
    )


@mcp.tool()
def pragmatics_ingest_dialogues(
    utterances: list[DialogueTurnInput],
    privacy_mode: PrivacyMode = "features_only",
) -> DialogueIngestionResult:
    """Accumulate dialogue turns via reaction bootstrapping and retrain N-gram weights."""
    return get_corpus_engine().ingest_dialogue_turns(
        turns=utterances,
        privacy_mode=privacy_mode,
    )


@mcp.tool()
def pragmatics_get_corpus_stats(top_k: int = 8) -> PragmaticCorpusStats:
    """Return pragmatics corpus counts and top morphological N-gram log-odds features."""
    return get_corpus_engine().get_stats(top_k=top_k)


@mcp.tool()
def collect_slack_dialogues(
    channel_id: str,
    limit: int = 100,
    include_threads: bool = True,
    max_threads: int = 15,
    privacy_mode: PrivacyMode = "features_only",
) -> ChannelCollectionResult:
    """Pull incremental Slack channel & thread messages via SLACK_BOT_TOKEN with privacy."""
    return collect_slack_channel(
        channel_id=channel_id,
        limit=limit,
        include_threads=include_threads,
        max_threads=max_threads,
        privacy_mode=privacy_mode,
    )


@mcp.tool()
def collect_telegram_dialogues(
    limit: int = 100,
    privacy_mode: PrivacyMode = "features_only",
) -> ChannelCollectionResult:
    """Pull incremental Telegram Bot updates via TELEGRAM_BOT_TOKEN with privacy."""
    return collect_telegram_updates(limit=limit, privacy_mode=privacy_mode)


@mcp.tool()
def collect_telegram_export_json(
    messages: list[dict[str, object]],
    chat_label: str = "telegram_export",
    privacy_mode: PrivacyMode = "features_only",
) -> ChannelCollectionResult:
    """Ingest a Telegram Desktop/Telethon JSON message list with reply-chain linking."""
    return ingest_telegram_export(
        messages=messages,
        chat_label=chat_label,
        privacy_mode=privacy_mode,
    )


@mcp.tool()
def collect_web_dialogues_and_oov(
    url: str,
    privacy_mode: PrivacyMode = "public_full",
    also_mine_oov: bool = True,
) -> ChannelCollectionResult:
    """Fetch public web/RSS Korean dialogues, bootstrap pragmatic reactions, and mine OOV."""
    return collect_web_dialogues(
        url=url,
        privacy_mode=privacy_mode,
        also_mine_oov=also_mine_oov,
    )


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


@mcp.tool()
def autopilot_run_cycle(max_utterances: int = 20) -> AutopilotRunReport:
    """Run one bounded autonomous cycle: collect new Korean texts and distill via Claude Teacher."""
    return run_autopilot_cycle(max_utterances=max_utterances)


@mcp.tool()
def autopilot_get_status() -> AutopilotState:
    """Return cumulative autonomous LLM distillation stats and circuit-breaker state."""
    return get_autopilot_store().state


def main() -> None:
    """Entrypoint supporting both local stdio and remote streamable-http (e.g. Railway)."""
    transport = os.environ.get("MCP_TRANSPORT", "stdio").strip().lower()
    if transport in ("http", "streamable-http"):
        host = os.environ.get("HOST", "0.0.0.0")
        port = int(os.environ.get("PORT", "8000"))
        mcp.run(transport="streamable-http", host=host, port=port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
