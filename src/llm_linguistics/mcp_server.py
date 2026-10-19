"""FastMCP server exposing Korean morphology, pragmatics, and formal semantics tools."""

from __future__ import annotations

from fastmcp import FastMCP

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
from llm_linguistics.pragmatics import (
    DialogueTurnInput,
    DiscourseAnalysisResult,
    analyze_discourse,
)
from llm_linguistics.semantics import (
    ConsistencyReport,
    LogicalMarkerExtractionResult,
    PolicyRule,
    extract_logical_markers,
    verify_rule_consistency,
)

mcp = FastMCP(
    name="llm-linguistics",
    instructions=(
        "Deterministic Korean linguistic middleware for LLM agents: "
        "(1) morph_* tools for exact Hangul syllable/Jamo constraint verification, "
        "allomorphic Josa attachment, and irregular predicate/compound normalization for RAG; "
        "(2) pragmatics_analyze_discourse for 6-tier Korean speech-level tracking, "
        "honorific shifts, and indirect speech-act detection; "
        "(3) semantics_* tools for scope/quantifier/negation extraction and Z3 theorem prover "
        "verification of policy/contract rule consistency."
    ),
)


@mcp.tool()
def morph_verify_constraints(
    text: str,
    constraints: MorphConstraintSpec,
) -> ConstraintCheckResult:
    """Verify Korean syllable count, length, batchim rules, choseong, and vowel harmony."""
    return verify_constraints(text=text, constraints=constraints)


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


def main() -> None:
    """Entrypoint for running the FastMCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
