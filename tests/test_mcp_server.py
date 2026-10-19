"""Integration tests for FastMCP server tool registration and execution."""

from __future__ import annotations

from llm_linguistics.mcp_server import (
    mcp,
    morph_attach_josa,
    morph_normalize_for_rag,
    morph_verify_constraints,
    pragmatics_analyze_discourse,
    semantics_extract_logical_markers,
    semantics_verify_rule_consistency,
)
from llm_linguistics.morph import MorphConstraintSpec
from llm_linguistics.pragmatics import DialogueTurnInput
from llm_linguistics.semantics import NumericBound, PolicyRule


def test_mcp_entrypoints_execute_end_to_end() -> None:
    assert mcp.name == "llm-linguistics"

    c_res = morph_verify_constraints(
        text="가나다",
        constraints=MorphConstraintSpec(
            exact_syllable_count=3,
            batchim_rule="none",
            expected_choseong_sequence="ㄱㄴㄷ",
        ),
    )
    assert c_res.is_valid is True

    j_res = morph_attach_josa(word="인공지능", josa_pair="과/와")
    assert j_res.combined == "인공지능과"

    r_res = morph_normalize_for_rag(text="빠르게 걸었다")
    assert "걷다" in r_res.content_lemmas

    p_res = pragmatics_analyze_discourse(
        utterances=[DialogueTurnInput(speaker="A", text="내부적으로 검토해 보겠습니다.")]
    )
    assert p_res.utterances[0].illocutionary_force == "indirect_refusal_or_deferral"

    l_res = semantics_extract_logical_markers(
        sentences=["모든 사용자는 10일 이내에 인증해야 한다."]
    )
    assert l_res.sentences[0].deontic_modality == "MUST"

    z_res = semantics_verify_rule_consistency(
        rules=[
            PolicyRule(
                rule_id="R1",
                source_sentence="10일 이내 승인 필수",
                action="approve",
                modality="MUST",
                numeric_conditions=[NumericBound(variable="days", operator="<=", value=10)],
            )
        ]
    )
    assert z_res.is_consistent is True
