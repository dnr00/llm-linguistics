"""Integration tests for FastMCP server tool registration, flat args, and online learning."""

from __future__ import annotations

from pathlib import Path

from llm_linguistics.learner import reset_learner_for_testing
from llm_linguistics.mcp_server import (
    learn_domain_corpus,
    learn_domain_word,
    learn_pragmatic_feedback,
    learn_semantic_pattern,
    mcp,
    morph_attach_josa,
    morph_normalize_for_rag,
    morph_verify_constraints,
    pragmatics_analyze_discourse,
    semantics_extract_logical_markers,
    semantics_verify_rule_consistency,
)
from llm_linguistics.pragmatics import DialogueTurnInput
from llm_linguistics.semantics import NumericBound, PolicyRule


def test_mcp_entrypoints_execute_end_to_end(tmp_path: Path) -> None:
    reset_learner_for_testing(tmp_path / "mcp_state.json")
    assert mcp.name == "llm-linguistics"

    # Flat argument call (no nested MorphConstraintSpec required)
    c_res = morph_verify_constraints(
        text="가나다",
        exact_syllable_count=3,
        batchim_rule="none",
        expected_choseong_sequence="ㄱㄴㄷ",
    )
    assert c_res.is_valid is True

    j_res = morph_attach_josa(word="인공지능", josa_pair="과/와")
    assert j_res.combined == "인공지능과"

    w_res = learn_domain_word(word="하이퍼볼트그랜트", tag="NNP")
    assert w_res.word == "하이퍼볼트그랜트"

    r_res = morph_normalize_for_rag(text="하이퍼볼트그랜트를 빠르게 걸었다")
    assert "하이퍼볼트그랜트" in r_res.content_lemmas
    assert "걷다" in r_res.content_lemmas

    c_mine = learn_domain_corpus(
        texts=["온체인레이블러를 실행하고 온체인레이블러를 검증했다."] * 3,
        min_cnt=2,
        min_pos_score=0.2,
    )
    assert c_mine.total_learned_words >= 1

    f_res = learn_pragmatic_feedback(
        phrase="홀딩해 두겠",
        category="indirect_refusal_cue",
        is_positive=True,
    )
    assert f_res.is_active is True

    p_res = pragmatics_analyze_discourse(
        utterances=[DialogueTurnInput(speaker="A", text="일단 홀딩해 두겠습니다.")]
    )
    assert p_res.utterances[0].illocutionary_force == "indirect_refusal_or_deferral"

    s_pat = learn_semantic_pattern(
        pattern="즉시 상환을 요한다",
        modality="MUST",
        is_exclusive=False,
    )
    assert s_pat.modality == "MUST"

    l_res = semantics_extract_logical_markers(
        sentences=["기한 이익 상실 시 모든 채무는 즉시 상환을 요한다."]
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
