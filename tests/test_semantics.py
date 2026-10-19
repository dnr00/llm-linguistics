"""Tests for Korean logical marker extraction and Z3-based policy rule consistency verification."""

from __future__ import annotations

import pytest

from llm_linguistics.semantics import (
    NumericBound,
    PolicyRule,
    extract_logical_markers,
    verify_rule_consistency,
)


def test_extract_logical_markers_quantifiers_negations_and_bounds() -> None:
    sentences = [
        "모든 회원은 가입 후 7일 이내에만 환불을 요청할 수 있다.",
        "일부 특가 상품은 어떠한 경우에도 환불이 불가하다.",
        "본 규정이 적용되지 않는 것은 아니다.",
        "확정된 계약 조건은 임의로 바꿀 수 없다.",
        "법령에 따라 결과를 통보하지 않을 수 없다.",
    ]
    res = extract_logical_markers(sentences)

    assert res.sentence_count == 5

    s0 = res.sentences[0]
    assert "모든" in s0.universal_quantifiers
    assert "만" in s0.exclusive_markers
    assert s0.deontic_modality == "MAY"
    assert len(s0.numeric_phrases) == 1
    assert s0.numeric_phrases[0].value == 7.0
    assert s0.numeric_phrases[0].operator == "<="

    s1 = res.sentences[1]
    assert "일부" in s1.existential_quantifiers
    assert s1.deontic_modality == "MUST_NOT"

    s2 = res.sentences[2]
    assert s2.is_double_negation is True

    s3 = res.sentences[3]
    assert s3.deontic_modality == "MUST_NOT"

    s4 = res.sentences[4]
    assert s4.is_double_negation is True
    assert s4.deontic_modality == "MUST"

    assert res.double_negation_indices == [2, 4]
    assert res.exclusive_constraint_indices == [0]


def test_z3_verify_rule_consistency_detects_conflict_with_witness() -> None:
    rules = [
        PolicyRule(
            rule_id="R1",
            source_sentence="결제 후 7일 이내인 모든 주문은 반드시 환불해야 한다.",
            action="refund",
            modality="MUST",
            numeric_conditions=[
                NumericBound(variable="days_since_purchase", operator="<=", value=7)
            ],
        ),
        PolicyRule(
            rule_id="R2",
            source_sentence="디지털 콘텐츠는 결제 후 30일 이내라도 환불이 불가하다.",
            action="refund",
            modality="MUST_NOT",
            boolean_conditions={"is_digital": True},
            numeric_conditions=[
                NumericBound(variable="days_since_purchase", operator="<=", value=30)
            ],
        ),
    ]
    report = verify_rule_consistency(rules)

    assert report.is_consistent is False
    assert len(report.conflicts) == 1
    conflict = report.conflicts[0]
    assert conflict.rule_a_id == "R1"
    assert conflict.rule_b_id == "R2"
    assert conflict.witness_assignment["is_digital"] is True
    assert float(conflict.witness_assignment["days_since_purchase"]) <= 7.0


def test_z3_verify_rule_consistency_passes_when_disjoint_and_detects_unreachable() -> None:
    consistent_rules = [
        PolicyRule(
            rule_id="R1",
            source_sentence="일반 상품은 7일 이내 환불해야 한다.",
            action="refund",
            modality="MUST",
            boolean_conditions={"is_digital": False},
            numeric_conditions=[
                NumericBound(variable="days_since_purchase", operator="<=", value=7)
            ],
        ),
        PolicyRule(
            rule_id="R2",
            source_sentence="디지털 상품은 환불이 불가하다.",
            action="refund",
            modality="MUST_NOT",
            boolean_conditions={"is_digital": True},
        ),
    ]
    ok_report = verify_rule_consistency(consistent_rules)
    assert ok_report.is_consistent is True
    assert ok_report.conflicts == []
    assert ok_report.unreachable_rule_ids == []

    unreachable_rules = [
        PolicyRule(
            rule_id="DEAD_1",
            source_sentence="결제 금액이 50000원 이상이면서 10000원 미만인 경우 환불한다.",
            action="refund",
            modality="MUST",
            numeric_conditions=[
                NumericBound(variable="amount", operator=">=", value=50000),
                NumericBound(variable="amount", operator="<", value=10000),
            ],
        )
    ]
    dead_report = verify_rule_consistency(unreachable_rules)
    assert dead_report.is_consistent is False
    assert dead_report.unreachable_rule_ids == ["DEAD_1"]


def test_semantics_empty_inputs_raise() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        extract_logical_markers([])

    with pytest.raises(ValueError, match="must not be empty"):
        verify_rule_consistency([])


def test_auto_verify_korean_sentences_from_raw_text() -> None:
    from llm_linguistics.semantics import auto_verify_korean_sentences

    raw_clauses = [
        "결제 후 7일 이내인 모든 주문은 반드시 환불해야 한다.",
        "디지털 콘텐츠는 결제 후 30일 이내라도 환불이 불가하다.",
    ]
    auto_rep = auto_verify_korean_sentences(raw_clauses)
    assert len(auto_rep.extracted_rules) == 2
    assert auto_rep.extracted_rules[0].action == "환불"
    assert auto_rep.extracted_rules[1].action == "환불"
    assert auto_rep.consistency_report.is_consistent is False
    assert len(auto_rep.consistency_report.conflicts) == 1
    witness = auto_rep.consistency_report.conflicts[0].witness_assignment
    assert witness["cond_디지털_콘텐츠"] is True
    assert float(witness["num_일"]) <= 7.0

