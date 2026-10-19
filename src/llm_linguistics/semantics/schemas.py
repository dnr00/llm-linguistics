"""Pydantic schemas for Korean logical marker extraction and Z3 policy consistency verification."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DeonticModality = Literal["MUST", "MAY", "MUST_NOT", "NONE"]
NumericOperator = Literal["==", "!=", "<", "<=", ">", ">="]


class ExtractedNumericPhrase(BaseModel):
    """Structured numeric comparison extracted from a Korean sentence."""

    raw_phrase: str
    value: float
    unit: str
    bound_keyword: str
    operator: NumericOperator


class SentenceLogicalMarkers(BaseModel):
    """Scope-critical logical markers extracted from a single Korean sentence."""

    sentence_index: int
    sentence: str
    universal_quantifiers: list[str]
    existential_quantifiers: list[str]
    exclusive_markers: list[str]
    negations: list[str]
    is_double_negation: bool
    deontic_modality: DeonticModality
    conditional_connectives: list[str]
    numeric_phrases: list[ExtractedNumericPhrase]


class LogicalMarkerExtractionResult(BaseModel):
    """Aggregated logical marker extraction output across sentences."""

    sentence_count: int
    sentences: list[SentenceLogicalMarkers]
    double_negation_indices: list[int]
    exclusive_constraint_indices: list[int]


class NumericBound(BaseModel):
    """Single numeric inequality or equality constraint on a named variable."""

    variable: str = Field(min_length=1)
    operator: NumericOperator
    value: float


class PolicyRule(BaseModel):
    """Formalized deontic policy rule for Z3 satisfiability and conflict verification."""

    rule_id: str = Field(min_length=1)
    source_sentence: str = Field(min_length=1)
    action: str = Field(
        min_length=1,
        description="Canonical identifier of the regulated action (e.g. 'refund', 'data_export')",
    )
    modality: Literal["MUST", "MAY", "MUST_NOT"]
    boolean_conditions: dict[str, bool] = Field(default_factory=dict)
    numeric_conditions: list[NumericBound] = Field(default_factory=list)


class RuleConflictDetail(BaseModel):
    """Witnessed conflict between two policy rules over the same action."""

    rule_a_id: str
    rule_b_id: str
    action: str
    modality_a: str
    modality_b: str
    witness_assignment: dict[str, bool | float]
    explanation: str


class ConsistencyReport(BaseModel):
    """Z3 solver verification report for a set of policy rules."""

    is_consistent: bool
    unreachable_rule_ids: list[str] = Field(
        description="Rules whose own conditions are internally unsatisfiable (dead rules)"
    )
    conflicts: list[RuleConflictDetail]
    summary: str
