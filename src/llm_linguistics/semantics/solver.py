"""Deterministic Korean logical marker extractor and Z3 policy consistency verifier."""

from __future__ import annotations

import re
from typing import Any

from z3 import And, Bool, BoolRef, Not, RatNumRef, Real, Solver, is_true, sat

from llm_linguistics.engine import get_kiwi, normalize_kiwi_jamo
from llm_linguistics.semantics.schemas import (
    ConsistencyReport,
    DeonticModality,
    ExtractedNumericPhrase,
    LogicalMarkerExtractionResult,
    NumericBound,
    NumericOperator,
    PolicyRule,
    RuleConflictDetail,
    SentenceLogicalMarkers,
)

UNIVERSAL_QUANTIFIERS: frozenset[str] = frozenset({
    "모두", "모든", "전부", "전체", "일체", "각", "매", "전원",
})

EXISTENTIAL_QUANTIFIERS: frozenset[str] = frozenset({
    "일부", "어떤", "몇몇", "특정", "개별",
})

EXCLUSIVE_JX_FORMS: frozenset[str] = frozenset({"만", "뿐", "밖에"})
EXCLUSIVE_PHRASES: tuple[str, ...] = ("외에는", "제외하고", "한하여", "한해", "한정하여")

CONDITIONAL_PATTERNS: tuple[str, ...] = (
    "경우", "다면", "라면", "으면", "면", "한하여", "조건으로", "때에만",
)

BOUND_KEYWORD_TO_OP: dict[str, NumericOperator] = {
    "이상": ">=",
    "초과": ">",
    "이하": "<=",
    "이내": "<=",
    "미만": "<",
    "전": "<",
    "후": ">",
}

NUMERIC_BOUND_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[가-힣a-zA-Z%]+)?\s*"
    r"(?P<kw>이상|초과|이하|이내|미만)"
)


def _detect_deontic_modality(sentence: str) -> DeonticModality:
    if any(
        pat in sentence
        for pat in ("해서는 안", "할 수 없", "불가", "금지", "불허", "아니 된다", "안 된다")
    ):
        return "MUST_NOT"
    if any(
        pat in sentence
        for pat in ("해야", "하여야", "의무", "필수", "반드시", "요한다")
    ):
        return "MUST"
    if any(pat in sentence for pat in ("할 수 있", "가능하", "가능함", "허용")):
        return "MAY"
    return "NONE"


def _extract_numeric_phrases(sentence: str) -> list[ExtractedNumericPhrase]:
    phrases: list[ExtractedNumericPhrase] = []
    for match in NUMERIC_BOUND_RE.finditer(sentence):
        kw = match.group("kw")
        phrases.append(
            ExtractedNumericPhrase(
                raw_phrase=match.group(0).strip(),
                value=float(match.group("num")),
                unit=match.group("unit") or "",
                bound_keyword=kw,
                operator=BOUND_KEYWORD_TO_OP[kw],
            )
        )
    return phrases


def extract_logical_markers(sentences: list[str]) -> LogicalMarkerExtractionResult:
    """Extract quantifiers, exclusive particles, negations, modality, and numeric bounds."""
    if not sentences:
        raise ValueError("sentences list must not be empty.")

    kiwi = get_kiwi()
    items: list[SentenceLogicalMarkers] = []
    double_negation_indices: list[int] = []
    exclusive_constraint_indices: list[int] = []

    for idx, sent in enumerate(sentences):
        tokens = kiwi.tokenize(sent)
        universals: list[str] = []
        existentials: list[str] = []
        exclusives: list[str] = []
        negations: list[str] = []
        conditionals: list[str] = []

        for tok in tokens:
            norm_form = normalize_kiwi_jamo(tok.form)
            if norm_form in UNIVERSAL_QUANTIFIERS and tok.tag in ("MM", "NNG", "MAG"):
                universals.append(norm_form)
            if norm_form in EXISTENTIAL_QUANTIFIERS and tok.tag in ("MM", "NNG"):
                existentials.append(norm_form)
            if tok.tag == "JX" and norm_form in EXCLUSIVE_JX_FORMS:
                exclusives.append(norm_form)
            if (
                (tok.tag == "MAG" and norm_form in ("안", "못", "결코", "절대"))
                or (tok.tag in ("VX", "VCN") and norm_form in ("않", "못하", "아니"))
                or (tok.tag == "VA" and norm_form == "없")
                or norm_form in ("불가", "불가능", "금지", "불허")
            ):
                if norm_form not in ("결코", "절대"):
                    negations.append(norm_form)
            if tok.tag == "EC" and any(norm_form.endswith(p) for p in ("면", "다면", "라면")):
                conditionals.append(norm_form)

        for phrase in EXCLUSIVE_PHRASES:
            if phrase in sent and phrase not in exclusives:
                exclusives.append(phrase)

        for cond_word in ("경우", "한하여", "조건으로", "때에만"):
            if cond_word in sent and cond_word not in conditionals:
                conditionals.append(cond_word)

        is_double_neg = len(negations) >= 2
        if is_double_neg:
            double_negation_indices.append(idx)
        if exclusives:
            exclusive_constraint_indices.append(idx)

        items.append(
            SentenceLogicalMarkers(
                sentence_index=idx,
                sentence=sent,
                universal_quantifiers=universals,
                existential_quantifiers=existentials,
                exclusive_markers=exclusives,
                negations=negations,
                is_double_negation=is_double_neg,
                deontic_modality=_detect_deontic_modality(sent),
                conditional_connectives=conditionals,
                numeric_phrases=_extract_numeric_phrases(sent),
            )
        )

    return LogicalMarkerExtractionResult(
        sentence_count=len(items),
        sentences=items,
        double_negation_indices=double_negation_indices,
        exclusive_constraint_indices=exclusive_constraint_indices,
    )


def _build_numeric_expr(bound: NumericBound) -> BoolRef:
    var = Real(bound.variable)
    val = bound.value
    if bound.operator == "==":
        return var == val
    if bound.operator == "!=":
        return var != val
    if bound.operator == "<":
        return var < val
    if bound.operator == "<=":
        return var <= val
    if bound.operator == ">":
        return var > val
    return var >= val


def _build_rule_constraints(rule: PolicyRule) -> list[BoolRef]:
    exprs: list[BoolRef] = []
    for var_name, expected_bool in rule.boolean_conditions.items():
        b_var = Bool(var_name)
        exprs.append(b_var if expected_bool else Not(b_var))
    for num_bound in rule.numeric_conditions:
        exprs.append(_build_numeric_expr(num_bound))
    return exprs


def _are_modalities_incompatible(mod_a: str, mod_b: str) -> bool:
    pair = {mod_a, mod_b}
    return pair in ({"MUST", "MUST_NOT"}, {"MAY", "MUST_NOT"})


def _z3_val_to_python(z3_val: Any) -> bool | float:
    if isinstance(z3_val, BoolRef):
        return bool(is_true(z3_val))
    if isinstance(z3_val, RatNumRef):
        return float(z3_val.numerator_as_long()) / float(z3_val.denominator_as_long())
    text = str(z3_val).rstrip("?")
    try:
        return float(text)
    except ValueError:
        return bool(text == "True")


def verify_rule_consistency(rules: list[PolicyRule]) -> ConsistencyReport:
    """Prove consistency or find concrete counterexample witnesses across policy rules via Z3."""
    if not rules:
        raise ValueError("rules list must not be empty.")

    unreachable_rule_ids: list[str] = []
    rule_exprs: dict[str, list[BoolRef]] = {}

    for rule in rules:
        exprs = _build_rule_constraints(rule)
        rule_exprs[rule.rule_id] = exprs
        solver = Solver()
        if exprs:
            solver.add(And(*exprs))
        if solver.check() != sat:
            unreachable_rule_ids.append(rule.rule_id)

    conflicts: list[RuleConflictDetail] = []
    n = len(rules)
    for i in range(n):
        for j in range(i + 1, n):
            r_a = rules[i]
            r_b = rules[j]
            if r_a.action != r_b.action:
                continue
            if not _are_modalities_incompatible(r_a.modality, r_b.modality):
                continue
            if r_a.rule_id in unreachable_rule_ids or r_b.rule_id in unreachable_rule_ids:
                continue

            solver = Solver()
            combined = rule_exprs[r_a.rule_id] + rule_exprs[r_b.rule_id]
            if combined:
                solver.add(And(*combined))

            if solver.check() == sat:
                model = solver.model()
                witness: dict[str, bool | float] = {}
                for decl in model.decls():
                    witness[decl.name()] = _z3_val_to_python(model[decl])

                conflicts.append(
                    RuleConflictDetail(
                        rule_a_id=r_a.rule_id,
                        rule_b_id=r_b.rule_id,
                        action=r_a.action,
                        modality_a=r_a.modality,
                        modality_b=r_b.modality,
                        witness_assignment=witness,
                        explanation=(
                            f"Rules {r_a.rule_id!r} ({r_a.modality}) and "
                            f"{r_b.rule_id!r} ({r_b.modality}) both apply to action "
                            f"{r_a.action!r} and simultaneously hold under {witness}."
                        ),
                    )
                )

    is_consistent = len(unreachable_rule_ids) == 0 and len(conflicts) == 0
    if is_consistent:
        summary = f"All {len(rules)} rules are logically consistent and satisfiable."
    else:
        parts: list[str] = []
        if conflicts:
            parts.append(f"{len(conflicts)} deontic conflict(s) detected")
        if unreachable_rule_ids:
            parts.append(f"{len(unreachable_rule_ids)} unreachable rule(s): {unreachable_rule_ids}")
        summary = "Inconsistency detected: " + "; ".join(parts) + "."

    return ConsistencyReport(
        is_consistent=is_consistent,
        unreachable_rule_ids=unreachable_rule_ids,
        conflicts=conflicts,
        summary=summary,
    )
