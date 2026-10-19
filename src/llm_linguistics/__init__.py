"""Deterministic Korean morphology, pragmatics, semantics, and online adaptation toolkit."""

from llm_linguistics.learner import (
    OnlineLinguisticLearner,
    get_learner,
)
from llm_linguistics.morph import (
    MorphConstraintSpec,
    attach_josa,
    normalize_for_rag,
    verify_constraints,
)
from llm_linguistics.pragmatics import (
    DialogueTurnInput,
    analyze_discourse,
)
from llm_linguistics.semantics import (
    NumericBound,
    PolicyRule,
    extract_logical_markers,
    verify_rule_consistency,
)

__version__ = "0.2.0"

__all__ = [
    "DialogueTurnInput",
    "MorphConstraintSpec",
    "NumericBound",
    "OnlineLinguisticLearner",
    "PolicyRule",
    "__version__",
    "analyze_discourse",
    "attach_josa",
    "extract_logical_markers",
    "get_learner",
    "normalize_for_rag",
    "verify_constraints",
    "verify_rule_consistency",
]
