"""Self-evolving Korean pragmatics corpus store, reaction bootstrapper, and N-gram learner."""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from llm_linguistics.engine import get_kiwi, normalize_kiwi_jamo
from llm_linguistics.pragmatics.schemas import DialogueTurnInput, IllocutionaryForce

REPO_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
DEFAULT_CORPUS_PATH = REPO_DATA_DIR / "pragmatics_corpus.jsonl"
DEFAULT_WEIGHTS_PATH = REPO_DATA_DIR / "pragmatics_weights.json"

TARGET_PRAGMATIC_CLASSES: tuple[IllocutionaryForce, ...] = (
    "indirect_refusal_or_deferral",
    "indirect_request_or_urge",
    "hedged_dissent_or_reservation",
    "direct_pressure_or_blame",
)

REFUSAL_REACTION_SIGNALS: tuple[str, ...] = (
    "아쉽네요",
    "어쩔 수 없",
    "다음 기회에",
    "보류하시는",
    "보류하겠",
    "내부 논의 결과 기다리",
    "다음 분기에",
)

URGE_REACTION_SIGNALS: tuple[str, ...] = (
    "바로 전달드리",
    "최대한 빨리 공유",
    "지금 바로 확인해서",
    "곧 회신드리",
    "오늘 중으로 바로",
)

DISSENT_REACTION_SIGNALS: tuple[str, ...] = (
    "우려하시는 부분",
    "수정안을 같이",
    "하루 더 검증해",
)

FEATURE_POS_TAGS: frozenset[str] = frozenset({
    "NNG", "NNP", "VV", "VA", "MAG", "IC",
})

STOP_MORPHEMES: frozenset[str] = frozenset({
    "이", "하", "있", "되", "것", "수", "들", "그", "나", "우리", "저", "네", "예",
    "확인", "내용", "제안", "말씀", "드리", "주", "일", "건", "번", "이번", "지금",
})


class PragmaticCorpusRecord(BaseModel):
    """Single dialogue turn record stored in the pragmatics training corpus."""

    speaker: str
    text: str
    next_reaction: str = ""
    label: IllocutionaryForce
    source: str = "online"


class PragmaticModelWeights(BaseModel):
    """Persisted morphological N-gram log-odds weights learned from accumulated corpus."""

    total_samples: int = 0
    class_counts: dict[str, int] = Field(default_factory=dict)
    feature_log_odds: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Mapping from class label -> {morph_ngram_feature: log_odds_weight}",
    )


class PragmaticCorpusStats(BaseModel):
    """Summary statistics and top discovered morphological features in the pragmatics engine."""

    corpus_path: str
    weights_path: str
    total_corpus_records: int
    class_distribution: dict[str, int]
    top_features_by_class: dict[str, list[tuple[str, float]]]


class DialogueIngestionResult(BaseModel):
    """Result of ingesting multi-turn dialogues into the self-evolving pragmatics corpus."""

    appended_records: int
    bootstrapped_from_reactions: int
    total_corpus_records: int
    top_newly_weighted_features: dict[str, list[tuple[str, float]]]


def extract_morph_features(text: str) -> list[str]:
    """Extract morphological content/pragmatic stem unigrams, bigrams, and trigrams."""
    kiwi = get_kiwi()
    tokens = kiwi.tokenize(text)
    filtered: list[str] = []

    for tok in tokens:
        base_tag = tok.tag.split("-")[0]
        if base_tag not in FEATURE_POS_TAGS:
            continue
        norm_form = normalize_kiwi_jamo(tok.form)
        if norm_form in STOP_MORPHEMES:
            continue
        filtered.append(f"{norm_form}/{base_tag}")

    features: list[str] = list(filtered)
    for i in range(len(filtered) - 1):
        features.append(f"{filtered[i]}+{filtered[i + 1]}")
    for i in range(len(filtered) - 2):
        features.append(f"{filtered[i]}+{filtered[i + 1]}+{filtered[i + 2]}")

    # Deduplicate per utterance while preserving order
    return list(dict.fromkeys(features))


def infer_weak_label_from_reaction(next_reaction: str) -> IllocutionaryForce | None:
    """Infer pragmatic speech-act label of utterance A from interlocutor reaction B."""
    if not next_reaction:
        return None
    if any(sig in next_reaction for sig in REFUSAL_REACTION_SIGNALS):
        return "indirect_refusal_or_deferral"
    if any(sig in next_reaction for sig in URGE_REACTION_SIGNALS):
        return "indirect_request_or_urge"
    if any(sig in next_reaction for sig in DISSENT_REACTION_SIGNALS):
        return "hedged_dissent_or_reservation"
    return None


class PragmaticCorpusEngine:
    """Manages pragmatics JSONL corpus accumulation and incremental N-gram log-odds training."""

    def __init__(
        self,
        corpus_path: Path | str | None = None,
        weights_path: Path | str | None = None,
    ) -> None:
        c_raw = corpus_path or os.environ.get("LLM_LINGUISTICS_CORPUS_PATH", DEFAULT_CORPUS_PATH)
        w_raw = weights_path or os.environ.get("LLM_LINGUISTICS_WEIGHTS_PATH", DEFAULT_WEIGHTS_PATH)
        self.corpus_path = Path(c_raw).expanduser().resolve()
        self.weights_path = Path(w_raw).expanduser().resolve()
        self.weights = self._load_or_train_weights()

    def load_records(self) -> list[PragmaticCorpusRecord]:
        if not self.corpus_path.exists():
            return []
        records: list[PragmaticCorpusRecord] = []
        for line in self.corpus_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                records.append(PragmaticCorpusRecord.model_validate_json(stripped))
        return records

    def _load_or_train_weights(self) -> PragmaticModelWeights:
        if self.weights_path.exists():
            return PragmaticModelWeights.model_validate_json(
                self.weights_path.read_text(encoding="utf-8")
            )
        if self.corpus_path.exists():
            return self.retrain_from_corpus()
        return PragmaticModelWeights()

    def retrain_from_corpus(self, min_feature_df: int = 1) -> PragmaticModelWeights:
        """Compute Dirichlet-smoothed log-odds weights for morphological N-grams across classes."""
        records = self.load_records()
        if not records:
            return PragmaticModelWeights()

        class_counts: Counter[str] = Counter()
        feature_class_counts: dict[str, Counter[str]] = defaultdict(Counter)
        feature_total_counts: Counter[str] = Counter()

        for rec in records:
            class_counts[rec.label] += 1
            feats = extract_morph_features(rec.text)
            for f in feats:
                feature_class_counts[rec.label][f] += 1
                feature_total_counts[f] += 1

        total_n = len(records)
        alpha = 0.5
        learned_log_odds: dict[str, dict[str, float]] = {}

        for cls in TARGET_PRAGMATIC_CLASSES:
            n_c = class_counts[cls]
            n_not_c = max(1, total_n - n_c)
            if n_c == 0:
                learned_log_odds[cls] = {}
                continue

            cls_map: dict[str, float] = {}
            for feat, tot_cnt in feature_total_counts.items():
                if tot_cnt < min_feature_df:
                    continue
                cnt_in_c = feature_class_counts[cls][feat]
                if cnt_in_c == 0:
                    continue
                cnt_in_not_c = tot_cnt - cnt_in_c

                # Smoothed log-odds ratio between class c and ~c
                p_c = (cnt_in_c + alpha) / (n_c + 2.0 * alpha)
                p_not_c = (cnt_in_not_c + alpha) / (n_not_c + 2.0 * alpha)
                log_odds = math.log(p_c / (1.0 - p_c)) - math.log(p_not_c / (1.0 - p_not_c))
                if log_odds > 0.4:
                    cls_map[feat] = round(log_odds, 4)

            learned_log_odds[cls] = cls_map

        weights = PragmaticModelWeights(
            total_samples=total_n,
            class_counts=dict(class_counts),
            feature_log_odds=learned_log_odds,
        )
        self.weights = weights
        self.save_weights()
        return weights

    def save_weights(self) -> None:
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.weights_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(self.weights.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_path, self.weights_path)

    def append_records(
        self,
        new_records: list[PragmaticCorpusRecord],
        retrain: bool = True,
    ) -> None:
        if not new_records:
            return
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        existing_texts = {r.text for r in self.load_records()}
        to_write = [r for r in new_records if r.text not in existing_texts]
        if not to_write:
            return

        with self.corpus_path.open("a", encoding="utf-8") as f:
            for rec in to_write:
                f.write(json.dumps(rec.model_dump(), ensure_ascii=False) + "\n")

        if retrain:
            self.retrain_from_corpus()

    def predict_from_ngram_weights(
        self,
        text: str,
        threshold: float = 2.5,
    ) -> tuple[IllocutionaryForce | None, float, list[str]]:
        """Score utterance using corpus-learned morphological N-gram log-odds weights."""
        if not self.weights.feature_log_odds:
            return None, 0.0, []

        feats = extract_morph_features(text)
        best_cls: IllocutionaryForce | None = None
        best_score = 0.0
        best_matched: list[str] = []

        for cls in TARGET_PRAGMATIC_CLASSES:
            cls_weights = self.weights.feature_log_odds.get(cls, {})
            matched = [f for f in feats if f in cls_weights]
            if len(matched) < 2:
                continue
            score = sum(cls_weights[f] for f in matched)
            if score > best_score:
                best_score = score
                best_cls = cls
                best_matched = matched

        if best_cls is None or best_score < threshold:
            return None, 0.0, []

        # Convert cumulative log-odds score to sigmoid probability in [0.55, 0.99]
        prob = round(1.0 / (1.0 + math.exp(-(best_score - threshold + 0.5))), 4)
        return best_cls, prob, best_matched

    def ingest_dialogue_turns(
        self,
        turns: list[DialogueTurnInput],
        explicit_labels: dict[int, IllocutionaryForce] | None = None,
    ) -> DialogueIngestionResult:
        """Ingest a multi-turn dialogue, bootstrap labels from reactions, and retrain weights."""
        if not turns:
            raise ValueError("turns list must not be empty.")

        explicit = explicit_labels or {}
        new_records: list[PragmaticCorpusRecord] = []
        bootstrapped_cnt = 0

        for idx, turn in enumerate(turns):
            next_text = turns[idx + 1].text if idx + 1 < len(turns) else ""
            label = explicit.get(idx)
            source = "explicit"
            if label is None:
                weak = infer_weak_label_from_reaction(next_text)
                if weak is not None:
                    label = weak
                    source = "reaction_bootstrap"
                    bootstrapped_cnt += 1

            if label is not None:
                new_records.append(
                    PragmaticCorpusRecord(
                        speaker=turn.speaker,
                        text=turn.text,
                        next_reaction=next_text,
                        label=label,
                        source=source,
                    )
                )

        self.append_records(new_records, retrain=True)
        stats = self.get_stats(top_k=5)
        return DialogueIngestionResult(
            appended_records=len(new_records),
            bootstrapped_from_reactions=bootstrapped_cnt,
            total_corpus_records=stats.total_corpus_records,
            top_newly_weighted_features=stats.top_features_by_class,
        )

    def get_stats(self, top_k: int = 8) -> PragmaticCorpusStats:
        top_by_cls: dict[str, list[tuple[str, float]]] = {}
        for cls, f_map in self.weights.feature_log_odds.items():
            sorted_feats = sorted(f_map.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            top_by_cls[cls] = sorted_feats

        return PragmaticCorpusStats(
            corpus_path=str(self.corpus_path),
            weights_path=str(self.weights_path),
            total_corpus_records=self.weights.total_samples,
            class_distribution=self.weights.class_counts,
            top_features_by_class=top_by_cls,
        )


_CORPUS_ENGINE_SINGLETON: PragmaticCorpusEngine | None = None


def get_corpus_engine() -> PragmaticCorpusEngine:
    """Return process-wide singleton PragmaticCorpusEngine."""
    global _CORPUS_ENGINE_SINGLETON
    if _CORPUS_ENGINE_SINGLETON is None:
        _CORPUS_ENGINE_SINGLETON = PragmaticCorpusEngine()
    return _CORPUS_ENGINE_SINGLETON


def reset_corpus_engine_for_testing(
    corpus_path: Path | str | None = None,
    weights_path: Path | str | None = None,
) -> PragmaticCorpusEngine:
    """Reinitialize singleton PragmaticCorpusEngine for isolated tests."""
    global _CORPUS_ENGINE_SINGLETON
    _CORPUS_ENGINE_SINGLETON = PragmaticCorpusEngine(
        corpus_path=corpus_path,
        weights_path=weights_path,
    )
    return _CORPUS_ENGINE_SINGLETON
