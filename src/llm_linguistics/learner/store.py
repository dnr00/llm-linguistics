"""Persistent online learner for Kiwi OOV words, Bayesian pragmatic cues, and semantic rules."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

from llm_linguistics.engine import get_kiwi, is_hangul_syllable
from llm_linguistics.learner.schemas import (
    CorpusMiningResult,
    LearnedPragmaticCue,
    LearnedSemanticPattern,
    LearnedWordEntry,
    OnlineKnowledgeState,
    PragmaticCategory,
    UserWordTag,
)

logger = logging.getLogger(__name__)

DEFAULT_STATE_ENV = "LLM_LINGUISTICS_STATE_PATH"
DEFAULT_STATE_FILE = "~/.llm_linguistics/learned_state.json"


def _compute_posterior_weight(pos: int, neg: int) -> float:
    """Compute Laplace-smoothed Beta(1, 1) posterior mean."""
    return round((pos + 1.0) / (pos + neg + 2.0), 4)


class OnlineLinguisticLearner:
    """Manages online linguistic adaptation and atomic JSON persistence."""

    def __init__(self, state_path: Path | str | None = None) -> None:
        raw_path = state_path or os.environ.get(DEFAULT_STATE_ENV, DEFAULT_STATE_FILE)
        self.state_path = Path(raw_path).expanduser().resolve()
        self.state = self._load_state()
        self._sync_words_to_kiwi()

    def _load_state(self) -> OnlineKnowledgeState:
        if not self.state_path.exists():
            return OnlineKnowledgeState()
        content = self.state_path.read_text(encoding="utf-8")
        return OnlineKnowledgeState.model_validate_json(content)

    def save_state(self) -> None:
        """Atomically persist current learned state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_suffix(".json.tmp")
        tmp_path.write_text(self.state.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp_path, self.state_path)

    def _sync_words_to_kiwi(self) -> None:
        kiwi = get_kiwi()
        for entry in self.state.user_words.values():
            kiwi.add_user_word(entry.word, entry.tag, entry.user_score)

    def register_domain_word(
        self,
        word: str,
        tag: UserWordTag = "NNP",
        user_score: float = 10.0,
    ) -> LearnedWordEntry:
        """Explicitly register a domain neologism or compound into Kiwi and persist it."""
        cleaned = word.strip()
        if len(cleaned) < 2:
            raise ValueError("Domain word must be at least 2 characters long.")

        entry = LearnedWordEntry(
            word=cleaned,
            tag=tag,
            user_score=user_score,
            pos_score=user_score,
            frequency=1,
            source="explicit_feedback",
        )
        get_kiwi().add_user_word(cleaned, tag, user_score)
        self.state.user_words[cleaned] = entry
        self.save_state()
        return entry

    def mine_corpus_oov_words(
        self,
        texts: list[str],
        min_cnt: int = 2,
        max_word_len: int = 10,
        min_pos_score: float = 0.2,
        user_score: float = 10.0,
    ) -> CorpusMiningResult:
        """Extract OOV multi-morpheme domain nouns from online texts and hot-load into Kiwi."""
        if not texts:
            raise ValueError("texts list must not be empty.")
        if min_cnt < 1:
            raise ValueError("min_cnt must be >= 1.")

        kiwi = get_kiwi()
        candidates = kiwi.extract_words(
            texts,
            min_cnt=min_cnt,
            max_word_len=max_word_len,
            min_score=0.0,
            pos_score=min_pos_score,
            lm_filter=False,
        )

        sent_spans: list[tuple[str, set[int], set[int]]] = []
        for s in texts:
            toks = kiwi.tokenize(s)
            starts = {t.start for t in toks}
            ends = {
                t.start + t.len
                for t in toks
                if t.tag.startswith(("NN", "XR", "XSN", "XPN", "NR")) or t.tag == "ETN"
            }
            sent_spans.append((s, starts, ends))

        trailing_forbidden_tags = {
            "EP",
            "EF",
            "EC",
            "ETM",
            "JKS",
            "JKC",
            "JKG",
            "JKO",
            "JKB",
            "JC",
            "VV",
            "VA",
            "VX",
            "VCP",
            "VCN",
            "MM",
            "Z_CODA",
        }
        internal_forbidden_tags = {
            "EP",
            "EF",
            "ETM",
            "VCP",
            "VCN",
            "Z_CODA",
            "JKS",
            "JKC",
            "JKG",
            "JKO",
            "JKB",
            "JX",
            "JC",
            "MM",
        }

        newly_registered: list[LearnedWordEntry] = []
        skipped = 0
        valid_candidates: list[tuple[str, int, float]] = []

        for cand in candidates:
            word_str = str(cand[0]).strip()
            freq = int(cand[2])
            p_score = float(cand[3])

            if len(word_str) < 3 or not all(is_hangul_syllable(ch) for ch in word_str):
                skipped += 1
                continue
            if p_score < min_pos_score or word_str in self.state.user_words:
                skipped += 1
                continue

            # Ensure candidate starts and ends on actual morpheme boundaries in the texts
            aligned_cnt = 0
            for s, starts, ends in sent_spans:
                idx = s.find(word_str)
                while idx != -1:
                    if idx in starts and (idx + len(word_str)) in ends:
                        aligned_cnt += 1
                    idx = s.find(word_str, idx + 1)
            if aligned_cnt < min_cnt:
                skipped += 1
                continue

            current_tokens = kiwi.tokenize(word_str)
            if len(current_tokens) < 2:
                skipped += 1
                continue
            tags = [t.tag for t in current_tokens]
            if (
                tags[-1] in trailing_forbidden_tags
                or any(t in internal_forbidden_tags for t in tags)
                or word_str.endswith(("으", "적", "만", "들"))
            ):
                skipped += 1
                continue

            valid_candidates.append((word_str, freq, p_score))

        for word_str, freq, p_score in valid_candidates:
            if any(
                word_str != other_w
                and word_str in other_w
                and other_f >= freq * 0.7
                for other_w, other_f, _ in valid_candidates
            ):
                skipped += 1
                continue

            kiwi.add_user_word(word_str, "NNP", user_score)
            entry = LearnedWordEntry(
                word=word_str,
                tag="NNP",
                user_score=user_score,
                pos_score=round(p_score, 4),
                frequency=freq,
                source="corpus_mining",
            )
            self.state.user_words[word_str] = entry
            newly_registered.append(entry)

        if newly_registered:
            self.save_state()

        return CorpusMiningResult(
            newly_registered_words=newly_registered,
            skipped_known_or_low_score=skipped,
            total_learned_words=len(self.state.user_words),
        )

    def record_pragmatic_feedback(
        self,
        phrase: str,
        category: PragmaticCategory,
        is_positive: bool = True,
    ) -> LearnedPragmaticCue:
        """Update Beta-Binomial posterior weight for a pragmatic marker from online feedback."""
        cleaned = phrase.strip()
        if len(cleaned) < 2:
            raise ValueError("Pragmatic cue phrase must be at least 2 characters long.")

        key = f"{category}:{cleaned}"
        existing = self.state.pragmatic_cues.get(key)
        pos = existing.positive_count if existing else 0
        neg = existing.negative_count if existing else 0

        if is_positive:
            pos += 1
        else:
            neg += 1

        updated = LearnedPragmaticCue(
            phrase=cleaned,
            category=category,
            positive_count=pos,
            negative_count=neg,
            posterior_weight=_compute_posterior_weight(pos, neg),
        )
        self.state.pragmatic_cues[key] = updated
        self.save_state()
        return updated

    def register_semantic_pattern(
        self,
        pattern: str,
        modality: Literal["MUST", "MAY", "MUST_NOT", "NONE"] = "NONE",
        is_exclusive: bool = False,
    ) -> LearnedSemanticPattern:
        """Register a domain-specific deontic modality or exclusivity phrase pattern."""
        cleaned = pattern.strip()
        if len(cleaned) < 2:
            raise ValueError("Semantic pattern must be at least 2 characters long.")
        if modality == "NONE" and not is_exclusive:
            raise ValueError("Pattern must specify a non-NONE modality or is_exclusive=True.")

        entry = LearnedSemanticPattern(
            pattern=cleaned,
            modality=modality,
            is_exclusive=is_exclusive,
        )
        self.state.semantic_patterns[cleaned] = entry
        self.save_state()
        return entry


_LEARNER_SINGLETON: OnlineLinguisticLearner | None = None


def get_learner() -> OnlineLinguisticLearner:
    """Return process-wide singleton OnlineLinguisticLearner."""
    global _LEARNER_SINGLETON
    if _LEARNER_SINGLETON is None:
        _LEARNER_SINGLETON = OnlineLinguisticLearner()
    return _LEARNER_SINGLETON


def reset_learner_for_testing(state_path: Path | str | None = None) -> OnlineLinguisticLearner:
    """Reinitialize the singleton learner with a specific state path for isolated testing."""
    global _LEARNER_SINGLETON
    _LEARNER_SINGLETON = OnlineLinguisticLearner(state_path=state_path)
    return _LEARNER_SINGLETON
