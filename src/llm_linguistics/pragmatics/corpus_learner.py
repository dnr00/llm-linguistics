"""Self-evolving Korean pragmatics corpus store, reaction bootstrapper, and N-gram learner."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from llm_linguistics.engine import get_kiwi, normalize_kiwi_jamo
from llm_linguistics.pragmatics.schemas import DialogueTurnInput, IllocutionaryForce

REPO_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
DEFAULT_CORPUS_PATH = REPO_DATA_DIR / "pragmatics_corpus.jsonl"
DEFAULT_PRIVATE_CORPUS_PATH = Path.home() / ".llm_linguistics" / "private_corpus.jsonl"
DEFAULT_WEIGHTS_PATH = REPO_DATA_DIR / "pragmatics_weights.json"

PrivacyMode = Literal["public_full", "private_scrubbed", "features_only"]

TARGET_PRAGMATIC_CLASSES: tuple[IllocutionaryForce, ...] = (
    "indirect_refusal_or_deferral",
    "indirect_request_or_urge",
    "hedged_dissent_or_reservation",
    "direct_pressure_or_blame",
)

REFUSAL_REACTION_SIGNALS: tuple[str, ...] = (
    "아쉽네요",
    "아쉽지만",
    "어쩔 수 없",
    "다음 기회에",
    "보류하시는",
    "보류하겠",
    "내부 논의 결과 기다리",
    "다음 분기에",
    "안 되는군요",
    "어려우시군요",
    "일정 나오면 알려주",
    "나중에 다시 말씀",
)

URGE_REACTION_SIGNALS: tuple[str, ...] = (
    "바로 전달드리",
    "최대한 빨리 공유",
    "지금 바로 확인해서",
    "곧 회신드리",
    "오늘 중으로 바로",
    "제가 처리할게요",
    "바로 해둘게요",
    "금방 확인해 드릴게요",
    "늦어서 죄송",
    "지금 챙기겠습니다",
)

DISSENT_REACTION_SIGNALS: tuple[str, ...] = (
    "우려하시는 부분",
    "수정안을 같이",
    "하루 더 검증해",
    "리스크가 있겠네요",
    "다시 검토해 볼게요",
    "지적해주신 대로",
)

PRESSURE_REACTION_SIGNALS: tuple[str, ...] = (
    "죄송합니다 지금",
    "제 불찰",
    "변명의 여지",
    "시정하겠습니다",
    "다시는 이런 일",
)

PUBLIC_FEATURE_POS_TAGS: frozenset[str] = frozenset({
    "NNG", "NNP", "VV", "VA", "MAG", "IC",
})

PRIVATE_FEATURE_POS_TAGS: frozenset[str] = frozenset({
    "NNG", "VV", "VA", "MAG", "IC",
})

STOP_MORPHEMES: frozenset[str] = frozenset({
    "이", "하", "있", "되", "것", "수", "들", "그", "나", "우리", "저", "네", "예",
    "확인", "내용", "제안", "말씀", "드리", "주", "일", "건", "번", "이번", "지금",
    "멘션", "이메일", "전화번호", "보안토큰", "핸들", "URL",
})

_SLACK_MENTION_RE = re.compile(r"<[@#][A-Z0-9]+(?:\|[^>]+)?>")
_URL_RE = re.compile(r"https?://\S+")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(r"(?:01[016789]|02|0[3-6][1-5])-?\d{3,4}-?\d{4}")
_SECRET_TOKEN_RE = re.compile(r"(?:0x[a-fA-F0-9]{16,}|xox[baprs]-[a-zA-Z0-9-]+)")
_HANDLE_RE = re.compile(r"(?<!\w)@[a-zA-Z0-9_]{3,}")


def scrub_pii(text: str) -> str:
    """Redact mentions, URLs, emails, phone numbers, tokens, and handles from raw text."""
    cleaned = _SLACK_MENTION_RE.sub("[멘션]", text)
    cleaned = _URL_RE.sub("[URL]", cleaned)
    cleaned = _EMAIL_RE.sub("[이메일]", cleaned)
    cleaned = _PHONE_RE.sub("[전화번호]", cleaned)
    cleaned = _SECRET_TOKEN_RE.sub("[보안토큰]", cleaned)
    cleaned = _HANDLE_RE.sub("[핸들]", cleaned)
    return cleaned.strip()


def compute_utterance_hash(text: str) -> str:
    """Compute deterministic SHA-256 fingerprint of PII-scrubbed utterance text."""
    normalized = re.sub(r"\s+", " ", scrub_pii(text)).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


class PragmaticCorpusRecord(BaseModel):
    """Single dialogue turn record stored in the pragmatics training corpus."""

    speaker: str
    text: str = ""
    next_reaction: str = ""
    label: IllocutionaryForce
    source: str = "online"
    utterance_hash: str = ""
    morph_features: list[str] = Field(default_factory=list)


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
    private_corpus_path: str
    weights_path: str
    public_corpus_records: int
    private_corpus_records: int
    total_corpus_records: int
    class_distribution: dict[str, int]
    top_features_by_class: dict[str, list[tuple[str, float]]]


class DialogueIngestionResult(BaseModel):
    """Result of ingesting multi-turn dialogues into the self-evolving pragmatics corpus."""

    appended_records: int
    bootstrapped_from_reactions: int
    privacy_mode: PrivacyMode
    target_store: str
    total_corpus_records: int
    top_newly_weighted_features: dict[str, list[tuple[str, float]]]


def extract_morph_features(text: str, allow_proper_nouns: bool = True) -> list[str]:
    """Extract morphological content/pragmatic stem unigrams, bigrams, and trigrams."""
    cleaned = scrub_pii(text)
    if not cleaned:
        return []
    kiwi = get_kiwi()
    tokens = kiwi.tokenize(cleaned)
    allowed_tags = PUBLIC_FEATURE_POS_TAGS if allow_proper_nouns else PRIVATE_FEATURE_POS_TAGS
    filtered: list[str] = []

    for tok in tokens:
        base_tag = tok.tag.split("-")[0]
        if base_tag not in allowed_tags:
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
    if any(sig in next_reaction for sig in PRESSURE_REACTION_SIGNALS):
        return "direct_pressure_or_blame"
    return None


class PragmaticCorpusEngine:
    """Manages public/private pragmatics JSONL corpora and incremental N-gram training."""

    def __init__(
        self,
        corpus_path: Path | str | None = None,
        weights_path: Path | str | None = None,
        private_corpus_path: Path | str | None = None,
    ) -> None:
        c_raw = corpus_path or os.environ.get(
            "LLM_LINGUISTICS_PRAGMATICS_CORPUS_PATH",
            os.environ.get("LLM_LINGUISTICS_CORPUS_PATH", DEFAULT_CORPUS_PATH),
        )
        p_raw = private_corpus_path or os.environ.get(
            "LLM_LINGUISTICS_PRIVATE_CORPUS_PATH",
            DEFAULT_PRIVATE_CORPUS_PATH,
        )
        w_raw = weights_path or os.environ.get(
            "LLM_LINGUISTICS_PRAGMATICS_WEIGHTS_PATH",
            os.environ.get("LLM_LINGUISTICS_WEIGHTS_PATH", DEFAULT_WEIGHTS_PATH),
        )
        self.corpus_path = Path(c_raw).expanduser().resolve()
        self.private_corpus_path = Path(p_raw).expanduser().resolve()
        self.weights_path = Path(w_raw).expanduser().resolve()
        self.weights = self._load_or_train_weights()

    def _read_jsonl(self, path: Path) -> list[PragmaticCorpusRecord]:
        if not path.exists():
            return []
        records: list[PragmaticCorpusRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                records.append(PragmaticCorpusRecord.model_validate_json(stripped))
        return records

    def load_public_records(self) -> list[PragmaticCorpusRecord]:
        return self._read_jsonl(self.corpus_path)

    def load_private_records(self) -> list[PragmaticCorpusRecord]:
        return self._read_jsonl(self.private_corpus_path)

    def load_records(self) -> list[PragmaticCorpusRecord]:
        """Load combined public and private corpus records."""
        return self.load_public_records() + self.load_private_records()

    def _load_or_train_weights(self) -> PragmaticModelWeights:
        if self.weights_path.exists():
            loaded = PragmaticModelWeights.model_validate_json(
                self.weights_path.read_text(encoding="utf-8")
            )
            current_records = self.load_records()
            if current_records and loaded.total_samples != len(current_records):
                return self.retrain_from_corpus()
            return loaded
        if self.corpus_path.exists() or self.private_corpus_path.exists():
            return self.retrain_from_corpus()
        return PragmaticModelWeights()

    def retrain_from_corpus(self, min_feature_df: int = 1) -> PragmaticModelWeights:
        """Compute Dirichlet-smoothed log-odds weights for morphological N-grams across classes."""
        pub_records = self.load_public_records()
        priv_records = self.load_private_records()
        total_n = len(pub_records) + len(priv_records)
        if total_n == 0:
            return PragmaticModelWeights()

        class_counts: Counter[str] = Counter()
        feature_class_counts: dict[str, Counter[str]] = defaultdict(Counter)
        feature_total_counts: Counter[str] = Counter()

        for rec in pub_records:
            class_counts[rec.label] += 1
            feats = (
                rec.morph_features
                if rec.morph_features
                else extract_morph_features(rec.text, allow_proper_nouns=True)
            )
            for f in feats:
                feature_class_counts[rec.label][f] += 1
                feature_total_counts[f] += 1

        for rec in priv_records:
            class_counts[rec.label] += 1
            raw_feats = (
                rec.morph_features
                if rec.morph_features
                else extract_morph_features(rec.text, allow_proper_nouns=False)
            )
            feats = [f for f in raw_feats if "/NNP" not in f]
            for f in feats:
                feature_class_counts[rec.label][f] += 1
                feature_total_counts[f] += 1

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

    def _existing_fingerprints(self) -> set[str]:
        seen: set[str] = set()
        for r in self.load_records():
            if r.utterance_hash:
                seen.add(r.utterance_hash)
            if r.text:
                seen.add(compute_utterance_hash(r.text))
                seen.add(r.text)
        return seen

    def append_records(
        self,
        new_records: list[PragmaticCorpusRecord],
        privacy_mode: PrivacyMode = "public_full",
        retrain: bool = True,
    ) -> int:
        if not new_records:
            return 0

        existing_fps = self._existing_fingerprints()
        to_write: list[PragmaticCorpusRecord] = []

        for rec in new_records:
            u_hash = rec.utterance_hash or compute_utterance_hash(rec.text)
            if u_hash in existing_fps or (rec.text and rec.text in existing_fps):
                continue

            if privacy_mode in ("features_only", "private_scrubbed"):
                raw_feats = rec.morph_features or extract_morph_features(
                    rec.text, allow_proper_nouns=False
                )
                feats = [f for f in raw_feats if "/NNP" not in f]
                if not feats:
                    continue
                existing_fps.add(u_hash)
                sanitized = PragmaticCorpusRecord(
                    speaker="anon",
                    text="" if privacy_mode == "features_only" else scrub_pii(rec.text),
                    next_reaction=(
                        "" if privacy_mode == "features_only" else scrub_pii(rec.next_reaction)
                    ),
                    label=rec.label,
                    source=rec.source,
                    utterance_hash=u_hash,
                    morph_features=feats,
                )
            else:
                existing_fps.add(u_hash)
                sanitized = PragmaticCorpusRecord(
                    speaker=rec.speaker,
                    text=scrub_pii(rec.text),
                    next_reaction=scrub_pii(rec.next_reaction),
                    label=rec.label,
                    source=rec.source,
                    utterance_hash=u_hash,
                    morph_features=rec.morph_features,
                )
            to_write.append(sanitized)

        if not to_write:
            return 0

        target_path = (
            self.corpus_path if privacy_mode == "public_full" else self.private_corpus_path
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("a", encoding="utf-8") as f:
            for rec in to_write:
                dumped = json.dumps(rec.model_dump(exclude_defaults=True), ensure_ascii=False)
                f.write(dumped + "\n")

        if privacy_mode != "public_full":
            try:
                os.chmod(target_path, 0o600)
            except OSError:
                pass

        if retrain:
            self.retrain_from_corpus()
        return len(to_write)

    def predict_from_ngram_weights(
        self,
        text: str,
        threshold: float = 2.5,
    ) -> tuple[IllocutionaryForce | None, float, list[str]]:
        """Score utterance using corpus-learned morphological N-gram log-odds weights."""
        if not self.weights.feature_log_odds:
            return None, 0.0, []

        feats = extract_morph_features(text, allow_proper_nouns=True)
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

        prob = round(1.0 / (1.0 + math.exp(-(best_score - threshold + 0.5))), 4)
        return best_cls, prob, best_matched

    def ingest_dialogue_sessions(
        self,
        sessions: list[list[DialogueTurnInput]],
        explicit_labels: dict[tuple[int, int], IllocutionaryForce] | None = None,
        privacy_mode: PrivacyMode = "public_full",
        source_tag: str = "online",
    ) -> DialogueIngestionResult:
        """Ingest multiple independent dialogue sessions without cross-boundary pairing."""
        if not sessions or all(not s for s in sessions):
            raise ValueError("sessions list must not be empty.")

        explicit = explicit_labels or {}
        candidate_records: list[PragmaticCorpusRecord] = []
        bootstrapped_cnt = 0

        for s_idx, turns in enumerate(sessions):
            for idx, turn in enumerate(turns):
                has_next = idx + 1 < len(turns)
                next_turn = turns[idx + 1] if has_next else None
                next_text = next_turn.text if next_turn is not None else ""
                is_different_speaker = (
                    next_turn is not None and next_turn.speaker != turn.speaker
                )

                label = explicit.get((s_idx, idx))
                source = source_tag
                if label is None and is_different_speaker:
                    weak = infer_weak_label_from_reaction(next_text)
                    if weak is not None:
                        label = weak
                        source = f"{source_tag}:reaction_bootstrap"
                        bootstrapped_cnt += 1

                if label is not None:
                    candidate_records.append(
                        PragmaticCorpusRecord(
                            speaker=turn.speaker,
                            text=turn.text,
                            next_reaction=next_text if is_different_speaker else "",
                            label=label,
                            source=source,
                        )
                    )

        written_cnt = self.append_records(
            candidate_records,
            privacy_mode=privacy_mode,
            retrain=True,
        )
        stats = self.get_stats(top_k=5)
        target_store = (
            str(self.corpus_path)
            if privacy_mode == "public_full"
            else str(self.private_corpus_path)
        )
        return DialogueIngestionResult(
            appended_records=written_cnt,
            bootstrapped_from_reactions=bootstrapped_cnt,
            privacy_mode=privacy_mode,
            target_store=target_store,
            total_corpus_records=stats.total_corpus_records,
            top_newly_weighted_features=stats.top_features_by_class,
        )

    def ingest_dialogue_turns(
        self,
        turns: list[DialogueTurnInput],
        explicit_labels: dict[int, IllocutionaryForce] | None = None,
        privacy_mode: PrivacyMode = "public_full",
        source_tag: str = "online",
    ) -> DialogueIngestionResult:
        """Ingest a single multi-turn dialogue session and retrain weights."""
        if not turns:
            raise ValueError("turns list must not be empty.")
        mapped_labels = (
            {(0, k): v for k, v in explicit_labels.items()} if explicit_labels else None
        )
        return self.ingest_dialogue_sessions(
            sessions=[turns],
            explicit_labels=mapped_labels,
            privacy_mode=privacy_mode,
            source_tag=source_tag,
        )

    def get_stats(self, top_k: int = 8) -> PragmaticCorpusStats:
        top_by_cls: dict[str, list[tuple[str, float]]] = {}
        for cls, f_map in self.weights.feature_log_odds.items():
            sorted_feats = sorted(f_map.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            top_by_cls[cls] = sorted_feats

        pub_cnt = len(self.load_public_records())
        priv_cnt = len(self.load_private_records())
        return PragmaticCorpusStats(
            corpus_path=str(self.corpus_path),
            private_corpus_path=str(self.private_corpus_path),
            weights_path=str(self.weights_path),
            public_corpus_records=pub_cnt,
            private_corpus_records=priv_cnt,
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
    private_corpus_path: Path | str | None = None,
) -> PragmaticCorpusEngine:
    """Reinitialize singleton PragmaticCorpusEngine for isolated tests."""
    global _CORPUS_ENGINE_SINGLETON
    _CORPUS_ENGINE_SINGLETON = PragmaticCorpusEngine(
        corpus_path=corpus_path,
        weights_path=weights_path,
        private_corpus_path=private_corpus_path,
    )
    return _CORPUS_ENGINE_SINGLETON
