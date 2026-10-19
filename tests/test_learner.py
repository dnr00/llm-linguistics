"""Tests for online linguistic adaptation: OOV mining, Bayesian pragmatic cues, and persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_linguistics.learner import OnlineLinguisticLearner, reset_learner_for_testing
from llm_linguistics.morph import normalize_for_rag
from llm_linguistics.pragmatics import DialogueTurnInput, analyze_discourse
from llm_linguistics.semantics import extract_logical_markers


def test_online_corpus_oov_mining_hotloads_into_kiwi(tmp_path: Path) -> None:
    state_file = tmp_path / "learned_state.json"
    learner = reset_learner_for_testing(state_file)

    corpus = [
        "우리는 제로슬립모드로 봇을 운영하고 제로슬립모드를 검증했다.",
        "이번 배포에서 제로슬립모드가 정상 작동하는지 확인했다.",
        "제로슬립모드로 전환한 뒤 지연 시간이 줄어들었다.",
        "라즈베리파이와 오픈소스를 활용해 네이버페이 결제를 연동했다.",
        "라즈베리파이에서 오픈소스 기반으로 네이버페이 모듈을 검증했다.",
        "라즈베리파이 환경의 오픈소스는 네이버페이와 호환된다.",
    ]
    res = learner.mine_corpus_oov_words(corpus, min_cnt=2, min_pos_score=0.2)
    learned_words = [w.word for w in res.newly_registered_words]
    assert "제로슬립모드" in learned_words
    assert "라즈베리파" not in learned_words
    assert "오픈소" not in learned_words
    assert "네이버페" not in learned_words
    assert "검증했" not in learned_words

    rag_out = normalize_for_rag("제로슬립모드를 활성화했다.")
    assert "제로슬립모드" in rag_out.content_lemmas

    # Verify persistence reload
    reloaded = OnlineLinguisticLearner(state_path=state_file)
    assert "제로슬립모드" in reloaded.state.user_words


def test_bayesian_pragmatic_feedback_activation_and_suppression(tmp_path: Path) -> None:
    state_file = tmp_path / "learned_state.json"
    learner = reset_learner_for_testing(state_file)

    utterance = [DialogueTurnInput(speaker="PM", text="네, 그 기능은 백로그에 넣어두겠습니다.")]

    # Before learning: treated as literal declarative
    before = analyze_discourse(utterance)
    assert before.utterances[0].illocutionary_force == "literal_declarative"

    # Positive feedback: posterior = (1 + 1) / (1 + 0 + 2) = 0.6667 >= 0.55 (active)
    cue = learner.record_pragmatic_feedback(
        phrase="백로그에 넣어두",
        category="indirect_refusal_cue",
        is_positive=True,
    )
    assert cue.posterior_weight == pytest.approx(0.6667, abs=1e-3)
    assert cue.is_active is True

    after = analyze_discourse(utterance)
    assert after.utterances[0].illocutionary_force == "indirect_refusal_or_deferral"
    assert after.utterances[0].confidence_score == pytest.approx(0.6667, abs=1e-3)

    # Suppress a static default cue ("검토해 보") via negative feedback
    for _ in range(3):
        sup_cue = learner.record_pragmatic_feedback(
            phrase="검토해 보",
            category="indirect_refusal_cue",
            is_positive=False,
        )
    assert sup_cue.is_suppressed is True

    suppressed_res = analyze_discourse(
        [DialogueTurnInput(speaker="A", text="제가 오늘 바로 검토해 보겠습니다.")]
    )
    assert suppressed_res.utterances[0].illocutionary_force == "literal_declarative"


def test_online_semantic_pattern_learning(tmp_path: Path) -> None:
    state_file = tmp_path / "learned_state.json"
    learner = reset_learner_for_testing(state_file)

    learner.register_semantic_pattern(
        pattern="사전 서면 승인을 득하",
        modality="MUST",
        is_exclusive=True,
    )

    res = extract_logical_markers(["을은 갑의 사전 서면 승인을 득한 경우에 진행한다."])
    s0 = res.sentences[0]
    assert s0.deontic_modality == "MUST"
    assert "사전 서면 승인을 득하" in s0.exclusive_markers


def test_learner_invalid_inputs_rejected(tmp_path: Path) -> None:
    learner = reset_learner_for_testing(tmp_path / "learned_state.json")

    with pytest.raises(ValueError, match="must not be empty"):
        learner.mine_corpus_oov_words([])

    with pytest.raises(ValueError, match="at least 2 characters"):
        learner.register_domain_word("가")

    with pytest.raises(ValueError, match="non-NONE modality or is_exclusive"):
        learner.register_semantic_pattern("아무패턴", modality="NONE", is_exclusive=False)


def test_html_or_xml_sentence_extraction_and_ssrf_guard() -> None:
    from llm_linguistics.learner import html_or_xml_to_sentences, ingest_online_url

    rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
      <item>
        <title>차세대 언어학 미들웨어 공개</title>
        <description><![CDATA[
          <p>우리는 제로슬립모드를 도입하여 응답 속도를 개선했습니다.</p>
        ]]></description>
      </item>
    </channel></rss>
    """
    sents = html_or_xml_to_sentences(rss_xml)
    assert len(sents) >= 2
    assert any("제로슬립모드" in s for s in sents)

    with pytest.raises(ValueError, match="Blocked or invalid hostname"):
        ingest_online_url("http://localhost:8080/feed.xml")

    with pytest.raises(ValueError, match="Private or reserved IP"):
        ingest_online_url("http://192.168.1.1/internal")

