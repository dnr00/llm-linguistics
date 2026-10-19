"""Tests for Korean pragmatic discourse analysis: speech levels, shifts, and speech acts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_linguistics.pragmatics import DialogueTurnInput, analyze_discourse


def test_speech_levels_and_politeness_downgrade() -> None:
    turns = [
        DialogueTurnInput(
            speaker="Client",
            text="제안해 주신 내용 잘 확인했습니다.",  # formal_high (하십시오체)
        ),
        DialogueTurnInput(
            speaker="Vendor",
            text="혹시 언제쯤 일정 공유 가능하실까요?",  # informal_high + -시- + soft urge
        ),
        DialogueTurnInput(
            speaker="Client",
            text="내부적으로 논의 후 추후에 검토해 보겠습니다.",  # formal_high + indirect refusal
        ),
        DialogueTurnInput(
            speaker="Client",
            text="글쎄요, 지금은 다소 어렵지 않을까 싶네요.",  # informal_high (downgraded)
        ),
        DialogueTurnInput(
            speaker="Client",
            text="그냥 이번 건은 없던 걸로 해.",  # informal_low (downgraded to 반말)
        ),
    ]
    res = analyze_discourse(turns)

    assert res.turn_count == 5
    assert res.utterances[0].speech_level == "formal_high"
    assert res.utterances[0].speaker_politeness_shift == "initial"

    assert res.utterances[1].speech_level == "informal_high"
    assert res.utterances[1].has_honorific_si is True
    assert res.utterances[1].illocutionary_force == "indirect_request_or_urge"

    assert res.utterances[2].speech_level == "formal_high"
    assert res.utterances[2].illocutionary_force == "indirect_refusal_or_deferral"

    assert res.utterances[3].speech_level == "informal_high"
    assert res.utterances[3].speaker_politeness_shift == "downgraded"
    assert res.utterances[3].illocutionary_force == "hedged_dissent_or_reservation"

    assert res.utterances[4].speech_level == "informal_low"
    assert res.utterances[4].speaker_politeness_shift == "downgraded"

    assert res.politeness_drop_turns == [3, 4]
    assert res.indirect_speech_act_turns == [1, 2, 3]


def test_asymmetry_and_remaining_speech_levels() -> None:
    turns = [
        DialogueTurnInput(speaker="Boss", text="자네가 이번 프로젝트를 한번 맡아주게."),
        DialogueTurnInput(speaker="Staff", text="네, 최선을 다해 수행하겠습니다."),
        DialogueTurnInput(speaker="Lead", text="이건 내가 직접 처리한다."),
        DialogueTurnInput(speaker="Memo", text="일정 확인 요망."),
    ]
    res = analyze_discourse(turns)

    assert res.utterances[0].speech_level == "neutral_mid"
    assert res.utterances[1].speech_level == "formal_high"
    assert res.utterances[2].speech_level == "formal_low"
    assert res.utterances[3].speech_level == "fragment"
    assert res.asymmetry_detected is True


def test_empty_utterances_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        analyze_discourse([])

    with pytest.raises(ValidationError):
        DialogueTurnInput(speaker="", text="안녕하세요.")


def test_self_evolving_pragmatics_reaction_bootstrapping_and_ngram_inference(
    tmp_path: pytest.TempPathFactory,
) -> None:
    import shutil
    from pathlib import Path

    from llm_linguistics.pragmatics import reset_corpus_engine_for_testing
    from llm_linguistics.pragmatics.corpus_learner import DEFAULT_CORPUS_PATH

    temp_dir = Path(str(tmp_path))
    temp_corpus = temp_dir / "pragmatics_corpus.jsonl"
    temp_weights = temp_dir / "pragmatics_weights.json"
    shutil.copyfile(DEFAULT_CORPUS_PATH, temp_corpus)

    engine = reset_corpus_engine_for_testing(corpus_path=temp_corpus, weights_path=temp_weights)
    engine.retrain_from_corpus()

    # Novel expression not in static lexicon nor seed corpus yet
    novel_probe = [
        DialogueTurnInput(speaker="Lead", text="그 프로젝트는 동결 상태로 넘기겠습니다.")
    ]
    before_res = analyze_discourse(novel_probe)
    assert before_res.utterances[0].illocutionary_force == "literal_declarative"

    # Ingest unlabelled dialogue where reaction signals disappointment/deferral acceptance
    unlabelled_dialogue = [
        DialogueTurnInput(
            speaker="Lead",
            text="이번 분기에는 그 프로젝트를 동결 상태로 넘기겠습니다.",
        ),
        DialogueTurnInput(
            speaker="Requester",
            text="네, 아쉽네요. 보류하시는 걸로 알겠습니다.",
        ),
    ]
    ingest_res = engine.ingest_dialogue_turns(unlabelled_dialogue)
    assert ingest_res.bootstrapped_from_reactions == 1
    assert ingest_res.appended_records == 1

    # Morphological N-gram model now recognizes '동결/NNG + 상태/NNG + 넘기/VV' as refusal
    after_res = analyze_discourse(novel_probe)
    assert after_res.utterances[0].illocutionary_force == "indirect_refusal_or_deferral"
    assert "corpus_ngram_match" in after_res.utterances[0].detected_markers

    # Reset back to default repo paths after test
    reset_corpus_engine_for_testing()

