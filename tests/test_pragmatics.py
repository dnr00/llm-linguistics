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


def test_privacy_separation_features_only_and_pii_scrubbing(
    tmp_path: pytest.TempPathFactory,
) -> None:
    import shutil
    from pathlib import Path

    from llm_linguistics.pragmatics import reset_corpus_engine_for_testing, scrub_pii
    from llm_linguistics.pragmatics.corpus_learner import DEFAULT_CORPUS_PATH

    raw_pii = (
        "<@U12345|홍길동>님, @sec_handle 문의하신 010-1234-5678 연락처와 "
        "user@company.co.kr 및 https://internal.corp/doc 토큰 xoxb-1234-abcd 확인 바랍니다."
    )
    scrubbed = scrub_pii(raw_pii)
    assert "홍길동" not in scrubbed
    assert "010-1234-5678" not in scrubbed
    assert "user@company.co.kr" not in scrubbed
    assert "https://internal.corp" not in scrubbed
    assert "xoxb-1234-abcd" not in scrubbed
    assert "@sec_handle" not in scrubbed

    temp_dir = Path(str(tmp_path))
    pub_corpus = temp_dir / "public.jsonl"
    priv_corpus = temp_dir / "private.jsonl"
    weights_file = temp_dir / "weights.json"
    shutil.copyfile(DEFAULT_CORPUS_PATH, pub_corpus)

    engine = reset_corpus_engine_for_testing(
        corpus_path=pub_corpus,
        private_corpus_path=priv_corpus,
        weights_path=weights_file,
    )

    private_dialogue = [
        DialogueTurnInput(
            speaker="U99999",
            text="<@U12345|김철수>님 해당 안건은 영구 폐기 절차로 돌리겠습니다.",
        ),
        DialogueTurnInput(
            speaker="U12345",
            text="네, 아쉽지만 어쩔 수 없네요.",
        ),
    ]
    res = engine.ingest_dialogue_turns(private_dialogue, privacy_mode="features_only")
    assert res.appended_records == 1
    assert res.privacy_mode == "features_only"

    # Verify raw text is NEVER written to private_corpus.jsonl and proper nouns are stripped
    priv_raw_content = priv_corpus.read_text(encoding="utf-8")
    assert "영구 폐기 절차로 돌리겠습니다" not in priv_raw_content
    assert "김철수" not in priv_raw_content
    priv_records = engine.load_private_records()
    assert len(priv_records) == 1
    assert priv_records[0].text == ""
    assert priv_records[0].speaker == "anon"
    assert any("폐기/NNG" in f for f in priv_records[0].morph_features)

    # Deduplication works via utterance_hash even when text="" on disk
    dup_res = engine.ingest_dialogue_turns(private_dialogue, privacy_mode="features_only")
    assert dup_res.appended_records == 0

    # Same-speaker consecutive utterances must NOT bootstrap a reaction label
    same_speaker_turns = [
        DialogueTurnInput(speaker="U1", text="이 부분 수정 가능할까요?"),
        DialogueTurnInput(speaker="U1", text="제가 바로 확인해서 곧 회신드리겠습니다."),
    ]
    same_res = engine.ingest_dialogue_turns(same_speaker_turns, privacy_mode="features_only")
    assert same_res.bootstrapped_from_reactions == 0
    assert same_res.appended_records == 0

    # private_scrubbed with only proper nouns must be skipped and never leak /NNP to weights
    nnp_only_turns = [
        DialogueTurnInput(speaker="U1", text="김철수"),
        DialogueTurnInput(speaker="U2", text="네, 아쉽네요."),
    ]
    nnp_res = engine.ingest_dialogue_turns(nnp_only_turns, privacy_mode="private_scrubbed")
    assert nnp_res.appended_records == 0
    weights_raw = weights_file.read_text(encoding="utf-8")
    assert "김철수" not in weights_raw

    # N-gram inference works on the newly learned private features
    probe = [DialogueTurnInput(speaker="A", text="이 건은 영구 폐기 절차로 돌리겠습니다.")]
    analyzed = analyze_discourse(probe)
    assert analyzed.utterances[0].illocutionary_force == "indirect_refusal_or_deferral"

    reset_corpus_engine_for_testing()


def test_multi_channel_collectors_slack_telegram_web_and_security(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    import shutil
    import urllib.error
    from email.message import Message
    from pathlib import Path
    from typing import Any

    import llm_linguistics.pragmatics.collectors as col_mod
    from llm_linguistics.learner import reset_learner_for_testing
    from llm_linguistics.pragmatics import (
        collect_slack_channel,
        collect_telegram_updates,
        collect_web_dialogues,
        ingest_telegram_export,
        reset_checkpoint_store_for_testing,
        reset_corpus_engine_for_testing,
    )
    from llm_linguistics.pragmatics.corpus_learner import DEFAULT_CORPUS_PATH

    temp_dir = Path(str(tmp_path))
    pub_corpus = temp_dir / "pub.jsonl"
    priv_corpus = temp_dir / "priv.jsonl"
    weights_file = temp_dir / "weights.json"
    checkpoints_file = temp_dir / "checkpoints.json"
    learner_file = temp_dir / "learned.json"
    shutil.copyfile(DEFAULT_CORPUS_PATH, pub_corpus)

    reset_corpus_engine_for_testing(
        corpus_path=pub_corpus,
        private_corpus_path=priv_corpus,
        weights_path=weights_file,
    )
    reset_learner_for_testing(state_path=learner_file)
    cp_store = reset_checkpoint_store_for_testing(path=checkpoints_file)

    # 1. Missing token raises explicit RuntimeError
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="Missing SLACK_BOT_TOKEN"):
        collect_slack_channel("C12345")

    # 2. Mock Slack API history & thread replies (exact 1 reaction bootstrap)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")

    def fake_slack_get(
        endpoint: str,
        params: dict[str, Any],
        token: str,
        timeout_sec: float,
    ) -> dict[str, Any]:
        if endpoint == "conversations.history":
            return {
                "ok": True,
                "messages": [
                    {
                        "ts": "1700000010.000100",
                        "user": "U1",
                        "text": "이번 주 내로 배포 일정 조율 가능하실까요?",
                        "reply_count": 1,
                        "thread_ts": "1700000010.000100",
                    }
                ],
            }
        return {
            "ok": True,
            "messages": [
                {
                    "ts": "1700000010.000100",
                    "user": "U1",
                    "text": "이번 주 내로 배포 일정 조율 가능하실까요?",
                },
                {
                    "ts": "1700000020.000200",
                    "user": "U2",
                    "text": "제가 바로 확인해서 곧 회신드리겠습니다.",
                },
            ],
        }

    monkeypatch.setattr(col_mod, "_slack_api_get", fake_slack_get)
    slack_res = collect_slack_channel("C12345", privacy_mode="features_only")
    assert slack_res.checkpoint_cursor == "1700000010.000100"
    assert cp_store.state.slack_cursors["C12345"] == "1700000010.000100"
    assert slack_res.ingestion_result is not None
    assert slack_res.ingestion_result.bootstrapped_from_reactions == 1
    assert slack_res.ingestion_result.appended_records == 1

    # 3. Telegram Desktop JSON export with reply_to_message_id linking
    tg_export = [
        {
            "id": 101,
            "type": "message",
            "from": "팀장",
            "text": "해당 기능은 장기 백로그로 이관해 두겠습니다.",
        },
        {
            "id": 102,
            "type": "message",
            "from": "기획자",
            "reply_to_message_id": 101,
            "text": ["네, ", {"type": "bold", "text": "아쉽네요."}, " 다음 기회에 진행하죠."],
        },
    ]
    tg_res = ingest_telegram_export(tg_export, privacy_mode="features_only")
    assert tg_res.ingestion_result is not None
    assert tg_res.ingestion_result.bootstrapped_from_reactions == 1
    assert tg_res.ingestion_result.appended_records == 1

    # 4. Telegram Bot getUpdates with exact 1 bootstrap and no duplicate reply pairing
    secret_tg_token = "999999:SUPER-SECRET-TG-TOKEN"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", secret_tg_token)

    class _FakeResp:
        def __init__(self, body: bytes) -> None:
            self._body = body
            self.headers = Message()

        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def read(self, _n: int = -1) -> bytes:
            return self._body

    tg_payload = json.dumps({
        "ok": True,
        "result": [
            {
                "update_id": 500,
                "message": {
                    "chat": {"id": 777},
                    "from": {"id": 11},
                    "text": "지금 바로 확인해서 오늘 중으로 바로 공유하겠습니다.",
                    "reply_to_message": {
                        "from": {"id": 22},
                        "text": "로그 분석 자료 언제쯤 받아볼 수 있을까요?",
                    },
                },
            }
        ],
    }).encode("utf-8")

    monkeypatch.setattr(
        col_mod.urllib.request,
        "urlopen",
        lambda *a, **kw: _FakeResp(tg_payload),
    )
    tg_upd_res = collect_telegram_updates(privacy_mode="features_only")
    assert tg_upd_res.checkpoint_cursor == "501"
    assert cp_store.state.telegram_offset == 501
    assert tg_upd_res.ingestion_result is not None
    assert tg_upd_res.ingestion_result.bootstrapped_from_reactions == 1
    assert tg_upd_res.ingestion_result.appended_records == 1

    # 5. Telegram HTTP error must NOT leak TELEGRAM_BOT_TOKEN in message or __cause__
    def _raise_http_err(req: Any, *a: object, **kw: object) -> Any:
        raise urllib.error.HTTPError(
            req.full_url, 401, "Unauthorized", Message(), None
        )

    monkeypatch.setattr(col_mod.urllib.request, "urlopen", _raise_http_err)
    with pytest.raises(RuntimeError) as exc_info:
        collect_telegram_updates()
    assert secret_tg_token not in str(exc_info.value)
    assert exc_info.value.__cause__ is None

    # 6. Web dialogue collector: initial fetch + OOV mining + hash skip + SSRF DNS block
    monkeypatch.setattr(
        col_mod.socket,
        "getaddrinfo",
        lambda host, port, *a, **kw: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )
    web_html = (
        "<html><body>"
        "<p>이번 분기에는 양자내성암호화모듈 도입을 잠정 유예 상태로 두겠습니다.</p>"
        "<p>네, 아쉽네요. 양자내성암호화모듈은 보류하시는 걸로 알겠습니다.</p>"
        "</body></html>"
    ).encode()
    monkeypatch.setattr(
        col_mod,
        "_safe_http_urlopen",
        lambda req, timeout_sec: _FakeResp(web_html),
    )
    web_res = collect_web_dialogues(
        "https://example.com/dialogues",
        privacy_mode="public_full",
        also_mine_oov=True,
    )
    assert web_res.skipped_unchanged is False
    assert web_res.ingestion_result is not None
    assert web_res.ingestion_result.bootstrapped_from_reactions == 1
    assert len(web_res.oov_words_added) >= 1

    # Repeat fetch with identical payload is skipped via checkpoint hash
    web_dup = collect_web_dialogues("https://example.com/dialogues")
    assert web_dup.skipped_unchanged is True

    # DNS resolving to private IP 10.0.0.1 is blocked
    monkeypatch.setattr(
        col_mod.socket,
        "getaddrinfo",
        lambda host, port, *a, **kw: [(2, 1, 6, "", ("10.0.0.1", 0))],
    )
    with pytest.raises(ValueError, match="non-public IP"):
        collect_web_dialogues("https://rebind.example.com/dialogues")

    reset_corpus_engine_for_testing()
    reset_checkpoint_store_for_testing()
    reset_learner_for_testing()


def test_cli_collect_batch_runner_and_degraded_handling(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    import shutil
    from pathlib import Path

    from llm_linguistics.cli_collect import CollectSourcesConfig, main, run_batch_collection
    from llm_linguistics.pragmatics import (
        reset_checkpoint_store_for_testing,
        reset_corpus_engine_for_testing,
    )
    from llm_linguistics.pragmatics.corpus_learner import DEFAULT_CORPUS_PATH

    temp_dir = Path(str(tmp_path))
    pub_corpus = temp_dir / "pub.jsonl"
    priv_corpus = temp_dir / "priv.jsonl"
    weights_file = temp_dir / "weights.json"
    checkpoints_file = temp_dir / "checkpoints.json"
    shutil.copyfile(DEFAULT_CORPUS_PATH, pub_corpus)

    reset_corpus_engine_for_testing(
        corpus_path=pub_corpus,
        private_corpus_path=priv_corpus,
        weights_path=weights_file,
    )
    reset_checkpoint_store_for_testing(path=checkpoints_file)

    export_file = temp_dir / "result.json"
    export_file.write_text(
        json.dumps(
            {
                "name": "Project Room",
                "messages": [
                    {
                        "id": 1,
                        "type": "message",
                        "from": "PM",
                        "text": "이번 주 내로 보안 감사 일정 확정 가능하실까요?",
                    },
                    {
                        "id": 2,
                        "type": "message",
                        "from": "SecLead",
                        "reply_to_message_id": 1,
                        "text": "지금 바로 확인해서 곧 회신드리겠습니다.",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # 1. Healthy run via CLI main
    exit_code = main(["--telegram-export", str(export_file)])
    assert exit_code == 0

    # 2. Degraded run (1 valid export + 1 missing Slack token)
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    cfg = CollectSourcesConfig(
        slack_channels=["C_MISSING_TOKEN"],
        telegram_export_files=[str(export_file)],
    )
    summary = run_batch_collection(cfg)
    assert summary.status == "degraded"
    assert summary.succeeded_sources == 1
    assert summary.failed_sources == 1
    assert len(summary.errors) == 1

    # --strict returns exit code 1 on degraded status
    strict_code = main(
        ["--telegram-export", str(export_file), "--slack-channel", "C_MISSING", "--strict"]
    )
    assert strict_code == 1

    reset_corpus_engine_for_testing()
    reset_checkpoint_store_for_testing()




