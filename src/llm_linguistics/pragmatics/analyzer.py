"""Deterministic Korean speech-level, honorific, and indirect speech-act analyzer."""

from __future__ import annotations

from llm_linguistics.engine import get_kiwi, normalize_kiwi_jamo
from llm_linguistics.pragmatics.schemas import (
    DialogueTurnInput,
    DiscourseAnalysisResult,
    IllocutionaryForce,
    PolitenessShift,
    SentenceType,
    SpeechLevel,
    UtterancePragmatics,
)

POLITENESS_RANK_MAP: dict[SpeechLevel, int] = {
    "formal_high": 5,
    "informal_high": 4,
    "neutral_mid": 3,
    "formal_low": 2,
    "fragment": 2,
    "informal_low": 1,
}

FORMAL_HIGH_ENDINGS: frozenset[str] = frozenset({
    "ㅂ니다", "습니다", "ㅂ니까", "습니까", "ㅂ시오", "십시오", "소서", "올시다", "나이다",
})

NEUTRAL_MID_ENDINGS: frozenset[str] = frozenset({
    "소", "오", "구려", "게", "세",
})

FORMAL_LOW_ENDINGS: frozenset[str] = frozenset({
    "다", "는다", "ㄴ다", "라", "어라", "아라", "거라", "너라",
    "자", "느냐", "니", "냐", "구나", "군", "렴", "으렴",
})

FRAGMENT_ENDINGS: frozenset[str] = frozenset({"음", "ㅁ", "기"})

IMPERATIVE_ENDINGS: frozenset[str] = frozenset({
    "ㅂ시오", "십시오", "소서", "라", "어라", "아라", "거라", "너라",
    "게", "렴", "으렴", "세요", "으세요",
})

PROPOSATIVE_ENDINGS: frozenset[str] = frozenset({
    "ㅂ시다", "읍시다", "자", "세", "시죠", "지요",
})

INTERROGATIVE_ENDINGS: frozenset[str] = frozenset({
    "ㅂ니까", "습니까", "느냐", "니", "냐", "ㄹ까요", "ㄹ까", "나요", "가요", "던가요",
})

HONORIFIC_LEXICAL_STEMS: frozenset[str] = frozenset({
    "드리", "모시", "뵙", "여쭙", "여쭈", "계시", "잡수시", "주무시", "말씀",
})

PRAGMATIC_MARKER_LEXICON: dict[str, tuple[str, ...]] = {
    "indirect_refusal_cue": (
        "검토해 보",
        "검토해보",
        "검토하겠",
        "내부적으로 논의",
        "내부 논의",
        "추후에",
        "다음 기회에",
        "여의치",
        "장담하기 어렵",
        "확답드리기",
        "확답하기",
    ),
    "soft_urge_cue": (
        "혹시 언제",
        "일정 공유",
        "가능하실까요",
        "가능할까요",
        "확인 부탁",
        "피드백 부탁",
        "회신 부탁",
        "부탁드렸던",
        "기다리고 있",
        "공유 부탁",
    ),
    "hedging": (
        "것 같",
        "듯하",
        "아무래도",
        "글쎄",
        "다소",
        "어렵지 않",
        "싶네요",
        "싶습니다",
        "편입니",
        "조심스럽",
    ),
    "apology_or_cushion": (
        "죄송합니다만",
        "죄송하지만",
        "송구하나",
        "실례지만",
        "번거로우시겠지만",
        "다름이 아니라",
        "양해 부탁",
    ),
    "direct_pressure_or_blame": (
        "당장",
        "도대체",
        "왜 아직",
        "말이 됩니까",
        "말이 돼",
        "책임지",
        "몇 번을",
    ),
}


def _normalize_jamo(form: str) -> str:
    """Normalize Hangul Jamo compatibility forms (e.g. '\\u11b8니다' -> 'ㅂ니다')."""
    return normalize_kiwi_jamo(form)


def _classify_speech_level(
    ef_forms: list[str],
    has_trailing_yo_jx: bool,
    has_nominal_etn_ending: bool,
) -> SpeechLevel:
    if not ef_forms:
        return "informal_high" if has_trailing_yo_jx else "fragment"

    last_ef = ef_forms[-1]
    if last_ef in FORMAL_HIGH_ENDINGS:
        return "formal_high"
    if last_ef.endswith("요") or last_ef.endswith("죠") or has_trailing_yo_jx:
        return "informal_high"
    if last_ef in FRAGMENT_ENDINGS or has_nominal_etn_ending:
        return "fragment"
    if last_ef in NEUTRAL_MID_ENDINGS:
        return "neutral_mid"
    if last_ef in FORMAL_LOW_ENDINGS:
        return "formal_low"
    return "informal_low"


def _classify_sentence_type(
    text: str,
    ef_forms: list[str],
    speech_level: SpeechLevel,
) -> SentenceType:
    if speech_level == "fragment" and not ef_forms:
        return "fragment"

    stripped = text.strip()
    last_ef = ef_forms[-1] if ef_forms else ""

    if stripped.endswith("?") or last_ef in INTERROGATIVE_ENDINGS:
        return "interrogative"
    if last_ef in IMPERATIVE_ENDINGS or (
        speech_level == "informal_low" and ("가져와" in stripped or "해라" in stripped)
    ):
        return "imperative"
    if last_ef in PROPOSATIVE_ENDINGS and not stripped.endswith("?"):
        return "proposative"
    return "declarative"


def _detect_markers(text: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for category, patterns in PRAGMATIC_MARKER_LEXICON.items():
        matches = [pat for pat in patterns if pat in text]
        if matches:
            found[category] = matches
    return found


def _infer_illocutionary_force(
    sentence_type: SentenceType,
    markers: dict[str, list[str]],
) -> tuple[IllocutionaryForce, str]:
    if "direct_pressure_or_blame" in markers:
        return (
            "direct_pressure_or_blame",
            "Direct pressure or blame markers detected in utterance.",
        )
    if "indirect_refusal_cue" in markers:
        return (
            "indirect_refusal_or_deferral",
            f"Surface form is {sentence_type}, but indirect refusal/deferral cues "
            f"({markers['indirect_refusal_cue']}) indicate polite rejection or postponement.",
        )
    if "soft_urge_cue" in markers:
        return (
            "indirect_request_or_urge",
            f"Surface form is {sentence_type}, but soft urge cues "
            f"({markers['soft_urge_cue']}) indicate an indirect request for action/schedule.",
        )
    if len(markers.get("hedging", [])) >= 2 or (
        "hedging" in markers and "apology_or_cushion" in markers
    ):
        return (
            "hedged_dissent_or_reservation",
            "Multiple hedging/cushion markers indicate reservation or softened disagreement.",
        )
    if sentence_type == "imperative":
        return ("direct_directive", "Direct command or instruction.")
    if sentence_type == "proposative":
        return ("proposal", "Direct proposal for joint action.")
    if sentence_type == "interrogative":
        return ("literal_interrogative", "Direct question.")
    if sentence_type == "fragment":
        return ("literal_fragment", "Elliptical or nominal-ending statement.")
    return ("literal_declarative", "Direct statement.")


def _compute_shift(prev_rank: int | None, curr_rank: int) -> PolitenessShift:
    if prev_rank is None:
        return "initial"
    if curr_rank > prev_rank:
        return "upgraded"
    if curr_rank < prev_rank:
        return "downgraded"
    return "maintained"


def analyze_discourse(utterances: list[DialogueTurnInput]) -> DiscourseAnalysisResult:
    """Deterministically analyze speech levels, honorific shifts, and pragmatic markers."""
    if not utterances:
        raise ValueError("utterances list must not be empty.")

    kiwi = get_kiwi()
    results: list[UtterancePragmatics] = []
    prev_global_rank: int | None = None
    speaker_last_rank: dict[str, int] = {}
    speaker_max_ranks: dict[str, int] = {}
    politeness_drop_turns: list[int] = []
    indirect_speech_act_turns: list[int] = []

    for idx, turn in enumerate(utterances):
        tokens = kiwi.tokenize(turn.text)
        ef_forms = [_normalize_jamo(t.form) for t in tokens if t.tag == "EF"]

        # Check if the sentence ends with an auxiliary particle '요' (JX)
        non_punct_tokens = [t for t in tokens if not t.tag.startswith("S")]
        has_trailing_yo_jx = bool(
            non_punct_tokens
            and non_punct_tokens[-1].tag == "JX"
            and _normalize_jamo(non_punct_tokens[-1].form) == "요"
        )
        has_nominal_etn_ending = bool(
            non_punct_tokens
            and non_punct_tokens[-1].tag == "ETN"
            and _normalize_jamo(non_punct_tokens[-1].form) in FRAGMENT_ENDINGS
        )

        has_honorific_si = any(
            (t.tag == "EP" and _normalize_jamo(t.form) in ("시", "으시"))
            or (_normalize_jamo(t.form) in HONORIFIC_LEXICAL_STEMS)
            for t in tokens
        )

        speech_level = _classify_speech_level(
            ef_forms=ef_forms,
            has_trailing_yo_jx=has_trailing_yo_jx,
            has_nominal_etn_ending=has_nominal_etn_ending,
        )
        rank = POLITENESS_RANK_MAP[speech_level]

        global_shift = _compute_shift(prev_global_rank, rank)
        speaker_shift = _compute_shift(speaker_last_rank.get(turn.speaker), rank)

        prev_global_rank = rank
        speaker_last_rank[turn.speaker] = rank
        speaker_max_ranks[turn.speaker] = max(speaker_max_ranks.get(turn.speaker, rank), rank)

        if speaker_shift == "downgraded":
            politeness_drop_turns.append(idx)

        sentence_type = _classify_sentence_type(turn.text, ef_forms, speech_level)
        markers = _detect_markers(turn.text)
        illocutionary_force, note = _infer_illocutionary_force(sentence_type, markers)

        if illocutionary_force in (
            "indirect_refusal_or_deferral",
            "indirect_request_or_urge",
            "hedged_dissent_or_reservation",
        ):
            indirect_speech_act_turns.append(idx)

        results.append(
            UtterancePragmatics(
                turn_index=idx,
                speaker=turn.speaker,
                text=turn.text,
                speech_level=speech_level,
                politeness_rank=rank,
                has_honorific_si=has_honorific_si,
                politeness_shift_from_prev=global_shift,
                speaker_politeness_shift=speaker_shift,
                sentence_type=sentence_type,
                illocutionary_force=illocutionary_force,
                detected_markers=markers,
                final_ending_morphemes=ef_forms,
                pragmatic_note=note,
            )
        )

    # Detect politeness asymmetry across distinct speakers (e.g., high tier >=4 vs low tier <=2)
    has_high_speaker = any(r >= 4 for r in speaker_max_ranks.values())
    has_low_only_speaker = any(r <= 2 for r in speaker_max_ranks.values())
    asymmetry_detected = len(speaker_max_ranks) >= 2 and has_high_speaker and has_low_only_speaker

    summary_notes: list[str] = []
    if asymmetry_detected:
        summary_notes.append(
            "Politeness tier asymmetry detected between speakers (존댓말 vs 해라체/해체)."
        )
    if politeness_drop_turns:
        summary_notes.append(
            f"Intra-speaker politeness downgrade detected at turns {politeness_drop_turns}."
        )
    if indirect_speech_act_turns:
        summary_notes.append(
            f"Indirect speech acts (refusal/urge/reservation) detected at turns "
            f"{indirect_speech_act_turns}."
        )

    return DiscourseAnalysisResult(
        turn_count=len(results),
        utterances=results,
        asymmetry_detected=asymmetry_detected,
        politeness_drop_turns=politeness_drop_turns,
        indirect_speech_act_turns=indirect_speech_act_turns,
        summary_notes=summary_notes,
    )
