"""Korean syllable/Jamo verification, Josa attachment, and RAG normalization."""

from __future__ import annotations

from llm_linguistics.engine import decompose_syllable, get_kiwi, is_hangul_syllable
from llm_linguistics.morph.schemas import (
    ConstraintCheckResult,
    JosaAttachmentResult,
    JosaPair,
    MorphConstraintSpec,
    MorphemeToken,
    RagNormalizedOutput,
    SyllableInfo,
)

SONORANT_JONGSEONG: frozenset[str] = frozenset({"ㄴ", "ㄹ", "ㅁ", "ㅇ"})

JOSA_TABLE: dict[JosaPair, tuple[str, str]] = {
    "은/는": ("은", "는"),
    "이/가": ("이", "가"),
    "을/를": ("을", "를"),
    "과/와": ("과", "와"),
    "으로/로": ("으로", "로"),
    "아/야": ("아", "야"),
    "이나/나": ("이나", "나"),
    "이란/란": ("이란", "란"),
    "이에요/예요": ("이에요", "예요"),
    "이라서/라서": ("이라서", "라서"),
}

PREDICATE_TAGS: frozenset[str] = frozenset({"VV", "VA", "VX", "XSV", "XSA"})
CONTENT_TAG_PREFIXES: tuple[str, ...] = (
    "NNG", "NNP", "NR", "NP", "VV", "VA", "XR", "SL", "SH", "SN",
)


def _strip_irregular_tag(tag: str) -> str:
    """Normalize Kiwi irregular tags like 'VV-I' or 'VA-R' to base POS tag ('VV', 'VA')."""
    return tag.split("-")[0]


def extract_syllables(text: str) -> list[SyllableInfo]:
    """Extract and decompose all precomposed Hangul syllables from text with exact positions."""
    syllables: list[SyllableInfo] = []
    syl_idx = 0
    for char_idx, ch in enumerate(text):
        if is_hangul_syllable(ch):
            dec = decompose_syllable(ch, index=char_idx)
            syllables.append(
                SyllableInfo(
                    char_index=char_idx,
                    syllable_index=syl_idx,
                    char=dec.char,
                    choseong=dec.choseong,
                    jungseong=dec.jungseong,
                    jongseong=dec.jongseong,
                    has_batchim=dec.has_batchim,
                    is_rieul_batchim=dec.is_rieul_batchim,
                    vowel_polarity=dec.vowel_polarity,
                )
            )
            syl_idx += 1
    return syllables


def verify_constraints(text: str, constraints: MorphConstraintSpec) -> ConstraintCheckResult:
    """Verify syllable count, character length, coda rules, choseong, and vowel harmony."""
    syllables = extract_syllables(text)
    char_count_with_space = len(text)
    char_count_no_space = sum(1 for ch in text if not ch.isspace())
    syllable_count = len(syllables)
    violations: list[str] = []

    if (
        constraints.exact_syllable_count is not None
        and syllable_count != constraints.exact_syllable_count
    ):
        violations.append(
            f"Syllable count mismatch: expected {constraints.exact_syllable_count}, "
            f"got {syllable_count}."
        )

    if (
        constraints.exact_char_count_no_space is not None
        and char_count_no_space != constraints.exact_char_count_no_space
    ):
        violations.append(
            f"Non-space character count mismatch: "
            f"expected {constraints.exact_char_count_no_space}, got {char_count_no_space}."
        )

    if (
        constraints.max_char_count_with_space is not None
        and char_count_with_space > constraints.max_char_count_with_space
    ):
        violations.append(
            f"Total character count exceeds maximum: max {constraints.max_char_count_with_space}, "
            f"got {char_count_with_space}."
        )

    if constraints.batchim_rule == "none":
        for syl in syllables:
            if syl.has_batchim:
                violations.append(
                    f"Batchim forbidden but found at syllable #{syl.syllable_index} "
                    f"(char_index={syl.char_index}, char={syl.char!r}, "
                    f"jongseong={syl.jongseong!r})."
                )
    elif constraints.batchim_rule == "all":
        for syl in syllables:
            if not syl.has_batchim:
                violations.append(
                    f"Batchim required but missing at syllable #{syl.syllable_index} "
                    f"(char_index={syl.char_index}, char={syl.char!r})."
                )
    elif constraints.batchim_rule == "only_sonorant":
        for syl in syllables:
            if syl.jongseong is not None and syl.jongseong not in SONORANT_JONGSEONG:
                violations.append(
                    f"Non-sonorant batchim at syllable #{syl.syllable_index} "
                    f"(char_index={syl.char_index}, char={syl.char!r}, "
                    f"jongseong={syl.jongseong!r}); only ㄴ/ㄹ/ㅁ/ㅇ allowed."
                )

    if constraints.expected_choseong_sequence is not None:
        expected_cho = [ch for ch in constraints.expected_choseong_sequence if not ch.isspace()]
        actual_cho = [syl.choseong for syl in syllables]
        if len(actual_cho) != len(expected_cho):
            violations.append(
                f"Choseong sequence length mismatch: expected {len(expected_cho)} "
                f"({''.join(expected_cho)!r}), got {len(actual_cho)} ({''.join(actual_cho)!r})."
            )
        else:
            for idx, (exp, act) in enumerate(zip(expected_cho, actual_cho, strict=True)):
                if exp != act:
                    syl = syllables[idx]
                    violations.append(
                        f"Choseong mismatch at syllable #{idx} (char={syl.char!r}): "
                        f"expected {exp!r}, got {act!r}."
                    )

    if constraints.enforce_vowel_harmony_per_word:
        for word_idx, word in enumerate(text.split()):
            word_syllables = extract_syllables(word)
            polarities = {
                syl.vowel_polarity
                for syl in word_syllables
                if syl.vowel_polarity in ("positive", "negative")
            }
            if len(polarities) > 1:
                violations.append(
                    f"Vowel harmony violation in word #{word_idx} ({word!r}): "
                    f"mixes positive and negative vowels."
                )

    return ConstraintCheckResult(
        is_valid=len(violations) == 0,
        char_count_with_space=char_count_with_space,
        char_count_no_space=char_count_no_space,
        syllable_count=syllable_count,
        violations=violations,
        syllables=syllables,
    )


def attach_josa(word: str, josa_pair: JosaPair) -> JosaAttachmentResult:
    """Deterministically select and attach the Korean postposition (조사)."""
    cleaned = word.strip()
    if not cleaned:
        raise ValueError("Input word must not be empty or whitespace-only.")
    if josa_pair not in JOSA_TABLE:
        raise ValueError(f"Unsupported josa_pair: {josa_pair!r}. Valid options: {list(JOSA_TABLE)}")

    last_char = cleaned[-1]
    if not is_hangul_syllable(last_char):
        raise ValueError(
            f"Last character {last_char!r} of word {cleaned!r} "
            f"is not a precomposed Hangul syllable."
        )

    syllables = extract_syllables(cleaned)
    last_syl = syllables[-1]
    with_batchim_form, without_batchim_form = JOSA_TABLE[josa_pair]

    if josa_pair == "으로/로":
        if not last_syl.has_batchim:
            selected = without_batchim_form
            reason = f"Final syllable {last_syl.char!r} has no batchim -> '{selected}'."
        elif last_syl.is_rieul_batchim:
            selected = without_batchim_form
            reason = f"Final syllable {last_syl.char!r} ends with 'ㄹ' batchim -> '{selected}'."
        else:
            selected = with_batchim_form
            reason = (
                f"Final syllable {last_syl.char!r} ends with non-'ㄹ' batchim "
                f"({last_syl.jongseong!r}) -> '{selected}'."
            )
    else:
        if last_syl.has_batchim:
            selected = with_batchim_form
            reason = (
                f"Final syllable {last_syl.char!r} has batchim ({last_syl.jongseong!r}) "
                f"-> '{selected}'."
            )
        else:
            selected = without_batchim_form
            reason = f"Final syllable {last_syl.char!r} has no batchim -> '{selected}'."

    return JosaAttachmentResult(
        word=cleaned,
        josa_pair=josa_pair,
        selected_josa=selected,
        combined=f"{cleaned}{selected}",
        last_syllable=last_syl,
        reason=reason,
    )


def normalize_for_rag(text: str) -> RagNormalizedOutput:
    """Decompose Korean text into canonical lemmas and split compounds for Hybrid RAG indexing."""
    if not text.strip():
        return RagNormalizedOutput(
            surface_text=text,
            morphemes=[],
            content_lemmas=[],
            decomposed_compounds={},
            normalized_query="",
        )

    kiwi = get_kiwi()
    coarse_tokens = kiwi.tokenize(text, normalize_coda=True, split_complex=False)
    fine_tokens = kiwi.tokenize(text, normalize_coda=True, split_complex=True)

    morphemes: list[MorphemeToken] = []
    content_lemmas: list[str] = []

    for tok in fine_tokens:
        base_tag = _strip_irregular_tag(tok.tag)
        lemma = f"{tok.form}다" if base_tag in PREDICATE_TAGS else tok.form
        morphemes.append(
            MorphemeToken(
                form=tok.form,
                tag=tok.tag,
                lemma=lemma,
                start=tok.start,
                length=tok.len,
            )
        )
        if base_tag.startswith(CONTENT_TAG_PREFIXES):
            content_lemmas.append(lemma)

    decomposed_compounds: dict[str, list[str]] = {}
    for c_tok in coarse_tokens:
        c_end = c_tok.start + c_tok.len
        sub_tokens = [
            f_tok
            for f_tok in fine_tokens
            if f_tok.start >= c_tok.start and (f_tok.start + f_tok.len) <= c_end
        ]
        if len(sub_tokens) > 1:
            sub_lemmas = [
                f"{st.form}다" if _strip_irregular_tag(st.tag) in PREDICATE_TAGS else st.form
                for st in sub_tokens
            ]
            decomposed_compounds[c_tok.form] = sub_lemmas

    return RagNormalizedOutput(
        surface_text=text,
        morphemes=morphemes,
        content_lemmas=content_lemmas,
        decomposed_compounds=decomposed_compounds,
        normalized_query=" ".join(content_lemmas),
    )
