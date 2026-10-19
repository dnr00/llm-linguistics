"""Tests for Korean morphological analysis, syllable constraints, Josa, and RAG normalization."""

from __future__ import annotations

import pytest

from llm_linguistics.engine import compose_syllable, decompose_syllable
from llm_linguistics.morph import (
    MorphConstraintSpec,
    attach_josa,
    normalize_for_rag,
    verify_constraints,
)


def test_decompose_and_compose_roundtrip() -> None:
    dec = decompose_syllable("한", index=0)
    assert dec.choseong == "ㅎ"
    assert dec.jungseong == "ㅏ"
    assert dec.jongseong == "ㄴ"
    assert dec.has_batchim is True
    assert dec.is_rieul_batchim is False
    assert dec.vowel_polarity == "positive"
    assert compose_syllable(dec.choseong, dec.jungseong, dec.jongseong) == "한"

    dec_open = decompose_syllable("가", index=1)
    assert dec_open.jongseong is None
    assert dec_open.has_batchim is False
    assert compose_syllable(dec_open.choseong, dec_open.jungseong, None) == "가"


def test_decompose_non_hangul_raises() -> None:
    with pytest.raises(ValueError, match="not a precomposed Hangul syllable"):
        decompose_syllable("A")


def test_verify_constraints_no_batchim_and_choseong() -> None:
    spec = MorphConstraintSpec(
        exact_syllable_count=5,
        batchim_rule="none",
        expected_choseong_sequence="ㅂㄴㄴㅌㅁ",
    )
    valid_res = verify_constraints("바나나 토마", spec)
    assert valid_res.is_valid is True
    assert valid_res.violations == []

    invalid_res = verify_constraints("바나나 토마토와 학", spec)
    assert invalid_res.is_valid is False
    assert any("Syllable count mismatch" in v for v in invalid_res.violations)
    assert any("Batchim forbidden" in v and "char='학'" in v for v in invalid_res.violations)


def test_verify_constraints_sonorant_and_vowel_harmony() -> None:
    spec = MorphConstraintSpec(
        batchim_rule="only_sonorant",
        enforce_vowel_harmony_per_word=True,
    )
    # '살랑살랑' has only ㄹ/ㅇ codas and positive vowel 'ㅏ'
    res_ok = verify_constraints("살랑살랑 말랑말랑", spec)
    assert res_ok.is_valid is True

    # '깡충깡충' mixes positive 'ㅏ' and negative 'ㅜ'; '국밥' has non-sonorant ㄱ/ㅂ codas
    res_fail = verify_constraints("깡충깡충 국밥", spec)
    assert res_fail.is_valid is False
    assert any("Vowel harmony violation" in v and "'깡충깡충'" in v for v in res_fail.violations)
    assert any("Non-sonorant batchim" in v and "char='국'" in v for v in res_fail.violations)


@pytest.mark.parametrize(
    ("word", "josa_pair", "expected_combined"),
    [
        ("서울", "으로/로", "서울로"),      # ㄹ 받침 예외 -> 로
        ("부산", "으로/로", "부산으로"),    # ㄴ 받침 -> 으로
        ("바다", "으로/로", "바다로"),      # 무받침 -> 로
        ("사과", "을/를", "사과를"),
        ("수박", "을/를", "수박을"),
        ("학생", "이에요/예요", "학생이에요"),
        ("의사", "이에요/예요", "의사예요"),
    ],
)
def test_attach_josa_valid(word: str, josa_pair: str, expected_combined: str) -> None:
    res = attach_josa(word, josa_pair)  # type: ignore[arg-type]
    assert res.combined == expected_combined


def test_attach_josa_invalid_input_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        attach_josa("   ", "은/는")

    with pytest.raises(ValueError, match="not a precomposed Hangul syllable"):
        attach_josa("AI", "은/는")


def test_normalize_for_rag_irregular_predicates_and_compounds() -> None:
    text = "날씨가 추워서 아름다운 거리를 빠르게 걸었다."
    out = normalize_for_rag(text)
    # ㅂ 불규칙 ('추워서' -> '춥다', '아름다운' -> '아름답다'), ㄷ 불규칙 ('걸었다' -> '걷다')
    assert "춥다" in out.content_lemmas
    assert "아름답다" in out.content_lemmas
    assert "걷다" in out.content_lemmas
    assert "날씨" in out.content_lemmas
    assert "거리" in out.content_lemmas
