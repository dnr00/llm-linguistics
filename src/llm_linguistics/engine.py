"""Shared Kiwi morphological engine singleton and Hangul Jamo Unicode utilities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from kiwipiepy import Kiwi

HANGUL_BASE = 0xAC00
HANGUL_END = 0xD7A3

CHOSEONG_LIST: tuple[str, ...] = (
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
)

JUNGSEONG_LIST: tuple[str, ...] = (
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
)

JONGSEONG_LIST: tuple[str | None, ...] = (
    None, "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
)

POSITIVE_VOWELS: frozenset[str] = frozenset({"ㅏ", "ㅑ", "ㅗ", "ㅘ", "ㅛ", "ㅐ", "ㅒ", "ㅙ", "ㅚ"})
NEGATIVE_VOWELS: frozenset[str] = frozenset({"ㅓ", "ㅕ", "ㅜ", "ㅝ", "ㅠ", "ㅔ", "ㅖ", "ㅞ", "ㅟ"})
NEUTRAL_VOWELS: frozenset[str] = frozenset({"ㅡ", "ㅣ", "ㅢ"})

VowelPolarity = Literal["positive", "negative", "neutral"]


@dataclass(frozen=True)
class DecomposedSyllable:
    """Immutable representation of a single decomposed Korean Hangul syllable."""

    index: int
    char: str
    choseong: str
    jungseong: str
    jongseong: str | None
    vowel_polarity: VowelPolarity

    @property
    def has_batchim(self) -> bool:
        return self.jongseong is not None

    @property
    def is_rieul_batchim(self) -> bool:
        return self.jongseong == "ㄹ"


@lru_cache(maxsize=1)
def get_kiwi() -> Kiwi:
    """Return a process-wide singleton Kiwi analyzer instance."""
    return Kiwi()


def is_hangul_syllable(ch: str) -> bool:
    """Return True if ch is a single precomposed Hangul syllable (U+AC00..U+D7A3)."""
    if len(ch) != 1:
        return False
    code = ord(ch)
    return HANGUL_BASE <= code <= HANGUL_END


def classify_vowel_polarity(jungseong: str) -> VowelPolarity:
    """Classify a Korean medial vowel into positive, negative, or neutral polarity."""
    if jungseong in POSITIVE_VOWELS:
        return "positive"
    if jungseong in NEGATIVE_VOWELS:
        return "negative"
    return "neutral"


def decompose_syllable(ch: str, index: int = 0) -> DecomposedSyllable:
    """Decompose a single Hangul syllable character into choseong, jungseong, and jongseong."""
    if not is_hangul_syllable(ch):
        raise ValueError(f"Character {ch!r} at index {index} is not a precomposed Hangul syllable.")
    offset = ord(ch) - HANGUL_BASE
    cho_idx = offset // (21 * 28)
    jung_idx = (offset // 28) % 21
    jong_idx = offset % 28
    jung = JUNGSEONG_LIST[jung_idx]
    return DecomposedSyllable(
        index=index,
        char=ch,
        choseong=CHOSEONG_LIST[cho_idx],
        jungseong=jung,
        jongseong=JONGSEONG_LIST[jong_idx],
        vowel_polarity=classify_vowel_polarity(jung),
    )


def compose_syllable(choseong: str, jungseong: str, jongseong: str | None = None) -> str:
    """Compose choseong, jungseong, and optional jongseong into a single Hangul syllable."""
    try:
        cho_idx = CHOSEONG_LIST.index(choseong)
        jung_idx = JUNGSEONG_LIST.index(jungseong)
        jong_idx = JONGSEONG_LIST.index(jongseong)
    except ValueError as exc:
        raise ValueError(
            f"Invalid Jamo combination: choseong={choseong!r}, "
            f"jungseong={jungseong!r}, jongseong={jongseong!r}"
        ) from exc
    code = HANGUL_BASE + (cho_idx * 21 * 28) + (jung_idx * 28) + jong_idx
    return chr(code)
