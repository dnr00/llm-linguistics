"""Online URL and RSS feed fetcher for continuous Korean corpus ingestion and OOV mining."""

from __future__ import annotations

import ipaddress
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

from llm_linguistics.engine import get_kiwi, is_hangul_syllable
from llm_linguistics.learner.schemas import OnlineUrlIngestionResult
from llm_linguistics.learner.store import get_learner

BLOCKED_HOSTS: frozenset[str] = frozenset({
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "169.254.169.254",
    "metadata.google.internal",
})


class _VisibleTextExtractor(HTMLParser):
    """Extract visible text from HTML while ignoring script/style blocks."""

    def __init__(self) -> None:
        super().__init__()
        self._ignored_depth = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in ("script", "style", "noscript", "head"):
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in ("script", "style", "noscript", "head") and self._ignored_depth > 0:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0:
            stripped = data.strip()
            if stripped:
                self.chunks.append(stripped)


def _validate_public_http_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Only http:// and https:// URLs are allowed; got {parsed.scheme!r}.")
    host = (parsed.hostname or "").lower()
    if not host or host in BLOCKED_HOSTS:
        raise ValueError(f"Blocked or invalid hostname: {host!r}.")
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError(f"Private or reserved IP addresses are forbidden: {host!r}.")
    except ValueError as exc:
        if "Private or reserved" in str(exc):
            raise


def html_or_xml_to_blocks(raw_content: str, min_hangul_syllables: int = 5) -> list[str]:
    """Parse RSS/Atom XML or HTML string into paragraph/element-level Korean text blocks."""
    text_blocks: list[str] = []
    stripped = raw_content.lstrip()

    if stripped.startswith("<?xml") or stripped.startswith("<rss") or stripped.startswith("<feed"):
        try:
            root = ET.fromstring(raw_content)
            for elem in root.iter():
                tag_name = elem.tag.split("}")[-1].lower()
                if (
                    tag_name in ("title", "description", "summary", "content", "encoded")
                    and elem.text
                ):
                    parser = _VisibleTextExtractor()
                    parser.feed(elem.text)
                    text_blocks.extend(parser.chunks)
        except ET.ParseError:
            pass

    if not text_blocks:
        parser = _VisibleTextExtractor()
        parser.feed(raw_content)
        text_blocks = parser.chunks

    filtered_blocks: list[str] = []
    for block in text_blocks:
        b_text = block.strip()
        hangul_cnt = sum(1 for ch in b_text if is_hangul_syllable(ch))
        if hangul_cnt >= min_hangul_syllables:
            filtered_blocks.append(b_text)
    return filtered_blocks


def html_or_xml_to_sentences(raw_content: str, min_hangul_syllables: int = 5) -> list[str]:
    """Parse RSS/Atom XML or HTML string into clean Korean sentences."""
    text_blocks = html_or_xml_to_blocks(raw_content, min_hangul_syllables=min_hangul_syllables)
    kiwi = get_kiwi()
    sentences: list[str] = []
    for block in text_blocks:
        for sent in kiwi.split_into_sents(block):
            s_text = sent.text.strip()
            hangul_cnt = sum(1 for ch in s_text if is_hangul_syllable(ch))
            if hangul_cnt >= min_hangul_syllables:
                sentences.append(s_text)

    return sentences


def ingest_online_url(
    url: str,
    min_cnt: int = 2,
    min_pos_score: float = 0.2,
    timeout_sec: float = 10.0,
) -> OnlineUrlIngestionResult:
    """Fetch an online HTTP/HTTPS page or RSS feed, extract Korean sentences, and mine OOV words."""
    _validate_public_http_url(url)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "llm-linguistics-mcp/0.3 (+https://github.com/dnr00/llm-linguistics)"},
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # noqa: S310
        raw_bytes = resp.read(2_000_000)  # Cap read at 2 MB for bounded memory
        charset = resp.headers.get_content_charset() or "utf-8"
        raw_text = raw_bytes.decode(charset, errors="replace")

    sentences = html_or_xml_to_sentences(raw_text)
    if not sentences:
        raise ValueError(f"No Korean sentences with >=5 Hangul syllables found at {url!r}.")

    mining_res = get_learner().mine_corpus_oov_words(
        texts=sentences,
        min_cnt=min_cnt,
        min_pos_score=min_pos_score,
    )
    return OnlineUrlIngestionResult(
        url=url,
        extracted_sentence_count=len(sentences),
        sample_sentences=sentences[:5],
        mining_result=mining_res,
    )
