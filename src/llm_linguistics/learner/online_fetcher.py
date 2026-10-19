"""Online URL and RSS feed fetcher for continuous Korean corpus ingestion and OOV mining."""

from __future__ import annotations

import ipaddress
import re
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


IGNORED_HTML_TAGS: frozenset[str] = frozenset({
    "script",
    "style",
    "noscript",
    "head",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "button",
    "select",
    "option",
    "svg",
})

UI_BOILERPLATE_SUBSTRINGS: tuple[str, ...] = (
    "바로가기",
    "로그인이 유지",
    "검색어 저장 기능이",
    "구독중인 채널이 없습니다",
    "야간 모드를 이용해",
    "불법촬영물",
    "개인정보처리방침",
    "새창으로 이동",
    "새로운 댓글이",
    "업로드 가능 확장자",
    "회원이 아니신가요",
    "불러오는 중입니다",
    "즐겨찾는 갤러리",
    "갤러리명 클릭 시",
    "모바일 앱 이용 안내",
    "추천을 많이 받은 게시물",
    "클릭하시면 해당 채널",
    "카테고리 별 최신글",
    "예고없이 삭제 또는 제한",
    "권한을 제한할 수 있습니다",
    "해당 IP나 ID를 제재",
    "이용에 주의를 부탁",
)

COMMUNITY_POST_LINK_PATTERNS: tuple[str, ...] = (
    r'href=["\'](/b/[a-z0-9_]+/[0-9]{8,12})(?:\?[^"\']*)?["\']',
    r'href=["\'](/service/board/park/[0-9]{7,10})(?:\?[^"\']*)?["\']',
    r'href=["\'](https://bbs\.ruliweb\.com/best/board/[0-9]+/read/[0-9]+)["\']',
    r'href=["\'](/talk/[0-9]{8,12})["\']',
    r'href=["\'](/dogdrip/[0-9]{8,12})(?:\?[^"\']*)?["\']',
)

DEFAULT_WEB_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
}


def extract_community_post_url(base_url: str, raw_html: str) -> str | None:
    """Extract a non-notice post URL from a Korean community board page for comment harvesting."""
    base_host = (urllib.parse.urlparse(base_url).hostname or "").lower()
    for pat in COMMUNITY_POST_LINK_PATTERNS:
        matches = list(dict.fromkeys(re.findall(pat, raw_html)))
        if not matches:
            continue
        chosen = matches[2] if len(matches) >= 3 else matches[-1]
        candidate = urllib.parse.urljoin(base_url, chosen)
        cand_host = (urllib.parse.urlparse(candidate).hostname or "").lower()
        if cand_host == base_host:
            return candidate
    return None


class _VisibleTextExtractor(HTMLParser):
    """Extract visible text from HTML while ignoring script/style/nav/chrome blocks."""

    def __init__(self) -> None:
        super().__init__()
        self._ignored_depth = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in IGNORED_HTML_TAGS:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in IGNORED_HTML_TAGS and self._ignored_depth > 0:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0:
            stripped = " ".join(data.split())
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
        if any(bad in b_text for bad in UI_BOILERPLATE_SUBSTRINGS):
            continue
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
