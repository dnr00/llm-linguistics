"""Multi-channel Korean dialogue collectors (Slack, Telegram, Web/RSS) with privacy separation."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import socket
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import IO, Any

from pydantic import BaseModel, Field

from llm_linguistics.engine import is_hangul_syllable
from llm_linguistics.learner.online_fetcher import (
    DEFAULT_WEB_HEADERS,
    _validate_public_http_url,
    html_or_xml_to_blocks,
    html_or_xml_to_sentences,
)
from llm_linguistics.learner.store import get_learner
from llm_linguistics.pragmatics.corpus_learner import (
    DialogueIngestionResult,
    PrivacyMode,
    get_corpus_engine,
)
from llm_linguistics.pragmatics.schemas import DialogueTurnInput

DEFAULT_CHECKPOINTS_PATH = Path.home() / ".llm_linguistics" / "collector_checkpoints.json"


class CollectorCheckpoints(BaseModel):
    """Durable incremental cursors for multi-channel dialogue collectors."""

    slack_cursors: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from Slack channel_id -> latest processed message ts",
    )
    telegram_offset: int = Field(
        default=0,
        description="Next Telegram getUpdates offset (max_update_id + 1)",
    )
    web_content_hashes: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from URL -> SHA-256 of last processed payload",
    )


class ChannelCollectionResult(BaseModel):
    """Result summary returned after collecting dialogues from an external channel."""

    channel_type: str
    target_id: str
    fetched_messages: int
    constructed_turns: int
    skipped_unchanged: bool = False
    checkpoint_cursor: str
    oov_words_added: list[str] = Field(default_factory=list)
    ingestion_result: DialogueIngestionResult | None = None


class CollectorCheckpointStore:
    """Manages atomic persistence of incremental collection cursors."""

    def __init__(self, path: Path | str | None = None) -> None:
        raw = path or os.environ.get(
            "LLM_LINGUISTICS_CHECKPOINTS_PATH",
            DEFAULT_CHECKPOINTS_PATH,
        )
        self.path = Path(raw).expanduser().resolve()
        self.state = self._load()

    def _load(self) -> CollectorCheckpoints:
        if not self.path.exists():
            return CollectorCheckpoints()
        return CollectorCheckpoints.model_validate_json(
            self.path.read_text(encoding="utf-8")
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(self.state.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_path, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass


_CHECKPOINT_SINGLETON: CollectorCheckpointStore | None = None


def get_checkpoint_store() -> CollectorCheckpointStore:
    """Return process-wide singleton CollectorCheckpointStore."""
    global _CHECKPOINT_SINGLETON
    if _CHECKPOINT_SINGLETON is None:
        _CHECKPOINT_SINGLETON = CollectorCheckpointStore()
    return _CHECKPOINT_SINGLETON


def reset_checkpoint_store_for_testing(
    path: Path | str | None = None,
) -> CollectorCheckpointStore:
    """Reinitialize singleton CollectorCheckpointStore for isolated tests."""
    global _CHECKPOINT_SINGLETON
    _CHECKPOINT_SINGLETON = CollectorCheckpointStore(path=path)
    return _CHECKPOINT_SINGLETON


def _has_min_hangul(text: str, min_syllables: int = 3) -> bool:
    return sum(1 for ch in text if is_hangul_syllable(ch)) >= min_syllables


def _validate_url_and_resolved_ips(url: str) -> None:
    """Validate public HTTP/HTTPS URL and verify all resolved DNS IPs are globally routable."""
    _validate_public_http_url(url)
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"Failed to resolve hostname {host!r}.") from exc
    if not infos:
        raise ValueError(f"Hostname {host!r} returned no DNS records.")
    for info in infos:
        ip_str = str(info[4][0])
        ip = ipaddress.ip_address(ip_str)
        if (
            not ip.is_global
            or ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            raise ValueError(f"Hostname {host!r} resolves to non-public IP {ip_str!r}.")


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validate every HTTP redirect target against SSRF and DNS rebinding rules."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        resolved_target = urllib.parse.urljoin(req.full_url, newurl)
        _validate_url_and_resolved_ips(resolved_target)
        return super().redirect_request(req, fp, code, msg, headers, resolved_target)


def _safe_http_urlopen(req: urllib.request.Request, timeout_sec: float) -> Any:
    """Open a public HTTP/HTTPS request with SSRF-checked redirect handling."""
    opener = urllib.request.build_opener(_SafeRedirectHandler())
    return opener.open(req, timeout=timeout_sec)


def _slack_api_get(
    endpoint: str,
    params: dict[str, str | int],
    token: str,
    timeout_sec: float,
) -> dict[str, Any]:
    """Execute a bounded GET call to Slack Web API without exposing token in errors."""
    query = urllib.parse.urlencode(params)
    url = f"https://slack.com/api/{endpoint}?{query}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # noqa: S310
            raw_bytes = resp.read(2_000_000)
            payload: dict[str, Any] = json.loads(raw_bytes.decode("utf-8", errors="replace"))
    except Exception as exc:
        raise RuntimeError(f"Slack API {endpoint} request failed ({type(exc).__name__}).") from None

    if not payload.get("ok"):
        err_code = payload.get("error", "unknown_error")
        raise RuntimeError(f"Slack API {endpoint} failed with error={err_code!r}.")
    return payload


def collect_slack_channel(
    channel_id: str,
    limit: int = 100,
    include_threads: bool = True,
    max_threads: int = 15,
    privacy_mode: PrivacyMode = "features_only",
    timeout_sec: float = 10.0,
) -> ChannelCollectionResult:
    """Pull incremental Slack channel & thread messages and train pragmatics with privacy."""
    if not channel_id or not channel_id.strip():
        raise ValueError("channel_id must not be empty.")
    if limit < 1 or limit > 200:
        raise ValueError("limit must be between 1 and 200.")

    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "Missing SLACK_BOT_TOKEN in environment. Set SLACK_BOT_TOKEN to pull Slack messages."
        )

    cp_store = get_checkpoint_store()
    oldest_cursor = cp_store.state.slack_cursors.get(channel_id, "0")

    params: dict[str, str | int] = {
        "channel": channel_id,
        "limit": limit,
    }
    if oldest_cursor != "0":
        params["oldest"] = oldest_cursor

    history = _slack_api_get("conversations.history", params, token=token, timeout_sec=timeout_sec)
    raw_messages: list[dict[str, Any]] = history.get("messages", [])
    if not raw_messages:
        return ChannelCollectionResult(
            channel_type="slack",
            target_id=channel_id,
            fetched_messages=0,
            constructed_turns=0,
            checkpoint_cursor=oldest_cursor,
        )

    sorted_msgs = sorted(raw_messages, key=lambda m: float(m.get("ts", "0")))
    newest_ts = str(sorted_msgs[-1].get("ts", oldest_cursor))

    sessions: list[list[DialogueTurnInput]] = []
    main_turns: list[DialogueTurnInput] = []
    fetched_count = len(sorted_msgs)

    for msg in sorted_msgs:
        if msg.get("subtype"):
            continue
        text = str(msg.get("text", "")).strip()
        user = str(msg.get("user", "slack_user"))
        if _has_min_hangul(text):
            main_turns.append(DialogueTurnInput(speaker=user, text=text))

    if main_turns:
        sessions.append(main_turns)

    if include_threads and max_threads > 0:
        threads_fetched = 0
        for msg in sorted_msgs:
            reply_cnt = int(msg.get("reply_count", 0))
            thread_ts = msg.get("thread_ts") or msg.get("ts")
            if reply_cnt > 0 and thread_ts and threads_fetched < max_threads:
                threads_fetched += 1
                replies_payload = _slack_api_get(
                    "conversations.replies",
                    {"channel": channel_id, "ts": str(thread_ts), "limit": 50},
                    token=token,
                    timeout_sec=timeout_sec,
                )
                thread_msgs: list[dict[str, Any]] = replies_payload.get("messages", [])
                fetched_count += max(0, len(thread_msgs) - 1)
                thread_turns: list[DialogueTurnInput] = []
                for t_msg in thread_msgs:
                    t_text = str(t_msg.get("text", "")).strip()
                    t_user = str(t_msg.get("user", "slack_user"))
                    if _has_min_hangul(t_text):
                        thread_turns.append(DialogueTurnInput(speaker=t_user, text=t_text))
                if thread_turns:
                    sessions.append(thread_turns)

    total_turns = sum(len(s) for s in sessions)
    ingest_res: DialogueIngestionResult | None = None
    if sessions:
        ingest_res = get_corpus_engine().ingest_dialogue_sessions(
            sessions=sessions,
            privacy_mode=privacy_mode,
            source_tag="slack",
        )

    cp_store.state.slack_cursors[channel_id] = newest_ts
    cp_store.save()

    return ChannelCollectionResult(
        channel_type="slack",
        target_id=channel_id,
        fetched_messages=fetched_count,
        constructed_turns=total_turns,
        checkpoint_cursor=newest_ts,
        ingestion_result=ingest_res,
    )


def _extract_telegram_text(raw_text: Any) -> str:
    """Normalize Telegram text field (string or list of rich-text entity segments)."""
    if isinstance(raw_text, str):
        return raw_text.strip()
    if isinstance(raw_text, list):
        parts: list[str] = []
        for item in raw_text:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "".join(parts).strip()
    return ""


def collect_telegram_updates(
    limit: int = 100,
    privacy_mode: PrivacyMode = "features_only",
    timeout_sec: float = 10.0,
) -> ChannelCollectionResult:
    """Pull incremental Telegram Bot updates and train pragmatics with privacy."""
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100.")

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "Missing TELEGRAM_BOT_TOKEN in environment. Set TELEGRAM_BOT_TOKEN to pull updates."
        )

    cp_store = get_checkpoint_store()
    offset = cp_store.state.telegram_offset
    query = urllib.parse.urlencode({"offset": offset, "limit": limit, "timeout": 0})
    url = f"https://api.telegram.org/bot{token}/getUpdates?{query}"

    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # noqa: S310
            raw_bytes = resp.read(2_000_000)
            payload: dict[str, Any] = json.loads(raw_bytes.decode("utf-8", errors="replace"))
    except Exception as exc:
        raise RuntimeError(
            f"Telegram getUpdates request failed ({type(exc).__name__})."
        ) from None

    if not payload.get("ok"):
        desc = payload.get("description", "unknown_error")
        raise RuntimeError(f"Telegram getUpdates returned ok=False: {desc!r}.")

    updates: list[dict[str, Any]] = payload.get("result", [])
    if not updates:
        return ChannelCollectionResult(
            channel_type="telegram",
            target_id="bot_updates",
            fetched_messages=0,
            constructed_turns=0,
            checkpoint_cursor=str(offset),
        )

    max_update_id = offset
    chat_streams: dict[str, list[DialogueTurnInput]] = defaultdict(list)
    sessions: list[list[DialogueTurnInput]] = []

    for upd in updates:
        u_id = int(upd.get("update_id", 0))
        if u_id >= max_update_id:
            max_update_id = u_id + 1

        msg = upd.get("message") or upd.get("channel_post")
        if not isinstance(msg, dict):
            continue

        chat_id = str(msg.get("chat", {}).get("id", "default"))
        sender = str(msg.get("from", {}).get("id", "tg_user"))
        text = str(msg.get("text") or msg.get("caption") or "").strip()
        if not _has_min_hangul(text):
            continue

        reply_to = msg.get("reply_to_message")
        if isinstance(reply_to, dict):
            parent_text = str(reply_to.get("text") or reply_to.get("caption") or "").strip()
            parent_sender = str(reply_to.get("from", {}).get("id", "tg_parent"))
            if _has_min_hangul(parent_text):
                sessions.append(
                    [
                        DialogueTurnInput(speaker=parent_sender, text=parent_text),
                        DialogueTurnInput(speaker=sender, text=text),
                    ]
                )
                continue

        chat_streams[chat_id].append(DialogueTurnInput(speaker=sender, text=text))

    for stream in chat_streams.values():
        if stream:
            sessions.append(stream)

    total_turns = sum(len(s) for s in sessions)
    ingest_res: DialogueIngestionResult | None = None
    if sessions:
        ingest_res = get_corpus_engine().ingest_dialogue_sessions(
            sessions=sessions,
            privacy_mode=privacy_mode,
            source_tag="telegram",
        )

    cp_store.state.telegram_offset = max_update_id
    cp_store.save()

    return ChannelCollectionResult(
        channel_type="telegram",
        target_id="bot_updates",
        fetched_messages=len(updates),
        constructed_turns=total_turns,
        checkpoint_cursor=str(max_update_id),
        ingestion_result=ingest_res,
    )


def ingest_telegram_export(
    messages: list[dict[str, Any]],
    chat_label: str = "telegram_export",
    privacy_mode: PrivacyMode = "features_only",
) -> ChannelCollectionResult:
    """Ingest messages from a Telegram Desktop / Telethon JSON export with reply linking."""
    if not messages:
        raise ValueError("messages list must not be empty.")

    by_id: dict[int, tuple[str, str]] = {}
    sessions: list[list[DialogueTurnInput]] = []
    sequential_stream: list[DialogueTurnInput] = []

    for msg in messages:
        if msg.get("type", "message") != "message":
            continue
        msg_id = msg.get("id")
        sender = str(msg.get("from_id") or msg.get("from") or "tg_user")
        text = _extract_telegram_text(msg.get("text", ""))
        if not _has_min_hangul(text):
            continue

        if isinstance(msg_id, int):
            by_id[msg_id] = (sender, text)

        reply_id = msg.get("reply_to_message_id")
        if isinstance(reply_id, int) and reply_id in by_id:
            p_sender, p_text = by_id[reply_id]
            sessions.append(
                [
                    DialogueTurnInput(speaker=p_sender, text=p_text),
                    DialogueTurnInput(speaker=sender, text=text),
                ]
            )
        else:
            sequential_stream.append(DialogueTurnInput(speaker=sender, text=text))

    if sequential_stream:
        sessions.append(sequential_stream)

    total_turns = sum(len(s) for s in sessions)
    ingest_res: DialogueIngestionResult | None = None
    if sessions:
        ingest_res = get_corpus_engine().ingest_dialogue_sessions(
            sessions=sessions,
            privacy_mode=privacy_mode,
            source_tag="telegram_export",
        )

    return ChannelCollectionResult(
        channel_type="telegram_export",
        target_id=chat_label,
        fetched_messages=len(messages),
        constructed_turns=total_turns,
        checkpoint_cursor=f"count:{len(messages)}",
        ingestion_result=ingest_res,
    )


def collect_web_dialogues(
    url: str,
    privacy_mode: PrivacyMode = "public_full",
    also_mine_oov: bool = True,
    timeout_sec: float = 10.0,
) -> ChannelCollectionResult:
    """Fetch public web/RSS content, extract Korean dialogue turns, and train pragmatics + OOV."""
    _validate_url_and_resolved_ips(url)
    req = urllib.request.Request(url, headers=DEFAULT_WEB_HEADERS)
    with _safe_http_urlopen(req, timeout_sec=timeout_sec) as resp:
        raw_bytes = resp.read(2_000_000)
        charset = resp.headers.get_content_charset() or "utf-8"
        try:
            raw_text = raw_bytes.decode(charset)
        except (LookupError, UnicodeDecodeError):
            raw_text = raw_bytes.decode("cp949", errors="replace")

    content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:24]
    cp_store = get_checkpoint_store()
    if cp_store.state.web_content_hashes.get(url) == content_hash:
        return ChannelCollectionResult(
            channel_type="web",
            target_id=url,
            fetched_messages=0,
            constructed_turns=0,
            skipped_unchanged=True,
            checkpoint_cursor=content_hash,
        )

    blocks = html_or_xml_to_blocks(raw_text, min_hangul_syllables=5)
    if not blocks:
        raise ValueError(f"No Korean text blocks with >=5 Hangul syllables found at {url!r}.")

    turns = [
        DialogueTurnInput(speaker=f"web_turn_{idx % 2}", text=block)
        for idx, block in enumerate(blocks)
    ]
    ingest_res = get_corpus_engine().ingest_dialogue_turns(
        turns=turns,
        privacy_mode=privacy_mode,
        source_tag="web",
    )

    oov_added: list[str] = []
    if also_mine_oov and privacy_mode == "public_full":
        sentences = html_or_xml_to_sentences(raw_text, min_hangul_syllables=5)
        oov_res = get_learner().mine_corpus_oov_words(
            texts=sentences or blocks,
            min_cnt=2,
            min_pos_score=0.2,
        )
        oov_added = [w.word for w in oov_res.newly_registered_words]

    cp_store.state.web_content_hashes[url] = content_hash
    cp_store.save()

    return ChannelCollectionResult(
        channel_type="web",
        target_id=url,
        fetched_messages=len(blocks),
        constructed_turns=len(turns),
        checkpoint_cursor=content_hash,
        oov_words_added=oov_added,
        ingestion_result=ingest_res,
    )
