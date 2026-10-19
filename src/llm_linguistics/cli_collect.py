"""CLI batch runner (`llm-linguistics-collect`) for multi-channel Korean corpus collection."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from llm_linguistics.pragmatics.collectors import (
    ChannelCollectionResult,
    collect_slack_channel,
    collect_telegram_updates,
    collect_web_dialogues,
    ingest_telegram_export,
)
from llm_linguistics.pragmatics.corpus_learner import (
    REPO_DATA_DIR,
    PrivacyMode,
    get_corpus_engine,
)

DEFAULT_SOURCES_CONFIG_PATH = REPO_DATA_DIR / "collect_sources.json"

BatchStatus = Literal["healthy", "degraded", "failed", "empty"]


class CollectSourcesConfig(BaseModel):
    """Configuration file schema for batch multi-channel collection."""

    web_urls: list[str] = Field(default_factory=list)
    slack_channels: list[str] = Field(default_factory=list)
    enable_telegram_updates: bool = False
    telegram_export_files: list[str] = Field(default_factory=list)
    slack_privacy_mode: PrivacyMode = "features_only"
    telegram_privacy_mode: PrivacyMode = "features_only"
    web_privacy_mode: PrivacyMode = "public_full"


class SourceErrorRecord(BaseModel):
    """Structured error record for a single failed collection target."""

    channel_type: str
    target_id: str
    error_type: str
    reason: str


class BatchCollectionSummary(BaseModel):
    """Aggregate report of a multi-channel batch collection run."""

    status: BatchStatus
    attempted_sources: int
    succeeded_sources: int
    failed_sources: int
    total_appended_records: int
    total_bootstrapped_reactions: int
    total_oov_words_added: int
    total_corpus_records: int
    results: list[ChannelCollectionResult] = Field(default_factory=list)
    errors: list[SourceErrorRecord] = Field(default_factory=list)


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger("llm_linguistics.collect")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
    return logger


def _load_telegram_export_file(file_path: Path) -> list[dict[str, Any]]:
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [m for m in raw if isinstance(m, dict)]
    if isinstance(raw, dict) and isinstance(raw.get("messages"), list):
        return [m for m in raw["messages"] if isinstance(m, dict)]
    raise ValueError(
        f"Telegram export file {str(file_path)!r} must be a JSON list or contain a 'messages' list."
    )


def run_batch_collection(
    config: CollectSourcesConfig,
    timeout_sec: float = 10.0,
) -> BatchCollectionSummary:
    """Execute all configured collection sources, checkpointing each successful source."""
    logger = _configure_logger()
    results: list[ChannelCollectionResult] = []
    errors: list[SourceErrorRecord] = []

    attempted = (
        len(config.web_urls)
        + len(config.slack_channels)
        + (1 if config.enable_telegram_updates else 0)
        + len(config.telegram_export_files)
    )
    if attempted == 0:
        stats = get_corpus_engine().get_stats(top_k=3)
        return BatchCollectionSummary(
            status="empty",
            attempted_sources=0,
            succeeded_sources=0,
            failed_sources=0,
            total_appended_records=0,
            total_bootstrapped_reactions=0,
            total_oov_words_added=0,
            total_corpus_records=stats.total_corpus_records,
        )

    for url in config.web_urls:
        try:
            logger.info("Collecting web/RSS source target=%s", url)
            res = collect_web_dialogues(
                url=url,
                privacy_mode=config.web_privacy_mode,
                also_mine_oov=(config.web_privacy_mode == "public_full"),
                timeout_sec=timeout_sec,
            )
            results.append(res)
        except Exception as exc:
            logger.warning("Web source failed target=%s error=%s: %s", url, type(exc).__name__, exc)
            errors.append(
                SourceErrorRecord(
                    channel_type="web",
                    target_id=url,
                    error_type=type(exc).__name__,
                    reason=str(exc),
                )
            )

    for channel_id in config.slack_channels:
        try:
            logger.info("Collecting Slack channel target=%s", channel_id)
            res = collect_slack_channel(
                channel_id=channel_id,
                privacy_mode=config.slack_privacy_mode,
                timeout_sec=timeout_sec,
            )
            results.append(res)
        except Exception as exc:
            logger.warning(
                "Slack source failed target=%s error=%s: %s",
                channel_id,
                type(exc).__name__,
                exc,
            )
            errors.append(
                SourceErrorRecord(
                    channel_type="slack",
                    target_id=channel_id,
                    error_type=type(exc).__name__,
                    reason=str(exc),
                )
            )

    if config.enable_telegram_updates:
        try:
            logger.info("Collecting Telegram bot updates")
            res = collect_telegram_updates(
                privacy_mode=config.telegram_privacy_mode,
                timeout_sec=timeout_sec,
            )
            results.append(res)
        except Exception as exc:
            logger.warning("Telegram updates failed error=%s: %s", type(exc).__name__, exc)
            errors.append(
                SourceErrorRecord(
                    channel_type="telegram",
                    target_id="bot_updates",
                    error_type=type(exc).__name__,
                    reason=str(exc),
                )
            )

    for export_path_str in config.telegram_export_files:
        try:
            export_path = Path(export_path_str).expanduser().resolve()
            logger.info("Ingesting Telegram export file=%s", export_path.name)
            messages = _load_telegram_export_file(export_path)
            res = ingest_telegram_export(
                messages=messages,
                chat_label=export_path.name,
                privacy_mode=config.telegram_privacy_mode,
            )
            results.append(res)
        except Exception as exc:
            logger.warning(
                "Telegram export failed file=%s error=%s: %s",
                export_path_str,
                type(exc).__name__,
                exc,
            )
            errors.append(
                SourceErrorRecord(
                    channel_type="telegram_export",
                    target_id=export_path_str,
                    error_type=type(exc).__name__,
                    reason=str(exc),
                )
            )

    succeeded = len(results)
    failed = len(errors)
    if failed == 0:
        status: BatchStatus = "healthy"
    elif succeeded > 0:
        status = "degraded"
    else:
        status = "failed"

    total_appended = sum(
        r.ingestion_result.appended_records if r.ingestion_result else 0 for r in results
    )
    total_bootstrapped = sum(
        r.ingestion_result.bootstrapped_from_reactions if r.ingestion_result else 0
        for r in results
    )
    total_oov = sum(len(r.oov_words_added) for r in results)
    stats = get_corpus_engine().get_stats(top_k=3)

    logger.info(
        "Batch collection complete status=%s succeeded=%d failed=%d appended=%d total_corpus=%d",
        status,
        succeeded,
        failed,
        total_appended,
        stats.total_corpus_records,
    )
    return BatchCollectionSummary(
        status=status,
        attempted_sources=attempted,
        succeeded_sources=succeeded,
        failed_sources=failed,
        total_appended_records=total_appended,
        total_bootstrapped_reactions=total_bootstrapped,
        total_oov_words_added=total_oov,
        total_corpus_records=stats.total_corpus_records,
        results=results,
        errors=errors,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-linguistics-collect",
        description="Batch multi-channel Korean dialogue & OOV collector with privacy separation.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON sources config file (default: data/collect_sources.json).",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Public web or RSS URL to collect (can be specified multiple times).",
    )
    parser.add_argument(
        "--slack-channel",
        action="append",
        default=[],
        help="Slack channel ID to pull via SLACK_BOT_TOKEN (can be specified multiple times).",
    )
    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Pull incremental Telegram Bot updates via TELEGRAM_BOT_TOKEN.",
    )
    parser.add_argument(
        "--telegram-export",
        action="append",
        default=[],
        help="Path to Telegram Desktop JSON export file (result.json).",
    )
    parser.add_argument(
        "--privacy-mode",
        choices=["features_only", "private_scrubbed", "public_full"],
        default=None,
        help="Override privacy mode for Slack/Telegram channels (default: features_only).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP request timeout in seconds per source (default: 10.0).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 even on partial (degraded) source failures.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for llm-linguistics-collect."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    has_cli_targets = bool(
        args.url or args.slack_channel or args.telegram or args.telegram_export
    )

    if args.config:
        cfg_path = Path(args.config).expanduser().resolve()
        config = CollectSourcesConfig.model_validate_json(cfg_path.read_text(encoding="utf-8"))
    elif not has_cli_targets and DEFAULT_SOURCES_CONFIG_PATH.exists():
        config = CollectSourcesConfig.model_validate_json(
            DEFAULT_SOURCES_CONFIG_PATH.read_text(encoding="utf-8")
        )
    else:
        config = CollectSourcesConfig()

    if args.url:
        config.web_urls.extend(args.url)
    if args.slack_channel:
        config.slack_channels.extend(args.slack_channel)
    if args.telegram:
        config.enable_telegram_updates = True
    if args.telegram_export:
        config.telegram_export_files.extend(args.telegram_export)
    if args.privacy_mode is not None:
        config.slack_privacy_mode = args.privacy_mode
        config.telegram_privacy_mode = args.privacy_mode

    summary = run_batch_collection(config=config, timeout_sec=args.timeout)
    sys.stdout.write(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2) + "\n")
    sys.stdout.flush()

    if summary.status == "failed":
        return 1
    if summary.status == "degraded" and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
