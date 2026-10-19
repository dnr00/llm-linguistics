"""Autonomous background agent + LLM Teacher distillation (`llm-linguistics-autopilot`)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import plistlib
import re
import shutil
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from llm_linguistics.cli_collect import (
    DEFAULT_SOURCES_CONFIG_PATH,
    CollectSourcesConfig,
)
from llm_linguistics.learner.online_fetcher import html_or_xml_to_blocks, html_or_xml_to_sentences
from llm_linguistics.learner.schemas import PragmaticCategory
from llm_linguistics.learner.store import get_learner
from llm_linguistics.pragmatics.analyzer import analyze_discourse
from llm_linguistics.pragmatics.collectors import (
    _safe_http_urlopen,
    _validate_url_and_resolved_ips,
    collect_slack_channel,
    collect_telegram_updates,
    get_checkpoint_store,
    ingest_telegram_export,
)
from llm_linguistics.pragmatics.corpus_learner import (
    TARGET_PRAGMATIC_CLASSES,
    PragmaticCorpusRecord,
    PrivacyMode,
    get_corpus_engine,
    scrub_pii,
)
from llm_linguistics.pragmatics.schemas import DialogueTurnInput, IllocutionaryForce

DEFAULT_AUTOPILOT_STATE_PATH = Path.home() / ".llm_linguistics" / "autopilot_state.json"
DEFAULT_AUTOPILOT_LOG_PATH = Path.home() / ".llm_linguistics" / "autopilot.log"
LAUNCHD_LABEL = "com.uksang.llm-linguistics-autopilot"
DEFAULT_LAUNCHD_PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"

MAX_CONSECUTIVE_FAILURES = 3
MAX_UTTERANCES_PER_LLM_CALL = 20

AutopilotStatus = Literal[
    "distilled",
    "idle_no_new_data",
    "circuit_open",
    "failed",
]

VALID_PRAGMATIC_CATEGORIES: frozenset[str] = frozenset({
    "indirect_refusal_cue",
    "soft_urge_cue",
    "hedged_dissent_cue",
    "pressure_blame_cue",
})

VALID_MODALITIES: frozenset[str] = frozenset({"MUST", "MUST_NOT", "MAY"})


class DistilledUtteranceItem(BaseModel):
    """Single utterance pragmatic classification returned by the LLM Teacher."""

    index: int
    label: IllocutionaryForce
    confidence: float = Field(ge=0.0, le=1.0)


class DistilledCueItem(BaseModel):
    """Newly discovered pragmatic cue phrase returned by the LLM Teacher."""

    phrase: str
    category: PragmaticCategory


class DistilledSemanticItem(BaseModel):
    """Newly discovered deontic modality pattern returned by the LLM Teacher."""

    pattern: str
    modality: Literal["MUST", "MUST_NOT", "MAY"]
    is_exclusive: bool = False


class TeacherDistillationPayload(BaseModel):
    """Structured JSON output schema expected from the Claude LLM Teacher."""

    pragmatic_labels: list[DistilledUtteranceItem] = Field(default_factory=list)
    pragmatic_cues: list[DistilledCueItem] = Field(default_factory=list)
    semantic_patterns: list[DistilledSemanticItem] = Field(default_factory=list)


class AutopilotState(BaseModel):
    """Durable state and circuit-breaker accounting for autonomous LLM distillation."""

    circuit_open: bool = False
    consecutive_failures: int = 0
    last_error: str = ""
    last_run_at: str = ""
    total_runs: int = 0
    total_llm_calls: int = 0
    total_distilled_utterances: int = 0
    total_distilled_cues: int = 0
    total_distilled_patterns: int = 0
    total_oov_words: int = 0


class AutopilotRunReport(BaseModel):
    """Result report of a single autonomous collection + LLM distillation cycle."""

    status: AutopilotStatus
    candidate_utterances_collected: int
    llm_calls_made: int
    distilled_utterances_added: int
    distilled_cues_added: int
    distilled_patterns_added: int
    oov_words_added: int
    total_corpus_records: int
    error: str = ""
    state: AutopilotState


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger("llm_linguistics.autopilot")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger


class AutopilotStateStore:
    """Atomic persistence for AutopilotState and circuit-breaker counters."""

    def __init__(self, path: Path | str | None = None) -> None:
        raw = path or os.environ.get(
            "LLM_LINGUISTICS_AUTOPILOT_STATE_PATH",
            DEFAULT_AUTOPILOT_STATE_PATH,
        )
        self.path = Path(raw).expanduser().resolve()
        self.state = self._load()

    def _load(self) -> AutopilotState:
        if not self.path.exists():
            return AutopilotState()
        return AutopilotState.model_validate_json(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self.state.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass


_AUTOPILOT_STORE_SINGLETON: AutopilotStateStore | None = None


def get_autopilot_store() -> AutopilotStateStore:
    """Return process-wide singleton AutopilotStateStore."""
    global _AUTOPILOT_STORE_SINGLETON
    if _AUTOPILOT_STORE_SINGLETON is None:
        _AUTOPILOT_STORE_SINGLETON = AutopilotStateStore()
    return _AUTOPILOT_STORE_SINGLETON


def reset_autopilot_store_for_testing(
    path: Path | str | None = None,
) -> AutopilotStateStore:
    """Reinitialize singleton AutopilotStateStore for isolated tests."""
    global _AUTOPILOT_STORE_SINGLETON
    _AUTOPILOT_STORE_SINGLETON = AutopilotStateStore(path=path)
    return _AUTOPILOT_STORE_SINGLETON


def build_billing_isolated_env(
    base_env: dict[str, str] | None = None,
    claude_settings_path: Path | None = None,
) -> dict[str, str]:
    """Build child env stripped of ANTHROPIC_*/CLAUDE_CODE_USE_* per Claude Max billing guard."""
    source = dict(base_env if base_env is not None else os.environ)
    scrubbed: dict[str, str] = {}

    for key, val in source.items():
        if key.startswith("ANTHROPIC_"):
            continue
        if key.startswith("CLAUDE_CODE_USE_"):
            continue
        if key.startswith("CLAUDE") and key != "CLAUDE_CODE_OAUTH_TOKEN":
            continue
        if key in ("NODE_OPTIONS", "ELECTRON_RUN_AS_NODE"):
            continue
        scrubbed[key] = val

    local_bin = str(Path.home() / ".local" / "bin")
    current_path = scrubbed.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    if local_bin not in current_path.split(":"):
        scrubbed["PATH"] = f"{local_bin}:{current_path}"
    scrubbed["PYTHONUNBUFFERED"] = "1"

    # Also verify ~/.claude/settings.json does not inject an API key or cloud provider route
    s_path = claude_settings_path or (Path.home() / ".claude" / "settings.json")
    if s_path.exists():
        try:
            settings_data = json.loads(s_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Billing guard tripped: failed to parse {s_path} as valid JSON."
            ) from exc
        if not isinstance(settings_data, dict):
            raise RuntimeError(
                f"Billing guard tripped: {s_path} root must be a JSON object."
            )
        if settings_data.get("apiKeyHelper"):
            raise RuntimeError(
                f"Billing guard tripped: apiKeyHelper is configured in {s_path}. "
                "Refusing to spawn child claude process outside subscription."
            )
        settings_env = settings_data.get("env", {})
        if settings_env is not None and not isinstance(settings_env, dict):
            raise RuntimeError(
                f"Billing guard tripped: {s_path} 'env' field must be a JSON object."
            )
        if isinstance(settings_env, dict):
            for f_key, f_val in settings_env.items():
                if f_key.startswith(("ANTHROPIC_", "CLAUDE_CODE_USE_")) and f_val:
                    raise RuntimeError(
                        f"Billing guard tripped: {f_key} is set in {s_path}. "
                        "Refusing to spawn child claude process outside subscription."
                    )

    return scrubbed


def resolve_claude_binary(env: dict[str, str]) -> str:
    """Resolve the installed `claude` CLI executable on PATH."""
    found = shutil.which("claude", path=env.get("PATH"))
    if found:
        return found
    fallback = Path.home() / ".local" / "bin" / "claude"
    if fallback.exists():
        return str(fallback)
    raise RuntimeError("Could not find `claude` CLI executable on PATH or ~/.local/bin/claude.")


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM response string or envelope."""
    stripped = raw_text.strip()
    try:
        outer = json.loads(stripped)
        if isinstance(outer, dict):
            if outer.get("is_error"):
                detail = outer.get("subtype") or outer.get("result") or outer
                raise RuntimeError(f"Claude teacher returned error envelope: {detail}")
            # If this is `claude -p --output-format json` envelope, unwrap `result`
            if "result" in outer and isinstance(outer["result"], str):
                inner_text = outer["result"].strip()
                return _extract_json_object(inner_text)
            if outer.get("type") == "result":
                detail = outer.get("subtype") or outer
                raise RuntimeError(f"Claude teacher envelope missing string result: {detail}")
            return outer
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        parsed_fenced = json.loads(fenced.group(1))
        if isinstance(parsed_fenced, dict):
            return parsed_fenced

    brace_match = re.search(r"(\{.*\})", stripped, flags=re.DOTALL)
    if brace_match:
        parsed_brace = json.loads(brace_match.group(1))
        if isinstance(parsed_brace, dict):
            return parsed_brace

    raise ValueError("LLM response did not contain a valid JSON object.")


def build_teacher_prompt(utterances: list[str]) -> str:
    """Construct single-turn distillation prompt for Korean pragmatics and semantics."""
    numbered = "\n".join(f"[{idx}] {text}" for idx, text in enumerate(utterances))
    return (
        "You are a Korean linguistics teacher model distilling knowledge into a local CPU "
        "morphological analyzer (`llm-linguistics`). Analyze the Korean sentences below.\n\n"
        "Return ONLY a single raw JSON object (no prose, no markdown fences) matching this exact "
        "schema:\n"
        "{\n"
        '  "pragmatic_labels": [\n'
        '    {"index": <int>, "label": "<indirect_refusal_or_deferral | indirect_request_or_urge | '
        'hedged_dissent_or_reservation | direct_pressure_or_blame>", "confidence": <float>}\n'
        "  ],\n"
        '  "pragmatic_cues": [\n'
        '    {"phrase": "<short Korean pragmatic marker substring>", "category": '
        '"<indirect_refusal_cue | soft_urge_cue | hedged_dissent_cue | pressure_blame_cue>"}\n'
        "  ],\n"
        '  "semantic_patterns": [\n'
        '    {"pattern": "<Korean deontic/policy phrase>", "modality": "<MUST | MUST_NOT | MAY>", '
        '"is_exclusive": <bool>}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "1. Only include `pragmatic_labels` for sentences that clearly exhibit one of the 4 "
        "non-literal speech acts with confidence >= 0.75. Omit purely literal news/facts.\n"
        "2. Only include generalizable multi-character Korean phrases in `pragmatic_cues` and "
        "`semantic_patterns` (never proper nouns, names, or numbers).\n\n"
        f"Sentences:\n{numbered}\n"
    )


def invoke_claude_teacher(
    utterances: list[str],
    timeout_sec: float = 60.0,
) -> TeacherDistillationPayload:
    """Invoke billing-isolated `claude -p` in 1-turn mode to label Korean utterances."""
    if not utterances:
        return TeacherDistillationPayload()

    env = build_billing_isolated_env()
    claude_bin = resolve_claude_binary(env)
    prompt = build_teacher_prompt(utterances[:MAX_UTTERANCES_PER_LLM_CALL])

    cmd = [
        claude_bin,
        "-p",
        prompt,
        "--output-format",
        "json",
        "--max-turns",
        "1",
        "--tools",
        "",
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        start_new_session=True,
    )
    try:
        stdout_str, stderr_str = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait()
        raise RuntimeError(f"Claude teacher timed out after {timeout_sec}s.") from exc

    if proc.returncode != 0:
        err_snip = (stderr_str or stdout_str or "").strip()[:240]
        raise RuntimeError(f"Claude teacher exited with code {proc.returncode}: {err_snip}")

    raw_dict = _extract_json_object(stdout_str)
    expected_keys = ("pragmatic_labels", "pragmatic_cues", "semantic_patterns")
    if not any(k in raw_dict for k in expected_keys):
        raise RuntimeError("LLM response JSON is missing expected distillation schema keys.")

    raw_labels = raw_dict.get("pragmatic_labels") or []
    raw_cues = raw_dict.get("pragmatic_cues") or []
    raw_patterns = raw_dict.get("semantic_patterns") or []
    if not (
        isinstance(raw_labels, list)
        and isinstance(raw_cues, list)
        and isinstance(raw_patterns, list)
    ):
        raise RuntimeError("LLM response distillation schema fields must be lists.")

    # Filter to valid enum values before validation so minor teacher hallucinations don't crash
    clean_labels: list[dict[str, Any]] = []
    for item in raw_labels:
        if (
            isinstance(item, dict)
            and isinstance(item.get("index"), int)
            and 0 <= item["index"] < len(utterances)
            and item.get("label") in TARGET_PRAGMATIC_CLASSES
        ):
            clean_labels.append(item)

    clean_cues: list[dict[str, Any]] = []
    for item in raw_cues:
        if (
            isinstance(item, dict)
            and isinstance(item.get("phrase"), str)
            and len(item["phrase"].strip()) >= 2
            and item.get("category") in VALID_PRAGMATIC_CATEGORIES
        ):
            clean_cues.append({"phrase": item["phrase"].strip(), "category": item["category"]})

    clean_patterns: list[dict[str, Any]] = []
    for item in raw_patterns:
        if (
            isinstance(item, dict)
            and isinstance(item.get("pattern"), str)
            and len(item["pattern"].strip()) >= 2
            and item.get("modality") in VALID_MODALITIES
        ):
            clean_patterns.append({
                "pattern": item["pattern"].strip(),
                "modality": item["modality"],
                "is_exclusive": bool(item.get("is_exclusive", False)),
            })

    return TeacherDistillationPayload.model_validate({
        "pragmatic_labels": clean_labels,
        "pragmatic_cues": clean_cues,
        "semantic_patterns": clean_patterns,
    })


def run_autopilot_cycle(
    config: CollectSourcesConfig | None = None,
    max_utterances: int = MAX_UTTERANCES_PER_LLM_CALL,
    http_timeout_sec: float = 10.0,
    llm_timeout_sec: float = 60.0,
) -> AutopilotRunReport:
    """Execute one bounded autonomous cycle: fetch -> LLM teacher -> local distillation."""
    logger = _configure_logger()
    ap_store = get_autopilot_store()

    if ap_store.state.circuit_open:
        logger.warning(
            "Autopilot circuit breaker is OPEN (consecutive_failures=%d). Skipping LLM run.",
            ap_store.state.consecutive_failures,
        )
        stats = get_corpus_engine().get_stats(top_k=1)
        return AutopilotRunReport(
            status="circuit_open",
            candidate_utterances_collected=0,
            llm_calls_made=0,
            distilled_utterances_added=0,
            distilled_cues_added=0,
            distilled_patterns_added=0,
            oov_words_added=0,
            total_corpus_records=stats.total_corpus_records,
            error=ap_store.state.last_error,
            state=ap_store.state,
        )

    if config is None:
        if DEFAULT_SOURCES_CONFIG_PATH.exists():
            config = CollectSourcesConfig.model_validate_json(
                DEFAULT_SOURCES_CONFIG_PATH.read_text(encoding="utf-8")
            )
        else:
            config = CollectSourcesConfig()

    cp_store = get_checkpoint_store()
    corpus_engine = get_corpus_engine()
    learner = get_learner()

    effective_limit = max(1, min(max_utterances, MAX_UTTERANCES_PER_LLM_CALL))
    candidate_utterances: list[tuple[str, PrivacyMode]] = []
    pending_web_hashes: dict[str, str] = {}
    oov_added_count = 0

    # 1. Fetch new Web/RSS blocks and extract unlabelled candidates
    for url in config.web_urls:
        if len(candidate_utterances) >= effective_limit:
            break
        try:
            _validate_url_and_resolved_ips(url)
            req = __import__("urllib.request").request.Request(
                url,
                headers={"User-Agent": "llm-linguistics-autopilot/0.4"},
            )
            with _safe_http_urlopen(req, timeout_sec=http_timeout_sec) as resp:
                raw_bytes = resp.read(2_000_000)
                charset = resp.headers.get_content_charset() or "utf-8"
                raw_text = raw_bytes.decode(charset, errors="replace")

            content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:24]
            if cp_store.state.web_content_hashes.get(url) == content_hash:
                continue

            blocks = html_or_xml_to_blocks(raw_text, min_hangul_syllables=8)
            sentences = html_or_xml_to_sentences(raw_text, min_hangul_syllables=6)
            if config.web_privacy_mode == "public_full" and sentences:
                oov_res = learner.mine_corpus_oov_words(
                    texts=sentences, min_cnt=2, min_pos_score=0.2
                )
                oov_added_count += len(oov_res.newly_registered_words)

            started_with_prior = len(candidate_utterances) > 0
            truncated_mid_url = False
            for s_text in (sentences or blocks):
                scrubbed = scrub_pii(s_text)
                if len(scrubbed) < 8 or len(scrubbed) > 160:
                    continue
                # Check if already classified by deterministic analyzer
                det = analyze_discourse(
                    [DialogueTurnInput(speaker="web", text=scrubbed)],
                    auto_harvest_reactions=False,
                )
                if det.utterances[0].illocutionary_force.startswith("literal_"):
                    if len(candidate_utterances) >= effective_limit:
                        truncated_mid_url = True
                        break
                    candidate_utterances.append((scrubbed, config.web_privacy_mode))

            # If a secondary URL was truncated only because an earlier URL took part of the batch,
            # leave its hash un-checkpointed so the next cycle gives it a full batch.
            if not (started_with_prior and truncated_mid_url):
                pending_web_hashes[url] = content_hash
        except Exception as exc:
            logger.warning("Autopilot web source skipped url=%s: %s", url, exc)

    # 2. Optional Slack / Telegram / Export incremental collection
    for ch_id in config.slack_channels:
        try:
            collect_slack_channel(
                channel_id=ch_id,
                privacy_mode=config.slack_privacy_mode,
                timeout_sec=http_timeout_sec,
            )
        except Exception as exc:
            logger.warning("Autopilot Slack channel skipped id=%s: %s", ch_id, exc)

    if config.enable_telegram_updates:
        try:
            collect_telegram_updates(
                privacy_mode=config.telegram_privacy_mode,
                timeout_sec=http_timeout_sec,
            )
        except Exception as exc:
            logger.warning("Autopilot Telegram updates skipped: %s", exc)

    for exp_file in config.telegram_export_files:
        try:
            p = Path(exp_file).expanduser().resolve()
            raw = json.loads(p.read_text(encoding="utf-8"))
            msgs = raw.get("messages", []) if isinstance(raw, dict) else raw
            if isinstance(msgs, list):
                ingest_telegram_export(
                    messages=msgs,
                    chat_label=p.name,
                    privacy_mode=config.telegram_privacy_mode,
                )
        except Exception as exc:
            logger.warning("Autopilot Telegram export skipped file=%s: %s", exp_file, exc)

    now_iso = datetime.now(UTC).isoformat()
    batch = candidate_utterances[:effective_limit]

    if not batch:
        for u, h in pending_web_hashes.items():
            cp_store.state.web_content_hashes[u] = h
        if pending_web_hashes:
            cp_store.save()

        ap_store.state.total_runs += 1
        ap_store.state.total_oov_words += oov_added_count
        ap_store.state.last_run_at = now_iso
        ap_store.save()
        stats = corpus_engine.get_stats(top_k=1)
        logger.info("Autopilot idle: 0 new unlabelled utterances (0 LLM calls made).")
        return AutopilotRunReport(
            status="idle_no_new_data",
            candidate_utterances_collected=0,
            llm_calls_made=0,
            distilled_utterances_added=0,
            distilled_cues_added=0,
            distilled_patterns_added=0,
            oov_words_added=oov_added_count,
            total_corpus_records=stats.total_corpus_records,
            state=ap_store.state,
        )

    # 3. Single bounded LLM Teacher call (max 1 call per cycle)
    try:
        texts_only = [t for t, _ in batch]
        logger.info("Invoking Claude Teacher on %d candidate utterances", len(texts_only))
        distilled = invoke_claude_teacher(texts_only, timeout_sec=llm_timeout_sec)
    except Exception as exc:
        ap_store.state.consecutive_failures += 1
        ap_store.state.last_error = str(exc)
        ap_store.state.last_run_at = now_iso
        ap_store.state.total_runs += 1
        if ap_store.state.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            ap_store.state.circuit_open = True
        ap_store.save()
        stats = corpus_engine.get_stats(top_k=1)
        logger.error(
            "Autopilot LLM distillation failed (consecutive=%d): %s",
            ap_store.state.consecutive_failures,
            exc,
        )
        return AutopilotRunReport(
            status="failed",
            candidate_utterances_collected=len(batch),
            llm_calls_made=1,
            distilled_utterances_added=0,
            distilled_cues_added=0,
            distilled_patterns_added=0,
            oov_words_added=oov_added_count,
            total_corpus_records=stats.total_corpus_records,
            error=str(exc),
            state=ap_store.state,
        )

    # 4. Persist distilled knowledge into local student models
    added_utterances = 0
    by_mode: dict[PrivacyMode, list[PragmaticCorpusRecord]] = {
        "public_full": [],
        "private_scrubbed": [],
        "features_only": [],
    }
    for item in distilled.pragmatic_labels:
        if item.confidence < 0.75 or not (0 <= item.index < len(batch)):
            continue
        u_text, p_mode = batch[item.index]
        by_mode[p_mode].append(
            PragmaticCorpusRecord(
                speaker="llm_teacher",
                text=u_text,
                label=item.label,
                source="autopilot_claude_teacher",
            )
        )

    for p_mode, recs in by_mode.items():
        if recs:
            added_utterances += corpus_engine.append_records(
                recs, privacy_mode=p_mode, retrain=True
            )

    added_cues = 0
    for cue in distilled.pragmatic_cues:
        learner.record_pragmatic_feedback(
            phrase=cue.phrase,
            category=cue.category,
            is_positive=True,
        )
        added_cues += 1

    added_patterns = 0
    for pat in distilled.semantic_patterns:
        learner.register_semantic_pattern(
            pattern=pat.pattern,
            modality=pat.modality,
            is_exclusive=pat.is_exclusive,
        )
        added_patterns += 1

    # 5. Checkpoint web hashes and reset circuit breaker after durable success
    for u, h in pending_web_hashes.items():
        cp_store.state.web_content_hashes[u] = h
    if pending_web_hashes:
        cp_store.save()

    ap_store.state.consecutive_failures = 0
    ap_store.state.circuit_open = False
    ap_store.state.last_error = ""
    ap_store.state.last_run_at = now_iso
    ap_store.state.total_runs += 1
    ap_store.state.total_llm_calls += 1
    ap_store.state.total_distilled_utterances += added_utterances
    ap_store.state.total_distilled_cues += added_cues
    ap_store.state.total_distilled_patterns += added_patterns
    ap_store.state.total_oov_words += oov_added_count
    ap_store.save()

    stats = corpus_engine.get_stats(top_k=1)
    logger.info(
        "Autopilot cycle complete utterances=+%d cues=+%d patterns=+%d oov=+%d",
        added_utterances,
        added_cues,
        added_patterns,
        oov_added_count,
    )
    return AutopilotRunReport(
        status="distilled",
        candidate_utterances_collected=len(batch),
        llm_calls_made=1,
        distilled_utterances_added=added_utterances,
        distilled_cues_added=added_cues,
        distilled_patterns_added=added_patterns,
        oov_words_added=oov_added_count,
        total_corpus_records=stats.total_corpus_records,
        state=ap_store.state,
    )


def install_launchd_agent(
    interval_sec: int = 14400,
    plist_path: Path | None = None,
    load_now: bool = True,
) -> Path:
    """Write and bootstrap macOS launchd user agent for zero-touch background execution."""
    if interval_sec < 300:
        raise ValueError("interval_sec must be at least 300 seconds (5 minutes).")

    target_plist = plist_path or DEFAULT_LAUNCHD_PLIST_PATH
    target_plist.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_AUTOPILOT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[2]
    venv_bin = repo_root / ".venv" / "bin" / "llm-linguistics-autopilot"
    program_args = (
        [str(venv_bin)]
        if venv_bin.exists()
        else ["uv", "--directory", str(repo_root), "run", "llm-linguistics-autopilot"]
    )

    plist_payload: dict[str, Any] = {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": program_args,
        "WorkingDirectory": str(repo_root),
        "RunAtLoad": True,
        "StartInterval": interval_sec,
        "StandardOutPath": str(DEFAULT_AUTOPILOT_LOG_PATH),
        "StandardErrorPath": str(DEFAULT_AUTOPILOT_LOG_PATH),
        "EnvironmentVariables": {
            "PATH": (
                f"{Path.home() / '.local' / 'bin'}:/opt/homebrew/bin:"
                "/usr/local/bin:/usr/bin:/bin"
            ),
            "PYTHONUNBUFFERED": "1",
        },
    }

    target_plist.write_bytes(plistlib.dumps(plist_payload))

    if load_now and sys.platform == "darwin":
        uid = os.getuid()
        subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}/{LAUNCHD_LABEL}"],
            check=False,
            capture_output=True,
        )
        subprocess.run(
            ["launchctl", "bootstrap", f"gui/{uid}", str(target_plist)],
            check=False,
            capture_output=True,
        )
    return target_plist


def uninstall_launchd_agent(plist_path: Path | None = None) -> bool:
    """Unload and remove macOS launchd user agent."""
    target_plist = plist_path or DEFAULT_LAUNCHD_PLIST_PATH
    if sys.platform == "darwin":
        uid = os.getuid()
        subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}/{LAUNCHD_LABEL}"],
            check=False,
            capture_output=True,
        )
    if target_plist.exists():
        target_plist.unlink()
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `llm-linguistics-autopilot`."""
    parser = argparse.ArgumentParser(
        prog="llm-linguistics-autopilot",
        description="Zero-touch background collector & Claude Teacher distillation daemon.",
    )
    parser.add_argument(
        "--install-launchd",
        action="store_true",
        help="Install and start macOS launchd background agent (runs every --interval-sec).",
    )
    parser.add_argument(
        "--uninstall-launchd",
        action="store_true",
        help="Stop and remove macOS launchd background agent.",
    )
    parser.add_argument(
        "--interval-sec",
        type=int,
        default=14400,
        help="Background launchd run interval in seconds (default: 14400 = 4 hours).",
    )
    parser.add_argument(
        "--reset-circuit",
        action="store_true",
        help="Reset circuit breaker after fixing an upstream LLM error.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current AutopilotState JSON without running a cycle.",
    )
    args = parser.parse_args(argv)

    ap_store = get_autopilot_store()
    if args.reset_circuit:
        ap_store.state.circuit_open = False
        ap_store.state.consecutive_failures = 0
        ap_store.state.last_error = ""
        ap_store.save()

    if args.install_launchd:
        plist = install_launchd_agent(interval_sec=args.interval_sec)
        sys.stdout.write(
            json.dumps({"installed_launchd_plist": str(plist), "interval_sec": args.interval_sec})
            + "\n"
        )
        return 0

    if args.uninstall_launchd:
        removed = uninstall_launchd_agent()
        sys.stdout.write(json.dumps({"uninstalled_launchd": removed}) + "\n")
        return 0

    if args.status:
        sys.stdout.write(
            json.dumps(ap_store.state.model_dump(), ensure_ascii=False, indent=2) + "\n"
        )
        return 0

    report = run_autopilot_cycle()
    sys.stdout.write(json.dumps(report.model_dump(), ensure_ascii=False, indent=2) + "\n")
    sys.stdout.flush()
    return 0 if report.status in ("distilled", "idle_no_new_data") else 1


if __name__ == "__main__":
    raise SystemExit(main())
