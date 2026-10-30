#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 AUTONOMOUS END-TO-END YOUTUBE VIRALITY PREDICTOR
================================================================================
 A single-file, self-sustaining pipeline that:

   STAGE 1  INGEST   (age 0-2h)   Discover brand-new, non-Short videos and
                                  persist them with is_viral = "PENDING".
   STAGE 2  TRACK    (age 8h)     Re-fetch statistics in BATCHED requests and
                                  engineer the 5 model features at this mark.
   STAGE 3  TRACK    (age 24h)    Re-fetch statistics (analysis / future work,
                                  deliberately NOT used as model input).
   STAGE 4  LABEL    (age 72h)    "The Teacher" applies the ground-truth rule
                                  and writes is_viral = 1 / 0.
   STAGE 5  TRAIN    (every cycle) "The Student" — a PyTorch MLP (9-32-16-1)
                                  retrained from scratch on every labelled row.
   STAGE 6  PREDICT  (every cycle) Score videos that have just crossed the 8h
                                  mark and write the probability back to Excel.

 The whole thing runs forever inside a `while True:` loop that sleeps
 CYCLE_SLEEP_SECONDS (2 hours) between cycles.

--------------------------------------------------------------------------------
 THE CENTRAL DESIGN DECISION — "THE 8-HOUR SHIFT"
--------------------------------------------------------------------------------
 Raw 0-2h statistics are garbage as predictors: a 40-minute-old video may have
 12 views purely because the subscriber feed has not fanned out yet. Variance at
 that age is dominated by publishing luck, not by content quality. We therefore
 use the *8-hour* observation as the model's baseline: it is late enough for the
 algorithmic distribution signal to appear, and early enough that a prediction
 still has business value (64 hours before the label exists).

--------------------------------------------------------------------------------
 ZERO-DATA-LEAKAGE CONTRACT (enforced in code, not just in prose)
--------------------------------------------------------------------------------
 The model may ONLY ever see the columns listed in FEATURE_COLUMNS, all of
 which are computed strictly from the 8h observation plus static metadata known
 at upload time. The 24h columns and every *_72h / final_* column are physically
 excluded by `assert_no_leakage()`, which raises at runtime if anyone ever adds a
 forbidden column to the feature list.

--------------------------------------------------------------------------------
 REQUIREMENTS
--------------------------------------------------------------------------------
   pip install google-api-python-client pandas openpyxl scikit-learn torch joblib

   export YOUTUBE_API_KEY="AIza..."          # Linux / macOS
   setx    YOUTUBE_API_KEY "AIza..."         # Windows

 USAGE
   python youtube_virality_engine.py                 # run forever (2h cycles)
   python youtube_virality_engine.py --once          # single cycle (debugging)
   python youtube_virality_engine.py --once --no-ingest   # no quota spent on search
   python youtube_virality_engine.py --self-test     # offline logic check, no API
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --- ML stack (optional at import time) --------------------------------------
# Stages 1-4 (collect, track, label) need no machine-learning library at all.
# Keeping these imports optional lets the collector run on a minimal host — a CI
# runner or a small VPS — with pandas and the Google client only, while the
# training host installs the full requirements.txt. Missing libraries disable
# stages 5-6 with a clear log line instead of crashing the whole engine.
try:
    import joblib
    import torch
    import torch.nn as nn
    from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
    ML_IMPORT_ERROR = ""
except ImportError as _ml_err:  # pragma: no cover - depends on the host
    ML_AVAILABLE = False
    ML_IMPORT_ERROR = str(_ml_err)

# --- Google API stack -------------------------------------------------------
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# ==============================================================================
#  SECTION 1 — CONFIGURATION
#  Everything tunable lives here. No magic numbers are buried in the logic.
# ==============================================================================

# Every output lands next to this file, never in the current working directory.
# Without this, launching from "/" (Finder, VS Code with no folder open, cron)
# tries to write to a read-only filesystem and the engine dies at startup.
BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    # ---------- Credentials & storage ----------
    # NEVER hardcode the key here: this file is committed to git, and a key in
    # git history stays there. It is read from the environment / CI secret.
    api_key: str = os.getenv("YOUTUBE_API_KEY", "")
    excel_path: Path = BASE_DIR / "youtube_predictive_datasetss.xlsx"
    model_dir: Path = BASE_DIR / "models"
    log_path: Path = BASE_DIR / "virality_engine.log"

    # ---------- Orchestration ----------
    cycle_sleep_seconds: int = 2 * 3600          # 2 hours between cycles

    # ---------- Stage 1: ingestion filters ----------
    target_new_videos: int = 50                  # how many keepers we want per cycle
    ingest_max_age_hours: float = 2.0            # "brand new" window (0-2h)
    min_duration_seconds: int = 180              # STRICTLY greater -> excludes Shorts
    min_views_at_discovery: int = 300
    min_subscribers: int = 500
    # Upper bound added after phase 1 of the real collection: the search endpoint
    # returns mostly mega-channels (median 4.5M subscribers in the first sample),
    # for which "views >= 3 x subscribers in 72h" is effectively unreachable —
    # 24 labelled rows produced zero positives. Capping the channel size keeps the
    # ground-truth rule attainable and the dataset learnable.
    max_subscribers: int = 300_000
    max_search_calls_per_cycle: int = 6          # 100 quota units each -> 400/cycle

    # ---------- Stage 2-4: checkpoint scheduling ----------
    # A cycle runs every 2h, so a checkpoint can never be observed at the exact
    # second. `grace_hours` is how late an observation may be and still count.
    checkpoint_hours: Tuple[int, int, int] = (8, 24, 72)
    grace_hours: float = 24.0                   # >3h late -> checkpoint MISSED

    # ---------- Stage 4: ground-truth rule ("The Teacher") ----------
    viral_views_multiplier: float = 3.0          # final_views >= 3 * subscribers
    # Lowered from 0.02 after phase 2 of the real collection: truly viral videos
    # reach non-subscribers who rarely like/comment, so a 2% floor rejected 30 of
    # 37 videos that hit the reach target. 0.3% still filters bot-inflated views.
    viral_engagement_ratio: float = 0.003        # (likes+comments)/final_views >= 0.3%

    # ---------- Stage 5: the PyTorch student ----------
    min_rows_to_train: int = 40                  # below this, training is noise
    min_rows_per_class: int = 8
    hidden_1: int = 32
    hidden_2: int = 16
    dropout: float = 0.20
    epochs: int = 400
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.20
    early_stopping_patience: int = 40
    random_seed: int = 42
    decision_threshold: float = 0.50

    # ---------- API resilience ----------
    api_max_retries: int = 5
    api_backoff_base: float = 2.0                # exponential: 2,4,8,16,32 s
    daily_quota_units: int = 10_000              # default YouTube Data API v3 cap


CFG = Config()

# Broad seed queries. `search.list` without a query returns low-quality spam, so
# we rotate through neutral topical seeds to obtain a diversified daily sample.
# Set to [""] if you prefer an unfiltered (noisier) firehose.
SEARCH_SEEDS: List[str] = [
    "tutorial", "review", "vlog", "news", "gaming", "music", "podcast",
    "documentary", "recipe", "workout", "travel", "explained", "reaction",
    "interview", "highlights", "analysis", "unboxing", "story",
]

# ------------------------------------------------------------------------------
#  THE FEATURE CONTRACT — the ONLY five columns the network is ever allowed to see.
#  All are derived from the 8h observation or from metadata known at upload time.
# ------------------------------------------------------------------------------
FEATURE_COLUMNS: List[str] = [
    "log_views_per_minute",  # log(1 + views_8h / measured age in minutes)
    "likes_per_view",        # likes_8h / views_8h
    "comments_per_view",     # comments_8h / views_8h
    "title_length",          # characters in the title (static, upload-time)
    "tags_count",            # number of tags (static, upload-time)
    # --- added in v2 (computed on the fly by add_derived_features) ---
    "log_views_per_sub_8h",  # log(1 + views_8h / subscribers): reach relative to audience
    "log_subscribers",       # log(1 + subscribers at discovery): channel size
    "log_acceleration",      # log(velocity at 8h / velocity at 0-2h): is it speeding up?
    "log_duration",          # log(1 + duration in seconds)
]

# The original five features (v1), kept so the thesis can report v1 vs v2.
FEATURE_COLUMNS_V1: List[str] = [
    "views_per_minute", "likes_per_view", "comments_per_view", "title_length", "tags_count",
]

# Any column whose name matches these patterns is future information relative to
# the 8h decision point and must never enter the feature matrix.
LEAKAGE_PATTERNS: Tuple[str, ...] = ("_24h", "_72h", "final_", "is_viral", "label")

TARGET_COLUMN = "is_viral"


# ==============================================================================
#  SECTION 2 — DATA SCHEMA
#  A single source of truth for the Excel layout. Column order = write order.
# ==============================================================================

SCHEMA: List[str] = [
    # --- identity / static metadata (known at upload time) ---
    "video_id", "title", "channel_id", "channel_title", "category_id",
    "published_at", "discovered_at", "duration_seconds",
    "title_length", "tags_count", "description_length",
    "subscribers", "channel_video_count",

    # --- t0 observation (0-2h) : stored for the thesis, NOT used by the model ---
    "views_0h", "likes_0h", "comments_0h", "age_minutes_0h",

    # --- 8h checkpoint : the model's baseline ---
    "checked_at_8h", "age_minutes_8h", "views_8h", "likes_8h", "comments_8h",
    "stage_8h_status",

    # --- engineered features (computed at the 8h checkpoint) ---
    "views_per_minute", "likes_per_view", "comments_per_view",

    # --- 24h checkpoint : trajectory analysis only, never a model input ---
    "checked_at_24h", "age_minutes_24h", "views_24h", "likes_24h", "comments_24h",
    "stage_24h_status",

    # --- 72h ground truth ("The Teacher") ---
    "checked_at_72h", "age_minutes_72h", "final_views", "final_likes",
    "final_comments", "subscribers_72h", "views_to_subs_ratio",
    "engagement_ratio", "stage_72h_status",

    # --- label + inference output ---
    "is_viral",
    "pred_prob", "pred_label", "pred_at", "pred_model_version",

    # --- housekeeping ---
    "row_status", "last_updated_at",
]

# Columns that must be numeric when read back from Excel.
NUMERIC_COLUMNS: List[str] = [
    "category_id", "duration_seconds", "title_length", "tags_count",
    "description_length", "subscribers", "channel_video_count",
    "views_0h", "likes_0h", "comments_0h", "age_minutes_0h",
    "age_minutes_8h", "views_8h", "likes_8h", "comments_8h",
    "views_per_minute", "likes_per_view", "comments_per_view",
    "age_minutes_24h", "views_24h", "likes_24h", "comments_24h",
    "age_minutes_72h", "final_views", "final_likes", "final_comments",
    "subscribers_72h", "views_to_subs_ratio", "engagement_ratio",
    "pred_prob", "pred_label",
]

PENDING = "PENDING"          # is_viral placeholder until the 72h teacher runs
STATUS_DUE = "DUE"           # checkpoint not reached yet
STATUS_OK = "OK"             # checkpoint observed inside the grace window
STATUS_MISSED = "MISSED"     # observed too late -> row excluded from training
STATUS_DEAD = "DEAD"         # video deleted / made private before the checkpoint


# ==============================================================================
#  SECTION 3 — LOGGING
# ==============================================================================

def setup_logging(verbose: bool = True) -> logging.Logger:
    """Console + rotating file logging. Google's HTTP client is muted to WARNING."""
    logger = logging.getLogger("virality")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = RotatingFileHandler(
        CFG.log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # Third-party noise suppression.
    logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)
    logging.getLogger("googleapiclient").setLevel(logging.WARNING)
    return logger


LOG = setup_logging()


# ==============================================================================
#  SECTION 4 — SMALL UTILITIES
# ==============================================================================

def utcnow() -> datetime:
    """Timezone-aware 'now'. Every timestamp in this program is UTC — never local."""
    return datetime.now(timezone.utc)


def to_iso(dt: Optional[datetime]) -> Optional[str]:
    """Serialize a datetime for Excel as a stable ISO-8601 string (not a date cell).

    Storing strings instead of native Excel datetimes avoids two classic bugs:
    silent timezone stripping by openpyxl, and pandas re-parsing columns into
    inconsistent dtypes on the next read.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def from_iso(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 / RFC-3339 string (or a datetime) into aware UTC."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, pd.Timestamp):
        ts = value.to_pydatetime()
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return None
    text = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = pd.to_datetime(text, utc=True, errors="raise").to_pydatetime()
        except Exception:
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?"
    r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?$"
)


def parse_iso_duration(duration: Optional[str]) -> int:
    """Convert an ISO-8601 duration ('PT4M13S') to whole seconds.

    Returns 0 for live streams ('P0D') or anything unparseable — such items are
    then rejected by the >180s Shorts filter, which is the safe default.
    """
    if not duration:
        return 0
    match = _DURATION_RE.match(str(duration).strip())
    if not match:
        return 0
    parts = match.groupdict(default="0")
    return int(
        int(parts["days"] or 0) * 86400
        + int(parts["hours"] or 0) * 3600
        + int(parts["minutes"] or 0) * 60
        + float(parts["seconds"] or 0)
    )


def safe_div(numerator: float, denominator: float) -> float:
    """Division that returns NaN instead of exploding on a zero/None denominator.

    NaN (not 0.0) is deliberate: 'we could not measure this' and 'the measured
    value is zero' are different statements, and the imputer must distinguish them.
    """
    try:
        numerator = float(numerator)
        denominator = float(denominator)
    except (TypeError, ValueError):
        return float("nan")
    if denominator == 0 or np.isnan(denominator) or np.isnan(numerator):
        return float("nan")
    return numerator / denominator


def to_int(value: Any, default: int = 0) -> int:
    """Coerce an API string ('12345') or NaN into an int without raising."""
    try:
        if value is None:
            return default
        if isinstance(value, float) and np.isnan(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def chunked(items: Sequence[Any], size: int) -> Iterable[List[Any]]:
    """Yield fixed-size chunks — this is what turns N requests into N/50 requests."""
    for start in range(0, len(items), size):
        yield list(items[start:start + size])


# ==============================================================================
#  SECTION 5 — QUOTA GUARD
#  The YouTube Data API gives 10,000 units/day. search.list costs 100 units;
#  videos.list and channels.list cost 1 unit each regardless of how many IDs are
#  batched into them. This guard makes the budget explicit and refuses calls that
#  would overdraw it, so tracking (cheap, and where the science lives) never dies
#  because ingestion (expensive) ate the whole day.
# ==============================================================================

class QuotaGuard:
    COST = {"search": 100, "videos": 1, "channels": 1}

    def __init__(self, daily_limit: int) -> None:
        self.daily_limit = daily_limit
        self.used = 0
        self.window_start = utcnow()

    def _roll_window(self) -> None:
        """YouTube quota resets at midnight Pacific; a rolling 24h reset is a
        conservative approximation that never *over*-spends."""
        if utcnow() - self.window_start >= timedelta(hours=24):
            LOG.info("Quota window rolled over (used %d units).", self.used)
            self.used = 0
            self.window_start = utcnow()

    def can_afford(self, kind: str, calls: int = 1) -> bool:
        self._roll_window()
        return self.used + self.COST[kind] * calls <= self.daily_limit

    def charge(self, kind: str, calls: int = 1) -> None:
        self._roll_window()
        self.used += self.COST[kind] * calls

    @property
    def remaining(self) -> int:
        self._roll_window()
        return max(0, self.daily_limit - self.used)


# ==============================================================================
#  SECTION 6 — YOUTUBE API CLIENT
# ==============================================================================

class YouTubeClient:
    """Thin, retry-hardened wrapper over the three endpoints we actually need."""

    def __init__(self, api_key: str, quota: QuotaGuard) -> None:
        if not api_key:
            raise RuntimeError(
                "YOUTUBE_API_KEY is not set. Export it before running the engine."
            )
        # cache_discovery=False avoids the noisy oauth2client file-cache warning.
        self.service = build("youtube", "v3", developerKey=api_key, cache_discovery=False)
        self.quota = quota

    # -- low-level ------------------------------------------------------------
    def _execute(self, request, kind: str) -> Dict[str, Any]:
        """Execute a request with exponential backoff.

        Distinguishes three failure classes:
          * quotaExceeded / rateLimitExceeded  -> fatal for this cycle, raise
          * 5xx / transient network            -> retry with backoff
          * 4xx (bad ID, deleted video)        -> raise to the caller to handle
        """
        last_error: Optional[Exception] = None
        for attempt in range(CFG.api_max_retries):
            try:
                response = request.execute()
                self.quota.charge(kind)
                return response
            except HttpError as err:
                status = getattr(err.resp, "status", None)
                reason = str(err)
                last_error = err
                if "quotaExceeded" in reason or "dailyLimitExceeded" in reason:
                    self.quota.used = self.quota.daily_limit  # hard stop for today
                    raise RuntimeError("YouTube API daily quota exhausted.") from err
                if status and 400 <= status < 500 and "rateLimitExceeded" not in reason:
                    raise
                sleep_for = CFG.api_backoff_base ** attempt + random.uniform(0, 1)
                LOG.warning(
                    "API %s error (status=%s), retry %d/%d in %.1fs",
                    kind, status, attempt + 1, CFG.api_max_retries, sleep_for,
                )
                time.sleep(sleep_for)
            except Exception as err:  # socket timeouts, SSL resets, DNS blips
                last_error = err
                sleep_for = CFG.api_backoff_base ** attempt + random.uniform(0, 1)
                LOG.warning("Network error (%s), retry in %.1fs", err, sleep_for)
                time.sleep(sleep_for)
        raise RuntimeError(f"API call failed after {CFG.api_max_retries} retries: {last_error}")

    # -- discovery ------------------------------------------------------------
    def search_recent_video_ids(self, published_after: datetime,
                                published_before: datetime,
                                query: str,
                                max_results: int = 50) -> List[str]:
        """One search.list page (100 quota units) restricted to the 0-2h window.

        `videoDuration='medium'` is intentionally NOT used: it would drop every
        video between 3 and 4 minutes, which our >180s rule explicitly keeps.
        """
        if not self.quota.can_afford("search"):
            LOG.warning("Skipping search: only %d quota units left.", self.quota.remaining)
            return []
        params: Dict[str, Any] = {
            "part": "id",
            "type": "video",
            # "viewCount", not "date": ordering by date returns videos uploaded
            # minutes ago with ~0 views, so ~99% failed the 300-view gate
            # (200 screened -> 2 kept). Within the same 0-2h window, ordering by
            # views surfaces the uploads that already have traction.
            "order": "viewCount",
            "maxResults": min(50, max_results),
            "publishedAfter": to_iso(published_after),
            "publishedBefore": to_iso(published_before),
        }
        if query:
            params["q"] = query
        response = self._execute(self.service.search().list(**params), "search")
        return [
            item["id"]["videoId"]
            for item in response.get("items", [])
            if item.get("id", {}).get("videoId")
        ]

    # -- batched detail fetches ----------------------------------------------
    def get_videos(self, video_ids: Sequence[str],
                   parts: str = "snippet,statistics,contentDetails") -> Dict[str, Dict[str, Any]]:
        """Fetch up to 50 videos per request — 1 quota unit per *request*.

        This is the single most important cost optimisation in the pipeline:
        tracking 500 pending videos costs 10 units, not 500.
        """
        out: Dict[str, Dict[str, Any]] = {}
        for batch in chunked(list(video_ids), 50):
            if not self.quota.can_afford("videos"):
                LOG.warning("Quota exhausted mid-batch; %d videos not refreshed.", len(batch))
                break
            response = self._execute(
                self.service.videos().list(part=parts, id=",".join(batch), maxResults=50),
                "videos",
            )
            for item in response.get("items", []):
                out[item["id"]] = item
        return out

    def get_channels(self, channel_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """Batched channels.list — 1 quota unit per request, 50 channels per call."""
        out: Dict[str, Dict[str, Any]] = {}
        unique_ids = list(dict.fromkeys(channel_ids))  # de-dup, order preserved
        for batch in chunked(unique_ids, 50):
            if not self.quota.can_afford("channels"):
                break
            response = self._execute(
                self.service.channels().list(
                    part="statistics,snippet", id=",".join(batch), maxResults=50
                ),
                "channels",
            )
            for item in response.get("items", []):
                out[item["id"]] = item
        return out


# ==============================================================================
#  SECTION 7 — EXCEL PERSISTENCE LAYER
#  The .xlsx file IS the database: crash-safe, human-inspectable, thesis-friendly.
# ==============================================================================

class ExcelStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        """Read the dataset, guaranteeing the full schema and correct dtypes."""
        if not self.path.exists():
            LOG.info("No dataset found — creating a fresh one at %s", self.path)
            return self._empty()
        try:
            df = pd.read_excel(self.path, dtype=object, engine="openpyxl")
        except Exception as err:
            # A corrupt/locked workbook must never kill the daemon: quarantine it.
            backup = self.path.with_suffix(f".corrupt-{int(time.time())}.xlsx")
            LOG.error("Could not read %s (%s). Moving it to %s", self.path, err, backup)
            try:
                shutil.move(str(self.path), str(backup))
            except Exception:
                pass
            return self._empty()
        return self._normalise(df)

    def save(self, df: pd.DataFrame) -> pd.DataFrame:
        """Atomic write: build a temp file, then os.replace it over the original.

        Without this, a crash (or Ctrl-C) during to_excel leaves a truncated,
        unreadable workbook — i.e. the entire labelled dataset is lost.

        Returns the normalised frame so the in-memory state stays dtype-identical
        to what is on disk (see `_normalise` for why that matters).
        """
        df = self._normalise(df)
        tmp = self.path.with_suffix(".tmp.xlsx")
        try:
            with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="dataset")
            os.replace(tmp, self.path)
            LOG.debug("Dataset saved (%d rows).", len(df))
        except PermissionError:
            LOG.error(
                "PERMISSION DENIED writing %s — the file is probably open in Excel. "
                "Close it; this cycle's changes are kept in memory and retried next cycle.",
                self.path,
            )
        except Exception as err:
            LOG.error("Failed to save dataset: %s", err)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
        return df

    # -- helpers --------------------------------------------------------------
    @classmethod
    def _empty(cls) -> pd.DataFrame:
        """An empty frame that already carries the correct per-column dtypes."""
        return cls._normalise(pd.DataFrame(columns=SCHEMA))

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        """Add missing columns, drop unknown ones, enforce column order + dtypes.

        The dtype pass is not cosmetic. A column created as all-NaN defaults to
        float64, and pandas >= 3 raises `TypeError: Invalid value ... for dtype
        float64` the moment you write an ISO timestamp or "PENDING" into it. Every
        non-numeric column is therefore pinned to `object` up front.
        """
        for column in SCHEMA:
            if column not in df.columns:
                df[column] = pd.Series(
                    np.nan,
                    index=df.index,
                    dtype="float64" if column in NUMERIC_COLUMNS else "object",
                )
        df = df[SCHEMA].copy()
        for column in SCHEMA:
            if column in NUMERIC_COLUMNS:
                df[column] = pd.to_numeric(df[column], errors="coerce")
            else:
                df[column] = df[column].astype(object)
        # Defensive de-duplication: the search endpoint can return the same video
        # under two different seed queries within the same cycle.
        df = df.drop_duplicates(subset=["video_id"], keep="first").reset_index(drop=True)
        return df


# ==============================================================================
#  SECTION 8 — STAGE 1: INGESTION (age 0-2h)
# ==============================================================================

def ingest_new_videos(client: YouTubeClient, df: pd.DataFrame) -> pd.DataFrame:
    """Discover up to CFG.target_new_videos fresh, filter-passing videos.

    Filter chain (all must pass):
        published within the last CFG.ingest_max_age_hours
        duration_seconds > 180            -> excludes Shorts
        views >= 300                      -> excludes dead-on-arrival uploads
        subscribers >= 500                -> excludes zero-audience channels
        not already in the dataset
    """
    LOG.info("STAGE 1 | Ingesting brand-new videos (0-%.0fh old)...", CFG.ingest_max_age_hours)

    now = utcnow()
    published_after = now - timedelta(hours=CFG.ingest_max_age_hours)
    known_ids = set(df["video_id"].dropna().astype(str))

    # ---- 1a. Collect candidate IDs from a few rotating seed queries ----------
    candidate_ids: List[str] = []
    seeds = random.sample(SEARCH_SEEDS, k=min(CFG.max_search_calls_per_cycle, len(SEARCH_SEEDS)))
    for seed in seeds:
        if len(candidate_ids) >= CFG.target_new_videos * 4:
            break  # we over-fetch 4x because the filters are aggressive
        try:
            ids = client.search_recent_video_ids(published_after, now, seed)
        except RuntimeError as err:
            LOG.error("Search aborted: %s", err)
            break
        fresh = [vid for vid in ids if vid not in known_ids and vid not in candidate_ids]
        LOG.debug("  seed '%s' -> %d ids (%d new)", seed, len(ids), len(fresh))
        candidate_ids.extend(fresh)

    if not candidate_ids:
        LOG.info("STAGE 1 | No new candidates this cycle.")
        return df

    # ---- 1b. One batched videos.list for every candidate (1 unit / 50 ids) ---
    videos = client.get_videos(candidate_ids)
    LOG.debug("  fetched details for %d/%d candidates", len(videos), len(candidate_ids))

    # ---- 1c. Cheap filters BEFORE spending a channels.list call --------------
    stage_one: List[Dict[str, Any]] = []
    for video_id, item in videos.items():
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        content = item.get("contentDetails", {})

        duration_seconds = parse_iso_duration(content.get("duration"))
        if duration_seconds <= CFG.min_duration_seconds:
            continue  # Short / live stream

        views = to_int(stats.get("viewCount"))
        if views < CFG.min_views_at_discovery:
            continue

        published_at = from_iso(snippet.get("publishedAt"))
        if published_at is None:
            continue
        age_minutes = (now - published_at).total_seconds() / 60.0
        if age_minutes < 0 or age_minutes > CFG.ingest_max_age_hours * 60 + 15:
            continue  # outside the 0-2h window (15 min of clock-skew tolerance)

        stage_one.append({
            "video_id": video_id,
            "title": snippet.get("title", ""),
            "channel_id": snippet.get("channelId", ""),
            "channel_title": snippet.get("channelTitle", ""),
            "category_id": to_int(snippet.get("categoryId"), default=-1),
            "published_at": to_iso(published_at),
            "discovered_at": to_iso(now),
            "duration_seconds": duration_seconds,
            "title_length": len(snippet.get("title", "") or ""),
            "tags_count": len(snippet.get("tags", []) or []),
            "description_length": len(snippet.get("description", "") or ""),
            "views_0h": views,
            "likes_0h": to_int(stats.get("likeCount")),
            "comments_0h": to_int(stats.get("commentCount")),
            "age_minutes_0h": round(age_minutes, 2),
        })

    if not stage_one:
        LOG.info("STAGE 1 | All %d candidates failed the duration/views filters.", len(videos))
        return df

    # ---- 1d. Batched channel lookup for the subscriber gate ------------------
    channels = client.get_channels([row["channel_id"] for row in stage_one])
    accepted: List[Dict[str, Any]] = []
    for row in stage_one:
        channel = channels.get(row["channel_id"])
        if not channel:
            continue
        channel_stats = channel.get("statistics", {})
        if channel_stats.get("hiddenSubscriberCount"):
            continue  # unusable: the 72h label rule needs a subscriber count
        subscribers = to_int(channel_stats.get("subscriberCount"))
        if subscribers < CFG.min_subscribers or subscribers > CFG.max_subscribers:
            continue
        row["subscribers"] = subscribers
        row["channel_video_count"] = to_int(channel_stats.get("videoCount"))
        # Every new row starts life unlabelled, with all checkpoints pending.
        row["is_viral"] = PENDING
        row["stage_8h_status"] = STATUS_DUE
        row["stage_24h_status"] = STATUS_DUE
        row["stage_72h_status"] = STATUS_DUE
        row["row_status"] = "TRACKING"
        row["last_updated_at"] = to_iso(now)
        accepted.append(row)
        if len(accepted) >= CFG.target_new_videos:
            break

    if not accepted:
        LOG.info("STAGE 1 | No candidate passed the %d-%d subscriber gate.",
                 CFG.min_subscribers, CFG.max_subscribers)
        return df

    new_df = ExcelStore._normalise(pd.DataFrame(accepted))
    df = ExcelStore._normalise(pd.concat([df, new_df], ignore_index=True, sort=False))
    LOG.info(
        "STAGE 1 | Ingested %d new videos (%d candidates screened). Dataset: %d rows.",
        len(accepted), len(candidate_ids), len(df),
    )
    return df


# ==============================================================================
#  SECTION 9 — CHECKPOINT SCHEDULER
# ==============================================================================

def find_due_rows(df: pd.DataFrame, checkpoint_hours: int, now: datetime) -> List[int]:
    """Return the indices of rows whose `checkpoint_hours` observation is due.

    "Exactly 8h" is impossible with a 2h cycle, so a row is due when
        age >= checkpoint  AND  age <= checkpoint + grace
    and it has not been observed yet. Rows that drift past the grace window are
    marked MISSED (below) rather than silently recorded with a wrong age, which
    would corrupt views_per_minute — the single strongest feature.
    """
    status_column = f"stage_{checkpoint_hours}h_status"
    due: List[int] = []
    for index, row in df.iterrows():
        if str(row.get(status_column)) != STATUS_DUE:
            continue
        published_at = from_iso(row.get("published_at"))
        if published_at is None:
            continue
        age_hours = (now - published_at).total_seconds() / 3600.0
        if age_hours >= checkpoint_hours:
            due.append(index)
    return due


def expire_missed(df: pd.DataFrame, checkpoint_hours: int, indices: List[int],
                  now: datetime) -> Tuple[List[int], int]:
    """Split due rows into (observable now, too late) and flag the late ones."""
    status_column = f"stage_{checkpoint_hours}h_status"
    observable, missed = [], 0
    for index in indices:
        published_at = from_iso(df.at[index, "published_at"])
        age_hours = (now - published_at).total_seconds() / 3600.0
        if age_hours > checkpoint_hours + CFG.grace_hours:
            df.at[index, status_column] = STATUS_MISSED
            missed += 1
        else:
            observable.append(index)
    return observable, missed


# ==============================================================================
#  SECTION 10 — STAGES 2 & 3: TIME-SERIES TRACKING (8h and 24h)
#  + STAGE 4a: FEATURE ENGINEERING AT THE 8-HOUR MARK
# ==============================================================================

def engineer_8h_features(df: pd.DataFrame, index: int) -> None:
    """Compute the five model features for one row, in place.

    Called exactly once per video, immediately after the 8h observation lands.
    Every term below is knowable at t+8h — nothing here touches 24h or 72h data.
    """
    views = df.at[index, "views_8h"]
    likes = df.at[index, "likes_8h"]
    comments = df.at[index, "comments_8h"]
    age_minutes = df.at[index, "age_minutes_8h"]

    # Velocity, not volume: divided by the ACTUAL measured age, so a row observed
    # at 9.4h is not treated as if it were observed at 8.0h.
    df.at[index, "views_per_minute"] = safe_div(views, age_minutes)
    df.at[index, "likes_per_view"] = safe_div(likes, views)
    df.at[index, "comments_per_view"] = safe_div(comments, views)
    # title_length and tags_count are static and were stored at ingestion time.


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a COPY of df with the v2 model features added.

    Computed on the fly (not stored in Excel) so they apply to old rows too.
    Inputs are only t0 / 8h information — subscribers and views_0h are captured
    at discovery, views_8h at the 8h checkpoint — so nothing leaks from 24h/72h.

    The log transforms matter: views and subscribers are heavy-tailed (a few
    videos have 100x the median), and a StandardScaler on raw values would let
    those few rows dominate every gradient step.
    """
    out = df.copy()
    num = lambda c: pd.to_numeric(out[c], errors="coerce") if c in out else pd.Series(np.nan, index=out.index)
    subs = num("subscribers").where(lambda x: x > 0)
    vpm_8h = num("views_per_minute")
    vpm_0h = num("views_0h") / num("age_minutes_0h").where(lambda x: x > 0)
    out["log_views_per_minute"] = np.log1p(vpm_8h.clip(lower=0))
    out["log_views_per_sub_8h"] = np.log1p((num("views_8h") / subs).clip(lower=0))
    out["log_subscribers"] = np.log1p(subs)
    out["log_acceleration"] = np.log((vpm_8h.clip(lower=0) + 1e-3) / (vpm_0h.clip(lower=0) + 1e-3))
    out["log_duration"] = np.log1p(num("duration_seconds").clip(lower=0))
    return out


def track_checkpoint(client: YouTubeClient, df: pd.DataFrame,
                     checkpoint_hours: int, now: datetime) -> pd.DataFrame:
    """Refresh statistics for every row that has reached `checkpoint_hours`."""
    status_column = f"stage_{checkpoint_hours}h_status"
    due = find_due_rows(df, checkpoint_hours, now)
    if not due:
        LOG.info("STAGE 2/3 | No rows due at the %dh checkpoint.", checkpoint_hours)
        return df

    due, missed = expire_missed(df, checkpoint_hours, due, now)
    if missed:
        LOG.warning("  %d rows missed the %dh window (>%.1fh late) -> MISSED.",
                    missed, checkpoint_hours, CFG.grace_hours)
    if not due:
        return df

    video_ids = [str(df.at[i, "video_id"]) for i in due]
    LOG.info("STAGE 2/3 | Refreshing %d videos at the %dh mark (%d batched calls).",
             len(video_ids), checkpoint_hours, (len(video_ids) + 49) // 50)

    fetched = client.get_videos(video_ids, parts="snippet,statistics")

    updated, dead = 0, 0
    for index in due:
        video_id = str(df.at[index, "video_id"])
        item = fetched.get(video_id)
        if item is None:
            # Absent from the response = deleted, privated, or region-blocked.
            df.at[index, status_column] = STATUS_DEAD
            df.at[index, "row_status"] = "DEAD"
            dead += 1
            continue

        stats = item.get("statistics", {})
        published_at = from_iso(df.at[index, "published_at"])
        age_minutes = (now - published_at).total_seconds() / 60.0

        df.at[index, f"checked_at_{checkpoint_hours}h"] = to_iso(now)
        df.at[index, f"age_minutes_{checkpoint_hours}h"] = round(age_minutes, 2)
        df.at[index, f"views_{checkpoint_hours}h"] = to_int(stats.get("viewCount"))
        df.at[index, f"likes_{checkpoint_hours}h"] = to_int(stats.get("likeCount"))
        df.at[index, f"comments_{checkpoint_hours}h"] = to_int(stats.get("commentCount"))
        df.at[index, status_column] = STATUS_OK
        df.at[index, "last_updated_at"] = to_iso(now)

        # Titles and tags can be edited post-upload; refresh the static features
        # at the 8h mark so they reflect what the audience actually saw.
        if checkpoint_hours == 8:
            snippet = item.get("snippet", {})
            df.at[index, "title"] = snippet.get("title", df.at[index, "title"])
            df.at[index, "title_length"] = len(snippet.get("title", "") or "")
            df.at[index, "tags_count"] = len(snippet.get("tags", []) or [])
            engineer_8h_features(df, index)

        updated += 1

    LOG.info("STAGE 2/3 | %dh checkpoint: %d updated, %d dead.",
             checkpoint_hours, updated, dead)
    return df


# ==============================================================================
#  SECTION 11 — STAGE 4: GROUND TRUTH AT 72h ("THE TEACHER")
# ==============================================================================

def label_ground_truth(client: YouTubeClient, df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Apply the virality rule to every row that has reached 72 hours.

        is_viral = 1  IFF  final_views >= 3 * subscribers
                     AND  (final_likes + final_comments) / final_views >= 0.003

    The first clause captures *reach beyond the existing audience* (algorithmic
    distribution); the second captures *audience quality* and filters out the
    inflated-view / zero-engagement pattern typical of click-farmed uploads.
    Both must hold — virality is reach AND resonance.
    """
    due = find_due_rows(df, 72, now)
    if not due:
        LOG.info("STAGE 4 | No rows due for labelling.")
        return df

    due, missed = expire_missed(df, 72, due, now)
    if missed:
        LOG.warning("  %d rows missed the 72h labelling window.", missed)
    if not due:
        return df

    video_ids = [str(df.at[i, "video_id"]) for i in due]
    fetched = client.get_videos(video_ids, parts="statistics")
    # Refresh subscribers too, purely for analysis (the rule uses the discovery
    # count so that the denominator matches the audience the video launched to).
    channels = client.get_channels([str(df.at[i, "channel_id"]) for i in due])

    viral_count, dud_count, dead = 0, 0, 0
    for index in due:
        video_id = str(df.at[index, "video_id"])
        item = fetched.get(video_id)
        if item is None:
            df.at[index, "stage_72h_status"] = STATUS_DEAD
            df.at[index, "row_status"] = "DEAD"
            dead += 1
            continue

        stats = item.get("statistics", {})
        final_views = to_int(stats.get("viewCount"))
        final_likes = to_int(stats.get("likeCount"))
        final_comments = to_int(stats.get("commentCount"))
        published_at = from_iso(df.at[index, "published_at"])

        # Subscriber baseline = the count captured at discovery (t0).
        subscribers = to_int(df.at[index, "subscribers"], default=0)
        channel = channels.get(str(df.at[index, "channel_id"]))
        if channel:
            df.at[index, "subscribers_72h"] = to_int(
                channel.get("statistics", {}).get("subscriberCount")
            )

        views_to_subs = safe_div(final_views, subscribers)
        engagement = safe_div(final_likes + final_comments, final_views)

        # --- THE RULE ---------------------------------------------------------
        reach_ok = subscribers > 0 and final_views >= CFG.viral_views_multiplier * subscribers
        resonance_ok = (not np.isnan(engagement)) and engagement >= CFG.viral_engagement_ratio
        is_viral = 1 if (reach_ok and resonance_ok) else 0
        # ----------------------------------------------------------------------

        df.at[index, "checked_at_72h"] = to_iso(now)
        df.at[index, "age_minutes_72h"] = round((now - published_at).total_seconds() / 60.0, 2)
        df.at[index, "final_views"] = final_views
        df.at[index, "final_likes"] = final_likes
        df.at[index, "final_comments"] = final_comments
        df.at[index, "views_to_subs_ratio"] = views_to_subs
        df.at[index, "engagement_ratio"] = engagement
        df.at[index, "is_viral"] = is_viral
        df.at[index, "stage_72h_status"] = STATUS_OK
        df.at[index, "row_status"] = "LABELLED"
        df.at[index, "last_updated_at"] = to_iso(now)

        viral_count += is_viral
        dud_count += 1 - is_viral

    LOG.info("STAGE 4 | Labelled %d rows -> %d viral / %d non-viral (%d dead).",
             viral_count + dud_count, viral_count, dud_count, dead)
    return df


def relabel_existing(df: pd.DataFrame) -> pd.DataFrame:
    """Re-apply the CURRENT ground-truth rule to every already-labelled row.

    No API calls: it only uses the stored final_views / final_likes /
    final_comments / subscribers. This keeps old and new rows consistent after
    the rule's thresholds change.
    """
    mask = df["row_status"].astype(str) == "LABELLED"
    if not mask.any():
        return df
    subs = pd.to_numeric(df.loc[mask, "subscribers"], errors="coerce").fillna(0)
    views = pd.to_numeric(df.loc[mask, "final_views"], errors="coerce").fillna(0)
    eng = pd.to_numeric(df.loc[mask, "engagement_ratio"], errors="coerce")
    reach_ok = (subs > 0) & (views >= CFG.viral_views_multiplier * subs)
    resonance_ok = eng.notna() & (eng >= CFG.viral_engagement_ratio)
    new_labels = (reach_ok & resonance_ok).astype(int)
    old_labels = pd.to_numeric(df.loc[mask, "is_viral"], errors="coerce")
    changed = int((old_labels != new_labels).sum())
    if changed:
        df["is_viral"] = df["is_viral"].astype(object)
        df.loc[mask, "is_viral"] = new_labels
        LOG.info("STAGE 4 | Re-labelled %d rows with the current rule -> %d viral / %d total.",
                 changed, int(new_labels.sum()), int(mask.sum()))
    return df


# ==============================================================================
#  SECTION 12 — STAGE 5: THE PYTORCH STUDENT
# ==============================================================================

def build_network(input_dim: int, hidden_1: int, hidden_2: int, dropout: float):
    """Construct the virality network. Defined inside a function because its base
    class, nn.Module, only exists once PyTorch has been imported (see ML_AVAILABLE).

    A deliberately small MLP: 9 -> 32 -> 16 -> 1 (logit).

    Why this shape: with a few hundred labelled rows and five features, anything
    deeper memorises the training set. Dropout + weight decay + early stopping do
    the regularising. The output is a raw logit — BCEWithLogitsLoss applies the
    sigmoid internally in a numerically stable way (log-sum-exp trick), so we
    must NOT add a Sigmoid layer here.
    """

    class ViralityNet(nn.Module):
        def __init__(self, input_dim: int, hidden_1: int, hidden_2: int, dropout: float) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_1),
                nn.BatchNorm1d(hidden_1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_1, hidden_2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_2, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)  # -> (batch,) logits

    return ViralityNet(input_dim, hidden_1, hidden_2, dropout)


def tensor_to_numpy(tensor: "torch.Tensor") -> np.ndarray:
    """Tensor -> ndarray, tolerant of a torch/NumPy ABI mismatch.

    A torch built against NumPy 1.x raises "Numpy is not available" from
    `.numpy()` when NumPy 2.x is installed. That is an environment problem, not a
    modelling one, and it should not take down a pipeline that has been running
    for days, so we fall back to a (slower) list conversion.
    """
    detached = tensor.detach().cpu()
    try:
        return detached.numpy()
    except RuntimeError:
        return np.asarray(detached.tolist(), dtype=np.float64)


def assert_no_leakage(feature_columns: Sequence[str]) -> None:
    """Runtime enforcement of the zero-leakage contract.

    If a future edit ever adds `final_views` or a 24h column to FEATURE_COLUMNS,
    the program crashes here instead of quietly producing a 0.99-AUC fantasy.
    """
    for column in feature_columns:
        for pattern in LEAKAGE_PATTERNS:
            if pattern in column:
                raise RuntimeError(
                    f"DATA LEAKAGE: feature '{column}' matches forbidden pattern "
                    f"'{pattern}'. The model may only use 8h-mark information."
                )


def set_seeds(seed: int) -> None:
    """Determinism so two runs on the same data produce the same thesis numbers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select rows eligible for supervised learning.

    Eligibility = a valid 0/1 label AND a clean 8h observation. Rows whose 8h
    checkpoint was MISSED/DEAD are excluded even if they carry a label, because
    their features describe a different point in time than the model assumes.
    """
    labels = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    eligible = (
        labels.isin([0, 1])
        & (df["stage_8h_status"].astype(str) == STATUS_OK)
        & df["views_per_minute"].notna()
    )
    derived = add_derived_features(df.loc[eligible])
    frame = derived[FEATURE_COLUMNS].copy()
    frame[TARGET_COLUMN] = labels.loc[eligible].astype(int)
    return frame


def train_model(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Retrain the network from scratch on every labelled row. Returns an artefact
    dict (model, scaler, imputation medians, metrics) or None if training is not
    yet statistically meaningful."""
    if not ML_AVAILABLE:
        LOG.info("STAGE 5 | ML libraries not installed on this host (%s). "
                 "Collection continues; train on a host with requirements.txt installed.",
                 ML_IMPORT_ERROR)
        return None

    assert_no_leakage(FEATURE_COLUMNS)
    set_seeds(CFG.random_seed)

    frame = build_training_frame(df)
    n_rows = len(frame)
    if n_rows < CFG.min_rows_to_train:
        LOG.info("STAGE 5 | Only %d labelled rows (need %d). Skipping training.",
                 n_rows, CFG.min_rows_to_train)
        return None

    positives = int(frame[TARGET_COLUMN].sum())
    negatives = n_rows - positives
    if min(positives, negatives) < CFG.min_rows_per_class:
        LOG.info("STAGE 5 | Class imbalance too extreme (%d viral / %d non-viral). Skipping.",
                 positives, negatives)
        return None

    X = frame[FEATURE_COLUMNS].astype(float).to_numpy()
    y = frame[TARGET_COLUMN].astype(float).to_numpy()

    # ---- Split FIRST, then fit every transformation on the training half only.
    # Fitting the scaler or the imputer on the full dataset would leak the test
    # distribution into training — a subtle but real form of leakage.
    stratify = y if min(positives, negatives) >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=CFG.validation_fraction,
        random_state=CFG.random_seed,
        stratify=stratify,
    )

    # ---- NaN handling: median imputation learned on the training split -------
    # NaNs arise legitimately (e.g. likes_per_view when views_8h == 0), and also
    # from infinities produced by pathological ratios.
    X_train = np.where(np.isfinite(X_train), X_train, np.nan)
    X_val = np.where(np.isfinite(X_val), X_val, np.nan)
    medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)  # all-NaN column -> 0
    X_train = np.where(np.isnan(X_train), medians, X_train)
    X_val = np.where(np.isnan(X_val), medians, X_val)

    # ---- Standardisation ----------------------------------------------------
    # views_per_minute lives on ~[0, 500] while likes_per_view lives on ~[0, 0.1].
    # Without scaling the first feature dominates every gradient step.
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.float32, device=device)
    Xva = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
    yva = torch.tensor(y_val, dtype=torch.float32, device=device)

    model = build_network(len(FEATURE_COLUMNS), CFG.hidden_1, CFG.hidden_2, CFG.dropout).to(device)

    # ---- Class imbalance ----------------------------------------------------
    # Virality is rare (typically 5-15% positives). pos_weight multiplies the loss
    # contribution of the positive class so the network cannot reach a low loss by
    # predicting "not viral" for everything.
    train_pos = float(y_train.sum())
    train_neg = float(len(y_train) - train_pos)
    pos_weight = torch.tensor([max(train_neg, 1.0) / max(train_pos, 1.0)],
                              dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate,
                                 weight_decay=CFG.weight_decay)

    LOG.info("STAGE 5 | Training on %d rows (%d viral / %d non-viral), pos_weight=%.2f, device=%s",
             n_rows, positives, negatives, pos_weight.item(), device.type)

    best_val_loss, best_state, patience_left = float("inf"), None, CFG.early_stopping_patience
    batch_size = min(CFG.batch_size, max(2, len(Xtr) // 2))

    for epoch in range(1, CFG.epochs + 1):
        # ---- train ----
        model.train()
        permutation = torch.randperm(len(Xtr), device=device)
        epoch_loss = 0.0
        for start in range(0, len(Xtr), batch_size):
            idx = permutation[start:start + batch_size]
            # BatchNorm requires >1 sample; skip a trailing singleton batch.
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            loss = criterion(model(Xtr[idx]), ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        # ---- validate ----
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xva), yva).item()

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = CFG.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                LOG.info("  Early stopping at epoch %d (best val loss %.4f).", epoch, best_val_loss)
                break

        if epoch % 50 == 0:
            LOG.debug("  epoch %3d | train %.4f | val %.4f", epoch,
                      epoch_loss / max(len(Xtr), 1), val_loss)

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Validation metrics -------------------------------------------------
    model.eval()
    with torch.no_grad():
        val_probs = tensor_to_numpy(torch.sigmoid(model(Xva)))
    val_preds = (val_probs >= CFG.decision_threshold).astype(int)
    y_val_int = y_val.astype(int)

    metrics = {
        "trained_at": to_iso(utcnow()),
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "n_total_labelled": int(n_rows),
        "positives": positives,
        "negatives": negatives,
        "best_val_loss": round(best_val_loss, 5),
        "val_accuracy": round(float(accuracy_score(y_val_int, val_preds)), 4),
        "val_precision": round(float(precision_score(y_val_int, val_preds, zero_division=0)), 4),
        "val_recall": round(float(recall_score(y_val_int, val_preds, zero_division=0)), 4),
        "val_f1": round(float(f1_score(y_val_int, val_preds, zero_division=0)), 4),
        # ROC-AUC is undefined when the validation split is single-class.
        "val_roc_auc": (round(float(roc_auc_score(y_val_int, val_probs)), 4)
                        if len(np.unique(y_val_int)) > 1 else None),
        # PR-AUC is the honest headline metric for a rare positive class: a model
        # that says "not viral" to everything gets a PR-AUC equal to the base rate.
        "val_pr_auc": (round(float(average_precision_score(y_val_int, val_probs)), 4)
                       if len(np.unique(y_val_int)) > 1 else None),
        "base_rate": round(float(positives / n_rows), 4),
        "features": FEATURE_COLUMNS,
    }
    metrics.update(cross_validated_baselines(frame))
    LOG.info("STAGE 5 | Validation — acc %.3f | prec %.3f | rec %.3f | f1 %.3f | auc %s",
             metrics["val_accuracy"], metrics["val_precision"], metrics["val_recall"],
             metrics["val_f1"], metrics["val_roc_auc"])

    artefacts = {
        "model": model,
        "scaler": scaler,
        "medians": medians,
        "metrics": metrics,
        "device": device,
        "version": f"v{utcnow().strftime('%Y%m%d-%H%M%S')}-n{n_rows}",
    }
    persist_artefacts(artefacts)
    return artefacts


def cross_validated_baselines(frame: pd.DataFrame, folds: int = 5) -> Dict[str, Any]:
    """5-fold stratified CV of simple baselines, for the thesis comparison table.

    A single 80/20 split on a few hundred rows (with ~25 positives) is noisy, so
    the baselines are cross-validated. The neural network must beat these to
    justify its existence:
      * logreg_v2   — logistic regression on the current (v2) features
      * views_per_sub_only — one feature: early reach relative to audience size
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
    except ImportError:
        return {}
    y = frame[TARGET_COLUMN].astype(int).to_numpy()
    if min(int(y.sum()), int(len(y) - y.sum())) < folds:
        return {}
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=CFG.random_seed)
    results: Dict[str, Any] = {}
    candidates = {
        "logreg_v2": FEATURE_COLUMNS,
        "views_per_sub_only": ["log_views_per_sub_8h"],
    }
    for name, columns in candidates.items():
        X = frame[columns].astype(float).replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median()).fillna(0.0).to_numpy()
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(class_weight="balanced", max_iter=2000))
        probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        preds = (probs >= 0.5).astype(int)
        results[f"cv_{name}_roc_auc"] = round(float(roc_auc_score(y, probs)), 4)
        results[f"cv_{name}_pr_auc"] = round(float(average_precision_score(y, probs)), 4)
        results[f"cv_{name}_f1"] = round(float(f1_score(y, preds, zero_division=0)), 4)
    LOG.info("STAGE 5 | CV baselines — logreg v2 ROC-AUC %s / PR-AUC %s | views/subs only ROC-AUC %s",
             results.get("cv_logreg_v2_roc_auc"), results.get("cv_logreg_v2_pr_auc"),
             results.get("cv_views_per_sub_only_roc_auc"))
    return results


def persist_artefacts(artefacts: Dict[str, Any]) -> None:
    """Write weights, scaler, medians and metrics to disk for reproducibility."""
    CFG.model_dir.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(
            {
                "state_dict": artefacts["model"].state_dict(),
                "input_dim": len(FEATURE_COLUMNS),
                "hidden_1": CFG.hidden_1,
                "hidden_2": CFG.hidden_2,
                "dropout": CFG.dropout,
                "features": FEATURE_COLUMNS,
                "version": artefacts["version"],
            },
            CFG.model_dir / "virality_net.pt",
        )
        joblib.dump(
            {"scaler": artefacts["scaler"], "medians": artefacts["medians"]},
            CFG.model_dir / "preprocessor.joblib",
        )
        with open(CFG.model_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(artefacts["metrics"], handle, indent=2)
        # Append-only history so the thesis can plot metric evolution over time.
        with open(CFG.model_dir / "metrics_history.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(artefacts["metrics"]) + "\n")
        LOG.debug("Artefacts persisted to %s", CFG.model_dir)
    except Exception as err:
        LOG.error("Could not persist model artefacts: %s", err)


# ==============================================================================
#  SECTION 13 — STAGE 6: INFERENCE ON FRESH 8h VIDEOS
# ==============================================================================

def predict_pending(df: pd.DataFrame, artefacts: Dict[str, Any], now: datetime) -> pd.DataFrame:
    """Score every still-unlabelled video that has a clean 8h observation.

    These are exactly the rows the system exists for: 8 hours old, 64 hours away
    from their ground truth, and now carrying a probability of going viral.
    """
    labels = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    target = (
        (df[TARGET_COLUMN].astype(str) == PENDING)
        & labels.isna()
        & (df["stage_8h_status"].astype(str) == STATUS_OK)
        & df["views_per_minute"].notna()
        & df["pred_prob"].isna()          # score once; never overwrite history
    )
    indices = df.index[target].tolist()
    if not indices:
        LOG.info("STAGE 6 | No newly-8h videos awaiting a prediction.")
        return df

    model = artefacts["model"]
    scaler = artefacts["scaler"]
    medians = artefacts["medians"]
    device = artefacts["device"]

    X = add_derived_features(df.loc[indices])[FEATURE_COLUMNS].astype(float).to_numpy()
    X = np.where(np.isfinite(X), X, np.nan)
    X = np.where(np.isnan(X), medians, X)     # same medians as training — no refit
    X_scaled = scaler.transform(X)

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_scaled, dtype=torch.float32, device=device))
        probabilities = tensor_to_numpy(torch.sigmoid(logits))

    for position, index in enumerate(indices):
        probability = float(probabilities[position])
        df.at[index, "pred_prob"] = round(probability, 6)
        df.at[index, "pred_label"] = int(probability >= CFG.decision_threshold)
        df.at[index, "pred_at"] = to_iso(now)
        df.at[index, "pred_model_version"] = artefacts["version"]
        df.at[index, "last_updated_at"] = to_iso(now)

    flagged = int((probabilities >= CFG.decision_threshold).sum())
    LOG.info("STAGE 6 | Scored %d videos — %d flagged as likely viral (mean p=%.3f).",
             len(indices), flagged, float(probabilities.mean()))
    return df


# ==============================================================================
#  SECTION 14 — CYCLE ORCHESTRATION
# ==============================================================================

def print_dashboard(df: pd.DataFrame, quota: QuotaGuard) -> None:
    """One-glance operational summary written to the log every cycle."""
    if df.empty:
        LOG.info("DASHBOARD | dataset empty.")
        return
    labels = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    pending = int((df[TARGET_COLUMN].astype(str) == PENDING).sum())
    viral = int((labels == 1).sum())
    non_viral = int((labels == 0).sum())
    labelled = viral + non_viral
    scored = int(df["pred_prob"].notna().sum())
    rate = (viral / labelled * 100) if labelled else 0.0

    LOG.info("-" * 78)
    LOG.info("DASHBOARD | rows=%d | pending=%d | labelled=%d (viral=%d, %.1f%%) | scored=%d",
             len(df), pending, labelled, viral, rate, scored)
    LOG.info("DASHBOARD | quota used=%d/%d units", quota.used, quota.daily_limit)
    LOG.info("-" * 78)


def run_cycle(client: YouTubeClient, store: ExcelStore, quota: QuotaGuard,
              do_ingest: bool = True, do_train: bool = True) -> None:
    """One complete pass over all six stages. Saves after each mutating stage so a
    crash never costs more than the stage in flight."""
    now = utcnow()
    LOG.info("=" * 78)
    LOG.info("CYCLE START | %s", to_iso(now))
    LOG.info("=" * 78)

    df = store.load()

    # ---- STAGE 1: ingest -----------------------------------------------------
    if do_ingest:
        try:
            df = ingest_new_videos(client, df)
            df = store.save(df)
        except RuntimeError as err:
            LOG.error("Ingestion halted: %s", err)
    else:
        LOG.info("STAGE 1 | Ingestion disabled for this run.")

    # ---- STAGES 2 & 3: 8h and 24h tracking -----------------------------------
    for checkpoint in (8, 24):
        try:
            df = track_checkpoint(client, df, checkpoint, utcnow())
            df = store.save(df)
        except RuntimeError as err:
            LOG.error("Tracking at %dh halted: %s", checkpoint, err)
            break

    # ---- STAGE 4: 72h ground truth -------------------------------------------
    try:
        df = label_ground_truth(client, df, utcnow())
        df = relabel_existing(df)
        df = store.save(df)
    except RuntimeError as err:
        LOG.error("Labelling halted: %s", err)

    # ---- STAGES 5 & 6: train, then predict -----------------------------------
    if do_train:
        try:
            artefacts = train_model(df)
            if artefacts is not None:
                df = predict_pending(df, artefacts, utcnow())
                df = store.save(df)
        except Exception as err:
            # ML failures must never take the data pipeline down with them.
            LOG.error("Training/inference failed: %s\n%s", err, traceback.format_exc())
    else:
        LOG.info("STAGES 5-6 | Training disabled for this run.")

    print_dashboard(df, quota)


# ==============================================================================
#  SECTION 15 — OFFLINE SELF-TEST (no API key, no network)
#  Verifies the pure logic: duration parsing, feature maths, the 72h rule, the
#  leakage guard, Excel round-tripping, and a full train/predict pass on
#  synthetic data. Run this before pointing the engine at a live API key.
# ==============================================================================

def self_test() -> bool:
    LOG.info("SELF-TEST | Running offline verification...")
    if not ML_AVAILABLE:
        LOG.error("SELF-TEST | Needs the full ML stack: pip install -r requirements.txt (%s)",
                  ML_IMPORT_ERROR)
        return False
    ok = True

    def check(name: str, condition: bool) -> None:
        nonlocal ok
        LOG.info("  [%s] %s", "PASS" if condition else "FAIL", name)
        ok = ok and condition

    # --- duration parsing ---
    check("PT4M13S -> 253s", parse_iso_duration("PT4M13S") == 253)
    check("PT59S -> 59s (Short)", parse_iso_duration("PT59S") == 59)
    check("PT1H2M3S -> 3723s", parse_iso_duration("PT1H2M3S") == 3723)
    check("P1DT2H -> 93600s", parse_iso_duration("P1DT2H") == 93600)
    check("garbage -> 0", parse_iso_duration("LIVE") == 0)
    check("Shorts filter rejects 180s", not (parse_iso_duration("PT3M") > CFG.min_duration_seconds))

    # --- safe division ---
    check("safe_div(10, 0) is NaN", np.isnan(safe_div(10, 0)))
    check("safe_div(10, 4) == 2.5", safe_div(10, 4) == 2.5)

    # --- timestamps ---
    now = utcnow()
    check("ISO round-trip", abs((from_iso(to_iso(now)) - now).total_seconds()) < 1.5)

    # --- the 72h rule ---
    def rule(views: int, subs: int, likes: int, comments: int) -> int:
        reach = subs > 0 and views >= CFG.viral_views_multiplier * subs
        eng = safe_div(likes + comments, views)
        return int(reach and not np.isnan(eng) and eng >= CFG.viral_engagement_ratio)

    check("30k views / 1k subs / 3% eng -> viral", rule(30_000, 1_000, 800, 100) == 1)
    check("30k views / 1k subs / 0.5% eng -> viral (>= 0.3%)", rule(30_000, 1_000, 100, 50) == 1)
    check("30k views / 1k subs / 0.17% eng -> not viral", rule(30_000, 1_000, 40, 10) == 0)
    check("2k views / 1k subs / 5% eng -> not viral", rule(2_000, 1_000, 90, 10) == 0)

    # --- leakage guard ---
    try:
        assert_no_leakage(["views_per_minute", "final_views"])
        check("leakage guard blocks final_views", False)
    except RuntimeError:
        check("leakage guard blocks final_views", True)
    try:
        assert_no_leakage(FEATURE_COLUMNS)
        check("production features pass the guard", True)
    except RuntimeError:
        check("production features pass the guard", False)

    # --- synthetic end-to-end: Excel round-trip + train + predict -------------
    rng = np.random.default_rng(CFG.random_seed)
    n = 240
    rows = []
    for i in range(n):
        published = now - timedelta(hours=100 + i)
        vpm = float(abs(rng.normal(8, 6)))
        lpv = float(abs(rng.normal(0.03, 0.02)))
        cpv = float(abs(rng.normal(0.004, 0.003)))
        # A learnable (noisy) relationship so the network has real signal to find.
        score = 0.45 * vpm + 60 * lpv + 200 * cpv + rng.normal(0, 1.5)
        label = 1 if score > 6.5 else 0
        rows.append({
            "video_id": f"TEST{i:05d}",
            "title": "x" * int(rng.integers(20, 90)),
            "channel_id": f"CH{i%40:03d}",
            "published_at": to_iso(published),
            "subscribers": int(rng.integers(500, 200_000)),
            "views_8h": int(vpm * 480),
            "likes_8h": int(vpm * 480 * lpv),
            "comments_8h": int(vpm * 480 * cpv),
            "age_minutes_8h": 480.0,
            "views_per_minute": vpm,
            "likes_per_view": lpv,
            "comments_per_view": cpv,
            "title_length": int(rng.integers(20, 90)),
            "tags_count": int(rng.integers(0, 30)),
            "stage_8h_status": STATUS_OK,
            "stage_24h_status": STATUS_OK,
            "stage_72h_status": STATUS_OK,
            "is_viral": label,
        })
    # 30 unlabelled rows sitting exactly at the 8h mark -> inference targets.
    for i in range(30):
        vpm = float(abs(rng.normal(8, 6)))
        rows.append({
            "video_id": f"LIVE{i:05d}",
            "published_at": to_iso(now - timedelta(hours=9)),
            "subscribers": 4200,
            "views_8h": int(vpm * 480),
            "likes_8h": 50, "comments_8h": 5, "age_minutes_8h": 485.0,
            "views_per_minute": vpm,
            "likes_per_view": 0.02, "comments_per_view": 0.002,
            "title_length": 55, "tags_count": 8,
            "stage_8h_status": STATUS_OK,
            "stage_24h_status": STATUS_DUE,
            "stage_72h_status": STATUS_DUE,
            "is_viral": PENDING,
        })
    # Inject NaNs to exercise the imputation path.
    synthetic = pd.DataFrame(rows)
    synthetic.loc[synthetic.sample(12, random_state=1).index, "likes_per_view"] = np.nan

    test_path = Path("selftest_dataset.xlsx")
    test_store = ExcelStore(test_path)
    test_store.save(synthetic)
    reloaded = test_store.load()
    check("Excel round-trip preserves rows", len(reloaded) == len(synthetic))
    check("Excel round-trip preserves schema", list(reloaded.columns) == SCHEMA)
    check("labels survive round-trip",
          int(pd.to_numeric(reloaded[TARGET_COLUMN], errors="coerce").isin([0, 1]).sum()) == n)

    artefacts = train_model(reloaded)
    check("model trains on synthetic data", artefacts is not None)
    if artefacts is not None:
        check("validation accuracy > 0.6", artefacts["metrics"]["val_accuracy"] > 0.6)
        scored = predict_pending(reloaded, artefacts, utcnow())
        probabilities = pd.to_numeric(scored["pred_prob"], errors="coerce").dropna()
        check("30 pending videos scored", len(probabilities) == 30)
        check("probabilities inside [0,1]",
              bool(((probabilities >= 0) & (probabilities <= 1)).all()))
        check("labelled rows were NOT re-scored",
              bool(pd.to_numeric(scored.loc[scored["video_id"].astype(str).str.startswith("TEST"),
                                            "pred_prob"], errors="coerce").isna().all()))

    # --- scheduler logic ---
    sched = pd.DataFrame([
        {"video_id": "A", "published_at": to_iso(now - timedelta(hours=3)),
         "stage_8h_status": STATUS_DUE},   # too young
        {"video_id": "B", "published_at": to_iso(now - timedelta(hours=9)),
         "stage_8h_status": STATUS_DUE},   # due, inside grace
        {"video_id": "C", "published_at": to_iso(now - timedelta(hours=40)),
         "stage_8h_status": STATUS_DUE},   # far too late
        {"video_id": "D", "published_at": to_iso(now - timedelta(hours=9)),
         "stage_8h_status": STATUS_OK},    # already done
    ])
    due = find_due_rows(sched, 8, now)
    check("scheduler finds only un-observed, aged rows", sorted(due) == [1, 2])
    observable, missed = expire_missed(sched, 8, due, now)
    check("grace window keeps B", observable == [1])
    check("grace window expires C", missed == 1 and sched.at[2, "stage_8h_status"] == STATUS_MISSED)

    # cleanup
    for artefact in (test_path,):
        try:
            artefact.unlink()
        except OSError:
            pass

    LOG.info("SELF-TEST | %s", "ALL CHECKS PASSED" if ok else "FAILURES DETECTED")
    return ok


# ==============================================================================
#  SECTION 16 — ENTRY POINT / THE ETERNAL LOOP
# ==============================================================================

_SHUTDOWN = False


def _handle_signal(signum, _frame) -> None:
    """Finish the current cycle, then exit cleanly instead of corrupting the file."""
    global _SHUTDOWN
    _SHUTDOWN = True
    LOG.warning("Signal %s received — shutting down after this cycle.", signum)


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous YouTube Virality Predictor")
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit")
    parser.add_argument("--no-ingest", action="store_true", help="skip Stage 1 (saves quota)")
    parser.add_argument("--no-train", action="store_true", help="skip Stages 5-6")
    parser.add_argument("--self-test", action="store_true",
                        help="offline logic verification, no API key needed")
    args = parser.parse_args()

    if args.self_test:
        return 0 if self_test() else 1

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    quota = QuotaGuard(CFG.daily_quota_units)
    store = ExcelStore(CFG.excel_path)
    try:
        client = YouTubeClient(CFG.api_key, quota)
    except RuntimeError as err:
        LOG.critical("%s", err)
        return 1

    LOG.info("Engine online. Dataset=%s | cycle=%ds | features=%s",
             CFG.excel_path, CFG.cycle_sleep_seconds, FEATURE_COLUMNS)

    while True:
        cycle_started = time.monotonic()
        try:
            run_cycle(client, store, quota,
                      do_ingest=not args.no_ingest,
                      do_train=not args.no_train)
        except Exception as err:
            # Absolute last line of defence: the daemon must survive anything.
            LOG.critical("UNHANDLED ERROR: %s\n%s", err, traceback.format_exc())

        if args.once or _SHUTDOWN:
            LOG.info("Exiting after a single cycle.")
            break

        elapsed = time.monotonic() - cycle_started
        sleep_for = max(60, CFG.cycle_sleep_seconds - elapsed)
        LOG.info("CYCLE END | took %.1fs — sleeping %.1f minutes.", elapsed, sleep_for / 60)
        # Sleep in short slices so Ctrl-C is responsive instead of hanging for 2h.
        slept = 0.0
        while slept < sleep_for and not _SHUTDOWN:
            time.sleep(min(5.0, sleep_for - slept))
            slept += 5.0
        if _SHUTDOWN:
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
