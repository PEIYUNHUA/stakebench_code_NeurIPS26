"""
get_market_comments.py  (v2 — fast, resumable, anti-ban)
---------------------------------------------------------
Collects Polymarket comments with historically-accurate position data.

v2 improvements over v1:
  • Parallel activity fetching (ThreadPoolExecutor, ACTIVITY_WORKERS threads)
  • Optional proxy pool rotation (proxies.txt, one per line)
  • Random User-Agent + jitter sleeps to reduce detection risk
  • Exponential backoff + jitter on 429 / 5xx errors
  • Incremental/resumable collection via polysent_data/progress.json
  • Auto-reads target markets from valid_topics_summary.csv (best_market_id column)

Proxy file format (proxies.txt, optional):
    http://host:port
    http://user:password@host:port
    socks5://host:port          <- requires:  pip install requests[socks]
"""

import os
import sys
import csv
import json
import time
import random
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

# ── API base URLs ──────────────────────────────────────────────────────────────
GAMMA = "https://gamma-api.polymarket.com"
DATA  = "https://data-api.polymarket.com"

# ── PolySent dataset filters ──────────────────────────────────────────────────
# MIN_COMMENT_LENGTH: minimum character count for comment body (c["body"]).
#   Comments shorter than this are silently dropped in _build_comment_row().
#   Note: Guide §4.3 recommends >20 chars; current value is intentionally
#   set to 3 to retain short-but-meaningful comments (e.g. "Yes", "No way").
#   Apply a stricter post-hoc filter (e.g. len(text) > 20) during cleaning.
MIN_COMMENT_LENGTH      = 3

# MIN_COMMENTS_PER_MARKET: markets whose collected comment count (after all
#   other filters) falls below this threshold are marked "skipped" in the
#   progress file and excluded from the output CSV entirely.
#   Applied in stream_markets_and_dump_comments() and run_from_api().
MIN_COMMENTS_PER_MARKET = 10

# ── Speed & retry knobs ───────────────────────────────────────────────────────
ACTIVITY_WORKERS = 8      # parallel threads for per-commenter activity fetch
MAX_RETRIES      = 6      # per-request max attempts
BASE_SLEEP       = 0.12   # base inter-request sleep (seconds)
JITTER           = 0.08   # +/- random jitter added to every sleep

# ── File paths ────────────────────────────────────────────────────────────────
PROXY_LIST_FILE = "proxies.txt"
TOPICS_CSV      = "polysent_data/valid_topics_summary.csv"
PROGRESS_FILE   = "polysent_data/progress.json"
DEFAULT_OUT_CSV = "polysent_data/comments.csv"

# ── User-Agent pool ────────────────────────────────────────────────────────────
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]

# ── Category keyword classification ───────────────────────────────────────────
QUESTION_CATEGORY_KEYWORDS = {
    "crypto":     ["bitcoin", "btc", "ethereum", "eth", "solana", "sol", "crypto",
                   "defi", "blockchain", "nft", "token", "binance", "coinbase",
                   "dogecoin", "doge", "xrp", "ripple", "stablecoin", "usdc",
                   "memecoin", "altcoin", "airdrop", "web3"],
    "politics":   ["election", "president", "congress", "senate", "vote", "voting",
                   "democrat", "republican", "parliament", "government",
                   "prime minister", "chancellor", "minister", "political party",
                   "campaign", "candidate", "ballot", "referendum"],
    "economics":  ["federal reserve", "interest rate", "gdp", "inflation",
                   "recession", "unemployment", "cpi", "fomc", "tariff",
                   "trade war", "rate cut", "rate hike"],
    "sports":     ["nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball",
                   "super bowl", "world cup", "olympics", "nascar", "tennis",
                   "golf", "baseball", "hockey", "playoff", "championship",
                   "tournament", "world series"],
    "technology": ["artificial intelligence", "openai", "chatgpt", "gpt",
                   "apple", "google", "microsoft", "spacex", "tesla",
                   "ipo", "semiconductor", "nvidia", "self-driving"],
}


# ══════════════════════════════════════════════════════════════════════════════
# Proxy pool
# ══════════════════════════════════════════════════════════════════════════════

class ProxyPool:
    """
    Round-robin proxy rotation with per-proxy failure tracking.
    Falls back to direct connection when no healthy proxies remain or when
    proxies.txt is absent.

    proxies.txt format (one proxy per line; lines starting with # are comments):
        http://host:port
        http://user:password@host:port
        socks5://host:port
    """
    MAX_FAILURES = 5

    def __init__(self, proxies=None):
        self._proxies = list(proxies or [])
        self._lock    = threading.Lock()
        self._idx     = 0
        self._fails   = {}   # proxy_str -> int

    @classmethod
    def from_file(cls, path=PROXY_LIST_FILE):
        if not os.path.exists(path):
            print(f"[ProxyPool] '{path}' not found — running without proxies.")
            return cls([])
        with open(path) as f:
            proxies = [ln.strip() for ln in f
                       if ln.strip() and not ln.strip().startswith("#")]
        print(f"[ProxyPool] loaded {len(proxies)} proxies from '{path}'")
        return cls(proxies)

    def get(self):
        """Return next healthy proxy dict, or None for direct connection."""
        if not self._proxies:
            return None
        with self._lock:
            n = len(self._proxies)
            for _ in range(n):
                p = self._proxies[self._idx % n]
                self._idx += 1
                if self._fails.get(p, 0) < self.MAX_FAILURES:
                    return {"http": p, "https": p, "_raw": p}
        return None   # all proxies exceeded failure threshold

    def report_failure(self, proxy_dict):
        if not proxy_dict:
            return
        p = proxy_dict.get("_raw")
        if p:
            with self._lock:
                self._fails[p] = self._fails.get(p, 0) + 1

    def report_success(self, proxy_dict):
        if not proxy_dict:
            return
        p = proxy_dict.get("_raw")
        if p:
            with self._lock:
                self._fails[p] = max(0, self._fails.get(p, 0) - 1)

    def __len__(self):
        return len(self._proxies)


# ══════════════════════════════════════════════════════════════════════════════
# Progress tracker  (incremental / resume)
# ══════════════════════════════════════════════════════════════════════════════

class ProgressTracker:
    """
    Persists collection status per market_id to PROGRESS_FILE (JSON).
    Allows resuming interrupted runs without reprocessing finished markets.

    Status values:
      "done"    — wrote >= min_comments rows to CSV
      "skipped" — API returned < min_comments valid comments (will NOT retry)
      "failed"  — unexpected error (WILL be retried on next run)
    """

    def __init__(self, path=PROGRESS_FILE):
        self.path  = path
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def is_done(self, market_id):
        """Return True if this market should be skipped (done or skipped)."""
        entry = self._data.get(str(market_id))
        if entry is None:
            return False
        return entry.get("status") in ("done", "skipped")

    def mark(self, market_id, status, n_comments=0):
        with self._lock:
            self._data[str(market_id)] = {
                "status":     status,
                "n_comments": n_comments,
                "ts":         datetime.now(timezone.utc).isoformat(),
            }
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)

    def summary(self):
        done    = sum(1 for v in self._data.values() if v["status"] == "done")
        skipped = sum(1 for v in self._data.values() if v["status"] == "skipped")
        failed  = sum(1 for v in self._data.values() if v["status"] == "failed")
        return done, skipped, failed


# ══════════════════════════════════════════════════════════════════════════════
# HTTP helpers
# ══════════════════════════════════════════════════════════════════════════════

def _sleep(base=BASE_SLEEP, jitter=JITTER):
    time.sleep(max(0.0, base + random.uniform(-jitter, jitter)))


def make_session(proxy_pool=None):
    """Create a requests.Session with a random User-Agent and optional proxy."""
    s = requests.Session()
    s.headers["User-Agent"] = random.choice(_USER_AGENTS)
    proxy_dict = proxy_pool.get() if proxy_pool else None
    if proxy_dict:
        s.proxies.update({k: v for k, v in proxy_dict.items() if k != "_raw"})
        s._proxy_dict = proxy_dict
    else:
        s._proxy_dict = None
    return s


def get_json(url, params=None, session=None, timeout=25,
             max_retry=MAX_RETRIES, proxy_pool=None):
    """
    GET + JSON parse with exponential backoff + jitter.
    Returns (status_code, data_dict).
    Rotates to the next proxy on 429 / 5xx / connection error.
    """
    _s = session or requests.Session()
    proxy_dict = getattr(_s, "_proxy_dict", None)

    for attempt in range(max_retry):
        try:
            r = _s.get(url, params=params, timeout=timeout)
        except requests.RequestException:
            wait = min(60.0, (2 ** attempt) * BASE_SLEEP + random.uniform(0, JITTER * 4))
            time.sleep(wait)
            if proxy_pool:
                proxy_pool.report_failure(proxy_dict)
                proxy_dict = proxy_pool.get()
                if proxy_dict:
                    _s.proxies.update({k: v for k, v in proxy_dict.items() if k != "_raw"})
                    _s._proxy_dict = proxy_dict
            continue

        if r.status_code == 200:
            if proxy_pool and proxy_dict:
                proxy_pool.report_success(proxy_dict)
            try:
                return 200, r.json()
            except Exception:
                return 200, {"_text": r.text[:300]}

        if r.status_code in (429, 500, 502, 503, 504):
            wait = min(120.0, (2 ** attempt) * BASE_SLEEP * 3 + random.uniform(0, JITTER * 8))
            if r.status_code == 429:
                wait = max(wait, 8.0)   # enforce minimum 8 s on rate-limit
            time.sleep(wait)
            if proxy_pool:
                proxy_pool.report_failure(proxy_dict)
                proxy_dict = proxy_pool.get()
                if proxy_dict:
                    _s.proxies.update({k: v for k, v in proxy_dict.items() if k != "_raw"})
                    _s._proxy_dict = proxy_dict
            continue

        # Non-retryable status (4xx other than 429)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"_text": r.text[:300]}

    return 0, {"_error": f"failed after {max_retry} retries — {url}"}


# ══════════════════════════════════════════════════════════════════════════════
# Market metadata helpers
# ══════════════════════════════════════════════════════════════════════════════

def classify_by_question(text):
    if not text:
        return None
    text_lower = text.lower()
    scores = {
        cat: sum(1 for kw in kws if kw in text_lower)
        for cat, kws in QUESTION_CATEGORY_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def extract_event_id(events_cell):
    if isinstance(events_cell, list) and events_cell and isinstance(events_cell[0], dict):
        eid = events_cell[0].get("id")
        try:
            return int(eid) if eid is not None else None
        except Exception:
            return None
    return None


def extract_series_id(market_obj):
    """Series is nested at market.events[0].series[0].id (not top-level)."""
    events = market_obj.get("events") or []
    if not events or not isinstance(events[0], dict):
        return None
    series_list = events[0].get("series") or []
    if not series_list or not isinstance(series_list, list):
        return None
    sid = series_list[0].get("id") if isinstance(series_list[0], dict) else None
    try:
        return int(sid) if sid is not None else None
    except Exception:
        return None


def parse_outcomes(market_obj):
    raw = market_obj.get("outcomes") or "[]"
    if isinstance(raw, list):
        return [str(o) for o in raw]
    if isinstance(raw, str):
        try:
            arr = json.loads(raw)
            if isinstance(arr, list):
                return [str(o) for o in arr]
        except Exception:
            pass
    return []


# ══════════════════════════════════════════════════════════════════════════════
# Activity / position reconstruction
# ══════════════════════════════════════════════════════════════════════════════

def fetch_user_activity(proxy_wallet, session, limit=500, max_pages=20):
    """Fetch complete TRADE+REDEEM history for one user. Returns sorted list."""
    all_events = []
    offset = 0
    for _ in range(max_pages):
        try:
            r = session.get(
                f"{DATA}/activity",
                params={"user": proxy_wallet, "limit": limit, "offset": offset},
                timeout=25,
            )
            if r.status_code != 200:
                break
            js = r.json()
        except Exception:
            break
        if not isinstance(js, list) or not js:
            break
        all_events.extend(js)
        if len(js) < limit:
            break
        offset += limit
        _sleep(BASE_SLEEP, JITTER)
    return sorted(all_events, key=lambda e: e.get("timestamp", 0))


def build_position_timeline(activity_events, condition_id, outcomes_list):
    """Reconstruct net position timeline for one user in one market."""
    timeline  = []
    net_shares = {}
    net_usdc   = {}

    for ev in activity_events:
        if ev.get("conditionId") != condition_id:
            continue
        ts      = ev.get("timestamp", 0)
        ev_type = ev.get("type", "")
        side    = ev.get("side", "")
        size    = float(ev.get("size") or 0)
        usdc    = float(ev.get("usdcSize") or 0)
        idx     = ev.get("outcomeIndex")
        if idx is None:
            continue

        if ev_type == "TRADE":
            delta_shares = size  if side == "BUY" else -size
            delta_usdc   = usdc  if side == "BUY" else -usdc
        elif ev_type == "REDEEM":
            delta_shares = -net_shares.get(idx, 0)
            delta_usdc   = -net_usdc.get(idx, 0)
        else:
            continue

        net_shares[idx] = net_shares.get(idx, 0) + delta_shares
        net_usdc[idx]   = net_usdc.get(idx, 0)   + delta_usdc

        if net_shares:
            best_idx = max(net_shares, key=lambda i: net_shares[i])
            try:
                best_side = (outcomes_list[best_idx]
                             if 0 <= best_idx < len(outcomes_list) else None)
            except (TypeError, IndexError):
                best_side = None
            timeline.append({
                "timestamp":  ts,
                "side":       best_side,
                "net_shares": net_shares.get(best_idx, 0),
                "net_usdc":   net_usdc.get(best_idx, 0),
            })

    return timeline


def get_position_at_time(timeline, comment_ts):
    """Binary-search the position timeline for the state at comment_ts."""
    if not timeline:
        return {}
    lo, hi, result_idx = 0, len(timeline) - 1, -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if timeline[mid]["timestamp"] <= comment_ts:
            result_idx = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if result_idx == -1:
        return {}
    entry = timeline[result_idx]
    if entry["net_shares"] <= 0:
        return {}
    return {"side":       entry["side"],
            "net_shares": entry["net_shares"],
            "net_usdc":   entry["net_usdc"]}


def fetch_activity_position_map(proxy_wallets, condition_id, outcomes_list,
                                proxy_pool=None, max_workers=ACTIVITY_WORKERS,
                                verbose=False):
    """
    Batch-fetch activity for all unique commenters IN PARALLEL.

    Each worker thread gets its own session (potentially a different proxy),
    spreading load across the pool.  Thread count is capped at ACTIVITY_WORKERS
    to avoid triggering rate limits.

    Returns dict: proxyWallet -> timeline (list of timed position snapshots).
    """
    unique_wallets = [pw for pw in set(proxy_wallets) if pw]
    result      = {}
    result_lock = threading.Lock()

    def _fetch_one(pw):
        session  = make_session(proxy_pool)
        activity = fetch_user_activity(pw, session)
        timeline = build_position_timeline(activity, condition_id, outcomes_list)
        _sleep(BASE_SLEEP / max(max_workers, 1), JITTER / 2)
        return pw, timeline

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_one, pw): pw for pw in unique_wallets}
        done = 0
        for fut in as_completed(futures):
            try:
                pw, tl = fut.result()
                if tl:
                    with result_lock:
                        result[pw] = tl
            except Exception:
                pass
            done += 1
            if verbose and done % 20 == 0:
                with result_lock:
                    n_active = len(result)
                print(f"    activity {done}/{len(unique_wallets)} wallets, "
                      f"{n_active} with market activity")

    return result


def fetch_holders_map(condition_id, outcomes_list, session, limit=500):
    """Fetch current holders snapshot (fast, 1 call). Used as fallback."""
    if not condition_id:
        return {}
    try:
        r = session.get(f"{DATA}/holders",
                        params={"market": condition_id, "limit": limit},
                        timeout=25)
        if r.status_code != 200:
            return {}
        js = r.json()
    except Exception:
        return {}

    result = {}
    for token_obj in (js if isinstance(js, list) else []):
        for h in token_obj.get("holders", []):
            pw = h.get("proxyWallet")
            if not pw:
                continue
            idx = h.get("outcomeIndex")
            try:
                side = (outcomes_list[idx]
                        if idx is not None and 0 <= idx < len(outcomes_list) else None)
            except (TypeError, IndexError):
                side = None
            try:
                size = float(h.get("amount") or 0)
            except (TypeError, ValueError):
                size = 0.0
            if pw not in result or size > result[pw]["size"]:
                result[pw] = {"side": side, "size": size}
    _sleep(BASE_SLEEP, JITTER)
    return result


def _parse_comment_ts(created_at_str):
    if not created_at_str:
        return 0
    try:
        dt = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except Exception:
        return 0


def _parse_market_times(market_obj):
    """Return (start_ts, end_ts, start_iso, end_iso) for a market dict.

    start_iso = startDate  (when the market opened for trading)
    end_iso   = closedTime (actual resolution) → umaEndDate → endDate (fallbacks)

    Timestamps are Unix ints (seconds); ISO strings are kept as-is for storage.
    Either value may be None if the market metadata lacks the field.
    """
    def _to_ts(s):
        if not s:
            return None
        try:
            import re
            s = str(s).strip()
            s = s.replace(" ", "T")          # "2024-11-06 15:17:41+00" → T-sep
            if s.endswith("+00"):
                s += ":00"                   # "+00" → "+00:00"
            s = s.replace("Z", "+00:00")     # "Z" → "+00:00"
            # Normalise fractional seconds to 6 digits (fromisoformat pre-3.11
            # only accepts 3 or 6 digits; ".18" → ".180000", ".960442" → kept)
            s = re.sub(r'\.(\d+)', lambda m: '.' + m.group(1).ljust(6, '0')[:6], s)
            return int(datetime.fromisoformat(s).timestamp())
        except Exception:
            return None

    start_iso = market_obj.get("startDate") or market_obj.get("startDateIso")
    end_iso   = (market_obj.get("closedTime")
                 or market_obj.get("umaEndDate")
                 or market_obj.get("endDate")
                 or market_obj.get("endDateIso"))

    return _to_ts(start_iso), _to_ts(end_iso), start_iso, end_iso


def _get_reply_count(comment_obj):
    """Read reply count from a comment dict, trying multiple field names.

    The Gamma API may omit replyCount at Series/Event level but include it
    at Comment level; trying several variants guards against naming changes.
    """
    for field in ("replyCount", "repliesCount", "reply_count", "replies"):
        v = comment_obj.get(field)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    return 0


def _build_comment_row(c, activity_timelines, holders_map, min_text_length,
                       parent_comment_id=None, start_ts=None, end_ts=None):
    """
    Convert a raw Gamma API comment object into a row dict.

    Filter conditions (returns None if any fail):
      1. len(body) < min_text_length  — body too short (default MIN_COMMENT_LENGTH=3)
      2. comment_ts < start_ts        — posted before market opened for trading
      3. comment_ts > end_ts          — posted after market closed/resolved

    Position data priority (first available wins):
      a) activity timeline: historically-accurate net position reconstructed from
         all TRADE/REDEEM events up to comment_ts via build_position_timeline().
         position_size = net_usdc (USDC paid net of sells); can be negative when
         the user sold more than they originally bought (e.g. after partial fills
         or rolling trades).  position_shares = net token shares.
      b) holders snapshot: current on-chain balance from data-api/holders.
         position_size = token amount (shares, not USDC); position_size is always
         >= 0 in this path, but reflects the end-of-market state, not the moment
         the comment was posted.
      c) no position: both lookups return empty → position_side=NaN, size=0.

    Shared by fetch_comments() (top-level) and fetch_reply_comments() (replies).
    """
    body = (c.get("body") or "").strip()
    if len(body) < min_text_length:
        return None

    profile    = c.get("profile") or {}
    proxy      = profile.get("proxyWallet", "")
    comment_ts = _parse_comment_ts(c.get("createdAt"))

    # ── Time-window filter ────────────────────────────────────────────────────
    if start_ts is not None and comment_ts < start_ts:
        return None
    if end_ts is not None and comment_ts > end_ts:
        return None

    pos      = {}
    timeline = activity_timelines.get(proxy, [])
    if timeline:
        pos = get_position_at_time(timeline, comment_ts)
    if not pos:
        snap = holders_map.get(proxy, {})
        if snap.get("size", 0) > 0:
            pos = {"side": snap["side"],
                   "net_shares": snap["size"], "net_usdc": None}

    position_side   = pos.get("side")
    position_size   = pos.get("net_usdc") or pos.get("net_shares") or 0.0
    position_shares = pos.get("net_shares", 0.0)

    reactions_raw  = c.get("reactions") or []
    reactions_json = (json.dumps(reactions_raw)
                      if isinstance(reactions_raw, list) else str(reactions_raw))

    return {
        "comment_id":        c.get("id"),
        "created_at":        c.get("createdAt"),
        "text":              body,
        "pseudonym":         profile.get("pseudonym"),
        "user_address":      c.get("userAddress"),
        "proxy_wallet":      proxy,
        "position_side":     position_side,
        "position_size":     round(float(position_size), 4) if position_size else 0.0,
        "position_shares":   round(float(position_shares), 4) if position_shares else 0.0,
        "reactions":         reactions_json,
        "reaction_count":    c.get("reactionCount") or 0,
        "report_count":      c.get("reportCount") or 0,
        "parent_comment_id": parent_comment_id,
        "reply_count":       _get_reply_count(c),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Comment fetching
# ══════════════════════════════════════════════════════════════════════════════

def fetch_comments(parent_type, parent_id, activity_timelines, holders_map,
                   session, limit=100, max_pages=10,
                   min_text_length=MIN_COMMENT_LENGTH,
                   verbose=False, proxy_pool=None,
                   start_ts=None, end_ts=None):
    """
    Fetch and filter comments for one parent entity, merging position data.

    parent_entity_type must be 'Event' or 'Series' (API rejects 'market').
    start_ts / end_ts: Unix timestamps; comments outside [start, end] are dropped.
    Returns list[dict] or None (HTTP error -> caller tries next entity type).
    """
    rows   = []
    offset = 0

    for _ in range(max_pages):
        params = {
            "parent_entity_type": parent_type,
            "parent_entity_id":   int(parent_id),
            "limit":              int(limit),
            "offset":             int(offset),
        }
        status, js = get_json(f"{GAMMA}/comments", params=params,
                              session=session, proxy_pool=proxy_pool)
        if status != 200:
            if verbose:
                print(f"  [{parent_type}:{parent_id}] HTTP {status} {js}")
            return None   # signal caller to try next entity type

        if not isinstance(js, list) or not js:
            break

        for c in js:
            row = _build_comment_row(c, activity_timelines, holders_map,
                                     min_text_length,
                                     parent_comment_id=c.get("parentCommentID"),
                                     start_ts=start_ts, end_ts=end_ts)
            if row is not None:
                rows.append(row)

        offset += limit
        _sleep(BASE_SLEEP, JITTER)

    return rows



def fetch_comments_for_market(market_obj, proxy_pool=None,
                               limit=100, max_pages=10,
                               min_text_length=MIN_COMMENT_LENGTH,
                               use_activity=True, verbose=True):
    """
    Fetch comments for a market with historically-accurate position data.

    Three-pass pipeline:
      Pass 1 — fetch raw comments via Event then Series fallback.
      Pass 2 — build position maps:
                  a) holders snapshot (1 API call, fast)
                  b) activity timelines (PARALLEL, 1 call per unique commenter)
      Pass 3 — re-fetch comments with positions merged.

    Returns (rows, entity_type_used, entity_id_used).
    """
    market_id    = int(market_obj["id"])
    condition_id = market_obj.get("conditionId")
    outcomes     = parse_outcomes(market_obj)
    event_id     = extract_event_id(market_obj.get("events"))
    series_id    = extract_series_id(market_obj)

    # ── Extract market active window ──────────────────────────────────────────
    start_ts, end_ts, start_iso, end_iso = _parse_market_times(market_obj)
    if verbose:
        print(f"  [market {market_id}] active window: {start_iso} → {end_iso}")

    session = make_session(proxy_pool)

    # ── Pass 1: raw comments (no position data yet) ───────────────────────────
    raw_rows, used_type, used_id = None, None, None

    if event_id is not None:
        raw_rows = fetch_comments("Event", event_id, {}, {}, session,
                                  limit=limit, max_pages=max_pages,
                                  min_text_length=min_text_length,
                                  proxy_pool=proxy_pool,
                                  start_ts=start_ts, end_ts=end_ts)
        if raw_rows:
            used_type, used_id = "Event", event_id

    if not raw_rows and series_id is not None:
        raw_rows = fetch_comments("Series", series_id, {}, {}, session,
                                  limit=limit, max_pages=max_pages,
                                  min_text_length=min_text_length,
                                  proxy_pool=proxy_pool,
                                  start_ts=start_ts, end_ts=end_ts)
        if raw_rows:
            used_type, used_id = "Series", series_id

    if not raw_rows:
        if verbose:
            print(f"[market {market_id}] comments=0 (no accessible event/series source)")
        return [], None, None

    proxy_wallets = [r["proxy_wallet"] for r in raw_rows if r.get("proxy_wallet")]

    # ── Pass 2a: holders snapshot (fast, 1 call) ──────────────────────────────
    holders_map = {}
    if condition_id:
        holders_map = fetch_holders_map(condition_id, outcomes, session)

    # ── Pass 2b: parallel activity timelines (main speedup) ──────────────────
    activity_timelines = {}
    if use_activity and condition_id:
        unique_n = len(set(proxy_wallets))
        if verbose:
            print(f"  [market {market_id}] fetching activity for "
                  f"{unique_n} commenters ({ACTIVITY_WORKERS} workers in parallel)...")
        activity_timelines = fetch_activity_position_map(
            proxy_wallets, condition_id, outcomes,
            proxy_pool=proxy_pool, max_workers=ACTIVITY_WORKERS, verbose=verbose,
        )

    # ── Pass 3: re-fetch comments with positions merged ───────────────────────
    session2 = make_session(proxy_pool)
    rows = None
    if used_type == "Event":
        rows = fetch_comments("Event", used_id, activity_timelines, holders_map,
                              session2, limit=limit, max_pages=max_pages,
                              min_text_length=min_text_length, proxy_pool=proxy_pool,
                              start_ts=start_ts, end_ts=end_ts)
    elif used_type == "Series":
        rows = fetch_comments("Series", used_id, activity_timelines, holders_map,
                              session2, limit=limit, max_pages=max_pages,
                              min_text_length=min_text_length, proxy_pool=proxy_pool,
                              start_ts=start_ts, end_ts=end_ts)

    rows = rows or []

    # ── Compute reply_count from collected data ───────────────────────────────
    # The Gamma API does not return replyCount, so we derive it here:
    # for each comment X, reply_count = number of rows whose parent_comment_id == X.
    # Note: only counts replies present in the fetched window (not all-time replies).
    from collections import Counter
    parent_tally = Counter(
        str(r["parent_comment_id"])
        for r in rows
        if r.get("parent_comment_id") is not None
    )
    for r in rows:
        r["reply_count"] = parent_tally.get(str(r["comment_id"]), 0)

    if verbose:
        n_pos    = sum(1 for r in rows if r["position_side"])
        n_active = sum(1 for pw in set(proxy_wallets) if pw in activity_timelines)
        n_reply  = sum(1 for r in rows if r.get("parent_comment_id") is not None)
        print(f"[market {market_id}] type={used_type}({used_id}) "
              f"comments={len(rows)} (top-level={len(rows)-n_reply} replies={n_reply}) "
              f"with_position={n_pos} activity_wallets={n_active}")
    return rows, used_type, used_id


# ══════════════════════════════════════════════════════════════════════════════
# Target market loading
# ══════════════════════════════════════════════════════════════════════════════

def load_target_markets_from_csv(csv_path=TOPICS_CSV):
    """
    Read best_market_id column from valid_topics_summary.csv.
    Returns list of unique integer market IDs in CSV order.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Topics CSV not found: {csv_path}")
    seen = set()
    ids  = []
    with open(csv_path, newline="", encoding="utf-8-sig", errors="replace") as f:
        for row in csv.DictReader(f):
            mid = row.get("best_market_id", "").strip()
            if not mid:
                continue
            try:
                mid_int = int(mid)
            except ValueError:
                continue
            if mid_int not in seen:
                seen.add(mid_int)
                ids.append(mid_int)
    print(f"Loaded {len(ids)} unique best_market_ids from '{csv_path}'")
    return ids


def fetch_markets_by_ids(market_ids, proxy_pool=None):
    """Fetch market objects from Gamma API by ID. Used when markets.jsonl absent."""
    markets = []
    session = make_session(proxy_pool)
    for mid in market_ids:
        try:
            status, js = get_json(f"{GAMMA}/markets", params={"id": int(mid)},
                                  session=session, proxy_pool=proxy_pool)
            m = (js[0] if isinstance(js, list) and js
                 else (js if isinstance(js, dict) else None))
            if m:
                markets.append(m)
                print(f"  fetched market {mid}: {str(m.get('question',''))[:70]}")
            else:
                print(f"  market {mid}: empty response (status={status})")
        except Exception as e:
            print(f"  failed to fetch market {mid}: {e}")
        _sleep(BASE_SLEEP * 1.5, JITTER)
    return markets


# ══════════════════════════════════════════════════════════════════════════════
# Shared CSV fieldnames & field descriptions
# ══════════════════════════════════════════════════════════════════════════════
#
# ── Market-level fields (one value per market, repeated on every comment row) ─
#
#   market_id               Integer Polymarket market ID (Gamma API "id").
#                           A market is a single binary-outcome prediction
#                           contract.
#
#   slug                    URL-friendly market identifier, e.g.
#                           "will-trump-win-2024-us-presidential-election".
#
#   question                Full prediction question text as shown in the UI.
#
#   condition_id            On-chain Polymarket condition ID (hex string).
#                           Used to query CLOB price history and data-api
#                           holder/trade records.
#
#   category                Raw category tag from market "_category" field in
#                           the Gamma API response.  Almost always NaN because
#                           the Gamma API does not expose this field publicly.
#
#   category_validated      Category inferred by keyword-matching "question"
#                           against QUESTION_CATEGORY_KEYWORDS (5 classes:
#                           crypto | politics | economics | sports | technology).
#                           NaN when no keyword matched.  Highest-scoring class
#                           wins; ties broken by dict insertion order.
#
#   market_start_time       ISO 8601 datetime when the market opened for trading
#                           (Gamma API "startDate").  Comments before this time
#                           are excluded by the time-window filter.
#                           ~1.6 % of rows are NaN (field absent in API).
#
#   market_end_time         ISO 8601 datetime when the market closed/resolved.
#                           Fallback priority: closedTime → umaEndDate → endDate.
#                           Comments after this time are excluded.
#
#   parent_entity_type_used The Gamma API entity type used to fetch comments:
#                           "Event" (standalone market) or "Series" (recurring
#                           market family).  The API rejects "market" as a type;
#                           Event is tried first, then Series as fallback.
#
#   parent_entity_id_used   Integer ID of the Event or Series whose /comments
#                           endpoint was called.  Multiple market_ids in the same
#                           Series share this ID, causing duplicate comment rows
#                           (same comment_id, different market_id / time window).
#
# ── Comment-level fields (one row per comment) ────────────────────────────────
#
#   comment_id              Integer Gamma API comment ID.  Globally unique per
#                           comment, but NOT unique per CSV row: a comment in a
#                           Series that spans multiple markets is written once per
#                           qualifying market (time-window intersection).
#                           Always deduplicate on this column before analysis.
#
#   created_at              ISO 8601 UTC timestamp of comment posting
#                           (Gamma API "createdAt").
#
#   text                    Cleaned comment body (strip() applied).
#                           Minimum length = MIN_COMMENT_LENGTH (currently 3).
#                           Guide §4.3 recommends >20 chars; apply post-hoc.
#
#   pseudonym               User's anonymous display name (e.g. "Scaly-Motion").
#                           ~1.6 % missing (NaN).
#
#   user_address            User's primary Ethereum address (lowercase hex).
#                           Stable user identifier across markets.
#
#   proxy_wallet            User's Polymarket delegated wallet address.
#                           This address signs on-chain transactions and is the
#                           key for data-api activity/holders lookups.
#
#   position_side           Outcome label of the user's net long position at
#                           comment time (e.g. "Yes", "No", "Up", "Down", team
#                           name for sports markets).  NaN = no position found.
#                           73 distinct values; Yes/No account for 90.8 %.
#
#   position_size           Net USDC cost basis of the position when sourced from
#                           the activity timeline (net_usdc = Σ buy USDC − Σ sell
#                           USDC).  Can be NEGATIVE when the user has recouped
#                           more than their cost (partial sell-offs or profit-
#                           taking result in a negative cost basis).
#                           When sourced from the holders snapshot (fallback),
#                           this field holds the token share amount (always ≥ 0)
#                           rather than USDC.  0.0 = no position data.
#
#   position_shares         Net token shares held at comment time (activity
#                           timeline net_shares, or holders snapshot amount).
#                           0.0 = no position data.
#
#   reactions               Full JSON array of reaction objects (type, reactor
#                           address, proxy wallet).  "[]" = no reactions.
#
#   reaction_count          Total reaction count (Gamma API "reactionCount").
#                           Occasionally −1 due to an API anomaly; filter in
#                           cleaning.
#
#   report_count            Times this comment was reported (Gamma API
#                           "reportCount").  0 = never reported.
#
#   parent_comment_id       For replies: integer comment_id of the parent.
#                           NaN for top-level comments.  Populated from
#                           c["parentCommentID"] in the Gamma API response.
#
#   reply_count             Number of replies to this comment captured in the
#                           same fetch batch.  Derived locally: count of rows
#                           whose parent_comment_id == this comment_id.
#                           NOT the all-time reply count; only reflects replies
#                           within the fetched time window and page limit.

_FIELDNAMES = [
    "market_id", "slug", "question", "condition_id",
    "category", "category_validated",
    "market_start_time", "market_end_time",
    "parent_entity_type_used", "parent_entity_id_used",
    "comment_id", "created_at", "text", "pseudonym",
    "user_address", "proxy_wallet",
    "position_side", "position_size", "position_shares",
    "reactions", "reaction_count", "report_count",
    "parent_comment_id", "reply_count",
]


# ══════════════════════════════════════════════════════════════════════════════
# Main collection loop  (markets.jsonl path)
# ══════════════════════════════════════════════════════════════════════════════

def stream_markets_and_dump_comments(
    markets_jsonl_path,
    target_market_ids,
    out_csv_path=DEFAULT_OUT_CSV,
    limit=100,
    max_pages=10,
    min_comments=MIN_COMMENTS_PER_MARKET,
    min_text_length=MIN_COMMENT_LENGTH,
    use_activity=True,
    proxy_pool=None,
    progress=None,
    verbose=True,
):
    """
    Stream markets from JSONL, collect comments, append to CSV.
    Skips markets already in progress tracker (done/skipped).
    Updates progress file after each market for crash-safe resuming.
    """
    target   = {int(x) for x in target_market_ids}
    progress = progress or ProgressTracker()

    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    write_header = not os.path.exists(out_csv_path)

    with open(out_csv_path, "a", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=_FIELDNAMES)
        if write_header:
            writer.writeheader()

        with open(markets_jsonl_path, encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    m = json.loads(line)
                except Exception:
                    continue

                mid = m.get("id")
                if mid is None:
                    continue
                try:
                    mid_int = int(mid)
                except Exception:
                    continue

                if mid_int not in target:
                    continue

                if progress.is_done(mid_int):
                    if verbose:
                        print(f"[market {mid_int}] already done — skipping")
                    continue

                try:
                    rows, used_type, used_id = fetch_comments_for_market(
                        m, proxy_pool=proxy_pool,
                        limit=limit, max_pages=max_pages,
                        min_text_length=min_text_length,
                        use_activity=use_activity, verbose=verbose,
                    )
                except Exception as exc:
                    print(f"[market {mid_int}] ERROR: {exc}")
                    progress.mark(mid_int, "failed")
                    continue

                if len(rows) < min_comments:
                    if verbose:
                        print(f"[market {mid_int}] skipped: {len(rows)} comments < min={min_comments}")
                    progress.mark(mid_int, "skipped", n_comments=len(rows))
                    continue

                slug               = m.get("slug")
                question           = m.get("question") or m.get("title")
                condition_id       = m.get("conditionId")
                category           = m.get("_category")
                category_validated = (m.get("_category_validated")
                                      or classify_by_question(question or ""))
                _, _, start_iso, end_iso = _parse_market_times(m)

                for r in rows:
                    writer.writerow({
                        "market_id":               str(mid_int),
                        "slug":                    slug,
                        "question":                question,
                        "condition_id":            condition_id,
                        "category":                category,
                        "category_validated":      category_validated,
                        "market_start_time":       start_iso,
                        "market_end_time":         end_iso,
                        "parent_entity_type_used": used_type,
                        "parent_entity_id_used":   used_id,
                        **r,
                    })
                fout.flush()

                n_pos = sum(1 for r in rows if r["position_side"])
                progress.mark(mid_int, "done", n_comments=len(rows))
                if verbose:
                    print(f"[market {mid_int}] wrote {len(rows)} comments "
                          f"({n_pos} with historical position data)")

    if verbose:
        done, skipped, failed = progress.summary()
        print(f"\nSaved: {out_csv_path}")
        print(f"Progress — done={done}  skipped={skipped}  failed={failed}")


# ══════════════════════════════════════════════════════════════════════════════
# Main collection loop  (API-only path, no markets.jsonl)
# ══════════════════════════════════════════════════════════════════════════════

def run_from_api(
    target_market_ids,
    out_csv_path=DEFAULT_OUT_CSV,
    limit=100,
    max_pages=20,
    min_comments=MIN_COMMENTS_PER_MARKET,
    min_text_length=MIN_COMMENT_LENGTH,
    use_activity=True,
    proxy_pool=None,
    progress=None,
    verbose=True,
):
    """
    Run comment collection without a pre-built markets.jsonl.
    Fetches market metadata from Gamma API for each pending ID.
    Supports incremental resume via ProgressTracker.
    """
    progress = progress or ProgressTracker()

    pending  = [mid for mid in target_market_ids if not progress.is_done(mid)]
    skipped_already = len(target_market_ids) - len(pending)
    if skipped_already and verbose:
        print(f"Skipping {skipped_already} already-finished markets.")
    if not pending:
        print("All target markets already done. Nothing to fetch.")
        return

    print(f"Fetching metadata for {len(pending)} markets from Gamma API...")
    markets = fetch_markets_by_ids(pending, proxy_pool=proxy_pool)
    if not markets:
        print("No market metadata retrieved. Exiting.")
        return

    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    write_header = not os.path.exists(out_csv_path)

    with open(out_csv_path, "a", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=_FIELDNAMES)
        if write_header:
            writer.writeheader()

        for m in markets:
            mid_int = int(m["id"])

            try:
                rows, used_type, used_id = fetch_comments_for_market(
                    m, proxy_pool=proxy_pool,
                    limit=limit, max_pages=max_pages,
                    min_text_length=min_text_length,
                    use_activity=use_activity, verbose=verbose,
                )
            except Exception as exc:
                print(f"[market {mid_int}] ERROR: {exc}")
                progress.mark(mid_int, "failed")
                continue

            if len(rows) < min_comments:
                if verbose:
                    print(f"[market {mid_int}] skipped: {len(rows)} < min={min_comments}")
                progress.mark(mid_int, "skipped", n_comments=len(rows))
                continue

            question           = m.get("question") or m.get("title")
            category_validated = classify_by_question(question or "")
            _, _, start_iso, end_iso = _parse_market_times(m)

            for r in rows:
                writer.writerow({
                    "market_id":               str(mid_int),
                    "slug":                    m.get("slug"),
                    "question":                question,
                    "condition_id":            m.get("conditionId"),
                    "category":                m.get("_category"),
                    "category_validated":      category_validated,
                    "market_start_time":       start_iso,
                    "market_end_time":         end_iso,
                    "parent_entity_type_used": used_type,
                    "parent_entity_id_used":   used_id,
                    **r,
                })
            fout.flush()

            n_pos = sum(1 for r in rows if r["position_side"])
            progress.mark(mid_int, "done", n_comments=len(rows))
            if verbose:
                print(f"[market {mid_int}] wrote {len(rows)} comments "
                      f"({n_pos} with historical position data)")

    if verbose:
        done, skipped, failed = progress.summary()
        print(f"\nSaved: {out_csv_path}")
        print(f"Progress — done={done}  skipped={skipped}  failed={failed}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

class _Tee:
    """Write every print() to both the terminal and a log file in real time."""

    def __init__(self, log_path, mode="a", encoding="utf-8"):
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._file    = open(log_path, mode, encoding=encoding, buffering=1)
        self._stdout  = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()


if __name__ == "__main__":
    # ── Log file (mirrors all print output to polysent_data/run.log) ───────────
    _log_path = "polysent_data/run.log"
    _tee = _Tee(_log_path)
    sys.stdout = _tee
    print(f"\n{'='*70}")
    print(f"Run started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Log: {_log_path}")
    print(f"{'='*70}\n")

    # ── Target markets ─────────────────────────────────────────────────────────
    # Option A (default): auto-read all best_market_ids from valid_topics_summary.csv
    # Option B: hard-code a list for quick testing
    #
    # --- Option A ---
    target_markets = load_target_markets_from_csv(TOPICS_CSV)

    # --- Option B (uncomment to override) ---
    # target_markets = [561829, 561263, 253591]

    # ── Proxy pool (reads proxies.txt if present, otherwise direct) ────────────
    pool = ProxyPool.from_file(PROXY_LIST_FILE)

    # ── Progress tracker (enables incremental / crash-safe collection) ─────────
    tracker = ProgressTracker(PROGRESS_FILE)
    done, skipped, failed = tracker.summary()
    if done or skipped or failed:
        print(f"Resuming — previous progress: done={done}  skipped={skipped}  failed={failed}")

    # ── Shared kwargs ──────────────────────────────────────────────────────────
    common = dict(
        out_csv_path    = DEFAULT_OUT_CSV,
        limit           = 100,
        max_pages       = 100,
        min_comments    = MIN_COMMENTS_PER_MARKET,
        min_text_length = MIN_COMMENT_LENGTH,
        use_activity    = True,
        proxy_pool      = pool,
        progress        = tracker,
        verbose         = True,
    )

    jsonl = "polysent_data/markets.jsonl"
    if os.path.exists(jsonl):
        stream_markets_and_dump_comments(
            markets_jsonl_path = jsonl,
            target_market_ids  = target_markets,
            **common,
        )
    else:
        print(f"'{jsonl}' not found — fetching market metadata from Gamma API.")
        run_from_api(target_market_ids=target_markets, **common)

    print(f"\n{'='*70}")
    print(f"Run finished: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*70}\n")
    _tee.close()
