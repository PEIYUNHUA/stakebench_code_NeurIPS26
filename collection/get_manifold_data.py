"""
get_manifold_data.py
====================
PolySent companion script for Manifold Markets.

Fetches market metadata + hourly probability (odds) history for a list of
market IDs, driven by a comments CSV or a manual TEST_MARKET_IDS list.

Outputs (in OUT_DIR):
  market_attrs.csv        – rich flat market attributes (single file)
  prices/{market_id}.csv  – hourly probability rows per market (one file each)

Resume-safe: markets with an existing prices/{market_id}.csv are skipped.

API docs: https://docs.manifold.markets/api
All read endpoints are public — no authentication required.
Rate limit: 500 req/min per IP.
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np

MANIFOLD = "https://api.manifold.markets/v0"

# ── Post-hoc category classification ─────────────────────────────────────────
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


def classify_by_question(text):
    """Keyword-based category inference from question text."""
    if not text:
        return None
    text_lower = text.lower()
    scores = {
        cat: sum(1 for kw in kws if kw in text_lower)
        for cat, kws in QUESTION_CATEGORY_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


# ── HTTP helper ───────────────────────────────────────────────────────────────

def get_json(url, params=None, timeout=30, max_retry=6, base_sleep=0.7):
    last_err = None
    for i in range(max_retry):
        try:
            r = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(base_sleep * (i + 1))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(base_sleep * (i + 1))
            continue
        try:
            last_err = r.json()
        except Exception:
            last_err = r.text[:300]
        raise RuntimeError(
            f"HTTP {r.status_code} {url} params={params} resp={last_err}"
        )
    raise RuntimeError(
        f"Failed after {max_retry} retries: {url} params={params} last={last_err}"
    )


# ── Market fetch ──────────────────────────────────────────────────────────────

def fetch_market_by_id(market_id):
    """
    Fetch full market metadata from Manifold API.
    market_id: string (Manifold uses alphanumeric IDs like 'abc123XYZ').
    Returns market dict, or None if not found.
    """
    try:
        data = get_json(f"{MANIFOLD}/market/{market_id}")
        if isinstance(data, dict) and data.get("id"):
            return data
        return None
    except RuntimeError as e:
        if "404" in str(e):
            return None
        raise


def extract_market_attrs(market_obj):
    """
    Extract a rich flat dict of Manifold market attributes.
    Covers identity, timing, liquidity, resolution, social, and group metadata.
    """
    if market_obj is None:
        return {}

    question_text = market_obj.get("question") or ""

    # ── groups ────────────────────────────────────────────────────────────────
    group_slugs = market_obj.get("groupSlugs") or []

    # ── pool (liquidity distribution) ─────────────────────────────────────────
    pool = market_obj.get("pool") or {}

    # ── description (rich text → plain string) ────────────────────────────────
    desc = market_obj.get("description")
    if isinstance(desc, dict):
        # Manifold stores description as ProseMirror JSON; extract plain text
        desc = _extract_text_from_prosemirror(desc)
    elif not isinstance(desc, str):
        desc = str(desc) if desc is not None else None

    return {
        # ── identity ──────────────────────────────────────────────────────────
        "market_id":              str(market_obj.get("id", "")),
        "slug":                   market_obj.get("slug"),
        "url":                    market_obj.get("url"),
        "question":               question_text,
        "description":            desc,

        # ── creator ───────────────────────────────────────────────────────────
        "creator_id":             market_obj.get("creatorId"),
        "creator_username":       market_obj.get("creatorUsername"),
        "creator_name":           market_obj.get("creatorName"),

        # ── timing ────────────────────────────────────────────────────────────
        "created_time":           _ms_to_iso(market_obj.get("createdTime")),
        "close_time":             _ms_to_iso(market_obj.get("closeTime")),
        "last_updated_time":      _ms_to_iso(market_obj.get("lastUpdatedTime")),
        "last_bet_time":          _ms_to_iso(market_obj.get("lastBetTime")),
        "resolution_time":        _ms_to_iso(market_obj.get("resolutionTime")),

        # ── status / resolution ───────────────────────────────────────────────
        "is_resolved":            market_obj.get("isResolved", False),
        "resolution":             market_obj.get("resolution"),          # YES/NO/MKT/CANCEL
        "resolution_probability": _safe_float(market_obj.get("resolutionProbability")),

        # ── current odds ──────────────────────────────────────────────────────
        "probability":            _safe_float(market_obj.get("probability")),

        # ── liquidity & trading ───────────────────────────────────────────────
        "volume":                 _safe_float(market_obj.get("volume")),
        "volume_24hr":            _safe_float(market_obj.get("volume24Hours")),
        "total_liquidity":        _safe_float(market_obj.get("totalLiquidity")),
        "pool_yes":               _safe_float(pool.get("YES")),
        "pool_no":                _safe_float(pool.get("NO")),
        "p":                      _safe_float(market_obj.get("p")),      # CPMM constant

        # ── market type ───────────────────────────────────────────────────────
        "outcome_type":           market_obj.get("outcomeType"),         # BINARY, MULTIPLE_CHOICE, …
        "mechanism":              market_obj.get("mechanism"),            # cpmm-1, dpm-2, …
        "token":                  market_obj.get("token"),               # MANA or CASH

        # ── social ────────────────────────────────────────────────────────────
        "unique_bettor_count":    int(market_obj.get("uniqueBettorCount") or 0),
        "comment_count":          int(market_obj.get("commentCount") or 0),

        # ── category / groups ─────────────────────────────────────────────────
        "group_slugs":            json.dumps(group_slugs),
        "category_validated":     classify_by_question(question_text),  # keyword-based
    }


def _ms_to_iso(ms):
    """Convert millisecond epoch to ISO-8601 string, or None."""
    if ms is None:
        return None
    try:
        return pd.to_datetime(int(ms), unit="ms", utc=True).isoformat()
    except Exception:
        return None


def _safe_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _extract_text_from_prosemirror(node, parts=None):
    """Recursively extract plain text from a ProseMirror JSON document."""
    if parts is None:
        parts = []
    if isinstance(node, dict):
        if node.get("type") == "text":
            parts.append(node.get("text", ""))
        for child in node.get("content") or []:
            _extract_text_from_prosemirror(child, parts)
    return " ".join(parts).strip()


# ── Bet / price-history helpers ───────────────────────────────────────────────

def fetch_all_bets(market_id, limit=1000):
    """
    Fetch all bets for a market via cursor-based pagination (before= bet ID).
    Returns list of bet dicts, chronologically ascending.
    """
    bets = []
    before_id = None

    while True:
        params = {"contractId": str(market_id), "limit": limit}
        if before_id is not None:
            params["before"] = before_id

        batch = get_json(f"{MANIFOLD}/bets", params=params)
        if not batch:
            break

        bets.extend(batch)

        if len(batch) < limit:
            break  # last page

        # next page: bets before the oldest in this batch
        # API returns newest-first, so last item is oldest
        before_id = batch[-1]["id"]
        time.sleep(0.12)

    # bets come newest-first from API; reverse to chronological order
    bets.reverse()
    return bets


def reconstruct_hourly_odds(bets, market_id, slug, question, category,
                             outcome_type="BINARY"):
    """
    Reconstruct an hourly probability time-series from raw bet events.

    Strategy:
      - Build a step function: at each bet's createdTime the probability
        becomes probAfter.
      - Resample to hourly by forward-filling the last known probability.

    Returns a DataFrame with columns:
      market_id, slug, question, category, outcome, ts, datetime, price
    where 'price' is the YES probability (0-1).
    """
    if not bets:
        return pd.DataFrame()

    # Filter to bets that carry probability fields (binary markets)
    prob_bets = [
        b for b in bets
        if b.get("probAfter") is not None and b.get("createdTime") is not None
    ]

    if not prob_bets:
        return pd.DataFrame()

    df_bets = pd.DataFrame([
        {
            "datetime": pd.to_datetime(b["createdTime"], unit="ms", utc=True),
            "prob":     float(b["probAfter"]),
            "outcome":  b.get("outcome", "YES"),
        }
        for b in prob_bets
    ]).sort_values("datetime")

    if df_bets.empty:
        return pd.DataFrame()

    # Build hourly index spanning market lifetime
    t_start = df_bets["datetime"].iloc[0].floor("h")
    t_end   = df_bets["datetime"].iloc[-1].ceil("h")
    hourly_index = pd.date_range(t_start, t_end, freq="h", tz="UTC")

    if hourly_index.empty:
        return pd.DataFrame()

    # Set prob as index, reindex to hourly, forward-fill
    prob_series = df_bets.set_index("datetime")["prob"]
    prob_series = prob_series[~prob_series.index.duplicated(keep="last")]
    prob_hourly = prob_series.reindex(
        prob_series.index.union(hourly_index)
    ).ffill().reindex(hourly_index)

    rows = []
    for dt, prob in prob_hourly.items():
        if pd.isna(prob):
            continue
        ts = int(dt.timestamp())
        base = {
            "market_id": str(market_id),
            "slug":      slug,
            "question":  question,
            "category":  category,
            "ts":        ts,
            "datetime":  dt,
        }
        rows.append({**base, "outcome": "YES", "price": float(prob)})
        rows.append({**base, "outcome": "NO",  "price": float(1.0 - prob)})

    return pd.DataFrame(rows)


# ── Main pipeline: by market_id list ─────────────────────────────────────────

def main_by_market_ids(
    market_ids,
    out_dir="manifold_data",
    comments_csv=None,
):
    """
    Fetch Manifold market metadata + hourly odds for a specific list of market_ids.

    Args:
        market_ids:   list of str market IDs (Manifold uses alphanumeric strings).
        out_dir:      output directory.
        comments_csv: path to a comments CSV with a 'market_id' column; if given,
                      its IDs are merged with market_ids before processing.

    Outputs:
        {out_dir}/market_attrs.csv       – rich flat market attributes (single file)
        {out_dir}/prices/{market_id}.csv – hourly probability rows, one file per market

    Resume-safe: markets with an existing prices/{market_id}.csv are skipped.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. resolve market_ids ──────────────────────────────────────────────────
    ids_set = set(str(mid).strip() for mid in market_ids if mid)

    if comments_csv and os.path.isfile(comments_csv):
        try:
            df_comments = pd.read_csv(comments_csv, usecols=["market_id"],
                                      encoding="utf-8-sig")
        except ValueError:
            df_comments = pd.read_csv(comments_csv, encoding="utf-8-sig")
        csv_ids = set(df_comments["market_id"].dropna().astype(str).str.strip().unique())
        print(f"Loaded {len(csv_ids)} unique market_ids from {comments_csv}")
        ids_set |= csv_ids
    elif comments_csv:
        print(f"[warn] comments_csv not found: {comments_csv}")

    market_ids_sorted = sorted(ids_set)
    print(f"Processing {len(market_ids_sorted)} market_ids total")

    # ── 2. fetch market metadata ───────────────────────────────────────────────
    market_rows = []

    for mid in market_ids_sorted:
        print(f"  Fetching market {mid} …", end=" ", flush=True)
        try:
            m = fetch_market_by_id(mid)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        if m is None:
            print("not found")
            continue

        attrs = extract_market_attrs(m)
        market_rows.append(attrs)

        print(
            f"ok  type={m.get('outcomeType')}  "
            f"resolved={m.get('isResolved')}  "
            f"groups={m.get('groupSlugs', [])}"
        )
        time.sleep(0.15)

    # save market attributes (single file — small, one row per market)
    attrs_df = pd.DataFrame(market_rows)
    attrs_csv = os.path.join(out_dir, "market_attrs.csv")
    attrs_df.to_csv(attrs_csv, index=False)
    print(f"Saved market attributes → {attrs_csv}  ({len(attrs_df)} markets)")

    # ── 3. fetch bets & reconstruct hourly odds — one file per market ──────────
    prices_dir = os.path.join(out_dir, "prices")
    os.makedirs(prices_dir, exist_ok=True)

    n_saved   = 0
    n_skipped = 0
    n_no_data = 0

    for row in market_rows:
        mid          = row["market_id"]
        outcome_type = row.get("outcome_type", "BINARY")
        slug         = row.get("slug")
        question     = row.get("question")
        category     = row.get("category_validated")
        out_path     = os.path.join(prices_dir, f"{mid}.csv")

        # Resume: skip markets already fetched in a previous run
        if os.path.exists(out_path):
            n_skipped += 1
            continue

        if outcome_type not in ("BINARY", "PSEUDO_NUMERIC"):
            print(f"  [market {mid}] outcome_type={outcome_type} — skipping price history")
            n_no_data += 1
            continue

        print(f"  Fetching bets for {mid} …", end=" ", flush=True)
        try:
            bets = fetch_all_bets(mid)
        except Exception as e:
            print(f"ERROR: {e}")
            n_no_data += 1
            continue

        print(f"{len(bets)} bets", end=" → ", flush=True)

        df_prices = reconstruct_hourly_odds(
            bets, mid, slug, question, category, outcome_type
        )

        if df_prices.empty:
            print("no probability data")
            n_no_data += 1
            continue

        df_prices.sort_values("ts").to_csv(out_path, index=False)
        print(f"{len(df_prices)} hourly rows → {os.path.basename(out_path)}")
        n_saved += 1
        time.sleep(0.05)

    print(f"\nDONE — {n_saved} markets saved, {n_skipped} skipped (already exist), "
          f"{n_no_data} with no data")
    print(f"Prices dir : {prices_dir}")
    print(f"Attrs      : {attrs_csv}")

    return attrs_df, prices_dir


# ── Quality filters (same logic as Polymarket pipeline) ───────────────────────

def compute_market_stats(prices_df, min_duration_hours=24, min_points=10, min_std=0.005):
    """
    Compute per-market statistics and apply quality filters.
    Returns (stats_df, filtered_df).
    """
    if prices_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    def belief_entropy(p, eps=1e-6):
        p = np.clip(p, eps, 1 - eps)
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))

    df = prices_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"])

    stats = (
        df.groupby(["market_id", "slug", "question", "category"])
        .agg(
            t_start=("datetime", "min"),
            t_end=("datetime", "max"),
            n_points=("price", "count"),
            mean_price=("price", "mean"),
            std_price=("price", "std"),
            mean_entropy=("price", lambda x: belief_entropy(x).mean()),
        )
        .reset_index()
    )
    stats["duration_hours"] = (
        stats["t_end"] - stats["t_start"]
    ).dt.total_seconds() / 3600.0

    filtered = stats[
        (stats["duration_hours"] >= min_duration_hours) &
        (stats["n_points"]       >= min_points) &
        (stats["std_price"].fillna(0) >= min_std)
    ].copy()

    return stats, filtered


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ──────────────────────────────────────────────────────────────────────────
    # Fill TEST_MARKET_IDS with Manifold market IDs (alphanumeric strings) to
    # validate specific markets without processing the full comments CSV.
    # Leave empty ([]) to process only the CSV markets.
    #
    # Example Manifold market ID format: "abc123XYZ" (visible in the URL:
    #   https://manifold.markets/username/market-slug#abc123XYZ)
    # ──────────────────────────────────────────────────────────────────────────

    TEST_MARKET_IDS = [
        # ← add Manifold market IDs here for targeted validation, e.g.:
        # "7RZ7CnNVPKOGSh3v5pnn"   # example binary market
        # "kto2oBkpaf6L5oNpKbaF",
    ]

    # STAKEBENCH_MANIFOLD_DIR: folder that receives market_attrs.csv and prices/ (default data/manifold_data).
    # STAKEBENCH_MANIFOLD_COMMENTS: comments CSV whose market_id values are processed in addition to
    # TEST_MARKET_IDS (default <dir>/comments_clean.csv).
    OUT_DIR      = os.environ.get("STAKEBENCH_MANIFOLD_DIR", "data/manifold_data")
    COMMENTS_CSV = os.environ.get("STAKEBENCH_MANIFOLD_COMMENTS", os.path.join(OUT_DIR, "comments_clean.csv"))

    attrs_df, prices_dir = main_by_market_ids(
        market_ids  = TEST_MARKET_IDS,
        out_dir     = OUT_DIR,
        comments_csv= COMMENTS_CSV,
    )

