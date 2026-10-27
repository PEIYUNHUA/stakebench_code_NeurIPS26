import os
import json
import time
import requests
import pandas as pd
import numpy as np

GAMMA = "https://gamma-api.polymarket.com"
CLOB  = "https://clob.polymarket.com"

# PolySent dataset: target research categories (Guide §4.3)
CATEGORY_KEYWORDS = {
    "crypto":     ["crypto", "cryptocurrency", "bitcoin", "btc", "ethereum", "eth",
                   "solana", "sol", "defi", "stablecoin", "memecoin", "altcoin", "doge"],
    "politics":   ["election", "president", "congress", "senate", "vote", "political",
                   "democrat", "republican", "parliament", "government"],
    "economics":  ["fed", "federal reserve", "interest rate", "gdp", "inflation",
                   "economy", "recession", "unemployment", "cpi", "fomc"],
    "sports":     ["nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball",
                   "championship", "super bowl", "world cup", "olympics", "nascar"],
    "technology": ["ai", "artificial intelligence", "openai", "tech", "apple", "google",
                   "microsoft", "spacex", "tesla", "ipo"],
}
MIN_VOLUME           = 100_000  # Guide §4.3: market total volume > $100,000
MIN_EVENT_COMMENTS   = 10       # Pre-filter: skip markets whose event has < N comments

# Keywords for post-hoc category validation on question text (broader than tag keywords)
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
    """
    Post-hoc category assignment based on question text keywords.
    Returns best-matching category name, or None if no keyword matches.
    """
    if not text:
        return None
    text_lower = text.lower()
    scores = {
        cat: sum(1 for kw in kws if kw in text_lower)
        for cat, kws in QUESTION_CATEGORY_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def get_json(url, params=None, timeout=30, max_retry=6, sleep=0.7):
    last_err = None
    for i in range(max_retry):
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (418, 429, 500, 502, 503, 504):
            time.sleep(sleep * (i + 1))
            continue
        try:
            last_err = r.json()
        except Exception:
            last_err = r.text[:300]
        raise RuntimeError(f"HTTP {r.status_code} {url} params={params} resp={last_err}")
    raise RuntimeError(f"Failed after retries: {url} params={params} last={last_err}")


# ── Gamma API helpers ──────────────────────────────────────────────────────────

def fetch_all_tags(limit=200, max_pages=50):
    tags = []
    offset = 0
    for _ in range(max_pages):
        data = get_json(f"{GAMMA}/tags", params={"limit": limit, "offset": offset})
        if not data:
            break
        tags.extend(data)
        offset += limit
        time.sleep(0.05)
    return tags


def pick_category_tag_ids(tags, categories=None):
    """
    Filter tags by target research categories for PolySent dataset.
    categories: list of keys from CATEGORY_KEYWORDS, or None for all.
    Returns list of dicts: {id, name, slug, category}.
    """
    if categories is None:
        categories = list(CATEGORY_KEYWORDS.keys())

    out = []
    seen = set()
    for t in tags:
        name = str(t.get("name", "")).lower()
        slug = str(t.get("slug", "")).lower()
        for cat in categories:
            kws = CATEGORY_KEYWORDS[cat]
            if any(k in name or k in slug for k in kws):
                tid = t.get("id")
                if tid is not None and tid not in seen:
                    seen.add(tid)
                    out.append({
                        "id":       tid,
                        "name":     t.get("name"),
                        "slug":     t.get("slug"),
                        "category": cat,
                    })
                break  # avoid double-counting same tag across categories
    return out


def fetch_markets_by_tag(tag_id, limit=200, closed=None):
    markets = []
    offset = 0
    while True:
        params = {
            "tag_id":       tag_id,
            "include_tag":  True,
            "related_tags": True,
            "limit":        limit,
            "offset":       offset,
        }
        if closed is not None:
            params["closed"] = bool(closed)

        data = get_json(f"{GAMMA}/markets", params=params)
        if not data:
            break
        markets.extend(data)
        offset += limit
        time.sleep(0.05)
    return markets


def fetch_market_by_id(market_id):
    """
    Fetch a single market's full metadata from Gamma API by its integer market_id.
    Returns the market dict, or None if not found.
    """
    data = get_json(
        f"{GAMMA}/markets",
        params={"id": int(market_id), "related_tags": True, "include_tag": True},
    )
    if isinstance(data, list) and len(data) > 0:
        return data[0]
    if isinstance(data, dict) and data.get("id"):
        return data
    return None


def extract_market_attrs(market_obj):
    """
    Extract a rich flat dict of market attributes from the Gamma API response.
    Covers identity, timing, liquidity, resolution, social, and tag metadata.
    """
    if market_obj is None:
        return {}

    # ── tags ──────────────────────────────────────────────────────────────────
    tags_raw = market_obj.get("tags") or []
    tag_slugs  = [t.get("slug")  for t in tags_raw if isinstance(t, dict)]
    tag_labels = [t.get("label") for t in tags_raw if isinstance(t, dict)]

    # ── events ────────────────────────────────────────────────────────────────
    events = market_obj.get("events") or []
    event  = events[0] if events and isinstance(events[0], dict) else {}

    # ── outcomes ──────────────────────────────────────────────────────────────
    outcomes_raw = market_obj.get("outcomes")
    if isinstance(outcomes_raw, str):
        try:
            outcomes_raw = json.loads(outcomes_raw)
        except Exception:
            outcomes_raw = [outcomes_raw]
    if not isinstance(outcomes_raw, list):
        outcomes_raw = []

    # ── outcomePrices ─────────────────────────────────────────────────────────
    outcome_prices_raw = market_obj.get("outcomePrices")
    if isinstance(outcome_prices_raw, str):
        try:
            outcome_prices_raw = json.loads(outcome_prices_raw)
        except Exception:
            outcome_prices_raw = []
    if not isinstance(outcome_prices_raw, list):
        outcome_prices_raw = []

    question_text = market_obj.get("question") or market_obj.get("title") or ""

    # ── resolution outcome ─────────────────────────────────────────────────────
    # Gamma API does not expose a "resolution" result field directly.
    # Derive it: whichever outcome has outcomePrices closest to 1.0 is the winner.
    resolution = None
    if outcomes_raw and outcome_prices_raw and len(outcomes_raw) == len(outcome_prices_raw):
        try:
            prices_float = [float(p) for p in outcome_prices_raw]
            resolution = outcomes_raw[prices_float.index(max(prices_float))]
        except Exception:
            pass

    return {
        # ── identity ──────────────────────────────────────────────────────────
        "market_id":            str(market_obj.get("id", "")),
        "slug":                 market_obj.get("slug"),
        "question":             question_text,
        "description":          market_obj.get("description"),
        "condition_id":         market_obj.get("conditionId"),
        "clob_token_ids":       json.dumps(parse_clob_token_ids(market_obj)),

        # ── timing ────────────────────────────────────────────────────────────
        "start_date":           market_obj.get("startDate") or market_obj.get("startDateIso"),
        "end_date":             market_obj.get("endDate")   or market_obj.get("endDateIso"),
        "created_at":           market_obj.get("createdAt"),
        "updated_at":           market_obj.get("updatedAt"),

        # ── status / resolution ───────────────────────────────────────────────
        "closed":               market_obj.get("closed", False),
        "active":               market_obj.get("active"),
        "archived":             market_obj.get("archived"),
        "resolved":             market_obj.get("resolved", market_obj.get("closed", False)),
        "resolution":           resolution,

        # ── liquidity & trading ───────────────────────────────────────────────
        "volume":               _safe_float(market_obj.get("volume")),
        "liquidity":            _safe_float(market_obj.get("liquidity")),
        "best_bid":             _safe_float(market_obj.get("bestBid")),
        "best_ask":             _safe_float(market_obj.get("bestAsk")),
        "spread":               _safe_float(market_obj.get("spread")),
        "last_trade_price":     _safe_float(market_obj.get("lastTradePrice")),

        # ── market-type ───────────────────────────────────────────────────────
        "market_type":          market_obj.get("marketType"),
        "is_binary":            bool(len(outcomes_raw) == 2),
        "outcomes":             json.dumps(outcomes_raw),
        "outcome_prices":       json.dumps(outcome_prices_raw),

        # ── category / tags ───────────────────────────────────────────────────
        "category_validated":   classify_by_question(question_text),
        "tag_slugs":            json.dumps(tag_slugs),
        "tag_labels":           json.dumps(tag_labels),

        # ── event metadata ────────────────────────────────────────────────────
        "event_id":             str(event.get("id", "")),
        "event_title":          event.get("title"),
        "event_slug":           event.get("slug"),
        "event_comment_count":  int(event.get("commentCount") or 0),
        "event_start_date":     event.get("startDate"),
        "event_end_date":       event.get("endDate"),

        # ── social ────────────────────────────────────────────────────────────
        "comment_count":        int(market_obj.get("commentCount") or 0),
        "favorites_count":      int(market_obj.get("favoritesCount") or 0),

        # ── misc ──────────────────────────────────────────────────────────────
        "image":                market_obj.get("image"),
        "icon":                 market_obj.get("icon"),
        "approved":             market_obj.get("approved"),
        "neg_risk":             market_obj.get("negRisk"),
        "neg_risk_market_id":   market_obj.get("negRiskMarketID"),
    }


def _safe_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── CLOB / price helpers ───────────────────────────────────────────────────────

def get_market_volume(market_obj):
    """Extract numeric volume from market object."""
    for key in ("volume", "volumeNum", "volume24hr"):
        v = market_obj.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0


def get_market_resolution(market_obj):
    """Extract resolution info (Guide §4.1: ground truth for resolved markets)."""
    return {
        "closed":     market_obj.get("closed", False),
        "resolved":   market_obj.get("resolved", market_obj.get("closed", False)),
        "resolution": market_obj.get("resolution") or market_obj.get("resolutionSource"),
        "end_date":   market_obj.get("endDate") or market_obj.get("endDateIso"),
    }


def get_event_comment_count(market_obj):
    """
    Extract commentCount from the first event associated with this market.
    Comments are stored at the Event level in the Polymarket API.
    """
    events = market_obj.get("events") or []
    if events and isinstance(events[0], dict):
        return int(events[0].get("commentCount") or 0)
    return 0


def parse_clob_token_ids(market_obj):
    v = market_obj.get("clobTokenIds")
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return [str(x) for x in arr]
            except Exception:
                pass
        if "," in s:
            return [x.strip() for x in s.split(",") if x.strip()]
        return [s]
    return []


def fetch_price_history(token_id, interval="max", fidelity=60, startTs=None, endTs=None):
    params = {"market": str(token_id)}
    if startTs is not None or endTs is not None:
        if startTs is not None: params["startTs"] = int(startTs)
        if endTs is not None:   params["endTs"]   = int(endTs)
    else:
        params["interval"] = interval
    params["fidelity"] = int(fidelity)
    js = get_json(f"{CLOB}/prices-history", params=params)
    return js.get("history", [])


def fetch_price_history_chunked(token_id, startTs, endTs, fidelity=60,
                                 chunk_days=5):
    """
    Fetch CLOB price history over a long time range by splitting into chunks.
    The API rejects requests where endTs - startTs is too large (~30+ days at
    hourly fidelity), so we page through in chunk_days-sized windows.
    Returns deduplicated list of {t, p} dicts sorted by t.
    """
    chunk_sec = chunk_days * 24 * 3600
    all_pts   = {}   # ts -> price (dedup by timestamp)

    t = int(startTs)
    end = int(endTs)

    while t < end:
        chunk_end = min(t + chunk_sec, end)
        try:
            pts = fetch_price_history(token_id, fidelity=fidelity,
                                      startTs=t, endTs=chunk_end)
            for pt in pts:
                all_pts[int(pt["t"])] = float(pt["p"])
        except Exception as e:
            print(f"    chunk [{t}→{chunk_end}] error: {e}")
        t = chunk_end
        time.sleep(0.1)

    return [{"t": k, "p": v} for k, v in sorted(all_pts.items())]


# ── Main pipeline A: by tag (original) ────────────────────────────────────────

def main(out_dir="polysent_data", fidelity=60, categories=None,
         min_volume=MIN_VOLUME, min_event_comments=MIN_EVENT_COMMENTS):
    """
    Collect PolySent dataset: resolved multi-category markets with price histories.
    Guide §4.1: only resolved markets (ground truth), volume > $100,000.
    Guide §4.3: min_event_comments pre-filters markets before price history fetch.
    fidelity=3600: hourly price data (3600 seconds) for comment-window construction (Guide §4.2).
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) tags → category tag ids
    print("Fetching all tags...")
    tags = fetch_all_tags(limit=200)
    category_tags = pick_category_tag_ids(tags, categories=categories)
    print(f"Found {len(category_tags)} category tags across {len(set(t['category'] for t in category_tags))} categories")

    with open(os.path.join(out_dir, "category_tags.json"), "w", encoding="utf-8") as f:
        json.dump(category_tags, f, ensure_ascii=False, indent=2)

    # 2) markets by tag (resolved only, volume-filtered)
    all_markets = {}      # market_id -> market_obj
    market_categories = {}  # market_id -> category

    for t in category_tags:
        tid = t["id"]
        cat = t["category"]
        mkts = fetch_markets_by_tag(tid, limit=200, closed=True)
        added = 0
        for m in mkts:
            mid = str(m.get("id"))
            if not mid or mid == "None":
                continue
            if get_market_volume(m) < min_volume:
                continue
            if mid not in all_markets:
                all_markets[mid] = m
                market_categories[mid] = cat
                added += 1
        print(f"[tag {tid} / {cat}] fetched={len(mkts)} added_after_filter={added} total_unique={len(all_markets)}")

    # Attach category, resolution info, and comment count to each market
    for mid, m in all_markets.items():
        m["_category"]             = market_categories.get(mid)
        m["_category_validated"]   = classify_by_question(m.get("question") or m.get("title") or "")
        m["_resolution_info"]      = get_market_resolution(m)
        m["_event_comment_count"]  = get_event_comment_count(m)

    # Apply minimum comment count pre-filter
    before = len(all_markets)
    all_markets = {
        mid: m for mid, m in all_markets.items()
        if m["_event_comment_count"] >= min_event_comments
    }
    print(f"Comment pre-filter (>={min_event_comments}): {before} → {len(all_markets)} markets")

    # Save markets metadata
    markets_path = os.path.join(out_dir, "markets.jsonl")
    with open(markets_path, "w", encoding="utf-8") as f:
        for m in all_markets.values():
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_markets)} markets → {markets_path}")

    # 3) hourly price histories
    rows = []
    for m in all_markets.values():
        mid      = str(m.get("id"))
        slug     = m.get("slug")
        question = m.get("question") or m.get("title")
        category = m.get("_category")
        token_ids = parse_clob_token_ids(m)

        if not token_ids:
            continue

        for token_id in token_ids:
            try:
                hist = fetch_price_history(token_id, interval="max", fidelity=fidelity)
            except Exception as e:
                print(f"  [token {token_id}] price history error: {e}")
                continue
            for pt in hist:
                rows.append({
                    "market_id": mid,
                    "slug":      slug,
                    "question":  question,
                    "category":  category,
                    "token_id":  str(token_id),
                    "ts":        int(pt["t"]),
                    "datetime":  pd.to_datetime(int(pt["t"]), unit="s", utc=True),
                    "price":     float(pt["p"]),
                })
            time.sleep(0.05)

        print(f"[market {mid}] tokens={len(token_ids)} rows_so_far={len(rows)}")

    df = pd.DataFrame(rows).sort_values(["market_id", "token_id", "ts"])
    df.to_parquet(os.path.join(out_dir, "prices.parquet"), index=False)
    df.to_csv(os.path.join(out_dir, "prices.csv"), index=False)

    print("DONE")
    print(f"Markets : {markets_path}")
    print(f"Prices  : {os.path.join(out_dir, 'prices.parquet')}")


# ── Main pipeline B: by market_id list (driven by comments_clean.csv) ─────────

def main_by_market_ids(
    market_ids,
    out_dir="polysent_data",
    fidelity=60,
    comments_csv=None,
):
    """
    Fetch market metadata + hourly odds for a specific list of market_ids.

    Args:
        market_ids:   list of int/str market IDs to process.
        out_dir:      output directory.
        fidelity:     CLOB price-history fidelity in minutes (60 = hourly).
        comments_csv: path to comments_clean.csv; if given, market_ids are
                      auto-populated from its 'market_id' column (then deduplicated
                      with any ids supplied in the market_ids argument).

    Outputs:
        {out_dir}/market_attrs.csv   – rich flat market attributes
        {out_dir}/market_attrs.jsonl – full raw Gamma API response per market
        {out_dir}/prices_by_id.csv   – hourly price rows with outcome labels
        {out_dir}/prices_by_id.parquet
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. resolve market_ids ──────────────────────────────────────────────────
    ids_set = set(str(mid) for mid in market_ids if mid)

    if comments_csv and os.path.isfile(comments_csv):
        df_comments = pd.read_csv(comments_csv, usecols=["market_id"], encoding="latin-1")
        csv_ids = set(df_comments["market_id"].dropna().astype(str).unique())
        print(f"Loaded {len(csv_ids)} unique market_ids from {comments_csv}")
        ids_set |= csv_ids
    else:
        if comments_csv:
            print(f"[warn] comments_csv not found: {comments_csv}")

    market_ids_sorted = sorted(ids_set, key=lambda x: int(x) if x.isdigit() else x)
    print(f"Processing {len(market_ids_sorted)} market_ids total")

    # ── 2. fetch market metadata from Gamma API ────────────────────────────────
    market_rows  = []   # flat attribute dicts for CSV
    market_raws  = []   # full raw API responses for JSONL
    token_map    = {}   # market_id -> list of token_ids
    outcome_map  = {}   # token_id  -> outcome label

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
        market_raws.append(m)

        token_ids = parse_clob_token_ids(m)
        token_map[mid] = token_ids

        # build token → outcome label mapping
        outcomes_raw = m.get("outcomes") or "[]"
        if isinstance(outcomes_raw, str):
            try:
                outcomes_raw = json.loads(outcomes_raw)
            except Exception:
                outcomes_raw = []
        if isinstance(outcomes_raw, list) and len(outcomes_raw) == len(token_ids):
            for tid, out in zip(token_ids, outcomes_raw):
                outcome_map[str(tid)] = str(out).strip()

        print(f"ok (tokens={len(token_ids)}, tags={len(json.loads(attrs['tag_slugs']))})")
        time.sleep(0.12)

    # save market attributes
    attrs_df = pd.DataFrame(market_rows)
    attrs_csv = os.path.join(out_dir, "market_attrs.csv")
    attrs_df.to_csv(attrs_csv, index=False)
    print(f"Saved market attributes → {attrs_csv}  ({len(attrs_df)} markets)")

    # ── 3. fetch hourly price histories — one file per market ─────────────────
    prices_dir = os.path.join(out_dir, "prices")
    os.makedirs(prices_dir, exist_ok=True)

    def _to_ts(iso_str):
        if not iso_str:
            return None
        try:
            return int(pd.to_datetime(iso_str, utc=True).timestamp())
        except Exception:
            return None

    n_saved = 0
    n_skipped = 0
    n_no_data = 0

    for mid, token_ids in token_map.items():
        out_path = os.path.join(prices_dir, f"{mid}.csv")

        # Resume: skip markets already fetched in a previous run
        if os.path.exists(out_path):
            n_skipped += 1
            continue

        if not token_ids:
            print(f"  [market {mid}] no CLOB token ids — skipping prices")
            n_no_data += 1
            continue

        m_row    = next((r for r in market_rows if r["market_id"] == mid), {})
        slug     = m_row.get("slug")
        question = m_row.get("question")
        category = m_row.get("category_validated") or m_row.get("category_api")
        start_ts = _to_ts(m_row.get("start_date")) or 1577836800  # fallback 2020-01-01
        end_ts   = _to_ts(m_row.get("end_date"))   or int(time.time())

        price_rows = []
        for token_id in token_ids:
            outcome_label = outcome_map.get(str(token_id), "unknown")
            hist = []

            # Strategy 1: chunked startTs/endTs — true hourly candles.
            # fidelity in minutes: 60 = 1-hour. API rejects spans > ~7 days.
            try:
                hist = fetch_price_history_chunked(str(token_id),
                                                   startTs=start_ts, endTs=end_ts,
                                                   fidelity=fidelity, chunk_days=5)
            except Exception as e:
                print(f"  [token …{str(token_id)[-12:]}] chunked error: {e}")

            # Strategy 2: interval=max fallback (coarser, but better than nothing)
            if not hist:
                try:
                    hist = fetch_price_history(str(token_id), interval="max", fidelity=fidelity)
                except Exception as e:
                    print(f"  [token …{str(token_id)[-12:]}] interval=max error: {e}")

            # Snap timestamps to clean hour boundaries (12:00:03 → 12:00:00)
            for pt in hist:
                dt_hour = pd.to_datetime(int(pt["t"]), unit="s", utc=True).floor("h")
                price_rows.append({
                    "market_id": mid,
                    "slug":      slug,
                    "question":  question,
                    "category":  category,
                    "token_id":  str(token_id),
                    "outcome":   outcome_label,
                    "ts":        int(dt_hour.timestamp()),
                    "datetime":  dt_hour,
                    "price":     float(pt["p"]),
                })
            time.sleep(0.05)

        if price_rows:
            mdf = pd.DataFrame(price_rows).sort_values(["token_id", "ts"])
            # Drop duplicate (token_id, ts) pairs that can arise from chunk overlaps
            mdf = mdf.drop_duplicates(subset=["token_id", "ts"])
            mdf.to_csv(out_path, index=False)
            print(f"  [market {mid}] {len(mdf)} rows → {os.path.basename(out_path)}")
            n_saved += 1
        else:
            print(f"  [market {mid}] no price data found")
            n_no_data += 1

    print(f"\nDONE — {n_saved} markets saved, {n_skipped} skipped (already exist), "
          f"{n_no_data} with no data")
    print(f"Prices dir : {prices_dir}")
    print(f"Attrs      : {attrs_csv}")

    return attrs_df, prices_dir


# ── helpers for adding outcome labels ─────────────────────────────────────────

def _to_list(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return [str(i) for i in arr]
            except Exception:
                pass
        if "," in s:
            return [t.strip().strip('"').strip("'") for t in s.split(",") if t.strip()]
        return [s.strip().strip('"').strip("'")]
    return [str(x)]


def build_token_outcome_map(markets_df: pd.DataFrame) -> pd.DataFrame:
    tmp = markets_df.copy()
    tmp["market_id"]      = tmp["id"].astype(str)
    tmp["token_ids"]      = tmp["clobTokenIds"].apply(_to_list)
    tmp["outcomes_list"]  = tmp["outcomes"].apply(_to_list)

    ok = (tmp["token_ids"].str.len().fillna(0).astype(int) ==
          tmp["outcomes_list"].str.len().fillna(0).astype(int))
    tmp = tmp.loc[ok, ["market_id", "token_ids", "outcomes_list"]].copy()
    tmp = tmp.explode(["token_ids", "outcomes_list"], ignore_index=True)
    tmp = tmp.rename(columns={"token_ids": "token_id", "outcomes_list": "outcome"})
    tmp["token_id"]    = tmp["token_id"].astype(str)
    tmp["outcome"]     = tmp["outcome"].astype(str).str.strip()
    tmp["outcome_norm"] = tmp["outcome"].str.lower()
    norm = {"y": "yes", "n": "no", "true": "yes", "false": "no"}
    tmp["outcome_norm"] = tmp["outcome_norm"].map(lambda x: norm.get(x, x))
    return tmp[["market_id", "token_id", "outcome", "outcome_norm"]]


def add_outcome_to_prices(prices_df: pd.DataFrame, markets_df: pd.DataFrame) -> pd.DataFrame:
    prices = prices_df.copy()
    prices["market_id"] = prices["market_id"].astype(str)
    prices["token_id"]  = prices["token_id"].astype(str)
    mapping_df = build_token_outcome_map(markets_df)
    return prices.merge(mapping_df, on=["market_id", "token_id"], how="left")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ──────────────────────────────────────────────────────────────────────────
    # Pipeline B: fetch odds by market_id
    # Driven by comments_clean.csv + optional manual TEST_MARKET_IDS list.
    #
    # Fill TEST_MARKET_IDS to validate specific markets without re-running the
    # full comments CSV.  Leave it empty ([]) to process only the CSV markets.
    # ──────────────────────────────────────────────────────────────────────────

    TEST_MARKET_IDS = [
        253591,
        519075
    ]

    # STAKEBENCH_POLYMARKET_DIR: folder that receives market_attrs.csv and prices/ (default data/polymarket_data).
    # STAKEBENCH_POLYMARKET_COMMENTS: optional comments CSV whose market_id values are processed in
    # addition to TEST_MARKET_IDS (e.g. <dir>/comments_clean.csv); unset = TEST_MARKET_IDS only.
    OUT_DIR      = os.environ.get("STAKEBENCH_POLYMARKET_DIR", "data/polymarket_data")
    COMMENTS_CSV = os.environ.get("STAKEBENCH_POLYMARKET_COMMENTS") or None

    attrs_df, prices_dir = main_by_market_ids(
        market_ids  = TEST_MARKET_IDS,
        out_dir     = OUT_DIR,
        fidelity    = 60,          # fidelity in minutes: 60 = 1-hour candles
        comments_csv= COMMENTS_CSV,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Pipeline A: full tag-based crawl (uncomment to run)
    # ──────────────────────────────────────────────────────────────────────────
    # main(
    #     out_dir="polysent_data",
    #     fidelity=60,
    #     categories=None,
    #     min_volume=MIN_VOLUME,
    #     min_event_comments=MIN_EVENT_COMMENTS,
    # )
    #
    # prices_df  = pd.read_csv("polysent_data/prices.csv")
    # markets_df = pd.read_json("polysent_data/markets.jsonl", lines=True)
    # prices_with_outcome = add_outcome_to_prices(prices_df, markets_df)
