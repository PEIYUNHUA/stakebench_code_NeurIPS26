"""
get_manifold_topics.py
----------------------
Scan resolved binary Manifold markets through /v0/search-markets (filter=resolved,
contractType=BINARY) and keep markets with volume >= MIN_VOLUME_MANA,
uniqueBettorCount >= MIN_BETTORS and a non-null lastCommentTime. The comment-count filter
is applied in get_manifold_comments.py because commentsCount is not returned by this
endpoint. /v0/markets only returns unresolved markets and is not used.

Outputs (in OUT_DIR, default "manifold_data"):
  valid_markets.jsonl         - full fields, one market per line
  valid_markets_summary.csv   - flat summary

Released as raw/manifold/valid_markets.jsonl and manifold/markets_summary.csv.
"""

import os
import csv
import json
import time
import random
import requests
from collections import Counter

# ── API ────────────────────────────────────────────────────────────────────────
MANIFOLD_API = "https://api.manifold.markets/v0"

MIN_VOLUME_MANA = 10_000
MIN_BETTORS     = 50

PAGE_LIMIT = 100
MAX_PAGES  = 5000
BASE_SLEEP = 0.15
JITTER     = 0.05

OUT_DIR   = "manifold_data"
OUT_JSONL = os.path.join(OUT_DIR, "valid_markets.jsonl")
OUT_CSV   = os.path.join(OUT_DIR, "valid_markets_summary.csv")

QUESTION_CATEGORY_KEYWORDS = {
    "crypto":     ["bitcoin", "btc", "ethereum", "eth", "solana", "sol", "crypto",
                   "defi", "blockchain", "nft", "token", "binance", "coinbase",
                   "dogecoin", "doge", "xrp", "ripple", "stablecoin", "memecoin"],
    "politics":   ["election", "president", "congress", "senate", "vote", "voting",
                   "democrat", "republican", "parliament", "government",
                   "prime minister", "chancellor", "campaign", "candidate", "ballot"],
    "economics":  ["federal reserve", "interest rate", "gdp", "inflation",
                   "recession", "unemployment", "cpi", "fomc", "tariff", "trade war"],
    "sports":     ["nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball",
                   "super bowl", "world cup", "olympics", "nascar", "tennis",
                   "golf", "baseball", "hockey", "playoff", "championship",
                   "tournament", "world series"],
    "technology": ["artificial intelligence", "openai", "chatgpt", "gpt",
                   "apple", "google", "microsoft", "spacex", "tesla",
                   "ipo", "semiconductor", "nvidia", "self-driving"],
}



def _sleep():
    time.sleep(max(0.0, BASE_SLEEP + random.uniform(-JITTER, JITTER)))


def get_json(url, params=None, timeout=30, max_retry=6):
    """See collection/README.md."""
    last_err = None
    for attempt in range(max_retry):
        try:
            r = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            last_err = str(exc)
            time.sleep(BASE_SLEEP * (2 ** attempt))
            continue

        if r.status_code == 200:
            try:
                return r.json()
            except Exception:
                return {"_text": r.text[:300]}

        if r.status_code in (429, 500, 502, 503, 504):
            wait = min(120.0, BASE_SLEEP * (3 ** attempt))
            if r.status_code == 429:
                wait = max(wait, 10.0)
            print(f"  [HTTP {r.status_code}] retry {attempt+1}/{max_retry}, wait {wait:.1f}s")
            time.sleep(wait)
            continue

        try:
            last_err = r.json()
        except Exception:
            last_err = r.text[:300]
        raise RuntimeError(f"HTTP {r.status_code} — {url} params={params} resp={last_err}")

    raise RuntimeError(f"Failed after {max_retry} retries: {url} last={last_err}")


def classify_category(text):
    if not text:
        return "other"
    t = text.lower()
    scores = {cat: sum(1 for kw in kws if kw in t)
              for cat, kws in QUESTION_CATEGORY_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"



def sweep_markets():
    """See collection/README.md."""
    os.makedirs(OUT_DIR, exist_ok=True)

    valid         = []
    seen          = set()
    offset        = 0
    page          = 0
    total_fetched = 0

    print("Sweeping Manifold Markets (resolved BINARY) via search-markets API …")
    print(f"  Filters: volume>={MIN_VOLUME_MANA:,} M$ | bettors>={MIN_BETTORS} | "
          f"lastCommentTime!=None")
    print(f"  (commentsCount filter applied later in get_manifold_comments.py)")
    print()

    while page < MAX_PAGES:
        params = {
            "filter":       "resolved",
            "contractType": "BINARY",
            "sort":         "liquidity",
            "limit":        PAGE_LIMIT,
            "offset":       offset,
        }

        try:
            data = get_json(f"{MANIFOLD_API}/search-markets", params=params)
        except RuntimeError as exc:
            print(f"  [page {page+1}] API error: {exc} — stopping sweep.")
            break

        if not data or not isinstance(data, list):
            print(f"  [page {page+1}] empty response — sweep complete.")
            break

        batch_pass = 0
        for m in data:
            mid = m.get("id")
            if not mid or mid in seen:
                continue
            seen.add(mid)

            vol     = float(m.get("volume") or 0)
            bettors = int(m.get("uniqueBettorCount") or 0)

            if vol < MIN_VOLUME_MANA:
                continue
            if bettors < MIN_BETTORS:
                continue
            if not m.get("lastCommentTime"):
                continue

            batch_pass += 1
            question = m.get("question") or ""

            valid.append({
                "market_id":        mid,
                "slug":             m.get("slug"),
                "question":         question,
                "outcome_type":     m.get("outcomeType"),
                "category":         classify_category(question),
                "resolution":       m.get("resolution"),
                "resolution_time":  m.get("resolutionTime"),
                "close_time":       m.get("closeTime"),
                "created_time":     m.get("createdTime"),
                "volume":           round(vol, 2),
                "unique_bettors":   bettors,
                "total_liquidity":  float(m.get("totalLiquidity") or 0),
                "last_comment_time": m.get("lastCommentTime"),
                "creator_id":       m.get("creatorId"),
                "creator_username": m.get("creatorUsername"),
                "url":              m.get("url"),
            })

        total_fetched += len(data)
        page          += 1
        offset        += PAGE_LIMIT

        print(f"  page {page:4d} | batch={len(data):3} | pass={batch_pass:3} | "
              f"total={total_fetched:7,} | valid={len(valid):5,}")

        _sleep()

        if len(data) < PAGE_LIMIT:
            print(f"  Last page ({len(data)} < {PAGE_LIMIT}) — sweep complete.")
            break

    print(f"\nTotal fetched: {total_fetched:,} | passed all filters: {len(valid):,}")
    return valid



def save_results(valid):
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for m in valid:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"Saved {len(valid):,} markets → {OUT_JSONL}")

    fieldnames = [
        "market_id", "slug", "category", "outcome_type",
        "resolution", "close_time", "resolution_time",
        "volume", "unique_bettors", "total_liquidity",
        "last_comment_time", "creator_username", "question",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(valid)
    print(f"Saved summary CSV → {OUT_CSV}")

    cat_counts = Counter(m["category"] for m in valid)
    res_counts = Counter(m["resolution"] for m in valid)
    total_vol  = sum(m["volume"] for m in valid)

    print(f"\n{'='*65}")
    print(f"  Valid markets   : {len(valid):>8,}")
    print(f"  Total volume    : M${total_vol:>13,.0f}")
    print(f"{'='*65}")

    print("\nCategory breakdown:")
    for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        cat_vol = sum(m["volume"] for m in valid if m["category"] == cat)
        print(f"  {cat:<12s}: {n:5,} markets   M${cat_vol:>13,.0f}")

    print("\nResolution breakdown:")
    for res, n in sorted(res_counts.items(), key=lambda x: -x[1]):
        print(f"  {str(res):<8}: {n:5,}")

    print(f"\nTop 30 markets by volume:")
    top = sorted(valid, key=lambda x: -x["volume"])[:30]
    print(f"  {'market_id':<26}  {'volume':>12}  {'bettors':>7}  question")
    print(f"  {'-'*26}  {'-'*12}  {'-'*7}  {'-'*50}")
    for m in top:
        q = (m["question"] or "")[:55]
        print(f"  {m['market_id']:<26}  "
              f"M${m['volume']:>11,.0f}  "
              f"{m['unique_bettors']:>7,}  {q}")


if __name__ == "__main__":
    valid = sweep_markets()
    if not valid:
        print("\nNo markets passed all filters.")
    else:
        save_results(valid)
        print(f"\nDone. Next step: run get_manifold_comments.py "
              f"using {OUT_JSONL} as input.")
