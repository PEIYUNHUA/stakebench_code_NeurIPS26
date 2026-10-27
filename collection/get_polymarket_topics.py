"""
get_polymarket_topics.py
------------------------
Scan all closed Polymarket markets through the Gamma API, aggregate them by event and keep
events with total volume >= MIN_TOTAL_VOLUME (USD) and event-level comment count
>= MIN_COMMENT_COUNT.

Outputs (in OUT_DIR, default "polysent_data"):
  valid_topics.jsonl        - one event per line with its member markets and the
                              highest-volume market as `best_market`
  valid_topics_summary.csv  - flat summary (one row per event)

Released as raw/polymarket/valid_topics.jsonl and polymarket/topics_summary.csv.
"""

import os
import csv
import json
import time
import requests
from collections import defaultdict

GAMMA     = "https://gamma-api.polymarket.com"
OUT_DIR   = "polysent_data"
OUT_JSONL = os.path.join(OUT_DIR, "valid_topics.jsonl")
OUT_CSV   = os.path.join(OUT_DIR, "valid_topics_summary.csv")

MIN_TOTAL_VOLUME  = 10_000
MIN_COMMENT_COUNT = 100

PAGE_LIMIT = 500
MAX_PAGES  = 2000
SLEEP      = 0.15



def get_json(url, params=None, timeout=30, max_retry=6):
    last_err = None
    for attempt in range(max_retry):
        try:
            r = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            last_err = str(exc)
            time.sleep(SLEEP * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(SLEEP * (attempt + 1) * 2)
            continue
        try:
            last_err = r.json()
        except Exception:
            last_err = r.text[:300]
        raise RuntimeError(f"HTTP {r.status_code} — {url} params={params} resp={last_err}")
    raise RuntimeError(f"Failed after {max_retry} retries: {url} last={last_err}")


def get_volume(m):
    for key in ("volume", "volumeNum", "volume24hr"):
        v = m.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0


def get_event_info(m):
    """See collection/README.md."""
    events = m.get("events") or []
    if not events or not isinstance(events[0], dict):
        return None, None, 0, None
    ev = events[0]
    eid   = ev.get("id")
    title = ev.get("title") or ev.get("name") or ""
    count = int(ev.get("commentCount") or 0)
    series_list = ev.get("series") or []
    sid = None
    if series_list and isinstance(series_list[0], dict):
        sid = series_list[0].get("id")
    try:
        eid = int(eid)
    except (TypeError, ValueError):
        eid = None
    try:
        sid = int(sid)
    except (TypeError, ValueError):
        sid = None
    return eid, title, count, sid


QUESTION_KW = {
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
                   "golf", "baseball", "hockey", "playoff", "championship"],
    "technology": ["artificial intelligence", "openai", "chatgpt", "gpt",
                   "apple", "google", "microsoft", "spacex", "tesla",
                   "ipo", "semiconductor", "nvidia", "self-driving"],
}

def classify_category(text):
    if not text:
        return "other"
    t = text.lower()
    scores = {cat: sum(1 for kw in kws if kw in t) for cat, kws in QUESTION_KW.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"



def sweep_and_aggregate(min_total_volume=MIN_TOTAL_VOLUME, min_comments=MIN_COMMENT_COUNT):
    """See collection/README.md."""
    os.makedirs(OUT_DIR, exist_ok=True)

    events: dict[int, dict] = {}
    no_event: dict[str, dict] = {}

    seen_markets = set()
    total_fetched = 0
    offset = 0
    page   = 0

    print("Sweeping ALL closed markets on Gamma API …")
    print(f"  Filters: event total_volume > ${min_total_volume:,}  |  event comments > {min_comments}")
    print()

    while page < MAX_PAGES:
        params = {"closed": "true", "limit": PAGE_LIMIT, "offset": offset}
        try:
            data = get_json(f"{GAMMA}/markets", params=params)
        except RuntimeError as exc:
            print(f"  [page {page+1}] API error: {exc} — stopping sweep.")
            break

        if not data or not isinstance(data, list):
            print(f"  [page {page+1}] empty response — sweep complete.")
            break

        for m in data:
            mid = m.get("id")
            if mid is None or str(mid) in seen_markets:
                continue
            seen_markets.add(str(mid))

            vol = get_volume(m)
            eid, etitle, cnt, sid = get_event_info(m)
            question = m.get("question") or m.get("title") or ""

            market_rec = {
                "market_id":    str(mid),
                "slug":         m.get("slug"),
                "question":     question,
                "condition_id": m.get("conditionId"),
                "volume":       round(vol, 2),
                "clob_token_ids": m.get("clobTokenIds"),
                "end_date":     m.get("endDate") or m.get("endDateIso"),
                "resolution":   m.get("resolution") or m.get("resolutionSource"),
            }

            if eid is not None:
                if eid not in events:
                    events[eid] = {
                        "event_id":      eid,
                        "event_title":   etitle,
                        "series_id":     sid,
                        "comment_count": cnt,
                        "total_volume":  0.0,
                        "market_count":  0,
                        "markets":       [],
                        "best_market":   None,   # market with highest volume
                    }
                ev = events[eid]
                ev["total_volume"]  = round(ev["total_volume"] + vol, 2)
                ev["market_count"] += 1
                ev["markets"].append(market_rec)
                # keep highest-volume market as representative
                if ev["best_market"] is None or vol > ev["best_market"]["volume"]:
                    ev["best_market"] = market_rec
                    ev["category"]    = classify_category(question)
            else:
                # Standalone market with no event
                key = f"m_{mid}"
                no_event[key] = {
                    "event_id":      None,
                    "event_title":   question,
                    "series_id":     sid,
                    "comment_count": cnt,
                    "total_volume":  round(vol, 2),
                    "market_count":  1,
                    "markets":       [market_rec],
                    "best_market":   market_rec,
                    "category":      classify_category(question),
                }

        total_fetched += len(data)
        page          += 1
        offset        += PAGE_LIMIT

        print(f"  page {page:4d} | fetched={total_fetched:7,} | "
              f"events_seen={len(events):6,} | no_event={len(no_event):5,}")

        time.sleep(SLEEP)

        if len(data) < PAGE_LIMIT:
            print(f"  Last page reached (returned {len(data)} < {PAGE_LIMIT}).")
            break

    all_events = list(events.values()) + list(no_event.values())

    valid = [
        ev for ev in all_events
        if ev["total_volume"] >= min_total_volume
        and ev["comment_count"] >= min_comments
    ]
    valid.sort(key=lambda x: x["comment_count"], reverse=True)

    print(f"\nAggregated: {len(all_events):,} unique events/markets "
          f"→ {len(valid):,} pass both filters")
    return valid



def save_results(valid):
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for ev in valid:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    print(f"Saved {len(valid):,} valid events → {OUT_JSONL}")

    fieldnames = [
        "event_id", "series_id", "category",
        "comment_count", "total_volume", "market_count",
        "event_title", "representative_question",
        "best_market_id", "best_market_volume",
        "end_date", "resolution",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ev in valid:
            bm = ev.get("best_market") or {}
            writer.writerow({
                "event_id":               ev["event_id"],
                "series_id":              ev["series_id"],
                "category":               ev.get("category", "other"),
                "comment_count":          ev["comment_count"],
                "total_volume":           f"{ev['total_volume']:,.2f}",
                "market_count":           ev["market_count"],
                "event_title":            ev["event_title"],
                "representative_question": bm.get("question", ""),
                "best_market_id":         bm.get("market_id", ""),
                "best_market_volume":     f"{bm.get('volume', 0):,.2f}",
                "end_date":               bm.get("end_date", ""),
                "resolution":             bm.get("resolution", ""),
            })
    print(f"Saved summary CSV → {OUT_CSV}")

    from collections import Counter
    cat_counts = Counter(ev.get("category", "other") for ev in valid)
    total_vol  = sum(ev["total_volume"] for ev in valid)
    total_cmts = sum(ev["comment_count"] for ev in valid)

    print(f"\n{'='*65}")
    print(f"  Valid events  : {len(valid):>8,}")
    print(f"  Total volume  : ${total_vol:>14,.0f}")
    print(f"  Total comments: {total_cmts:>14,}  (event-level, may overlap across markets)")
    print(f"{'='*65}")

    print("\nCategory breakdown (by event count):")
    for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        cat_vol = sum(ev["total_volume"] for ev in valid if ev.get("category") == cat)
        print(f"  {cat:<12s}: {n:5,} events   ${cat_vol:>14,.0f} total volume")

    print(f"\nTop 30 events by comment_count:")
    header = f"  {'event_id':>8}  {'comments':>8}  {'total_vol':>14}  {'mkts':>4}  event_title"
    print(header)
    print(f"  {'-'*8}  {'-'*8}  {'-'*14}  {'-'*4}  {'-'*45}")
    for ev in valid[:30]:
        title = (ev["event_title"] or "")[:55]
        print(f"  {str(ev['event_id'] or ''):>8}  "
              f"{ev['comment_count']:>8,}  "
              f"${ev['total_volume']:>13,.0f}  "
              f"{ev['market_count']:>4}  {title}")


if __name__ == "__main__":
    valid = sweep_and_aggregate(
        min_total_volume=MIN_TOTAL_VOLUME,
        min_comments=MIN_COMMENT_COUNT,
    )

    if not valid:
        print("\nNo events passed all three filters.")
    else:
        save_results(valid)
        print(f"\nDone. Next step: use event_id / best_market_id from {OUT_JSONL} "
              f"as input to get_market_comments.py")
