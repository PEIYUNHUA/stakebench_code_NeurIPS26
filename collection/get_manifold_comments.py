"""
get_manifold_comments.py
------------------------
Collect Manifold comments and each commenter's position at comment time.

  1. Read manifold_data/valid_markets.jsonl (from get_manifold_topics.py); keep markets
     resolved YES or NO.
  2. For each market: fetch all comments (GET /v0/comments?contractId=<id>), extract plain
     text from the ProseMirror JSON, keep comments within [created_time, resolution_time],
     fetch every commenter's bets on that market in parallel and reconstruct the position at
     comment time (net Mana per side; prob_at_comment = probAfter of the last bet before the
     comment).
  3. Append rows to manifold_data/comments.csv with columns
     market_id, slug, question, outcome_type, category, resolution,
     comment_id, created_at, text, user_id, user_name,
     position_side (YES / NO / empty), position_mana, position_shares, prob_at_comment,
     likes, parent_comment_id, reply_count (empty here; filled by clean_comments.py)

Mana is a virtual currency: position_mana is not comparable to Polymarket position_size (USD).
Historical probabilities are taken from bet records (probBefore / probAfter), which is less
precise than Polymarket's /prices-history. Resumable through progress.json.
Released as raw/manifold/comments.csv.
"""
from __future__ import annotations

import os
import csv
import json
import time
import random
import threading
import requests
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── API ────────────────────────────────────────────────────────────────────────
MANIFOLD_API = "https://api.manifold.markets/v0"

MIN_COMMENT_LENGTH      = 3
MIN_COMMENTS_PER_MARKET = 10

TOP_K = 10

BET_WORKERS  = 8
MAX_RETRIES  = 6
BASE_SLEEP   = 0.15
JITTER       = 0.07

IN_JSONL      = "manifold_data/valid_markets.jsonl"
PROGRESS_FILE = "manifold_data/progress.json"
OUT_CSV       = "manifold_data/comments.csv"

CSV_FIELDS = [
    "market_id", "slug", "question", "outcome_type", "category", "resolution",
    "comment_id", "created_at", "text",
    "user_id", "user_name",
    "position_side", "position_mana", "position_shares", "prob_at_comment",
    "likes", "parent_comment_id", "reply_count",
]

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
]


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class ProgressTracker:
    """See collection/README.md."""

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
        e = self._data.get(str(market_id))
        return e is not None and e.get("status") in ("done", "skipped")

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
# ══════════════════════════════════════════════════════════════════════════════

def _sleep(base=BASE_SLEEP, jitter=JITTER):
    time.sleep(max(0.0, base + random.uniform(-jitter, jitter)))


def make_session():
    s = requests.Session()
    s.headers["User-Agent"] = random.choice(_USER_AGENTS)
    return s


def get_json(url, params=None, session=None, timeout=25, max_retry=MAX_RETRIES):
    """See collection/README.md."""
    _s = session or requests.Session()
    for attempt in range(max_retry):
        try:
            r = _s.get(url, params=params, timeout=timeout)
        except requests.RequestException:
            wait = min(60.0, BASE_SLEEP * (2 ** attempt) + random.uniform(0, JITTER * 4))
            time.sleep(wait)
            continue

        if r.status_code == 200:
            try:
                return 200, r.json()
            except Exception:
                return 200, {"_text": r.text[:300]}

        if r.status_code in (429, 500, 502, 503, 504):
            wait = min(120.0, BASE_SLEEP * (3 ** attempt))
            if r.status_code == 429:
                wait = max(wait, 10.0)
            time.sleep(wait)
            continue

        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"_text": r.text[:300]}

    return 0, {"_error": f"failed after {max_retry} retries — {url}"}


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def extract_text(comment: dict) -> str:
    """See collection/README.md."""
    if comment.get("text") and isinstance(comment["text"], str):
        return comment["text"].strip()

    def _walk(node) -> str:
        if not isinstance(node, dict):
            return ""
        if node.get("type") == "text":
            return node.get("text", "")
        parts = [_walk(child) for child in node.get("content", [])]
        sep = "\n" if node.get("type") in ("doc", "paragraph", "blockquote",
                                            "bulletList", "orderedList",
                                            "listItem", "heading") else ""
        return sep.join(p for p in parts if p)

    content = comment.get("content")
    if content:
        return _walk(content).strip()

    return ""


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def fetch_comments(market_id: str, session) -> list[dict]:
    """See collection/README.md."""
    all_comments = []
    seen_ids     = set()
    cursor       = None
    page         = 0
    MAX_PAGES    = 50

    while page < MAX_PAGES:
        params = {"contractId": market_id, "limit": 1000}
        if cursor:
            params["before"] = cursor

        status, data = get_json(f"{MANIFOLD_API}/comments",
                                params=params, session=session)
        if status != 200 or not isinstance(data, list):
            break

        new_batch = []
        for c in data:
            cid = c.get("id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                new_batch.append(c)

        all_comments.extend(new_batch)
        page += 1
        _sleep()

        if len(data) < 1000:
            break

        if data:
            cursor = data[-1].get("id")

    return all_comments


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def fetch_bets(market_id: str, user_id: str, session) -> list[dict]:
    """See collection/README.md."""
    all_bets = []
    seen_ids = set()
    cursor   = None
    page     = 0
    MAX_PAGES = 100

    while page < MAX_PAGES:
        params = {"contractId": market_id, "userId": user_id, "limit": 1000}
        if cursor:
            params["before"] = cursor

        status, data = get_json(f"{MANIFOLD_API}/bets",
                                params=params, session=session)
        if status != 200 or not isinstance(data, list):
            break

        new_batch = []
        for b in data:
            bid = b.get("id")
            if bid and bid not in seen_ids:
                seen_ids.add(bid)
                new_batch.append(b)

        all_bets.extend(new_batch)
        page += 1
        _sleep(BASE_SLEEP / 2, JITTER / 2)

        if len(data) < 1000:
            break
        if data:
            cursor = data[-1].get("id")

    all_bets.sort(key=lambda b: b.get("createdTime", 0))
    return all_bets


def build_position_at_time(bets: list[dict], comment_ts_ms: int) -> tuple:
    """See collection/README.md."""
    net_mana   = {"YES": 0.0, "NO": 0.0}
    net_shares = {"YES": 0.0, "NO": 0.0}

    for bet in bets:
        if bet.get("createdTime", 0) > comment_ts_ms:
            break
        if bet.get("isAnte", False):
            continue

        outcome = bet.get("outcome")
        if outcome not in net_mana:
            continue

        amount = float(bet.get("amount") or 0)
        shares = float(bet.get("shares") or 0)

        net_mana[outcome]   += amount
        net_shares[outcome] += shares

    yes_m, no_m = net_mana["YES"], net_mana["NO"]

    if yes_m > 0 and yes_m >= no_m:
        return "YES", round(yes_m, 4), round(net_shares["YES"], 4)
    elif no_m > 0 and no_m > yes_m:
        return "NO", round(no_m, 4), round(net_shares["NO"], 4)
    return None, 0.0, 0.0


def get_prob_at_time(bets: list[dict], comment_ts_ms: int) -> float | None:
    """See collection/README.md."""
    prob = None
    for bet in bets:
        if bet.get("createdTime", 0) > comment_ts_ms:
            break
        pa = bet.get("probAfter")
        if pa is not None:
            try:
                prob = float(pa)
            except (TypeError, ValueError):
                pass
    return prob


def batch_fetch_bets(market_id: str, user_ids: list[str],
                     max_workers: int = BET_WORKERS) -> dict[str, list[dict]]:
    """See collection/README.md."""
    unique_ids  = list(set(user_ids))
    result      = {}
    result_lock = threading.Lock()

    def _fetch_one(uid):
        session = make_session()
        bets    = fetch_bets(market_id, uid, session)
        return uid, bets

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_one, uid): uid for uid in unique_ids}
        done = 0
        for fut in as_completed(futures):
            uid = futures[fut]
            try:
                uid, bets = fut.result()
                if bets:
                    with result_lock:
                        result[uid] = bets
            except Exception:
                pass
            done += 1

    return result


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def _ms_to_iso(ms) -> str:
    """See collection/README.md."""
    if ms is None:
        return ""
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return str(ms)


def process_market(market: dict, session, writer, writer_lock,
                   progress: ProgressTracker, verbose: bool = True) -> int:
    """See collection/README.md."""
    mid      = market["market_id"]
    slug     = market.get("slug", "")
    question = market.get("question", "")

    if verbose:
        print(f"\n  [{mid}] {question[:60]}")

    comments = fetch_comments(mid, session)
    if verbose:
        print(f"    comments fetched: {len(comments)}")

    if not comments:
        return 0

    lifecycle_start = market.get("created_time")
    lifecycle_end   = market.get("resolution_time") or market.get("close_time")

    valid_comments = []
    user_ids       = []
    n_out_of_range = 0
    for c in comments:
        ctime = c.get("createdTime", 0)
        if lifecycle_start and ctime < lifecycle_start:
            n_out_of_range += 1
            continue
        if lifecycle_end and ctime > lifecycle_end:
            n_out_of_range += 1
            continue

        text = extract_text(c)
        if len(text) < MIN_COMMENT_LENGTH:
            continue
        valid_comments.append((c, text))
        uid = c.get("userId")
        if uid:
            user_ids.append(uid)

    if verbose and n_out_of_range:
        print(f"    dropped {n_out_of_range} comments outside lifecycle")

    if len(valid_comments) < MIN_COMMENTS_PER_MARKET:
        if verbose:
            print(f"    only {len(valid_comments)} valid comments — skipping.")
        return 0

    if verbose:
        print(f"    fetching bets for {len(set(user_ids))} unique users …")
    bets_map = batch_fetch_bets(mid, user_ids)
    if verbose:
        n_with_bets = sum(1 for v in bets_map.values() if v)
        print(f"    {n_with_bets} users have bet records in this market")

    rows = []
    for c, text in valid_comments:
        uid         = c.get("userId")
        created_ms  = c.get("createdTime", 0)
        user_bets   = bets_map.get(uid, [])

        pos_side, pos_mana, pos_shares = build_position_at_time(user_bets, created_ms)
        prob = get_prob_at_time(user_bets, created_ms)

        rows.append({
            "market_id":        mid,
            "slug":             slug,
            "question":         question,
            "outcome_type":     market.get("outcome_type", ""),
            "category":         market.get("category", "other"),
            "resolution":       market.get("resolution", ""),
            "comment_id":       c.get("id", ""),
            "created_at":       _ms_to_iso(created_ms),
            "text":             text,
            "user_id":          uid or "",
            "user_name":        c.get("userUsername") or c.get("userName") or "",
            "position_side":    pos_side or "",
            "position_mana":    pos_mana if pos_mana else "",
            "position_shares":  pos_shares if pos_shares else "",
            "prob_at_comment":  round(prob, 4) if prob is not None else "",
            "likes":            int(c.get("likes") or 0),
            "parent_comment_id": c.get("replyToCommentId") or "",
            "reply_count":      "",
        })

    with writer_lock:
        writer.writerows(rows)

    if verbose:
        has_pos = sum(1 for r in rows if r["position_side"])
        print(f"    wrote {len(rows)} rows | {has_pos} with position data")

    return len(rows)


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def run(in_jsonl=IN_JSONL, out_csv=OUT_CSV, verbose=True):
    if not os.path.exists(in_jsonl):
        raise FileNotFoundError(
            f"{in_jsonl} not found — run get_manifold_topics.py first.")

    markets = []
    with open(in_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                markets.append(json.loads(line))

    print(f"Loaded {len(markets):,} markets from {in_jsonl}")

    markets = [m for m in markets if m.get("resolution") in ("YES", "NO")]
    print(f"  After resolution filter (YES/NO only): {len(markets):,} markets")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    progress = ProgressTracker(PROGRESS_FILE)

    pending = [m for m in markets if not progress.is_done(m["market_id"])]
    if TOP_K is not None:
        pending = pending[:TOP_K]
    done0, skipped0, failed0 = progress.summary()
    print(f"  Progress: {done0} done / {skipped0} skipped / {failed0} failed")
    print(f"  Pending : {len(pending)}" + (f"  (TOP_K={TOP_K})" if TOP_K is not None else ""))

    if not pending:
        print("All markets already processed. Done.")
        return

    file_exists = os.path.exists(out_csv)
    open_mode   = "a" if file_exists else "w"

    writer_lock = threading.Lock()
    session     = make_session()

    with open(out_csv, open_mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        if not file_exists or open_mode == "w":
            writer.writeheader()

        total_rows    = 0
        n_done        = 0
        n_skipped     = 0
        n_failed      = 0

        for i, market in enumerate(pending):
            mid = market["market_id"]
            print(f"\n[{i+1}/{len(pending)}] market {mid}")

            try:
                n = process_market(market, session, writer, writer_lock,
                                   progress, verbose=verbose)
                csvfile.flush()

                if n >= MIN_COMMENTS_PER_MARKET:
                    progress.mark(mid, "done", n)
                    total_rows += n
                    n_done     += 1
                else:
                    progress.mark(mid, "skipped", n)
                    n_skipped  += 1

            except KeyboardInterrupt:
                print("\nInterrupted by user — progress saved.")
                break
            except Exception as exc:
                print(f"  ERROR: {exc}")
                progress.mark(mid, "failed")
                n_failed += 1

            if (i + 1) % 10 == 0:
                print(f"\n  ── Progress summary ──────────────────────────────")
                print(f"     processed={i+1}/{len(pending)} | "
                      f"done={n_done} | skipped={n_skipped} | failed={n_failed}")
                print(f"     total rows written: {total_rows:,}")

            _sleep(BASE_SLEEP, JITTER)

    done_f, skipped_f, failed_f = progress.summary()
    print(f"\n{'='*60}")
    print(f"  Finished.")
    print(f"  done={done_f} | skipped={skipped_f} | failed={failed_f}")
    print(f"  Total rows in this run: {total_rows:,}")
    print(f"  Output: {out_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()
