"""
get_manifold_position.py
------------------------
Collect the full bet history of every Manifold commenter that holds a position in
comments_clean.csv (used to build the G3 action labels).

  1. Read comments_clean.csv: user IDs with a non-empty position_side, and the set of
     market IDs.
  2. For each user, page GET /v0/bets with afterTime / beforeTime and bet-id cursors.
  3. Keep bets whose contractId is in the comment markets.
  4. Write user_positions.csv with columns
     user_id, username, bet_id, created_at, contract_id, slug, question,
     outcome, amount, shares, prob_before, prob_after,
     is_redemption, is_cancelled, is_filled

Resumable through positions_progress.json. API: https://docs.manifold.markets/api
(500 requests/min, no authentication). Released as raw/manifold/user_positions.csv.
"""

import os
import csv
import json
import time
import random
import threading
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

BASE_DIR      = "data/manifold_data"
INPUT_CSV     = os.path.join(BASE_DIR, "comments_clean.csv")
OUTPUT_CSV    = os.path.join(BASE_DIR, "user_positions.csv")
PROGRESS_FILE = os.path.join(BASE_DIR, "positions_progress.json")

MANIFOLD_API = "https://api.manifold.markets/v0"
PAGE_LIMIT   = 1000
MAX_RETRIES  = 6
BASE_SLEEP   = 0.15
JITTER       = 0.10

START_MS = int(datetime(2021, 12, 30, tzinfo=timezone.utc).timestamp() * 1000)
END_MS   = int(datetime(2026,  3,  8, tzinfo=timezone.utc).timestamp() * 1000)

WORKERS = 6

OUTPUT_FIELDS = [
    "user_id", "username", "bet_id", "created_at",
    "contract_id", "slug", "question",
    "outcome", "amount", "shares", "prob_before", "prob_after",
    "is_redemption", "is_cancelled", "is_filled",
]


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def _sleep(base=BASE_SLEEP, jitter=JITTER):
    time.sleep(max(0.0, base + random.uniform(-jitter, jitter)))


def get_json(url, params=None, timeout=30, max_retry=MAX_RETRIES):
    """See collection/README.md."""
    last_err = None
    for attempt in range(max_retry):
        try:
            r = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            last_err = str(exc)
            _sleep(BASE_SLEEP * (attempt + 1), JITTER)
            continue

        if r.status_code == 200:
            return r.json()

        if r.status_code in (429, 500, 502, 503, 504):
            wait = BASE_SLEEP * (2 ** attempt) + random.uniform(0, JITTER)
            time.sleep(wait)
            last_err = f"HTTP {r.status_code}"
            continue

        try:
            last_err = r.json()
        except Exception:
            last_err = r.text[:300]
        raise RuntimeError(
            f"HTTP {r.status_code} — {url} params={params} resp={last_err}"
        )

    raise RuntimeError(
        f"Failed after {max_retry} retries: {url} last={last_err}"
    )


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

    def is_done(self, user_id):
        return self._data.get(user_id, {}).get("status") == "done"

    def mark(self, user_id, status, n_rows=0):
        with self._lock:
            self._data[user_id] = {
                "status": status,
                "n_rows": n_rows,
                "ts":     datetime.now(timezone.utc).isoformat(),
            }
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)

    def summary(self):
        done   = sum(1 for v in self._data.values() if v["status"] == "done")
        failed = sum(1 for v in self._data.values() if v["status"] == "failed")
        return done, failed


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def _make_row(bet, user_id, username, market_meta):
    """See collection/README.md."""
    created_ms = bet.get("createdTime")
    created_at = (
        datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc).isoformat()
        if created_ms else None
    )
    contract_id = bet.get("contractId", "")
    meta = market_meta.get(contract_id, {})
    return {
        "user_id":       user_id,
        "username":      username,
        "bet_id":        bet.get("id") or bet.get("betId"),
        "created_at":    created_at,
        "contract_id":   contract_id,
        "slug":          meta.get("slug"),
        "question":      meta.get("question"),
        "outcome":       bet.get("outcome"),
        "amount":        bet.get("amount"),
        "shares":        bet.get("shares"),
        "prob_before":   bet.get("probBefore"),
        "prob_after":    bet.get("probAfter"),
        "is_redemption": bet.get("isRedemption"),
        "is_cancelled":  bet.get("isCancelled"),
        "is_filled":     bet.get("isFilled"),
    }


def fetch_user_positions(user_id, username, relevant_market_ids, market_meta):
    """See collection/README.md."""
    results   = []
    seen_ids  = set()
    before_id = None

    while True:
        params = {
            "userId":     user_id,
            "limit":      PAGE_LIMIT,
            "afterTime":  START_MS,
            "beforeTime": END_MS,
        }
        if before_id is not None:
            params["before"] = before_id

        try:
            page = get_json(f"{MANIFOLD_API}/bets", params=params)
        except RuntimeError as exc:
            raise RuntimeError(
                f"[{user_id[:10]}…] fetch error: {exc}"
            ) from exc

        if not page or not isinstance(page, list):
            break

        for bet in page:
            bet_id = bet.get("id") or bet.get("betId")
            if bet_id in seen_ids:
                continue
            seen_ids.add(bet_id)

            if bet.get("contractId") in relevant_market_ids:
                results.append(_make_row(bet, user_id, username, market_meta))

        _sleep()

        if len(page) < PAGE_LIMIT:
            break

        before_id = page[-1].get("id") or page[-1].get("betId")
        if not before_id:
            break

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def load_inputs(csv_path=INPUT_CSV):
    """See collection/README.md."""
    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)

    relevant_ids = set(df["market_id"].dropna().astype(str))

    market_meta = {}
    for _, row in df[["market_id", "slug", "question"]].drop_duplicates(
        subset="market_id"
    ).iterrows():
        mid = str(row["market_id"])
        market_meta[mid] = {
            "slug":     row.get("slug"),
            "question": row.get("question"),
        }

    has_pos = df[df["position_side"].notna() & df["user_address"].notna()].copy()
    if "pseudonym" in df.columns:
        uid_to_name = (
            has_pos[["user_address", "pseudonym"]]
            .drop_duplicates(subset="user_address")
            .set_index("user_address")["pseudonym"]
            .to_dict()
        )
    else:
        uid_to_name = {}

    users = [
        (uid, uid_to_name.get(uid, ""))
        for uid in has_pos["user_address"].unique()
    ]

    print(f"  Total comments   : {len(df):,}")
    print(f"  Relevant markets : {len(relevant_ids):,} unique market_ids")
    print(f"  Users w/ position: {len(users):,} unique user_ids")
    return users, relevant_ids, market_meta


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def run(workers=WORKERS):
    os.makedirs(BASE_DIR, exist_ok=True)

    users, relevant_ids, market_meta = load_inputs()
    progress = ProgressTracker()

    done_before, _ = progress.summary()
    todo = [(uid, uname) for uid, uname in users if not progress.is_done(uid)]
    total_users = len(users)
    print(f"\nProgress: {done_before}/{total_users} already done. "
          f"Remaining: {len(todo)}")

    if not todo:
        print("All users already processed. Output:", OUTPUT_CSV)
        return

    write_header = not os.path.exists(OUTPUT_CSV)
    csv_lock     = threading.Lock()
    csv_file     = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer       = csv.DictWriter(csv_file, fieldnames=OUTPUT_FIELDS)
    if write_header:
        writer.writeheader()

    total_done = done_before
    total_rows = 0

    start_str = datetime.fromtimestamp(START_MS / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    end_str   = datetime.fromtimestamp(END_MS   / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    print(f"\nDate window : {start_str} ~ {end_str}")
    print(f"Starting collection with {workers} workers …\n")

    def process_user(uid, uname):
        try:
            rows = fetch_user_positions(uid, uname, relevant_ids, market_meta)
        except RuntimeError as exc:
            progress.mark(uid, "failed")
            return uid, 0, str(exc)

        with csv_lock:
            for row in rows:
                writer.writerow(row)
            csv_file.flush()

        progress.mark(uid, "done", n_rows=len(rows))
        return uid, len(rows), None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_user, uid, uname): uid
                   for uid, uname in todo}

        for fut in as_completed(futures):
            uid, n_rows, err = fut.result()
            total_done += 1
            total_rows += n_rows

            status_str = f"ERROR: {err[:60]}" if err else f"{n_rows} rows"
            pct = total_done / total_users * 100
            print(f"  [{total_done:5d}/{total_users}  {pct:5.1f}%]  "
                  f"{uid[:12]}…  {status_str}")

    csv_file.close()

    done_after, failed_after = progress.summary()
    print(f"\n{'='*60}")
    print(f"  Done    : {done_after:,}")
    print(f"  Failed  : {failed_after:,}")
    print(f"  New rows: {total_rows:,}")
    print(f"  Output  : {OUTPUT_CSV}")
    print(f"{'='*60}")

    if failed_after > 0:
        print(f"\n{failed_after} users failed — re-run the script to retry them.")


if __name__ == "__main__":
    run()
