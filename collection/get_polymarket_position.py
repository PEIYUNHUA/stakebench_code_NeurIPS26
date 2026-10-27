"""
get_polymarket_position.py
--------------------------
Collect the full activity history of every Polymarket commenter that holds a position in
comments_clean.csv (used to build the G3 action labels).

  1. Read comments_clean.csv: proxy wallets with a non-empty position_side, and the set of
     condition IDs of the comment markets.
  2. For each wallet, page data-api.polymarket.com/activity (time cursors for heavy traders).
  3. Keep records whose conditionId is in the comment markets.
  4. Write user_positions.csv with columns
     proxy_wallet, timestamp, conditionId, type, side, outcome, outcomeIndex,
     size, usdcSize, price, asset, transactionHash, title, slug, eventSlug

Resumable through positions_progress.json. Released as raw/polymarket/user_positions.csv.
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

BASE_DIR      = "data/polymarket_data"
INPUT_CSV     = os.path.join(BASE_DIR, "comments_clean.csv")
OUTPUT_CSV    = os.path.join(BASE_DIR, "user_positions.csv")
PROGRESS_FILE = os.path.join(BASE_DIR, "positions_progress.json")

DATA_API     = "https://data-api.polymarket.com"
PAGE_LIMIT   = 500
MAX_RETRIES  = 6
BASE_SLEEP   = 0.15
JITTER       = 0.10

import datetime as _dt
CURSOR_START_TS = int(_dt.datetime(2024,  1,  1, tzinfo=_dt.timezone.utc).timestamp())
CURSOR_END_TS   = int(_dt.datetime(2026,  1,  1, tzinfo=_dt.timezone.utc).timestamp())

WORKERS = 8

OUTPUT_FIELDS = [
    "proxy_wallet", "timestamp", "conditionId", "type",
    "side", "outcome", "outcomeIndex",
    "size", "usdcSize", "price",
    "transactionHash", "title", "slug", "eventSlug",
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

def _sleep(base=BASE_SLEEP, jitter=JITTER):
    time.sleep(max(0.0, base + random.uniform(-jitter, jitter)))


def get_json(url, params=None, timeout=30, max_retry=MAX_RETRIES):
    """See collection/README.md."""
    last_err = None
    for attempt in range(max_retry):
        headers = {"User-Agent": random.choice(_USER_AGENTS)}
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
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
        raise RuntimeError(f"HTTP {r.status_code} — {url} params={params} resp={last_err}")

    raise RuntimeError(f"Failed after {max_retry} retries: {url} last={last_err}")


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

    def is_done(self, wallet):
        return self._data.get(wallet, {}).get("status") == "done"

    def mark(self, wallet, status, n_rows=0):
        with self._lock:
            self._data[wallet] = {
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

def _make_row(rec, proxy_wallet):
    return {
        "proxy_wallet":    rec.get("proxyWallet", proxy_wallet),
        "timestamp":       rec.get("timestamp"),
        "conditionId":     rec.get("conditionId"),
        "type":            rec.get("type"),
        "side":            rec.get("side"),
        "outcome":         rec.get("outcome"),
        "outcomeIndex":    rec.get("outcomeIndex"),
        "size":            rec.get("size"),
        "usdcSize":        rec.get("usdcSize"),
        "price":           rec.get("price"),
        "transactionHash": rec.get("transactionHash"),
        "title":           rec.get("title"),
        "slug":            rec.get("slug"),
        "eventSlug":       rec.get("eventSlug"),
    }


def fetch_user_positions(proxy_wallet, relevant_cids):
    """See collection/README.md."""
    results     = []
    seen_hashes = set()
    end_ts      = CURSOR_END_TS
    consecutive_dup_pages = 0

    while end_ts > CURSOR_START_TS:
        params = {"user": proxy_wallet, "limit": PAGE_LIMIT, "end": end_ts}
        try:
            page = get_json(f"{DATA_API}/activity", params=params)
        except RuntimeError as exc:
            raise RuntimeError(f"[{proxy_wallet[:10]}…] fetch error: {exc}") from exc

        if not page or not isinstance(page, list):
            break

        new_records = 0
        oldest_ts   = end_ts
        for rec in page:
            tx = rec.get("transactionHash")
            ts = rec.get("timestamp", 0)
            oldest_ts = min(oldest_ts, ts)
            if tx in seen_hashes:
                continue
            seen_hashes.add(tx)
            new_records += 1
            if ts >= CURSOR_START_TS and rec.get("conditionId") in relevant_cids:
                results.append(_make_row(rec, proxy_wallet))

        _sleep()

        if new_records == 0:
            consecutive_dup_pages += 1
            if consecutive_dup_pages > 3:
                break
            end_ts = oldest_ts - 1
            continue

        consecutive_dup_pages = 0

        if oldest_ts <= CURSOR_START_TS:
            break

        end_ts = oldest_ts if len(page) == PAGE_LIMIT else oldest_ts - 1

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def load_inputs(csv_path=INPUT_CSV):
    """See collection/README.md."""
    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path, encoding="latin1", low_memory=False)

    relevant_cids = set(df["condition_id"].dropna().astype(str))

    has_pos = df[df["position_side"].notna() & df["proxy_wallet"].notna()]
    wallets = list(has_pos["proxy_wallet"].unique())

    print(f"  Total comments   : {len(df):,}")
    print(f"  Relevant topics  : {len(relevant_cids):,} unique conditionIds")
    print(f"  Users w/ position: {len(wallets):,} unique proxy_wallets")
    return wallets, relevant_cids


def run(workers=WORKERS):
    os.makedirs(BASE_DIR, exist_ok=True)

    wallets, relevant_cids = load_inputs()
    progress = ProgressTracker()

    done_before, _ = progress.summary()
    todo = [w for w in wallets if not progress.is_done(w)]
    print(f"\nProgress: {done_before}/{len(wallets)} already done. "
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

    total_done    = done_before
    total_rows    = 0
    total_wallets = len(wallets)

    print(f"\nStarting collection with {workers} workers …\n")

    def process_wallet(wallet):
        try:
            rows = fetch_user_positions(wallet, relevant_cids)
        except RuntimeError as exc:
            progress.mark(wallet, "failed")
            return wallet, 0, str(exc)

        with csv_lock:
            for row in rows:
                writer.writerow(row)
            csv_file.flush()

        progress.mark(wallet, "done", n_rows=len(rows))
        return wallet, len(rows), None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_wallet, w): w for w in todo}

        for fut in as_completed(futures):
            wallet, n_rows, err = fut.result()
            total_done += 1
            total_rows += n_rows

            if err:
                status_str = f"ERROR: {err[:60]}"
            else:
                status_str = f"{n_rows} rows"

            pct = total_done / total_wallets * 100
            print(f"  [{total_done:5d}/{total_wallets}  {pct:5.1f}%]  "
                  f"{wallet[:12]}…  {status_str}")

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
