"""
clean_positions_polymarket.py
-----------------------------
Running position snapshots for Polymarket: user_positions.csv -> positions_clean.csv.
For every TRADE record (REDEEM / MERGE / SPLIT are excluded) in a market present in
comments_clean.csv, compute the user's position after the trade:

  position_side         - YES / NO / FLAT
  position_shares       - |YES shares - NO shares|
  position_size         - USD cost basis of the net position
  market_price          - nearest hourly price from prices/{market_id}.csv
  position_market_value - position_shares x market_price
  yes_cum_shares, no_cum_shares, potential_arbitrage - cumulative per-side holdings and a
                          flag for positions that hold both sides (set-arbitrage pattern)
"""

import os
import numpy as np
import pandas as pd

HERE          = os.path.dirname(os.path.abspath(__file__))
POSITIONS_CSV = os.path.join(HERE, "user_positions.csv")
COMMENTS_CSV  = os.path.join(HERE, "comments_clean.csv")
PRICES_DIR    = os.path.join(HERE, "prices")
OUTPUT_CSV    = os.path.join(HERE, "positions_clean.csv")

FLAT_THRESHOLD = 0.5
ARB_MIN_SHARES = 1.0


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def load_cid_to_mid():
    df = pd.read_csv(COMMENTS_CSV, encoding="latin1", low_memory=False,
                     usecols=["condition_id", "market_id"])
    df = df.dropna(subset=["condition_id", "market_id"]).drop_duplicates("condition_id")
    return dict(zip(df["condition_id"].astype(str), df["market_id"].astype(int)))


def load_all_prices(cid_to_mid):
    """See collection/README.md."""
    rows = []
    missing = 0
    for cid, mid in cid_to_mid.items():
        path = os.path.join(PRICES_DIR, f"{mid}.csv")
        if not os.path.exists(path):
            missing += 1
            continue
        df = pd.read_csv(path, usecols=["outcome", "ts", "price"])
        df["conditionId"] = cid
        df.rename(columns={"price": "market_price"}, inplace=True)
        rows.append(df[["conditionId", "outcome", "ts", "market_price"]])
    if missing:
        print(f"  [warn] {missing} markets have no price file")
    prices = pd.concat(rows, ignore_index=True)
    prices["outcome"] = prices["outcome"].str.capitalize()
    prices = prices.sort_values(["conditionId", "outcome", "ts"]).reset_index(drop=True)
    print(f"  Price rows loaded: {len(prices):,}  "
          f"({prices['conditionId'].nunique()} markets × 2 outcomes)")
    return prices


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def compute_snapshots(trades: pd.DataFrame) -> pd.DataFrame:
    """See collection/README.md."""
    trades = trades.copy()
    trades["outcome"] = trades["outcome"].str.strip().str.capitalize()

    # BUY=+1, SELL=-1
    sign = trades["side"].map({"BUY": 1.0, "SELL": -1.0}).fillna(0.0)

    is_yes = (trades["outcome"] == "Yes")
    is_no  = (trades["outcome"] == "No")

    trades["_dyes_sh"]   = np.where(is_yes, sign * trades["size"],    0.0)
    trades["_dno_sh"]    = np.where(is_no,  sign * trades["size"],    0.0)
    trades["_dyes_cost"] = np.where(is_yes, sign * trades["usdcSize"],0.0)
    trades["_dno_cost"]  = np.where(is_no,  sign * trades["usdcSize"],0.0)

    trades = trades.sort_values(
        ["proxy_wallet", "conditionId", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)

    grp = trades.groupby(["proxy_wallet", "conditionId"], sort=False)
    for col in ["_dyes_sh", "_dno_sh", "_dyes_cost", "_dno_cost"]:
        trades[col.replace("_d", "_cum_")] = grp[col].cumsum()

    yes_cum_sh = trades["_cum_yes_sh"]
    no_cum_sh  = trades["_cum_no_sh"]
    net_sh     = yes_cum_sh - no_cum_sh
    net_cost   = trades["_cum_yes_cost"] - trades["_cum_no_cost"]

    trades["position_side"] = np.where(
        net_sh >  FLAT_THRESHOLD, "YES",
        np.where(net_sh < -FLAT_THRESHOLD, "NO", "FLAT")
    )
    trades["position_shares"] = net_sh.abs()
    trades["position_size"]   = net_cost.abs()

    trades["yes_cum_shares"] = yes_cum_sh
    trades["no_cum_shares"]  = no_cum_sh

    flat_mask = trades["position_side"] == "FLAT"
    trades.loc[flat_mask, "position_shares"] = 0.0
    trades.loc[flat_mask, "position_size"]   = 0.0

    cond_a = (
        flat_mask &
        (yes_cum_sh >= ARB_MIN_SHARES) &
        (no_cum_sh  >= ARB_MIN_SHARES)
    )
    cond_b = (
        ~flat_mask &
        (yes_cum_sh >= ARB_MIN_SHARES) &
        (no_cum_sh  >= ARB_MIN_SHARES)
    )
    trades["potential_arbitrage"] = np.where(cond_a | cond_b, "Y", "N")

    trades["_price_outcome"] = np.where(
        trades["position_side"] == "YES", "Yes",
        np.where(trades["position_side"] == "NO", "No", None)
    )

    drop_cols = [c for c in trades.columns if c.startswith("_d") or c.startswith("_cum_")]
    trades = trades.drop(columns=drop_cols)
    return trades


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def attach_prices(trades: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """See collection/README.md."""
    trades = trades.copy()
    trades["timestamp"] = trades["timestamp"].astype(np.int64)
    prices = prices.copy()
    prices["ts"] = prices["ts"].astype(np.int64)

    result_chunks = []

    for outcome in ("Yes", "No"):
        mask = trades["_price_outcome"] == outcome
        if not mask.any():
            continue

        sub = trades[mask].copy()
        pr  = prices[prices["outcome"] == outcome][
            ["conditionId", "ts", "market_price"]
        ].copy()

        sub = sub.sort_values("timestamp")
        pr  = pr.sort_values("ts")

        chunks = []
        for cid, grp in sub.groupby("conditionId", sort=False):
            pr_cid = pr[pr["conditionId"] == cid].sort_values("ts")
            if pr_cid.empty:
                grp = grp.copy()
                grp["market_price"] = np.nan
                chunks.append(grp)
                continue
            merged = pd.merge_asof(
                grp.sort_values("timestamp"),
                pr_cid[["ts", "market_price"]],
                left_on="timestamp", right_on="ts",
                direction="nearest",
            )
            chunks.append(merged)

        result_chunks.append(pd.concat(chunks, ignore_index=True))

    flat_mask = trades["_price_outcome"].isna()
    if flat_mask.any():
        flat = trades[flat_mask].copy()
        flat["market_price"] = np.nan
        result_chunks.append(flat)

    result = pd.concat(result_chunks, ignore_index=True)

    result = result.sort_values(
        ["proxy_wallet", "conditionId", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)

    result["position_market_value"] = result["position_shares"] * result["market_price"]

    result = result.drop(columns=["_price_outcome", "ts"], errors="ignore")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Step 1 — Loading auxiliary data …")
    cid_to_mid = load_cid_to_mid()
    print(f"  {len(cid_to_mid):,} relevant conditionIds")

    prices = load_all_prices(cid_to_mid)

    print("\nStep 2 — Loading & filtering user_positions.csv …")
    pos = pd.read_csv(POSITIONS_CSV, low_memory=False)
    pos["conditionId"] = pos["conditionId"].astype(str)

    valid_cids = set(cid_to_mid.keys())
    pos = pos[pos["conditionId"].isin(valid_cids)]
    trades = pos[pos["type"] == "TRADE"].copy()
    n_excl = (pos["type"] != "TRADE").sum()
    print(f"  TRADE rows: {len(trades):,}  |  excluded (REDEEM/MERGE/SPLIT): {n_excl:,}")

    print("\nStep 3 — Computing rolling position snapshots …")
    trades = compute_snapshots(trades)

    print("\nStep 4 — Attaching market prices …")
    result = attach_prices(trades, prices)

    print(f"\nStep 5 — Saving → {OUTPUT_CSV}")
    result.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"  Output rows   : {len(result):,}")
    print(f"  Unique users  : {result['proxy_wallet'].nunique():,}")
    print(f"  Unique markets: {result['conditionId'].nunique():,}")

    side = result["position_side"].value_counts()
    print(f"\n  position_side distribution:")
    for s, c in side.items():
        print(f"    {s:5s}: {c:>8,}  ({c/len(result)*100:.1f}%)")

    nf = result[result["position_side"] != "FLAT"]["position_size"]
    print(f"\n  position_size USD (non-FLAT):")
    print(f"    median=${nf.median():.2f}  mean=${nf.mean():.2f}  "
          f"p75=${nf.quantile(.75):.2f}  max=${nf.max():.2f}")

    cov = result["market_price"].notna().sum()
    print(f"\n  market_price coverage: {cov:,}/{len(result):,} "
          f"({cov/len(result)*100:.1f}%)")

    arb = result["potential_arbitrage"].value_counts()
    arb_y = arb.get("Y", 0)
    print(f"\n  potential_arbitrage:")
    print(f"    Y: {arb_y:>8,}  ({arb_y/len(result)*100:.1f}%)")
    print(f"    N: {arb.get('N',0):>8,}  ({arb.get('N',0)/len(result)*100:.1f}%)")
    arb_users = result[result["potential_arbitrage"] == "Y"]["proxy_wallet"].nunique()
    arb_mkts  = result[result["potential_arbitrage"] == "Y"]["conditionId"].nunique()
    print(f"    unique arb users  : {arb_users:,}")
    print(f"    unique arb markets: {arb_mkts:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
