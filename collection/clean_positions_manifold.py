"""
clean_positions_manifold.py
---------------------------
Running position snapshots for Manifold: user_positions.csv -> positions_clean.csv.
For every valid bet (is_redemption == False and is_cancelled == False) in a market present in
comments_clean.csv, compute the user's position after the bet:

  position_side         - YES / NO / FLAT (net shares within FLAT_THRESHOLD count as FLAT)
  position_shares       - net shares on the dominant side
  position_size         - Mana cost basis of the net position
  market_price          - prob_after of the bet
  position_market_value - position_shares x market_price
  yes_cum_shares, no_cum_shares, potential_arbitrage - cumulative per-side holdings and a
                          flag for FLAT or hedged positions that hold both sides
"""

import os
import numpy as np
import pandas as pd

HERE          = os.path.dirname(os.path.abspath(__file__))
POSITIONS_CSV = os.path.join(HERE, "user_positions.csv")
COMMENTS_CSV  = os.path.join(HERE, "comments_clean.csv")
OUTPUT_CSV    = os.path.join(HERE, "positions_clean.csv")

FLAT_THRESHOLD = 0.5
ARB_MIN_SHARES = 1.0


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def load_relevant_markets():
    """See collection/README.md."""
    df = pd.read_csv(COMMENTS_CSV, encoding="utf-8-sig", low_memory=False,
                     usecols=["market_id"])
    df = df.dropna(subset=["market_id"])
    markets = set(df["market_id"].astype(str))
    return markets


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def compute_snapshots(bets: pd.DataFrame) -> pd.DataFrame:
    """See collection/README.md."""
    bets = bets.copy()
    bets["outcome"] = bets["outcome"].str.strip().str.capitalize()

    is_yes = (bets["outcome"] == "Yes")
    is_no  = (bets["outcome"] == "No")

    bets["_dyes_sh"]   = np.where(is_yes, bets["shares"], 0.0)
    bets["_dno_sh"]    = np.where(is_no,  bets["shares"], 0.0)
    bets["_dyes_cost"] = np.where(is_yes, bets["amount"], 0.0)
    bets["_dno_cost"]  = np.where(is_no,  bets["amount"], 0.0)

    bets = bets.sort_values(
        ["user_id", "contract_id", "created_at"], kind="mergesort"
    ).reset_index(drop=True)

    grp = bets.groupby(["user_id", "contract_id"], sort=False)
    for col in ["_dyes_sh", "_dno_sh", "_dyes_cost", "_dno_cost"]:
        bets[col.replace("_d", "_cum_")] = grp[col].cumsum()

    yes_cum_sh = bets["_cum_yes_sh"]
    no_cum_sh  = bets["_cum_no_sh"]
    net_sh     = yes_cum_sh - no_cum_sh
    net_cost   = bets["_cum_yes_cost"] - bets["_cum_no_cost"]

    bets["position_side"] = np.where(
        net_sh >  FLAT_THRESHOLD, "YES",
        np.where(net_sh < -FLAT_THRESHOLD, "NO", "FLAT")
    )
    bets["position_shares"] = net_sh.abs()
    bets["position_size"]   = net_cost.abs()

    bets["yes_cum_shares"] = yes_cum_sh
    bets["no_cum_shares"]  = no_cum_sh

    flat_mask = bets["position_side"] == "FLAT"
    bets.loc[flat_mask, "position_shares"] = 0.0
    bets.loc[flat_mask, "position_size"]   = 0.0

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
    bets["potential_arbitrage"] = np.where(cond_a | cond_b, "Y", "N")

    bets["market_price"] = np.where(
        bets["position_side"] == "YES", bets["prob_after"],
        np.where(bets["position_side"] == "NO", 1.0 - bets["prob_after"], np.nan)
    )
    bets["position_market_value"] = bets["position_shares"] * bets["market_price"]

    drop_cols = [c for c in bets.columns if c.startswith("_d") or c.startswith("_cum_")]
    bets = bets.drop(columns=drop_cols)
    return bets


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Step 1 — Loading relevant markets from comments_clean.csv …")
    relevant_markets = load_relevant_markets()
    print(f"  {len(relevant_markets):,} relevant contract_ids")

    print("\nStep 2 — Loading & filtering user_positions.csv …")
    pos = pd.read_csv(POSITIONS_CSV, low_memory=False)
    pos["contract_id"] = pos["contract_id"].astype(str)

    pos = pos[pos["contract_id"].isin(relevant_markets)]

    n_total      = len(pos)
    n_redemption = pos["is_redemption"].sum()
    n_cancelled  = (~pos["is_redemption"]) & pos["is_cancelled"]
    n_cancelled  = n_cancelled.sum()
    print(f"  After market filter  : {n_total:,} rows")
    print(f"  is_redemption = True : {n_redemption:,}")
    print(f"  is_cancelled  = True : {n_cancelled:,}  (excluding redemptions)")

    bets = pos[
        (pos["is_redemption"] == False) &
        (pos["is_cancelled"]  == False)
    ].copy()
    n_excl = n_total - len(bets)
    print(f"  Valid bets kept      : {len(bets):,}  (excluded: {n_excl:,})")

    n_buy  = (bets["shares"] > 0).sum()
    n_sell = (bets["shares"] < 0).sum()
    print(f"  BUY (shares>0): {n_buy:,}  |  SELL (shares<0): {n_sell:,}")

    print("\nStep 3 — Computing rolling position snapshots …")
    result = compute_snapshots(bets)

    print(f"\nStep 4 — Saving → {OUTPUT_CSV}")
    result.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"  Output rows   : {len(result):,}")
    print(f"  Unique users  : {result['user_id'].nunique():,}")
    print(f"  Unique markets: {result['contract_id'].nunique():,}")

    side = result["position_side"].value_counts()
    print(f"\n  position_side distribution:")
    for s, c in side.items():
        print(f"    {s:5s}: {c:>8,}  ({c/len(result)*100:.1f}%)")

    nf = result[result["position_side"] != "FLAT"]["position_size"]
    print(f"\n  position_size Mana (non-FLAT):")
    print(f"    median={nf.median():.1f}  mean={nf.mean():.1f}  "
          f"p75={nf.quantile(.75):.1f}  max={nf.max():.1f}")

    cov = result["market_price"].notna().sum()
    print(f"\n  market_price coverage: {cov:,}/{len(result):,} "
          f"({cov/len(result)*100:.1f}%)")

    arb_y = (result["potential_arbitrage"] == "Y").sum()
    print(f"\n  potential_arbitrage:")
    print(f"    Y: {arb_y:>8,}  ({arb_y/len(result)*100:.1f}%)")
    print(f"    N: {len(result)-arb_y:>8,}  ({(len(result)-arb_y)/len(result)*100:.1f}%)")
    arb_users = result[result["potential_arbitrage"] == "Y"]["user_id"].nunique()
    arb_mkts  = result[result["potential_arbitrage"] == "Y"]["contract_id"].nunique()
    print(f"    unique arb users  : {arb_users:,}")
    print(f"    unique arb markets: {arb_mkts:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
