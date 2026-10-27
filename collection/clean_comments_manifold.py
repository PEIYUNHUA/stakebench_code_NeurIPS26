"""
clean_comments_manifold.py
--------------------------
Market-level cleaning of the Manifold comment crawl: comments.csv -> comments_clean.csv.
Steps: back-calculate reply_count from parent_comment_id; rename columns to the Polymarket
schema (user_id -> user_address, user_name -> pseudonym, position_mana -> position_size,
likes -> reaction_count); drop markets whose question is empty; drop markets with fewer than
100 comments; reorder columns.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR   = Path(__file__).parent
INPUT_FILE  = DATA_DIR / "comments.csv"
OUTPUT_FILE = DATA_DIR / "comments_clean.csv"

df = pd.read_csv(INPUT_FILE, dtype={"reply_count": object})
print(f"input: {len(df):,} rows, {df['market_id'].nunique():,} markets")

reply_counts = (
    df["parent_comment_id"]
    .dropna()
    .value_counts()
    .rename("reply_count")
    .astype(int)
)
df = df.drop(columns=["reply_count"])
df = df.join(reply_counts, on="comment_id", how="left")
df["reply_count"] = df["reply_count"].fillna(0).astype(int)

# Manifold             → Polymarket reference
# user_id              → user_address
# user_name            → pseudonym
# position_mana        → position_size
# likes                → reaction_count
df = df.rename(columns={
    "user_id":       "user_address",
    "user_name":     "pseudonym",
    "position_mana": "position_size",
    "likes":         "reaction_count",
})

# market_start_time / market_end_time / condition_id /
# category_validated / parent_entity_type_used / parent_entity_id_used /
# proxy_wallet / reactions(raw JSON) / report_count

before = len(df)
df["question"] = df["question"].astype(str).str.strip()
empty_q_markets = df[df["question"].isin(["", "nan", "None"])]["market_id"].unique()
df = df[~df["market_id"].isin(empty_q_markets)]
print(f"after dropping markets with an empty question: {len(df):,} rows (removed {before - len(df):,})")

before = len(df)
valid_markets = df.groupby("market_id").size()
valid_markets = valid_markets[valid_markets >= 100].index
df = df[df["market_id"].isin(valid_markets)]
print(f"after dropping markets with <100 comments: {len(df):,} rows (removed {before - len(df):,}), {df['market_id'].nunique():,} markets")

col_order = [
    "market_id", "slug", "question", "outcome_type",
    "category", "resolution",
    "comment_id", "created_at", "text",
    "pseudonym", "user_address",
    "position_side", "position_size", "position_shares",
    "prob_at_comment",
    "reaction_count",
    "parent_comment_id", "reply_count",
]
col_order = [c for c in col_order if c in df.columns]
df = df[col_order]

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"written: {OUTPUT_FILE}")
print(f"final: {len(df):,} rows, {df['market_id'].nunique():,} markets, {df['user_address'].nunique():,} users")
