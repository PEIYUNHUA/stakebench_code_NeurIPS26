"""
clean_comments_polymarket.py
----------------------------
Market-level cleaning of the Polymarket comment crawl: comments.csv -> comments_clean.csv.
Steps: drop the always-empty `category` column; drop markets whose question is empty; drop
markets with fewer than 10 comments; reorder columns. See raw/README.md in the dataset for
the row counts of the released files.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR    = Path(__file__).parent
INPUT_FILE  = DATA_DIR / "comments.csv"
OUTPUT_FILE = DATA_DIR / "comments_clean.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"input: {len(df):,} rows, {df['market_id'].nunique():,} markets")

df = df.drop(columns=["category"])
print("dropped column 'category' (100% missing)")

before = len(df)
df["question"] = df["question"].astype(str).str.strip()
empty_q = df[df["question"].isin(["", "nan", "None"])]["market_id"].unique()
df = df[~df["market_id"].isin(empty_q)]
print(f"after dropping markets with an empty question: {len(df):,} rows (removed {before - len(df):,})")

before = len(df)
valid_markets = df.groupby("market_id").size()
valid_markets = valid_markets[valid_markets >= 10].index
df = df[df["market_id"].isin(valid_markets)]
print(f"after dropping markets with <10 comments: {len(df):,} rows (removed {before - len(df):,}), {df['market_id'].nunique():,} markets")

col_order = [
    "market_id", "slug", "question", "condition_id",
    "category_validated", "market_start_time", "market_end_time",
    "parent_entity_type_used", "parent_entity_id_used",
    "comment_id", "created_at", "text",
    "pseudonym", "user_address", "proxy_wallet",
    "position_side", "position_size", "position_shares",
    "reactions", "reaction_count", "report_count",
    "parent_comment_id", "reply_count",
]
col_order = [c for c in col_order if c in df.columns]
df = df[col_order]

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"written: {OUTPUT_FILE}")
print(f"final: {len(df):,} rows, {df['market_id'].nunique():,} markets, {df['user_address'].nunique():,} users")
print(f"category distribution:\n{df['category_validated'].value_counts().to_string()}")
