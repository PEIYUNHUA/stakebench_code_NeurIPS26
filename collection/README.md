# Collection and cleaning scripts

The scripts that produced the StakeBench data, in run order. Their unmodified outputs are released in
the dataset's `raw/` folder (see `raw/README.md` there), and `data_processing.md` in the dataset documents
every parameter. All endpoints are public and need no authentication; `proxies.txt` (optional, one proxy
URL per line) enables proxy rotation in the comment crawlers.

| Step | Polymarket | Manifold | Output |
|---|---|---|---|
| 1. Candidate markets | `get_polymarket_topics.py` | `get_manifold_topics.py` | `valid_topics.jsonl` + `valid_topics_summary.csv` / `valid_markets.jsonl` + `valid_markets_summary.csv` |
| 2. Comments with position at comment time | `get_polymarket_comments.py` | `get_manifold_comments.py` | `comments.csv` |
| 3. Market-level cleaning | `clean_comments_polymarket.py` | `clean_comments_manifold.py` | `comments_clean.csv` |
| 4. Position / bet histories of positioned commenters | `get_polymarket_position.py` | `get_manifold_position.py` | `user_positions.csv` |
| 5. Market attributes and hourly prices | `get_polymarket_data.py` | `get_manifold_data.py` | `market_attrs.csv`, `prices/<market_id>.csv` |
| 6. Running position snapshots | `clean_positions_polymarket.py` | `clean_positions_manifold.py` | `positions_clean.csv` |

Paths. Steps 1–2 write to `polysent_data/` (Polymarket) and `manifold_data/` (Manifold) relative to the
working directory; the released dataset keeps the same files under `polymarket/` and `manifold/`
(`valid_*_summary.csv` renamed to `topics_summary.csv` / `markets_summary.csv`). The cleaning scripts read
and write in their own directory (`Path(__file__).parent`), so copy them next to the data they process.
Step 5 takes `STAKEBENCH_POLYMARKET_DIR` / `STAKEBENCH_MANIFOLD_DIR` (output folder, default
`data/<platform>_data`) and `STAKEBENCH_<PLATFORM>_COMMENTS` (comments CSV whose markets are processed);
step 2 reads the step-1 output from the default paths at the top of each file.

Notes.
- `clean_comments_polymarket.py` applied to the released raw file removes no rows (514,314 remain); the
  `comments_clean.csv` shipped since v1.0 has 514,282 rows and is GB18030-encoded. The evaluation code
  reads it as latin1. See `raw/README.md` in the dataset.
- The original scripts carried Chinese comments; they were removed for release and the module docstrings
  rewritten in English. Logic is unchanged.
- Requirements: `requests`, `pandas`, `numpy`.
