# Data

- `processed/cleaned_feature_engineered_bank_marketing.csv` is the reproducible, tracked analysis input (41,176 deduplicated records).
- `raw/` contains instructions only; original UCI ZIP files are intentionally excluded from Git.

Run `python3 scripts/prepare_data.py --refresh-data` to download the public source and regenerate the processed file and Notebook 01 handoff artifacts.
