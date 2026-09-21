# Raw data

Raw UCI archives are intentionally **not committed** to this repository. To recreate them, run:

```bash
python3 scripts/prepare_data.py --refresh-data
```

The script downloads the public UCI archive into a temporary location, rebuilds the tracked processed dataset, and removes the temporary raw files when it finishes.
