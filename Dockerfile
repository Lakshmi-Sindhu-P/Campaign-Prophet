FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Reproduce the full pipeline and verify every contract.
CMD ["bash", "-c", "python scripts/prepare_data.py && python scripts/run_sql.py && python scripts/run_modeling.py && python scripts/tune_sensitivity.py && python scripts/build_report.py && python scripts/experiment_power.py && python -m pytest -q"]
