# 📊 Campaign Prophet: Bank Marketing Campaign Analytics

## 🎯 Project Overview

Campaign Prophet is a multi-notebook marketing analytics project built using the UCI Bank Marketing dataset. The project analyzes financial marketing campaign performance, estimates campaign return on investment (ROI), validates key business assumptions statistically, and builds predictive models to identify customers most likely to subscribe to a term deposit product.

The project is designed as a portfolio-ready analytics workflow that connects business understanding, statistical validation, machine learning, and ROI-based campaign optimization.

---

## 💼 Business Problem

A financial institution conducts outbound marketing campaigns to promote term deposit subscriptions. Each customer contact incurs a cost, while successful subscriptions generate revenue.

This project answers key business questions such as:

- Which customer segments generate the highest marketing ROI?
- Which customer and campaign factors significantly influence conversion?
- Can customer subscription likelihood be predicted before contact?
- Which customers should be prioritized for outreach?
- How can marketing budgets be allocated more profitably?
- Which predictive model is most suitable for operational targeting?

---

## 🗂️ Dataset

**Dataset:** UCI Bank Marketing Dataset  
**Industry Context:** Portuguese retail banking marketing campaigns  
**Target Variable:** Whether the customer subscribed to a term deposit (`yes` / `no`)

### Dataset Summary

| Metric | Value |
|---|---:|
| Original records | 41,188 |
| Duplicate records removed | 12 |
| Cleaned records used | 41,176 |
| Subscribers | 4,639 |
| Non-subscribers | 36,537 |
| Baseline subscription rate | 11.27% |
| Original features | 21 |

### Key Variable Groups

- **Customer demographics:** age, job, marital status, education
- **Campaign information:** contact method, month, day of week, call duration, number of contacts
- **Previous campaign history:** previous contacts, previous outcome, days since prior contact
- **Macroeconomic indicators:** employment variation rate, consumer price index, consumer confidence index, Euribor rate, number of employees
- **Response variable:** subscription outcome

---

## 🧭 Project Structure

```text
Campaign_Prophet/
├── notebooks/
│   ├── 01_business_roi_statistical_validation.ipynb
│   └── 02_modeling_targeting_campaign_optimization.ipynb
│
├── outputs/
│   ├── notebook_01/
│   │   ├── cleaned_feature_engineered_bank_marketing.csv
│   │   ├── job_summary.csv
│   │   ├── expected_value_usd_base.csv
│   │   ├── expected_value_eur_base.csv
│   │   ├── usd_eur_ranking_comparison.csv
│   │   ├── job_subscription_summary.csv
│   │   ├── poutcome_subscription_summary.csv
│   │   ├── duration_summary.csv
│   │   ├── age_summary.csv
│   │   ├── roi_scenarios_usd.pkl
│   │   ├── breakeven_rates_usd.pkl
│   │   ├── roi_scenarios_eur.pkl
│   │   └── breakeven_rates_eur.pkl
│   │
│   └── notebook_02/
│
├── visuals/
├── reports/
├── data/
├── docs/
├── README.md
└── requirements.txt
```

---

## 📒 Notebook Workflow

### ✅ Notebook 01 — Business Understanding, ROI Analysis, and Statistical Validation

Notebook 01 establishes the analytical foundation of the project.

Completed work includes:

- Dataset loading and validation
- Duplicate removal and data quality checks
- Feature engineering
- Subscription outcome exploration
- ROI assumption development
- USD and EUR / CLV profitability frameworks
- Breakeven analysis
- Segment-level expected value analysis
- Statistical hypothesis testing
- Notebook 01 handoff artifact generation

Key outputs from Notebook 01 are saved in:

```text
outputs/notebook_01/
```

### 🚧 Notebook 02 — Predictive Modeling, Targeting, and Campaign Optimization

Notebook 02 builds on the validated outputs from Notebook 01 and focuses on predictive and prescriptive analytics.

Current and planned work includes:

- Full Model vs Pre-Contact Model strategy
- Machine learning preprocessing
- Logistic Regression baseline modeling
- Logistic Regression tuning
- Random Forest modeling
- XGBoost modeling
- Cross-model comparison
- Customer probability scoring
- ROI-based threshold optimization
- Budget-constrained targeting
- Campaign insight visualizations

### 🔜 Notebook 03 — SQL/PySpark, Executive Summary, and Packaging

Planned later-stage work includes:

- SQL reproduction of selected analyses
- PySpark / Spark SQL reproduction
- Executive summary and recommendations
- Final limitations and scalability discussion
- GitHub packaging and portfolio documentation

---

## 💰 ROI Framework

The dataset does not include actual campaign costs, revenue values, deposit amounts, or customer lifetime value. Therefore, ROI is estimated using scenario-based assumptions.

The project uses two ROI frameworks:

### 1. Illustrative USD Framework

| Scenario | Cost per Contact | Revenue per Success |
|---|---:|---:|
| Conservative | $15 | $300 |
| Base Case | $10 | $500 |
| Optimistic | $5 | $1,000 |

### 2. Data-Anchored EUR / CLV Framework

| Scenario | Cost per Contact | Estimated CLV per Success |
|---|---:|---:|
| Conservative | €12 | €150 |
| Base Case | €10 | €250 |
| Optimistic | €7 | €450 |

The EUR / CLV framework is used as a contextual robustness check because the original dataset represents Portuguese banking campaigns.

---

## 📈 Key Findings So Far

### Notebook 01 Findings

- Overall campaign subscription rate is **11.27%**.
- The dataset is structurally sound after removing **12 duplicate records**.
- Students and retired customers show the strongest subscription rates.
- Blue-collar and services customers show weaker conversion performance.
- Students, retired customers, and unemployed customers generate the highest ROI under the scenario framework.
- Blue-collar, services, and entrepreneur segments generate the lowest ROI.
- ROI segment rankings are stable across USD and EUR / CLV frameworks.
- Job category is statistically associated with subscription success.
- Previous campaign outcome is strongly associated with future subscription success.
- Successful subscriptions have significantly longer call durations.
- Successful subscribers are slightly older on average, though the practical age difference is modest.

### Notebook 02 Current Findings

- The project now uses two modeling strategies:
  - **Full Model:** includes all available predictors, including post-contact variables.
  - **Pre-Contact Model:** excludes variables unavailable before outreach.
- The Full Model includes **21 raw predictors** before encoding.
- The Pre-Contact Model includes **19 raw predictors** before encoding.
- After one-hot encoding:
  - Full Model: **54 predictors**
  - Pre-Contact Model: **52 predictors**
- The train/test split preserves the original class distribution:
  - Training rows: **32,940**
  - Testing rows: **8,236**
  - Subscriber rate: **11.27%**
- The current Logistic Regression baseline confirms that pre-contact variables contain meaningful predictive signal.

---

## 🤖 Modeling Strategy

Notebook 02 compares two modeling perspectives.

### Full Model

The Full Model includes all available variables, including post-contact variables such as call duration. This model helps measure the maximum predictive signal available after campaign interaction data is known.

### Pre-Contact Model

The Pre-Contact Model excludes variables unavailable before outreach, such as call duration and current campaign interaction count. This model is operationally useful because it can support customer prioritization before calls are made.

### Why This Distinction Matters

Call duration is highly predictive, but it is only known after the call begins. A model that depends on call duration may perform well analytically, but it cannot decide whom to contact before outreach. For real campaign targeting, the Pre-Contact Model is more deployable.

---

## 🧪 Statistical Testing Summary

| Test | Result | Business Interpretation |
|---|---|---|
| Job category vs subscription | Significant | Occupation-based segments differ in conversion performance. |
| Previous campaign outcome vs subscription | Significant | Prior successful outcomes strongly indicate future subscription likelihood. |
| Call duration vs subscription | Significant | Successful calls are much longer, but duration is post-contact. |
| Age vs subscription | Significant but modest | Age has signal, but should not be treated as a primary driver. |

---

## 🧠 Business Question Traceability

The project uses a master business question matrix to connect technical analysis with business decisions.

Examples of key questions addressed so far:

| ID | Question | Current Status |
|---|---|---|
| MQ1 | What is the overall subscription success rate? | Completed |
| MQ7 | Which job categories generate the highest ROI? | Completed |
| MQ9 | Are job category differences statistically significant? | Completed |
| MQ10 | Does previous campaign outcome affect subscription success? | Completed |
| MQ13 | Which variables are most important in predicting customer response? | In Progress |
| MQ14 | How accurately can subscription be predicted? | In Progress |
| MQ17 | Which customers should be prioritized before contact? | Planned |
| MQ23 | What is the best operational model for ROI optimization? | Planned |

---

## 🛠️ Technologies Used

- Python
- pandas
- NumPy
- SciPy
- scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Google Colab
- Google Drive

Planned later-stage additions:

- SQL
- PySpark
- Spark SQL / Hive-style queries
- GitHub packaging

---

## 📁 Output Artifacts

Notebook 01 generates reusable artifacts for Notebook 02, including:

- Cleaned and feature-engineered dataset
- Segment-level job summary
- USD expected value outputs
- EUR expected value outputs
- USD/EUR ROI ranking comparison
- Statistical summary tables
- ROI scenario dictionaries
- Breakeven threshold dictionaries

These outputs are stored in:

```text
outputs/notebook_01/
```

Notebook 02 will generate additional modeling and optimization artifacts under:

```text
outputs/notebook_02/
```

---

## 🚦 Current Project Status

| Component | Status |
|---|---|
| Notebook 01: Business, ROI, and Statistical Validation | ✅ Completed |
| Notebook 02: Predictive Modeling and Campaign Optimization | 🚧 In Progress |
| Notebook 03: SQL/PySpark, Executive Summary, and Packaging | 🔜 Planned |

---

## 🌟 Project Differentiators

- Connects business questions directly to analytical outputs.
- Includes both statistical testing and machine learning.
- Separates explanatory modeling from operationally deployable pre-contact modeling.
- Uses dual USD and EUR / CLV ROI frameworks.
- Converts marketing analytics into expected value and targeting decisions.
- Maintains reusable notebook handoff artifacts.
- Designed for portfolio presentation, GitHub documentation, and interview discussion.

---

## ⚠️ Limitations

- The dataset does not contain actual revenue, deposit amount, campaign cost, or customer lifetime value fields.
- ROI values are scenario-based estimates, not observed financial records.
- No true control group is available, so uplift modeling is not possible in the current dataset.
- Call duration is highly predictive but cannot be used for pre-contact targeting.
- Model performance should be interpreted as historical prediction, not guaranteed future campaign performance.
- Probability calibration and temporal validation may be considered in future improvements.

---

## 👩‍💻 Author

**Lakshmi Sindhu Pulugundla**  
Data Analyst, Effexoft Inc.

---

## 📌 Repository Status

This project is currently being organized for GitHub packaging. Notebook 01 is complete, and Notebook 02 is actively being executed.