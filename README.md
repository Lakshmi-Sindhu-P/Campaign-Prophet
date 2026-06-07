<div align="center">

# 🔮 Campaign Prophet

### *Turning Bank Marketing Data Into Precision Targeting Intelligence*

<br>

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.1+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-189B4E?style=for-the-badge)
![PySpark](https://img.shields.io/badge/PySpark-3.3+-E25A1C?style=for-the-badge&logo=apache-spark&logoColor=white)
![Spark SQL](https://img.shields.io/badge/Spark_SQL-Active-4479A1?style=for-the-badge&logo=mysql&logoColor=white)
![Status](https://img.shields.io/badge/Status-In_Progress-FFB300?style=for-the-badge)

<br>

> **An end-to-end marketing analytics system that transforms raw campaign data into ranked, profit-optimized contact decisions — connecting ROI modeling, predictive machine learning, and enterprise-scale SQL/PySpark reproduction.**

<br>

[📊 Overview](#-overview) · [💰 ROI Framework](#-roi-framework) · [🤖 Models](#-model-performance) · [📓 Notebooks](#-notebook-workflow) · [🔑 Findings](#-key-findings) · [🛠️ Setup](#%EF%B8%8F-setup)

</div>

---

## 📊 Overview

A Portuguese retail bank ran thousands of outbound phone calls between **May 2008 and November 2010**, trying to sell customers term deposit subscriptions. Only **11.27% said yes**.

Campaign Prophet answers the questions that matter:

| Layer | Question |
|-------|---------|
| 📈 Descriptive | Which segments are most profitable and why? |
| 🔬 Inferential | Are the patterns real or random? |
| 🤖 Predictive | Who will subscribe *before* we call them? |
| 🎯 Prescriptive | Who do we call first with a fixed budget? |

### Dataset at a Glance

| Metric | Value |
|--------|-------|
| 📁 Source | UCI Bank Marketing Dataset (Moro et al., 2014) |
| 🧹 Records after deduplication | **41,176** (12 duplicates removed from 41,188) |
| 📐 Raw features | 21 |
| 🎯 Positive class rate | **11.27%** (4,639 subscribers / 36,537 non-subscribers) |
| 📅 Campaign period | May 2008 – November 2010 |
| 💶 Currency context | Euro — Portuguese retail banking |

<details>
<summary>📋 Variable groups</summary>

<br>

🧑 **Customer demographics** — age, job, marital, education, default, housing loan, personal loan

📞 **Campaign information** — contact method, month, day of week, call duration, contacts made

🕐 **Prior campaign history** — previous outcome `poutcome`, days since last contact `pdays`

📈 **Macroeconomic indicators** — `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`

> 💡 The five macroeconomic columns are **Banco de Portugal published statistics** embedded in the dataset by the original authors — not a separate data source.

</details>

---

## 💰 ROI Framework

The dataset contains **no actual costs or revenue figures**. Profitability is estimated using scenario-based assumptions across two independent frameworks — designed so conclusions can be validated regardless of which assumptions are used.

<details>
<summary>💵 Stage 1 — USD Reference Framework (Illustrative)</summary>

<br>

| Scenario | Cost / Contact | Revenue / Success | Breakeven Rate |
|----------|---------------:|------------------:|---------------:|
| 🔴 Conservative | \$15 | \$300 | 5.00% |
| 🟡 Base Case | \$10 | \$500 | 2.00% |
| 🟢 Optimistic | \$5 | \$1,000 | 0.50% |

All three cleared by the observed 11.27% conversion rate — the **campaign is profitable in aggregate without any targeting optimization**.

</details>

<details>
<summary>💶 Stage 2 — EUR Data-Anchored Framework (Empirical)</summary>

<br>

Revenue derived from a CLV model using Banco de Portugal published deposit rate statistics and Portuguese retail banking benchmarks for 2008–2010.

| Scenario | Cost / Contact | CLV / Success | Derivation Basis | Breakeven |
|----------|---------------:|--------------:|-----------------|----------:|
| 🔴 Conservative | €12 | €150 | €3,000 deposit × 2.0% margin × 1yr | 8.00% |
| 🟡 Base Case | €10 | €250 | €5,000 deposit × 2.5% margin × 1.5yr | 4.00% |
| 🟢 Optimistic | €7 | €450 | €8,000 deposit × 3.0% margin × 2yr | 1.56% |

</details>

> ✅ **Robustness result:** Segment rankings are **identical** across all 6 scenarios and both frameworks — zero position changes across 12 segments and two independent frameworks.

---

## 🤖 Model Performance

All three models are trained on **pre-contact variables only** — features available before the call is made, enabling proactive targeting.

| Model | ROC-AUC | Avg Precision | Precision | Recall | F1 Score |
|-------|--------:|--------------:|----------:|-------:|---------:|
| 🥇 🌲 **Random Forest** `Primary` | **0.8116** | **0.4811** | **0.4171** | 0.6369 | **0.5041** |
| 🥈 ⚡ XGBoost `Secondary` | 0.8096 | 0.4770 | 0.3864 | **0.6509** | 0.4849 |
| 🥉 📐 Logistic Regression `Baseline` | 0.8009 | 0.4438 | 0.3593 | 0.6455 | 0.4617 |

**Selected model:** 🌲 Random Forest Pre-Contact — leads on ROC-AUC, Average Precision, Precision, F1, and Accuracy

**Secondary model:** ⚡ XGBoost — preferred when maximising subscriber capture (highest Recall) is the priority

**Post-contact ceiling:** Full Model ROC-AUC = **0.9476** when call duration is included — quantifying the **0.136 predictive lift** from post-contact variables

<details>
<summary>🧠 Why two models (Pre-Contact vs Full)?</summary>

<br>

A model that requires call duration to predict subscription **cannot decide who to call** — it can only describe what happened after the call began.

| Model | Features | ROC-AUC | Operational Use |
|-------|---------|--------:|----------------|
| Full Model | All 21 (incl. duration) | 0.9476 | Benchmarking predictive ceiling |
| **Pre-Contact Model** | 19 (excl. duration, campaign) | **0.8116** | **Proactive targeting before calls** |

</details>

<details>
<summary>🏋️ How class imbalance is handled</summary>

<br>

With only 11.27% positive class rate, a naive model predicting "no" for everyone achieves **88.73% accuracy** and catches zero subscribers.

Campaign Prophet uses:
- Logistic Regression + Random Forest → `class_weight='balanced'`
- XGBoost → `scale_pos_weight=7.88` (ratio of negative to positive)
- Evaluation → ROC-AUC and Average Precision prioritised over accuracy

</details>

---

## 🔑 Key Findings

<details>
<summary>📈 ROI and profitability</summary>

<br>

- 🥇 **Students** generate the highest ROI at **1,471%** under USD Base Case (31.43% conversion)
- 🥈 **Retired customers** rank second at **1,163%** ROI (25.26% conversion)
- 📉 **Blue-collar** generates the lowest ROI at **245%** — still profitable, priority is lower
- 💡 All 12 segments are profitable → **prioritize, don't exclude**
- 💰 Total expected campaign profit under USD Base Case: **\$1,907,736**

</details>

<details>
<summary>🔬 Statistical validation</summary>

<br>

| Test | Statistic | p-value | Business Interpretation |
|------|----------:|--------:|------------------------|
| Job category vs subscription | χ² = 961.74, V = 0.15 | < 0.0001 | Significant — combine with other variables |
| Prior campaign outcome vs subscription | χ² = 4230.14, V = 0.32 | < 0.0001 | **Strongest signal** — prior success predicts future success |
| Call duration vs subscription | t = 55.50 | < 0.0001 | 553s vs 221s — explanatory, not pre-contact actionable |
| Age vs subscription | t = 4.78 | < 0.0001 | 40.91 vs 39.91 years — significant, modest practical effect |

</details>

<details>
<summary>🔑 Cross-model feature consistency</summary>

<br>

**6 features in top 10 of ALL THREE models:**

`emp.var.rate` · `euribor3m` · `cons.price.idx` · `pdays` · `contact_telephone` · `month_may`

**9 features consistent across at least two models** (adds): `nr.employed` · `cons.conf.idx` · `poutcome_success`

**Actionable signals:**
- 📉 Low `emp.var.rate` and favorable `euribor3m` → higher conversion likelihood
- 🗓️ **March** and **October** contacts outperform **May** and **November** significantly
- 📱 **Cellular** outreach outperforms **telephone** consistently across all models
- 🏆 `nr.employed` accounts for **37.1%** of XGBoost importance — the single strongest pre-contact signal

</details>

---

## 📓 Notebook Workflow

### ✅ Notebook 01 — Business · ROI · Statistical Validation

<details>
<summary>View phase breakdown</summary>

<br>

| Phase | Section | Key Output |
|-------|---------|-----------|
| ✅ 0 | Environment setup | Configured Colab workspace |
| ✅ 1.0 | Dataset preview | 41,176 clean records, data quality confirmed |
| ✅ 1.1–1.2 | Exploration + visualization | 11.27% baseline, segment distributions |
| ✅ 2.1 | Feature engineering | `success`, `age_group`, `duration_category`, `had_previous_contact` |
| ✅ 2.2.1–2.2.4 | ROI framework | Profitability rankings across 3 scenarios |
| ✅ 2.2.5 | Breakeven economics | Mathematically grounded targeting thresholds |
| ✅ 2.2.6 | Expected value framework | EV = p × R − C at segment level |
| ✅ 2.2.7 | Cross-framework robustness | Identical rankings across USD and EUR |
| ✅ 3.1–3.4 | Statistical hypothesis testing | Chi-square and t-test validation |

</details>

---

### 🚧 Notebook 02 — Predictive Modeling · Targeting

<details>
<summary>View phase breakdown</summary>

<br>

| Phase | Section | Status | Key Output |
|-------|---------|--------|-----------|
| 4.1 | Feature selection | ✅ | Full (21) vs Pre-Contact (19) strategy |
| 4.2 | Data preprocessing | ✅ | 54/52 encoded features, 80/20 stratified split |
| 4.3 | Logistic Regression | ✅ | Pre-Contact ROC-AUC 0.8009, AP 0.4438 |
| 4.3.2 | Hyperparameter tuning | ✅ | C=0.1, L1, balanced — marginal improvement confirmed |
| 4.3.3 | Coefficient analysis | ✅ | `emp.var.rate` dominant negative, `month_mar` dominant positive |
| 4.4 | Random Forest | ✅ | Pre-Contact ROC-AUC 0.8116, AP 0.4811 |
| 4.5 | XGBoost | ✅ | Pre-Contact ROC-AUC 0.8096, AP 0.4770 |
| 4.6 | Cross-model comparison | ✅ | Random Forest selected as primary operational model |
| 5 | ROI optimization + targeting | 🚧 | Customer scoring, threshold optimization, budget allocation |
| 6 | Visualizations | 📋 | Model performance, targeting strategy, campaign insights |

</details>

---

### 📋 Notebook 03 — SQL · PySpark · Executive Summary

<details>
<summary>View planned phases</summary>

<br>

| Phase | Section | Description |
|-------|---------|-------------|
| 📋 6.5 | SQL + PySpark analytics | Core analyses reproduced at enterprise scale using Spark SQL and PySpark DataFrame API |
| 📋 7 | Executive summary | Consolidated findings and strategic recommendations |
| 📋 8 | GitHub packaging | Portfolio documentation and repository finalization |

</details>

---

## 🗂️ Project Structure

```
🔮 campaign-prophet/
│
├── 📓 notebooks/
│   ├── 01_business_roi_statistical_validation.ipynb      ✅ Complete
│   ├── 02_modeling_targeting_campaign_optimization.ipynb 🚧 In Progress
│   └── 03_sql_pyspark_summary_packaging.ipynb            📋 Planned
│
├── 📊 outputs/
│   ├── notebook_01/
│   │   ├── job_summary.csv
│   │   ├── expected_value_usd_base.csv
│   │   ├── expected_value_eur_base.csv
│   │   ├── usd_eur_ranking_comparison.csv
│   │   ├── job_subscription_summary.csv
│   │   ├── poutcome_subscription_summary.csv
│   │   ├── duration_summary.csv
│   │   └── age_summary.csv
│   └── notebook_02/                                      📋 Pending Phase 5
│
├── 📁 data/
│   └── README.md                                         ← dataset not included, see setup
│
├── 🖼️ visuals/
├── 📄 reports/
├── 📚 docs/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🏆 What Makes This Different

| Differentiator | Description |
|----------------|-------------|
| 🔄 **Dual-currency robustness** | Rankings validated across USD illustrative and EUR empirically grounded frameworks independently — not assumption artifacts |
| 🎯 **Pre-contact model distinction** | Operational vs analytical modeling explicitly separated — demonstrates production deployment thinking |
| 📋 **Question traceability** | Every inference maps to a named business question with tracked completion across 23 master questions and 25 clarification questions |
| 💰 **EV framework integration** | ROI assumptions connect to model probabilities via EV = p × R − C at individual customer level |
| ⚡ **Enterprise-scale reproduction** | Core pipeline reproduced in PySpark and Spark SQL demonstrating scalability beyond notebook analytics |
| 📐 **Grounded targeting thresholds** | Breakeven rate (Cost/Revenue) replaces arbitrary 0.50 cutoff as the minimum profitable contact probability |

---

## 🛠️ Setup

```bash
pip install -r requirements.txt
```

**Dataset download:**

```
1. Visit  → https://archive.ics.uci.edu/dataset/222/bank+marketing
2. Download → bank+marketing.zip
3. Extract  → bank-additional-full.csv from the nested zip
4. Place in → data/raw/bank-additional-full.csv
5. Run      → Notebook 01 end-to-end to generate all processed outputs
```

All outputs required by Notebook 02 are generated by Notebook 01 and saved automatically to `outputs/notebook_01/`. No manual data wrangling required.

---

## ⚠️ Limitations

- ROI values are scenario-based estimates — no actual costs, deposit amounts, or CLV figures in the dataset
- No randomized control group — uplift modeling is not implementable on this data
- Call duration is highly predictive but unavailable pre-contact — excluded from the operational model
- Model performance reflects historical prediction on held-out test data, not guaranteed future results

---

## 📚 Citation

```bibtex
@article{moro2014data,
  title={A data-driven approach to predict the success of bank telemarketing},
  author={Moro, S{\'e}rgio and Cortez, Paulo and Rita, Paulo},
  journal={Decision Support Systems},
  volume={62},
  pages={22--31},
  year={2014},
  publisher={Elsevier}
}
```

---

<div align="center">

**Built by Lakshmi Sindhu Pulugundla · Data Analyst, Effexoft Inc.**

*Campaign Prophet is a portfolio project for educational and demonstration purposes.*

<br>

⭐ If this project was useful, consider giving it a star!

</div>
