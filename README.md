<div align="center">

<img src="visuals/banner.svg" alt="Campaign Prophet — Bank Marketing Campaign Analytics" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.1+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-189B=4E?style=flat-square)](https://xgboost.ai)
[![PySpark](https://img.shields.io/badge/PySpark-3.3+-E25A1C?style=flat-square&logo=apache-spark&logoColor=white)](https://spark.apache.org)
[![Status](https://img.shields.io/badge/Status-In_Progress-FFB300?style=flat-square)](https://github.com/Lakshmi-Sindhu-P/Campaign-Prophet)
[![License](https://img.shields.io/badge/License-MIT-1D9E75?style=flat-square)](LICENSE)

<br/>

> **An end-to-end marketing analytics system that transforms raw campaign data into ranked,
> profit-optimized contact decisions — connecting business understanding, statistical validation,
> predictive machine learning, and enterprise-scale SQL/PySpark reproduction.**

<br/>

[📊 Overview](#-overview) &nbsp;·&nbsp;
[💰 ROI Framework](#-roi-framework) &nbsp;·&nbsp;
[🤖 Models](#-model-performance) &nbsp;·&nbsp;
[🔑 Findings](#-key-findings) &nbsp;·&nbsp;
[🎯 Targeting Results](#-targeting-results) &nbsp;·&nbsp;
[📓 Notebooks](#-notebook-workflow) &nbsp;·&nbsp;
[📸 Visuals](#-visual-outputs) &nbsp;·&nbsp;
[🚀 Live Demo](#-live-demo) &nbsp;·&nbsp;
[🗺️ Roadmap](#%EF%B8%8F-roadmap) &nbsp;·&nbsp;
[🛠️ Setup](#%EF%B8%8F-setup) &nbsp;·&nbsp;
[📬 Connect](#-connect)

</div>

---

## 📊 Overview

A Portuguese retail bank ran thousands of outbound phone calls between **May 2008 and November 2010**, trying to sell customers term deposit subscriptions. Only **11.27%** said yes.

Campaign Prophet answers the questions that actually matter to a business:

<br/>

| Layer | Question Answered |
|:------|:-----------------|
| 📈 **Descriptive** | Which segments are profitable and by how much? |
| 🔬 **Inferential** | Are the patterns statistically real or random noise? |
| 🤖 **Predictive** | Who will subscribe *before* we make the call? |
| 🎯 **Prescriptive** | Who do we call first when the budget is constrained? |

<br/>

### Dataset at a Glance

| | |
|:---|:---|
| 📁 **Source** | UCI Bank Marketing Dataset — Moro, Cortez & Rita (2014) |
| 🧹 **Records** | **41,176** after removing 12 duplicates from 41,188 |
| 🎯 **Positive class** | **11.27%** — 4,639 subscribers out of 36,537 non-subscribers |
| 📐 **Features** | 21 raw predictors across demographics, campaign, and macroeconomics |
| 📅 **Period** | May 2008 – November 2010 |
| 💶 **Currency context** | Euro — Portuguese retail banking |

<details>
<summary>🔍 &nbsp;<b>Expand variable groups</b></summary>

<br/>

**🧑 Customer Demographics**
`age` `job` `marital` `education` `default` `housing` `loan`

**📞 Campaign Information**
`contact` `month` `day_of_week` `duration` `campaign` `pdays` `previous`

**🕐 Prior Campaign History**
`poutcome` — previous campaign outcome: success, failure, or nonexistent

**📈 Macroeconomic Indicators**
`emp.var.rate` `cons.price.idx` `cons.conf.idx` `euribor3m` `nr.employed`

> 💡 **Note:** The five macroeconomic columns are **Banco de Portugal published statistics**
> embedded in the dataset by the original paper authors — not a separate external data source.

</details>

---

## 💰 ROI Framework

The dataset contains **no actual costs or revenues**. Profitability is estimated using scenario-based assumptions across two independent frameworks. Running both is a deliberate robustness design — if conclusions hold under materially different assumptions, they are real.

<details>
<summary>💵 &nbsp;<b>Stage 1 — USD Reference Framework</b> &nbsp;<i>(Illustrative)</i></summary>

<br/>

| Scenario | Cost / Contact | Revenue / Success | Breakeven Rate |
|:---------|---------------:|------------------:|---------------:|
| 🔴 Conservative | \$15 | \$300 | **5.00%** |
| 🟡 Base Case | \$10 | \$500 | **2.00%** |
| 🟢 Optimistic | \$5 | \$1,000 | **0.50%** |

> All three scenarios cleared by the observed **11.27%** conversion rate.
> The campaign is profitable in aggregate without any targeting optimization applied.

</details>

<details>
<summary>💶 &nbsp;<b>Stage 2 — EUR Data-Anchored Framework</b> &nbsp;<i>(Empirical)</i></summary>

<br/>

Revenue derived from a simplified CLV model using Banco de Portugal published deposit
rate statistics and Portuguese retail banking benchmarks for 2008–2010.

| Scenario | Cost / Contact | CLV / Success | Derivation Basis | Breakeven |
|:---------|---------------:|--------------:|:----------------|----------:|
| 🔴 Conservative | €12 | €150 | €3,000 deposit × 2.0% margin × 1yr | **8.00%** |
| 🟡 Base Case | €10 | €250 | €5,000 deposit × 2.5% margin × 1.5yr | **4.00%** |
| 🟢 Optimistic | €7 | €450 | €8,000 deposit × 3.0% margin × 2yr | **1.56%** |

</details>

<br/>

> ✅ **Robustness Result** — Segment rankings are **identical** across all 6 scenarios and both
> frameworks. Zero position changes across 12 segments. Conclusions are not assumption artifacts.

---

## 🤖 Model Performance

All three models use **pre-contact variables only** — features available before the call is made,
enabling proactive customer targeting before any outreach begins.

<br/>

| Model | ROC-AUC | Avg Precision | Precision | Recall | F1 Score |
|:------|--------:|--------------:|----------:|-------:|---------:|
| 🥇 &nbsp;🌲 **Random Forest** &nbsp;`Primary` | **0.8116** | **0.4811** | **0.4171** | 0.6369 | **0.5041** |
| 🥈 &nbsp;⚡ XGBoost &nbsp;`Secondary` | 0.8096 | 0.4770 | 0.3864 | **0.6509** | 0.4849 |
| 🥉 &nbsp;📐 Logistic Regression &nbsp;`Baseline` | 0.8009 | 0.4438 | 0.3593 | 0.6455 | 0.4617 |

<br/>

**🌲 Random Forest Pre-Contact** is selected as the primary operational model — it leads on
every metric except Recall.

**⚡ XGBoost Pre-Contact** is retained as the secondary model for scenarios where maximising
subscriber capture takes priority over contact efficiency.

**Post-contact ceiling:** Including call duration lifts Full Model ROC-AUC to **0.9476** — a gap
of **0.136** that quantifies exactly how much predictive value the post-contact variable adds,
and why it cannot be used operationally.

<details>
<summary>🧠 &nbsp;<b>Why two models? Pre-Contact vs Full explained</b></summary>

<br/>

A model that requires call duration to predict subscription **cannot decide who to call**.
It can only describe what happened after the call began.

| Model | Features | ROC-AUC | Purpose |
|:------|:---------|--------:|:--------|
| Full Model | All 21 (incl. `duration`) | 0.9476 | Benchmarking the predictive ceiling |
| **Pre-Contact Model** | 19 (excl. `duration`, `campaign`) | **0.8116** | **Operational proactive targeting** |

</details>

<details>
<summary>🏋️ &nbsp;<b>How class imbalance is handled</b></summary>

<br/>

With only 11.27% positive class, a naive model predicting "no" for everyone achieves
**88.73% accuracy** and misses every single subscriber.

- **Logistic Regression + Random Forest** → `class_weight='balanced'`
- **XGBoost** → `scale_pos_weight=7.88` (negative-to-positive ratio)
- **Evaluation priority** → ROC-AUC and Average Precision, not accuracy

</details>

---

## 🔑 Key Findings

<details>
<summary>📈 &nbsp;<b>ROI and Profitability</b></summary>

<br/>

| Segment | USD Base Case ROI | Conversion Rate | Verdict |
|:--------|------------------:|----------------:|:--------|
| 🥇 Students | **1,471%** | 31.43% | Highest ROI |
| 🥈 Retired | 1,163% | 25.26% | Strong performer |
| 🥉 Unemployed | 610% | 14.20% | Solid third |
| 📉 Blue-collar | 245% | 6.90% | Lowest ROI — still profitable |

- All 12 job segments are profitable under Base Case assumptions
- Total expected campaign profit (USD Base): **\$1,907,736**
- Strategy: **prioritise by EV per contact — do not exclude entire segments**

</details>

<details>
<summary>🔬 &nbsp;<b>Statistical Validation</b></summary>

<br/>

| Test | Statistic | p-value | Business Interpretation |
|:-----|----------:|--------:|:------------------------|
| Job category vs subscription | χ² = 961.74, V = 0.15 | < 0.0001 | Significant — combine with other variables |
| Prior outcome vs subscription | χ² = 4230.14, V = 0.32 | < 0.0001 | **Strongest signal** — prior success predicts future conversion |
| Call duration vs subscription | t = 55.50 | < 0.0001 | 553s vs 221s — explanatory, not pre-contact actionable |
| Age vs subscription | t = 4.78 | < 0.0001 | 40.91 vs 39.91 years — significant, modest practical effect |

</details>

<details>
<summary>🔑 &nbsp;<b>Cross-Model Feature Consistency</b></summary>

<br/>

**6 features in the top 10 of ALL THREE models:**

| Feature | Type | Signal |
|:--------|:-----|:-------|
| `emp.var.rate` | Macroeconomic | ↓ Lower → higher conversion |
| `euribor3m` | Macroeconomic | 21.0% of RF importance |
| `cons.price.idx` | Macroeconomic | Consistent positive signal |
| `pdays` | Campaign history | Prior contact recency matters |
| `contact_telephone` | Contact method | Telephone underperforms cellular |
| `month_may` | Campaign timing | Weak conversion month |

**9 features consistent across at least two models** — adds `nr.employed`, `cons.conf.idx`, `poutcome_success`

**Key actionable signals:**
- `nr.employed` drives **37.1%** of XGBoost importance — single strongest pre-contact signal
- **March** and **October** outperform **May** and **November** significantly
- Cellular outreach consistently outperforms telephone across all three models

</details>

---

## 🎯 Targeting Results

> 🚧 **This section will be populated after Phase 5 completes.**

Phase 5 converts the Random Forest probability scores into concrete business decisions.
The following outputs are planned:

| Metric | Description | Status |
|:-------|:------------|:------:|
| Optimal threshold | Probability cutoff that maximises campaign profit | 📋 Pending |
| Recommended contacts | Number of customers to call from 41,176 | 📋 Pending |
| Expected subscribers | Predicted conversions at optimal threshold | 📋 Pending |
| Profit improvement | Gain over default 0.50 threshold | 📋 Pending |
| Cumulative gains | Top X% customers → Y% of all subscribers | 📋 Pending |
| Budget allocation | Ranked contact list under fixed budget scenarios | 📋 Pending |
| Monte Carlo CI | 95% confidence interval on expected campaign profit | 📋 Pending |

*Results will be added here once Phase 5 threshold optimization and Monte Carlo simulation are complete.*

---

## 📓 Notebook Workflow

<details>
<summary>✅ &nbsp;<b>Notebook 01 — Business Understanding · ROI Analysis · Statistical Validation</b> &nbsp;<code>Complete</code></summary>

<br/>

| Phase | Section | Key Output |
|:------|:--------|:-----------|
| ✅ Phase 0 | Environment setup | Configured Colab workspace |
| ✅ Phase 1.0 | Dataset preview | 41,176 records, 12 duplicates removed, no missing values |
| ✅ Phase 1.1–1.2 | Exploration + visualization | 11.27% baseline conversion, segment distributions |
| ✅ Phase 2.1 | Feature engineering | `success` `age_group` `duration_category` `had_previous_contact` |
| ✅ Phase 2.2.1–2.2.4 | ROI scenarios | Profitability and rankings across 3 scenarios |
| ✅ Phase 2.2.5 | Breakeven economics | Mathematically grounded targeting thresholds per scenario |
| ✅ Phase 2.2.6 | Expected value framework | EV = p × R − C applied at segment level |
| ✅ Phase 2.2.7 | Cross-framework robustness | Identical rankings across USD and EUR frameworks |
| ✅ Phase 3.1 | Chi-square: job vs subscription | χ² = 961.74, p < 0.0001 |
| ✅ Phase 3.2 | Chi-square: prior outcome vs subscription | χ² = 4230.14, p < 0.0001 |
| ✅ Phase 3.3 | T-test: call duration | 553s vs 221s — explanatory, not operational |
| ✅ Phase 3.4 | T-test: age | 40.91 vs 39.91 years — significant, modest practical effect |

</details>

<details>
<summary>🚧 &nbsp;<b>Notebook 02 — Predictive Modeling · Targeting · Campaign Optimization</b> &nbsp;<code>Phase 4 Complete · Phase 5 In Progress</code></summary>

<br/>

| Phase | Section | Status | Key Output |
|:------|:--------|:------:|:-----------|
| 4.1 | Feature selection | ✅ | Full (21 feat) vs Pre-Contact (19 feat) |
| 4.2 | Data preprocessing | ✅ | 54/52 encoded features · 32,940 train · 8,236 test |
| 4.3 | Logistic Regression | ✅ | Pre-Contact ROC-AUC 0.8009 · AP 0.4438 |
| 4.3.2 | Hyperparameter tuning | ✅ | C=0.1, L1, balanced — near-optimal baseline confirmed |
| 4.3.3 | Coefficient analysis | ✅ | `emp.var.rate` dominant negative · `month_mar` dominant positive |
| 4.4 | Random Forest | ✅ | Pre-Contact ROC-AUC 0.8116 · AP 0.4811 |
| 4.5 | XGBoost | ✅ | Pre-Contact ROC-AUC 0.8096 · AP 0.4770 |
| 4.6 | Cross-model comparison | ✅ | Random Forest selected as primary operational model |
| 5.1–5.4 | Customer scoring · EV framework | 🚧 | Individual-level probability scoring and EV ranking |
| 5.5 | Monte Carlo simulation | 🚧 | Uncertainty quantification with antithetic variates |
| 5.6 | Threshold optimization | 🚧 | Profit-maximising threshold sweep 0.05 → 0.95 |
| 5.7 | Budget-constrained optimization | 🚧 | Ranked contact list under fixed budget scenarios |
| 5.8 | Cumulative gains and lift | 🚧 | Contact percentage → subscriber capture trade-off |
| 6 | Visualizations | 📋 | Model performance · targeting strategy · campaign insights |

</details>

<details>
<summary>📋 &nbsp;<b>Notebook 03 — SQL · PySpark · Executive Summary · GitHub Packaging</b> &nbsp;<code>Planned</code></summary>

<br/>

| Phase | Section | Description |
|:------|:--------|:------------|
| 📋 6.5.1–6.5.3 | Spark SQL analytics | ROI and segmentation reproduced as SQL queries |
| 📋 6.5.4–6.5.6 | PySpark DataFrame API | Customer scoring pipeline at enterprise scale |
| 📋 6.5.7–6.5.8 | Hive-style queries | Threshold and budget analysis in distributed SQL |
| 📋 Phase 7 | Executive summary | Consolidated findings and strategic recommendations |
| 📋 Phase 8 | GitHub packaging | Portfolio documentation and repository finalization |

</details>

---

## 📸 Visual Outputs

> 🚧 **Charts and visualizations will be added here after Phase 6 completes.**

The following visuals are planned for this section:

| Visual | Description | Status |
|:-------|:------------|:------:|
| Precision-Recall curves | All three Pre-Contact models compared | 📋 Pending export |
| ROI by job segment | USD and EUR frameworks side by side | 📋 Pending export |
| Profit vs threshold curve | Phase 5 threshold optimization output | 📋 Pending Phase 5 |
| Cumulative gains chart | Contact percentage vs subscriber capture | 📋 Pending Phase 5 |
| Monte Carlo distribution | Expected profit with confidence intervals | 📋 Pending Phase 5 |
| Cross-model feature importance | Top predictors across all three models | 📋 Pending export |

*All charts are generated in Matplotlib/Seaborn and will be exported to `visuals/` and embedded here.*

---

## 🚀 Live Demo

> 🚧 **A Streamlit dashboard is planned after Phase 5 and Phase 6 complete.**

The planned dashboard will include four interactive sections:

| Section | Description |
|:--------|:------------|
| 📊 Campaign Overview | Subscription rates, segment distributions, baseline performance |
| 🎯 Customer Scorer | Input customer attributes → predicted subscription probability |
| 📈 Threshold Optimizer | Slider: move threshold → see contacts, expected profit, ROI update live |
| 💰 Budget Allocator | Input budget → ranked contact list with expected ROI |

**Deployment target:** [Streamlit Community Cloud](https://streamlit.io/cloud) — free, permanent, shareable URL

*Demo link will appear here once deployed.*

---

## 🗺️ Roadmap

- [x] Notebook 01 — Business understanding, ROI framework, statistical validation
- [x] Notebook 02 Phase 4 — Logistic Regression, Random Forest, XGBoost, model selection
- [ ] Notebook 02 Phase 5 — Customer scoring, Monte Carlo, threshold optimization, budget allocation
- [ ] Notebook 02 Phase 6 — Visualizations and campaign insight charts
- [ ] Notebook 03 — SQL, PySpark, Spark SQL, executive summary
- [ ] Export visual outputs to `visuals/` and embed in README
- [ ] Build and deploy Streamlit dashboard
- [ ] Add targeting results to README after Phase 5
- [ ] Final GitHub packaging and portfolio documentation

---

## 🏆 What Makes This Different

| Differentiator | Why It Matters |
|:--------------|:--------------|
| 🔄 **Dual-currency robustness** | USD illustrative + EUR empirically grounded frameworks run independently. Identical rankings across both prove conclusions are not assumption artifacts. |
| 🎯 **Pre-contact model distinction** | Operational vs analytical modeling explicitly separated. Only variables available before the call is made are used for targeting — demonstrates production deployment thinking. |
| 📋 **Business question traceability** | Every inference maps to a named business question with tracked completion status across 23 master questions and 25 clarification questions. |
| 💰 **EV framework integration** | ROI assumptions from Phase 2 connect directly to model probabilities in Phase 5 through EV = p × R − C at the individual customer level — not just segment averages. |
| 📐 **Mathematically grounded thresholds** | The breakeven rate (Cost / Revenue) replaces the arbitrary 0.50 cutoff. Any customer whose predicted probability exceeds breakeven generates positive expected profit. |
| ⚡ **Enterprise-scale reproduction** | Core analytical pipeline reproduced in PySpark and Spark SQL in Notebook 03 — demonstrating how the system scales from 41,176 rows to millions. |

---

## 🗂️ Project Structure

```
🔮 Campaign-Prophet/
│
├── 📓 notebooks/
│   ├── 01_business_roi_statistical_validation.ipynb       ✅ Complete
│   ├── 02_modeling_targeting_campaign_optimization.ipynb  🚧 In Progress
│   └── 03_sql_pyspark_summary_packaging.ipynb             📋 Planned
│
├── 📊 outputs/
│   ├── notebook_01/          ← 8 CSV artifacts from Notebook 01
│   └── notebook_02/          ← Populated after Phase 5 completes
│
├── 📁 data/
│   ├── processed/            ← cleaned_feature_engineered_bank_marketing.csv
│   ├── raw/                  ← dataset download instructions
│   └── README.md             ← dataset sourcing guide
│
├── 🖼️  visuals/               ← chart exports (Phase 6)
├── 📄 reports/               ← executive summaries (Phase 7)
├── 📚 docs/                  ← data dictionary and project notes
├── README.md
├── requirements.txt
└── LICENSE
```

---

## ⚠️ Limitations

- ROI values are scenario-based estimates — the dataset contains no actual campaign costs, deposit amounts, or CLV figures
- No randomized control group — uplift modeling is not implementable on this data
- Call duration is highly predictive but unavailable pre-contact — correctly excluded from the operational model
- Model performance reflects historical prediction on a held-out test set, not guaranteed future campaign results
- Probability calibration and temporal validation are identified future improvements

---

## 🛠️ Setup

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Dataset — not included in repository:**

```
1. Visit    → https://archive.ics.uci.edu/dataset/222/bank+marketing
2. Download → bank+marketing.zip
3. Extract  → bank-additional-full.csv from the nested zip
4. Place in → data/raw/
5. Run      → Notebook 01 end-to-end to generate all processed outputs
```

All artifacts required by Notebook 02 are generated automatically by Notebook 01 and
saved to `outputs/notebook_01/`. No manual data wrangling required after the initial
dataset download.

**Running in Google Colab:** Each notebook is built for Google Colab with Google Drive
integration. Notebook 01 saves all artifacts to Drive. Notebook 02 loads them automatically.

---

## 📚 Citation

```bibtex
@article{moro2014data,
  title     = {A data-driven approach to predict the success of bank telemarketing},
  author    = {Moro, S{\'e}rgio and Cortez, Paulo and Rita, Paulo},
  journal   = {Decision Support Systems},
  volume    = {62},
  pages     = {22--31},
  year      = {2014},
  publisher = {Elsevier}
}
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

The MIT License means anyone can use, copy, modify, and distribute this code freely,
provided the original author attribution is retained.

The UCI Bank Marketing Dataset is used under its original terms.
See the [dataset page](https://archive.ics.uci.edu/dataset/222/bank+marketing) for citation requirements.

---

## 📬 Connect

Have questions about the methodology, want to discuss the project, or interested in collaboration?

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Lakshmi_Sindhu_Pulugundla-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lakshmi-sindhu-p/)
[![Email](https://img.shields.io/badge/Email-sindhu.pl@outlook.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:sindhu.pl@outlook.com)

---

<div align="center">

**Built by Lakshmi Sindhu Pulugundla &nbsp;·&nbsp; Data Analyst, Effexoft Inc.**

*Campaign Prophet is a portfolio project for educational and professional demonstration purposes.*

<br/>

⭐ &nbsp;If this project was useful, give it a star!

</div>
