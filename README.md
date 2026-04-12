# NBA Salary Prediction

Predicting NBA player salaries from per 100 possession stats using regression models — deployed as a live Streamlit app.

**Live app:** https://nba-salary-prediction-regression-model.streamlit.app/
---

## Project overview

- **Goal:** Predict a player's 2025-26 salary from their 2024-25 per 100 possession stats
- **Dataset:** Basketball Reference — 424 players with both 2024-25 stats and 2025-26 contracts
- **Best model:** Lasso Regression (alpha=0.01) — Test R²: 0.544, RMSE: $8.7M
- **Key finding:** Lasso outperformed Random Forest and XGBoost on this dataset

## Modeling logic

Teams sign players based on past performance. Using 2024-25 stats to predict 2025-26 salary mirrors how real NBA contracts work — making this a genuinely predictive model, not a descriptive one.

---

## Results

| Model | Train R² | Test R² | Test RMSE |
|---|---|---|---|
| Linear Regression | 0.487 | 0.539 | $8.7M |
| Ridge (alpha=10) | 0.486 | 0.540 | $8.8M |
| **Lasso (alpha=0.01)** | **0.484** | **0.544** | **$8.7M** |
| Random Forest (tuned) | 0.713 | 0.505 | $9.8M |
| XGBoost (tuned) | 0.838 | 0.512 | $9.8M |

Lasso Regression won. 424 training samples is too small for ensemble methods to build diverse generalizable trees. Lasso's automatic feature selection (zeroing out 6 features) produced a clean, interpretable model that outperformed both tree-based approaches.

---

## Key insights

- **Scoring and playmaking drive salary most** — PTS (SHAP: 0.31) and AST (0.24) are the top two predictors
- **Position carries a salary premium** — point guards earn more independent of stats
- **Availability is explicitly priced in** — games played (G) is the 4th strongest predictor
- **Efficiency is undervalued** — shooting % (eFG%, 3P%) barely correlates with salary (0.05-0.07)
- **Defense is underpaid** — BLK, STL, and DRtg rank near the bottom of feature importance
- **Paolo Banchero is the most underpaid player** — elite stats on a rookie scale contract ($14M actual vs $33M predicted)
- **Max contracts are systematically underpredicted** — reputation and market timing cannot be captured by stats alone

---

## Data sources

| Dataset | Source | Description |
|---|---|---|
| 2024-25 per 100 possession stats | Basketball Reference | 735 rows, 34 columns — all players this season |
| 2025-26 salary contracts | Basketball Reference | 524 rows — current season contracts |

Both datasets merged on Basketball Reference Player ID — avoiding name spelling mismatches. After removing traded player duplicates and players without contracts, 424 players remain.

---

## Project structure

```
nba-salary-prediction/
├── data/                   # Raw CSVs (not tracked — see setup below)
├── notebooks/              # One notebook per DS phase
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation.ipynb
├── outputs/                # Saved charts and figures
├── app.py                  # Streamlit app
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Data science process

| Phase | Notebook | Key decisions |
|---|---|---|
| Data understanding | 01 | Identified TotalCharges null pattern, salary survivor bias, merge strategy |
| EDA | 02 | QQ plots confirmed log transform needed, MP excluded for multicollinearity, 9 visualizations |
| Preprocessing | 03 | Min 10 games filter, dropped redundant features, log salary target, stratified split |
| Modeling | 04 | Simplest first philosophy — Lasso beat Random Forest and XGBoost |
| Evaluation | 05 | Residual analysis, actual vs predicted, over/underpaid chart, SHAP importance |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Drew-Zeimetz/nba-salary-prediction.git
cd nba-salary-prediction
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
source venv/Scripts/activate    # Windows (Git Bash)
source venv/bin/activate        # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get the data

Download both CSV files from Basketball Reference and place them in the `data/` folder:

- **Stats:** basketball-reference.com/leagues/NBA_2025_per_100_poss.html → Export CSV
- **Salaries:** basketball-reference.com/contracts/players.html → Export CSV

Rename files to:
- `24-25_season_stats_per_100_possesions.csv`
- `25-26_nba_salary_contracts.csv`

### 5. Run notebooks in order

Start with `01_data_understanding.ipynb` — each notebook saves output that the next notebook loads.

### 6. Run the Streamlit app locally

```bash
streamlit run app.py
```

---

## Modeling decisions

**Why log transform salary?**
QQ plots confirmed salary is right-skewed. Log transformation normalizes the distribution and significantly improves linear regression performance. Predictions are converted back to dollars using `exp()`.

**Why exclude minutes played (MP)?**
MP is a proxy variable — high-usage players earn more AND play more minutes, creating a circular relationship. Since we use per 100 possession stats, playing time is already normalized. MP adds no independent signal.

**Why did Lasso beat tree-based models?**
NBA salary has largely linear relationships with stats. 328 training samples is too small for ensemble methods — Random Forest and XGBoost overfit aggressively regardless of regularization. Lasso's L1 penalty also performed automatic feature selection, zeroing out 6 features with no independent salary signal.

**Why use per 100 possessions instead of per game?**
Per 100 possessions normalizes for pace and playing time — a player who played 40 games vs 70 games is on equal footing. Critical for a mid-season dataset where players have different games played totals.

---

## Model limitations

- Cannot capture contract timing or market dynamics — players signed at peak value appear overpaid
- Playoff performance and leadership are unquantified
- The $8.7M RMSE reflects irreducible error in a stat-only model
- Sample size of 410 players limits tree-based model performance

---

## What I would do differently

- Add years remaining on current contract as a feature
- Include playoff stats separately — regular season undersells some players
- Collect market size data — large market teams may pay premiums
- Build separate models by position — salary drivers differ between guards and big men
- Use a rolling multi-year average of stats rather than a single season

---

## Tools

Python · Pandas · Scikit-learn · XGBoost · SHAP · Matplotlib · Seaborn · SciPy · Streamlit · Git · GitHub

---
