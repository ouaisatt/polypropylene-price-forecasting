# 🏭 Polypropylene Price Forecasting

> Walk-Forward XGBoost model forecasting monthly Polypropylene futures 
> prices (CNY/ton) on the Dalian Commodity Exchange (DCE).

## 🚀 Live Demo
👉 **[polypropylene-price-forecasting.streamlit.app](https://polypropylene-price-forecasting.streamlit.app)**

---

## 📊 Results

| Metric | Value | Assessment |
|--------|-------|------------|
| R² | 0.8009 | explains 80% of price variation |
| MAE | 178.6 CNY/ton | average error per month |
| MAPE | 2.30% | percentage error rate |
| RMSE | 226.6 CNY/ton | penalized error metric |
| Coverage | 87.5% | months inside 95% confidence interval |

---

## 🗂️ Project Structure
```
├── app.py                          # Streamlit dashboard
├── pp_model.json                   # trained XGBoost model
├── feature_list.json               # list of 92 features
├── best_params.json                # optimized hyperparameters
├── merged_final.csv                # final dataset with all features
├── predictions_with_intervals.csv  # test predictions with CI
├── notebook.ipynb                  # full pipeline notebook
└── requirements.txt                # dependencies
```

---

## 📈 Dashboard Pages

| Page | Description |
|------|-------------|
| Dashboard | main forecast chart + error distribution |
| Historical Predictions | full table with confidence intervals |
| New Prediction | input current market values → get forecast |
| Feature Importance | which variables drive the model |
| Model Info | architecture + full improvement journey |

---

## 🔧 Pipeline
```
1. Data Collection
   └── Investing.com  → Polypropylene, LLDPE, BDI futures
   └── FRED API       → USD/CNY exchange rate
   └── yfinance       → Naphtha (BZ=F), Copper (HG=F)
   └── CPIAI          → US Inflation index

2. Feature Engineering
   └── 92 features from 9 raw variables
   └── Lag features (lag1, lag2, lag3)
   └── Rolling averages (roll3, roll6)
   └── First differences (diff)
   └── Propylene proxy (naphtha - oil spread)
   └── Time features (month, year)

3. Validation Strategy
   └── Walk-Forward validation
   └── 36-month rolling window
   └── Chronological train/test split (no data leakage)
   └── Train: Feb 2014 – Dec 2021 (90 months)
   └── Test:  Jan 2022 – Dec 2023 (24 months)

4. Hyperparameter Tuning
   └── Optuna Bayesian optimization
   └── 60 trials
   └── Best MAE: 178.6 CNY/ton

5. Confidence Intervals
   └── Quantile regression (2.5th – 97.5th percentile)
   └── Minimum width: 500 CNY/ton
   └── Coverage: 87.5% of actual prices inside band

6. Deployment
   └── Streamlit dashboard
   └── Hosted on Streamlit Cloud (free)
```

---

## 📦 Data Sources

| Source | Data | Access |
|--------|------|--------|
| Investing.com | Polypropylene futures | manual download |
| Investing.com | LLDPE futures | manual download |
| Investing.com | Baltic Dry Index | manual download |
| FRED | USD/CNY exchange rate | pandas-datareader |
| yfinance | Naphtha (BZ=F) | automatic |
| yfinance | Copper (HG=F) | automatic |
| CPIAI | US Inflation | manual download |

---

## 🏆 Improvement Journey

| Model | MAE | R² |
|-------|-----|-----|
| XGBoost baseline | 1,622 | -10.30 |
| Window-36 XGBoost | 331 | 0.395 |
| Window-36 Optimized | 259 | 0.545 |
| Window-36 + 4 Features | 231 | 0.618 |
| Window-36 + LLDPE + proxy | 178 | 0.800 |
| **Final + Confidence Intervals** | **178** | **0.801** |

---

## ⚙️ Top 10 Features

| Rank | Feature | Importance | Meaning |
|------|---------|------------|---------|
| 1 | usdcny_lag2 | 19.9% | USD/CNY rate 2 months ago |
| 2 | usdcny_lag1 | 9.9% | USD/CNY rate 1 month ago |
| 3 | usdcny_lag3 | 8.9% | USD/CNY rate 3 months ago |
| 4 | inflation_roll3 | 7.1% | 3-month inflation average |
| 5 | naphtha_lag3 | 3.8% | naphtha price 3 months ago |
| 6 | usdcny | 3.6% | current USD/CNY rate |
| 7 | naphtha_diff_lag3 | 3.5% | naphtha change 3 months ago |
| 8 | inflation_roll6 | 3.2% | 6-month inflation average |
| 9 | bdi_lag3 | 2.5% | Baltic Dry Index 3 months ago |
| 10 | lldpe | 2.5% | LLDPE futures price |

> USD/CNY dominates with 42% of total importance — polypropylene is 
> priced in CNY but produced using USD-priced oil. Currency effects 
> take 1–3 months to flow through to prices.

---

## 🚀 How to Run Locally
```bash
# clone the repository
git clone https://github.com/ouaisatt/polypropylene-price-forecasting.git
cd polypropylene-price-forecasting

# install dependencies
pip install -r requirements.txt

# run the app
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.13 |
| ML | XGBoost, Scikit-learn |
| Tuning | Optuna |
| Dashboard | Streamlit, Plotly |
| Data | Pandas, NumPy |
| Data sources | yfinance, pandas-datareader |
| Deployment | Streamlit Cloud |
| Version control | Git, GitHub |

---

## 👤 Author

**Ouais A.** — Machine Learning Engineer  
Master's degree · Time Series Forecasting · Classification  
🔗 [Upwork Profile](https://www.upwork.com/freelancers/~01b003c4fff0dcec92?viewMode=1)

---

## 📄 License

This project is open source and available under the MIT License.
