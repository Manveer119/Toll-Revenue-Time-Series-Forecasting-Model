# Toll-Revenue-Time-Series-Forecasting-Model

A financial analytics project simulating the type of forecasting work done at transportation authorities like the New Jersey Turnpike Authority (NJTA). Built to demonstrate skills in Python-based data engineering, time-series modeling, and business reporting.

---

##  Business Problem

Transportation authorities collect monthly toll revenue data and need to:
- **Forecast future revenue** to support budget planning
- **Detect seasonal patterns** to allocate resources efficiently
- **Quantify uncertainty** so stakeholders can plan for best/worst case scenarios

---

##  Tools & Technologies

| Area | Tools Used |
|---|---|
| Language | Python 3 |
| Data Manipulation | pandas, numpy |
| Forecasting Model | statsmodels (Holt-Winters Exponential Smoothing) |
| Model Evaluation | scikit-learn (MAE, RMSE, MAPE) |
| Visualization | matplotlib |
| Reporting | openpyxl (Excel export) |

---

##  What the Model Does

1. **Generates** 5 years of realistic monthly toll revenue data including trend, seasonality, and a COVID-era disruption period
2. **Splits** data into training (48 months) and test (12 months) sets
3. **Trains** a Holt-Winters Exponential Smoothing model — a standard method for business forecasting that handles both trend and seasonality
4. **Evaluates** model accuracy on the held-out test set
5. **Forecasts** 12 months into the future with 95% confidence intervals
6. **Exports** results to a multi-sheet Excel report and a dashboard PNG

---

##  Model Results

| Metric | Value | Interpretation |
|---|---|---|
| MAPE | 3.28% | **Excellent** — industry standard is < 10% |
| MAE | $294,435/month | Average dollar error per month |
| RMSE | $344,337 | Penalizes larger misses more |

**12-Month Forward Forecast:**
- Average Monthly Revenue: ~$9.4M
- Projected Annual Revenue: ~$113M
- 95% Confidence Interval: $7.2M – $11.6M/month

---

##  Project Structure

```
forecasting_project/
│
├── forecast.py                  # Main script (run this)
├── README.md                    # This file
│
├── data/
│   └── toll_revenue_raw.csv     # Generated source data
│
└── outputs/
    ├── forecasting_dashboard.png  # 4-panel visual dashboard
    └── forecast_results.xlsx      # Excel report (4 sheets)
        ├── Historical Data
        ├── Model Validation
        ├── 12-Month Forecast
        └── Model Summary
```

---

##  How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib scikit-learn statsmodels openpyxl

# Run the model
python forecast.py
```

Output files will appear in the `outputs/` folder.

---

##  Key Insights

- **Summer seasonality**: June–August generates ~6–8% more revenue than the annual average due to higher traffic volume — useful for capacity and staffing planning.
- **COVID impact modeled**: The data includes a realistic shock-and-recovery pattern, and the model still achieves excellent accuracy on post-recovery data.
- **Uncertainty quantification**: Confidence intervals widen further into the future, accurately representing that forecasts become less certain over time.

---

##  Relevance to Financial Analytics Roles

This project demonstrates:
-  Time-series forecasting and scenario modeling
-  Financial variance analysis (actual vs. predicted)
-  Data pipeline: raw data → model → business-ready report
-  Communicating technical results to non-technical stakeholders
-  Python proficiency (pandas, scikit-learn, statsmodels, matplotlib)

