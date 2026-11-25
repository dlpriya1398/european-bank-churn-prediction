# European Bank Customer Churn Risk Prediction & Analytics Platform

An end-to-end Machine Learning solution and interactive web application for anticipating customer churn, calculating quantitative risk probabilities, and simulating proactive retention strategies across 10,000 retail bank accounts in Europe.

---

## Project Overview & Key Deliverables

Retail bank account cancellations result in lost customer lifetime value, diminished interest margins, and operational inefficiencies. This platform transitions financial institutions from reactive customer loss to predictive retention analytics.

### Features Included:
- Executive Dashboard: High-level KPIs and multi-dimensional churn breakdowns across Geography, Product Count, Age distribution, and Active Member status.
- Individual Risk Calculator: Interactive risk-scoring engine yielding quantitative churn probabilities with actionable retention guidelines.
- What-If Scenario Simulator: Real-time simulation testing how customer interventions reduce churn probability.
- Batch CSV Scoring: Portfolio-level prediction engine with automated CSV export for targeted marketing campaigns.
- Model Methodology: Full transparency on feature engineering, model benchmarks, and SHAP feature importance.

---

## Key Empirical Findings

1. Product Saturation Risk:
   - Customers holding 1 product churn at 27.7%.
   - Customers holding 2 products demonstrate optimal retention with only 7.6% churn.
   - Customers with 3 or 4 products exhibit catastrophic churn (82.7% and 100% respectively).
2. Geographic Variance: German banking clients churn at double the rate (32.4%) of French (16.2%) and Spanish (16.7%) clients.
3. Engagement Protection: Active membership cuts churn risk nearly in half (14.3% active vs. 26.9% inactive).
4. Age Demographics: Peak attrition occurs between ages 40 and 60 (averaging 34% to 56% churn).

---

## Quick Start & Installation Guide

### Prerequisites
- Python 3.9+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/european-bank-churn-prediction.git
cd european-bank-churn-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Model Training Pipeline
```bash
python train_model.py
```

### 4. Launch the Streamlit Web Application
```bash
streamlit run app.py
```

---

## Project Structure

```
├── app.py                     # Main Streamlit Web Application script
├── train_model.py             # Model training, evaluation, and serialization script
├── European_Bank_part1.csv    # Partitioned European Bank dataset (Part 1)
├── European_Bank_part2.csv    # Partitioned European Bank dataset (Part 2)
├── requirements.txt           # Python dependencies list
└── README.md                  # Project documentation & GitHub submission guide
```
