![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-3F4F75)

````markdown
# Manufacturing Defect Detection

This project focuses on building an end-to-end production-style machine learning system rather than training a standalone model.

This project simulates a real-world industrial AI workflow, starting from raw manufacturing data and ending with an interactive prediction dashboard. The focus is not only on building a machine learning model, but also on designing a maintainable, modular, and production-like data pipeline.

---
````markdown
# Project Overview

Manufacturing facilities continuously generate large amounts of process data from equipment and production lines. Identifying defective products early can reduce production costs, improve product quality, and minimize downtime.

This project demonstrates a complete defect detection workflow by:

- Generating synthetic manufacturing data
- Validating and preprocessing raw manufacturing CSV files
- Performing automated data cleaning and feature engineering
- Training a Gradient Boosting defect prediction model
- Serving predictions through a FastAPI backend
- Visualizing prediction results in an interactive Streamlit dashboard

---
````markdown
# Features

- Generate synthetic manufacturing datasets
- Automated data cleaning pipeline
- Automated feature engineering pipeline
- Gradient Boosting defect prediction model
- Interactive Streamlit dashboard
- Raw CSV upload
- Automatic preprocessing
- Automatic prediction
- KPI summary cards
- Interactive Plotly visualizations
- Prediction result download
- CSV validation
- Friendly error handling
- Modular and reusable project structure

---
````markdown
# Project Architecture

```text
        User
         │
         ▼
+------------------+
|    Streamlit     |
+------------------+
         │
   HTTP POST
         │
         ▼
+------------------+
|     FastAPI      |
+------------------+
         │
         ▼
+------------------+
| Data Pipeline    |
| - Validation     |
| - Cleaning       |
| - Feature Eng.   |
+------------------+
         │
         ▼
+------------------+
| Gradient Boosting|
+------------------+
         │
         ▼
 Prediction Results
         │
         ▼
    Streamlit UI
```

---
````markdown
# Machine Learning Pipeline

1. Generate synthetic manufacturing data
2. Clean invalid and missing values
3. Create engineered features
4. Train Gradient Boosting classifier
5. Evaluate model performance
6. Save trained model
7. Predict defects on unseen data
8. Visualize prediction results

---
````markdown
# Model Performance

| Metric    |   Score  |
|-----------|----------|
| Accuracy  |  1.0000  |
| Precision |  1.0000  |
| Recall    |  1.0000  |
| F1-score  |  1.0000  |

> **Note:** The synthetic dataset uses deterministic rule-based defect generation. Therefore, near-perfect performance is expected and primarily demonstrates the end-to-end machine learning workflow rather than real-world predictive performance.

---
````markdown
# Dashboard Features

The Streamlit dashboard provides:

- Upload raw manufacturing CSV files
- Automatic preprocessing
- Automatic feature engineering
- Defect prediction
- KPI summary cards
- Defect distribution pie chart
- Defect count bar chart
- Interactive prediction table
- Highlighted defect rows
- Download prediction results as CSV

---
````markdown
# Tech Stack

### Programming Language

- Python 3.11

### Backend

- FastAPI

### Machine Learning

- Scikit-learn
- Gradient Boosting Classifier

### Data Processing

- Pandas
- NumPy

### Frontend & Visualization

- Streamlit
- Plotly

### Development Tools

- Git
- GitHub
---
````markdown
# Project Structure

```text
ManufacturingDefectDetection/
│

│
├── dashboard/
│   └── app.py                 # Streamlit dashboard
│
├── data/
│   ├── raw/                   # Raw manufacturing data
│   └── processed/             # Processed datasets
│
├── models/
│   ├── gradient_boosting_model.pkl
│   └── feature_columns.pkl
│
├── notebooks/                 # Experimentation and EDA
│
├── reports/                   # Figures and evaluation reports
│
├── src/
│   ├── data_generator.py      # Generate synthetic manufacturing data
│   ├── data_cleaning.py       # Data validation and cleaning
│   ├── feature_engineering.py # Feature engineering pipeline
│   ├── train_model.py         # Model training
│   ├── evaluate_model.py      # Model evaluation
│   ├── predict.py              # Prediction utilities
    └── api.py                 # FastAPI backend for model inference
│
├── requirements.txt
├── README.md
└── .gitignore
---

# Installation

Clone the repository

```bash
git clone https://github.com/JK1030/ManufacturingDefectDetection.git
```

Move into the project

```bash
cd ManufacturingDefectDetection
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---
````markdown
# Running the Project

Launch the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

Upload a raw manufacturing CSV file and the application will automatically:

- Validate the CSV
- Clean the data
- Generate engineered features
- Load the trained model
- Predict manufacturing defects
- Display interactive visualizations
- Allow downloading prediction results

---
````markdown
# Dashboard Preview

*Dashboard screenshots will be added after deployment.*

---
````markdown
# Future Improvements

- Docker containerization
- Cloud deployment (Render, AWS, or Azure)
- Model monitoring and logging
- SHAP model explainability
- Hyperparameter optimization
- Compare additional ML models (XGBoost, LightGBM)
- CI/CD pipeline with GitHub Actions
- Real-time prediction API


End-to-end Machine Learning pipeline for manufacturing defect prediction with automated preprocessing, Gradient Boosting inference, and an interactive Streamlit dashboard.
---
````markdown
# Author

**Jenny Jinmyeong Kim**

Industrial AI | Machine Learning | Manufacturing Analytics | Data Engineering

GitHub: https://github.com/JK1030/ManufacturingDefectDetection
