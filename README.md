# E-Commerce-User-Behavior-Analysis-Purchase-Prediction
This project involves cleaning, analyzing, and modeling e-commerce user behavior data to uncover insights and predict purchase intent. The goal is to help analysts and data scientists quickly understand customer journeys and apply machine learning to enhance business decisions.

# Introduction

Inspired by user log analysis practices and practical business use cases, this project focuses on building a reproducible, scalable pipeline for:

* Cleaning and transforming raw user activity logs.

* Exploratory data analysis (EDA) to uncover purchase patterns and funnel drop-offs.

* Engineering user and session-level features.

* Predicting user purchase intent using machine learning (Logistic Regression, XGBoost).

* Visualizing evaluation metrics such as confusion matrix and ROC curves.

This project is especially useful for understanding e-commerce funnels and optimizing conversion strategies.

# Requirements & Getting Started

Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost

Run the analysis notebook or script sequentially:

data_preprocessing.ipynb - Load and clean dataset.

eda_feature_engineering.ipynb - Perform EDA and generate features.

model_training.ipynb - Train ML models.

evaluation.ipynb - Compare model performance.

Data Source

The dataset includes anonymized user interactions with the e-commerce platform. Each row represents a user event, containing:

* event_time: Timestamp of the user event.

* user_id, session_id: User and session identifiers.

* event_type: One of view, cart, or purchase.

* product_id, category_code, brand: Item-related features.

* price: Product price.

This project filters and focuses on relevant events for modeling user conversion behavior.

# Project Structure

* 0_load_dataset.ipynb: Load dataset from source and change it to a parquet file.

* 1_Summary_Statistics.ipynb: Explores user activity, constructs features such as prior views, session event counts, previous purchases of a brand, etc.

* 2_Supervised_Learnings.ipynb: Trains classification model: XGBoost. Applies class balancing if needed. Visualizes confusion matrices and ROC curves. Displays performance metrics in formatted tables.

# Results

* Most purchases are preceded by 2-4 views.

* Certain brands have significantly higher conversion rates.

* XGBoost performs with high precision and ROC-AUC.

## Model accuracy:
  Precision  Recall  F1-Score       Support  ROC AUC
0                 1.000   0.836     0.911  1.030207e+07      NaN
1                 0.089   0.998     0.163  1.647770e+05      NaN
accuracy          0.839   0.839     0.839  8.390000e-01      NaN
macro avg         0.544   0.917     0.537  1.046685e+07      NaN
weighted avg      0.986   0.839     0.899  1.046685e+07      NaN
ROC AUC             NaN     NaN       NaN           NaN     0.93

# Future Work

* Tune hyperparameters using cross-validation.

* Add time-series features (e.g., session duration).

* Explore deep learning approaches.
