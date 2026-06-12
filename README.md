#  Customer Churn Prediction Web App

An end-to-end Machine Learning–driven web application designed to predict customer churn for subscription-based businesses.
The project implements a production-ready ML pipeline, from data exploration and feature engineering to model deployment using Flask.

Pipeline:
Exploratory Data Analysis → Feature Engineering → Model Training → Evaluation → Deployment

---

##  Problem Statement

Customer churn significantly impacts revenue and growth in subscription-based industries.
The objective of this project is to identify customers at high risk of churn in advance, enabling businesses to implement targeted retention strategies and reduce revenue loss.

---

##  Dataset
- **Source:** Telecom Customer Churn dataset (Kaggle)
- **Rows:** 7,043
- **Features:** 20 customer attributes 
- **Target Variable:** `Churn` (Yes / No)

---

##  Exploratory Data Analysis (EDA)


- Customers with short tenure exhibit significantly higher churn rates

- Month-to-month contracts are the most churn-prone

- Higher monthly charges strongly correlate with churn

- Fiber optic users show higher churn compared to other services

- Electronic check payment method has the highest churn rate
---

##  Feature Engineering & Preprocessing

- Detected and handled hidden missing values in TotalCharges

- Applied business-logic-driven imputation strategies

- Feature transformation strategy:

    - Numerical features → Standard scaling

    - Binary categorical features → Label encoding

    - Multi-class categorical features → One-hot encoding

- Implemented Pipeline and ColumnTransformer to:

- Prevent data leakage

- Ensure reproducibility

- Enable seamless deployment

---

##  Machine Learning Model
- Algorithm: Logistic Regression

- Rationale:

 - High interpretability

 - Probabilistic output for risk scoring

 - Strong baseline for churn prediction problems

Class Imbalance Handling: class_weight = 'balanced'

## Tech Stack

- Languages & Libraries: Python, NumPy, Pandas, Scikit-learn

- Visualization: Matplotlib, Seaborn

- Backend: Flask

- Model Serialization: Pickle

###  Model Performance
| Metric               | Score      |
| -------------------- | ---------- |
| Accuracy             | **74.82%** |
| Recall (Churn = Yes) | **0.78**   |
| F1-Score             | **0.62**   |
| ROC-AUC              | **0.84**   |


Recall was prioritized to minimize false negatives, as missing a potential churner is more costly from a business perspective.

---

##  Business Cost Framing

- False Negatives (missed churners) have higher business cost than False Positives

- Model optimized to maximize churn detection

- Predictions are suitable for:

 - Retention campaigns

 - Personalized offers

 - Customer lifetime value optimization

---

##  Web Application (Flask)
- Interactive and user-friendly web interface

- Accepts real-time customer inputs via form

- Outputs:

 - Churn Probability

 - Risk Category: Low / Medium / High

- Designed for business stakeholders and non-technical users

---

##  Project Structure
```bash 
customer-churn-prediction/
│
├── app.py
├── train_logistic_regression.py
│
├── model/
│   ├── model.pkl
│   └── pipeline.pkl
│
├── notebooks/
│   ├── eda.ipynb
│   ├── exploring_data.ipynb   
│   ├── main.ipynb
│   ├── train_random_forest.ipynb
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── requirements.txt
└── README.md
```
##  Model Behavior Validation

To validate real-world usefulness, the model was tested on contrasting customer profiles:

| Customer Type | Churn Probability | Risk |
|--------------|------------------|------|
| Short-tenure, month-to-month, high charges | 0.93 | High |
| Long-tenure, two-year contract, low charges | 0.02 | Low |

The predictions align strongly with business intuition and EDA insights.

## Deployment & Source Code
- Deployment: Flask-based Machine Learning Web Application

- Live Demo: 🌐 https://customer-churn-prediction-8wxl.onrender.com/

- Platform: Render

- Purpose: Enables real-time customer churn risk prediction through an interactive web interface


##  How to Run Locally
```bash
pip install -r requirements.txt
python app.py
```
## Author

Robiul Hasan Jisan