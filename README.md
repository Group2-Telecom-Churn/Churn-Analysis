# Churn-Analysis
Telecom churn prediction project

# Telecom Customer Churn AI

A machine learning application for predicting telecom customer churn and helping retention officers identify customers who may need intervention.

The project combines a trained Random Forest model with a Streamlit dashboard, customer search and registry, customer-level risk signals, retention recommendations, analytics, and system logging.

## Project Overview

The main goal of the project is to predict the likelihood that a telecom customer will churn and turn that prediction into something useful for a retention team.

The application allows a retention officer to:

- Enter a customer's details and run a churn prediction.
- Search for an existing customer using their Customer ID.
- View the customer's previous prediction and historical churn result.
- See the customer's churn and retention probabilities.
- View the main features influencing the individual prediction.
- Get suggested retention actions.
- View all customers stored in the registry.
- Review application and model scan analytics.
- View system activity through the system logs.

## Machine Learning Pipeline

The machine learning workflow used for the project is:

```text
Dataset
   ↓
Data Cleaning & Preparation
   ↓
Exploratory Data Analysis
   ↓
Feature Preparation
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Hyperparameter Tuning
   ↓
Random Forest Model
   ↓
Saved Preprocessor + Model
   ↓
Streamlit Application

The application uses the saved preprocessing pipeline and trained model. It does not retrain the model when a prediction is made.

Churn Classification

The final application uses a locked classification threshold of 40%.

Churn Probability >= 40%  →  Churn Risk
Churn Probability < 40%   →  Retention

The threshold is defined in the application as:

CHURN_THRESHOLD = 0.40

Risk levels are displayed as:

Churn Probability	Risk
Below 40%	Low
40% – below 60%	Medium
60% and above	High

The circular risk indicator changes from green to yellow to red as the churn probability increases.

Customer Search and Registry

Each customer is identified using their customerID.

The Customer Registry stores customer information and prediction results. The demonstration database is initially populated with approximately 70 customers:

35 historical churn customers
35 historical non-churn customers

These customers are scored when the application initializes, so searching for one of them immediately returns its stored result.

Customers entered through a new prediction are also added to the registry.

Customer Signals

For a selected customer, the application calculates which features have the largest effect on the model's predicted churn probability.

The application tests changes to individual customer features and compares the resulting churn probability with the original prediction.

The strongest signals are displayed to the retention officer.

These signals describe the model's sensitivity to the customer's features. They should not be interpreted as proof that a feature directly causes a customer to churn.

Suggested Retention Actions

The application provides suggested retention actions based on the customer's churn risk and profile.

The recommendations can consider factors such as:

Contract type
Customer tenure
Monthly charges
Online security
Technical support
Payment method

The recommendations are intended to support the retention officer's decision. They do not automatically contact or change anything for the customer.

Analytics

The Analytics page provides an overview of customer scans performed through the application.

It includes information such as:

Total scans
Churn flags
Non-churn flags
Average churn probability
Prediction distribution
Churn probability across scans
Accuracy for customers with known historical churn outcomes

Accuracy is only calculated where an actual historical churn result is available. A newly scanned customer cannot be counted as correct or incorrect until an actual outcome is known.

System Logs

The System Logs page records important application activity.

Examples include:

Application startup
Database initialization
Customer searches
Customer predictions
System errors

Logs include information such as timestamps, event types, customer IDs where applicable, and event details.

Project Structure
Churn-Analysis/
│
├── app.py
│
├── models/
│   ├── preprocessor.joblib
│   └── best_random_forest_tuned.pkl
│
├── prototype_seed_customers.csv
│
├── telecom_churn.db
│
├── requirements.txt
│
└── README.md
Main Files
File	Description
app.py	Main Streamlit application
preprocessor.joblib	Saved preprocessing pipeline
best_random_forest_tuned.pkl	Saved Random Forest model
prototype_seed_customers.csv	Demonstration customer records
telecom_churn.db	SQLite database
requirements.txt	Python dependencies
Dataset

The dataset contains telecom customer information including:

Customer ID
Gender
Senior Citizen status
Partner
Dependents
Tenure
Phone service
Multiple lines
Internet service
Online security
Online backup
Device protection
Technical support
Streaming TV
Streaming movies
Contract
Paperless billing
Payment method
Monthly charges
Total charges
Churn

Churn is the target variable.

Technology

The project was built using:

Python
Pandas
NumPy
Scikit-learn
Joblib
Streamlit
SQLite
GitHub
Running the Application

Clone the repository:

git clone https://github.com/Group2-Telecom-Churn/Churn-Analysis.git

Enter the project directory:

cd Churn-Analysis

Install the required packages:

pip install -r requirements.txt

Run the application:

streamlit run app.py
Deployment

The application is deployed through Streamlit Community Cloud.

The deployment uses:

GitHub
   ↓
Streamlit Community Cloud
   ↓
requirements.txt
   ↓
app.py
   ↓
Live Application

The main application file is app.py and the deployment branch is main.

Model Compatibility

The saved model and preprocessing pipeline depend on compatible versions of the machine learning libraries used to create them.

The versions specified in requirements.txt should therefore be kept consistent with the versions used when the model artifacts were created.

Changes to packages such as Scikit-learn, NumPy, Pandas, or Joblib should be tested against the saved model before deployment.

Limitations
The model predicts the likelihood of churn; it does not guarantee that a customer will churn.
Customer signals show model sensitivity and are not causal explanations.
The preloaded customers are for demonstration and evaluation.
SQLite is suitable for the current project but would not be the preferred database for a large production deployment.
Retention recommendations are decision-support suggestions and should be reviewed by a retention officer.
Future Improvements

If the application were taken into a full production environment, possible improvements would include:

Moving from SQLite to a managed database such as PostgreSQL.
Adding authentication and user roles.
Adding stronger customer-data security.
Adding model and data drift monitoring.
Adding automated model retraining.
Adding automated tests and CI/CD.
Adding production monitoring and alerting.
Introducing formal model governance.
Project Information

Project: Telecom Customer Churn AI

Project Group: Group 2 – Telecom Churn

Final Project Status

The machine learning pipeline and Streamlit application have been completed and integrated.

The final system includes:

Model
 ↓
Prediction
 ↓
40% Churn Threshold
 ↓
Risk Classification
 ↓
Customer Signals
 ↓
Retention Recommendations
 ↓
Customer Registry
 ↓
Analytics & System Logs

The application is deployed and ready for demonstration and final project submission.
