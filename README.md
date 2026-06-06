# 📊 Customer Churn Prediction using Machine Learning

## 🚀 Project Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses. Losing existing customers can significantly impact revenue and growth. This project aims to predict whether a customer is likely to **churn** (leave the service) or **stay**, using Machine Learning techniques.

The project includes complete data preprocessing, feature engineering, model training, evaluation, feature importance analysis, model serialization using **Joblib**, and deployment through a **Streamlit web application**.

# 🎯 Objective

The goal of this project is to build a machine learning model that can accurately predict customer churn based on customer demographics, subscription details, and usage behavior.

Businesses can use this prediction to:

* Improve customer retention
* Identify high-risk customers
* Reduce revenue loss
* Create targeted marketing strategies

# 📂 Project Structure

   text
Customer-Churn-Prediction/
│
├── data/
│   └── customer_churn_dataset-training-master.csv
│
├── app.py
├── streamlit_app.py
├── model.pkl
├── pipeline.pkl
├── requirements.txt
└── README.md


# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit

# 📋 Features

* Data Cleaning
* Duplicate Removal
* Missing Value Handling
* Feature Encoding
* Feature Scaling
* Train-Test Split
* Machine Learning Model Training
* Cross Validation
* Performance Evaluation
* Feature Importance Analysis
* Model Serialization
* Interactive Streamlit Web Application


# 📊 Machine Learning Models Used

The following models were trained and evaluated:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

# 📈 Model Performance

| Model               | Accuracy   |
| ------------------- | ---------- |
| Logistic Regression | **89.33%** |
| Decision Tree       | **97.76%** |
| Random Forest       | **98.81%** |

### ✅ Best Model

**Random Forest Classifier**

Performance:

* Accuracy: **98.81%**
* Precision: **99.65%**
* Recall: **98.25%**
* F1 Score: **98.94%**

# 🔥 Top Important Features

According to the Random Forest model, the most influential features are:

1. Support Calls
2. Total Spend
3. Payment Delay
4. Age
5. Contract Length
6. Last Interaction

These factors play a significant role in predicting customer churn.

# ⚙️ Installation

Clone the repository:
   
   bash
git clone https://github.com/your-username/Customer-Churn-Prediction.git

Go to the project directory:

   bash
cd Customer-Churn-Prediction
   

Install dependencies:

   bash
pip install -r requirements.txt
   

   

# ▶️ Run the Training Script

   bash
python app.py
   

This will:

* Load the dataset
* Preprocess the data
* Train the models
* Evaluate performance
* Save:

  * `model.pkl`
  * `pipeline.pkl`



# 🌐 Run the Streamlit App

   bash
streamlit run streamlit_app.py
   

Then open the local URL displayed in your terminal to access the application in your browser.

# 📌 Workflow

text
Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Preprocessing
      │
      ▼
Train-Test Split
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Feature Importance
      │
      ▼
Save Model (.pkl)
      │
      ▼
Streamlit Deployment

# 💡 Future Improvements

* Hyperparameter tuning
* ROC-AUC visualization
* SHAP explainability
* Probability score prediction
* Docker deployment
* Cloud deployment (Render/AWS/Azure)


# 📚 Learning Outcomes

Through this project, I gained hands-on experience with:

* Data preprocessing
* Machine Learning pipelines
* Classification algorithms
* Model evaluation metrics
* Cross-validation
* Feature importance analysis
* Model serialization using Joblib
* Streamlit deployment


# 👨‍💻 Author

**Yash Joshi**

Machine Learning & AI || Data Science
If you found this project useful, feel free to ⭐ the repository and share your feedback.
