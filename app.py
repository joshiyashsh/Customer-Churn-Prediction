import pandas as pd
import pickle
import joblib
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
# 2. Load Dataset

data = pd.read_csv("data/customer_churn_dataset-training-master.csv")

# 3. Remove Duplicate Rows

print("Duplicate Rows :", data.duplicated().sum())

data = data.drop_duplicates()

# 4. Remove Missing Target

data = data.dropna(subset=["Churn"])

# 5. Remove CustomerID

if "CustomerID" in data.columns:
    data = data.drop(columns=["CustomerID"])

# 6. Separate Features and Target


X = data.drop("Churn", axis=1)
y = data["Churn"]

# 7. Train Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 8. Define Columns

cat_cols = [
    "Gender",
    "Subscription Type",
    "Contract Length"
]

num_cols = [
    col
    for col in X_train.columns
    if col not in cat_cols
]

# 9. Numerical Pipeline

num_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

# 10. Categorical Pipeline

cat_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


# 11. Full Pipeline

full_pipeline = ColumnTransformer(
    transformers=[
        (
            "num",
            num_pipeline,
            num_cols
        ),
        (
            "cat",
            cat_pipeline,
            cat_cols
        )
    ]
)


# 12. Transform Data

X_train_processed = full_pipeline.fit_transform(X_train)
X_test_processed = full_pipeline.transform(X_test)

# 13. Evaluation Function

def evaluate(model_name, model):

    model.fit(
        X_train_processed,
        y_train
    )

    pred = model.predict(
        X_test_processed
    )

    print("\n")
    print("=" * 60)
    print(model_name)
    print("=" * 60)

    print(
        "Accuracy :",
        accuracy_score(
            y_test,
            pred
        )
    )

    print(
        "Precision :",
        precision_score(
            y_test,
            pred
        )
    )

    print(
        "Recall :",
        recall_score(
            y_test,
            pred
        )
    )

    print(
        "F1 Score :",
        f1_score(
            y_test,
            pred
        )
    )

    print("\nConfusion Matrix\n")

    print(
        confusion_matrix(
            y_test,
            pred
        )
    )

    print("\nClassification Report\n")

    print(
        classification_report(
            y_test,
            pred
        )
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X_train_processed,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    print("5-Fold CV Accuracy :", scores)
    print("Mean Accuracy :", scores.mean())

# 14. Logistic Regression

log_model = LogisticRegression(
    max_iter=1000,
    C=0.5
)

evaluate(
    "Logistic Regression",
    log_model
)

# 15. Decision Tree

dt_model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=50,
    min_samples_leaf=25,
    random_state=42
)

evaluate(
    "Decision Tree",
    dt_model
)


# 16. Random Forest


rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=50,
    min_samples_leaf=25,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

evaluate(
    "Random Forest",
    rf_model
)


# 17. Feature Importance


feature_names = full_pipeline.get_feature_names_out()

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n")
print("=" * 60)
print("TOP 20 IMPORTANT FEATURES")
print("=" * 60)

print(
    importance.head(20)
)


# Save preprocessing pipeline

joblib.dump(
    full_pipeline,
    "pipeline.pkl"
)
# Save trained Random Forest model
joblib.dump(
    rf_model,
    "model.pkl"
)

print("\n====================================")
print("Files Saved Successfully")
print("pipeline.pkl")
print("model.pkl")
print("====================================")