"""
STEP 2: TRAIN & COMPARE MODELS
--------------------------------
This script trains 3 different machine learning models on the cleaned
data and compares how well each one predicts heart disease:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest

It prints accuracy + a report for each model, then saves all three
so predict.py can use them later.
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

RANDOM_STATE = 42

# 1. Load the cleaned, feature-selected dataset
df = pd.read_csv("heart_disease_processed_feature_selected.csv")

X = df.drop("Heart_Disease", axis=1)
y = df["Heart_Disease"]

# 1b. Load the scaler that was already fit in preprocess.py, and scale the
#     features here (the saved CSV stays human-readable - the scaling
#     only what goes into the model gets scaled)
with open("artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

scaler = artifacts["scaler"]
X = pd.DataFrame(scaler.transform(X), columns=X.columns)

# 2. Split into training data (80%) and testing data (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 3. Define the 3 models to train
models = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE),
}

trained_models = {}
results = {}

# 4. Train each model and check how accurate it is
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    trained_models[name] = model
    results[name] = accuracy

    print("=" * 50)
    print(name)
    print("=" * 50)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=["No Heart Disease", "Heart Disease"]))
    print()

# 5. Show which model performed best
best_model_name = max(results, key=results.get)
print("=" * 50)
print("MODEL COMPARISON")
print("=" * 50)
for name, acc in sorted(results.items(), key=lambda x: -x[1]):
    marker = "  <-- BEST" if name == best_model_name else ""
    print(f"{name}: {acc * 100:.2f}%{marker}")

# 6. Save all trained models so predict.py can load them
with open("trained_models.pkl", "wb") as f:
    pickle.dump({
        "models": trained_models,
        "best_model_name": best_model_name,
    }, f)

print("\nSaved: trained_models.pkl")
print("Model training completed.")
