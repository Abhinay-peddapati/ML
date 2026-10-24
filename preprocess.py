"""
STEP 1: DATA PREPROCESSING
---------------------------
This script:
  1. Loads the raw dataset
  2. Removes duplicate rows
  3. Fills missing values
  4. Converts text (categorical) columns into numbers
  5. Picks the 10 most useful columns (features) for prediction
     (Gender is always kept, since it's a common factor to ask about)
  6. Saves the cleaned data (original, readable numbers - no negatives)
     + the tools used to clean it (so predict.py can prepare new user
     answers in exactly the same way)
"""

import pandas as pd
import pickle
from functools import partial
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

RANDOM_STATE = 42
# mutual_info_classif has some internal randomness - fix the seed so the
# same 10 features are picked every time you re-run this script.
mutual_info_seeded = partial(mutual_info_classif, random_state=RANDOM_STATE)

# 1. Load the dataset
df = pd.read_csv("heart_disease_dataset_5000.csv")
print("Original Shape:", df.shape)

# 2. Remove duplicate records
df = df.drop_duplicates()

# 3. Handle missing values
#    Alcohol_Intake had some blank rows - instead of creating a separate
#    "Unknown" category, fill them with the most common answer in the
#    dataset so every value stays a real, meaningful category.
most_common_alcohol = df["Alcohol_Intake"].mode()[0]
df["Alcohol_Intake"] = df["Alcohol_Intake"].fillna(most_common_alcohol)

# 4. Convert categorical (text) columns into numbers
categorical_columns = [
    "Gender",
    "Smoking",
    "Alcohol_Intake",
    "Physical_Activity",
    "Diet",
    "Stress_Level"
]

encoders = {}  # we save these so predict.py can convert user answers the same way
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# 5. Separate input features (X) and target/answer column (y)
X = df.drop("Heart_Disease", axis=1)
y = df["Heart_Disease"]

# 6. Select the most useful features for predicting heart disease
#    Gender is always kept (common, expected demographic factor in a heart
#    disease project), and the remaining 9 "best" columns are picked
#    automatically using mutual information.
FORCE_INCLUDE = ["Gender"]
TOTAL_FEATURES = 10

remaining_columns = [c for c in X.columns if c not in FORCE_INCLUDE]
k_remaining = TOTAL_FEATURES - len(FORCE_INCLUDE)

selector = SelectKBest(score_func=mutual_info_seeded, k=k_remaining)
selector.fit(X[remaining_columns], y)
auto_selected = list(pd.Index(remaining_columns)[selector.get_support()])

selected_features = FORCE_INCLUDE + auto_selected
X_selected = X[selected_features].copy()

print("\nSelected Features:")
for feature in selected_features:
    tag = " (always included)" if feature in FORCE_INCLUDE else ""
    print(f" - {feature}{tag}")

# 7. Attach the target column
X_selected["Heart_Disease"] = y.values

# 8. Fit a scaler on the selected features (used later during training and
#    prediction) but DO NOT overwrite the saved CSV with scaled numbers -
#    keep the CSV in original, human-readable values.
scaler = StandardScaler()
scaler.fit(X_selected[selected_features])

# 9. Save the processed dataset (original, readable values - not scaled)
X_selected.to_csv("heart_disease_processed_feature_selected.csv", index=False)
print("\nFinal Dataset Shape:", X_selected.shape)

# 10. Save encoders, scaler and selected feature list
#     predict.py needs these to turn a user's simple answers into the
#     exact same numeric format the model was trained on.
artifacts = {
    "encoders": encoders,
    "scaler": scaler,
    "selected_features": selected_features,
    "categorical_columns": categorical_columns,
}
with open("artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("\nSaved: heart_disease_processed_feature_selected.csv")
print("Saved: artifacts.pkl (encoders + scaler + selected feature list)")
print("\nData processing and feature selection completed.")
