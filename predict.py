"""
STEP 3: PREDICT HEART DISEASE RISK
------------------------------------
This is the part a normal user interacts with.
It asks a few SIMPLE questions (no confusing medical terms), then uses
the trained models to say whether the person is likely at risk of
heart disease.

Run this only AFTER running preprocess.py and train_models.py once.
"""

import pickle
import pandas as pd

# Load the saved preprocessing tools (encoders + scaler + feature list)
with open("artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

encoders = artifacts["encoders"]
scaler = artifacts["scaler"]
selected_features = artifacts["selected_features"]

# Load the trained models
with open("trained_models.pkl", "rb") as f:
    saved = pickle.load(f)

models = saved["models"]
best_model_name = saved["best_model_name"]

# -----------------------------------------------------------------
# Simple, everyday-language questions for every possible column.
# Only the ones needed (selected_features) will actually be asked.
# -----------------------------------------------------------------
QUESTIONS = {
    "Age": ("What is your age? (18-99)", "number"),
    "Gender": ("Are you Male or Female? (M/F)", "gender"),
    "Weight": ("What is your weight in kg? (50-119)", "number"),
    "Height": ("What is your height in cm? (150-199)", "number"),
    "BMI": ("What is your BMI (weight compared to height)? (15-40)", "number"),
    "Smoking": ("Do you smoke? (Never / Current / Former)", "category"),
    "Alcohol_Intake": ("How much alcohol do you drink? (Low / Medium / High)", "category"),
    "Physical_Activity": ("How active are you day to day? (Low / Medium / High)", "category"),
    "Diet": ("Is your diet mostly Healthy, Average or Unhealthy?", "category"),
    "Stress_Level": ("How would you rate your stress? (Low / Medium / High)", "category"),
    "Hypertension": ("Do you have high blood pressure? (Yes/No)", "yesno"),
    "Diabetes": ("Do you have diabetes? (Yes/No)", "yesno"),
    "Hyperlipidemia": ("Do you have high cholesterol? (Yes/No)", "yesno"),
    "Family_History": ("Does anyone in your family have heart disease? (Yes/No)", "yesno"),
    "Previous_Heart_Attack": ("Have you ever had a heart attack before? (Yes/No)", "yesno"),
    "Systolic_BP": ("What is the TOP number of your blood pressure reading? (90-180)", "number"),
    "Diastolic_BP": ("What is the BOTTOM number of your blood pressure reading? (60-120)", "number"),
    "Heart_Rate": ("What is your resting heart rate in beats per minute? (60-109)", "number"),
    "Blood_Sugar_Fasting": ("What is your fasting blood sugar level? (70-179)", "number"),
    "Cholesterol_Total": ("What is your total cholesterol number? (150-299)", "number"),
}

# -----------------------------------------------------------------
# Some columns store values in the dataset (e.g. "Moderate", "Sedentary")
# that we want to ask about using simpler Low/Medium/High wording.
# This maps the simple word the user types to the real dataset value.
# -----------------------------------------------------------------
LABEL_MAPS = {
    "Alcohol_Intake": {"Low": "Low", "Medium": "Moderate", "High": "High"},
    "Physical_Activity": {"Low": "Sedentary", "Medium": "Moderate", "High": "Active"},
}


def ask(column):
    prompt, kind = QUESTIONS[column]

    if kind == "number":
        while True:
            value = input(f"{prompt} ")
            try:
                return float(value)
            except ValueError:
                print("Please enter a valid number.")

    if kind == "yesno":
        while True:
            value = input(f"{prompt} ").strip().lower()
            if value in ("yes", "y"):
                return 1
            if value in ("no", "n"):
                return 0
            print("Please answer Yes or No.")

    if kind == "gender":
        le = encoders["Gender"]
        while True:
            value = input(f"{prompt} ").strip().lower()
            if value in ("m", "male"):
                return le.transform(["Male"])[0]
            if value in ("f", "female"):
                return le.transform(["Female"])[0]
            print("Please answer Male or Female.")

    if kind == "category":
        le = encoders[column]
        label_map = LABEL_MAPS.get(column)

        if label_map:
            # Simple wording (e.g. "Medium") shown to the user, translated
            # to the real dataset category (e.g. "Moderate") behind the scenes.
            options_display = " / ".join(label_map.keys())
            while True:
                value = input(f"{prompt} ").strip()
                match_label = next((o for o in label_map if o.lower() == value.lower()), None)
                if match_label:
                    real_value = label_map[match_label]
                    return le.transform([real_value])[0]
                print(f"Please choose one of: {options_display}")
        else:
            options = list(le.classes_)
            options_display = " / ".join(options)
            while True:
                value = input(f"{prompt} ").strip()
                match = next((o for o in options if o.lower() == value.lower()), None)
                if match:
                    return le.transform([match])[0]
                print(f"Please choose one of: {options_display}")


def collect_answers():
    print("Answer a few simple questions about yourself.\n")
    answers = {}

    # Always ask Height and Weight first, so we can calculate and show BMI
    # right away - even if the model itself doesn't use Weight directly.
    height = ask("Height")
    weight = ask("Weight")
    bmi = weight / ((height / 100) ** 2)
    print(f"\nYour BMI is: {bmi:.1f}\n")

    answers["Height"] = height
    answers["Weight"] = weight
    answers["BMI"] = bmi

    for column in selected_features:
        if column in answers:
            continue
        answers[column] = ask(column)

    return answers


def predict(answers, model_name=None):
    model_name = model_name or best_model_name
    model = models[model_name]

    # Build the input row in the SAME column order used during training
    row = pd.DataFrame([[answers[col] for col in selected_features]], columns=selected_features)
    row_scaled = pd.DataFrame(scaler.transform(row), columns=selected_features)

    prediction = model.predict(row_scaled)[0]
    probability = model.predict_proba(row_scaled)[0][1]  # probability of heart disease

    return prediction, probability


def get_clinical_summary(average_risk):
    if average_risk >= 75:
        return "Significant cardiac risk indicators present. Medical follow-up recommended."
    if average_risk >= 50:
        return "Moderate cardiac risk indicators present. Lifestyle review advised."
    if average_risk >= 25:
        return "Mild cardiac risk indicators present. Routine monitoring suggested."
    return "No significant cardiac risk indicators found. Maintain a healthy lifestyle."


if __name__ == "__main__":
    user_answers = collect_answers()

    # Get each model's estimated chance of heart disease, then average them
    probabilities = [predict(user_answers, model_name=name)[1] for name in models]
    average_risk = sum(probabilities) / len(probabilities) * 100
    risk_category = "HIGH RISK" if average_risk >= 50 else "LOW RISK"
    clinical_summary = get_clinical_summary(average_risk)

    print("\n" + "=" * 68)
    print("ASSESSMENT RESULT".center(68))
    print("=" * 68)
    print(f"\n Patient Heart Disease Risk : {average_risk:.2f}%")
    print(f" BMI                         : {user_answers['BMI']:.1f}")
    print(f" Risk Category               : {risk_category}")
    print(f" Clinical Summary            : {clinical_summary}")
    print("\n" + "=" * 68)

    print("\nNote: This is a simple learning project, NOT a medical diagnosis.")
    print("Please consult a doctor for any real health concerns.")
