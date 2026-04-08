"""
Diabetes Detection - Live Prediction Demo (Symptom-Based)
==========================================================
Just answer Yes/No questions about your symptoms.
No blood tests, no clinic visit needed!

USAGE: venv\Scripts\python.exe predict.py
"""

import joblib
import numpy as np
from pathlib import Path
import os

MODEL_PATH = Path(r"d:\Miniproject\model_outputs\diabetes_rf_model.pkl")
FEATURES_PATH = Path(r"d:\Miniproject\model_outputs\feature_names.pkl")

# Symptom display names and descriptions
SYMPTOM_INFO = {
    'Age': {'question': 'What is your age?', 'type': 'number'},
    'Gender': {'question': 'What is your gender?', 'type': 'gender'},
    'Polyuria': {'question': 'Do you urinate frequently (more than usual)?', 'type': 'yesno'},
    'Polydipsia': {'question': 'Do you feel excessively thirsty?', 'type': 'yesno'},
    'sudden weight loss': {'question': 'Have you experienced sudden/unexplained weight loss?', 'type': 'yesno'},
    'weakness': {'question': 'Do you feel weakness or fatigue regularly?', 'type': 'yesno'},
    'Polyphagia': {'question': 'Do you feel increased hunger (eating more than usual)?', 'type': 'yesno'},
    'Genital thrush': {'question': 'Do you have genital infections or thrush?', 'type': 'yesno'},
    'visual blurring': {'question': 'Do you experience blurred vision?', 'type': 'yesno'},
    'Itching': {'question': 'Do you have persistent itchy skin?', 'type': 'yesno'},
    'Irritability': {'question': 'Do you feel unusually irritable?', 'type': 'yesno'},
    'delayed healing': {'question': 'Do your wounds take a long time to heal?', 'type': 'yesno'},
    'partial paresis': {'question': 'Do you experience muscle weakness or partial paralysis?', 'type': 'yesno'},
    'muscle stiffness': {'question': 'Do you have muscle stiffness?', 'type': 'yesno'},
    'Alopecia': {'question': 'Do you have unusual hair loss?', 'type': 'yesno'},
    'Obesity': {'question': 'Are you overweight or obese?', 'type': 'yesno'},
}

DISPLAY_NAMES = {
    'Polyuria': 'Frequent Urination',
    'Polydipsia': 'Excessive Thirst',
    'Polyphagia': 'Increased Hunger',
    'sudden weight loss': 'Sudden Weight Loss',
    'weakness': 'Weakness/Fatigue',
    'Genital thrush': 'Genital Infections',
    'visual blurring': 'Blurred Vision',
    'Itching': 'Itchy Skin',
    'Irritability': 'Irritability',
    'delayed healing': 'Slow-Healing Wounds',
    'partial paresis': 'Muscle Weakness',
    'muscle stiffness': 'Muscle Stiffness',
    'Alopecia': 'Hair Loss',
    'Obesity': 'Obesity',
    'Age': 'Age',
    'Gender': 'Gender',
}


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    print()
    print("  " + "=" * 58)
    print("  |                                                        |")
    print("  |     DIABETES RISK SCREENING SYSTEM                    |")
    print("  |     Symptom-Based AI Detection                        |")
    print("  |                                                        |")
    print("  |     No blood tests needed!                            |")
    print("  |     Just answer simple Yes/No questions.              |")
    print("  |                                                        |")
    print("  " + "=" * 58)
    print()


def get_yes_no(prompt):
    while True:
        val = input(f"  {prompt} (yes/no): ").strip().lower()
        if val in ('yes', 'y', '1'):
            return 1
        elif val in ('no', 'n', '0'):
            return 0
        print("    -> Please enter 'yes' or 'no'")


def get_number(prompt, min_val=1, max_val=120):
    while True:
        try:
            val = int(input(f"  {prompt}: ").strip())
            if min_val <= val <= max_val:
                return val
            print(f"    -> Enter a number between {min_val} and {max_val}")
        except ValueError:
            print("    -> Please enter a valid number")


def predict_patient(model, feature_names):
    print("\n  " + "-" * 50)
    print("  PATIENT SYMPTOM ASSESSMENT")
    print("  " + "-" * 50)
    print("  Answer the following questions honestly.\n")

    values = {}

    for feat in feature_names:
        info = SYMPTOM_INFO[feat]
        if info['type'] == 'number':
            values[feat] = get_number(info['question'])
        elif info['type'] == 'gender':
            while True:
                g = input(f"  {info['question']} (male/female): ").strip().lower()
                if g in ('male', 'm'):
                    values[feat] = 1
                    break
                elif g in ('female', 'f'):
                    values[feat] = 0
                    break
                print("    -> Please enter 'male' or 'female'")
        elif info['type'] == 'yesno':
            values[feat] = get_yes_no(info['question'])

    # Build feature vector
    features = np.array([[values[f] for f in feature_names]])

    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    prob_neg = probability[0] * 100
    prob_pos = probability[1] * 100

    # Count symptoms
    symptom_cols = [f for f in feature_names if f not in ['Age', 'Gender']]
    symptom_count = sum(values[f] for f in symptom_cols)

    # Display results
    print("\n\n  " + "=" * 58)
    if prediction == 1:
        print("  |                                                        |")
        print("  |  RESULT: *** HIGH DIABETES RISK DETECTED ***          |")
        print("  |                                                        |")
        print("  " + "=" * 58)
        print(f"\n  Diabetes Risk Score: {prob_pos:.1f}%")
        print(f"  Symptoms reported: {symptom_count} out of 14")
        print("\n  RECOMMENDATION:")
        print("  >> Please visit a doctor for a confirmatory HbA1c test.")
        print("  >> Early detection can prevent serious complications.")
        print("  >> This is a screening tool, not a diagnosis.")
    else:
        print("  |                                                        |")
        print("  |  RESULT: LOW DIABETES RISK                            |")
        print("  |                                                        |")
        print("  " + "=" * 58)
        print(f"\n  Diabetes Risk Score: {prob_pos:.1f}%")
        print(f"  Symptoms reported: {symptom_count} out of 14")
        print("\n  RECOMMENDATION:")
        print("  >> Low risk based on current symptoms.")
        print("  >> Continue healthy lifestyle and annual checkups.")

    # Show which symptoms flagged
    print("\n  " + "-" * 50)
    print("  SYMPTOM SUMMARY")
    print("  " + "-" * 50)

    flagged = []
    clear_list = []
    for feat in symptom_cols:
        name = DISPLAY_NAMES.get(feat, feat)
        if values[feat] == 1:
            flagged.append(name)
        else:
            clear_list.append(name)

    if flagged:
        print("\n  Symptoms PRESENT:")
        for s in flagged:
            print(f"    [!] {s}")
    if clear_list:
        print(f"\n  Symptoms ABSENT: {len(clear_list)} out of 14")

    # Risk factors
    print("\n  " + "-" * 50)
    print("  RISK FACTORS")
    print("  " + "-" * 50)
    age = values['Age']
    if age >= 60:
        print(f"  [!] Age {age} - HIGH risk category (60+)")
    elif age >= 45:
        print(f"  [~] Age {age} - MODERATE risk (45-59)")
    else:
        print(f"  [OK] Age {age} - Lower risk category")

    if values.get('Polyuria') and values.get('Polydipsia'):
        print("  [!] CLASSIC SIGNS: Both frequent urination AND excessive thirst present")

    if symptom_count >= 8:
        print(f"  [!] HIGH symptom count ({symptom_count}/14) - multiple warning signs!")
    elif symptom_count >= 4:
        print(f"  [~] MODERATE symptom count ({symptom_count}/14)")
    else:
        print(f"  [OK] LOW symptom count ({symptom_count}/14)")

    print("\n  DISCLAIMER: This is an AI screening tool, NOT a medical")
    print("  diagnosis. Always consult a healthcare professional.")
    print()


def run_demo_cases(model, feature_names):
    """Pre-built demo cases for quick presentation."""
    demos = [
        {
            "name": "Healthy Young Person (28, Female)",
            "values": {'Age': 28, 'Gender': 0, 'Polyuria': 0, 'Polydipsia': 0,
                       'sudden weight loss': 0, 'weakness': 0, 'Polyphagia': 0,
                       'Genital thrush': 0, 'visual blurring': 0, 'Itching': 0,
                       'Irritability': 0, 'delayed healing': 0, 'partial paresis': 0,
                       'muscle stiffness': 0, 'Alopecia': 0, 'Obesity': 0}
        },
        {
            "name": "High-Risk Elderly Patient (65, Male)",
            "values": {'Age': 65, 'Gender': 1, 'Polyuria': 1, 'Polydipsia': 1,
                       'sudden weight loss': 1, 'weakness': 1, 'Polyphagia': 1,
                       'Genital thrush': 0, 'visual blurring': 1, 'Itching': 1,
                       'Irritability': 1, 'delayed healing': 1, 'partial paresis': 1,
                       'muscle stiffness': 1, 'Alopecia': 0, 'Obesity': 1}
        },
        {
            "name": "Borderline Case (50, Female - Few Symptoms)",
            "values": {'Age': 50, 'Gender': 0, 'Polyuria': 1, 'Polydipsia': 1,
                       'sudden weight loss': 0, 'weakness': 1, 'Polyphagia': 0,
                       'Genital thrush': 0, 'visual blurring': 0, 'Itching': 1,
                       'Irritability': 0, 'delayed healing': 1, 'partial paresis': 0,
                       'muscle stiffness': 0, 'Alopecia': 0, 'Obesity': 0}
        },
        {
            "name": "Classic Diabetes Signs (55, Male)",
            "values": {'Age': 55, 'Gender': 1, 'Polyuria': 1, 'Polydipsia': 1,
                       'sudden weight loss': 1, 'weakness': 1, 'Polyphagia': 1,
                       'Genital thrush': 0, 'visual blurring': 1, 'Itching': 0,
                       'Irritability': 0, 'delayed healing': 0, 'partial paresis': 1,
                       'muscle stiffness': 0, 'Alopecia': 0, 'Obesity': 0}
        },
    ]

    for demo in demos:
        features = np.array([[demo['values'][f] for f in feature_names]])
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0]

        status = "DIABETIC (High Risk)" if pred == 1 else "NON-DIABETIC (Low Risk)"
        risk_pct = prob[1] * 100
        symptom_count = sum(v for k, v in demo['values'].items() if k not in ['Age', 'Gender'])

        print(f"\n  Patient: {demo['name']}")
        print(f"  Symptoms: {symptom_count}/14 | Risk Score: {risk_pct:.1f}%")
        print(f"  Prediction: {status}")
        print(f"  " + "-" * 50)


# --- Main ---
if __name__ == "__main__":
    print("\nLoading model...")
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print("Model loaded!\n")

    while True:
        clear_screen()
        print_banner()
        print("  Choose an option:")
        print("  [1] Screen a patient (enter symptoms)")
        print("  [2] Run demo cases (for presentation)")
        print("  [3] Exit")
        print()

        choice = input("  Your choice (1/2/3): ").strip()

        if choice == '1':
            predict_patient(model, feature_names)
            input("\n  Press Enter to continue...")
        elif choice == '2':
            run_demo_cases(model, feature_names)
            input("\n  Press Enter to continue...")
        elif choice == '3':
            print("\n  Goodbye!\n")
            break
        else:
            print("  Invalid choice.")
            input("\n  Press Enter to continue...")
