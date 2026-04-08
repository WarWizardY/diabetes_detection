"""
Diabetes Detection (Symptom-Based) - Random Forest Training
=============================================================
Trains a Random Forest on 16 symptom features + Age + Gender.
NO blood tests needed - purely questionnaire-based prediction.

FEATURES (16 symptoms + 2 demographics = 18 total):
  Demographics: Age, Gender
  Symptoms: Polyuria, Polydipsia, sudden weight loss, weakness,
            Polyphagia, Genital thrush, visual blurring, Itching,
            Irritability, delayed healing, partial paresis,
            muscle stiffness, Alopecia, Obesity

TARGET: class (Positive = Diabetic, Negative = Non-Diabetic)

USAGE: venv\Scripts\python.exe train.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder

DATA_PATH = Path(r"d:\Miniproject\data\diabetes_data_upload.csv")
OUTPUT_DIR = Path(r"d:\Miniproject\model_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 65)
print("  DIABETES DETECTION (SYMPTOM-BASED) - TRAINING")
print("=" * 65)

# --- 1. Load & Encode ---
print("\n[1/7] Loading and encoding data...")
df = pd.read_csv(DATA_PATH)

# Encode target
df['diabetes'] = (df['class'] == 'Positive').astype(int)
df = df.drop(columns=['class'])

# Encode binary symptom columns (Yes=1, No=0)
binary_cols = [c for c in df.columns if c not in ['Age', 'Gender', 'diabetes']]
for col in binary_cols:
    df[col] = (df[col] == 'Yes').astype(int)

# Encode gender (Male=1, Female=0)
df['Gender'] = (df['Gender'] == 'Male').astype(int)

print(f"  Total samples: {len(df)}")
print(f"  Diabetic: {df['diabetes'].sum()} | Non-Diabetic: {(df['diabetes'] == 0).sum()}")

# --- 2. Features & Target ---
print("\n[2/7] Preparing features...")
X = df.drop(columns=['diabetes'])
y = df['diabetes']

feature_names = list(X.columns)
print(f"  Total features: {len(feature_names)}")
print(f"  Features: {feature_names}")

# --- 3. Train/Test Split ---
print("\n[3/7] Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
print(f"  Train distribution: {dict(y_train.value_counts())}")

# --- 4. No SMOTE needed (dataset is roughly balanced: 61%/39%) ---
print("\n[4/7] Class balance check...")
pos_pct = y.sum() / len(y) * 100
print(f"  Positive: {pos_pct:.1f}% | Negative: {100-pos_pct:.1f}%")
print(f"  Dataset is reasonably balanced - no SMOTE needed!")

# --- 5. Train Random Forest ---
print("\n[5/7] Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
print("  Model trained!")

# --- 6. Evaluate ---
print("\n[6/7] Evaluating model...")
print("=" * 65)

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"""
  RESULTS ON TEST SET ({len(X_test)} patients)
  -------------------------------------------
  Accuracy  : {acc:.2%}
  Precision : {prec:.2%}
  Recall    : {rec:.2%}
  F1 Score  : {f1:.2%}
  AUC-ROC   : {auc:.4f}
""")

print("  CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic']))

# Cross-validation
print("  5-FOLD CROSS VALIDATION:")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='f1')
print(f"  F1 scores: {[f'{s:.3f}' for s in cv_scores]}")
print(f"  Mean F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

cv_acc = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
print(f"  Mean Accuracy: {cv_acc.mean():.3f} (+/- {cv_acc.std():.3f})")

# --- 7. Save Plots & Model ---
print(f"\n[7/7] Saving outputs...")

# Confusion Matrix + ROC
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Non-Diabetic', 'Diabetic'],
            yticklabels=['Non-Diabetic', 'Diabetic'])
axes[0].set_xlabel('Predicted', fontsize=12)
axes[0].set_ylabel('Actual', fontsize=12)
axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
tn, fp, fn, tp = cm.ravel()
axes[0].text(0.5, -0.15,
    f"True Neg: {tn} | False Pos: {fp} | False Neg: {fn} | True Pos: {tp}",
    transform=axes[0].transAxes, ha='center', fontsize=10, style='italic')

fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[1].plot(fpr, tpr, color='#e74c3c', linewidth=2, label=f'Random Forest (AUC = {auc:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random Guess')
axes[1].fill_between(fpr, tpr, alpha=0.15, color='#e74c3c')
axes[1].set_xlabel('False Positive Rate', fontsize=12)
axes[1].set_ylabel('True Positive Rate', fontsize=12)
axes[1].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrix_roc.png', bbox_inches='tight', dpi=150)
plt.close()
print("  Saved: confusion_matrix_roc.png")

# Feature Importance
importances = rf.feature_importances_
feat_imp = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=True)

# Clean feature names for display
display_names = {
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
    'Gender': 'Gender'
}

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(feat_imp)))
display_labels = [display_names.get(f, f) for f in feat_imp['Feature']]
ax.barh(display_labels, feat_imp['Importance'], color=colors, edgecolor='black', alpha=0.85)
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Feature Importance - What Symptoms Matter Most',
             fontsize=14, fontweight='bold')
for i, (feat, imp) in enumerate(zip(display_labels, feat_imp['Importance'])):
    ax.text(imp + 0.003, i, f'{imp:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_importance.png', bbox_inches='tight', dpi=150)
plt.close()
print("  Saved: feature_importance.png")

# Save model
joblib.dump(rf, OUTPUT_DIR / 'diabetes_rf_model.pkl')
print("  Saved: diabetes_rf_model.pkl")

# Save feature importance CSV
feat_imp_sorted = feat_imp.sort_values('Importance', ascending=False)
feat_imp_sorted.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)
print("  Saved: feature_importance.csv")

# Save feature names for the prediction script
joblib.dump(feature_names, OUTPUT_DIR / 'feature_names.pkl')
print("  Saved: feature_names.pkl")

# Final summary
print("\n" + "=" * 65)
print("  TRAINING COMPLETE!")
print("=" * 65)
print(f"""
  Model: Random Forest (200 trees, max_depth=12)
  Features: {len(feature_names)} (16 symptoms + Age + Gender)
  NO blood tests required - symptom-based only!

  Key metrics:
    Accuracy  = {acc:.2%}
    F1 Score  = {f1:.2%}
    AUC-ROC   = {auc:.4f}

  Top 5 most important symptoms:
""")
for i, (_, row) in enumerate(feat_imp_sorted.head(5).iterrows()):
    name = display_names.get(row['Feature'], row['Feature'])
    print(f"    {i+1}. {name:25s} -> {row['Importance']:.4f}")

print(f"""
  All outputs saved to: {OUTPUT_DIR}
""")
print("=" * 65)
