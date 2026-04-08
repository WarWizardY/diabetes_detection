"""
Diabetes Detection (Symptom-Based) — EDA
==========================================
Analyzes the UCI Early Stage Diabetes Risk Prediction Dataset.
520 patients, 16 symptom features, NO blood tests required.

USAGE: venv\Scripts\python.exe eda.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Config
DATA_PATH = Path(r"d:\Miniproject\data\diabetes_data_upload.csv")
OUTPUT_DIR = Path(r"d:\Miniproject\eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['figure.dpi'] = 120

print("=" * 65)
print("  DIABETES DETECTION (SYMPTOM-BASED) - EDA")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
print(f"\nDataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# --- 1. Basic Info ---
print("\n" + "-" * 65)
print("  DATA TYPES & MISSING VALUES")
print("-" * 65)
print(df.dtypes.to_string())
missing = df.isnull().sum()
print(f"\nMissing values: {missing.sum()} total")

# --- 2. Target Distribution ---
print("\n" + "-" * 65)
print("  TARGET: DIABETES CLASS")
print("-" * 65)
target_counts = df['class'].value_counts()
for label, count in target_counts.items():
    pct = count / len(df) * 100
    print(f"  {label}: {count} ({pct:.1f}%)")

# --- 3. Encode for analysis ---
df_encoded = df.copy()
binary_cols = [c for c in df.columns if c not in ['Age', 'Gender', 'class']]
for col in binary_cols:
    df_encoded[col] = (df_encoded[col] == 'Yes').astype(int)
df_encoded['Gender'] = (df_encoded['Gender'] == 'Male').astype(int)
df_encoded['class_bin'] = (df_encoded['class'] == 'Positive').astype(int)

# --- 4. Target Distribution Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = ['#2ecc71', '#e74c3c']
axes[0].bar(['Negative\n(No Diabetes)', 'Positive\n(Diabetic)'],
            [target_counts.get('Negative', 0), target_counts.get('Positive', 0)],
            color=colors[::-1], edgecolor='black', alpha=0.85)
axes[0].set_title('Diabetes Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count')
for i, (label, count) in enumerate(zip(['Negative', 'Positive'],
                                        [target_counts.get('Negative', 0), target_counts.get('Positive', 0)])):
    axes[0].text(i, count + 5, str(count), ha='center', fontweight='bold', fontsize=13)

# Age distribution by class
for cls, color, label in zip(['Positive', 'Negative'], ['#e74c3c', '#2ecc71'], ['Diabetic', 'Non-Diabetic']):
    subset = df[df['class'] == cls]['Age']
    axes[1].hist(subset, bins=15, alpha=0.6, label=label, color=color, edgecolor='black')
axes[1].set_title('Age Distribution by Diabetes Status', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'target_distribution.png', bbox_inches='tight')
plt.close()
print("\nSaved: target_distribution.png")

# --- 5. Symptom Prevalence by Diabetes Status ---
symptom_cols = [c for c in df.columns if c not in ['Age', 'Gender', 'class']]

pos_rates = []
neg_rates = []
for col in symptom_cols:
    pos_rate = (df[df['class'] == 'Positive'][col] == 'Yes').mean() * 100
    neg_rate = (df[df['class'] == 'Negative'][col] == 'Yes').mean() * 100
    pos_rates.append(pos_rate)
    neg_rates.append(neg_rate)

fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(symptom_cols))
width = 0.35
bars1 = ax.barh(x + width/2, pos_rates, width, label='Diabetic', color='#e74c3c', alpha=0.8, edgecolor='black')
bars2 = ax.barh(x - width/2, neg_rates, width, label='Non-Diabetic', color='#2ecc71', alpha=0.8, edgecolor='black')

ax.set_yticks(x)
ax.set_yticklabels([c.replace('_', ' ').title() for c in symptom_cols], fontsize=10)
ax.set_xlabel('Prevalence (%)', fontsize=12)
ax.set_title('Symptom Prevalence: Diabetic vs Non-Diabetic Patients', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.invert_yaxis()

# Value labels
for bar, val in zip(bars1, pos_rates):
    ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.0f}%', va='center', fontsize=8)
for bar, val in zip(bars2, neg_rates):
    ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.0f}%', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'symptom_prevalence.png', bbox_inches='tight')
plt.close()
print("Saved: symptom_prevalence.png")

# --- 6. Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(14, 10))
corr_cols = ['Age', 'Gender'] + symptom_cols + ['class_bin']
corr_labels = [c.replace('_', ' ').replace('class_bin', 'Diabetes').title() for c in corr_cols]
corr_matrix = df_encoded[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, ax=ax, linewidths=0.5, cbar_kws={'shrink': 0.8},
            xticklabels=corr_labels, yticklabels=corr_labels)
ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("Saved: correlation_heatmap.png")

# --- 7. Correlation with target ---
print("\n" + "-" * 65)
print("  CORRELATION WITH DIABETES (Target)")
print("-" * 65)
corr_with_target = df_encoded[['Age', 'Gender'] + symptom_cols].corrwith(df_encoded['class_bin']).sort_values(ascending=False)
for feat, corr in corr_with_target.items():
    bar = "|" * int(abs(corr) * 30)
    sign = "+" if corr > 0 else "-"
    name = feat.replace('_', ' ').title()
    print(f"  {name:25s}: {corr:+.4f}  {sign}{bar}")

# --- 8. Gender vs Diabetes ---
fig, ax = plt.subplots(figsize=(8, 6))
gender_diabetes = df.groupby(['Gender', 'class']).size().unstack(fill_value=0)
gender_diabetes.plot(kind='bar', stacked=True, ax=ax, color=['#2ecc71', '#e74c3c'], edgecolor='black')
ax.set_title('Diabetes Distribution by Gender', fontsize=14, fontweight='bold')
ax.set_xlabel('Gender')
ax.set_ylabel('Count')
ax.legend(['Negative', 'Positive'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'diabetes_by_gender.png', bbox_inches='tight')
plt.close()
print("Saved: diabetes_by_gender.png")

# --- 9. Top symptom combinations ---
print("\n" + "-" * 65)
print("  TOP SYMPTOM COMBINATIONS IN DIABETIC PATIENTS")
print("-" * 65)
diabetic = df[df['class'] == 'Positive']
symptom_counts = diabetic[symptom_cols].apply(lambda x: (x == 'Yes').sum())
top_symptoms = symptom_counts.sort_values(ascending=False)
for sym, cnt in top_symptoms.items():
    pct = cnt / len(diabetic) * 100
    name = sym.replace('_', ' ').title()
    print(f"  {name:25s}: {cnt:3d} patients ({pct:.0f}%)")

# --- 10. Summary ---
print("\n" + "=" * 65)
print("  EDA SUMMARY")
print("=" * 65)
pos_count = target_counts.get('Positive', 0)
neg_count = target_counts.get('Negative', 0)
print(f"""
  DATASET: UCI Early Stage Diabetes Risk Prediction
  Source: Sylhet Diabetes Hospital, Bangladesh (questionnaire-based)
  Total patients: {len(df)}
  Diabetic (Positive): {pos_count} ({pos_count/len(df)*100:.1f}%)
  Non-Diabetic (Negative): {neg_count} ({neg_count/len(df)*100:.1f}%)

  FEATURES: 16 symptom-based (Yes/No) + Age + Gender
  NO blood tests or lab results required!

  TOP PREDICTIVE SYMPTOMS (by correlation):
    1. Polyuria (frequent urination)
    2. Polydipsia (excessive thirst)
    3. Gender (males slightly higher risk)
    4. Sudden weight loss
    5. Partial paresis (muscle weakness)

  All plots saved to: {OUTPUT_DIR}
""")
print("=" * 65)
