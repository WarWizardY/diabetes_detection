# 🩺 Diabetes Detection Using Random Forest (Symptom-Based)

A machine learning system that predicts diabetes risk using only symptoms and basic demographics — **no blood tests or clinic visits required**.
Built using a **Random Forest Classifier** trained on the *UCI Early Stage Diabetes Risk Prediction Dataset*.

---

## 📋 Problem Statement

In many communities, especially rural and resource-limited areas, access to diagnostic tests like HbA1c is limited.

This project provides a **symptom-based screening tool** that predicts diabetes risk using simple **Yes/No responses**, enabling:

* Early detection
* Low-cost screening
* No dependence on clinical infrastructure

---

## 🎯 How It Works

The system asks **16 simple Yes/No questions** about symptoms such as:

* Frequent urination, excessive thirst, sudden weight loss
* Blurred vision, fatigue, slow-healing wounds
* Itchy skin, muscle weakness, irritability

The trained model processes these inputs and outputs:

* ✅ Diabetes risk prediction (Positive / Negative)
* 📊 Confidence score

---

## 📊 Model Performance

| Metric       | Score         |
| ------------ | ------------- |
| Accuracy     | 98.08%        |
| Precision    | 100.00%       |
| Recall       | 96.88%        |
| F1 Score     | 98.41%        |
| AUC-ROC      | 1.0000        |
| Cross-Val F1 | 0.978 ± 0.014 |

---

## 🔬 Top Predictive Features

| Rank | Symptom                       | Importance |
| ---- | ----------------------------- | ---------- |
| 1    | Excessive Thirst (Polydipsia) | 21.95%     |
| 2    | Frequent Urination (Polyuria) | 20.54%     |
| 3    | Gender                        | 11.01%     |
| 4    | Age                           | 8.49%      |
| 5    | Sudden Weight Loss            | 7.27%      |

---

## 📁 Project Structure

```
diabetes-detection/
│
├── data/
│   ├── diabetes_data_upload.csv
│   └── diabetes.csv
│
├── eda_outputs/
│   ├── target_distribution.png
│   ├── symptom_prevalence.png
│   ├── correlation_heatmap.png
│   └── diabetes_by_gender.png
│
├── model_outputs/
│   ├── diabetes_rf_model.pkl
│   ├── feature_names.pkl
│   ├── feature_importance.png
│   ├── feature_importance.csv
│   └── confusion_matrix_roc.png
│
├── eda.py
├── train.py
├── predict.py
├── import_data.py
├── generate_report.py
└── requirements.txt
```

---

## 🚀 Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/WarWizardY/diabetes_detection.git
cd diabetes_detection
```

---

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run EDA

```bash
python eda.py
```

---

### 5. Train the model

```bash
python train.py
```

---

### 6. Run prediction demo

```bash
python predict.py
```

---

## 🖥️ Prediction Demo

The system provides two modes:

* **Manual Input Mode**
  Answer Yes/No symptom-based questions

* **Demo Mode**
  Run predefined cases:

  * Healthy
  * High-risk
  * Borderline
  * Classic diabetic

👉 No blood tests required — just symptom inputs → instant prediction.

---

## 📦 Dataset

**UCI Early Stage Diabetes Risk Prediction Dataset**

* Source: Sylhet Diabetes Hospital, Bangladesh
* Method: Direct patient questionnaires
* Size: 520 patients
* Features: 16 symptoms + Age + Gender
* Target: Positive / Negative
* License: CC BY 4.0

---

## 🛠️ Tech Stack

* Python 3.10+
* scikit-learn (Random Forest Classifier)
* pandas, numpy
* matplotlib, seaborn
* imbalanced-learn (SMOTE)
* fpdf2 (PDF report generation)
* joblib (model persistence)

---
