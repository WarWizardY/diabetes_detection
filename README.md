🩺 Diabetes Detection Using Random Forest (Symptom-Based)
A machine learning system that predicts diabetes risk using only symptoms and demographics — no blood tests or clinic visits required. Built with a Random Forest Classifier trained on the UCI Early Stage Diabetes Risk Prediction Dataset.

📋 Problem Statement
In many communities, especially rural areas, access to advanced lab tests like HbA1c is limited. This project builds a symptom-based diabetes screening tool that can predict whether a person is at risk of diabetes by simply answering Yes/No questions about their symptoms — enabling early detection without clinical infrastructure.
🎯 How It Works
A patient answers 16 simple Yes/No questions about symptoms like:
Frequent urination, excessive thirst, sudden weight loss
Blurred vision, slow-healing wounds, fatigue
Itchy skin, muscle weakness, irritability
The trained AI model analyzes these responses and outputs a diabetes risk prediction with a confidence score.
📊 Model Performance
Metric
Score
Accuracy
98.08%
Precision
100.00%
Recall
96.88%
F1 Score
98.41%
AUC-ROC
1.0000
Cross-Val F1
0.978 ± 0.014

🔬 Top Predictive Features
Rank
Symptom
Importance
1
Excessive Thirst (Polydipsia)
21.95%
2
Frequent Urination (Polyuria)
20.54%
3
Gender
11.01%
4
Age
8.49%
5
Sudden Weight Loss
7.27%

📁 Project Structure
├── data/
│   ├── diabetes_data_upload.csv    # UCI symptom-based dataset (520 patients)
│   └── diabetes.csv                # Original clinical dataset (reference)
├── eda_outputs/                    # EDA plots and visualizations
│   ├── target_distribution.png
│   ├── symptom_prevalence.png
│   ├── correlation_heatmap.png
│   └── diabetes_by_gender.png
├── model_outputs/                  # Trained model and results
│   ├── diabetes_rf_model.pkl       # Saved Random Forest model
│   ├── feature_names.pkl           # Feature order for prediction
│   ├── feature_importance.png
│   ├── feature_importance.csv
│   └── confusion_matrix_roc.png
├── eda.py                          # Exploratory Data Analysis script
├── train.py                        # Model training script
├── predict.py                      # Interactive prediction demo
├── import_data.py                  # Dataset download script
├── generate_report.py              # PDF report generator
└── requirements.txt                # Python dependencies
🚀 Setup & Usage
1. Clone the repository
bash
git clone https://github.com/YOUR_USERNAME/diabetes-detection.git
cd diabetes-detection
2. Create virtual environment
bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
3. Install dependencies
bash
pip install -r requirements.txt
4. Run EDA
bash
python eda.py
5. Train the model
bash
python train.py
6. Run the prediction demo
bash
python predict.py
🖥️ Prediction Demo
The interactive demo offers two modes:
Option 1 — Enter patient symptoms manually (Yes/No questions)
Option 2 — Run pre-built demo cases (healthy, high-risk, borderline, classic signs)
No blood tests needed — just answer symptom questions and get an instant risk prediction!
📦 Dataset
UCI Early Stage Diabetes Risk Prediction Dataset
Source: Sylhet Diabetes Hospital, Bangladesh
Collection: Direct questionnaires from patients
Size: 520 patients, 16 symptom features + Age + Gender
Target: Positive (Diabetic) / Negative (Non-Diabetic)
License: CC BY 4.0
UCI Repository Link
🛠️ Tech Stack
Python 3.10+
scikit-learn — Random Forest Classifier
pandas / numpy — Data processing
matplotlib / seaborn — Visualization
imbalanced-learn — SMOTE (for clinical dataset)
fpdf2 — PDF report generation
joblib — Model persistence
This project is for educational purposes (college mini project). Dataset licensed under CC BY 4.0.

