![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Flask](https://img.shields.io/badge/Framework-Flask-black.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Enabled-green.svg)

# 📌 Multiple Disease Prediction System (ML + Flask Web App)

Predict **Heart Disease**, **Liver Disease**, **Kidney Disease**, and **Breast Cancer** using trained Machine Learning models integrated into a Flask-based web application.

---

## 🚀 Overview

This project is an end-to-end **Machine Learning + Web Application** designed to predict multiple diseases from clinical input features.

It includes:

- Complete **model training pipelines** in Jupyter notebooks  
- **Model comparison & best selection** for each disease  
- A Flask-based **interactive web interface**  
- Clean, responsive UI with symptom descriptions  
- Real-time predictions using serialized `.pkl` models  

### ⚠️ Disclaimer  
This tool is intended strictly for **educational and research purposes**.  
It must **not** be used for medical diagnosis or decision-making.

---

## 🧠 Supported Diseases & Best Models

| Disease                 | Best Model           | Test Accuracy | Notes                                  |
|------------------------|-----------------------|---------------|----------------------------------------|
| Heart Disease          | Logistic Regression   | ~88.52%       | Strong generalization, low overfitting |
| Liver Disease          | Logistic Regression   | ~73.50%       | Most stable across splits              |
| Chronic Kidney Disease | Decision Tree         | ~100%         | Dataset is highly separable            |
| Breast Cancer          | Random Forest         | ~97.37%       | Best precision & F1                    |

Each selected model is trained on full data and saved as:  


The Flask app loads these models during prediction.

---

## 📂 Project Structure

```text
multiple-disease-prediction/
│
├── App/
│   ├── app.py                     # Flask application
│   ├── heart_best_model.pkl       # Best Heart model
│   ├── liver_best_model.pkl       # Best Liver model
│   ├── kidney_best_model.pkl      # Best Kidney model
│   ├── cancer_best_model.pkl      # Best Cancer model
│
├── templates/                     # HTML pages (Jinja2)
│   ├── index.html                 # Home page
│   ├── heart.html
│   ├── liver.html
│   ├── kidney.html
│   ├── cancer.html
│   ├── predict.html
│   ├── result_history.html        # (DB logging disabled)
│
├── static/
│   ├── style.css                  # Modern CSS design
│   ├── main.js                    # Menu + loading animation
│   └── images/                    # Icons / UI assets
│
├── Notebooks/                     # Full ML workflows
│   ├── Heart.ipynb
│   ├── Liver.ipynb
│   ├── Kidney.ipynb
│   └── Cancer.ipynb
│
├── Dataset/                       # Raw datasets
│   ├── heart.csv
│   ├── kidney_disease.csv
│   ├── indian_liver_patient.csv
│   └── cancer.csv
│
├── requirements.txt               # Dependencies
├── Procfile                       # Deployment (Gunicorn)
├── README.md
├── LICENSE
└── venv/                          # Local virtual environment (ignored)

✨ Features
🔸 Machine Learning

Comprehensive model comparison (LogReg, SVM, KNN, Random Forest, XGBoost…)

Confusion matrices for error analysis

ROC curves + AUC scoring

Train vs Test bias–variance analysis

Automatic best model selection

Deployment-ready .pkl models

🔸 Flask Web Application

Responsive, modern UI

Disease information cards with icons

Clean input forms with validation

Real-time predictions with styled output

Mobile-friendly navigation

Loading animation during model prediction

🔸 UI & UX Highlights

Professional layout for educational or demo settings

Centered cards and structured forms

Symptom sections for clarity

Color-coded prediction output (Green = Safe, Red = Risk)

📊 ML Development Workflow

Each notebook follows a complete workflow:

Load dataset

Handle missing values

Clean & preprocess features

Define multiple candidate models

Train models & compute metrics

Visualize performance:

Accuracy

Precision

Recall

F1 Score

Confusion matrices

ROC curves

Compare overfitting/underfitting

Select the best model

Save model to .pkl for deployment

🖼️ Screenshots
<p align="center"> <img src="Screenshot (106).png" alt="Home page" width="45%"> <img src="Screenshot (107).png" alt="Prediction Form" width="45%"> </p> <p align="center"> <img src="Screenshot (108).png" alt="Result Page" width="45%"> <img src="Screenshot (109).png" alt="Additional Form" width="45%"> </p>
⚙️ Setup & Installation
1️⃣ Clone the repository
git clone https://github.com/shahid-iqbal-er/multiple-disease-prediction.git
cd multiple-disease-prediction

2️⃣ Create and activate a virtual environment
python -m venv venv

Windows:
.\venv\Scripts\Activate.ps1

Linux/Mac:
source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the application
cd App
python app.py


Visit in your browser:

http://127.0.0.1:5000

🛠️ Technologies Used
Machine Learning

Python (NumPy, Pandas)

scikit-learn

XGBoost

Matplotlib

Seaborn

Pipelines & Imputation

Web Development

Flask

HTML5, CSS3, JavaScript

Font Awesome

Responsive UI

⚠️ Important Disclaimer

This project is not a medical device.
Predictions are based on ML models trained on publicly available datasets and should never replace professional medical advice.

📜 License

This project is licensed under the MIT License — see the LICENSE
 file for details.

🙌 Acknowledgements

UCI Machine Learning Repository

Kaggle datasets

scikit-learn & XGBoost communities

Flask documentation

Developed by Shahid Iqbal (2025)
