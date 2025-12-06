![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Flask](https://img.shields.io/badge/Framework-Flask-black.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Enabled-green.svg)

📌 Multiple Disease Prediction System (ML + Flask Web App)

Predict Heart Disease, Liver Disease, Kidney Disease, and Breast Cancer using trained Machine Learning models and a Flask-based web interface.

🚀 Overview

This project is an end-to-end Machine Learning and Web Application system that predicts the likelihood of multiple diseases based on clinical input features.

It includes:

Complete model training pipelines in Jupyter notebooks

Best model selection for each disease

A Flask web application for user interaction

A clean, responsive UI with symptom descriptions and model performance

Real-time ML predictions using .pkl models

⚠️ Disclaimer:
This tool is intended strictly for educational and research purposes.
It must not be used for professional diagnosis or medical decision-making.

🧠 Supported Diseases & Best Models
Disease	Best Model	Test Accuracy	Notes
Heart Disease	Logistic Regression	~88.52%	Strong generalization, low overfitting
Liver Disease	Logistic Regression	~73.50%	Most stable across splits
Chronic Kidney Disease	Decision Tree	~100%	Dataset very separable
Breast Cancer	Random Forest	~97.37%	Best precision & F1

Each best-performing model is trained on full data and saved as
<disease>_best_model.pkl, which the Flask app loads for prediction.

📂 Project Structure
multiple-disease-prediction-final/
│
├── App/
│   ├── app.py                         # Flask application
│   ├── heart_best_model.pkl           # Best Heart model
│   ├── liver_best_model.pkl           # Best Liver model
│   ├── kidney_best_model.pkl          # Best Kidney model
│   ├── cancer_best_model.pkl          # Best Cancer model
│
├── templates/
│   ├── index.html                     # Home page
│   ├── heart.html
│   ├── liver.html
│   ├── kidney.html
│   ├── cancer.html
│   ├── predict.html
│   ├── result_history.html            # History disabled (no DB)
│
├── static/
│   ├── style.css                      # Custom CSS (modern UI)
│   ├── main.js                        # Mobile menu + prediction loading state
│   ├── images (optional)              # Icons, logos, backgrounds
│
├── Notebooks/                         # Full ML workflow notebooks
│   ├── Heart.ipynb
│   ├── Liver.ipynb
│   ├── Kidney.ipynb
│   ├── Cancer.ipynb
│
├── Dataset/                           # Raw datasets for ML
│   ├── heart.csv
│   ├── kidney_disease.csv
│   ├── indian_liver_patient.csv
│   ├── cancer.csv
│
├── requirements.txt                   # Python dependencies
├── README.md                          # (this file)
└── venv/                              # Virtual environment (ignored on GitHub)

✨ Features
🔸 Machine Learning

Complete model comparison for each disease (LogReg, SVM, KNN, RF, XGBoost, etc.)

Confusion matrices

ROC curves + AUC metrics

Bias–variance analysis (Train vs Test gap)

Best model auto-selection

Deployment-ready .pkl models

🔸 Web Application (Flask)

Responsive UI built with HTML + CSS + JS + Font Awesome

Mobile-friendly navbar

Disease information cards with icons

Input validation

Beautiful result screen with clear health guidance

Loading animation on prediction

🔸 Clean UI Highlights

Professional layout

Centered cards and forms

Animated predict button

Color-coded results (green = healthy, red = risk)

📊 Model Development Workflow

Each disease notebook follows:

Data loading

Missing value handling

Feature preprocessing

Model zoo definition

Training & evaluation

Comparison plots:

Accuracy

Precision

Recall

F1 Score

Confusion matrices

ROC curves

Best model selection

Saving model for deployment

🖥️ Screenshots

![alt text](<Screenshot (106).png>) ![alt text](<Screenshot (107).png>) ![alt text](<Screenshot (108).png>) ![alt text](<Screenshot (109).png>)

/screenshots
    home.png
    heart_form.png
    prediction_result.png

⚙️ Setup & Installation
1️⃣ Clone the repository
git clone https://github.com/yourusername/multiple-disease-prediction.git
cd multiple-disease-prediction-final

2️⃣ Create and activate virtual environment
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


Open in your browser:

http://127.0.0.1:5000

🛠️ Technologies Used
Machine Learning

Python (NumPy, Pandas)

scikit-learn

XGBoost

Matplotlib & Seaborn

Pipelines + Imputation

Web Development

Flask

HTML5 / CSS3 / JavaScript

Font Awesome

Responsive layout

⚠️ Important Disclaimer

This project is not a medical device.
Predictions are based on machine learning models trained on publicly available datasets and should never be used for clinical decision-making.

📜## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.



🙌 Acknowledgements

UCI Machine Learning Repository

Kaggle datasets for disease prediction

scikit-learn & XGBoost communities

Flask documentation
