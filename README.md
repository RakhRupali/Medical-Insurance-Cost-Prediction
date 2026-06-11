# 🏥 Medical Insurance Cost Prediction

A Machine Learning web app built with Streamlit to predict medical insurance costs based on patient data.



## 📖 Overview
This project analyzes individual healthcare data to understand how factors like age, BMI, smoking habits, and region impact medical insurance costs. It includes interactive visualizations and an ML-based prediction tool.

## 🎯 Project Goals
- Explore key drivers of insurance charges
- Visualize data insights interactively
- Predict insurance cost for new patients using Machine Learning

## 📊 Dataset
- 1338 records
- Features: age, sex, bmi, children, smoker, region
- Target: charges (insurance cost in USD)

## 🔍 Features
- 18 EDA visual questions with interactive charts
- Plotly and Seaborn visualizations
- ML-based cost prediction using Random Forest
- Clean UI with sidebar navigation

## 🤖 Model Performance
- Algorithm: Random Forest Regressor
- Features used: age, sex, bmi, children, smoker, region
- Model saved as .pkl file using Pickle

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (Random Forest Regressor)
- Streamlit, Plotly, Matplotlib, Seaborn
- Pickle (model saving)

## 📁 Project Structure
- app.py — Main Streamlit app
- creat_model.py — Model training script
- medical_insurance_model.pkl — Trained model
- requirements.txt — Required libraries

## 🚀 How to Run
pip install -r requirements.txt
streamlit run app.py

## 👤 Author
**Rupali Kendre**
- 📍 Nanded, Maharashtra, India
- 💼 LinkedIn: https://www.linkedin.com/in/rupali-kendre-1680b3386/
- 🐙 GitHub: https://github.com/RakhRupali