# 🏏 IPL Team Score Prediction App

A Machine Learning-powered web application that predicts the final score of an IPL team based on match conditions such as overs, wickets, run rate, and player form. Built using Python and Streamlit for interactive, real-time use.

---

## 🚀 Features

- ⚡ Predict final scores based on live or hypothetical match inputs
- 📊 Interactive and user-friendly UI using Streamlit
- 🤖 Powered by trained ML models (Linear Regression / Random Forest)
- 📈 Visual insights using Matplotlib and Seaborn
- 📦 Clean data pipeline for consistent predictions

---

## 🧠 How It Works

This app uses historical IPL data to train a machine learning model. By inputting details like current over, wickets fallen, and run rate, the model predicts the most probable final score for the batting team.

---

## 🛠 Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python, Pandas, NumPy  
- **Machine Learning**: Scikit-learn (Regression Models)  
- **Visualization**: Matplotlib, Seaborn  
- **IDE**: Visual Studio Code  
- **Deployment**: Streamlit Cloud / Localhost  

---

## 📂 Project Structure

├── app.py # Streamlit main application file
├── model.pkl # Trained ML model
├── IPL_data.csv # Cleaned IPL dataset
├── requirements.txt # Required Python packages
├── README.md # Project documentation

yaml
Copy
Edit

---

## ▶️ How to Run Locally

1. **Clone the repo**

```bash
git clone https://github.com/your-username/IPL-Score-Predictor.git
cd IPL-Score-Predictor
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the app

bash
Copy
Edit
streamlit run app.py
📈 Model Details
Trained on IPL data from past seasons

Models used: Linear Regression, Random Forest Regressor

Evaluated using MAE, RMSE, and R² Score

💡 Use Cases
Fantasy League Strategy

Sports Analytics Projects

Cricket Match Commentary & Insights

Data Science Learning Projects

