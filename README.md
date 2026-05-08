# 🏠 House Price Prediction — Production ML System

An end-to-end Machine Learning system predicting California house prices using Random Forest. Built following **Aurélien Géron's "Hands-On Machine Learning" Chapter 2** and deployed to production.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B)
![Render](https://img.shields.io/badge/Render-Deployed-46E3B7)

## 🌐 Live Demo

| Component | URL |
|-----------|-----|
| 🖥️ Web App | [house-price-prediction.streamlit.app](https://house-price-prediction-mlops-eleqsctggryvmy9hqyjevp.streamlit.app) |
| ⚙️ REST API | [house-price-api-9hmc.onrender.com](https://house-price-api-9hmc.onrender.com) |
| 📖 API Docs | [/docs](https://house-price-api-9hmc.onrender.com/docs) |

## 🏗️ System Architecture

```Raw California Housing Data (1990 Census)
↓
Data Cleaning & EDA
(Removed capped values, handled missing data)
↓
Feature Engineering
(rooms_per_household, bedrooms_per_room, population_per_household)
↓
Model Training
(Random Forest → 83% R², ~$47K RMSE)
↓
FastAPI REST API  ←→  Streamlit Web App
(Render)               (Streamlit Cloud)```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| R² Score | 83% |
| Models Compared | Linear Regression, Decision Tree, Random Forest |
| Tuning Method | GridSearchCV |
| Target Transform | Log transformation (fixes skewness) |

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | Scikit-learn, Pandas, NumPy |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit |
| API Hosting | Render |
| Frontend Hosting | Streamlit Cloud |
| Version Control | Git + GitHub |

## 🔍 Key Findings

- **Median income** is the strongest predictor of house price (27% feature importance)
- **Ocean proximity** (INLAND vs coastal) is the second most important feature (21.6%)
- **Log transforming** the target variable significantly improved model performance
- Engineered features (`rooms_per_household`, `bedrooms_per_room`) outperformed raw counts

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/SahilDogra23/House-Price-Prediction-MLOps.git
cd House-Price-Prediction-MLOps
pip install -r requirements.txt
```

### 2. Start the API
```bash
uvicorn app:app --reload
```

### 3. Start the frontend (new terminal)
```bash
streamlit run streamlit_app.py
```

## 🐳 Docker Deployment

### Build the image
```bash
docker build -t house-price-api .
```

### Run the container
```bash
docker run -p 8000:8000 house-price-api
```

### Access the API
Visit `http://127.0.0.1:8000/docs`

### Pull from Docker Hub
```bash
docker pull sahil2323dogra/house-price-api:v1
docker run -p 8000:8000 sahil2323dogra/house-price-api:v1
```

## 📁 Project Structure

```House_Price_Prediction/
├── data/
│   └── housing.csv                ← California Housing dataset
├── notebooks/
│   └── house_price.ipynb          ← EDA, training, evaluation
├── app.py                         ← FastAPI backend
├── streamlit_app.py               ← Streamlit frontend
├── Procfile                       ← Render deployment config
├── requirements.txt
└── README.md```

## 📚 Reference
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* — Chapter 2
- Dataset: [California Housing Prices — Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

## 👤 Author
**Sahil Dogra**
[![GitHub](https://img.shields.io/badge/GitHub-SahilDogra23-black)](https://github.com/SahilDogra23)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-sahildogra23-blue)](https://www.linkedin.com/in/sahildogra23)