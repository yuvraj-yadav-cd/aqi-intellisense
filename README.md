# 🌫️ AQI IntelliSense X

<div align="center">

## Enterprise-Grade AI Air Quality Forecasting Platform

Predict future Air Quality Index (AQI) levels for major Indian cities using Deep Learning, interactive dashboards, and cloud deployment.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📌 Overview

AQI IntelliSense X is a production-ready machine learning web application that forecasts future air quality levels using trained LSTM neural networks.

The platform uses historical AQI trends, engineered time-series features, and city-specific trained models to predict hourly AQI for multiple Indian cities.

### Core Highlights

- 📊 Forecast analytics  
- 🤖 AI prediction models  
- 🌆 Multi-city intelligence  
- 📈 Interactive dashboards  
- ☁️ Cloud deployment  

---

## 🚀 Live Demo

🔗 **Deployed App:** [Aqi-Intellisense](https://aqi-intellisense.streamlit.app)

---

## 🌍 Supported Cities

- Delhi  
- Mumbai  
- Bengaluru  

### Expansion Ready

- Pune  
- Chennai  
- Kolkata  
- Hyderabad  

---

## ✨ Key Features

### 🤖 AI Forecasting Engine

- LSTM Deep Learning architecture
- Multi-step AQI forecasting
- Recursive prediction engine
- City-specific trained models

### 📈 Interactive Dashboard

- Premium enterprise UI
- Forecast line chart
- AQI Gauge meter
- Summary KPI cards
- City switching controls

### 📋 Forecast Export

- Download forecast as CSV
- Clean prediction tables
- Forecast comparison ready

### ☁️ Production Deployment

- Streamlit Cloud hosting
- GitHub integrated deployment
- Public access web application

---

## 🧠 Machine Learning Pipeline

### Data Sources

Historical hourly AQI datasets for each city.

### Data Preprocessing

- Missing value handling
- Datetime parsing
- Feature engineering
- Normalization using MinMaxScaler

### Engineered Features

```text
AQI
Hour
Day of Week
Month
Weekend Flag
Lag1
Lag2
Lag24
Rolling Mean (3h, 6h, 24h)
Difference Features
Sin/Cos Hour Encoding
```

### Model Architecture

```python
Sequential([
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
```

### Evaluation Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

## 📊 Sample Performance

| City | Accuracy Level | Notes |
|------|----------------|------|
| Delhi | High | Strong urban AQI trend learning |
| Mumbai | Excellent | R² ≈ 0.95 |
| Bengaluru | High | Stable forecast performance |

---

## 🛠️ Tech Stack

### Frontend

- Streamlit

### Machine Learning

- TensorFlow  
- Keras  
- Scikit-learn  

### Data Processing

- Pandas  
- NumPy  

### Visualization

- Plotly

### Deployment

- Streamlit Community Cloud

---

## 📁 Project Structure

```text
aqi-intellisense/
│── app.py
│── requirements.txt
│── runtime.txt
│── README.md
│
├── models/
│   ├── delhi_weights.weights.h5
│   ├── delhi_scaler.pkl
│   ├── delhi_last_sequence.npy
│   ├── mumbai_weights.weights.h5
│   ├── mumbai_scaler.pkl
│   ├── mumbai_last_sequence.npy
│   ├── bengaluru_weights.weights.h5
│   ├── bengaluru_scaler.pkl
│   └── bengaluru_last_sequence.npy
│
├── notebooks/
│   ├── 01_delhi_training.ipynb
│   ├── 02_mumbai_training.ipynb
│   └── 03_bengaluru_training.ipynb
```

---

## ⚙️ Installation Guide

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/aqi-intellisense.git
cd aqi-intellisense
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Locally

```bash
streamlit run app.py
```

---

## 📸 Screenshots

### Dashboard
<img width="895" height="581" alt="Screenshot 2026-04-30 112344" src="https://github.com/user-attachments/assets/11e328a1-7adb-4e3e-997f-cabda2e20dfc" />

### Forecast Graph

<img width="760" height="546" alt="Screenshot 2026-04-30 112313" src="https://github.com/user-attachments/assets/b2322a3a-986c-45fd-bb97-90152f0ea902" />

### Gauge Meter

<img width="960" height="578" alt="Screenshot 2026-04-30 111933" src="https://github.com/user-attachments/assets/33917f64-c3c5-4635-8c07-3a7e14e12855" />

---

## 🎯 Real-World Use Cases

- Smart city monitoring
- Pollution trend forecasting
- Public awareness dashboards
- Academic AI projects
- Environmental analytics

---

## 👨‍💻 Author

**Yuvraj Yadav**

---

## 🤝 Contributions

Contributions, ideas, and improvements are welcome.

---

## 📜 License

MIT License

---

<div align="center">

## ⭐ If you found this project useful, star the repository.

</div>
