# ☀️ Solar Generation Forecasting Microservice

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E.svg)

## 📌 Project Overview
This project is a production-ready machine learning microservice designed to predict solar power generation (in Megawatts) for a microgrid. By analyzing 24-hour meteorological forecasts, the AI model allows grid operators to anticipate power supply, optimize battery storage, and prevent grid instability.

The model was trained on real-world historical data from the Liege microgrid, capturing complex relationships between time of day, surface temperature, cloud cover, and solar irradiance.

## 🚀 Architecture & Tech Stack
*   **API Framework:** FastAPI (Python)
*   **Machine Learning:** Scikit-Learn (Random Forest Regressor)
*   **Data Processing:** Pandas
*   **Data Visualization:** Matplotlib, Requests
*   **Server:** Uvicorn

## ⚙️ Features
*   **RESTful API:** Exposes a POST endpoint (`/api/v1/forecast`) for submitting weather payloads.
*   **Data Validation:** Utilizes Pydantic to ensure all incoming API requests are strictly structured.
*   **Real-Time Inference:** Generates instant Megawatt output predictions for every hour of a 24-hour cycle.
*   **Dynamic Visualization:** Includes a client-side script that fetches live API data and generates a professional business chart.

## 💻 How to Run Locally

### 1. Install Dependencies
Ensure you have Python installed, then install the required libraries:
```bash
pip install fastapi uvicorn pandas scikit-learn matplotlib requests