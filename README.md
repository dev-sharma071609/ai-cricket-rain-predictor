# 🏏 AI Cricket Rain-Adjusted Score Predictor

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)]()
[![ML](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)]()
[![Status](https://img.shields.io/badge/Status-Live-success)]()

🔗 **Live App:**  
https://ai-cricket-rain-predictor-7cecgvcdwcrxxrlpztcuno.streamlit.app/

---

## 📌 Overview

This project predicts **first-innings cricket scores** for T20 and ODI matches using machine learning.

It also models **rain-shortened scenarios** and compares predictions against a simplified DLS-style baseline.

---

## ⚡ Features

- 🎯 Predicts:
  - Safe score
  - Expected score
  - Aggressive score
- 🌧️ Rain-adjusted predictions
- 📊 Visual score distributions
- 🧠 Match insights (momentum, pressure, phase)
- 📉 AI vs DLS comparison
- 🔄 What-if simulation (change overs dynamically)

---

## 🧠 Model Details

- Algorithm: **XGBoost Regressor**
- Data: Ball-by-ball cricket dataset
- Separate models:
  - T20 model
  - ODI model

### Input Features:
- Current score
- Wickets lost
- Balls remaining
- Run rate
- Recent momentum (last 6 & 12 balls)
- Phase indicators (death, middle, powerplay)

---


##  Run Locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py

 Author:
Dev Sharma
