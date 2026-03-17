# 🏏 AI Cricket Rain-Adjusted Score Predictor

[Live App](https://ai-cricket-rain-predictor-ysgsuwjyu9tty633lym9yb.streamlit.app/)

![App Screenshot]([live app screenshot 1.png](url))
([[live app screenshot 2.png](url)])
([[live app screenshot 3.png](url)])
([[live app screenshot 4.png](url)])
([[live app screenshot 5.png](url)])
An AI-based T20 cricket score prediction system built using ball-by-ball international cricket data.  
The model predicts safe, expected, and aggressive score ranges and estimates revised totals when rain shortens the innings.
---

##  Features

-  Predicts **safe, expected, and aggressive** final scores  
-  Handles **rain-shortened matches**  
-  Compares AI predictions with a **DLS-style baseline**  
-  Uses **real ball-by-ball international cricket data**  
-  Captures **wickets, momentum, and match context**

---

## Model Approach

### Dataset
- Source: Cricsheet ball-by-ball data  
- Matches: T20 internationals  
- Filtered:
  - First innings only  
  - Removed first 2 overs (noise reduction)

### Target
- Predicts: **runs remaining**
- Final score = current score + predicted runs

### Model
- Algorithm: **XGBoost Regressor**
- Final Performance:
  - **MAE: 16.12**

---

## Input Features

- current score  
- wickets lost  
- wickets in hand  
- balls remaining  
- overs remaining  
- current run rate  
- runs in last 6 balls  
- wickets in last 6 balls  
- runs in last 12 balls  
- wickets in last 12 balls  
- aggression index  
- death overs flag  
- middle overs flag  
- powerplay flag  
- required attack  
- scoring momentum  

---

##  Example

**Input:**
- Score: 92/3  
- Overs: 12.0  
- Rain reduces innings to 16 overs  

**Output:**
- Normal range: **165 / 167 / 170**  
- Rain range: **136 / 138 / 144**  
- Suggested target: **139**  
- AI vs DLS baseline: **+15 runs**

---

##  Streamlit App

The project includes a web app where you can:

- Enter match situation  
- Simulate rain interruptions  
- View score ranges  
- Compare with DLS-style baseline  

---

##  Run Locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py

 Author:
Dev Sharma
