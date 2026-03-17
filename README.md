# ai-cricket-rain-predictor
AI-based T20 rain-adjusted score predictor using ball-by-ball cricket data, XGBoost, and Streamlit.
# AI Cricket Rain-Adjusted Score Predictor

An AI-based T20 cricket score prediction system built using ball-by-ball international cricket data.  
The model predicts safe, expected, and aggressive final score ranges from live match situations and estimates revised totals when rain shortens the innings.

## Features

- Predicts **safe, expected, and aggressive** final score ranges
- Supports **rain-shortened innings** prediction
- Compares AI prediction with a **simple DLS-style baseline**
- Built with **Python, XGBoost, Streamlit, and ball-by-ball match data**

## Final Model

- Format: **T20 only**
- Innings: **First innings only**
- Early overs removed: **first 2 overs excluded**
- Target: **runs remaining**
- Model: **XGBoost**
- Final MAE: **16.12**

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

## Example

Input:
- Score: 92/3
- Overs: 12.0
- Rain reduces innings to 16 overs

Output:
- Normal range: 165 / 167 / 170
- Rain-adjusted range: 136 / 138 / 144
- Suggested revised target: 139
- AI vs simple DLS-style baseline: +15 runs

## Files

- `src/extract_zips.py` - extracts raw zip files
- `src/build_datasets.py` - builds ML dataset from Cricsheet JSON files
- `src/train_model.py` - trains the final T20 model
- `src/predict.py` - prediction logic
- `src/app.py` - Streamlit UI

## Run the app

```bash
python -m streamlit run src/app.py
