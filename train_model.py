import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


print("Loading dataset...")
df = pd.read_csv("data/processed/limited_overs_dataset.csv")

print("Creating extra features...")

# Phase features
df["death_overs_flag"] = (df["overs_remaining"] <= 5).astype(int)
df["middle_overs_flag"] = ((df["overs_remaining"] > 5) & (df["overs_remaining"] <= 15)).astype(int)
df["powerplay_like_flag"] = (df["ball_number"] <= 36).astype(int)

# Aggression & momentum
df["required_attack"] = df["wickets_in_hand"] * df["current_run_rate"]
df["scoring_momentum"] = df["runs_last_12"] - df["runs_last_6"]

# 🔥 IMPORTANT FILTERS (THIS IS THE BIG UPGRADE)
df_t20 = df[
    (df["is_t20"] == 1) &
    (df["innings_no"] == 1) &        # only first innings
    (df["ball_number"] > 12)         # remove very early overs (noise)
].copy()

print("Rows being used:", len(df_t20))

# Features
X = df_t20[[
    "current_score",
    "wickets_lost",
    "wickets_in_hand",
    "balls_remaining",
    "overs_remaining",
    "current_run_rate",
    "runs_last_6",
    "wickets_last_6",
    "runs_last_12",
    "wickets_last_12",
    "aggression_index",
    "death_overs_flag",
    "middle_overs_flag",
    "powerplay_like_flag",
    "required_attack",
    "scoring_momentum"
]]

# Target = runs remaining
y = df_t20["runs_remaining"]

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training XGBoost model...")

model = XGBRegressor(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.07,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model trained!")

# Predictions
pred_runs_remaining = model.predict(X_test)

# Convert to final score
pred_final_score = X_test["current_score"].values + pred_runs_remaining
actual_final_score = X_test["current_score"].values + y_test.values

mae = mean_absolute_error(actual_final_score, pred_final_score)

print(f"🔥 FINAL T20 MODEL MAE: {mae:.2f}")

# Save model
joblib.dump(model, "models/final_t20_model.pkl")

print("Saved model to models/final_t20_model.pkl")