import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "limited_overs_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Creating extra features...")

# Phase features
df["death_overs_flag"] = (df["overs_remaining"] <= 5).astype(int)
df["middle_overs_flag"] = ((df["overs_remaining"] > 5) & (df["overs_remaining"] <= 15)).astype(int)
df["powerplay_like_flag"] = (df["ball_number"] <= 36).astype(int)

# Aggression + momentum
df["required_attack"] = df["wickets_in_hand"] * df["current_run_rate"]
df["scoring_momentum"] = df["runs_last_12"] - df["runs_last_6"]

FEATURES = [
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
]


def train_and_save_model(df_subset: pd.DataFrame, model_path: str, label: str) -> None:
    X = df_subset[FEATURES]
    y = df_subset["runs_remaining"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training {label} model...")

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

    pred_runs_remaining = model.predict(X_test)

    pred_final_score = X_test["current_score"].values + pred_runs_remaining
    actual_final_score = X_test["current_score"].values + y_test.values

    mae = mean_absolute_error(actual_final_score, pred_final_score)

    print(f"✅ {label} MAE: {mae:.2f}")

    joblib.dump(model, model_path)
    print(f"✅ Saved {label} model to {model_path}")


# ---------------------------
# T20 model
# ---------------------------
df_t20 = df[
    (df["is_t20"] == 1) &
    (df["innings_no"] == 1) &
    (df["ball_number"] > 12)
].copy()

print("T20 rows:", len(df_t20))
train_and_save_model(
    df_t20,
    os.path.join(MODELS_DIR, "final_t20_model.pkl"),
    "T20"
)

# ---------------------------
# ODI model
# ---------------------------
df_odi = df[
    (df["is_odi"] == 1) &
    (df["innings_no"] == 1) &
    (df["ball_number"] > 18)
].copy()

print("ODI rows:", len(df_odi))
train_and_save_model(
    df_odi,
    os.path.join(MODELS_DIR, "final_odi_model.pkl"),
    "ODI"
)