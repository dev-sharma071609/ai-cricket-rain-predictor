import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOTAL_OVERS = {
    "T20": 20,
    "ODI": 50,
}

MODEL_PATHS = {
    "T20": os.path.join(BASE_DIR, "models", "final_t20_model.pkl"),
    "ODI": os.path.join(BASE_DIR, "models", "final_odi_model.pkl"),
}

MODELS = {}
AVAILABLE_FORMATS = []

for fmt, path in MODEL_PATHS.items():
    if os.path.exists(path):
        MODELS[fmt] = joblib.load(path)
        AVAILABLE_FORMATS.append(fmt)


def overs_to_balls(overs, balls):
    return overs * 6 + balls


def build_features(
    match_format,
    current_score,
    wickets_lost,
    overs_completed,
    balls_into_over,
    runs_last_6,
    wickets_last_6,
    runs_last_12,
    wickets_last_12,
    total_overs=20,
    aggression_multiplier=1.0,
):
    match_format = match_format.upper()

    if match_format not in MODELS:
        raise FileNotFoundError(f"Model for format '{match_format}' not found.")

    balls_bowled = overs_to_balls(overs_completed, balls_into_over)
    total_balls = total_overs * 6
    balls_remaining = total_balls - balls_bowled

    if balls_remaining < 0:
        raise ValueError("Overs completed cannot exceed total overs")

    overs_remaining = balls_remaining / 6
    wickets_in_hand = 10 - wickets_lost

    current_rr = current_score / (balls_bowled / 6) if balls_bowled > 0 else 0.0

    adjusted_rr = current_rr * aggression_multiplier
    adjusted_last6 = runs_last_6 * aggression_multiplier
    adjusted_last12 = runs_last_12 * aggression_multiplier

    aggression_index = wickets_in_hand / max(1, overs_remaining)

    return pd.DataFrame([{
        "current_score": current_score,
        "wickets_lost": wickets_lost,
        "wickets_in_hand": wickets_in_hand,
        "balls_remaining": balls_remaining,
        "overs_remaining": overs_remaining,
        "current_run_rate": adjusted_rr,
        "runs_last_6": adjusted_last6,
        "wickets_last_6": wickets_last_6,
        "runs_last_12": adjusted_last12,
        "wickets_last_12": wickets_last_12,
        "aggression_index": aggression_index,
        "death_overs_flag": int(overs_remaining <= 5),
        "middle_overs_flag": int((overs_remaining > 5) and (overs_remaining <= 15)),
        "powerplay_like_flag": int(balls_bowled <= 36),
        "required_attack": wickets_in_hand * adjusted_rr,
        "scoring_momentum": adjusted_last12 - adjusted_last6
    }])


def predict_runs_remaining(df, match_format):
    model = MODELS[match_format]
    return float(model.predict(df)[0])


def predict_range(
    match_format,
    current_score,
    wickets_lost,
    overs_completed,
    balls_into_over,
    runs_last_6,
    wickets_last_6,
    runs_last_12,
    wickets_last_12,
    total_overs,
):
    safe_df = build_features(
        match_format,
        current_score,
        wickets_lost,
        overs_completed,
        balls_into_over,
        runs_last_6,
        wickets_last_6,
        runs_last_12,
        wickets_last_12,
        total_overs,
        aggression_multiplier=0.90,
    )

    expected_df = build_features(
        match_format,
        current_score,
        wickets_lost,
        overs_completed,
        balls_into_over,
        runs_last_6,
        wickets_last_6,
        runs_last_12,
        wickets_last_12,
        total_overs,
        aggression_multiplier=1.00,
    )

    aggressive_df = build_features(
        match_format,
        current_score,
        wickets_lost,
        overs_completed,
        balls_into_over,
        runs_last_6,
        wickets_last_6,
        runs_last_12,
        wickets_last_12,
        total_overs,
        aggression_multiplier=1.20,
    )

    safe = int(round(current_score + predict_runs_remaining(safe_df, match_format)))
    expected = int(round(current_score + predict_runs_remaining(expected_df, match_format)))
    aggressive = int(round(current_score + predict_runs_remaining(aggressive_df, match_format)))

    ordered = sorted([safe, expected, aggressive])

    return {
        "safe": ordered[0],
        "expected": ordered[1],
        "aggressive": ordered[2],
    }


def predict_rain_range(
    match_format,
    current_score,
    wickets_lost,
    overs_completed,
    balls_into_over,
    runs_last_6,
    wickets_last_6,
    runs_last_12,
    wickets_last_12,
    reduced_overs,
):
    return predict_range(
        match_format,
        current_score,
        wickets_lost,
        overs_completed,
        balls_into_over,
        runs_last_6,
        wickets_last_6,
        runs_last_12,
        wickets_last_12,
        reduced_overs,
    )


def simple_dls_baseline(
    match_format,
    current_score,
    overs_completed,
    balls_into_over,
    reduced_overs,
):
    original_total_overs = TOTAL_OVERS[match_format]
    balls_bowled = overs_to_balls(overs_completed, balls_into_over)

    if balls_bowled == 0:
        return current_score

    current_rr_per_ball = current_score / balls_bowled
    reduced_total_balls = reduced_overs * 6

    return int(round(current_rr_per_ball * reduced_total_balls))


def build_distribution_points(safe, expected, aggressive):
    return [safe, expected, aggressive], [0.25, 0.50, 0.25]


if __name__ == "__main__":
    print("Looking for model files at:")
    for fmt, path in MODEL_PATHS.items():
        print(fmt, "->", path, "| exists:", os.path.exists(path))
    print("Available formats:", AVAILABLE_FORMATS)