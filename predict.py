import os
import joblib
import pandas as pd


# project root = parent of src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATHS = {
    "T20": os.path.join(BASE_DIR, "models", "final_t20_model.pkl"),
    "ODI": os.path.join(BASE_DIR, "models", "final_odi_model.pkl"),
}

MODELS = {
    fmt: joblib.load(path) for fmt, path in MODEL_PATHS.items()
}

TOTAL_OVERS = {
    "T20": 20,
    "ODI": 50,
}


def overs_balls_to_balls(overs: int, balls: int) -> int:
    if balls < 0 or balls > 5:
        raise ValueError("Balls into current over must be between 0 and 5")
    return overs * 6 + balls


def balls_to_cricket_overs(total_balls: int) -> str:
    overs = total_balls // 6
    balls = total_balls % 6
    return f"{overs}.{balls}"


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
    total_overs=None,
    aggression_multiplier=1.0,
):
    match_format = match_format.upper()

    if total_overs is None:
        total_overs = TOTAL_OVERS[match_format]

    balls_bowled = overs_balls_to_balls(overs_completed, balls_into_over)
    total_balls = int(total_overs * 6)
    balls_remaining = total_balls - balls_bowled

    if balls_remaining < 0:
        raise ValueError("Overs completed cannot exceed total overs")

    overs_remaining = balls_remaining / 6
    wickets_in_hand = 10 - wickets_lost

    current_run_rate = current_score / (balls_bowled / 6) if balls_bowled > 0 else 0.0

    adjusted_rr = current_run_rate * aggression_multiplier
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


def predict_runs_remaining(features: pd.DataFrame, match_format: str) -> float:
    model = MODELS[match_format.upper()]
    return float(model.predict(features)[0])


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
    total_overs=None,
):
    safe_features = build_features(
        match_format=match_format,
        current_score=current_score,
        wickets_lost=wickets_lost,
        overs_completed=overs_completed,
        balls_into_over=balls_into_over,
        runs_last_6=runs_last_6,
        wickets_last_6=wickets_last_6,
        runs_last_12=runs_last_12,
        wickets_last_12=wickets_last_12,
        total_overs=total_overs,
        aggression_multiplier=0.90
    )

    expected_features = build_features(
        match_format=match_format,
        current_score=current_score,
        wickets_lost=wickets_lost,
        overs_completed=overs_completed,
        balls_into_over=balls_into_over,
        runs_last_6=runs_last_6,
        wickets_last_6=wickets_last_6,
        runs_last_12=runs_last_12,
        wickets_last_12=wickets_last_12,
        total_overs=total_overs,
        aggression_multiplier=1.00
    )

    aggressive_features = build_features(
        match_format=match_format,
        current_score=current_score,
        wickets_lost=wickets_lost,
        overs_completed=overs_completed,
        balls_into_over=balls_into_over,
        runs_last_6=runs_last_6,
        wickets_last_6=wickets_last_6,
        runs_last_12=runs_last_12,
        wickets_last_12=wickets_last_12,
        total_overs=total_overs,
        aggression_multiplier=1.20
    )

    safe_pred = round(current_score + predict_runs_remaining(safe_features, match_format))
    expected_pred = round(current_score + predict_runs_remaining(expected_features, match_format))
    aggressive_pred = round(current_score + predict_runs_remaining(aggressive_features, match_format))

    ordered = sorted([safe_pred, expected_pred, aggressive_pred])

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
        match_format=match_format,
        current_score=current_score,
        wickets_lost=wickets_lost,
        overs_completed=overs_completed,
        balls_into_over=balls_into_over,
        runs_last_6=runs_last_6,
        wickets_last_6=wickets_last_6,
        runs_last_12=runs_last_12,
        wickets_last_12=wickets_last_12,
        total_overs=reduced_overs,
    )


def simple_dls_baseline(match_format, current_score, overs_completed, balls_into_over, reduced_overs):
    match_format = match_format.upper()
    original_total_overs = TOTAL_OVERS[match_format]
    balls_bowled = overs_balls_to_balls(overs_completed, balls_into_over)

    original_remaining = max(0, original_total_overs * 6 - balls_bowled)
    reduced_remaining = max(0, int(reduced_overs * 6) - balls_bowled)

    if balls_bowled <= 0:
        return current_score

    baseline = current_score + (current_score / balls_bowled) * reduced_remaining
    return round(baseline)


def build_distribution_points(safe_score, expected_score, aggressive_score):
    scores = [safe_score, expected_score, aggressive_score]
    probs = [0.25, 0.50, 0.25]
    return scores, probs


if __name__ == "__main__":
    current_score = 92
    wickets_lost = 3
    overs_completed = 12
    balls_into_over = 0
    runs_last_6 = 10
    wickets_last_6 = 0
    runs_last_12 = 18
    wickets_last_12 = 1
    reduced_overs = 16

    normal = predict_range(
        match_format="T20",
        current_score=current_score,
        wickets_lost=wickets_lost,
        overs_completed=overs_completed,
        balls_into_over=balls_into_over,
        runs_last_6=runs_last_6,
        wickets_last_6=wickets_last_6,
        runs_last_12=runs_last_12,
        wickets_last_12=wickets_last_12,
        total_overs=20
    )

    rain = predict_rain_range(
        match_format="T20",
        current_score=current_score,
        wickets_lost=wickets_lost,
        overs_completed=overs_completed,
        balls_into_over=balls_into_over,
        runs_last_6=runs_last_6,
        wickets_last_6=wickets_last_6,
        runs_last_12=runs_last_12,
        wickets_last_12=wickets_last_12,
        reduced_overs=reduced_overs
    )

    dls_baseline = simple_dls_baseline(
        match_format="T20",
        current_score=current_score,
        overs_completed=overs_completed,
        balls_into_over=balls_into_over,
        reduced_overs=reduced_overs
    )

    print("Normal range:", normal)
    print("Rain range:", rain)
    print("Suggested revised target:", rain["expected"] + 1)
    print("Simple DLS-style baseline:", dls_baseline)

    difference = rain["expected"] - dls_baseline
    sign = "+" if difference >= 0 else ""

    print(f"AI vs DLS difference: {sign}{difference} runs")

    if difference > 10:
        print("AI predicts significantly higher scoring potential than DLS")
    elif difference < -10:
        print("AI predicts lower scoring potential than DLS")
    else:
        print("AI and DLS predictions are similar")