import joblib
import pandas as pd


MODEL_PATH = "models/final_t20_model.pkl"
model = joblib.load("final_t20_model.pkl")


def overs_to_balls(overs: float) -> int:
    whole = int(overs)
    decimal = round(overs - whole, 1)
    balls = int(decimal * 10)

    if balls < 0 or balls > 5:
        raise ValueError("Invalid overs format. Use cricket style like 12.3 or 17.5")

    return whole * 6 + balls


def balls_to_overs(balls: int) -> float:
    return (balls // 6) + ((balls % 6) / 10)


def build_features(
    current_score,
    wickets_lost,
    overs_completed,
    runs_last_6,
    wickets_last_6,
    runs_last_12,
    wickets_last_12,
    total_overs,
    aggression_multiplier=1.0,
):
    balls_bowled = overs_to_balls(overs_completed)
    total_balls = int(total_overs * 6)
    balls_remaining = total_balls - balls_bowled

    if balls_remaining < 0:
        raise ValueError("Overs completed cannot be greater than total overs")

    overs_remaining = balls_remaining / 6
    wickets_in_hand = 10 - wickets_lost

    current_run_rate = current_score / (balls_bowled / 6) if balls_bowled > 0 else 0

    adjusted_rr = current_run_rate * aggression_multiplier
    adjusted_last6 = runs_last_6 * aggression_multiplier
    adjusted_last12 = runs_last_12 * aggression_multiplier

    aggression_index = wickets_in_hand / max(1, overs_remaining)

    features = pd.DataFrame([{
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

    return features


def predict_runs_remaining(features: pd.DataFrame) -> float:
    return float(model.predict(features)[0])


def predict_range(
    current_score,
    wickets_lost,
    overs_completed,
    runs_last_6,
    wickets_last_6,
    runs_last_12,
    wickets_last_12,
    total_overs=20,
):
    safe_features = build_features(
        current_score, wickets_lost, overs_completed,
        runs_last_6, wickets_last_6,
        runs_last_12, wickets_last_12,
        total_overs,
        aggression_multiplier=0.90
    )

    expected_features = build_features(
        current_score, wickets_lost, overs_completed,
        runs_last_6, wickets_last_6,
        runs_last_12, wickets_last_12,
        total_overs,
        aggression_multiplier=1.00
    )

    aggressive_features = build_features(
        current_score, wickets_lost, overs_completed,
        runs_last_6, wickets_last_6,
        runs_last_12, wickets_last_12,
        total_overs,
        aggression_multiplier=1.20
    )

    safe_pred = round(current_score + predict_runs_remaining(safe_features))
    expected_pred = round(current_score + predict_runs_remaining(expected_features))
    aggressive_pred = round(current_score + predict_runs_remaining(aggressive_features))

    ordered = sorted([safe_pred, expected_pred, aggressive_pred])

    return {
        "safe": ordered[0],
        "expected": ordered[1],
        "aggressive": ordered[2],
    }

def predict_rain_range(
    current_score,
    wickets_lost,
    overs_completed,
    runs_last_6,
    wickets_last_6,
    runs_last_12,
    wickets_last_12,
    reduced_overs,
):
    return predict_range(
        current_score=current_score,
        wickets_lost=wickets_lost,
        overs_completed=overs_completed,
        runs_last_6=runs_last_6,
        wickets_last_6=wickets_last_6,
        runs_last_12=runs_last_12,
        wickets_last_12=wickets_last_12,
        total_overs=reduced_overs,
    )


def simple_dls_baseline(current_score, overs_completed, reduced_overs):
    """
    Rough baseline, not official DLS.
    Uses proportional overs-remaining resource idea for comparison only.
    """
    original_total_overs = 20
    balls_bowled = overs_to_balls(overs_completed)
    original_remaining = max(0, original_total_overs * 6 - balls_bowled)
    reduced_remaining = max(0, int(reduced_overs * 6) - balls_bowled)

    if original_remaining == 0:
        return current_score

    baseline = current_score + (current_score / max(1, balls_bowled)) * reduced_remaining
    return round(baseline)


def build_distribution_points(safe_score, expected_score, aggressive_score):
    """
    Creates simple plotting points for a probability-style chart.
    Not a true calibrated probability distribution, but a useful visual.
    """
    scores = [safe_score, expected_score, aggressive_score]
    probs = [0.25, 0.50, 0.25]
    return scores, probs


if __name__ == "__main__":

    current_score = 92
    wickets_lost = 3
    overs_completed = 12.0
    runs_last_6 = 10
    wickets_last_6 = 0
    runs_last_12 = 18
    wickets_last_12 = 1
    reduced_overs = 16

    normal = predict_range(
        current_score=current_score,
        wickets_lost=wickets_lost,
        overs_completed=overs_completed,
        runs_last_6=runs_last_6,
        wickets_last_6=wickets_last_6,
        runs_last_12=runs_last_12,
        wickets_last_12=wickets_last_12,
        total_overs=20
    )

    rain = predict_rain_range(
        current_score=current_score,
        wickets_lost=wickets_lost,
        overs_completed=overs_completed,
        runs_last_6=runs_last_6,
        wickets_last_6=wickets_last_6,
        runs_last_12=runs_last_12,
        wickets_last_12=wickets_last_12,
        reduced_overs=reduced_overs
    )

    dls_baseline = simple_dls_baseline(current_score, overs_completed, reduced_overs)

    print("Normal range:", normal)
    print("Rain range:", rain)
    print("Suggested revised target:", rain["expected"] + 1)
    print("Simple DLS-style baseline:", dls_baseline)

    #  FIXED INDENTATION BLOCK
    difference = rain["expected"] - dls_baseline

    sign = "+" if difference >= 0 else ""

    print(f"AI vs DLS difference: {sign}{difference} runs")

    if difference > 10:
        print("AI predicts significantly higher scoring potential than DLS")
    elif difference < -10:
        print("AI predicts lower scoring potential than DLS")
    else:
        print("AI and DLS predictions are similar")