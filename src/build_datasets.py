from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

FORMAT_CONFIG = {
    "T20": {
        "folder": RAW_DIR / "t20s",
        "max_balls": 120,
    },
    "ODI": {
        "folder": RAW_DIR / "odis",
        "max_balls": 300,
    },
}


def safe_load_json(file_path: Path) -> dict[str, Any] | None:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Could not read {file_path.name}: {e}")
        return None


def get_innings_rows(
    match_id: str,
    match_format: str,
    innings_index: int,
    innings: dict[str, Any],
    max_balls: int,
) -> list[dict[str, Any]]:
    rows = []

    cumulative_score = 0
    wickets_lost = 0
    ball_number = 0

    overs = innings.get("overs", [])
    batting_team = innings.get("team", "Unknown")

    for over in overs:
        deliveries = over.get("deliveries", [])

        for delivery in deliveries:
            ball_number += 1

            runs_info = delivery.get("runs", {})
            runs_total = runs_info.get("total", 0)

            wicket_on_ball = 1 if "wickets" in delivery else 0

            cumulative_score += runs_total
            wickets_lost += wicket_on_ball
            wickets_in_hand = 10 - wickets_lost

            current_run_rate = cumulative_score / (ball_number / 6) if ball_number > 0 else 0.0
            balls_remaining = max_balls - ball_number
            overs_remaining = balls_remaining / 6

            aggression_index = wickets_in_hand / max(1, overs_remaining)

            rows.append(
                {
                    "match_id": match_id,
                    "match_format": match_format,
                    "innings_no": innings_index,
                    "batting_team": batting_team,
                    "ball_number": ball_number,
                    "runs_scored_on_ball": runs_total,
                    "current_score": cumulative_score,
                    "wickets_lost": wickets_lost,
                    "wickets_in_hand": wickets_in_hand,
                    "balls_remaining": max(0, balls_remaining),
                    "overs_remaining": max(0, overs_remaining),
                    "current_run_rate": current_run_rate,
                    "aggression_index": aggression_index,
                }
            )

    final_score = cumulative_score
    for row in rows:
        row["final_score"] = final_score
        row["runs_remaining"] = final_score - row["current_score"]

    return rows


def add_recent_form_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["match_id", "innings_no", "ball_number"]).copy()
    group_cols = ["match_id", "innings_no"]

    df["runs_last_6"] = (
        df.groupby(group_cols)["runs_scored_on_ball"]
        .rolling(window=6, min_periods=1)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )

    wicket_on_ball = (
        df.groupby(group_cols)["wickets_lost"]
        .diff()
        .fillna(df["wickets_lost"])
        .clip(lower=0)
    )

    df["wickets_last_6"] = (
        wicket_on_ball.groupby([df["match_id"], df["innings_no"]])
        .rolling(window=6, min_periods=1)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )

    df["runs_last_12"] = (
        df.groupby(group_cols)["runs_scored_on_ball"]
        .rolling(window=12, min_periods=1)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )

    df["wickets_last_12"] = (
        wicket_on_ball.groupby([df["match_id"], df["innings_no"]])
        .rolling(window=12, min_periods=1)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )

    return df


def parse_match(file_path: Path, match_format: str, max_balls: int) -> list[dict[str, Any]]:
    data = safe_load_json(file_path)
    if data is None:
        return []

    innings_list = data.get("innings", [])
    if not innings_list:
        return []

    all_rows = []

    for innings_index, innings in enumerate(innings_list, start=1):
        rows = get_innings_rows(
            match_id=file_path.stem,
            match_format=match_format,
            innings_index=innings_index,
            innings=innings,
            max_balls=max_balls,
        )
        all_rows.extend(rows)

    return all_rows


def build_dataset() -> pd.DataFrame:
    all_rows = []

    for match_format, cfg in FORMAT_CONFIG.items():
        folder = cfg["folder"]
        max_balls = cfg["max_balls"]

        files = sorted(folder.glob("*.json"))
        print(f"{match_format}: found {len(files)} files")

        for file_path in tqdm(files, desc=f"Parsing {match_format}"):
            rows = parse_match(file_path, match_format, max_balls)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    if df.empty:
        raise ValueError("No rows created. Check extracted folders.")

    df = add_recent_form_features(df)

    df["is_t20"] = (df["match_format"] == "T20").astype(int)
    df["is_odi"] = (df["match_format"] == "ODI").astype(int)

    return df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = build_dataset()

    output_path = PROCESSED_DIR / "limited_overs_dataset.csv"
    df.to_csv(output_path, index=False)

    print("\nSaved dataset to:", output_path)
    print("Shape:", df.shape)
    print(df.head())


if __name__ == "__main__":
    main()