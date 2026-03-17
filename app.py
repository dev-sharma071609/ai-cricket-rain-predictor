import matplotlib.pyplot as plt
import streamlit as st

from predict import (
    predict_range,
    predict_rain_range,
    simple_dls_baseline,
    build_distribution_points,
    overs_to_balls,
)

st.set_page_config(
    page_title="AI Cricket Rain Predictor",
    page_icon="🏏",
    layout="centered"
)

st.title("🏏 AI Cricket Rain-Adjusted Score Predictor")
st.write(
    "Predict normal score range, rain-shortened score range, and compare it with a simple DLS-style baseline."
)

st.subheader("Match input")

current_score = st.number_input(
    "Current score",
    min_value=0,
    max_value=300,
    value=92,
    step=1
)

wickets_lost = st.number_input(
    "Wickets lost",
    min_value=0,
    max_value=10,
    value=3,
    step=1
)

overs_completed = st.number_input(
    "Overs completed (cricket format like 12.3)",
    min_value=0.0,
    max_value=20.0,
    value=12.0,
    step=0.1,
    format="%.1f"
)


runs_last_6 = st.number_input(
    "Runs in last 6 balls",
    min_value=0,
    max_value=36,
    value=10,
    step=1
)

wickets_last_6 = st.number_input(
    "Wickets in last 6 balls",
    min_value=0,
    max_value=6,
    value=0,
    step=1
)

runs_last_12 = st.number_input(
    "Runs in last 12 balls",
    min_value=0,
    max_value=72,
    value=18,
    step=1
)

wickets_last_12 = st.number_input(
    "Wickets in last 12 balls",
    min_value=0,
    max_value=10,
    value=1,
    step=1
)

st.subheader("Rain setting")

use_rain = st.checkbox("Apply rain-shortened innings", value=True)

reduced_overs = st.number_input(
    "Reduced total overs",
    min_value=5.0,
    max_value=20.0,
    value=16.0,
    step=0.1,
    format="%.1f"
)

if st.button("Predict"):
    try:
        _ = overs_to_balls(overs_completed)

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

        st.subheader("Normal score range")

        col1, col2, col3 = st.columns(3)
        col1.metric("Safe", normal["safe"])
        col2.metric("Expected", normal["expected"])
        col3.metric("Aggressive", normal["aggressive"])

        normal_scores, normal_probs = build_distribution_points(
            normal["safe"],
            normal["expected"],
            normal["aggressive"]
        )

        fig1 = plt.figure(figsize=(7, 4))
        plt.plot(normal_scores, normal_probs, marker="o")
        plt.xlabel("Predicted final score")
        plt.ylabel("Relative likelihood")
        plt.title("Normal-conditions score range")
        plt.grid(True)
        st.pyplot(fig1)

        if use_rain:
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

            st.subheader("Rain-adjusted score range")

            r1, r2, r3 = st.columns(3)
            r1.metric("Safe", rain["safe"])
            r2.metric("Expected", rain["expected"])
            r3.metric("Aggressive", rain["aggressive"])

            st.success(f"Suggested revised target: {rain['expected'] + 1}")

            baseline = simple_dls_baseline(
                current_score=current_score,
                overs_completed=overs_completed,
                reduced_overs=reduced_overs
            )

            difference = rain["expected"] - baseline
            sign = "+" if difference >= 0 else ""

            st.info(f"Simple DLS-style baseline: {baseline}")
            st.write(f"AI vs DLS difference: **{sign}{difference} runs**")

            if difference > 10:
                st.write("AI predicts significantly higher scoring potential than DLS.")
            elif difference < -10:
                st.write("AI predicts lower scoring potential than DLS.")
            else:
                st.write("AI and DLS predictions are fairly similar.")

            rain_scores, rain_probs = build_distribution_points(
                rain["safe"],
                rain["expected"],
                rain["aggressive"]
            )

            fig2 = plt.figure(figsize=(7, 4))
            plt.plot(rain_scores, rain_probs, marker="o")
            plt.xlabel("Predicted rain-adjusted final score")
            plt.ylabel("Relative likelihood")
            plt.title("Rain-adjusted score range")
            plt.grid(True)
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error: {e}")