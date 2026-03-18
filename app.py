import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from predict import (
    predict_range,
    predict_rain_range,
    simple_dls_baseline,
    build_distribution_points,
    AVAILABLE_FORMATS,
)

st.set_page_config(page_title="AI Cricket Rain Predictor", page_icon="🏏", layout="wide")

st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }
    .stMetric {
        text-align: center;
        border: 1px solid rgba(150,150,150,0.2);
        border-radius: 12px;
        padding: 8px;
        background-color: rgba(255,255,255,0.03);
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏏 AI Cricket Rain-Adjusted Score Predictor")
st.caption("T20 and ODI first-innings score forecasting with rain-shortening analysis and DLS-style comparison.")

if not AVAILABLE_FORMATS:
    st.error("No trained model files were found. Please upload at least one model file.")
    st.stop()

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.header("About")
st.sidebar.write(
    "This tool predicts safe, expected, and aggressive final scores using trained ML models."
)
st.sidebar.write(
    "It also compares rain-adjusted output with a simple DLS-style baseline."
)

# ----------------------------
# TOP INPUT SECTION
# ----------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Match context")
    match_format = st.selectbox("Format", AVAILABLE_FORMATS)
    batting_team = st.text_input("Batting team", value="England")
    bowling_team = st.text_input("Bowling team", value="Australia")
    venue = st.text_input("Venue", value="Bengaluru")

with right:
    st.subheader("Current innings state")
    current_score = st.number_input("Current score", min_value=0, max_value=500, value=189, step=1)
    wickets_lost = st.number_input("Wickets lost", min_value=0, max_value=10, value=4, step=1)

    c1, c2 = st.columns(2)
    with c1:
        max_overs_input = 20 if match_format == "T20" else 50
        default_overs = 12 if match_format == "T20" else 31
        overs_completed = st.number_input(
            "Completed overs",
            min_value=0,
            max_value=max_overs_input,
            value=default_overs,
            step=1
        )
    with c2:
        default_balls = 0 if match_format == "T20" else 2
        balls_into_over = st.number_input(
            "Balls into current over",
            min_value=0,
            max_value=5,
            value=default_balls,
            step=1
        )

    default_last6 = 10 if match_format == "T20" else 5
    default_last12 = 18 if match_format == "T20" else 13
    default_wkts12 = 1 if match_format == "T20" else 0

    runs_last_6 = st.number_input("Runs in last 6 balls", min_value=0, max_value=36, value=default_last6, step=1)
    wickets_last_6 = st.number_input("Wickets in last 6 balls", min_value=0, max_value=6, value=0, step=1)
    runs_last_12 = st.number_input("Runs in last 12 balls", min_value=0, max_value=72, value=default_last12, step=1)
    wickets_last_12 = st.number_input("Wickets in last 12 balls", min_value=0, max_value=10, value=default_wkts12, step=1)

st.markdown("---")

# ----------------------------
# RAIN SECTION
# ----------------------------
st.subheader("Rain scenario")
use_rain = st.checkbox("Apply rain-shortened innings", value=True)

default_reduced = 16 if match_format == "T20" else 41
reduced_overs = None

if use_rain:
    max_overs = 20 if match_format == "T20" else 50
    reduced_overs = st.slider(
        "Reduced total overs",
        min_value=5,
        max_value=max_overs,
        value=default_reduced,
        step=1
    )

st.markdown("---")

# ----------------------------
# PREDICTION BUTTON
# ----------------------------
if st.button("Predict"):
    try:
        total_overs = 20 if match_format == "T20" else 50
        total_balls = total_overs * 6
        balls_bowled = overs_completed * 6 + balls_into_over
        balls_remaining = total_balls - balls_bowled
        wickets_in_hand = 10 - wickets_lost
        current_rr = current_score / (balls_bowled / 6) if balls_bowled > 0 else 0

        # Header summary
        st.markdown(
            f"""
            ### {batting_team} vs {bowling_team}
            **Venue:** {venue}  
            **Format:** {match_format}
            """
        )

        # Normal prediction
        normal = predict_range(
            match_format=match_format,
            current_score=current_score,
            wickets_lost=wickets_lost,
            overs_completed=overs_completed,
            balls_into_over=balls_into_over,
            runs_last_6=runs_last_6,
            wickets_last_6=wickets_last_6,
            runs_last_12=runs_last_12,
            wickets_last_12=wickets_last_12,
            total_overs=total_overs
        )

        st.subheader("📊 Normal score forecast")
        m1, m2, m3 = st.columns(3)
        m1.metric("Safe", normal["safe"])
        m2.metric("Expected", normal["expected"])
        m3.metric("Aggressive", normal["aggressive"])

        i1, i2, i3 = st.columns(3)
        i1.metric("Wickets in hand", wickets_in_hand)
        i2.metric("Balls remaining", balls_remaining)
        i3.metric("Current RR", f"{current_rr:.2f}")

        # Confidence
        confidence_gap = normal["aggressive"] - normal["safe"]
        if confidence_gap < 15:
            st.success("High confidence prediction")
        elif confidence_gap < 30:
            st.info("Moderate uncertainty")
        else:
            st.warning("Low confidence — match highly unpredictable")

        # Insights
        st.subheader("🧠 Match insights")
        insights = []

        if wickets_in_hand >= 7:
            insights.append("Strong batting depth available — aggressive finish likely.")
        if current_rr > 8:
            insights.append("High scoring momentum detected.")
        if balls_remaining < 30:
            insights.append("Death overs phase — volatility is high.")
        if wickets_lost >= 6:
            insights.append("Batting side is under pressure — collapse risk is higher.")

        if insights:
            for x in insights:
                st.write(f"• {x}")
        else:
            st.write("• No major positive or negative pressure signal detected.")

        # Normal distribution chart
        st.subheader("📈 Normal forecast shape")
        normal_scores, normal_probs = build_distribution_points(
            normal["safe"], normal["expected"], normal["aggressive"]
        )

        fig1 = plt.figure(figsize=(7, 4))
        plt.plot(normal_scores, normal_probs, marker="o")
        plt.fill_between(normal_scores, normal_probs, alpha=0.2)
        plt.xlabel("Predicted final score")
        plt.ylabel("Relative likelihood")
        plt.title(f"{match_format} normal-conditions score range")
        plt.grid(True, alpha=0.3)
        st.pyplot(fig1)

        # Rain prediction
        if use_rain and reduced_overs is not None:
            rain = predict_rain_range(
                match_format=match_format,
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

            st.subheader("🌧️ Rain-adjusted forecast")
            r1, r2, r3 = st.columns(3)
            r1.metric("Safe", rain["safe"])
            r2.metric("Expected", rain["expected"])
            r3.metric("Aggressive", rain["aggressive"])

            st.success(f"Suggested revised target: {rain['expected'] + 1}")

            baseline = simple_dls_baseline(
                match_format=match_format,
                current_score=current_score,
                overs_completed=overs_completed,
                balls_into_over=balls_into_over,
                reduced_overs=reduced_overs
            )

            difference = rain["expected"] - baseline
            sign = "+" if difference >= 0 else ""

            st.info(f"Simple DLS-style baseline: {baseline}")
            st.write(f"**AI vs DLS difference:** {sign}{difference} runs")

            if difference > 10:
                st.write("AI predicts significantly higher scoring potential than the simple DLS-style baseline.")
            elif difference < -10:
                st.write("AI predicts lower scoring potential than the simple DLS-style baseline.")
            else:
                st.write("AI and the simple DLS-style baseline are fairly similar.")

            # Rain distribution chart
            rain_scores, rain_probs = build_distribution_points(
                rain["safe"], rain["expected"], rain["aggressive"]
            )

            fig2 = plt.figure(figsize=(7, 4))
            plt.plot(rain_scores, rain_probs, marker="o")
            plt.fill_between(rain_scores, rain_probs, alpha=0.2)
            plt.xlabel("Predicted rain-adjusted final score")
            plt.ylabel("Relative likelihood")
            plt.title(f"{match_format} rain-adjusted score range")
            plt.grid(True, alpha=0.3)
            st.pyplot(fig2)

            # Comparison chart
            st.subheader("📉 Expected score comparison")
            fig3 = plt.figure(figsize=(7, 4))
            labels = ["Normal", "Rain", "DLS"]
            values = [normal["expected"], rain["expected"], baseline]

            plt.bar(labels, values)
            plt.ylabel("Score")
            plt.title("Expected Score Comparison")
            st.pyplot(fig3)

            # What-if
            st.subheader("🔄 What-if scenario")
            what_if_overs = st.slider(
                "Try another reduced overs value",
                min_value=5,
                max_value=total_overs,
                value=reduced_overs,
                step=1,
                key="what_if_slider"
            )

            if what_if_overs != reduced_overs:
                what_if = predict_rain_range(
                    match_format=match_format,
                    current_score=current_score,
                    wickets_lost=wickets_lost,
                    overs_completed=overs_completed,
                    balls_into_over=balls_into_over,
                    runs_last_6=runs_last_6,
                    wickets_last_6=wickets_last_6,
                    runs_last_12=runs_last_12,
                    wickets_last_12=wickets_last_12,
                    reduced_overs=what_if_overs
                )

                w1, w2 = st.columns(2)
                w1.metric("What-if expected score", what_if["expected"])
                w2.metric("What-if target", what_if["expected"] + 1)

    except Exception as e:
        st.error(f"Error: {e}")