import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets", "logos")

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from predict import (
    predict_range,
    predict_rain_range,
    simple_dls_baseline,
    build_distribution_points,
    AVAILABLE_FORMATS,
)

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Cricket AI Rain Predictor",
    page_icon="🏏",
    layout="wide"
)

# ----------------------------
# TEAM DATA
# ----------------------------
IPL_TEAMS = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Delhi Capitals",
    "Punjab Kings",
    "Lucknow Super Giants",
    "Gujarat Titans",
]

INTERNATIONAL_TEAMS = [
    "India",
    "Australia",
    "England",
    "New Zealand",
    "Pakistan",
    "South Africa",
    "Sri Lanka",
    "Bangladesh",
    "Afghanistan",
    "West Indies",
    "Ireland",
    "Zimbabwe",
    "Netherlands",
]

TEAM_SCOPE_OPTIONS = ["Local / Custom Teams", "IPL Teams", "International Teams"]

# ----------------------------
# STYLING
# ----------------------------
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: "Inter", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(0, 194, 255, 0.10), transparent 30%),
            radial-gradient(circle at top right, rgba(139, 92, 246, 0.10), transparent 28%),
            linear-gradient(180deg, #06101d 0%, #091424 42%, #0b1830 100%);
    }

    .block-container {
        max-width: 1460px;
        padding-top: 0.8rem;
        padding-bottom: 2.5rem;
    }

    .top-nav {
        position: sticky;
        top: 0.4rem;
        z-index: 999;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 18px;
        margin-bottom: 18px;
        border-radius: 18px;
        background: rgba(8, 16, 29, 0.82);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 24px rgba(0,0,0,0.16);
    }

    .brand-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-badge {
        width: 46px;
        height: 46px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #0ea5e9, #8b5cf6);
        font-size: 1.45rem;
        box-shadow: 0 10px 20px rgba(14,165,233,0.28);
    }

    .brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #f7fbff;
        margin-bottom: 2px;
    }

    .brand-sub {
        font-size: 0.88rem;
        color: #a8bddb;
    }

    .nav-pill {
        display: inline-block;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        color: #dce9fb;
        font-size: 0.86rem;
        font-weight: 600;
        margin-left: 8px;
    }

    .hero-box {
        background: linear-gradient(135deg, rgba(0,183,255,0.18), rgba(140,82,255,0.15));
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 26px;
        padding: 30px 30px;
        margin-bottom: 22px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.28);
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.04;
        margin-bottom: 0.55rem;
        color: #f5f9ff;
    }

    .hero-sub {
        color: #d8e5f8;
        font-size: 1.06rem;
        line-height: 1.7;
        max-width: 930px;
    }

    .tag-row {
        margin-top: 16px;
    }

    .tag {
        display: inline-block;
        padding: 8px 14px;
        margin-right: 10px;
        margin-bottom: 8px;
        border-radius: 999px;
        font-size: 0.88rem;
        font-weight: 600;
        background: rgba(255,255,255,0.09);
        border: 1px solid rgba(255,255,255,0.08);
        color: #ebf3ff;
    }

    .glass-card {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 20px 20px 10px 20px;
        margin-bottom: 18px;
        box-shadow: 0 10px 26px rgba(0,0,0,0.18);
        backdrop-filter: blur(10px);
    }

    .section-title {
        font-size: 1.8rem;
        font-weight: 750;
        color: #ffffff;
        margin-bottom: 4px;
    }

    .section-sub {
        color: #b9c9e2;
        font-size: 0.97rem;
        margin-bottom: 16px;
    }

    .match-banner {
        background: linear-gradient(135deg, rgba(14,165,233,0.15), rgba(34,197,94,0.10));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 22px 24px;
        margin-bottom: 18px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.16);
    }

    .versus-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 14px;
        flex-wrap: wrap;
    }

    .team-chip {
        flex: 1;
        min-width: 260px;
        border-radius: 18px;
        padding: 16px 18px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        min-height: 96px;
        display: flex;
        align-items: center;
    }

    .team-logo-fallback {
        width: 58px;
        height: 58px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.08);
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 800;
    }

    .team-name {
        font-size: 1.18rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
    }

    .vs-pill {
        padding: 10px 16px;
        border-radius: 999px;
        background: linear-gradient(90deg, #0ea5e9, #8b5cf6);
        color: white;
        font-weight: 800;
        letter-spacing: 0.6px;
        box-shadow: 0 10px 20px rgba(14,165,233,0.22);
    }

    .insight-card {
        background: linear-gradient(180deg, rgba(0,183,255,0.10), rgba(0,183,255,0.06));
        border: 1px solid rgba(0,183,255,0.16);
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 10px;
        color: #eaf5ff;
    }

    .stMetric {
        background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.03));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 10px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.12);
    }

    .result-banner {
        padding: 16px 18px;
        border-radius: 18px;
        font-weight: 650;
        margin-top: 8px;
        margin-bottom: 12px;
    }

    .green-banner {
        background: linear-gradient(90deg, rgba(34,197,94,0.22), rgba(34,197,94,0.10));
        border: 1px solid rgba(34,197,94,0.25);
        color: #d8ffe7;
    }

    .blue-banner {
        background: linear-gradient(90deg, rgba(59,130,246,0.22), rgba(59,130,246,0.10));
        border: 1px solid rgba(59,130,246,0.25);
        color: #ddecff;
    }

    .purple-banner {
        background: linear-gradient(90deg, rgba(168,85,247,0.22), rgba(168,85,247,0.10));
        border: 1px solid rgba(168,85,247,0.24);
        color: #f1e6ff;
    }

    .stButton > button {
        width: 100%;
        border-radius: 14px;
        padding: 0.84rem 1rem;
        font-size: 1rem;
        font-weight: 700;
        border: none;
        color: white;
        background: linear-gradient(90deg, #0ea5e9, #8b5cf6);
        box-shadow: 0 10px 24px rgba(14,165,233,0.22);
    }

    .stButton > button:hover {
        filter: brightness(1.05);
        transform: translateY(-1px);
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0a1322 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    .sidebar-box {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 18px;
        padding: 14px 14px 4px 14px;
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# HELPERS
# ----------------------------
def choose_team(label_prefix: str, exclude_team=None):
    st.markdown(f"**{label_prefix} team selection type**")
    scope = st.selectbox(
        f"{label_prefix} type",
        TEAM_SCOPE_OPTIONS,
        key=f"{label_prefix}_scope",
        label_visibility="collapsed",
    )

    if scope == "Local / Custom Teams":
        team_name = st.text_input(
            f"{label_prefix} team name",
            value="My Local XI" if label_prefix == "Batting" else "Opposition XI",
            key=f"{label_prefix}_local_name",
        )
        return team_name.strip() if team_name.strip() else f"{label_prefix} Team"

    teams = IPL_TEAMS.copy() if scope == "IPL Teams" else INTERNATIONAL_TEAMS.copy()

    if exclude_team and exclude_team in teams:
        teams = [t for t in teams if t != exclude_team]

    default_index = 0 if label_prefix == "Batting" else min(1, len(teams) - 1)

    return st.selectbox(
        f"{label_prefix} team",
        teams,
        index=default_index,
        key=f"{label_prefix}_dropdown",
    )


def get_total_overs(match_format: str) -> int:
    return 20 if match_format == "T20" else 50


def slugify_team_name(team_name: str) -> str:
    return team_name.strip().lower().replace("&", "and").replace(" ", "_")


def get_logo_path(team_name: str):
    filename = f"{slugify_team_name(team_name)}.png"
    path = os.path.join(ASSETS_DIR, filename)
    return path if os.path.exists(path) else None


def render_team_card(team_name: str):
    logo_path = get_logo_path(team_name)
    col_logo, col_text = st.columns([1, 4])

    with col_logo:
        if logo_path:
            st.image(logo_path, width=60)
        else:
            initials = "".join([word[0] for word in team_name.split()[:2]]).upper()
            st.markdown(f'<div class="team-logo-fallback">{initials}</div>', unsafe_allow_html=True)

    with col_text:
        st.markdown(f'<div class="team-name">{team_name}</div>', unsafe_allow_html=True)


def style_dark_chart(fig):
    ax = fig.axes[0]
    ax.set_facecolor("#0b1322")
    fig.patch.set_facecolor("#0b1322")

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(colors="#dbeafe")
    ax.xaxis.label.set_color("#dbeafe")
    ax.yaxis.label.set_color("#dbeafe")
    ax.title.set_color("#f8fbff")
    ax.grid(alpha=0.16, color="#8fb4ff")


# ----------------------------
# TOP NAV
# ----------------------------
st.markdown(
    """
    <div class="top-nav">
        <div class="brand-left">
            <div class="brand-badge">🏏</div>
            <div>
                <div class="brand-title">Cricket AI Forecast Studio</div>
                <div class="brand-sub">Match intelligence • Rain forecasting • DLS comparison</div>
            </div>
        </div>
        <div>
            <span class="nav-pill">Premium UI</span>
            <span class="nav-pill">Logo Ready</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# HERO
# ----------------------------
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">AI Cricket Rain-Adjusted Score Predictor</div>
        <div class="hero-sub">
            A polished forecasting interface for <b>T20</b> and <b>ODI</b> innings,
            designed for match analysis, rain-shortening decision support, and future
            product deployment.
        </div>
        <div class="tag-row">
            <span class="tag">Professional UI</span>
            <span class="tag">Real Logo Support</span>
            <span class="tag">AI vs DLS</span>
            <span class="tag">Interactive Forecasting</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if not AVAILABLE_FORMATS:
    st.error("No trained model files were found. Please upload at least one model file.")
    st.stop()

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.markdown("## Product Info")
st.sidebar.markdown(
    """
    <div class="sidebar-box">
        <b>What this platform does</b><br><br>
        Predicts safe, expected, and aggressive first-innings totals using trained ML models,
        then compares rain-adjusted output against a DLS-style baseline.
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div class="sidebar-box">
        <b>Next upgrade path</b><br><br>
        • Official logos<br>
        • Custom brand theme<br>
        • Mobile app version<br>
        • Extra analytics modules
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# INPUTS
# ----------------------------
left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Match setup</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Select the format, team category, and venue context.</div>', unsafe_allow_html=True)

    match_format = st.selectbox("Format", AVAILABLE_FORMATS)
    batting_team = choose_team("Batting")
    bowling_team = choose_team("Bowling", exclude_team=batting_team)
    venue = st.text_input("Venue", value="Bengaluru")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Current innings state</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter live match inputs with cricket-accurate overs and balls.</div>', unsafe_allow_html=True)

    total_overs_for_format = get_total_overs(match_format)

    current_score = st.number_input("Current score", min_value=0, max_value=500, value=189, step=1)
    wickets_lost = st.number_input("Wickets lost", min_value=0, max_value=10, value=4, step=1)

    c1, c2 = st.columns(2)
    with c1:
        default_overs = 12 if match_format == "T20" else 31
        overs_completed = st.number_input(
            "Completed overs",
            min_value=0,
            max_value=total_overs_for_format,
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
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# RAIN SECTION
# ----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Rain scenario</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Activate shortened-innings analysis and test revised-over outcomes.</div>', unsafe_allow_html=True)

use_rain = st.checkbox("Apply rain-shortened innings", value=True)

default_reduced = 16 if match_format == "T20" else 41
reduced_overs = None

if use_rain:
    reduced_overs = st.slider(
        "Reduced total overs",
        min_value=5,
        max_value=total_overs_for_format,
        value=default_reduced,
        step=1
    )

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# ACTION
# ----------------------------
if st.button("Generate Forecast"):
    try:
        total_overs = total_overs_for_format
        total_balls = total_overs * 6
        balls_bowled = overs_completed * 6 + balls_into_over
        balls_remaining = total_balls - balls_bowled
        wickets_in_hand = 10 - wickets_lost
        current_rr = current_score / (balls_bowled / 6) if balls_bowled > 0 else 0

        st.markdown('<div class="match-banner">', unsafe_allow_html=True)
        c_a, c_mid, c_b = st.columns([5, 1, 5])

        with c_a:
            st.markdown('<div class="team-chip">', unsafe_allow_html=True)
            render_team_card(batting_team)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_mid:
            st.markdown(
                '<div style="display:flex;justify-content:center;align-items:center;height:100%;"><div class="vs-pill">VS</div></div>',
                unsafe_allow_html=True
            )

        with c_b:
            st.markdown('<div class="team-chip">', unsafe_allow_html=True)
            render_team_card(bowling_team)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"**Venue:** {venue} &nbsp;&nbsp;&nbsp; **Format:** {match_format}")
        st.markdown('</div>', unsafe_allow_html=True)

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

        tabs = st.tabs(["Overview", "Forecast", "Comparison"])

        with tabs[0]:
            st.subheader("Overview")
            m1, m2, m3 = st.columns(3)
            m1.metric("Safe", normal["safe"])
            m2.metric("Expected", normal["expected"])
            m3.metric("Aggressive", normal["aggressive"])

            k1, k2, k3 = st.columns(3)
            k1.metric("Wickets in hand", wickets_in_hand)
            k2.metric("Balls remaining", balls_remaining)
            k3.metric("Current RR", f"{current_rr:.2f}")

            confidence_gap = normal["aggressive"] - normal["safe"]
            if confidence_gap < 15:
                st.markdown('<div class="result-banner green-banner">High confidence prediction</div>', unsafe_allow_html=True)
            elif confidence_gap < 30:
                st.markdown('<div class="result-banner blue-banner">Moderate uncertainty</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-banner purple-banner">Low confidence — match highly unpredictable</div>', unsafe_allow_html=True)

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
                for item in insights:
                    st.markdown(f'<div class="insight-card">• {item}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="insight-card">• No major positive or negative pressure signal detected.</div>', unsafe_allow_html=True)

        with tabs[1]:
            st.subheader("Forecast")
            normal_scores, normal_probs = build_distribution_points(
                normal["safe"], normal["expected"], normal["aggressive"]
            )

            fig1 = plt.figure(figsize=(8, 4.5))
            plt.plot(normal_scores, normal_probs, marker="o", linewidth=3, color="#22c55e")
            plt.fill_between(normal_scores, normal_probs, alpha=0.20, color="#22c55e")
            plt.xlabel("Predicted final score")
            plt.ylabel("Relative likelihood")
            plt.title(f"{match_format} normal-conditions score range")
            plt.grid(True)
            style_dark_chart(fig1)
            st.pyplot(fig1)

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

                st.subheader("Rain-adjusted forecast")
                r1, r2, r3 = st.columns(3)
                r1.metric("Safe", rain["safe"])
                r2.metric("Expected", rain["expected"])
                r3.metric("Aggressive", rain["aggressive"])

                st.markdown(
                    f'<div class="result-banner green-banner">Suggested revised target: {rain["expected"] + 1}</div>',
                    unsafe_allow_html=True
                )

                rain_scores, rain_probs = build_distribution_points(
                    rain["safe"], rain["expected"], rain["aggressive"]
                )

                fig2 = plt.figure(figsize=(8, 4.5))
                plt.plot(rain_scores, rain_probs, marker="o", linewidth=3, color="#0ea5e9")
                plt.fill_between(rain_scores, rain_probs, alpha=0.20, color="#0ea5e9")
                plt.xlabel("Predicted rain-adjusted final score")
                plt.ylabel("Relative likelihood")
                plt.title(f"{match_format} rain-adjusted score range")
                plt.grid(True)
                style_dark_chart(fig2)
                st.pyplot(fig2)

        with tabs[2]:
            st.subheader("Comparison")

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

                baseline = simple_dls_baseline(
                    match_format=match_format,
                    current_score=current_score,
                    overs_completed=overs_completed,
                    balls_into_over=balls_into_over,
                    reduced_overs=reduced_overs
                )

                difference = rain["expected"] - baseline
                sign = "+" if difference >= 0 else ""

                st.markdown(
                    f'<div class="result-banner blue-banner">Simple DLS-style baseline: {baseline}</div>',
                    unsafe_allow_html=True
                )
                st.write(f"**AI vs DLS difference:** {sign}{difference} runs")

                if difference > 10:
                    st.write("AI predicts significantly higher scoring potential than the simple DLS-style baseline.")
                elif difference < -10:
                    st.write("AI predicts lower scoring potential than the simple DLS-style baseline.")
                else:
                    st.write("AI and the simple DLS-style baseline are fairly similar.")

                fig3 = plt.figure(figsize=(8, 4.5))
                labels = ["Normal", "Rain", "DLS"]
                values = [normal["expected"], rain["expected"], baseline]
                colors = ["#22c55e", "#0ea5e9", "#8b5cf6"]
                plt.bar(labels, values, color=colors)
                plt.ylabel("Score")
                plt.title("Expected Score Comparison")
                style_dark_chart(fig3)
                st.pyplot(fig3)

                st.subheader("What-if scenario")
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
            else:
                st.info("Enable rain-shortened innings to unlock AI vs DLS comparison.")
    except Exception as e:
        st.error(f"Error: {e}")