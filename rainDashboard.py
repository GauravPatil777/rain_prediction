import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Rain Prediction Dashboard",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS - Dark Atmospheric Theme
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

    /* Global Reset & Background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 40%, #0f2336 100%);
        font-family: 'Syne', sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #091525 100%);
        border-right: 1px solid rgba(56, 189, 248, 0.15);
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] p {
        color: #94a3b8 !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background: rgba(15, 35, 54, 0.8) !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }

    /* Main title */
    h1 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
        font-size: 2.4rem !important;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        padding-bottom: 0.5rem;
    }

    /* Subheaders */
    h2, h3 {
        font-family: 'Syne', sans-serif !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }

    /* Metric Cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(15, 35, 54, 0.9), rgba(13, 27, 42, 0.9));
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    [data-testid="metric-container"]:hover {
        border-color: rgba(56, 189, 248, 0.5);
        box-shadow: 0 8px 32px rgba(56, 189, 248, 0.1);
        transform: translateY(-2px);
    }
    [data-testid="metric-container"] label {
        color: #64748b !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
    }

    /* DataFrames */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden;
        border: 1px solid rgba(56, 189, 248, 0.15) !important;
    }
    .stDataFrame thead tr th {
        background: rgba(56, 189, 248, 0.1) !important;
        color: #38bdf8 !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.75rem !important;
    }

    /* Select boxes */
    .stSelectbox > div > div {
        background: rgba(15, 35, 54, 0.8) !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        color: #e2e8f0 !important;
        border-radius: 10px !important;
        font-family: 'Syne', sans-serif !important;
    }
    .stSelectbox label {
        color: #94a3b8 !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* Sliders */
    .stSlider label {
        color: #94a3b8 !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .stSlider [data-testid="stThumbValue"] {
        color: #38bdf8 !important;
        font-family: 'Space Mono', monospace !important;
    }
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: #38bdf8 !important;
        border: 3px solid #0d1b2a !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.6) !important;
    }

    /* Predict Button */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.05em;
        padding: 0.75rem 2.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(14, 165, 233, 0.35) !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(14, 165, 233, 0.5) !important;
    }

    /* Success / info boxes */
    .stSuccess {
        background: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid rgba(16, 185, 129, 0.4) !important;
        border-radius: 12px !important;
        color: #6ee7b7 !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        text-align: center;
    }

    /* Section Divider */
    hr {
        border-color: rgba(56, 189, 248, 0.1) !important;
        margin: 1.5rem 0;
    }

    /* Info card container */
    .info-card {
        background: linear-gradient(135deg, rgba(15,35,54,0.95), rgba(13,27,42,0.95));
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }

    /* Prediction result card */
    .result-rain {
        background: linear-gradient(135deg, rgba(6,78,137,0.4), rgba(14,165,233,0.15));
        border: 2px solid rgba(56, 189, 248, 0.5);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        font-family: 'Syne', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: #38bdf8;
        box-shadow: 0 0 40px rgba(56, 189, 248, 0.15), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .result-sun {
        background: linear-gradient(135deg, rgba(120,53,15,0.4), rgba(251,191,36,0.15));
        border: 2px solid rgba(251, 191, 36, 0.5);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        font-family: 'Syne', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: #fbbf24;
        box-shadow: 0 0 40px rgba(251, 191, 36, 0.1), inset 0 1px 0 rgba(255,255,255,0.05);
    }

    /* Stat description text */
    p, .stMarkdown p {
        color: #94a3b8 !important;
        font-family: 'Syne', sans-serif !important;
    }

    /* Chart containers */
    .js-plotly-plot {
        border-radius: 16px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("<h1>🌧️ Rain Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown('<p style="color:#64748b; font-family: Space Mono, monospace; font-size:0.8rem; letter-spacing:0.1em; margin-top:-1rem; margin-bottom:1.5rem;">USA WEATHER INTELLIGENCE · 2024–2025</p>', unsafe_allow_html=True)

# -----------------------------
# Plotly chart theme
# -----------------------------
CHART_THEME = {
    "paper_bgcolor": "rgba(10,14,26,0)",
    "plot_bgcolor": "rgba(13,27,42,0.6)",
    "font": {"color": "#94a3b8", "family": "Space Mono, monospace", "size": 11},
    "gridcolor": "rgba(56,189,248,0.08)",
    "linecolor": "rgba(56,189,248,0.15)",
}

COLOR_SEQ = ["#38bdf8", "#818cf8", "#c084fc", "#f472b6", "#fb923c", "#34d399"]
RAIN_COLORS = {"0": "#38bdf8", "1": "#c084fc", 0: "#38bdf8", 1: "#c084fc"}

def style_chart(fig, title=""):
    fig.update_layout(
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        title=dict(text=title, font=dict(color="#e2e8f0", size=16, family="Syne, sans-serif"), x=0.02),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(gridcolor=CHART_THEME["gridcolor"], linecolor=CHART_THEME["linecolor"], zerolinecolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor=CHART_THEME["gridcolor"], linecolor=CHART_THEME["linecolor"], zerolinecolor="rgba(0,0,0,0)"),
        legend=dict(bgcolor="rgba(13,27,42,0.8)", bordercolor="rgba(56,189,248,0.2)", borderwidth=1),
    )
    return fig

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("usa_rain_prediction_dataset_2024_2025.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown('<p style="color:#38bdf8; font-family: Space Mono, monospace; font-size:0.7rem; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:0.5rem;">Navigation</p>', unsafe_allow_html=True)
    menu = st.selectbox(
        "Select Section",
        ["📊 Dataset Overview", "📈 Data Visualization", "🤖 Rain Prediction"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(f'<p style="color:#475569; font-family: Space Mono, monospace; font-size:0.65rem; text-transform:uppercase; letter-spacing:0.1em;">Dataset Info</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:#64748b; font-family: Space Mono, monospace; font-size:0.7rem;">📍 {df["Location"].nunique()} Locations</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:#64748b; font-family: Space Mono, monospace; font-size:0.7rem;">📋 {len(df):,} Records</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:#64748b; font-family: Space Mono, monospace; font-size:0.7rem;">📅 {df["Date"].min().strftime("%b %Y")} – {df["Date"].max().strftime("%b %Y")}</p>', unsafe_allow_html=True)

# =============================
# 1 DATASET OVERVIEW
# =============================
if menu == "📊 Dataset Overview":
    st.subheader("Dataset Preview")

    col1, col2, col3, col4 = st.columns(4)
    rain_pct = round(df["Rain Tomorrow"].mean() * 100, 1)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Locations", df["Location"].nunique())
    col3.metric("Rain Days", f"{int(df['Rain Tomorrow'].sum()):,}")
    col4.metric("Rain Rate", f"{rain_pct}%")

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(
        df.head(10).style.set_properties(**{"background-color": "transparent", "color": "#e2e8f0"}),
        use_container_width=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Statistical Summary")

    desc = df.describe().round(2)
    st.dataframe(desc, use_container_width=True)

    # Quick overview charts in 2 columns
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Quick Insights")
    c1, c2 = st.columns(2)

    with c1:
        rain_counts = df["Rain Tomorrow"].value_counts().reset_index()
        rain_counts.columns = ["Rain Tomorrow", "Count"]
        rain_counts["Label"] = rain_counts["Rain Tomorrow"].map({0: "No Rain", 1: "Rain"})
        fig = px.pie(
            rain_counts, values="Count", names="Label",
            color_discrete_sequence=["#38bdf8", "#c084fc"],
            hole=0.55
        )
        fig = style_chart(fig, "Rain Tomorrow Distribution")
        fig.update_traces(textfont_color="#e2e8f0", marker=dict(line=dict(color="#0a0e1a", width=2)))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        monthly = df.copy()
        monthly["Month"] = monthly["Date"].dt.strftime("%b")
        monthly["MonthNum"] = monthly["Date"].dt.month
        rain_monthly = monthly.groupby(["MonthNum", "Month"])["Rain Tomorrow"].mean().reset_index()
        rain_monthly = rain_monthly.sort_values("MonthNum")
        fig = px.bar(
            rain_monthly, x="Month", y="Rain Tomorrow",
            color="Rain Tomorrow",
            color_continuous_scale=["#0ea5e9", "#c084fc", "#f472b6"]
        )
        fig = style_chart(fig, "Monthly Rain Probability")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

# =============================
# 2 DATA VISUALIZATION
# =============================
elif menu == "📈 Data Visualization":
    st.subheader("Interactive Weather Visualizations")

    graph = st.selectbox(
        "Choose Visualization",
        [
            "🌡️ Temperature Distribution",
            "💧 Humidity Distribution",
            "🌬️ Pressure Distribution",
            "☔ Rain Tomorrow Count",
            "🔥 Temperature vs Humidity",
            "💨 Wind Speed vs Rain",
            "📅 Weather Over Time",
            "🗺️ Rain by Location"
        ]
    )

    if graph == "🌡️ Temperature Distribution":
        fig = px.histogram(
            df, x="Temperature", nbins=40,
            color_discrete_sequence=["#38bdf8"],
            opacity=0.85
        )
        fig.update_traces(marker_line_color="#0ea5e9", marker_line_width=0.5)
        fig = style_chart(fig, "Temperature Distribution (°F)")
        st.plotly_chart(fig, use_container_width=True)

    elif graph == "💧 Humidity Distribution":
        fig = px.histogram(
            df, x="Humidity", nbins=40,
            color_discrete_sequence=["#818cf8"],
            opacity=0.85
        )
        fig.update_traces(marker_line_color="#6366f1", marker_line_width=0.5)
        fig = style_chart(fig, "Humidity Distribution (%)")
        st.plotly_chart(fig, use_container_width=True)

    elif graph == "🌬️ Pressure Distribution":
        fig = px.histogram(
            df, x="Pressure", nbins=40,
            color_discrete_sequence=["#c084fc"],
            opacity=0.85
        )
        fig.update_traces(marker_line_color="#a855f7", marker_line_width=0.5)
        fig = style_chart(fig, "Pressure Distribution (hPa)")
        st.plotly_chart(fig, use_container_width=True)

    elif graph == "☔ Rain Tomorrow Count":
        rain_counts = df["Rain Tomorrow"].value_counts().reset_index()
        rain_counts.columns = ["Rain Tomorrow", "Count"]
        rain_counts["Label"] = rain_counts["Rain Tomorrow"].map({0: "☀️ No Rain", 1: "🌧️ Rain"})
        fig = px.bar(
            rain_counts, x="Label", y="Count",
            color="Label",
            color_discrete_map={"☀️ No Rain": "#fbbf24", "🌧️ Rain": "#38bdf8"},
            text="Count"
        )
        fig.update_traces(marker_line_color="rgba(0,0,0,0.3)", marker_line_width=1, textfont_color="#e2e8f0")
        fig = style_chart(fig, "Rain Tomorrow: Count Breakdown")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    elif graph == "🔥 Temperature vs Humidity":
        sample = df.sample(min(2000, len(df)), random_state=42)
        sample["Rain Label"] = sample["Rain Tomorrow"].map({0: "No Rain", 1: "Rain"})
        fig = px.scatter(
            sample, x="Temperature", y="Humidity",
            color="Rain Label",
            color_discrete_map={"No Rain": "#fbbf24", "Rain": "#38bdf8"},
            opacity=0.65,
            marginal_x="histogram",
            marginal_y="histogram"
        )
        fig = style_chart(fig, "Temperature vs Humidity (colored by Rain)")
        st.plotly_chart(fig, use_container_width=True)

    elif graph == "💨 Wind Speed vs Rain":
        df_plot = df.copy()
        df_plot["Rain Label"] = df_plot["Rain Tomorrow"].map({0: "No Rain", 1: "Rain"})
        fig = px.box(
            df_plot, x="Rain Label", y="Wind Speed",
            color="Rain Label",
            color_discrete_map={"No Rain": "#fbbf24", "Rain": "#38bdf8"},
            points="outliers"
        )
        fig.update_traces(marker_color="#c084fc", marker_opacity=0.4)
        fig = style_chart(fig, "Wind Speed Distribution by Rain Outcome")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    elif graph == "📅 Weather Over Time":
        time_df = df.copy()
        time_df["Month"] = time_df["Date"].dt.to_period("M").astype(str)
        monthly_avg = time_df.groupby("Month")[["Temperature", "Humidity", "Pressure"]].mean().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_avg["Month"], y=monthly_avg["Temperature"],
                                  name="Temperature", line=dict(color="#f87171", width=2.5), mode="lines+markers",
                                  marker=dict(size=6, color="#f87171")))
        fig.add_trace(go.Scatter(x=monthly_avg["Month"], y=monthly_avg["Humidity"],
                                  name="Humidity", line=dict(color="#38bdf8", width=2.5), mode="lines+markers",
                                  marker=dict(size=6, color="#38bdf8")))
        fig = style_chart(fig, "Monthly Average: Temperature & Humidity")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    elif graph == "🗺️ Rain by Location":
        loc_rain = df.groupby("Location")["Rain Tomorrow"].agg(["mean", "sum", "count"]).reset_index()
        loc_rain.columns = ["Location", "Rain Rate", "Rain Days", "Total Days"]
        loc_rain = loc_rain.sort_values("Rain Rate", ascending=True)
        fig = px.bar(
            loc_rain, x="Rain Rate", y="Location",
            orientation="h",
            color="Rain Rate",
            color_continuous_scale=["#0ea5e9", "#818cf8", "#f472b6"],
            text=loc_rain["Rain Rate"].apply(lambda x: f"{x:.1%}")
        )
        fig.update_traces(textfont_color="#e2e8f0")
        fig = style_chart(fig, "Rain Probability by Location")
        fig.update_layout(height=max(400, len(loc_rain) * 28), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

# =============================
# 3 RAIN PREDICTION
# =============================
elif menu == "🤖 Rain Prediction":
    st.subheader("Predict Rain Tomorrow")
    st.markdown('<p style="color:#64748b; font-size:0.85rem;">Adjust weather parameters below to get a prediction.</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])

    with col_a:
        c1, c2 = st.columns(2)
        with c1:
            temperature = st.slider("🌡️ Temperature (°F)", 30, 100, 60)
            humidity = st.slider("💧 Humidity (%)", 20, 100, 50)
            wind_speed = st.slider("💨 Wind Speed (mph)", 0, 50, 10)
        with c2:
            precipitation = st.slider("🌧️ Precipitation (in)", 0.0, 10.0, 1.0)
            cloud_cover = st.slider("☁️ Cloud Cover (%)", 0, 100, 50)
            pressure = st.slider("🔵 Pressure (hPa)", 970, 1040, 1010)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡ Predict Rain Tomorrow")

    with col_b:
        # Gauge-style indicator
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=humidity,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Humidity %", "font": {"color": "#94a3b8", "family": "Space Mono", "size": 13}},
            number={"font": {"color": "#38bdf8", "family": "Syne", "size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#475569", "size": 10}},
                "bar": {"color": "#38bdf8"},
                "bgcolor": "rgba(13,27,42,0.8)",
                "borderwidth": 1,
                "bordercolor": "rgba(56,189,248,0.2)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(251,191,36,0.15)"},
                    {"range": [40, 70], "color": "rgba(56,189,248,0.1)"},
                    {"range": [70, 100], "color": "rgba(192,132,252,0.2)"},
                ],
                "threshold": {
                    "line": {"color": "#f472b6", "width": 2},
                    "thickness": 0.75,
                    "value": 70
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
            height=240,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Radar chart of current conditions
        categories = ["Temp", "Humidity", "Wind", "Precip×10", "Cloud", "Pressure-940"]
        values_raw = [temperature, humidity, wind_speed, precipitation * 10, cloud_cover, pressure - 940]
        maxvals = [100, 100, 50, 100, 100, 100]
        norm = [min(v / m * 100, 100) for v, m in zip(values_raw, maxvals)]

        fig_radar = go.Figure(go.Scatterpolar(
            r=norm + [norm[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(56,189,248,0.12)",
            line=dict(color="#38bdf8", width=2),
            marker=dict(color="#38bdf8", size=5)
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(13,27,42,0.6)",
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(56,189,248,0.1)", tickfont=dict(color="#475569", size=9)),
                angularaxis=dict(gridcolor="rgba(56,189,248,0.1)", tickfont=dict(color="#94a3b8", size=10, family="Space Mono"))
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=240,
            margin=dict(l=30, r=30, t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    if predict_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        # Prediction logic (extended)
        rain_score = 0
        if humidity > 70: rain_score += 2
        if cloud_cover > 60: rain_score += 2
        if precipitation > 2.0: rain_score += 2
        if wind_speed > 20: rain_score += 1
        if pressure < 1005: rain_score += 1
        if temperature < 50: rain_score += 1

        if rain_score >= 4:
            st.markdown('<div class="result-rain">🌧️ Rain Expected Tomorrow<br><span style="font-size:0.9rem; color:#7dd3fc; font-weight:400;">High probability based on current conditions</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-sun">☀️ No Rain Expected Tomorrow<br><span style="font-size:0.9rem; color:#fde68a; font-weight:400;">Conditions indicate dry weather</span></div>', unsafe_allow_html=True)

        # Confidence bar
        st.markdown("<br>", unsafe_allow_html=True)
        confidence = min(rain_score / 8 * 100, 100)
        c_col1, c_col2, c_col3 = st.columns(3)
        c_col1.metric("Rain Score", f"{rain_score}/8")
        c_col2.metric("Confidence", f"{confidence:.0f}%")
        c_col3.metric("Status", "🌧️ Rain" if rain_score >= 4 else "☀️ Clear")