"""
System Capacity & Care Load Analytics for Unaccompanied Children
Unified Mentor Internship Project — Data Analyst Batch Jan 2026
HHS / U.S. Department of Health and Human Services
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pyngrok import ngrok

public_url = ngrok.connect(8501)
print("HTTPS URL:", public_url)

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UAC Care Load Analytics | HHS",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Data Loading & Structuring ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program.csv")

    # Drop blank rows
    df = df.dropna(subset=["Date"])

    df.columns = [
        "Date", "CBP_Apprehended", "CBP_Custody",
        "CBP_Transferred", "HHS_Care", "HHS_Discharged"
    ]

    # Clean HHS_Care comma formatting
    df["HHS_Care"] = (
        df["HHS_Care"].astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["HHS_Care"] = pd.to_numeric(df["HHS_Care"], errors="coerce")

    # Convert Date to datetime, chronological ordering
    df["Date"] = pd.to_datetime(df["Date"], format="%B %d, %Y", errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # ── Derived Healthcare Capacity Metrics (as required) ─────────────────────
    # Total System Load: CBP Custody + HHS Care
    df["Total_System_Load"] = df["CBP_Custody"] + df["HHS_Care"]

    # Net Daily Intake: Transfers into HHS − Discharges from HHS
    df["Net_Daily_Intake"] = df["CBP_Transferred"] - df["HHS_Discharged"]

    # Care Load Growth Rate: day-over-day percentage change
    df["Care_Load_Growth"] = df["HHS_Care"].pct_change() * 100

    # Backlog Indicator: cumulative net intake (sustained positive net intake)
    df["Backlog"] = df["Net_Daily_Intake"].cumsum()

    # Rolling averages: 7-day and 14-day
    df["Rolling7"]  = df["HHS_Care"].rolling(7,  min_periods=1).mean()
    df["Rolling14"] = df["HHS_Care"].rolling(14, min_periods=1).mean()

    # Time columns
    df["Year"]      = df["Date"].dt.year
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
    df["Week"]      = df["Date"].dt.to_period("W").astype(str)

    return df


df = load_data()
# print(df['Date'])

# ─── Sidebar — User Capabilities ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## UAC Analytics Dashboard")
    st.markdown("**U.S. Dept of Health & Human Services**  \nUnified Mentor · Data Analyst Intern · Jan 2026")
    st.divider()

    # Date range selector
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    date_range = st.date_input(
        "Date range selector",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Time granularity filter
    granularity = st.selectbox(
        "Time granularity",
        ["Daily", "Weekly", "Monthly"]
    )

    # Metric toggles
    metric_toggles = st.multiselect(
        "Metric toggles",
        ["HHS Care", "CBP Custody", "Total System Load"],
        default=["HHS Care", "CBP Custody", "Total System Load"]
    )

    st.divider()
    st.caption("Data: HHS UAC Program · Jan 2026 - July 2026 · 180 reporting days")


# ─── Filter by date range ─────────────────────────────────────────────────────
dff = df.copy()
if len(date_range) == 2:
    s, e = date_range
    dff = dff[(dff["Date"].dt.date >= s) & (dff["Date"].dt.date <= e)]


# ─── Apply time granularity ───────────────────────────────────────────────────
def aggregate(data, gran):
    if gran == "Monthly":
        grp = data.groupby("YearMonth").agg(
            HHS_Care          = ("HHS_Care",          "mean"),
            CBP_Custody       = ("CBP_Custody",        "mean"),
            Total_System_Load = ("Total_System_Load",  "mean"),
            Net_Daily_Intake  = ("Net_Daily_Intake",   "sum"),
            Backlog           = ("Backlog",             "last"),
            CBP_Transferred   = ("CBP_Transferred",    "sum"),
            HHS_Discharged    = ("HHS_Discharged",     "sum"),
            Rolling7          = ("Rolling7",            "mean"),
            Rolling14         = ("Rolling14",           "mean"),
        ).reset_index().rename(columns={"YearMonth": "Period"})
    elif gran == "Weekly":
        grp = data.groupby("Week").agg(
            HHS_Care          = ("HHS_Care",          "mean"),
            CBP_Custody       = ("CBP_Custody",        "mean"),
            Total_System_Load = ("Total_System_Load",  "mean"),
            Net_Daily_Intake  = ("Net_Daily_Intake",   "sum"),
            Backlog           = ("Backlog",             "last"),
            CBP_Transferred   = ("CBP_Transferred",    "sum"),
            HHS_Discharged    = ("HHS_Discharged",     "sum"),
            Rolling7          = ("Rolling7",            "mean"),
            Rolling14         = ("Rolling14",           "mean"),
        ).reset_index().rename(columns={"Week": "Period"})
    else:
        data = data.copy()
        data["Period"] = data["Date"].dt.strftime("%Y-%m-%d")
        grp = data[["Period","HHS_Care","CBP_Custody","Total_System_Load",
                    "Net_Daily_Intake","Backlog","CBP_Transferred",
                    "HHS_Discharged","Rolling7","Rolling14"]]
    return grp


agg = aggregate(dff, granularity)

# ─── KPI calculations ─────────────────────────────────────────────────────────
latest = dff.iloc[-1] if len(dff) else df.iloc[-1]

# KPI: Total Children Under Care (system-wide responsibility)
total_under_care = int(latest["HHS_Care"] + latest["CBP_Custody"])

# KPI: Net Intake Pressure (inflow vs outflow imbalance)
net_intake_pressure = round(dff["Net_Daily_Intake"].mean(), 1)

# KPI: Care Load Volatility Index (stability of system)
volatility_index = round(dff["Total_System_Load"].std(), 1)

# KPI: Backlog Accumulation Rate (sustained care pressure)
backlog_accum = int(dff["Backlog"].iloc[-1]) if len(dff) else 0

# KPI: Discharge Offset Ratio (ability to relieve load)
total_transferred = dff["CBP_Transferred"].sum()
discharge_offset  = round(
    (dff["HHS_Discharged"].sum() / total_transferred) * 100, 1
) if total_transferred > 0 else 0


# ─── Page Header ─────────────────────────────────────────────────────────────
st.markdown("# 🏥 System Capacity & Care Load Analytics — UAC Program")
st.markdown(
    f"**U.S. Department of Health and Human Services** · "
    f"Unaccompanied Alien Children Program · "
    f"**{dff['Date'].min().strftime('%b %Y')} – {dff['Date'].max().strftime('%b %Y')}**"
)
st.divider()


# ─── KPI Summary Cards ───────────────────────────────────────────────────────
st.subheader("📊 KPI Summary Cards")
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric("Total Children Under Care",   f"{total_under_care:,}",
              help="System-wide responsibility")
with k2:
    st.metric("Net Intake Pressure",          f"{net_intake_pressure:+}",
              help="Inflow vs outflow imbalance")
with k3:
    st.metric("Care Load Volatility Index",   f"{volatility_index:,.0f}",
              help="Stability of system")
with k4:
    st.metric("Backlog Accumulation Rate",    f"{backlog_accum:,}",
              help="Sustained care pressure")
with k5:
    st.metric("Discharge Offset Ratio",       f"{discharge_offset}%",
              help="Ability to relieve load")

st.divider()


# ─── Core Modules (as required) ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📈 System Load Overview",
    "⚖️ CBP vs HHS Load Comparison",
    "📥 Net Intake & Backlog Trends"
])

color_map = {
    "HHS Care":          "#378ADD",
    "CBP Custody":       "#D85A30",
    "Total System Load": "#1D9E75"
}
col_map = {
    "HHS Care":          "HHS_Care",
    "CBP Custody":       "CBP_Custody",
    "Total System Load": "Total_System_Load"
}


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — SYSTEM LOAD OVERVIEW PANE
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("System Load Overview")

    # Total system load trend
    fig1 = go.Figure()
    for m in metric_toggles:
        fig1.add_trace(go.Scatter(
            x=agg["Period"],
            y=agg[col_map[m]].round().astype(int),
            name=m,
            mode="lines+markers",
            line=dict(color=color_map[m], width=2),
            marker=dict(size=3)
        ))
    fig1.update_layout(
        height=400, template="plotly_white",
        hovermode="x unified",
        xaxis_title=granularity, yaxis_title="Children",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    fig1.update_yaxes(tickformat=",d")
    fig1.update_xaxes(tickangle=45, tickfont=dict(size=9))
    st.plotly_chart(fig1, width="stretch")

    # 7-day & 14-day rolling averages
    st.subheader("Rolling Averages — 7-day & 14-day (HHS Care)")
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(
        x=dff["Date"], y=dff["HHS_Care"],
        name="Daily HHS Care", mode="lines",
        line=dict(color="#B5D4F4", width=1), opacity=0.5
    ))
    fig_roll.add_trace(go.Scatter(
        x=dff["Date"], y=dff["Rolling7"].round().astype(int),
        name="7-day Avg", mode="lines",
        line=dict(color="#378ADD", width=2)
    ))
    fig_roll.add_trace(go.Scatter(
        x=dff["Date"], y=dff["Rolling14"].round().astype(int),
        name="14-day Avg", mode="lines",
        line=dict(color="#185FA5", width=2, dash="dash")
    ))
    fig_roll.update_layout(
        height=300, template="plotly_white",
        hovermode="x unified",
        yaxis_title="Children in HHS Care",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    fig_roll.update_yaxes(tickformat=",d")
    st.plotly_chart(fig_roll, width="stretch")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — CBP VS HHS LOAD COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("CBP Custody vs HHS Care — Load Comparison")

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=agg["Period"], y=agg["CBP_Custody"].round().astype(int),
        name="CBP Custody", marker_color="#D85A30", opacity=0.85
    ))
    fig2.add_trace(go.Bar(
        x=agg["Period"], y=agg["HHS_Care"].round().astype(int),
        name="HHS Care", marker_color="#378ADD", opacity=0.85
    ))
    fig2.update_layout(
        barmode="group", height=420, template="plotly_white",
        hovermode="x unified",
        yaxis_title="Children",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    fig2.update_yaxes(tickformat=",d")
    fig2.update_xaxes(tickangle=45, tickfont=dict(size=9))
    st.plotly_chart(fig2, width="stretch")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — NET INTAKE & BACKLOG TRENDS
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Net Intake & Backlog Trends")

    # Net Daily Intake
    st.markdown("**Net Daily Intake (Transfers into HHS − Discharges)**")
    net_vals = agg["Net_Daily_Intake"].fillna(0)
    fig3 = go.Figure(go.Bar(
        x=agg["Period"],
        y=net_vals.round().astype(int),
        marker_color=["#E24B4A" if v > 0 else "#1D9E75" for v in net_vals],
        name="Net Intake"
    ))
    fig3.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
    fig3.update_layout(
        height=320, template="plotly_white",
        hovermode="x unified",
        yaxis_title="Net Children",
        margin=dict(l=10, r=10, t=10, b=10)
    )
    fig3.update_yaxes(tickformat=",d")
    fig3.update_xaxes(tickangle=45, tickfont=dict(size=9))
    st.plotly_chart(fig3, width="stretch")

    # Backlog Accumulation
    st.markdown("**Backlog Accumulation — Sustained Positive Net Intake Over Time**")
    fig_bl = go.Figure()
    fig_bl.add_trace(go.Scatter(
        x=dff["Date"], y=dff["Backlog"],
        mode="lines", fill="tozeroy",
        fillcolor="rgba(226,75,74,0.08)",
        line=dict(color="#E24B4A", width=2),
        name="Backlog"
    ))
    fig_bl.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
    fig_bl.update_layout(
        height=280, template="plotly_white",
        hovermode="x unified",
        yaxis_title="Cumulative Net Children",
        margin=dict(l=10, r=10, t=10, b=10)
    )
    fig_bl.update_yaxes(tickformat=",d")
    st.plotly_chart(fig_bl, width="stretch")


# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='font-size:12px;color:#999;text-align:center;'>"
    "System Capacity & Care Load Analytics · HHS UAC Program · "
    "Unified Mentor Internship Jan 2026 Batch · Built with Streamlit & Plotly"
    "</p>",
    unsafe_allow_html=True
)
