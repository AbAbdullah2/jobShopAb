# upmc.py
# Neurosurgery demo: Dynamic vs Template scheduling (Streamlit)
# - RESULT (Plotly): lines only, shaded bands (Mean→Max) for Static / 3 Categories / Dynamic
#   with clock x-axis 08:00 → 10:55 (175 minutes) to match your results figure.
# - SCHEDULE: Built directly from jobshop results (soln3Times.csv), limited to 11:50 (08:00→11:50),
#   with labels on each appointment bar (e.g., "PAT 17"). Tabs per doctor include DF + mini timeline.

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import os

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────────────────────
# Page
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Neurosurgery Scheduling Demo", layout="wide")
st.title("UPMC Neurosurgery - Dynamic vs Template Based Scheduling")

# ──────────────────────────────────────────────────────────────────────────────
# File helpers
# ──────────────────────────────────────────────────────────────────────────────
def find_csv_dir() -> Optional[str]:
    candidates = [
        ".",  # current dir
        "17.01 UPMC Demo",
        "/mnt/data/jobShop-main/jobShop-main/17.01 UPMC Demo",
        "/mnt/data/jobShop-main/17.01 UPMC Demo",
        "/mnt/data",
    ]
    for base in candidates:
        paths = [os.path.join(base, f) for f in ("solnStatic.csv", "soln3Times.csv", "solnDynamic.csv")]
        if all(os.path.exists(p) for p in paths):
            return base
    for root, _, files in os.walk("."):
        if {"solnStatic.csv", "soln3Times.csv", "solnDynamic.csv"}.issubset(set(files)):
            return root
    return None

csv_dir = find_csv_dir()

def load_result_data() -> Dict[str, pd.DataFrame]:
    if not csv_dir:
        return {}
    df1 = pd.read_csv(f"{csv_dir}/solnStatic.csv", index_col=0)
    df2 = pd.read_csv(f"{csv_dir}/soln3Times.csv", index_col=0)
    df3 = pd.read_csv(f"{csv_dir}/solnDynamic.csv", index_col=0)
    need = {"arrival_time", "flow_time"}
    for d in (df1, df2, df3):
        if not need.issubset(d.columns):
            raise RuntimeError("CSV columns must include arrival_time and flow_time.")
    return {"Static": df1, "3 Categories": df2, "Dynamic": df3}

graphs_data = load_result_data() if csv_dir else {}

# ──────────────────────────────────────────────────────────────────────────────
# RESULT chart — Plotly (lines only, shaded for all 3, clock x-axis 08:00→10:55)
# ──────────────────────────────────────────────────────────────────────────────
st.header("Result")

if not graphs_data:
    st.error("Could not find solnStatic.csv / soln3Times.csv / solnDynamic.csv.")
else:
    strategy_choices = ["Static", "3 Categories", "Dynamic"]
    show = st.multiselect("Show strategies", strategy_choices, default=strategy_choices)

    colors = {
        "Static":       "rgb(220, 20, 60)",   # red
        "3 Categories": "rgb(255, 140, 0)",   # orange
        "Dynamic":      "rgb(34, 139, 34)",   # green
    }
    fills = {
        "Static":       "rgba(220, 20, 60, 0.15)",
        "3 Categories": "rgba(255, 140, 0, 0.15)",
        "Dynamic":      "rgba(34, 139, 34, 0.15)",
    }

    # Aggregate like notebook; clamp to 0..175 min
    agg = {}
    for name in strategy_choices:
        df = graphs_data[name]
        mean_s = df.groupby("arrival_time")["flow_time"].mean()
        max_s  = df.groupby("arrival_time")["flow_time"].max()
        mean_s = mean_s[(mean_s.index >= 0) & (mean_s.index <= 175)].sort_index()
        max_s  = max_s[(max_s.index >= 0) & (max_s.index <= 175)].sort_index()
        agg[name] = (mean_s, max_s)

    base_dt_res = pd.Timestamp("2024-01-01 08:00:00")
    def mins_to_dt_list(min_list: List[int]) -> List[pd.Timestamp]:
        return [base_dt_res + pd.to_timedelta(int(m), unit="m") for m in min_list]

    fig = go.Figure()

    # Thin black legend items (Mean/Max)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             line=dict(color="black", width=2, dash="solid"),
                             name="Mean", showlegend=True, hoverinfo="skip", legendgroup="style"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             line=dict(color="black", width=2, dash="dash"),
                             name="Max", showlegend=True, hoverinfo="skip", legendgroup="style"))

    # Thick color swatches for strategies
    for name in strategy_choices:
        if name in show:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                     line=dict(color=colors[name], width=12),
                                     name=name, showlegend=True, hoverinfo="skip", legendgroup=name))

    for name in strategy_choices:
        if name not in show:
            continue
        mean_s, max_s = agg[name]
        if len(mean_s) == 0: continue

        x_mins = sorted(set(mean_s.index).intersection(set(max_s.index)))
        x_dt   = mins_to_dt_list(x_mins)
        y_mean = [float(mean_s.loc[m]) for m in x_mins]
        y_max  = [float(max_s.loc[m])  for m in x_mins]

        # Shaded band
        fig.add_trace(go.Scatter(
            x = x_dt + x_dt[::-1],
            y = y_max + y_mean[::-1],
            fill="toself", fillcolor=fills[name],
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
            legendgroup=name
        ))
        # Mean (solid)
        fig.add_trace(go.Scatter(
            x=x_dt, y=y_mean, mode="lines",
            line=dict(color=colors[name], width=2, dash="solid"),
            name=f"{name} Mean", showlegend=False, legendgroup=name
        ))
        # Max (dashed)
        fig.add_trace(go.Scatter(
            x=x_dt, y=y_max, mode="lines",
            line=dict(color=colors[name], width=2, dash="dash"),
            name=f"{name} Max", showlegend=False, legendgroup=name
        ))

    # 08:00 → 10:55 (175 min)
    tick_mins  = [0, 30, 60, 90, 120, 150, 175]
    tick_vals  = mins_to_dt_list(tick_mins)
    tick_text  = [(base_dt_res + pd.to_timedelta(m, "m")).strftime("%H:%M") for m in tick_mins]
    end_dt_res = base_dt_res + pd.to_timedelta(175, "m")

    fig.update_layout(
        template="plotly_white",
        height=460,
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis=dict(
            title="Arrival time (clock)",
            range=[base_dt_res, end_dt_res],
            tickmode="array", tickvals=tick_vals, ticktext=tick_text,
            zeroline=False, type="date"
        ),
        yaxis=dict(title="Wait-Time (minutes)", zeroline=False),
        title="Mean/Max Wait-Time by Arrival Time",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )

    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# SCHEDULE — jobshop results with labels, limited to 11:50 (08:00→11:50)
# ──────────────────────────────────────────────────────────────────────────────
st.header("Schedule")

# Display window for schedule view:
SCHED_BASE_DT = pd.Timestamp("2024-01-01 08:00:00")
SCHED_END_DT  = pd.Timestamp("2024-01-01 11:50:00")  # 08:00 → 11:50

def load_jobshop_schedule() -> Optional[pd.DataFrame]:
    if not csv_dir:
        return None
    path = os.path.join(csv_dir, "soln3Times.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    need = {"job_id", "machine", "start_time", "end_time", "processing_time", "arrival_time"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        st.error(f"soln3Times.csv missing columns: {missing}")
        return None
    return df

def minutes_to_clock_ts(minutes: int) -> pd.Timestamp:
    return SCHED_BASE_DT + pd.to_timedelta(int(minutes), unit="m")

raw_sched = load_jobshop_schedule()

if raw_sched is None or raw_sched.empty:
    st.error("Could not load jobshop results from soln3Times.csv.")
else:
    # Build doctor_dfs with Start/Finish timestamps and type by duration
    doctor_dfs: Dict[int, pd.DataFrame] = {}
    for doctor_num in range(8):
        d = raw_sched[raw_sched["machine"] == doctor_num].copy()
        d = d[["job_id", "start_time", "end_time", "processing_time", "arrival_time"]]
        d = d.rename(columns={"job_id": "patient_id"})
        d["StartDT"]   = d["start_time"].apply(minutes_to_clock_ts)
        d["FinishDT"]  = d["end_time"].apply(minutes_to_clock_ts)
        d["ArrivalDT"] = d["arrival_time"].apply(minutes_to_clock_ts)
        def appt_type(proc):
            if proc < 10: return "Recurring Follow-up"
            elif proc <= 20: return "Follow-up"
            return "New"
        d["Type"] = d["processing_time"].apply(appt_type)
        d["Doctor"] = f"Doctor {doctor_num}"
        d = d.sort_values("StartDT").reset_index(drop=True)
        doctor_dfs[doctor_num] = d

    # Concatenate for All Doctors timeline
    df_all = pd.concat([doctor_dfs[i] for i in range(8)], ignore_index=True)

    # Colors to match your Matplotlib legend
    type_colors = {
        "New": "lightcoral",
        "Follow-up": "lightyellow",
        "Recurring Follow-up": "lightgreen",
    }

    # Limit view to 08:00 → 11:50
    def clip_view(df: pd.DataFrame) -> pd.DataFrame:
        # Keep rows that start before end of window and finish after start of window
        mask = (df["StartDT"] < SCHED_END_DT) & (df["FinishDT"] > SCHED_BASE_DT)
        return df.loc[mask].copy()

    # All Doctors timeline with labels on bars ("PAT #")
    st.subheader("All Doctors — Timeline (08:00→11:50)")
    df_all_view = clip_view(df_all)
    if df_all_view.empty:
        st.info("No scheduled patients in this window.")
    else:
        df_all_view["Label"] = "PAT " + df_all_view["patient_id"].astype(str)
        fig_all = px.timeline(
            df_all_view,
            x_start="StartDT", x_end="FinishDT", y="Doctor",
            color="Type",
            text="Label",  # ← labels on bars
            color_discrete_map=type_colors,
            hover_data=["patient_id","processing_time","start_time","end_time","arrival_time"]
        )
        # Style labels: centered inside bars
        fig_all.update_traces(textposition="inside", insidetextanchor="middle",
                              textfont=dict(size=10, family="Arial", color="black"))
        fig_all.update_layout(
            template="plotly_white",
            height=540,
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            xaxis_title="Time",
            yaxis_title="",
        )
        fig_all.update_xaxes(range=[SCHED_BASE_DT, SCHED_END_DT], tickformat="%H:%M")
        st.plotly_chart(fig_all, use_container_width=True)

    # Tabs per doctor: DF + mini timeline with labels
    st.subheader("Per-Doctor Views (08:00→11:50)")
    tabs = st.tabs([f"Doctor {i}" for i in range(8)])
    for i, tab in enumerate(tabs):
        with tab:
            d = doctor_dfs[i].copy()
            d_view = clip_view(d)
            if d_view.empty:
                st.info(f"No patients for Doctor {i} in this window.")
            else:
                # Requested DF view
                show_cols = [
                    "patient_id", "start_time", "end_time", "processing_time", "arrival_time",
                    "StartDT", "FinishDT", "ArrivalDT", "Type"
                ]
                st.dataframe(d_view[show_cols], use_container_width=True)

                d_view["Label"] = "PAT " + d_view["patient_id"].astype(str)
                fig_doc = px.timeline(
                    d_view,
                    x_start="StartDT", x_end="FinishDT", y="Doctor",
                    color="Type",
                    text="Label",  # labels
                    color_discrete_map=type_colors,
                    hover_data=["patient_id","processing_time","start_time","end_time","arrival_time"]
                )
                fig_doc.update_traces(textposition="inside", insidetextanchor="middle",
                                      textfont=dict(size=10, family="Arial", color="black"))
                fig_doc.update_layout(
                    template="plotly_white",
                    height=380,
                    legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
                    xaxis_title="Time", yaxis_title=""
                )
                fig_doc.update_xaxes(range=[SCHED_BASE_DT, SCHED_END_DT], tickformat="%H:%M")
                st.plotly_chart(fig_doc, use_container_width=True)
