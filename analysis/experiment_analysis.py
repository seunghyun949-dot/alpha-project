import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

CSV_PATH = "experiment_results.csv"


def load_data():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame()
    try:
        df = pd.read_csv(CSV_PATH)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    numeric_cols = [
        "test_id", "emitter_diameter_mm", "gap_mm", "measured_voltage_kV",
        "measured_current_mA", "measured_velocity_ms", "sim_force_mN",
        "sim_u_outlet_ms", "velocity_error_percent", "electrical_power_W",
        "force_per_watt_mN_W", "measured_velocity_gain_ms"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def gap_summary(df):
    if "gap_mm" not in df.columns or "measured_velocity_ms" not in df.columns:
        return pd.DataFrame()

    d = df.dropna(subset=["gap_mm", "measured_velocity_ms"]).copy()
    if d.empty:
        return pd.DataFrame()

    # 1 mm 단위로 가장 가까운 정수 gap에 묶음
    d["gap_1mm"] = d["gap_mm"].round().astype(int)

    out = (
        d.groupby("gap_1mm", as_index=False)
        .agg(
            n=("measured_velocity_ms", "count"),
            mean_velocity_ms=("measured_velocity_ms", "mean"),
            std_velocity_ms=("measured_velocity_ms", "std"),
            mean_current_mA=("measured_current_mA", "mean"),
            mean_voltage_kV=("measured_voltage_kV", "mean"),
            mean_error_percent=("velocity_error_percent", "mean"),
        )
        .sort_values("gap_1mm")
    )
    return out


def shape_summary(df):
    required = {"emitter_type", "collector_type", "measured_velocity_ms"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    d = df.dropna(subset=["measured_velocity_ms"]).copy()
    if d.empty:
        return pd.DataFrame()

    d["emitter_type"] = d["emitter_type"].fillna("unknown").astype(str)
    d["collector_type"] = d["collector_type"].fillna("unknown").astype(str)
    d["shape"] = d["emitter_type"] + " / " + d["collector_type"]

    out = (
        d.groupby("shape", as_index=False)
        .agg(
            n=("measured_velocity_ms", "count"),
            mean_velocity_ms=("measured_velocity_ms", "mean"),
            std_velocity_ms=("measured_velocity_ms", "std"),
            mean_gain_ms=("measured_velocity_gain_ms", "mean"),
            mean_power_W=("electrical_power_W", "mean"),
            mean_error_percent=("velocity_error_percent", "mean"),
        )
        .sort_values("mean_velocity_ms", ascending=False)
    )
    return out


def error_summary(df):
    if "velocity_error_percent" not in df.columns:
        return {}
    e = pd.to_numeric(df["velocity_error_percent"], errors="coerce").dropna()
    if e.empty:
        return {}
    return {
        "n": int(e.count()),
        "mean": float(e.mean()),
        "median": float(e.median()),
        "std": float(e.std(ddof=1)) if len(e) > 1 else np.nan,
        "max": float(e.max()),
    }


def recommend_best(df, objective):
    if df.empty:
        return None, "No experiment data."

    objectives = {
        "Max measured velocity": ("measured_velocity_ms", False),
        "Max velocity gain": ("measured_velocity_gain_ms", False),
        "Max force per watt": ("force_per_watt_mN_W", False),
        "Min simulation error": ("velocity_error_percent", True),
        "Min electrical power": ("electrical_power_W", True),
    }

    col, ascending = objectives[objective]
    if col not in df.columns:
        return None, f"Column unavailable: {col}"

    d = df.dropna(subset=[col]).copy()
    if d.empty:
        return None, f"No valid values for {col}."

    # 동일 목표값이면: 낮은 전력 → 낮은 오차 → 빠른 실측 풍속 순으로 보조 정렬
    sort_cols = [col]
    asc = [ascending]
    if col != "electrical_power_W" and "electrical_power_W" in d.columns:
        sort_cols.append("electrical_power_W")
        asc.append(True)
    if col != "velocity_error_percent" and "velocity_error_percent" in d.columns:
        sort_cols.append("velocity_error_percent")
        asc.append(True)
    if col != "measured_velocity_ms" and "measured_velocity_ms" in d.columns:
        sort_cols.append("measured_velocity_ms")
        asc.append(False)

    best = d.sort_values(sort_cols, ascending=asc, na_position="last").iloc[0]
    return best, None


def _fmt(v, digits=4):
    try:
        if pd.isna(v):
            return "n/a"
        return f"{float(v):.{digits}g}"
    except Exception:
        return str(v)


def show_analysis(_=None):
    with analysis_output:
        clear_output(wait=True)
        df = load_data()
        if df.empty:
            print("No experiment data. Save or upload experiment_results.csv first.")
            return

        gs = gap_summary(df)
        ss = shape_summary(df)
        es = error_summary(df)
        best, best_err = recommend_best(df, objective.value)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("EHD Experiment Analysis", fontsize=15)

        ax = axes[0, 0]
        if not gs.empty:
            yerr = gs["std_velocity_ms"].fillna(0.0)
            ax.errorbar(
                gs["gap_1mm"], gs["mean_velocity_ms"],
                yerr=yerr, marker="o", capsize=4
            )
        ax.set_xlabel("Electrode gap [mm] — 1 mm bins")
        ax.set_ylabel("Measured velocity [m/s]")
        ax.set_title("Gap vs mean measured velocity")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        if not ss.empty:
            x = np.arange(len(ss))
            err = ss["std_velocity_ms"].fillna(0.0)
            ax.bar(x, ss["mean_velocity_ms"], yerr=err, capsize=4)
            ax.set_xticks(x)
            ax.set_xticklabels(ss["shape"], rotation=35, ha="right")
        ax.set_ylabel("Measured velocity [m/s]")
        ax.set_title("Emitter / collector shape mean ± SD")
        ax.grid(True, axis="y", alpha=0.3)

        ax = axes[1, 0]
        if {"test_id", "velocity_error_percent"}.issubset(df.columns):
            d = df.dropna(subset=["test_id", "velocity_error_percent"])
            if len(d):
                ax.plot(d["test_id"], d["velocity_error_percent"], marker="o")
        ax.set_xlabel("Test ID")
        ax.set_ylabel("Absolute velocity error [%]")
        ax.set_title("Simulation error by test")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        if {"measured_velocity_ms", "electrical_power_W"}.issubset(df.columns):
            d = df.dropna(subset=["measured_velocity_ms", "electrical_power_W"])
            if len(d):
                ax.scatter(d["electrical_power_W"], d["measured_velocity_ms"])
        ax.set_xlabel("Measured electrical power [W]")
        ax.set_ylabel("Measured velocity [m/s]")
        ax.set_title("Power vs measured velocity")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print("\n=== DATA SUMMARY ===")
        print(f"Rows: {len(df)}")
        if es:
            print(
                f"Simulation velocity error: mean={es['mean']:.2f}% | "
                f"median={es['median']:.2f}% | max={es['max']:.2f}%"
            )

        print("\n=== 1 mm GAP SUMMARY ===")
        if gs.empty:
            print("No usable gap/velocity data.")
        else:
            print(gs.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

        print("\n=== SHAPE SUMMARY ===")
        if ss.empty:
            print("No usable shape data.")
        else:
            print(ss.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

        print("\n=== BEST OBSERVED CONDITION ===")
        print(f"Objective: {objective.value}")
        if best_err:
            print(best_err)
        else:
            fields = [
                "test_id", "emitter_type", "emitter_diameter_mm", "collector_type",
                "gap_mm", "measured_voltage_kV", "measured_current_mA",
                "measured_velocity_ms", "measured_velocity_gain_ms",
                "electrical_power_W", "force_per_watt_mN_W",
                "velocity_error_percent"
            ]
            for field in fields:
                if field in best.index:
                    print(f"{field}: {_fmt(best[field])}")
            print(
                "\nRecommendation is only the best OBSERVED row in the current CSV; "
                "it is not an extrapolated physical optimum."
            )


def export_summary(_=None):
    df = load_data()
    if df.empty:
        analysis_status.value = "<b style='color:#b00020'>No data to export.</b>"
        return

    gs = gap_summary(df)
    ss = shape_summary(df)
    gs.to_csv("gap_summary_1mm.csv", index=False)
    ss.to_csv("shape_summary.csv", index=False)

    analysis_status.value = (
        "<b style='color:green'>Summary CSVs created:</b> "
        "gap_summary_1mm.csv, shape_summary.csv"
    )

    try:
        from google.colab import files
        files.download("gap_summary_1mm.csv")
        files.download("shape_summary.csv")
    except Exception:
        pass


objective = widgets.Dropdown(
    options=[
        "Max measured velocity",
        "Max velocity gain",
        "Max force per watt",
        "Min simulation error",
        "Min electrical power",
    ],
    value="Max measured velocity",
    description="Optimize for",
    style={"description_width": "120px"},
    layout=widgets.Layout(width="360px")
)

analyze_button = widgets.Button(
    description="ANALYZE DATA",
    button_style="primary",
    icon="line-chart",
    layout=widgets.Layout(width="170px", height="42px")
)
export_button = widgets.Button(
    description="EXPORT SUMMARY",
    icon="download",
    layout=widgets.Layout(width="180px", height="42px")
)
analysis_status = widgets.HTML(
    value="<b>Ready.</b> Choose an objective and analyze the current CSV."
)
analysis_output = widgets.Output()

analyze_button.on_click(show_analysis)
export_button.on_click(export_summary)

analysis_panel = widgets.VBox([
    widgets.HTML("<h3>Experiment Statistics & Best Observed Condition</h3>"),
    widgets.HTML(
        "Gap results are grouped into 1 mm bins. Shape bars show mean ± standard deviation. "
        "The recommendation selects only among measured rows in the CSV."
    ),
    widgets.HBox([objective, analyze_button, export_button]),
    analysis_status,
])

display(analysis_panel, analysis_output)
