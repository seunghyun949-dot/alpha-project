import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

CSV_PATH = "experiment_results.csv"

COLUMNS = [
    "test_id", "timestamp", "emitter_type", "emitter_diameter_mm",
    "collector_type", "gap_mm", "measured_voltage_kV",
    "measured_current_mA", "measured_velocity_ms", "r_in_m", "r_th_m",
    "r_out_m", "x_th_m", "u_inf_ms", "v_dis_V", "v_ind_V", "v_acc_V",
    "acc_offset_m", "acc_length_m", "sim_force_mN", "sim_du_throat_ms",
    "sim_u_throat_ms", "sim_u_outlet_ms", "velocity_error_ms",
    "velocity_error_percent", "electrical_power_W", "force_per_watt_mN_W",
    "measured_velocity_gain_ms"
]


def _df():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(CSV_PATH)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=COLUMNS)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[COLUMNS]


def _f(widget):
    try:
        return float(widget.value)
    except Exception:
        return np.nan


def _next_id(df):
    if len(df) == 0:
        return 1
    ids = pd.to_numeric(df["test_id"], errors="coerce").dropna()
    return 1 if len(ids) == 0 else int(ids.max()) + 1


def build_row():
    p = current_params()
    res = run_simulation(p)
    df = _df()

    measured_v = _f(measured_voltage)
    measured_i = _f(measured_current)
    measured_u = _f(measured_velocity)
    sim_u = float(res["u_out_mean"])

    if np.isfinite(measured_u):
        error_ms = measured_u - sim_u
        error_pct = abs(error_ms) / abs(measured_u) * 100 if abs(measured_u) > 1e-12 else np.nan
        gain_ms = measured_u - float(p["u_inf"])
    else:
        error_ms = np.nan
        error_pct = np.nan
        gain_ms = np.nan

    power_w = measured_v * measured_i if np.isfinite(measured_v) and np.isfinite(measured_i) else np.nan
    force_mn = float(res["force_x"]) * 1e3
    force_per_watt = force_mn / power_w if np.isfinite(power_w) and abs(power_w) > 1e-12 else np.nan

    return {
        "test_id": _next_id(df),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "emitter_type": emitter_type.value.strip(),
        "emitter_diameter_mm": _f(emitter_diameter),
        "collector_type": collector_type.value.strip(),
        "gap_mm": _f(gap),
        "measured_voltage_kV": measured_v,
        "measured_current_mA": measured_i,
        "measured_velocity_ms": measured_u,
        "r_in_m": float(p["r_in"]),
        "r_th_m": float(p["r_th"]),
        "r_out_m": float(p["r_ou"]),
        "x_th_m": float(p["x_th"]),
        "u_inf_ms": float(p["u_inf"]),
        "v_dis_V": float(p["v_dis"]),
        "v_ind_V": float(p["v_ind"]),
        "v_acc_V": float(p["v_acc"]),
        "acc_offset_m": float(p["acc_offset"]),
        "acc_length_m": float(p["acc_length"]),
        "sim_force_mN": force_mn,
        "sim_du_throat_ms": float(res["du_throat_mean"]),
        "sim_u_throat_ms": float(res["u_throat_mean"]),
        "sim_u_outlet_ms": sim_u,
        "velocity_error_ms": error_ms,
        "velocity_error_percent": error_pct,
        "electrical_power_W": power_w,
        "force_per_watt_mN_W": force_per_watt,
        "measured_velocity_gain_ms": gain_ms,
    }


def save_experiment(_=None):
    save_button.disabled = True
    status.value = "<b>Saving...</b> simulation is being recalculated with the current sliders."
    try:
        row = build_row()
        df = _df()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)

        err = row["velocity_error_percent"]
        err_text = "n/a" if not np.isfinite(err) else f"{err:.2f}%"
        status.value = (
            f"<b style='color:green'>Saved test #{row['test_id']}.</b> "
            f"Rows: {len(df)} | velocity error: {err_text}"
        )
    except Exception as exc:
        status.value = f"<b style='color:#b00020'>Save error:</b> {exc}"
    finally:
        save_button.disabled = False


def show_comparison(_=None):
    with graph_output:
        clear_output(wait=True)
        df = _df()
        if len(df) == 0:
            print("No experiment rows yet. Save at least one experiment first.")
            return

        numeric = [
            "test_id", "measured_voltage_kV", "measured_current_mA",
            "measured_velocity_ms", "sim_u_outlet_ms", "velocity_error_percent"
        ]
        for col in numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("Experiment vs Simulation Comparison", fontsize=14)

        ax = axes[0, 0]
        d = df[["measured_velocity_ms", "sim_u_outlet_ms"]].dropna()
        if len(d):
            ax.scatter(d["measured_velocity_ms"], d["sim_u_outlet_ms"])
            lo = float(np.nanmin(d.to_numpy()))
            hi = float(np.nanmax(d.to_numpy()))
            if abs(hi - lo) < 1e-12:
                lo -= 0.5
                hi += 0.5
            ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlabel("Measured outlet velocity [m/s]")
        ax.set_ylabel("Simulated outlet velocity [m/s]")
        ax.set_title("Measured vs simulated velocity")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        d = df[["measured_voltage_kV", "measured_velocity_ms"]].dropna()
        if len(d):
            ax.scatter(d["measured_voltage_kV"], d["measured_velocity_ms"])
        ax.set_xlabel("Measured voltage [kV]")
        ax.set_ylabel("Measured velocity [m/s]")
        ax.set_title("Voltage vs measured velocity")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        d = df[["measured_current_mA", "measured_velocity_ms"]].dropna()
        if len(d):
            ax.scatter(d["measured_current_mA"], d["measured_velocity_ms"])
        ax.set_xlabel("Measured current [mA]")
        ax.set_ylabel("Measured velocity [m/s]")
        ax.set_title("Current vs measured velocity")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        d = df[["test_id", "measured_velocity_ms", "sim_u_outlet_ms"]].dropna(subset=["test_id"])
        if len(d):
            ax.plot(d["test_id"], d["measured_velocity_ms"], marker="o", label="Measured")
            ax.plot(d["test_id"], d["sim_u_outlet_ms"], marker="o", label="Simulation")
            ax.legend()
        ax.set_xlabel("Test ID")
        ax.set_ylabel("Velocity [m/s]")
        ax.set_title("Test-by-test velocity comparison")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        err = pd.to_numeric(df["velocity_error_percent"], errors="coerce").dropna()
        if len(err):
            print(f"Mean absolute velocity error: {err.mean():.2f}%")
            print(f"Median absolute velocity error: {err.median():.2f}%")


def download_csv(_=None):
    if not os.path.exists(CSV_PATH):
        status.value = "<b style='color:#b00020'>No CSV exists yet.</b>"
        return
    try:
        from google.colab import files
        files.download(CSV_PATH)
        status.value = f"<b>Download requested:</b> {CSV_PATH}"
    except Exception:
        status.value = f"<b>CSV saved locally:</b> {CSV_PATH}"


def upload_csv(_=None):
    try:
        from google.colab import files
        uploaded = files.upload()
        if not uploaded:
            return
        name = next(iter(uploaded))
        with open(CSV_PATH, "wb") as f:
            f.write(uploaded[name])
        df = _df()
        status.value = f"<b style='color:green'>Loaded CSV.</b> Rows: {len(df)}"
    except Exception as exc:
        status.value = f"<b style='color:#b00020'>Upload error:</b> {exc}"


emitter_type = widgets.Text(value="wire", description="Emitter type", style={"description_width": "150px"})
emitter_diameter = widgets.FloatText(value=0.10, description="Emitter dia [mm]", style={"description_width": "150px"})
collector_type = widgets.Text(value="ring", description="Collector type", style={"description_width": "150px"})
gap = widgets.FloatText(value=10.0, description="Electrode gap [mm]", style={"description_width": "150px"})
measured_voltage = widgets.FloatText(value=15.0, description="Measured V [kV]", style={"description_width": "150px"})
measured_current = widgets.FloatText(value=0.20, description="Measured I [mA]", style={"description_width": "150px"})
measured_velocity = widgets.FloatText(value=np.nan, description="Measured U [m/s]", style={"description_width": "150px"})

for w in [emitter_type, emitter_diameter, collector_type, gap, measured_voltage, measured_current, measured_velocity]:
    w.layout = widgets.Layout(width="340px")

save_button = widgets.Button(description="SAVE EXPERIMENT", button_style="success", icon="save", layout=widgets.Layout(width="180px", height="42px"))
compare_button = widgets.Button(description="SHOW COMPARISON", button_style="info", icon="bar-chart", layout=widgets.Layout(width="190px", height="42px"))
download_button = widgets.Button(description="DOWNLOAD CSV", icon="download", layout=widgets.Layout(width="160px", height="42px"))
upload_button = widgets.Button(description="UPLOAD CSV", icon="upload", layout=widgets.Layout(width="150px", height="42px"))

save_button.on_click(save_experiment)
compare_button.on_click(show_comparison)
download_button.on_click(download_csv)
upload_button.on_click(upload_csv)

status = widgets.HTML(value="<b>Ready.</b> Enter measured values, then save the experiment.")
graph_output = widgets.Output()

panel = widgets.VBox([
    widgets.HTML("<h3>Experiment Data Logger</h3>"),
    widgets.HTML("Current simulator sliders are saved together with the measured values below."),
    widgets.HBox([
        widgets.VBox([emitter_type, emitter_diameter, collector_type, gap]),
        widgets.VBox([measured_voltage, measured_current, measured_velocity]),
    ]),
    widgets.HBox([save_button, compare_button, download_button, upload_button]),
    status,
])

display(panel, graph_output)
