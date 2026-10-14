# ============================================================
# EHD VENTURI AXISYMMETRIC SIMULATOR — GOOGLE COLAB
# ============================================================
# 목적:
#   - 축대칭 벤츄리 형상
#   - 3종 전극(방전/유도/가속)
#   - 전위장/전기장 계산
#   - 경험적 공간전하 분포
#   - Coulomb 체적력 f = rho_e * E
#   - 벤츄리 배경 유속 + EHD 유속 증가량 근사
#   - Colab용 ipywidgets UI
#
# 주의:
#   이 코드는 정량 검증된 코로나 방전/플라즈마 CFD 모델이 아닙니다.
#   형상/전극 위치/전압 변화에 따른 상대 비교용입니다.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

from IPython.display import display, clear_output
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import distance_transform_edt

# ============================================================
# 0. CONSTANTS / GRID
# ============================================================

NX = 141
NR = 81

X_MIN = -0.22
X_MAX =  0.72
R_MAX =  0.18

TUBE_X0 = 0.00
TUBE_X1 = 0.50
WALL_T  = 0.006

AIR_RHO = 1.225
RHO_E_MAX = 2.0e-6
CHARGE_DECAY = 0.020
E_REF = 1.0e5
EHD_RESPONSE_TIME = 0.015

xv = np.linspace(X_MIN, X_MAX, NX)
rv = np.linspace(0.0, R_MAX, NR)

dx = xv[1] - xv[0]
dr = rv[1] - rv[0]

X, R = np.meshgrid(xv, rv)
N = NX * NR

print(f"[Grid] {NX} x {NR} = {N} cells | dx={dx:.5f} m | dr={dr:.5f} m")

# ============================================================
# 1. VENTURI GEOMETRY
# ============================================================

def venturi_r(xi, r_in, r_th, r_ou, x_th):
    xi = np.asarray(xi, dtype=float)
    rad = np.full_like(xi, np.nan, dtype=float)

    m1 = (xi >= TUBE_X0) & (xi <= x_th)
    m2 = (xi > x_th) & (xi <= TUBE_X1)

    if np.any(m1):
        den1 = max(x_th - TUBE_X0, 1e-12)
        t1 = (xi[m1] - TUBE_X0) / den1
        rad[m1] = (
            r_th
            + (r_in - r_th) * 0.5 * (1.0 + np.cos(np.pi * t1))
        )

    if np.any(m2):
        den2 = max(TUBE_X1 - x_th, 1e-12)
        t2 = (xi[m2] - x_th) / den2
        rad[m2] = (
            r_th
            + (r_ou - r_th) * 0.5 * (1.0 - np.cos(np.pi * t2))
        )

    return rad

# ============================================================
# 2. WALL / ELECTRODE MASKS
# ============================================================

def make_masks(r_in, r_th, r_ou, x_th, acc_offset=0.010, acc_length=0.080):
    rad = venturi_r(X, r_in, r_th, r_ou, x_th)

    in_tube = (
        (X >= TUBE_X0)
        & (X <= TUBE_X1)
        & np.isfinite(rad)
    )

    solid = (
        in_tube
        & (R > rad)
        & (R <= rad + WALL_T)
    )

    emitter_x0 = TUBE_X0 - 0.060
    emitter_x1 = TUBE_X0 - 0.025
    emitter_r  = 0.004

    discharge_body = (
        (X >= emitter_x0)
        & (X <= emitter_x1)
        & (R <= emitter_r)
        & ~solid
    )

    tip_end = TUBE_X0 - 0.004
    tip_len = max(tip_end - emitter_x1, 1e-12)

    taper_radius = (
        emitter_r
        * np.clip(
            1.0 - (X - emitter_x1) / tip_len,
            0.0,
            1.0
        )
    )

    discharge_tip = (
        (X > emitter_x1)
        & (X <= tip_end)
        & (R <= taper_radius)
        & ~solid
    )

    discharge = discharge_body | discharge_tip

    induced = (
        in_tube
        & (X >= TUBE_X0)
        & (X <= TUBE_X0 + 0.060)
        & (R >= rad - 2.0 * dr)
        & (R < rad)
        & ~solid
    )

    acc_x0 = np.clip(x_th + acc_offset, TUBE_X0, TUBE_X1)
    acc_x1 = np.clip(acc_x0 + acc_length, TUBE_X0, TUBE_X1)

    accel = (
        in_tube
        & (X >= acc_x0)
        & (X <= acc_x1)
        & (R >= rad - 2.0 * dr)
        & (R < rad)
        & ~solid
        & ~induced
    )

    return solid, discharge, induced, accel, rad

# ============================================================
# 3. AXISYMMETRIC LAPLACE MATRIX
# ============================================================

def build_axisym_laplace(dirichlet_mask, dirichlet_values):
    mask = np.asarray(dirichlet_mask, dtype=bool).ravel()
    vals = np.asarray(dirichlet_values, dtype=float).ravel()

    A = lil_matrix((N, N), dtype=np.float64)
    b = np.zeros(N, dtype=float)

    dx2 = dx * dx
    dr2 = dr * dr

    def g(j, i):
        return j * NX + i

    for j in range(NR):
        rj = rv[j]

        for i in range(NX):
            k = g(j, i)

            if mask[k]:
                A[k, k] = 1.0
                b[k] = vals[k]
                continue

            if i == 0:
                A[k, k] = 1.0
                A[k, g(j, 1)] = -1.0
                continue

            if i == NX - 1:
                A[k, k] = 1.0
                A[k, g(j, NX - 2)] = -1.0
                continue

            if j == NR - 1:
                A[k, k] = 1.0
                A[k, g(j - 1, i)] = -1.0
                continue

            if j == 0:
                A[k, k] = -4.0 / dr2 - 2.0 / dx2
                A[k, g(1, i)] = 4.0 / dr2
                A[k, g(j, i + 1)] = 1.0 / dx2
                A[k, g(j, i - 1)] = 1.0 / dx2
                continue

            A[k, k] = -2.0 / dr2 - 2.0 / dx2
            A[k, g(j + 1, i)] = 1.0 / dr2 + 1.0 / (2.0 * rj * dr)
            A[k, g(j - 1, i)] = 1.0 / dr2 - 1.0 / (2.0 * rj * dr)
            A[k, g(j, i + 1)] = 1.0 / dx2
            A[k, g(j, i - 1)] = 1.0 / dx2

    return A.tocsr(), b

# ============================================================
# 4. SIMULATION
# ============================================================

def run_simulation(p):
    r_in = float(p["r_in"])
    r_th = float(p["r_th"])
    r_ou = float(p["r_ou"])
    x_th = float(p["x_th"])

    u_inf = float(p["u_inf"])

    v_dis = float(p["v_dis"])
    v_ind = float(p["v_ind"])
    v_acc = float(p["v_acc"])

    acc_offset = float(p["acc_offset"])
    acc_length = float(p["acc_length"])

    if not (0.0 < r_th < min(r_in, r_ou)):
        raise ValueError("Throat radius must be smaller than inlet/outlet radii.")

    if not (TUBE_X0 + 0.05 < x_th < TUBE_X1 - 0.05):
        raise ValueError("Throat position is too close to a tube end.")

    solid, discharge, induced, accel, rad = make_masks(
        r_in, r_th, r_ou, x_th, acc_offset, acc_length
    )

    if not discharge.any():
        raise ValueError("Discharge electrode mask is empty.")
    if not induced.any():
        raise ValueError("Induced electrode mask is empty.")
    if not accel.any():
        raise ValueError("Accelerator electrode mask is empty.")

    electrode = discharge | induced | accel

    bc_values = np.zeros_like(X, dtype=float)
    bc_values[discharge] = v_dis
    bc_values[induced] = v_ind
    bc_values[accel] = v_acc

    A_phi, b_phi = build_axisym_laplace(electrode, bc_values)
    phi = spsolve(A_phi, b_phi).reshape(NR, NX)

    if not np.isfinite(phi).all():
        raise RuntimeError("Electrostatic solve produced NaN/Inf.")

    dphi_dr, dphi_dx = np.gradient(phi, dr, dx)

    Ex = -dphi_dx
    Er = -dphi_dr
    E_mag = np.hypot(Ex, Er)

    dist_d = distance_transform_edt(~discharge, sampling=(dr, dx))
    dist_i = distance_transform_edt(~induced, sampling=(dr, dx))
    dist_a = distance_transform_edt(~accel, sampling=(dr, dx))

    field_activation = 1.0 - np.exp(-E_mag / E_REF)

    rho_e = (
        RHO_E_MAX
        * (
            np.exp(-dist_d / CHARGE_DECAY)
            + 0.35 * np.exp(-dist_i / CHARGE_DECAY)
            + 0.35 * np.exp(-dist_a / CHARGE_DECAY)
        )
        * field_activation
    )

    rho_e[solid] = 0.0
    rho_e = np.clip(rho_e, 0.0, RHO_E_MAX)

    fx = rho_e * Ex
    fr = rho_e * Er

    fx[solid] = 0.0
    fr[solid] = 0.0

    fluid_inside = (
        (X >= TUBE_X0)
        & (X <= TUBE_X1)
        & np.isfinite(rad)
        & (R <= rad)
    )

    A_in = np.pi * r_in**2

    local_area = np.pi * np.where(
        np.isfinite(rad),
        np.maximum(rad, 1e-9)**2,
        r_in**2
    )

    u_base = np.full_like(X, u_inf, dtype=float)

    u_base[fluid_inside] = (
        u_inf
        * A_in
        / local_area[fluid_inside]
    )

    u_base[solid] = np.nan

    du = (fx / AIR_RHO) * EHD_RESPONSE_TIME

    du[~fluid_inside] = 0.0
    du[solid] = np.nan

    u_total = u_base + du

    dV = 2.0 * np.pi * R * dr * dx

    force_x = np.nansum(
        np.where(fluid_inside, fx, 0.0) * dV
    )

    force_r = np.nansum(
        np.where(fluid_inside, fr, 0.0) * dV
    )

    i_th = int(np.argmin(np.abs(xv - x_th)))

    throat_mask = R[:, i_th] <= r_th

    du_throat_mean = np.nanmean(
        du[:, i_th][throat_mask]
    )

    u_throat_mean = np.nanmean(
        u_total[:, i_th][throat_mask]
    )

    x_out_eval = TUBE_X1 - 2.0 * dx

    i_out = int(np.argmin(np.abs(xv - x_out_eval)))

    r_out_eval = venturi_r(
        np.array([xv[i_out]]),
        r_in,
        r_th,
        r_ou,
        x_th
    )[0]

    outlet_mask = R[:, i_out] <= r_out_eval

    u_out_mean = np.nanmean(
        u_total[:, i_out][outlet_mask]
    )

    return {
        "phi": phi,
        "Ex": Ex,
        "Er": Er,
        "E_mag": E_mag,
        "rho_e": rho_e,
        "fx": fx,
        "fr": fr,
        "solid": solid,
        "discharge": discharge,
        "induced": induced,
        "accel": accel,
        "rad": rad,
        "fluid_inside": fluid_inside,
        "u_base": u_base,
        "du": du,
        "u_total": u_total,
        "force_x": force_x,
        "force_r": force_r,
        "du_throat_mean": du_throat_mean,
        "u_throat_mean": u_throat_mean,
        "u_out_mean": u_out_mean,
        "max_E": float(np.nanmax(E_mag)),
        "max_rho_e": float(np.nanmax(rho_e)),
    }

# ============================================================
# 5. VISUALIZATION HELPERS
# ============================================================

def mirror_scalar(a):
    return np.vstack([
        a[:0:-1, :],
        a
    ])

r_full = np.concatenate([
    -rv[:0:-1],
    rv
])

extent_full = [
    X_MIN,
    X_MAX,
    -R_MAX,
    R_MAX
]

# ============================================================
# 6. RENDER
# ============================================================

def render_result(p, res):
    phi_m = mirror_scalar(res["phi"])
    E_m = mirror_scalar(res["E_mag"])
    rho_m = mirror_scalar(res["rho_e"])
    du_m = mirror_scalar(res["du"])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    plt.subplots_adjust(
        left=0.07,
        right=0.95,
        bottom=0.08,
        top=0.82,
        hspace=0.34,
        wspace=0.28
    )

    x_geo = np.linspace(TUBE_X0, TUBE_X1, 500)

    r_geo = venturi_r(
        x_geo,
        p["r_in"],
        p["r_th"],
        p["r_ou"],
        p["x_th"]
    )

    def draw_geometry(ax):
        ax.plot(x_geo, r_geo, linewidth=2)
        ax.plot(x_geo, -r_geo, linewidth=2)
        ax.plot(x_geo, r_geo + WALL_T, linewidth=1)
        ax.plot(x_geo, -(r_geo + WALL_T), linewidth=1)

        dmask = mirror_scalar(res["discharge"].astype(float))
        imask = mirror_scalar(res["induced"].astype(float))
        amask = mirror_scalar(res["accel"].astype(float))

        ax.contour(xv, r_full, dmask, levels=[0.5], linewidths=2)
        ax.contour(xv, r_full, imask, levels=[0.5], linewidths=2)
        ax.contour(xv, r_full, amask, levels=[0.5], linewidths=2)

        ax.axvline(p["x_th"], linestyle="--", linewidth=0.8)

        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(-R_MAX, R_MAX)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("r [m]")

    ax = axes[0, 0]
    im = ax.imshow(
        rho_m,
        origin="lower",
        extent=extent_full,
        aspect="auto"
    )
    draw_geometry(ax)
    ax.set_title("(A) Geometry + empirical charge density")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("rho_e [C/m^3]")

    ax = axes[0, 1]
    im = ax.imshow(
        phi_m / 1000.0,
        origin="lower",
        extent=extent_full,
        aspect="auto"
    )
    draw_geometry(ax)
    ax.set_title("(B) Electric potential")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Potential [kV]")

    ax = axes[1, 0]

    E_plot = np.log10(
        np.maximum(E_m, 1.0)
    )

    im = ax.imshow(
        E_plot,
        origin="lower",
        extent=extent_full,
        aspect="auto"
    )
    draw_geometry(ax)
    ax.set_title("(C) Electric field magnitude")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("log10(|E| [V/m])")

    ax = axes[1, 1]

    finite_du = du_m[np.isfinite(du_m)]

    if finite_du.size > 0:
        vmax = max(
            np.nanpercentile(np.abs(finite_du), 99),
            1e-12
        )
    else:
        vmax = 1.0

    im = ax.imshow(
        du_m,
        origin="lower",
        extent=extent_full,
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
        cmap="coolwarm"
    )

    draw_geometry(ax)
    ax.set_title("(D) EHD-only velocity increment du")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("du [m/s]")

    fig.suptitle(
        (
            "Axisymmetric EHD Venturi Simulator\n"
            f"Vdis={p['v_dis']/1000:.1f} kV   |   "
            f"Vind={p['v_ind']/1000:.1f} kV   |   "
            f"Vacc={p['v_acc']/1000:.1f} kV   |   "
            f"Uin={p['u_inf']:.2f} m/s"
        ),
        fontsize=14,
        y=0.98
    )

    summary = (
        f"Model body-force integral Fx = {res['force_x']:.3e} N "
        f"({res['force_x']*1e3:.3f} mN)    |    "
        f"mean du@throat = {res['du_throat_mean']:.4g} m/s    |    "
        f"mean U@throat = {res['u_throat_mean']:.4g} m/s    |    "
        f"mean U@outlet = {res['u_out_mean']:.4g} m/s\n"
        "NOTE: empirical charge + simplified flow response; "
        "not a validated thrust prediction."
    )

    fig.text(
        0.5,
        0.905,
        summary,
        ha="center",
        va="top",
        fontsize=9
    )

    plt.show()

# ============================================================
# 7. DEFAULT PARAMETERS
# ============================================================

defaults = {
    "r_in": 0.090,
    "r_th": 0.045,
    "r_ou": 0.080,
    "x_th": 0.250,
    "u_inf": 1.0,
    "v_dis": 15000.0,
    "v_ind": 0.0,
    "v_acc": 5000.0,
    "acc_offset": 0.010,
    "acc_length": 0.080
}

# ============================================================
# 8. GOOGLE COLAB IPYWIDGETS UI
# ============================================================

slider_specs = [
    ("r_in", "Inlet radius [m]", 0.055, 0.130, 0.001, ".3f"),
    ("r_th", "Throat radius [m]", 0.020, 0.090, 0.001, ".3f"),
    ("r_ou", "Outlet radius [m]", 0.055, 0.130, 0.001, ".3f"),
    ("x_th", "Throat position [m]", 0.150, 0.350, 0.005, ".3f"),
    ("u_inf", "Inlet velocity [m/s]", 0.0, 5.0, 0.1, ".1f"),
    ("v_dis", "Discharge [V]", 0.0, 30000.0, 500.0, ".0f"),
    ("v_ind", "Induced [V]", -15000.0, 15000.0, 500.0, ".0f"),
    ("v_acc", "Accelerator [V]", -15000.0, 15000.0, 500.0, ".0f"),
    ("acc_offset", "Accel offset [m]", -0.050, 0.100, 0.005, ".3f"),
    ("acc_length", "Accel length [m]", 0.030, 0.200, 0.005, ".3f"),
]

controls = {}

for key, label, low, high, step, fmt in slider_specs:
    controls[key] = widgets.FloatSlider(
        value=defaults[key],
        min=low,
        max=high,
        step=step,
        description=label,
        continuous_update=False,
        readout_format=fmt,
        style={"description_width": "180px"},
        layout=widgets.Layout(width="700px")
    )

recalc_button = widgets.Button(
    description="RECALCULATE",
    button_style="primary",
    icon="refresh",
    layout=widgets.Layout(width="180px", height="42px")
)

reset_button = widgets.Button(
    description="RESET",
    button_style="",
    icon="undo",
    layout=widgets.Layout(width="120px", height="42px")
)

status_html = widgets.HTML(
    value="<b>Ready.</b> Change sliders, then press RECALCULATE."
)

plot_output = widgets.Output()

# ============================================================
# 9. UI FUNCTIONS
# ============================================================

def current_params():
    return {
        key: float(w.value)
        for key, w in controls.items()
    }

def calculate_and_draw(_=None):
    p = current_params()

    recalc_button.disabled = True
    status_html.value = "<b>Calculating...</b>"

    try:
        res = run_simulation(p)

        with plot_output:
            clear_output(wait=True)
            render_result(p, res)

        status_html.value = (
            "<b style='color:green'>Done.</b> "
            f"Fx(model) = {res['force_x']*1e3:.4f} mN, "
            f"mean du@throat = {res['du_throat_mean']:.4g} m/s"
        )

    except Exception as exc:
        with plot_output:
            clear_output(wait=True)
            print("Simulation error:", repr(exc))

        status_html.value = (
            "<b style='color:#b00020'>Error:</b> "
            + str(exc)
        )

    finally:
        recalc_button.disabled = False

def reset_controls(_=None):
    for key, value in defaults.items():
        controls[key].value = value

    status_html.value = (
        "<b>Reset complete.</b> Press RECALCULATE."
    )

recalc_button.on_click(calculate_and_draw)
reset_button.on_click(reset_controls)

# ============================================================
# 10. UI LAYOUT
# ============================================================

left_box = widgets.VBox([
    controls["r_in"],
    controls["r_th"],
    controls["r_ou"],
    controls["x_th"],
    controls["u_inf"],
])

right_box = widgets.VBox([
    controls["v_dis"],
    controls["v_ind"],
    controls["v_acc"],
    controls["acc_offset"],
    controls["acc_length"],
])

ui = widgets.VBox([
    widgets.HTML(
        "<h3>EHD Venturi Axisymmetric Simulator — Colab UI</h3>"
    ),
    widgets.HBox([
        left_box,
        right_box
    ]),
    widgets.HBox([
        recalc_button,
        reset_button
    ]),
    status_html
])

display(
    ui,
    plot_output
)

# ============================================================
# 11. INITIAL RUN
# ============================================================

calculate_and_draw()
