import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path.cwd()))

    # Initialization code that runs before all other cells
    import marimo as mo

    # Import dependencies
    from core import _defaults
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np
    import pandas as pd
    from core import atmos
    from core import aircraft as ac
    from core import plot_utils
    # from core.plot_utils import OptimumGridView

    # Set local/online filepath
    _defaults.FILEURL = _defaults.get_url()

    # Plotly dark mode template
    _defaults.set_plotly_template()

    # Data directory
    data_dir = str(mo.notebook_location().parent.parent / "data" / "AircraftDB_Standard.csv")

    # Source tables report altitude as flight level, everything else is SI
    FT_TO_M = 0.3048


@app.cell
def _():
    # Set navbar on the right
    _defaults.set_sidebar()
    return


@app.cell
def _():
    mo.md(r"""
    As shown in [[AircraftCustom.py]], for custom aircraft, we no longer rely on any models, such as jet or propeller, but we can directly use the aircraft's parameters to compute the minimum speed. The data for custom aircraft is stored in a CSV file, which contains the aircraft's parameters such as weight, wing area, maximum lift coefficient, and the performance data, dependent on the Mach number and other variables. Given that this data is tabular, a closed form for minimum speed cannot be derived, and the solution must be obtained numerically.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    $$
    \begin{aligned}
        \min_{C_L, \delta_T}
        & \quad V \\
        \text{subject to}
        & \quad c_1^\mathrm{eq} = L-W = \frac{1}{2}\rho V^2 S C_L - W = 0 \\
        & \quad c_2^\mathrm{eq} = T-D = \delta_T T_a - \frac{1}{2} \rho V^2 S (C_{D_0}+K C_L^2) =0 \\
        \text{for }
        & \quad C_L \in [0, C_{L_\mathrm{max}}] \\
        & \quad \delta_T \in [0, 1] \\
        \text{with }
        & \quad T_a = T_a(M, h, s), \quad s \in \{\mathrm{Max}, \mathrm{Mil}\} \\
        & \quad C_{D_0} = C_{D_0}(M) \\
        & \quad K = K(M, C_L) \\
    \end{aligned}
    $$
    """)
    return


@app.cell
def _():
    mo.md(r"""
    As with the jet and propeller aircraft we started with, the problem can be simplified by eliminating the speed $V$ from the constraints, since it can be expressed as a function of the lift coefficient $C_L$ and the weight $W$. The optimization problem can then be rewritten as follows:


    $$
    \begin{aligned}
        \min_{C_L, \delta_T}
        & \quad V(C_L, W) = \sqrt{\frac{2W}{\rho S C_L}} \\
        \text{subject to}
        & \quad c_2^\mathrm{eq} = T-D = \delta_T T_a - \frac{1}{2} \rho V^2 S (C_{D_0}+K C_L^2) =0 \\
        \text{for }
        & \quad C_L \in [0, C_{L_\mathrm{max}}] \\
        & \quad \delta_T \in [0, 1] \\
        \text{with }
        & \quad T_a = T_a(M, h, s), \quad s \in \{\mathrm{Max}, \mathrm{Mil}\} \\
        & \quad C_{D_0} = C_{D_0}(M) \\
        & \quad K = K(M, C_L) \\
    \end{aligned}
    $$
    """)
    return


@app.cell
def _():
    mo.md(r"""
    A custom aircraft differs from standard jet and propeller models by accounting for compressibility effects, which are significant at high Mach numbers. Given that most of these aircraft operate at high speeds, the drag polar is no longer a simple parabolic function of the lift coefficient, but is dependent on the Mach number, clearly noticeable in the sudden increase in parasitic drag close to Mach.

    In this case, the parasitic drag coefficient $C_{D_0}$ and the induced drag factor $K$ are functions of the Mach number, which is a function of the speed $V$ and the altitude $h$. The thrust available $T_a$ is also a function of the Mach number, altitude, and thrust setting $s$.

    The optimization problem is solved numerically using the `scipy.optimize.minimize` function, which takes as input the objective function, the initial guess for the optimization variables, and the constraints. The optimization variables are the lift coefficient $C_L$ and the thrust setting $\delta_T$, which are bounded by their respective limits. The constraints are the lift and thrust equations, which are set to zero. The optimization is performed for a given weight $W$, altitude $h$, and thrust setting $s$.
    """)
    return


@app.cell
def _():
    # Custom aircraft database: one row per aircraft, "folder" points at its tables
    data_root = mo.notebook_location().parent.parent / "data"
    ac_db = pd.read_csv(str(data_root / "AircraftDB_Custom.csv"))

    ac_options = {folder.replace("_", " "): folder for folder in ac_db["folder"]}

    ac_dropdown = mo.ui.dropdown(
        options=ac_options,
        value=next(iter(ac_options)),
        label="Aircraft",
    )
    return ac_db, ac_dropdown, data_root


@app.cell
def _(ac_db, ac_dropdown, data_root):
    # Load the selected aircraft's tabular data and scalar parameters
    ac_id = ac_dropdown.value
    aircraft = ac.Aircraft(str(data_root / ac_id), "", custom=True)

    params = ac_db[ac_db["folder"] == ac_id].iloc[0]

    ac_name = params["full_name"]

    # Wing area [m^2]
    S = params["S"].item()

    # Maximum lift coefficient [-]
    CLmax = params["CLmax"].item()

    # Mass sweep between OEM and MTOM of the selected aircraft [kg],
    # rounded inwards so the ends stay within the certified envelope
    m_min = params["OEM"].item()
    m_max = params["MTOM"].item()

    m_slider = mo.ui.slider(
        start=float(np.ceil(m_min / 50) * 50),
        stop=float(np.floor(m_max / 50) * 50),
        step=50,
        value=float(np.ceil(m_min / 50) * 50),
        label=r"$m$ (kg)",
        show_value=True,
    )

    # Altitude sweep bounded by the aircraft's own thrust table, so a new
    # aircraft gets its ceiling from its data. FL is hundreds of feet: this is
    # the one conversion the source tables force on us, applied once here so
    # everything downstream stays in metres.
    h_max = aircraft.df_dictionary["TvsM"]["FL"].max() * 100 * FT_TO_M

    h_slider = mo.ui.slider(
        start=0,
        stop=float(np.floor(h_max / 500) * 500),
        step=500,
        value=0,
        label=r"$h$ (m)",
        show_value=True,
    )

    # Thrust setting s, read off the TvsM table
    setting_dropdown = mo.ui.dropdown(
        options=list(aircraft.df_dictionary["TvsM"]["Setting"].unique()),
        value="Max",
        label=r"$s$",
    )
    return CLmax, S, ac_name, aircraft, h_slider, m_slider, setting_dropdown


@app.cell
def _(ac_dropdown, h_slider, m_slider, setting_dropdown):
    mo.hstack(
        [ac_dropdown, m_slider, h_slider, setting_dropdown],
        justify="center",
    )
    return


@app.cell
def _(h_slider, m_slider):
    # Weight [N]
    W = m_slider.value * atmos.g0

    # Altitude [m]
    h = h_slider.value
    return W, h


@app.cell
def _(CLmax, S, W, h):
    # Optimization domain. CL is swept up to CLmax with the zero excluded,
    # since V -> inf there, and the throttle spans its full range.
    n_mesh = plot_utils.meshgrid_n

    CL_array = np.linspace(0, CLmax, n_mesh + 1)[1:]
    dT_array = np.linspace(0, 1, n_mesh)

    # Objective function, obtained by eliminating V from the lift equation.
    # It does not depend on the throttle, so the surface is constant along dT.
    V_CLarray = np.sqrt(2 * W / (atmos.rho(h) * S * CL_array))
    V_surface = np.broadcast_to(V_CLarray[np.newaxis, :], (n_mesh, n_mesh))
    return CL_array, V_CLarray, V_surface, dT_array


@app.cell
def _(CL_array, CLmax, V_surface, ac_name, dT_array):
    # Objective surface over the (CL, dT) domain. Written out in full rather
    # than through plot_utils, so the numerical constraint traces can be
    # dropped straight in once c2 is solved.

    # The surface is clipped at twice the minimum speed, otherwise the
    # low-CL branch flattens everything else out
    _V_min = np.min(V_surface)
    _V_max = 2 * _V_min

    fig_initial = go.Figure()

    fig_initial.add_trace(
        go.Surface(
            x=CL_array,
            y=dT_array,
            z=V_surface,
            name="Velocity",
            opacity=0.9,
            colorscale="viridis",
            cmin=_V_min,
            cmax=_V_max,
            colorbar={"title": "V (m/s)"},
        )
    )

    # --- constraint traces go here -------------------------------------
    # c2 (T - D = 0) gives the equilibrium throttle dT_eq(CL), to be solved
    # numerically against the TvsM / CD0vsM / KvsM tables. It rides on the
    # surface, so it plots as
    #     go.Scatter3d(x=CL_array, y=dT_eq, z=V_CLarray, mode="lines", ...)
    # The design point marker (CL, dT, V) goes here too.
    # -------------------------------------------------------------------

    fig_initial.update_layout(
        scene_dragmode="turntable",
        scene=dict(
            xaxis=dict(
                title="C<sub>L</sub> (-)",
                range=[plot_utils.xy_lowerbound, CLmax],
            ),
            yaxis=dict(
                title="δ<sub>T</sub> (-)",
                range=[plot_utils.xy_lowerbound, 1],
            ),
            zaxis=dict(title="V (m/s)", range=[0, _V_max]),
        ),
        scene_camera=dict(eye=dict(x=1.35, y=1.35, z=1.35)),
        title={
            "text": f"Minimum speed for {ac_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        },
    )

    fig_initial
    return


if __name__ == "__main__":
    app.run()
