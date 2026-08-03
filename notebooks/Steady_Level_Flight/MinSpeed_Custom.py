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
    from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
    from scipy.optimize import brentq

    # Set local/online filepath
    _defaults.FILEURL = _defaults.get_url()

    # Plotly dark mode template
    _defaults.set_plotly_template()

    # Data directory
    data_dir = str(mo.notebook_location().parent.parent / "data" / "AircraftDB_Standard.csv")

    # Source tables report altitude as flight level, everything else is SI
    FT_TO_M = 0.3048

    # Resolution of the CL sweep used to solve the constraint.
    N_SOLVE = 400


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
    """)
    return


@app.cell
def _():
    mo.md(r"""
    At first sight these dependencies look circular: $C_{D_0}$ needs $M$, $M$ needs $V$, and $V$ is precisely the quantity being minimised. They are not. Once the weight $W$ and the altitude $h$ are chosen, the lift equation $c_1^\mathrm{eq}$ pins the speed to the lift coefficient, so **picking $C_L$ fixes the entire flight condition**, and every table lookup downstream becomes an explicit evaluation:

    $$
    C_L \;\longrightarrow\; V = \sqrt{\frac{2W}{\rho(h) S C_L}} \;\longrightarrow\; M = \frac{V}{a(h)} \;\longrightarrow\; C_{D_0}(M),\; K(M, C_L),\; T_a(M, h, s)
    $$

    Nothing here is iterative. Nor does $c_2^\mathrm{eq}$ have to be solved as an equation, because the throttle enters it *linearly*: it can be solved **for** $\delta_T$ in closed form, giving the throttle setting that balances thrust and drag at each lift coefficient,

    $$
    \delta_T^\mathrm{eq}(C_L) = \frac{D(C_L)}{T_a(M(C_L), h, s)}, \qquad D(C_L) = \frac{W\left(C_{D_0}(M) + K(M, C_L)\, C_L^2\right)}{C_L}
    $$

    which is the same relation used for the simplified jet and propeller aircraft, only with $C_{D_0}$, $K$ and $T_a$ read from tables instead of held constant.

    The two-variable problem has therefore collapsed to a one-dimensional one. Since $V(C_L)$ decreases monotonically, the minimum speed is reached at the **largest lift coefficient that remains feasible**, feasibility meaning $\delta_T^\mathrm{eq}(C_L) \leq 1$ at a flight condition the engine data actually covers. Only a scalar root-find on the boundary $\delta_T^\mathrm{eq}(C_L) = 1$ is needed, and only in the case where the throttle saturates before the wing stalls.
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
def _(aircraft, setting_dropdown):
    # Table interpolators. They depend only on the aircraft and the thrust
    # setting, never on the mass or altitude sliders, so marimo rebuilds them
    # only when the selection changes -- LinearNDInterpolator is by far the
    # most expensive object here and has no business in the slider hot path.

    cd0_table = aircraft.df_dictionary["CD0vsM"]


    def CD0_of(M):
        """Parasitic drag coefficient at Mach number M.

        np.interp holds the end values outside the table, which is what we
        want: the digitised curve is flat well before its low-Mach end.
        """
        return np.interp(M, cd0_table["M"], cd0_table["CD0"])


    K_table = aircraft.df_dictionary["KvsM"].pivot(index="M", columns="CL", values="K")
    K_M = K_table.index.to_numpy(dtype=float)
    K_CL = K_table.columns.to_numpy(dtype=float)
    K_interp = RegularGridInterpolator(
        (K_M, K_CL),
        K_table.to_numpy(dtype=float),
        bounds_error=False,
        fill_value=None,
    )


    def K_of(M, CL):
        """Induced drag factor at Mach number M and lift coefficient CL.

        Mach is clipped to the table first, so fill_value=None extrapolates
        along CL only. That extrapolation is unavoidable: CLmax sits past the
        table's last column, and it is exactly the stall point that decides
        the answer. Extrapolating rather than clamping, since clamping would
        understate the induced drag right where it matters most.
        """
        M = np.clip(np.atleast_1d(M), K_M[0], K_M[-1])
        return K_interp(np.column_stack([M, np.atleast_1d(CL)]))


    T_table = aircraft.df_dictionary["TvsM"]
    T_table = T_table[T_table["Setting"] == setting_dropdown.value].dropna(subset=["Ta"])
    Ta_interp = LinearNDInterpolator(
        np.column_stack(
            [
                T_table["M"].to_numpy(dtype=float),
                T_table["FL"].to_numpy(dtype=float) * 100 * FT_TO_M,
            ]
        ),
        # The source table is in kN; everything else in this notebook is SI
        T_table["Ta"].to_numpy(dtype=float) * 1e3,
    )


    def Ta_of(M, h):
        """Thrust available at Mach number M and altitude h.

        The table is digitised one flight level at a time, each covering a
        different Mach window, so its points are scattered rather than
        gridded. Outside their convex hull the interpolator returns NaN --
        which is precisely the meaning we want: the source chart says nothing
        about that flight condition, so it is not a usable one.
        """
        M = np.atleast_1d(M)
        return Ta_interp(M, np.full_like(M, h, dtype=float))
    return CD0_of, K_of, Ta_of


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
    return CL_array, V_surface, dT_array


@app.cell
def _(CD0_of, CLmax, K_of, S, Ta_of, W, h):
    # The c2 constraint, solved for the throttle. With W and h fixed, c1 makes
    # V a function of CL alone, so this is a single forward pass per CL: no
    # iteration and no simultaneous root-find.


    def equilibrium_throttle(CL):
        """Throttle setting that balances thrust and drag at lift coefficient CL.

        Returns NaN wherever the resulting flight condition falls outside the
        digitised thrust envelope.
        """
        CL = np.atleast_1d(CL).astype(float)
        V = np.sqrt(2 * W / (atmos.rho(h) * S * CL))
        M = V / atmos.a(h)
        D = W * (CD0_of(M) + K_of(M, CL) * CL**2) / CL
        return D / Ta_of(M, h)


    # Own sweep, finer than the plotting mesh, so the thrust-limited optimum
    # can be bracketed properly
    CL_fine = np.linspace(0, CLmax, N_SOLVE + 1)[1:]
    V_fine = np.sqrt(2 * W / (atmos.rho(h) * S * CL_fine))
    dT_fine = equilibrium_throttle(CL_fine)

    feasible = np.isfinite(dT_fine) & (dT_fine <= 1)
    return CL_fine, V_fine, dT_fine, equilibrium_throttle, feasible


@app.cell
def _(CL_fine, CLmax, S, W, dT_fine, equilibrium_throttle, feasible, h):
    # V(CL) decreases monotonically, so the minimum speed sits at the largest
    # feasible lift coefficient. Which limit binds there is the whole story.

    if not feasible.any():
        # Nothing the aircraft can hold in steady level flight at this
        # combination of weight, altitude and thrust setting
        regime = "none"
        CL_opt = dT_opt = V_min = np.nan
    else:
        i_last = np.flatnonzero(feasible)[-1]

        if i_last == len(CL_fine) - 1:
            # The wing stalls before the throttle saturates
            regime = "stall"
            CL_opt = CLmax
        elif np.isfinite(dT_fine[i_last + 1]):
            # A genuine deltaT = 1 crossing: refine it on the bracket, since
            # the sweep only locates it to within one grid step
            regime = "thrust"
            CL_opt = brentq(
                lambda CL: float(equilibrium_throttle(CL)[0]) - 1,
                CL_fine[i_last],
                CL_fine[i_last + 1],
            )
        else:
            # The thrust table ran out, not the engine. Report where the data
            # stops rather than passing this off as a thrust limit.
            regime = "data"
            CL_opt = CL_fine[i_last]

        dT_opt = float(equilibrium_throttle(CL_opt)[0])
        V_min = float(np.sqrt(2 * W / (atmos.rho(h) * S * CL_opt)))
    return CL_opt, V_min, dT_opt, regime


@app.cell
def _(
    CL_array,
    CL_fine,
    CL_opt,
    CLmax,
    V_fine,
    V_min,
    V_surface,
    ac_name,
    dT_array,
    dT_fine,
    dT_opt,
    feasible,
    regime,
):
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

    # The c2 constraint rides on the surface. Infeasible points are blanked
    # rather than dropped, so the line breaks where the thrust envelope ends
    # instead of being bridged by a segment that means nothing.
    fig_initial.add_trace(
        go.Scatter3d(
            x=CL_fine,
            y=np.where(feasible, dT_fine, np.nan),
            z=np.where(feasible, V_fine, np.nan),
            mode="lines",
            name="c<sub>2</sub> constraint",
            showlegend=False,
            line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
        )
    )

    if regime != "none":
        fig_initial.add_trace(
            go.Scatter3d(
                x=[CL_opt],
                y=[dT_opt],
                z=[V_min],
                mode="markers",
                name="Minimum speed",
                showlegend=False,
                marker=dict(size=4, color="white", symbol="circle"),
            )
        )

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


@app.cell
def _(CL_opt, V_min, dT_opt, h, regime, setting_dropdown):
    _verdict = {
        "stall": (
            "**stall-limited**: the wing reaches $C_{L_\\mathrm{max}}$ while the "
            "engine still has throttle in hand, so the minimum speed is the stall "
            "speed and the constraint curve stops short of $\\delta_T = 1$."
        ),
        "thrust": (
            "**thrust-limited**: the throttle saturates before the wing stalls, so "
            "the minimum speed is set by where the constraint curve crosses "
            "$\\delta_T = 1$, at a lift coefficient below $C_{L_\\mathrm{max}}$."
        ),
        "data": (
            "**data-limited**: the constraint curve runs off the edge of the "
            "digitised thrust table before either the wing or the engine reaches "
            "its limit. The value below is where the source data stops, not a "
            "physical limit of the aircraft."
        ),
    }

    if regime == "none":
        print_output = mo.md(r"""
        No point of the $C_L$ sweep satisfies $c_2^\mathrm{eq}$ within the digitised
        thrust envelope: at this combination of weight, altitude and thrust setting
        the aircraft has no steady level flight condition the engine tables can
        vouch for. Lower the altitude or the mass, or switch the thrust setting.
        """)
    else:
        print_output = mo.md(f"""
        At $h = {h:.0f}$ m on the {setting_dropdown.value} thrust setting, the case is
        {_verdict[regime]}

        | | |
        |---|---|
        | $V_\\mathrm{{min}}$ | {V_min:.1f} m/s |
        | $M$ | {V_min / atmos.a(h):.3f} |
        | $C_L^*$ | {CL_opt:.3f} |
        | $\\delta_T^*$ | {dT_opt:.3f} |

        Two assumptions are worth keeping in view when reading these numbers. $K$ is
        extrapolated linearly in $C_L$ past the last column of `KvsM`, since
        $C_{{L_\\mathrm{{max}}}}$ lies beyond it; and $T_a$ is restricted to the convex
        hull of the digitised `TvsM` points, so a flight condition the source chart
        never covered counts as unavailable rather than being invented by
        extrapolation.
        """)

    print_output
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
