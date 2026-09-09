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
    from plotly.subplots import make_subplots
    import numpy as np
    import pandas as pd
    from core import atmos
    from core import aircraft as ac
    from core import plot_utils

    # from core.plot_utils import OptimumGridView
    from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
    from scipy.optimize import minimize
    from scipy.spatial import Delaunay

    # Set local/online filepath
    _defaults.FILEURL = _defaults.get_url()

    # Plotly dark mode template
    _defaults.set_plotly_template()

    # Data directory
    data_dir = str(mo.notebook_location().parent.parent / "data" / "AircraftDB_Standard.csv")

    # Source tables report altitude as flight level, everything else is SI
    FT_TO_M = 0.3048

    # Resolution of the CL sweep used to draw the constraint curve
    N_CURVE = 400


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Minimum drag: custom aircraft
    """)
    return


@app.cell
def _():
    # Set navbar on the right
    _defaults.set_sidebar()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    As shown in [Aircraft Custom](/?file=Models_Library/AircraftCustom.py), a
    custom aircraft is described by scalar parameters (such as wing area and
    maximum lift coefficient) and by tables for its aerodynamic coefficients
    and available thrust. This replaces the simplified analytical jet or
    propeller model with a data-driven one. The lift equation still gives speed
    explicitly, but the optimal lift coefficient, and therefore the minimum
    drag, must be found numerically because the remaining quantities are
    tabulated.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $$
    \begin{aligned}
        \min_{V, C_L, \delta_T}
        & \quad D = \frac{1}{2}\rho V^2S\left(C_{D_0}+K C_L^2\right) \\
        \text{subject to}
        & \quad c_1^\mathrm{eq} = L-W = \frac{1}{2}\rho V^2 S C_L - W = 0 \\
        & \quad c_2^\mathrm{eq} = T-D = \delta_T T_a - \frac{1}{2} \rho V^2 S (C_{D_0}+K C_L^2) =0 \\
        \text{for }
        & \quad C_L \in (0, C_{L_\mathrm{max}}] \\
        & \quad \delta_T \in [0, 1] \\
        \text{with }
        & \quad T_a = T_a(M, h, s), \quad s \in \{\mathrm{Max}, \mathrm{Mil}\} \\
        & \quad C_{D_0} = C_{D_0}(M) \\
        & \quad K = K(M, C_L) \\
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    As with the jet and propeller aircraft we started with, the problem can be simplified by eliminating the speed $V$ from the constraints, since it can be expressed as a function of the lift coefficient $C_L$ and the weight $W$. The optimization problem can then be rewritten as follows:


    $$
    \begin{aligned}
        \min_{C_L, \delta_T}
        & \quad D = \frac{W\left(C_{D_0}+K C_L^2\right)}{C_L} \\
        \text{subject to}
        & \quad c_2^\mathrm{eq} = T-D = \delta_T T_a - \frac{1}{2} \rho V^2 S (C_{D_0}+K C_L^2) =0 \\
        \text{for }
        & \quad C_L \in (0, C_{L_\mathrm{max}}] \\
        & \quad \delta_T \in [0, 1] \\
        \text{with }
        & \quad T_a = T_a(M, h, s), \quad s \in \{\mathrm{Max}, \mathrm{Mil}\} \\
        & \quad C_{D_0} = C_{D_0}(M) \\
        & \quad K = K(M, C_L) \\
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The custom-aircraft tables can account for compressibility effects that the
    simplified jet and propeller models omit. In the data used here, the drag
    polar depends on Mach number, including the rapid rise in parasitic drag
    near $M = 1$.

    Once $W$ and $h$ are fixed, choosing $C_L$ fixes the whole flight condition,
    so every table lookup follows from it:

    $$
    C_L \longrightarrow V(C_L) \longrightarrow M(C_L)
    \longrightarrow C_{D_0}(M),\ K(M,C_L),\ T_a(M,h,s).
    $$

    What is left to minimize is therefore a **surface** $D(C_L, \delta_T)$ over
    the rectangle $C_L \in (0, C_{L_\mathrm{max}}]$, $\delta_T \in [0,1]$. It
    has a bucket in $C_L$, where parasitic and induced drag trade against each
    other, and is flat along $\delta_T$, because the throttle does not appear
    in the drag written this way. Unlike the minimum- and maximum-speed
    problems the optimum is therefore usually **interior**: the shape of the
    polar sets it rather than a bound, and the limits below only take over
    where the bucket falls outside the flyable range.
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
    mass_step = 50  # kg

    m_slider = mo.ui.slider(
        start=float(np.ceil(m_min / mass_step) * mass_step),
        stop=float(np.floor(m_max / mass_step) * mass_step),
        step=mass_step,
        value=float(np.ceil(m_min / mass_step) * mass_step),
        label=r"$m$ (kg)",
        show_value=True,
    )

    # Altitude sweep bounded by the aircraft's own thrust table, so a new
    # aircraft gets its ceiling from its data.
    h_max = aircraft.df_dictionary["TvsM"]["FL"].max() * 100 * FT_TO_M
    h_step = 500  # m

    h_slider = mo.ui.slider(
        start=0,
        stop=float(np.floor(h_max / h_step) * h_step),
        step=h_step,
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
    return (
        CLmax,
        S,
        ac_name,
        aircraft,
        h_max,
        h_slider,
        m_slider,
        setting_dropdown,
    )


@app.cell
def _(aircraft, setting_dropdown):
    # Table interpolators. They depend only on the aircraft and the thrust setting,
    # never on the mass or altitude sliders, so marimo rebuilds them only when the
    # selection changes -- triangulating the thrust table is by far the most
    # expensive work here and has no business in the slider hot path.

    # Every table is digitised by hand from the source charts, and is read the same
    # way: interpolated where the chart has data, and never read outside it.

    cd0_table = aircraft.df_dictionary["CD0vsM"]
    CD0_M = cd0_table["M"].to_numpy(dtype=float)
    CD0_values = cd0_table["CD0"].to_numpy(dtype=float)


    def CD0_interp(M):
        """Parasitic drag coefficient at Mach number M."""
        # np.interp holds the end values outside the table, where the digitised curve is flat
        return np.interp(M, CD0_M, CD0_values)


    K_table = aircraft.df_dictionary["KvsM"].pivot(index="M", columns="CL", values="K")
    K_M = K_table.index.to_numpy(dtype=float)
    K_CL = K_table.columns.to_numpy(dtype=float)
    K_lookup = RegularGridInterpolator((K_M, K_CL), K_table.to_numpy(dtype=float))


    def K_interp(M, CL):
        """Induced drag factor at Mach number M and lift coefficient CL."""
        # The table is a full rectangle whose last column is CLmax, so the only thing
        # the clip ever catches is a hundredth of a Mach at the low end, where the
        # value is held rather than extrapolated, as for CD0 above.
        M, CL = np.broadcast_arrays(np.atleast_1d(M), np.atleast_1d(CL))
        M = np.clip(M, K_M[0], K_M[-1])
        CL = np.clip(CL, K_CL[0], K_CL[-1])
        return K_lookup(np.column_stack([M.ravel(), CL.ravel()])).reshape(M.shape)


    # The thrust chart is digitised one flight level at a time, each over its own Mach
    # window, so the grid it fills is rectangular but its domain is not.
    T_table = aircraft.df_dictionary["TvsM"].dropna(subset=["FL", "Ta"])
    T_table = T_table[T_table["Setting"] == setting_dropdown.value]

    T_grid = T_table.pivot(index="M", columns="FL", values="Ta")
    T_M = T_grid.index.to_numpy(dtype=float)
    T_h = T_grid.columns.to_numpy(dtype=float) * 100 * FT_TO_M

    # Gridded interpolation needs a table without holes: gaps inside a Mach window are
    # filled linearly, and the empty corners are held at the end values. Neither invents
    # data, because the hull test below masks them out again.
    T_filled = T_grid.interpolate(method="index", axis=0).ffill().bfill()

    # A shape-preserving spline is what these curves deserve, but evaluating one costs
    # some 300 us per call, which the sliders cannot afford. So it is sampled once onto
    # a 100 m altitude grid, the way the MATLAB pre-processing of the same charts does,
    # and everything below reads that grid linearly. The Mach axis needs no resampling:
    # the table is already spaced 0.01 apart, the step such a grid would use anyway.
    T_h_fine = np.union1d(np.arange(T_h[0], T_h[-1], 100.0), T_h)

    # The source table is in kN; everything else in this notebook is SI
    T_fine = PchipInterpolator(T_h, T_filled.to_numpy(dtype=float) * 1e3, axis=1)(T_h_fine)

    Ta_lookup = RegularGridInterpolator((T_M, T_h_fine), T_fine, bounds_error=False, fill_value=np.nan)

    # Convex hull of the digitised points, as the boundary of the usable domain
    T_hull = Delaunay(
        np.column_stack(
            [
                T_table["M"].to_numpy(dtype=float),
                T_table["FL"].to_numpy(dtype=float) * 100 * FT_TO_M,
            ]
        )
    )


    def Ta_interp(M, h):
        """Thrust available at Mach number M and altitude h.

        NaN outside the convex hull of the digitised points, which is precisely the
        meaning we want: the source chart says nothing about that flight condition,
        so it is not a usable one, and no value is invented for it.
        """
        M, h = np.broadcast_arrays(np.atleast_1d(M), np.atleast_1d(h))
        points = np.column_stack([M.ravel(), h.ravel()]).astype(float)
        inside = T_hull.find_simplex(points) >= 0
        return np.where(inside, Ta_lookup(points), np.nan).reshape(M.shape)


    # Largest thrust the table holds at this setting [N]. Used further down as
    # an axis anchor, so the performance diagrams stay framed on the thrust the
    # engine can actually produce rather than on the drag curve alone.
    Ta_max = float(T_table["Ta"].max()) * 1e3

    # The sweeps stop where the table does: below this CL the flight condition sits
    # past the Mach extent of TvsM, where the chart has nothing to say
    M_ceiling = T_M[-1]
    return CD0_interp, K_interp, M_ceiling, Ta_interp, Ta_max


@app.cell
def _(h_slider, m_slider):
    # Weight [N]
    W = m_slider.value * atmos.g0

    # Altitude [m]
    h = h_slider.value
    return W, h


@app.cell
def _(W, h, lift_coefficient_sweep, required_drag):
    # Optimization domain. CL is swept over the range the tables cover,
    # and the throttle spans its full range.
    n_mesh = plot_utils.meshgrid_n

    CL_array = lift_coefficient_sweep(W, h, n_mesh)
    dT_array = np.linspace(0, 1, n_mesh)

    # Objective function, obtained after eliminating V with the lift equation.
    # It does not depend on the throttle, so the surface is constant along dT.
    D_CLarray = required_drag(CL_array, W, h)
    D_surface = np.broadcast_to(D_CLarray[np.newaxis, :], (n_mesh, n_mesh))
    return CL_array, D_surface, dT_array


@app.cell
def _(CD0_interp, CLmax, K_interp, M_ceiling, S, Ta_interp):
    # The flight condition a lift coefficient implies, written as plain functions of
    # (W, h) because the flight envelope re-solves the same problem at every altitude
    def level_flight_speed(CL, W, h):
        """Speed in steady level flight at lift coefficient CL, from c1 solved for V."""
        return np.sqrt(2 * W / (atmos.rho(h) * S * np.atleast_1d(CL).astype(float)))


    def required_drag(CL, W, h):
        """Drag in steady level flight at lift coefficient CL."""
        # The polar is read at the Mach number the lift equation implies
        CL = np.atleast_1d(CL).astype(float)
        M = level_flight_speed(CL, W, h) / atmos.a(h)
        return W * (CD0_interp(M) + K_interp(M, CL) * CL**2) / CL


    def equilibrium_throttle(CL, W, h):
        """Throttle that satisfies c2 at lift coefficient CL, tracing the constraint curve."""
        CL = np.atleast_1d(CL).astype(float)
        M = level_flight_speed(CL, W, h) / atmos.a(h)
        return required_drag(CL, W, h) / Ta_interp(M, h)


    def lift_coefficient_sweep(W, h, n):
        """CL sweep from the tables' Mach ceiling up to CLmax."""
        CL_min = 2 * W / (atmos.rho(h) * S * (M_ceiling * atmos.a(h)) ** 2)
        return np.linspace(CL_min, CLmax, n)
    return (
        equilibrium_throttle,
        level_flight_speed,
        lift_coefficient_sweep,
        required_drag,
    )


@app.cell
def _(
    CLmax,
    Ta_interp,
    equilibrium_throttle,
    level_flight_speed,
    lift_coefficient_sweep,
    required_drag,
):
    # A design variable counts as sitting on its bound within this tolerance
    ACTIVE_TOL = 1e-4


    # Define the residual of the C2 constraints as scipy minimize reduces the residual up to 0, to have equivalence
    def c2_eq(CL, dT, W, h):
        """Residual of the equality constraint c2"""
        M = level_flight_speed(CL, W, h) / atmos.a(h)
        return float(dT * Ta_interp(M, h)[0] - required_drag(CL, W, h)[0]) / W


    def solve_min_drag(W, h):
        """Minimum drag in steady level flight, and the limit that sets it."""
        # Solving c2 for the throttle traces the constraint curve, which supplies the starting point
        CL_curve = lift_coefficient_sweep(W, h, N_CURVE)
        dT_curve = equilibrium_throttle(CL_curve, W, h)

        # A NaN throttle marks a flight condition the thrust chart never covered, so it
        # is dropped along with the ones that ask for more than full throttle
        covered = np.isfinite(dT_curve)
        on_surface = covered & (dT_curve <= 1)

        if not on_surface.any():
            # No point of the curve is both covered and flyable: no steady level flight here
            return None, np.nan, np.nan, np.nan, np.nan

        # The search stays inside the covered stretch of the sweep, so the tables are
        # never read where the source chart has nothing to say
        CL_floor = float(CL_curve[covered][0])
        CL_ceiling = CLmax if covered[-1] else float(CL_curve[covered][-1])

        # The best point of the curve starts the refinement off in the drag bucket
        CL_start = CL_curve[on_surface][np.argmin(required_drag(CL_curve[on_surface], W, h))]

        result = minimize(
            # Scaled by the weight, so the objective is the inverse lift-to-drag ratio
            # and scipy's tolerance means the same thing at every mass
            lambda x: float(required_drag(x[0], W, h)[0]) / W,
            np.array([CL_start, float(equilibrium_throttle(CL_start, W, h)[0])]),
            method="SLSQP",
            bounds=[(CL_floor, CL_ceiling), (0.0, 1.0)],
            constraints={"type": "eq", "fun": lambda x: c2_eq(x[0], x[1], W, h)},
            options={"disp": False, "ftol": 1e-9},
        )

        CL_opt, dT_opt = float(result.x[0]), float(result.x[1])

        # The drag bucket usually sits inside the flyable range, so unlike the speed
        # problems the optimum is interior unless one of the bounds cuts it off
        if CL_opt >= CLmax - ACTIVE_TOL:
            limit = "stall"
        elif dT_opt >= 1 - ACTIVE_TOL:
            limit = "thrust"
        else:
            limit = "interior"

        return (
            limit,
            CL_opt,
            dT_opt,
            float(required_drag(CL_opt, W, h)[0]),
            float(level_flight_speed(CL_opt, W, h)[0]),
        )


    mo.show_code()
    return (solve_min_drag,)


@app.cell
def _(
    W,
    equilibrium_throttle,
    h,
    level_flight_speed,
    lift_coefficient_sweep,
    required_drag,
):
    # The constraint curve at the selected flight condition, on the same sweep the
    # solver uses, so the traces below and the optimum come from one grid
    CL_fine = lift_coefficient_sweep(W, h, N_CURVE)
    V_fine = level_flight_speed(CL_fine, W, h)
    D_fine = required_drag(CL_fine, W, h)
    dT_fine = equilibrium_throttle(CL_fine, W, h)

    # Beyond full throttle the curve leaves the rectangle, and the aircraft cannot follow it
    on_surface = dT_fine <= 1
    return CL_fine, D_fine, V_fine, dT_fine, on_surface


@app.cell
def _(W, h, solve_min_drag):
    limit, CL_opt, dT_opt, D_min, V_opt = solve_min_drag(W, h)
    return CL_opt, D_min, V_opt, dT_opt, limit


@app.cell
def _(ac_dropdown, h_slider, m_slider, setting_dropdown):
    mo.hstack(
        [ac_dropdown, m_slider, h_slider, setting_dropdown],
        justify="center",
    )
    return


@app.cell
def _(
    CL_array,
    CL_fine,
    CL_opt,
    CLmax,
    D_fine,
    D_min,
    D_surface,
    ac_name,
    dT_array,
    dT_fine,
    dT_opt,
    limit,
    on_surface,
):
    # Objective surface over the (CL, dT) domain. Written out in full rather
    # than through plot_utils, so the numerical constraint traces can be
    # dropped straight in once c2 is solved.

    # The surface is clipped at twice the minimum drag, otherwise the two
    # branches climbing out of the bucket flatten everything else out
    _D_min = np.min(D_surface)
    _D_max = 2 * _D_min

    fig_initial = go.Figure()

    fig_initial.add_trace(
        go.Surface(
            x=CL_array,
            y=dT_array,
            z=D_surface,
            name="Drag",
            opacity=0.9,
            colorscale="viridis",
            cmin=_D_min,
            cmax=_D_max,
            colorbar={"title": "D (N)"},
        )
    )

    # The c2 constraint rides on the surface, and is blanked where it asks for
    # more than full throttle rather than being drawn outside the rectangle
    fig_initial.add_trace(
        go.Scatter3d(
            x=CL_fine,
            y=np.where(on_surface, dT_fine, np.nan),
            z=np.where(on_surface, D_fine, np.nan),
            mode="lines",
            name="c<sub>2</sub> constraint",
            showlegend=False,
            line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
        )
    )

    if limit:
        fig_initial.add_trace(
            go.Scatter3d(
                x=[CL_opt],
                y=[dT_opt],
                z=[D_min],
                mode="markers",
                name="Minimum drag",
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
            zaxis=dict(title="D (N)", range=[0, _D_max]),
        ),
        scene_camera=dict(eye=dict(x=1.35, y=1.35, z=1.35)),
        title={
            "text": f"Minimum drag for {ac_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        },
    )

    fig_initial
    return


@app.cell
def _(CL_opt, D_min, V_opt, dT_opt, h, limit, setting_dropdown):
    if limit is None:
        print_output = mo.md(r"""
        Every point of the constraint curve asks for more than full throttle: at this
        combination of weight, altitude and thrust setting the aircraft has no steady
        level flight condition at all. Lower the altitude or the mass, or switch the
        thrust setting.
        """)
    else:
        _verdict = {
            "interior": (
                "the drag bucket falls inside the flyable range, so the minimum is "
                "**interior** and no limiting bound is active"
            ),
            "stall": (
                "the wing reaches $C_{L_\\mathrm{max}}$ before the bucket is reached, "
                "so the minimum drag is **stall-limited**"
            ),
            "thrust": (
                "the throttle reaches $\\delta_T = 1$ before the bucket is reached, "
                "so the minimum drag is **thrust-limited**"
            ),
        }[limit]

        print_output = mo.md(f"""
        At $h = {h:.0f}$ m on the {setting_dropdown.value} thrust setting,
        {_verdict}.

        | | |
        |---|---|
        | $D_\\mathrm{{min}}$ | {D_min:.0f} N |
        | $V^*$ | {V_opt:.1f} m/s |
        | $M$ | {V_opt / atmos.a(h):.3f} |
        | $C_L^*$ | {CL_opt:.3f} |
        | $\\delta_T^*$ | {dT_opt:.3f} |

        None of the tables behind these numbers is ever extrapolated: $T_a$ is
        interpolated inside the convex hull of the digitised `TvsM` points and is
        undefined outside it, so a flight condition the source chart never covered
        counts as unavailable rather than being invented.
        """)

    print_output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Performance diagrams and flight envelope
    """)
    return


@app.cell
def _(
    CL_fine,
    CLmax,
    Ta_interp,
    Ta_max,
    V_fine,
    W,
    h,
    level_flight_speed,
    required_drag,
):
    # Performance diagrams at the selected flight condition. Every curve is a
    # function of the same fine CL sweep the solver uses, mapped onto speed
    # through c1, so the diagrams and the optimum cannot drift apart.
    drag_curve = required_drag(CL_fine, W, h)
    thrust_curve = Ta_interp(V_fine / atmos.a(h), h)

    power_required = drag_curve * V_fine
    power_available = thrust_curve * V_fine

    V_E = float(V_fine[np.argmin(drag_curve)])
    V_P = float(V_fine[np.argmin(power_required)])
    V_stall = float(level_flight_speed(CLmax, W, h)[0])

    # Axis anchors. Taken at sea level so the frame stays put as the altitude
    # slider moves, and floored on the installed thrust so the available curves
    # stay on the plot even where the drag bucket is shallow.
    _drag_sl = required_drag(CL_fine, W, 0)
    _V_sl = level_flight_speed(CL_fine, W, 0)

    drag_ylim = max(6 * float(_drag_sl.min()), 1.2 * Ta_max)
    power_ylim = drag_ylim * float(_V_sl[np.argmin(_drag_sl)]) / 1e3
    return (
        V_E,
        V_P,
        V_stall,
        drag_curve,
        drag_ylim,
        power_available,
        power_required,
        power_ylim,
        thrust_curve,
    )


@app.cell
def _(CLmax, W, h_max, level_flight_speed, solve_min_drag):
    # The flight envelope is the same problem re-solved at every altitude the
    # thrust table covers, by the same call: one curve sweep and one SLSQP
    # solve each, so a full envelope costs a couple hundred milliseconds.
    h_envelope = np.linspace(0, h_max, 61)

    _solved = [solve_min_drag(W, _h) for _h in h_envelope]
    limits_envelope = [_s[0] for _s in _solved]
    V_envelope = np.array([_s[4] for _s in _solved])

    Vstall_envelope = level_flight_speed(CLmax, W, h_envelope)
    return V_envelope, Vstall_envelope, h_envelope, limits_envelope


@app.cell
def _(ac_dropdown, h_slider, m_slider, setting_dropdown):
    mo.hstack(
        [ac_dropdown, m_slider, h_slider, setting_dropdown],
        justify="center",
    )
    return


@app.cell
def _(
    CL_array,
    CL_fine,
    CL_opt,
    CLmax,
    D_surface,
    V_E,
    V_P,
    V_envelope,
    V_fine,
    V_opt,
    V_stall,
    Vstall_envelope,
    dT_array,
    dT_fine,
    dT_opt,
    drag_curve,
    drag_ylim,
    h,
    h_envelope,
    h_max,
    limit,
    on_surface,
    power_available,
    power_required,
    power_ylim,
    thrust_curve,
):
    # Drag (top left), power (top right), optimization domain (bottom left) and
    # flight envelope (bottom right). Written out in full rather than through
    # plot_utils.OptimumGridView, which reads its curves off the closed-form
    # jet and propeller models and frames the envelope on a fixed 13 km ceiling.
    # Every trace names the panel it belongs to through its xaxis/yaxis pair:
    # x1/y1 is top left, x2/y2 top right, x3/y3 bottom left, x4/y4 bottom right.

    fig_grid = make_subplots(rows=2, cols=2, horizontal_spacing=0.1, vertical_spacing=0.15)

    # With no optimum to speak of there is no meaningful throttle setting, and
    # full throttle is the only honest thing to draw against the required curves.
    _dT_shown = dT_opt if np.isfinite(dT_opt) else 1.0

    # Top left: drag required and thrust available, with dotted verticals at the
    # minimum drag and minimum power speeds, the stall speed in its own colour,
    # and a grey arrow pointing the way the lift coefficient grows.
    fig_grid.add_traces(
        [
            go.Scattergl(
                x=V_fine,
                y=drag_curve,
                xaxis="x1",
                yaxis="y1",
                mode="lines",
                name="D",
                showlegend=False,
                line=dict(color=plot_utils.DRAG_COLOR, width=2),
            ),
            go.Scattergl(
                x=V_fine,
                y=_dT_shown * thrust_curve,
                xaxis="x1",
                yaxis="y1",
                mode="lines",
                name="T",
                showlegend=False,
                line=dict(color=plot_utils.AVAILABLE_COLOR),
            ),
            go.Scattergl(
                x=[V_E, V_E],
                y=[0, drag_ylim],
                xaxis="x1",
                yaxis="y1",
                mode="lines",
                showlegend=False,
                line=dict(dash="dot", color=plot_utils.LIGHTGREY),
            ),
            go.Scattergl(
                x=[V_P, V_P],
                y=[0, drag_ylim],
                xaxis="x1",
                yaxis="y1",
                mode="lines",
                showlegend=False,
                line=dict(dash="dot", color=plot_utils.LIGHTGREY),
            ),
            go.Scattergl(
                x=[V_stall, V_stall],
                y=[0, drag_ylim],
                xaxis="x1",
                yaxis="y1",
                mode="lines",
                showlegend=False,
                line=dict(dash="dot", color=plot_utils.CLMAX_AXES),
            ),
            go.Scattergl(
                x=[V_stall - 20, 2 * plot_utils.axes_max_speed],
                y=[0.1 * drag_ylim, 0.1 * drag_ylim],
                xaxis="x1",
                yaxis="y1",
                mode="lines",
                showlegend=False,
                line=dict(color=plot_utils.LIGHTGREY, width=1),
            ),
            go.Scattergl(
                x=[V_stall - 20],
                y=[0.1 * drag_ylim],
                xaxis="x1",
                yaxis="y1",
                mode="markers",
                showlegend=False,
                marker=dict(color=plot_utils.LIGHTGREY, size=10, symbol="arrow-left"),
            ),
        ]
    )

    # Top right: the same story in power, required against available, with the
    # same three verticals and the same lift coefficient arrow.
    fig_grid.add_traces(
        [
            go.Scattergl(
                x=V_fine,
                y=power_required / 1e3,
                xaxis="x2",
                yaxis="y2",
                mode="lines",
                name="P",
                showlegend=False,
                line=dict(color=plot_utils.POWER_COLOR, width=2),
            ),
            go.Scattergl(
                x=V_fine,
                y=_dT_shown * power_available / 1e3,
                xaxis="x2",
                yaxis="y2",
                mode="lines",
                name="P<sub>a</sub>",
                showlegend=False,
                line=dict(color=plot_utils.AVAILABLE_COLOR),
            ),
            go.Scattergl(
                x=[V_E, V_E],
                y=[0, power_ylim],
                xaxis="x2",
                yaxis="y2",
                mode="lines",
                showlegend=False,
                line=dict(dash="dot", color=plot_utils.LIGHTGREY),
            ),
            go.Scattergl(
                x=[V_P, V_P],
                y=[0, power_ylim],
                xaxis="x2",
                yaxis="y2",
                mode="lines",
                showlegend=False,
                line=dict(dash="dot", color=plot_utils.LIGHTGREY),
            ),
            go.Scattergl(
                x=[V_stall, V_stall],
                y=[0, power_ylim],
                xaxis="x2",
                yaxis="y2",
                mode="lines",
                showlegend=False,
                line=dict(dash="dot", color=plot_utils.CLMAX_AXES),
            ),
            go.Scattergl(
                x=[V_stall - 20, 2 * plot_utils.axes_max_speed],
                y=[0.1 * power_ylim, 0.1 * power_ylim],
                xaxis="x2",
                yaxis="y2",
                mode="lines",
                showlegend=False,
                line=dict(color=plot_utils.LIGHTGREY, width=1),
            ),
            go.Scattergl(
                x=[V_stall - 20],
                y=[0.1 * power_ylim],
                xaxis="x2",
                yaxis="y2",
                mode="markers",
                showlegend=False,
                marker=dict(color=plot_utils.LIGHTGREY, size=10, symbol="arrow-left"),
            ),
        ]
    )

    # Bottom left: the optimization domain, the surface plot above seen from
    # overhead, with the level flight constraint drawn on top of it.
    fig_grid.add_traces(
        [
            go.Heatmap(
                x=CL_array,
                y=dT_array,
                z=D_surface,
                xaxis="x3",
                yaxis="y3",
                zsmooth="best",
                opacity=0.9,
                colorscale="viridis",
                zmin=np.min(D_surface),
                zmax=2 * np.min(D_surface),
                hovertemplate="C<sub>L</sub>=%{x:.3f}<br>δ<sub>T</sub>=%{y:.2f}<br>D=%{z:.0f} N<extra></extra>",
                colorbar={"title": ""},
            ),
            go.Scattergl(
                x=CL_fine,
                y=np.where(on_surface, dT_fine, np.nan),
                xaxis="x3",
                yaxis="y3",
                mode="lines",
                showlegend=False,
                line=dict(color=plot_utils.CONSTRAINT_CLR, width=10),
            ),
        ]
    )

    # Bottom right: the minimum-drag speed against altitude, with the stall
    # speed and the speed of sound as references. The labels sit at 80% of the
    # way up the envelope, high enough to clear the boundary.
    _i_label = int(0.8 * len(h_envelope))

    fig_grid.add_traces(
        [
            go.Scattergl(
                x=V_envelope,
                y=h_envelope / 1e3,
                xaxis="x4",
                yaxis="y4",
                mode="lines",
                name="V<sub>E</sub>",
                showlegend=False,
                line=dict(color=plot_utils.SALMON, width=2),
            ),
            go.Scattergl(
                x=Vstall_envelope,
                y=h_envelope / 1e3,
                xaxis="x4",
                yaxis="y4",
                mode="lines",
                showlegend=False,
                line=dict(color=plot_utils.LIGHTGREY, width=1, dash="dash"),
            ),
            go.Scattergl(
                x=atmos.a(h_envelope),
                y=h_envelope / 1e3,
                xaxis="x4",
                yaxis="y4",
                mode="lines",
                showlegend=False,
                line=dict(color=plot_utils.LIGHTGREY, width=2, dash="dash"),
            ),
            go.Scatter(
                x=[Vstall_envelope[_i_label]],
                y=[h_envelope[_i_label] / 1e3],
                xaxis="x4",
                yaxis="y4",
                mode="markers+text",
                marker=dict(size=1, color=plot_utils.LIGHTGREY),
                text=["V<sub>stall</sub>"],
                textposition="top left",
                hoverinfo="skip",
                showlegend=False,
            ),
            go.Scatter(
                x=[atmos.a(h_envelope[_i_label]) - 5],
                y=[h_envelope[_i_label] / 1e3],
                xaxis="x4",
                yaxis="y4",
                mode="markers+text",
                marker=dict(size=1, color=plot_utils.LIGHTGREY),
                text=["M1.0"],
                textposition="top left",
                hoverinfo="skip",
                showlegend=False,
            ),
        ]
    )

    if limit:
        # The solved minimum, marked on all four panels: on the drag and power
        # curves at the speed where it occurs, at its lift coefficient and
        # throttle setting in the domain, and at the current altitude on the
        # envelope.
        _i_min = np.argmin(np.abs(V_fine - V_opt))

        fig_grid.add_traces(
            [
                go.Scattergl(
                    x=[V_opt],
                    y=[drag_curve[_i_min]],
                    xaxis="x1",
                    yaxis="y1",
                    mode="markers",
                    showlegend=False,
                    marker=dict(size=10, color=plot_utils.WHITE, symbol="circle"),
                ),
                go.Scattergl(
                    x=[V_opt],
                    y=[power_required[_i_min] / 1e3],
                    xaxis="x2",
                    yaxis="y2",
                    mode="markers",
                    showlegend=False,
                    marker=dict(size=10, color=plot_utils.WHITE, symbol="circle"),
                ),
                go.Scattergl(
                    x=[CL_opt],
                    y=[dT_opt],
                    xaxis="x3",
                    yaxis="y3",
                    mode="markers",
                    showlegend=False,
                    marker=dict(size=10, color=plot_utils.WHITE, symbol="circle"),
                ),
                go.Scattergl(
                    x=[V_opt],
                    y=[h / 1e3],
                    xaxis="x4",
                    yaxis="y4",
                    mode="markers",
                    showlegend=False,
                    marker=dict(size=10, color=plot_utils.WHITE, symbol="circle"),
                ),
            ]
        )

    fig_grid.update_layout(
        height=800,
        dragmode="pan",
        showlegend=False,
        xaxis=dict(title=r"$V \; (\text{m/s})$", range=[0, plot_utils.axes_max_speed]),
        yaxis=dict(title=r"$D \: (\text{N})$", range=[0, drag_ylim]),
        xaxis2=dict(title=r"$V \; (\text{m/s})$", range=[0, plot_utils.axes_max_speed]),
        yaxis2=dict(title=r"$P \: (\text{kW})$", range=[0, power_ylim]),
        xaxis3=dict(
            title=r"$C_L\:(\text{-})$",
            range=[plot_utils.xy_lowerbound, CLmax],
            showgrid=True,
            gridcolor="#515151",
            gridwidth=1,
        ),
        yaxis3=dict(
            title=r"$\delta_T \:(\text{-})$",
            range=[plot_utils.axes_min_dT, plot_utils.axes_max_dT],
            showgrid=True,
            gridcolor="#515151",
            gridwidth=1,
        ),
        xaxis4=dict(
            title=r"$V \: \text{(m/s)}$",
            range=[0, plot_utils.axes_max_speed],
            showgrid=True,
            gridcolor="#515151",
            gridwidth=1,
        ),
        yaxis4=dict(
            title=r"$h \: \text{(km)}$",
            range=[0, h_max / 1e3],
            showgrid=True,
            gridcolor="#515151",
            gridwidth=1,
        ),
    )

    fig_grid
    return


@app.cell
def _(V_E, V_P, V_stall, limits_envelope):
    # V_E and V_P are found by searching over CL in (0, CLmax], so a minimum
    # that would want a lift coefficient past the stall comes back pinned to
    # the stall speed rather than reported as unreachable.
    _at_stall = [
        _name for _name, _speed in (("minimum drag", V_E), ("minimum power", V_P)) if np.isclose(_speed, V_stall)
    ]
    if not _at_stall:
        _reachable = (
            "Both sampled aerodynamic minima lie above the stall speed, so they "
            "are interior to the admissible lift-coefficient range."
        )
    else:
        _pinned = " and ".join(_at_stall)
        _reachable = (
            f"The sampled {_pinned} search{'es' if len(_at_stall) > 1 else ''} "
            "reaches the stall boundary before finding an interior minimum; an "
            "unconstrained optimum beyond $C_{L_\\mathrm{max}}$ is not reachable "
            "in steady level flight."
        )

    # How the solved optimum is set, counted over the altitudes of the envelope
    _tally = ", ".join(
        f"{_label} at {_n} of them"
        for _key, _label in (
            ("interior", "interior"),
            ("stall", "stall-limited"),
            ("thrust", "thrust-limited"),
            ("data", "data-limited"),
            (None, "with no level flight condition at all"),
        )
        if (_n := limits_envelope.count(_key))
    )

    mo.md(f"""
    At the selected flight condition the minimum-drag speed is $V_E = {V_E:.1f}$ m/s and the
    minimum-power speed is $V_P = {V_P:.1f}$ m/s, against a stall speed of $V_s = {V_stall:.1f}$ m/s.
    {_reachable}
    """)
    return


if __name__ == "__main__":
    app.run()
