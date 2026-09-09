import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path.cwd()))

    import marimo as mo

    from core import _defaults
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    import pandas as pd
    from core import atmos
    from core import aircraft as ac
    from core import plot_utils

    from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
    from scipy.optimize import minimize

    _defaults.FILEURL = _defaults.get_url()
    _defaults.set_plotly_template()

    data_dir = str(
        mo.notebook_location().parent.parent / "data" / "AircraftDB_Standard.csv"
    )

    # Source tables report altitude as flight level, everything else is SI
    FT_TO_M = 0.3048

    # Resolution of the CL sweep used to draw the constraint curve
    N_CURVE = 400

    # A design variable counts as sitting on its bound within this tolerance
    ACTIVE_TOL = 1e-4


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Maximum speed: custom aircraft
    """)
    return


@app.cell
def _():
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
    explicitly, but the limiting lift coefficient, and therefore the maximum
    speed, must be found numerically because the remaining quantities are
    tabulated.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $$
    \begin{aligned}
        \max_{V, C_L, \delta_T}
        & \quad V \\
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
    As with the jet and propeller aircraft, the problem can be simplified by
    eliminating the speed $V$ through the lift constraint. Maximizing $V$ is
    equivalent to minimizing its reciprocal, $1/V$, so the problem can be put
    in the standard minimization form


    $$
    \begin{aligned}
        \min_{C_L, \delta_T}
        & \quad \frac{1}{V} = \sqrt{\frac{\rho S C_L}{2W}} \\
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

    What is left to minimize is therefore a **surface** $1/V(C_L, \delta_T)$
    over the rectangle $C_L \in (0, C_{L_\mathrm{max}}]$,
    $\delta_T \in [0,1]$. It rises as $C_L$ grows and is flat along
    $\delta_T$, because the throttle does not appear in the lift equation.
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
    ac_id = ac_dropdown.value
    aircraft = ac.Aircraft(str(data_root / ac_id), "", custom=True)

    params = ac_db[ac_db["folder"] == ac_id].iloc[0]

    ac_name = params["full_name"]

    S = params["S"].item()
    CLmax = params["CLmax"].item()

    # Mass sweep from OEM to MTOM [kg], rounded inwards to stay inside the envelope
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

    # Altitude sweep bounded by the aircraft's own thrust table
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
    # Table interpolators, rebuilt only when the aircraft or thrust setting changes

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
        # The clip only catches a hundredth of a Mach at the low end, held not extrapolated
        M, CL = np.broadcast_arrays(np.atleast_1d(M), np.atleast_1d(CL))
        M = np.clip(M, K_M[0], K_M[-1])
        CL = np.clip(CL, K_CL[0], K_CL[-1])
        return K_lookup(np.column_stack([M.ravel(), CL.ravel()])).reshape(M.shape)


    # Digitised one flight level at a time, each over its own Mach window
    T_table = aircraft.df_dictionary["TvsM"].dropna(subset=["FL", "Ta"])
    T_table = T_table[T_table["Setting"] == setting_dropdown.value]

    T_grid = T_table.pivot(index="M", columns="FL", values="Ta")
    T_M = T_grid.index.to_numpy(dtype=float)
    T_h = T_grid.columns.to_numpy(dtype=float) * 100 * FT_TO_M

    # Gaps filled linearly and empty corners held; the window test below masks them out
    T_filled = T_grid.interpolate(method="index", axis=0).ffill().bfill()

    T_h_fine = np.union1d(np.arange(T_h[0], T_h[-1], 100.0), T_h)

    # The source table is in kN; everything else in this notebook is SI
    T_fine = PchipInterpolator(T_h, T_filled.to_numpy(dtype=float) * 1e3, axis=1)(T_h_fine)

    Ta_lookup = RegularGridInterpolator(
        (T_M, T_h_fine), T_fine, bounds_error=False, fill_value=np.nan
    )

    # Usable domain: the chart's Mach window per flight level, interpolated in altitude
    T_window = T_table.groupby("FL")["M"].agg(["min", "max"])
    T_window_h = T_window.index.to_numpy(dtype=float) * 100 * FT_TO_M
    T_window_lo = T_window["min"].to_numpy(dtype=float)
    T_window_hi = T_window["max"].to_numpy(dtype=float)


    def Ta_interp(M, h):
        """Thrust available at Mach number M and altitude h.

        NaN outside the Mach window the chart covers at that altitude, which is
        precisely the meaning we want: the source chart says nothing about that flight
        condition, so it is not a usable one, and no value is invented for it. A convex
        hull will not do in its place: the windows are not nested, so a hull reaches
        past the chart's own edge and reads the held corner values there.
        """
        M, h = np.broadcast_arrays(np.atleast_1d(M), np.atleast_1d(h))
        points = np.column_stack([M.ravel(), h.ravel()]).astype(float)
        inside = (points[:, 0] >= np.interp(points[:, 1], T_window_h, T_window_lo)) & (
            points[:, 0] <= np.interp(points[:, 1], T_window_h, T_window_hi)
        )
        return np.where(inside, Ta_lookup(points), np.nan).reshape(M.shape)


    # Largest thrust the table holds [N], used below to frame the performance diagrams
    Ta_max = float(T_table["Ta"].max()) * 1e3

    # The sweeps stop here: below this CL the condition sits past the Mach extent of TvsM
    M_ceiling = T_M[-1]
    return CD0_interp, K_interp, M_ceiling, Ta_interp, Ta_max


@app.cell
def _(h_slider, m_slider):
    W = m_slider.value * atmos.g0
    h = h_slider.value
    return W, h


@app.cell
def _(W, h, inverse_level_flight_speed, lift_coefficient_sweep):
    # Optimization domain: CL over the range the tables cover, throttle over its own
    n_mesh = plot_utils.meshgrid_n

    CL_array = lift_coefficient_sweep(W, h, n_mesh)
    dT_array = np.linspace(0, 1, n_mesh)

    # The objective does not depend on the throttle, so the surface is flat along dT
    inverse_V_CLarray = inverse_level_flight_speed(CL_array, W, h)
    inverse_V_surface = np.broadcast_to(
        inverse_V_CLarray[np.newaxis, :], (n_mesh, n_mesh)
    )
    return CL_array, dT_array, inverse_V_surface


@app.cell
def _(CD0_interp, CLmax, K_interp, M_ceiling, S, Ta_interp):
    # The flight condition a CL implies, as functions of (W, h) for reuse by the envelope
    def level_flight_speed(CL, W, h):
        """Speed in steady level flight at lift coefficient CL, from c1 solved for V."""
        return np.sqrt(2 * W / (atmos.rho(h) * S * np.atleast_1d(CL).astype(float)))

    def inverse_level_flight_speed(CL, W, h):
        """Reciprocal speed objective after eliminating the lift constraint."""
        CL = np.atleast_1d(CL).astype(float)
        return np.sqrt(atmos.rho(h) * S * CL / (2 * W))

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
        """CL sweep from the tables' Mach ceiling up to CLmax, evenly spaced in CL.

        The objective surface is smooth in CL, so an even mesh in CL is what draws it.
        """
        CL_min = 2 * W / (atmos.rho(h) * S * (M_ceiling * atmos.a(h)) ** 2)
        return np.linspace(CL_min, CLmax, n)

    def mach_uniform_sweep(W, h, n):
        """The same CL range, evenly spaced in Mach number instead.

        Since M ~ 1/sqrt(CL), an even CL grid spends almost all its samples at the
        slow end: at sea level the first two samples of a 400-point CL sweep straddle
        the entire transonic drag rise, across which the drag falls by a third and the
        equilibrium throttle by a quarter. That is exactly where the maximum speed
        sits, so the curves that have to resolve it are laid out evenly in Mach and
        mapped back to CL through c1.
        """
        M_stall = float(level_flight_speed(CLmax, W, h)[0]) / atmos.a(h)
        M = np.linspace(M_stall, M_ceiling, n)
        return np.sort(2 * W / (atmos.rho(h) * S * (M * atmos.a(h)) ** 2))
    return (
        equilibrium_throttle,
        inverse_level_flight_speed,
        level_flight_speed,
        lift_coefficient_sweep,
        mach_uniform_sweep,
        required_drag,
    )


@app.cell
def _(
    CLmax,
    Ta_interp,
    equilibrium_throttle,
    inverse_level_flight_speed,
    level_flight_speed,
    mach_uniform_sweep,
    required_drag,
):
    # scipy drives an equality residual to zero, so c2 is written as one
    def c2_eq(CL, dT, W, h):
        """Residual of the equality constraint c2"""
        M = level_flight_speed(CL, W, h) / atmos.a(h)
        return float(dT * Ta_interp(M, h)[0] - required_drag(CL, W, h)[0]) / W

    def solve_max_speed(W, h):
        """Maximum speed in steady level flight, and the limit that sets it."""
        # Solving c2 for the throttle traces the constraint curve, and starts the search
        CL_curve = mach_uniform_sweep(W, h, N_CURVE)
        dT_curve = equilibrium_throttle(CL_curve, W, h)

        # A NaN throttle marks a condition the chart never covered; >1 asks for too much
        covered = np.isfinite(dT_curve)
        on_surface = covered & (dT_curve <= 1)

        if not on_surface.any():
            # No point of the curve is both covered and flyable: no steady level flight here
            return None, np.nan, np.nan, np.nan

        # V rises as CL falls, so the search runs down to the smallest covered CL
        CL_floor = float(CL_curve[covered][0])

        result = minimize(
            lambda x: float(inverse_level_flight_speed(x[0], W, h)[0]),
            np.array([CL_curve[on_surface][0], dT_curve[on_surface][0]]),
            method="SLSQP",
            bounds=[(CL_floor, CLmax), (0.0, 1.0)],
            constraints={"type": "eq", "fun": lambda x: c2_eq(x[0], x[1], W, h)},
            options={"disp": False, "ftol": 1e-9},
        )

        CL_opt, dT_opt = float(result.x[0]), float(result.x[1])

        # Where both bind at once, the saturated throttle is the limit and is reported
        if dT_opt >= 1 - ACTIVE_TOL:
            limit = "thrust"
        elif CL_opt <= CL_floor + ACTIVE_TOL:
            limit = "data"
        else:
            limit = "thrust"

        return limit, CL_opt, dT_opt, float(level_flight_speed(CL_opt, W, h)[0])

    mo.show_code()
    return (solve_max_speed,)


@app.cell
def _(CL_opt, W, equilibrium_throttle, h, level_flight_speed, mach_uniform_sweep):
    # The solver's own sweep, with the solved optimum spliced in so the curve reaches it
    _CL_sweep = mach_uniform_sweep(W, h, N_CURVE)
    CL_fine = np.sort(np.append(_CL_sweep, CL_opt)) if np.isfinite(CL_opt) else _CL_sweep

    V_fine = level_flight_speed(CL_fine, W, h)
    dT_fine = equilibrium_throttle(CL_fine, W, h)

    # The spliced optimum sits on the boundary, so the test carries the same tolerance
    on_surface = dT_fine <= 1 + ACTIVE_TOL
    return CL_fine, V_fine, dT_fine, on_surface


@app.cell
def _(W, h, solve_max_speed):
    limit, CL_opt, dT_opt, V_max = solve_max_speed(W, h)
    return CL_opt, V_max, dT_opt, limit


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
    V_fine,
    V_max,
    ac_name,
    dT_array,
    dT_fine,
    dT_opt,
    inverse_V_surface,
    limit,
    on_surface,
):
    # Objective surface over (CL, dT), written out rather than through plot_utils

    # The CL sweep is floored on the tables' Mach ceiling, so the objective stays bounded
    _inverse_V_min = np.min(inverse_V_surface)
    _inverse_V_max = np.max(inverse_V_surface)

    fig_initial = go.Figure()

    fig_initial.add_trace(
        go.Surface(
            x=CL_array,
            y=dT_array,
            z=inverse_V_surface,
            name="Inverse velocity",
            opacity=0.9,
            colorscale="viridis",
            cmin=_inverse_V_min,
            cmax=_inverse_V_max,
            colorbar={"title": "1 / V (s/m)"},
        )
    )

    # The c2 constraint rides on the surface, blanked where it asks past full throttle
    fig_initial.add_trace(
        go.Scatter3d(
            x=CL_fine,
            y=np.where(on_surface, dT_fine, np.nan),
            z=np.where(on_surface, 1 / V_fine, np.nan),
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
                z=[1 / V_max],
                mode="markers",
                name="Maximum speed",
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
            zaxis=dict(title="1 / V (s/m)", range=[0, _inverse_V_max]),
        ),
        scene_camera=dict(eye=dict(x=1.35, y=1.35, z=1.35)),
        title={
            "text": f"Maximum speed for {ac_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        },
    )

    fig_initial
    return


@app.cell
def _(CL_opt, V_max, dT_opt, h, limit, setting_dropdown):
    if limit is None:
        print_output = mo.md(r"""
        Every point of the constraint curve asks for more than full throttle: at this
        combination of weight, altitude and thrust setting the aircraft has no steady
        level flight condition at all. Lower the altitude or the mass, or switch the
        thrust setting.
        """)
    else:
        _verdict = {
            "thrust": (
                "the throttle reaches $\\delta_T = 1$ before the thrust chart runs "
                "out of Mach number, so the maximum speed is **thrust-limited**"
            ),
            "data": (
                "the `TvsM` chart runs out of data before the throttle saturates, so "
                "the value below is only the fastest speed **the source data covers**"
            ),
        }[limit]

        print_output = mo.md(f"""
        At $h = {h:.0f}$ m on the {setting_dropdown.value} thrust setting,
        {_verdict}.

        | | |
        |---|---|
        | $V_\\mathrm{{max}}$ | {V_max:.1f} m/s |
        | $M$ | {V_max / atmos.a(h):.3f} |
        | $C_L^*$ | {CL_opt:.3f} |
        | $\\delta_T^*$ | {dT_opt:.3f} |

        None of the tables behind these numbers is ever extrapolated: $T_a$ is
        interpolated inside the Mach window the digitised `TvsM` chart covers at
        that altitude and is undefined outside it, so a flight condition the source
        chart never covered counts as unavailable rather than being invented.
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
    # Performance diagrams on the same CL sweep the solver uses, mapped onto speed by c1
    drag_curve = required_drag(CL_fine, W, h)
    thrust_curve = Ta_interp(V_fine / atmos.a(h), h)

    power_required = drag_curve * V_fine
    power_available = thrust_curve * V_fine

    V_E = float(V_fine[np.argmin(drag_curve)])
    V_P = float(V_fine[np.argmin(power_required)])
    V_stall = float(level_flight_speed(CLmax, W, h)[0])

    # Axis anchors taken at sea level and floored on the installed thrust
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
def _(CLmax, W, h_max, level_flight_speed, solve_max_speed):
    # The flight envelope is the same problem re-solved at every altitude the table covers
    h_envelope = np.linspace(0, h_max, 61)

    _solved = [solve_max_speed(W, _h) for _h in h_envelope]
    limits_envelope = [_s[0] for _s in _solved]
    V_envelope = np.array([_s[3] for _s in _solved])

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
    V_E,
    V_P,
    V_envelope,
    V_fine,
    V_max,
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
    inverse_V_surface,
    limit,
    on_surface,
    power_available,
    power_required,
    power_ylim,
    thrust_curve,
):
    # Drag, power, domain, envelope: x1/y1 top left, x2/y2 top right, then x3, x4 below

    fig_grid = make_subplots(
        rows=2, cols=2, horizontal_spacing=0.1, vertical_spacing=0.15
    )

    # With no optimum there is no throttle setting to draw but full throttle
    _dT_shown = dT_opt if np.isfinite(dT_opt) else 1.0

    # Top left: drag required against thrust available, with V_E, V_P and V_stall marked
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

    # Top right: the same story in power
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

    # Bottom left: the optimization domain, the surface above seen from overhead
    fig_grid.add_traces(
        [
            go.Heatmap(
                x=CL_array,
                y=dT_array,
                z=inverse_V_surface,
                xaxis="x3",
                yaxis="y3",
                zsmooth="best",
                opacity=0.9,
                colorscale="viridis",
                zmin=np.min(inverse_V_surface),
                zmax=np.max(inverse_V_surface),
                hovertemplate="C<sub>L</sub>=%{x:.3f}<br>δ<sub>T</sub>=%{y:.2f}<br>1 / V=%{z:.5f} s/m<extra></extra>",
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

    # Bottom right: the flight envelope, with the stall speed and M1 as references
    _i_label = int(0.8 * len(h_envelope))

    fig_grid.add_traces(
        [
            go.Scattergl(
                x=V_envelope,
                y=h_envelope / 1e3,
                xaxis="x4",
                yaxis="y4",
                mode="lines",
                name="V<sub>max</sub>",
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
        # The solved maximum, marked on all four panels; spliced in, so this is exact
        _i_max = np.argmin(np.abs(CL_fine - CL_opt))

        fig_grid.add_traces(
            [
                go.Scattergl(
                    x=[V_max],
                    y=[drag_curve[_i_max]],
                    xaxis="x1",
                    yaxis="y1",
                    mode="markers",
                    showlegend=False,
                    marker=dict(size=10, color=plot_utils.WHITE, symbol="circle"),
                ),
                go.Scattergl(
                    x=[V_max],
                    y=[power_required[_i_max] / 1e3],
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
                    x=[V_max],
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
def _(V_E, V_P, V_stall):
    # A minimum past the stall comes back pinned to the stall speed, not unreachable
    _at_stall = [
        _name
        for _name, _speed in (("minimum drag", V_E), ("minimum power", V_P))
        if np.isclose(_speed, V_stall)
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

    mo.md(f"""
    At the selected flight condition the minimum-drag speed is $V_E = {V_E:.1f}$ m/s and the
    minimum-power speed is $V_P = {V_P:.1f}$ m/s, against a stall speed of $V_s = {V_stall:.1f}$ m/s.
    {_reachable}
    """)
    return


if __name__ == "__main__":
    app.run()
