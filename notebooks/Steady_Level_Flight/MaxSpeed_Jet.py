import marimo

__generated_with = "0.17.6"
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
    from core import atmos
    from core import aircraft as ac
    from core import plot_utils
    from core.plot_utils import OptimumGridView

    # Set local/online filepath
    _defaults.FILEURL = _defaults.get_url()

    # Plotly dark mode template
    _defaults.set_plotly_template()

    # Data directory
    data_dir = str(mo.notebook_location().parent / "public" / "AircraftDB_Standard.csv")
    return OptimumGridView, ac, atmos, data_dir, go, mo, np, plot_utils


@app.cell
def _():
    # Set navbar on the right
    _defaults.set_sidebar()
    return


@app.cell
def _(ac, atmos, data_dir, mo, np, plot_utils):
    # Define constants, this cell runs once and is not dependent in any way on any interactive element (not even the ac database)
    dT_slider = mo.ui.slider(start=0, stop=1, step=0.1, label=r"$\delta_T$", value=0.5)

    meshgrid_n = 41
    xy_lowerbound = -0.1

    dT_array = np.linspace(0, 1, meshgrid_n)  # -
    h_array = np.linspace(0, 20e3, meshgrid_n)  # meters

    m_slider = mo.ui.slider(start=0, stop=1, step=0.1, label=r"", show_value=True)

    h_slider = mo.ui.slider(
        start=0,
        stop=20,
        step=0.5,
        label=r"Altitude (km)",
        value=10,
        show_value=True,
    )

    data = ac.available_aircrafts(data_dir, ac_type="Jet")[:8]

    labels = ["Power (kW)", -15]

    # Database cell
    ac_table = mo.ui.table(
        data=data,
        pagination=True,
        show_column_summaries=False,
        selection="single",
        initial_selection=[0],
        page_size=4,
        show_data_types=False,
    )
    a_0 = atmos.a(0)

    hover_name = "P<sub>min</sub>"

    mass_stack = mo.hstack(
        [mo.md("**OEW**"), m_slider, mo.md("**MTOW**")],
        align="start",
        justify="start",
    )
    variables_stack = mo.hstack([mass_stack, h_slider])

    rho_array = atmos.rho(h_array)
    sigma_array = atmos.rhoratio(h_array)
    min_sigma = atmos.rhoratio(atmos.hmax)
    a_harray = atmos.a(h_array)

    # Visual computations
    mach_trace = plot_utils.create_mach_trace(h_array, a_harray)
    return (
        a_0,
        a_harray,
        ac_table,
        dT_array,
        dT_slider,
        data,
        h_array,
        h_slider,
        m_slider,
        mach_trace,
        mass_stack,
        meshgrid_n,
        min_sigma,
        rho_array,
        sigma_array,
        variables_stack,
        xy_lowerbound,
    )


@app.cell
def _(a_0, ac_table, dT_array, data, meshgrid_n, mo, np, xy_lowerbound):
    # Define constants dependent on the ac database. This runs every time another aircraft is selected

    if ac_table.value is not None and ac_table.value.any().any():
        active_selection = ac_table.value.iloc[0]
    else:
        active_selection = data.iloc[0]

    # avoid having zeros for velocity computation
    CL_array = np.linspace(0, active_selection["CLmax_ld"], meshgrid_n + 1)[1:]

    # Extract essential values
    CD0 = active_selection["CD0"]
    S = active_selection["S"]
    K = active_selection["K"]
    CLmax = active_selection["CLmax_ld"]
    Ta0 = active_selection["Ta0"] * 1e3  # Watts
    beta = active_selection["beta"]
    OEM = active_selection["OEM"]
    MTOM = active_selection["MTOM"]

    # Compute design values
    CL_P = np.sqrt(3 * CD0 / K)
    CL_E = np.sqrt(CD0 / K)
    E_max = CL_E / (CD0 + K * CL_E**2)
    E_S = CLmax / (CD0 + K * CLmax**2)
    E_P = (np.sqrt(3) / 2) * E_max
    E_array = CL_array / (CD0 + K * CL_array**2)

    CL_slider = mo.ui.slider(
        start=0,
        stop=CLmax,
        step=0.2,
        label=r"$C_L$",
        value=0.5,
    )

    ranges = [
        xy_lowerbound,
        CLmax + 0.05,
        xy_lowerbound,
        1 + 0.05,
        xy_lowerbound,
        a_0,
        xy_lowerbound,
        20,
    ]

    axes = (CL_array, dT_array)
    return (
        CD0,
        CL_E,
        CL_P,
        CL_array,
        CL_slider,
        CLmax,
        E_S,
        E_array,
        E_max,
        K,
        MTOM,
        OEM,
        S,
        Ta0,
        active_selection,
        beta,
    )


@app.cell
def _(CL_array, CL_slider):
    # Define variables, this cell runs every time the CL slider is run
    step_CL = CL_array[2] - CL_array[1]
    CL_selected = float(CL_slider.value)
    idx_CL_selected = int((CL_selected - CL_array[0]) / step_CL)
    return (idx_CL_selected,)


@app.cell
def _(
    CD0,
    CLmax,
    E_array,
    K,
    MTOM,
    OEM,
    S,
    a_0,
    atmos,
    h_array,
    m_slider,
    np,
    plot_utils,
    rho_array,
):
    # Define variables, this cell runs every time the mass slider is run
    W_selected = (OEM + (MTOM - OEM) * m_slider.value) * atmos.g0  # Netwons
    drag_curve = W_selected / E_array

    velocity_stall_harray = np.sqrt(2 * W_selected / (rho_array * S * CLmax))

    # Visual computations
    stall_trace = plot_utils.create_stall_trace(h_array, velocity_stall_harray)

    CL_a0 = W_selected * 2 / (atmos.rho0 * S * a_0**2)

    drag_yrange = 0.2 * W_selected * (CD0 + K * CL_a0**2) / CL_a0
    power_yrange = 0.5 * drag_yrange * a_0 / 1e3
    return (
        W_selected,
        drag_curve,
        drag_yrange,
        power_yrange,
        stall_trace,
        velocity_stall_harray,
    )


@app.cell
def _(Ta0, atmos, beta, h_array, h_slider, meshgrid_n, np):
    # Define variables, this cell runs every time the altitude slider is run
    h_selected = int(h_slider.value * 1e3)  # meters
    step_h = h_array[1] - h_array[0]
    idx_h_selected = int((h_selected - h_array[0]) / step_h)

    a_selected = atmos.a(h_selected)

    sigma_selected = atmos.rhoratio(h_selected)

    rho_selected = atmos.rho(h_selected)

    thrust_scalar = Ta0 * sigma_selected**beta

    thrust_vector = np.repeat(thrust_scalar, meshgrid_n)
    return (
        h_selected,
        idx_h_selected,
        rho_selected,
        sigma_selected,
        thrust_scalar,
        thrust_vector,
    )


@app.cell
def _(
    CL_E,
    CL_P,
    CL_array,
    CLmax,
    S,
    Ta0,
    W_selected,
    beta,
    dT_array,
    drag_curve,
    drag_yrange,
    idx_h_selected,
    mach_trace,
    np,
    plot_utils,
    power_yrange,
    rho_selected,
    sigma_selected,
    stall_trace,
    thrust_scalar,
    thrust_vector,
    velocity_stall_harray,
):
    # Computation only cell, indexing happens in another cell
    velocity_CLarray = np.sqrt(2 * W_selected / (rho_selected * S * CL_array))
    velocity_CL_E = velocity_CLarray[-1] * np.sqrt(CLmax / CL_E)
    velocity_CL_P = velocity_CLarray[-1] * np.sqrt(CLmax / CL_P)

    power_available = thrust_scalar * velocity_CLarray / 1e3
    power_required = drag_curve * velocity_CLarray / 1e3

    constraint = drag_curve / Ta0 / (sigma_selected**beta)

    range_performance_diagrams = (drag_yrange, power_yrange, CLmax, 400)

    inv_velocity_surface = np.broadcast_to(
        (1 / velocity_CLarray)[np.newaxis, :],  # Shape: (101, 1)
        (len(CL_array), len(dT_array)),  # Target shape: (101, 101)
    )

    plot_utils.axes_max_speed = 500
    min_colorbar = np.min(inv_velocity_surface)
    max_colorbar = np.max(inv_velocity_surface)
    zcolorbar = (min_colorbar, max_colorbar)

    # Create graphic traces
    configTraces = plot_utils.ConfigTraces(
        CL_array,
        dT_array,
        constraint,
        drag_curve,
        thrust_vector,
        power_required,
        power_available,
        inv_velocity_surface,
        velocity_CLarray,
        velocity_CL_P,
        velocity_CL_E,
        velocity_stall_harray,
        velocity_stall_harray[idx_h_selected],
        range_performance_diagrams,
        zcolorbar,
        mach_trace,
        stall_trace,
    )
    return (
        configTraces,
        constraint,
        inv_velocity_surface,
        max_colorbar,
        min_colorbar,
        range_performance_diagrams,
        velocity_CL_E,
        velocity_CL_P,
        velocity_CLarray,
    )


@app.cell
def _(idx_CL_selected, inv_velocity_surface):
    inv_velocity_selected = inv_velocity_surface[1, idx_CL_selected]
    return (inv_velocity_selected,)


@app.cell
def _(mo):
    mo.md(r"""
    # Maximum airspeed: simplified jet aircraft

    $$
    \begin{aligned}
        \max_{C_L, \delta_T}
        & \quad V \\
        \text{subject to}
        & \quad c_1^\mathrm{eq} = L-W = \frac{1}{2}\rho V^2 S C_L - W = 0 \\
        & \quad c_2^\mathrm{eq} = T-D = \delta_T T_a(V,h) - \frac{1}{2} \rho V^2 S (C_{D_0}+K C_L^2) =0 \\
        \text{for }
        & \quad C_L \in [0, C_{L_\mathrm{max}}] \\
        & \quad \delta_T \in [0, 1] \\
        \text{with }
        & \quad T_a(V,h) = T_a(h) = T_{a0}\sigma^\beta \\
    \end{aligned}
    $$
    """)
    return


@app.cell
def _(
    CL_array,
    CL_slider,
    active_selection,
    constraint,
    dT_array,
    dT_slider,
    go,
    inv_velocity_selected,
    inv_velocity_surface,
    max_colorbar,
    min_colorbar,
    mo,
    xy_lowerbound,
):
    # Initial Figure
    fig_initial = go.Figure()

    # Minimum velocity surface
    fig_initial.add_traces(
        [
            go.Surface(
                x=CL_array,
                y=dT_array,
                z=inv_velocity_surface,
                opacity=0.9,
                name="1/Velocity",
                colorscale="viridis",
                cmin=min_colorbar,
                cmax=max_colorbar,
                colorbar={"title": "V<sup>-1</sup> (s/m)"},
            ),
            go.Scatter3d(
                x=CL_array,
                y=constraint,
                z=inv_velocity_surface[0],
                opacity=0.7,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[CL_array[30]],
                y=[constraint[30]],
                z=[inv_velocity_surface[0, 30]],
                opacity=1,
                textposition="middle left",
                mode="markers+text",
                text=["g<sub>1</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="g1 constraint",
                textfont=dict(size=14, family="Arial"),
            ),
            go.Scatter3d(
                x=[CL_slider.value],
                y=[dT_slider.value],
                z=[inv_velocity_selected],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=3,
                    color="white",
                    symbol="circle",
                ),
                name="Design Point",
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub> : %{y}<br>1/V: %{z}<extra>%{fullData.name}</extra>",
            ),
        ]
    )
    # Set the camera to show the end of both axes
    camera = dict(eye=dict(x=-1.35, y=-1.35, z=1.35))

    fig_initial.update_layout(
        scene=dict(
            xaxis=dict(
                title="C<sub>L</sub> (-)",
                range=[xy_lowerbound, active_selection["CLmax_ld"]],
            ),
            yaxis=dict(title="δ<sub>T</sub> (-)", range=[xy_lowerbound, 1]),
            zaxis=dict(
                title="V<sup>-1</sup> (s/m)", range=[min_colorbar, max_colorbar]
            ),
        ),
    )

    fig_initial.update_layout(
        scene_camera=camera,
        title={
            "text": f"Maximum airspeed domain for {active_selection.full_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        },
    )

    mo.output.clear()
    return (fig_initial,)


@app.cell
def _(mo):
    mo.md(r"""
    ## KKT formulation
    To be reconducted in the standard KKT analysis format, the objective function is expressed in terms of the controls by direct elimination of $c_1^\mathrm{eq}$.
    Also, maximizing $V$ is equivalent to minimizing its inverse, $1/V$.
    Therefore, to simplify the calculations, the problem is rewritten as follows:
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    $$
    \begin{aligned}
        \min_{C_L, \delta_T}
        & \quad \frac{1}{V} = \sqrt{\frac{\rho S C_L}{2W}} \\
        \text{subject to}
        & \quad g_1 = \delta_T T_{a0}\sigma^\beta - W \left(\frac{C_{D_0} + KC_L^2}{C_L}\right) = 0 \\
        & \quad h_1 = C_L - C_{L_\mathrm{max}} \le 0 \\
        & \quad h_2 = \delta_T - 1 \le 0 \\
    \end{aligned}
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    In the interactive graph below, select a simplified jet aircraft of your choice and experiment in finding an optimum by changing the control variables, $C_L$ and $\delta_T$. The design point is marked in white in the 3D velocity surface.
    """)
    return


@app.cell
def _(ac_table):
    # Database cell (1)
    ac_table
    return


@app.cell
def _(CL_slider, dT_slider, mo):
    mo.md(f"""
    Here you can modify the control variables to understand how it affects the design: {mo.hstack([dT_slider, CL_slider])}
    """)
    return


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell
def _(fig_initial):
    fig_initial
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Lagrangian function and KKT conditions

    The Lagrangian function combines the objective function with equality constraints using Lagrange multipliers ($\lambda_i$) and inequality constraints using KKT multipliers ($\mu_j$).

    $$
    \begin{aligned}
    \mathcal{L}(C_L, \delta_T, \lambda_1, \mu_1, \mu_2) =
    \quad \sqrt{\frac{\rho S C_L}{2W}}
    & + \\
    & + \lambda_1 \left[\delta_T T_{a0}\sigma^\beta - W \left(\frac{C_{D_0} + KC_L^2}{C_L}\right)\right] + \\
    & + \mu_1 (C_L - C_{L_\mathrm{max}}) + \\
    & + \mu_2 (\delta_T - 1)\\
    \end{aligned}
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    A necessary condition for an optimal solution of the optimization problem $(C_L^*, \delta_T^*)$ to exist, the multipliers $\lambda_1, \mu_1, \mu_2$ have to meet the following conditions:

    **A. Stationarity ($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \frac{\partial \mathcal{L}}{\partial C_L} = \frac{1}{2}\sqrt{\rho_0\frac{S}{2}\frac{\sigma}{W}}C_L^{-1/2} - \lambda_1 W\left(\frac{KC_L^2 - C_{D_0}}{C_L^2}\right) + \mu_1 = 0$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1 T_{a0}\sigma^\beta + \mu_2 = 0$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $\displaystyle \delta_T T_{a0}\sigma^\beta - W \left(\frac{C_{D_0} + KC_L^2}{C_L}\right) = 0$
    4.  $C_L - C_{L_\mathrm{max}} \le 0$
    5.  $\delta_T - 1 \le 0$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    6.  $\mu_1, \mu_2\ge 0$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraint have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

    7.  $\mu_1 (C_L - C_{L_\mathrm{max}}) = 0$
    8. $\mu_2 (\delta_T - 1) = 0$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## KKT analysis

    We can now proceed to systematically examine the conditions where various inequality constraints are active or inactive.
    """)
    return


@app.cell
def _(mo):
    titles_dict = {
        "### Interior solutions": "",
        "### Lift limited solutions": "",
        "### Thrust limited solutions": "",
        "### Lift-thrust limited solutions": "",
    }

    tab = mo.ui.tabs(titles_dict)
    tab.style({"height": "60px", "overflow": "auto"}).callout(kind="info").center()
    return tab, titles_dict


@app.cell
def _(tab, titles_dict):
    title_keys = list(titles_dict.keys())
    tab_value = tab.value
    return tab_value, title_keys


@app.cell
def _(
    CL_array,
    CLmax,
    CLopt_maxlift,
    CLopt_maxthrust_selected,
    E_array,
    OptimumGridView,
    W_selected,
    active_selection,
    configTraces,
    dT_array,
    dTopt_maxlift,
    dTopt_maxthrust,
    drag_curve,
    drag_yrange,
    h_maxliftThrust,
    h_maxlift_array,
    h_maxthrust_array,
    h_selected,
    mach_trace,
    maxliftThrust_multiplier,
    np,
    plot_utils,
    power_available_maxliftThrust_array,
    power_maxliftThrust_selected,
    power_maxlift_selected,
    power_maxthrust_selected,
    power_yrange,
    range_performance_diagrams,
    stall_trace,
    tab_value,
    thrust_maxliftThrust_vector,
    title_keys,
    true_maxlift,
    true_maxliftThrust,
    true_maxthrust,
    velocity_CL_E,
    velocity_CL_P,
    velocity_CLarray_maxliftThrust,
    velocity_maxliftThrust_selected,
    velocity_maxlift_harray,
    velocity_maxlift_selected,
    velocity_maxthrust_harray,
    velocity_maxthrust_selected,
):
    if tab_value == title_keys[1]:
        figure_optimum = OptimumGridView(
            configTraces,
            h_selected,
            (velocity_maxlift_harray, velocity_maxlift_selected),
            (np.nan, power_maxlift_selected),
            (h_maxlift_array, dTopt_maxlift, CLopt_maxlift, true_maxlift),
            f"Lift-limited minimum power for {active_selection.full_name}",
        )

    elif tab_value == title_keys[2]:
        figure_optimum = OptimumGridView(
            configTraces,
            h_selected,
            (velocity_maxthrust_harray, velocity_maxthrust_selected),
            (np.nan, power_maxthrust_selected),
            (
                h_maxthrust_array,
                dTopt_maxthrust,
                CLopt_maxthrust_selected,
                true_maxthrust,
            ),
            f"Thrust-limited minimum power for {active_selection.full_name}",
        )

    elif tab_value == title_keys[3]:
        inv_velocity_surface_maxliftThrust = np.broadcast_to(
            1 / velocity_CLarray_maxliftThrust[np.newaxis, :],  # Shape: (101, 1)
            (len(CL_array), len(dT_array)),  # Target shape: (101, 101)
        )

        min_colorbar_maxliftThrust = np.min(inv_velocity_surface_maxliftThrust)
        max_colorbar_maxliftThrust = np.max(inv_velocity_surface_maxliftThrust)
        zcolorbar_maxliftThrust = (
            min_colorbar_maxliftThrust,
            max_colorbar_maxliftThrust,
        )

        constraint_maxliftThrust = W_selected / E_array / thrust_maxliftThrust_vector

        configTraces_maxliftThrust = plot_utils.ConfigTraces(
            CL_array,
            dT_array,
            constraint_maxliftThrust,
            drag_curve,
            thrust_maxliftThrust_vector,
            power_available_maxliftThrust_array / 1e3,
            thrust_maxliftThrust_vector * velocity_CLarray_maxliftThrust / 1e3,
            inv_velocity_surface_maxliftThrust,
            velocity_CLarray_maxliftThrust,
            velocity_CL_P * maxliftThrust_multiplier,
            velocity_CL_E * maxliftThrust_multiplier,
            velocity_maxliftThrust_selected,
            velocity_maxliftThrust_selected,
            (drag_yrange, power_yrange / 1e3, CLmax),
            zcolorbar_maxliftThrust,
            mach_trace,
            stall_trace,
        )

        # maxliftThrust graphics
        figure_optimum = OptimumGridView(
            configTraces_maxliftThrust,
            h_maxliftThrust,
            (velocity_CLarray_maxliftThrust, velocity_maxliftThrust_selected),
            (np.nan, power_maxliftThrust_selected),
            (h_maxliftThrust, 1 * true_maxliftThrust, CLmax, true_maxliftThrust),
            f"Thrust-lift limited minimum drag for {active_selection.full_name}",
            equality=True,
        )

    if tab_value != title_keys[0]:
        figure_optimum.update_axes_ranges(range_performance_diagrams)
    return (figure_optimum,)


@app.cell
def _(mo, tab_value, title_keys):
    if tab_value != title_keys[0]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ### _Interior solutions_

    Assuming that that $C_L < C_{L_\mathrm{max}}$ and $\delta_T < 1$ is equivalent to consider all inequality constraints as inactive.

    Therefore: $\mu_1,\mu_2=0$.

    From stationarity condition (2): $\lambda_1 = 0$.

    It can now be seen that stationarity condition (1) is never verified.

    It can be concluded that the maximum speed cannot be achieved in the interior of the domain. The maximum speed must lie on at least one of the boundaries defined by $C_L = C_{L_\mathrm{max}}$ or $\delta_T = 1$.
    """)
        ]
    ).callout()
    return


@app.cell
def _(figure_optimum, mo, tab_value, title_keys, variables_stack):
    if tab_value != title_keys[2]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ## _Thrust-limited maximum airspeed_

    $\delta_T=1 \quad \Rightarrow \quad \mu_2 > 0$

    $C_L < C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 = 0$

    From stationarity condition (2):

    $$
    \lambda_1 = -\frac{\mu_2}{T_{a0}\sigma^\beta} \lt 0
    $$

    Stationarity condition (1) then becomes:

    $$
    \begin{align}
    \mu_1 &= -\frac{1}{2}\sqrt{\rho_0\frac{S}{2}\frac{\sigma}{W}}C_L^{-1/2} + \lambda_1 W\left(\frac{KC_L^2 - C_{D_0}}{C_L^2}\right) = 0 \nonumber \\
    \Leftrightarrow \quad \lambda_1 &= \frac{\frac{1}{2}\sqrt{\rho_0\frac{S}{2}\frac{\sigma}{W}}C_L^{-1/2}}{W\left(\frac{KC_L^2 - C_{D_0}}{C_L^2}\right)} \lt 0 \quad \Leftrightarrow \frac{1}{KC_L^2 - C_{D_0}} \lt 0 \quad \Leftrightarrow \quad C_L \lt \sqrt{\frac{C_{D_0}}{K}} = C_{L_E} \nonumber
    \end{align}
    $$

    This shows that maximum speed is obtained, intuitively, on the positive (right-hand side) branch of the performance diagram.

    Note: both $C_L \lt C_{L_E}$ and $C_L\lt C_{L_\mathrm{max}}$ can be true in either case of $C_{L_\mathrm{max}} \geq C_{L_E}$. The loosest, and best-case design is of course when $C_{L_E} \lt C_{L_\mathrm{max}}$, meaning that the aircraft is able to fly on the induced (left-hand side) branch of the performance diagram.

    The corresponding optimum value of the $C_L$ is obtained by solving the primal feasibiliy condition (3), resulting in the well known:

    $$
    C_{L_{1, 2}}^* = \frac{T_{a0}\sigma^\beta}{2KW} \left[1\pm\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}\right]
    $$

    Which exists for:

    $$
    1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2 \ge 0
    \quad \Leftrightarrow \quad \frac{W}{\sigma^\beta} \le  T_{a0} E_\mathrm{max}
    $$

    Where we are interested in the lower value, with the - sign. In the case where $C_{L_E} \lt C_{L_\mathrm{max}}$, this value is feasible when:

    $$
    C_L^* \lt C_{L_E} \quad \Leftrightarrow \quad \frac{W}{\sigma^\beta} \lt  T_{a0} E_\mathrm{max}
    $$

    Meaning that the minimum drag at current altitude and weight is less then the available thrust.

    Thus the optimal values are:

    $$
    \delta_T^* = 1, \quad C_L^* = \frac{T_{a0}\sigma^\beta}{2KW} \left[1-\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}\right], \quad \text{for} \:\:\frac{W}{\sigma^\beta} \lt  T_{a0} E_\mathrm{max}, \quad\text{if}\:\: C_{L_\mathrm{max}} \gt \sqrt{\frac{C_{D_0}}{K}}
    $$
    """),
            variables_stack,
            figure_optimum.figure,
        ]
    ).callout()
    return


@app.cell
def _(atmos, np):
    def maxthrust_condition(
        W, h_selected, K, E_max, E_S, h_array, Ta0, beta, sigma_array
    ):
        B = W / E_max / Ta0
        min_sigma = (B) ** (1 / beta)

        if min_sigma >= 1:
            return np.array([np.nan]), 1, np.nan, np.nan, False

        dT_optimum = 1
        max_h = atmos.altitude(min_sigma)

        hopt_array = h_array[h_array < max_h]

        sigma_optimum = sigma_array[np.isin(h_array, hopt_array)]

        A = Ta0 * sigma_optimum**beta / (2 * K * W)
        B = (B / (sigma_optimum**beta)) ** 2

        CL_optimum = A * (
            1 - np.sqrt(1 - B, where=(B < 1), out=np.full_like(B, np.nan))
        )

        mask_CL = ~np.isnan(CL_optimum)

        hopt_array = hopt_array[mask_CL]
        CL_optimum = CL_optimum[mask_CL]
        CL_selected = CL_optimum[np.isclose(hopt_array, h_selected)]

        CL_selected = CL_selected.item() if CL_selected.size == 1 else np.nan

        cond = 1 if h_selected <= max_h else np.nan
        return (
            hopt_array,
            dT_optimum,
            CL_optimum,
            CL_selected,
            cond,
        )

    return (maxthrust_condition,)


@app.cell
def _(
    CD0,
    E_S,
    E_max,
    K,
    S,
    Ta0,
    W_selected,
    atmos,
    beta,
    h_array,
    h_selected,
    maxthrust_condition,
    np,
    rho_selected,
    sigma_array,
):
    # Maxthrust computations
    (
        h_maxthrust_array,
        dTopt_maxthrust,
        CLopt_maxthrust,
        CLopt_maxthrust_selected,
        true_maxthrust,
    ) = maxthrust_condition(
        W_selected, h_selected, K, E_max, E_S, h_array, Ta0, beta, sigma_array
    )

    velocity_maxthrust_harray = np.sqrt(
        2 * W_selected / (atmos.rho(h_maxthrust_array) * S * CLopt_maxthrust)
    )

    velocity_maxthrust_selected = (
        np.sqrt(2 * W_selected / (rho_selected * S * CLopt_maxthrust_selected))
        * true_maxthrust
    )

    power_maxthrust_selected = (
        W_selected
        * (CD0 + K * CLopt_maxthrust_selected**2)
        / CLopt_maxthrust_selected
        * velocity_maxthrust_selected
    )
    return (
        CLopt_maxthrust_selected,
        dTopt_maxthrust,
        h_maxthrust_array,
        power_maxthrust_selected,
        true_maxthrust,
        velocity_maxthrust_harray,
        velocity_maxthrust_selected,
    )


@app.cell
def _(mo, tab_value, title_keys):
    if tab_value != title_keys[1]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ###_Lift-limited minimum airspeed_

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 > 0$

    $0 < \delta_T < 1 \quad \Rightarrow \quad \mu_2 = 0$.

    From stationarity condition (2): $\lambda_1 = 0$.

    From stationarity condition (1):

    $$
    \mu_1 = -\frac{1}{2}\sqrt{\frac{\rho_0 S \sigma }{2WC_{L_\mathrm{max}}}}>0, \quad \mathrm{for} \quad C_{L_\mathrm{max}} \lt 0, \mathrm{impossible}
    $$

    The solution cannot be obtained at $C_{L_\mathrm{max}}$, which is intuitive. As a matter of fact:

    $$
    \min_{C_L} \frac{1}{V} = \sqrt{\frac{\rho S C_L}{2W}} \quad \Leftrightarrow \quad \min_{C_L} \sqrt{C_L} \quad \Leftrightarrow \quad \min_{C_L} C_L
    $$
    """)
        ]
    ).callout()
    return


@app.cell
def _(figure_optimum, mo, tab_value, title_keys, variables_stack):
    if tab_value != title_keys[3]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ### _Thrust- and lift-limited minimum speed_

    $\delta_T = 1 \quad \Rightarrow \quad \mu_2 > 0$

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 > 0$.

    From the stationary conditions (2):

    $$
    \lambda_1 = -\frac{\mu_2}{T_{a0}\sigma^\beta} < 0
    $$

    From stationary condition (1):

    $$
    \mu_1 = \lambda_1 W\left(\frac{KC_{L_\mathrm{max}}^2 - C_{D_0}}{C_{L_\mathrm{max}}^2}\right) -\frac{1}{2}\sqrt{\rho_0\frac{S}{2}\frac{\sigma}{W}\frac{1}{C_{L_\mathrm{max}}}} \gt 0
    $$

    $$
    \frac{\frac{1}{2}\sqrt{\rho_0\frac{S}{2}\frac{\sigma}{W}\frac{1}{C_{L_\mathrm{max}}}}}{W\left(\frac{KC_{L_\mathrm{max}}^2 - C_{D_0}}{C_{L_\mathrm{max}}^2}\right)} \lt \lambda_1 \lt 0 \quad \Leftarrow \quad C_{L_\mathrm{max}} \lt \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$


    In other words, this condition is verified only if the aircraft would not be able to fly in the condition of maximum aerodynamic efficiency (or minimum drag in steady level flight) because it woudl stall at a higher speed.

    From (3), the same derivation as the previous case results in

    $$
    C_L^* = C_{L_\mathrm{max}}, \quad \delta_T^*=1, \quad \frac{W}{\sigma^\beta} = T_{a0}E_S, \quad \mathrm{if} \quad C_{L_\mathrm{max}} \lt \sqrt{\frac{C_{D_0}}{K}}
    $$
    """),
            variables_stack,
            figure_optimum.figure,
        ]
    ).callout()
    return


@app.cell
def _(atmos, np):
    def maxliftThrust_condition(W, Ta0, CL_E, E_S, beta, min_sigma, CLmax):
        sigma_maxliftThrust = (W / Ta0 / E_S) ** (1 / beta)
        h_maxliftThrust_selected = atmos.altitude(sigma_maxliftThrust)

        if sigma_maxliftThrust >= min_sigma or CLmax > CL_E:
            return h_maxliftThrust_selected, sigma_maxliftThrust, np.nan, 1, np.nan

        condition = True

        return (
            h_maxliftThrust_selected,
            sigma_maxliftThrust,
            CLmax,
            1,
            condition,
        )

    return (maxliftThrust_condition,)


@app.cell
def _(
    CL_E,
    CLmax,
    E_S,
    Ta0,
    W_selected,
    atmos,
    beta,
    drag_curve,
    maxliftThrust_condition,
    meshgrid_n,
    min_sigma,
    np,
    rho_selected,
    velocity_CLarray,
):
    (
        h_maxliftThrust,
        sigma_maxliftThrust,
        CLopt_maxliftThrust,
        dTopt_maxliftThrust,
        true_maxliftThrust,
    ) = maxliftThrust_condition(W_selected, Ta0, CL_E, E_S, beta, min_sigma, CLmax)

    maxliftThrust_multiplier = np.sqrt(rho_selected / atmos.rho(h_maxliftThrust))
    velocity_maxliftThrust_selected = (
        velocity_CLarray[-1] * maxliftThrust_multiplier * true_maxliftThrust
    )
    velocity_CLarray_maxliftThrust = velocity_CLarray * maxliftThrust_multiplier

    thrust_maxliftThrust_vector = np.repeat(Ta0 * sigma_maxliftThrust**beta, meshgrid_n)
    power_available_maxliftThrust_array = drag_curve * velocity_CLarray_maxliftThrust
    power_maxliftThrust_selected = W_selected / E_S * velocity_maxliftThrust_selected
    return (
        h_maxliftThrust,
        maxliftThrust_multiplier,
        power_available_maxliftThrust_array,
        power_maxliftThrust_selected,
        thrust_maxliftThrust_vector,
        true_maxliftThrust,
        velocity_CLarray_maxliftThrust,
        velocity_maxliftThrust_selected,
    )


@app.cell
def _(mo):
    mo.md(r"""
    Now after deriving all the optima for each condition we can summarize the flight envelopes in one graph, as shown below. Experiment with the weight of the aircrarft to understand how the theoretical ceiling for maximum speed moves in the graph.
    """)
    return


@app.cell
def _(
    a_harray,
    h_array,
    h_maxliftThrust,
    h_maxthrust_array,
    mass_stack,
    mo,
    np,
    plot_utils,
    velocity_maxliftThrust_selected,
    velocity_maxthrust_harray,
    velocity_stall_harray,
):
    flight_envelope = plot_utils.create_final_flightenvelope(
        velocity_stall_harray,
        a_harray,
        h_array,
        (np.nan, np.nan, False),
        (h_maxthrust_array, velocity_maxthrust_harray, True),
        (np.nan, np.nan, False),
        (h_maxliftThrust, velocity_maxliftThrust_selected, False),
    )

    mo.vstack([mass_stack, flight_envelope])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Name | Condition | $C_L^*$ | $\delta_T^*$ | $V^*$ |
    |:-|:----------|:-------:|:------------:|:------|
    |Thrust and Lift-limited    | $\displaystyle \frac{W}{\sigma^\beta} =  T_{a0} E_S$ | $C_{L_\mathrm{max}}$ | $1$ | $\displaystyle V_s =\sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}$ |
    |Thrust-limited    | $\displaystyle \frac{W}{\sigma^\beta} \lt  T_{a0} E_\mathrm{max}$ | $\displaystyle \frac{T_{a0}\sigma^\beta}{2KW} \left[1-\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}\right]$ | $1$ | $\displaystyle V_s \sqrt{\frac{2KWC_{L_\mathrm{max}}/T_{a0}\sigma^\beta}{1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}}}$ |
    """)
    return


@app.cell
def _():
    _defaults.nav_footer(
        after_file="MaxSpeed_Prop.py",
        after_title="Maximum Speed Simplified Propeller",
        above_file="MaxSpeed.py",
        above_title="Maximum Speed Homepage",
        above_before=True,
    )
    return


if __name__ == "__main__":
    app.run()
