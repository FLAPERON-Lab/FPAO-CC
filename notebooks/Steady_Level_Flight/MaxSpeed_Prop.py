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
    from scipy.optimize import root_scalar
    from core import atmos
    from core import aircraft as ac
    from core import plot_utils
    from core.plot_utils import OptimumGridView

    # Set local/online filepath
    _defaults.FILEURL = _defaults.get_url()

    # Plotly dark mode template
    _defaults.set_plotly_template()

    # Data directory
    data_dir = str(
        mo.notebook_location().parent.parent / "data" / "AircraftDB_Standard.csv"
    )


@app.cell
def _():
    # Set navbar on the right
    _defaults.set_sidebar()
    return


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
        & \quad T_a(V,h) = \frac{P_a(h)}{V} = \frac{P_{a0}\sigma^\beta}{V} \\
    \end{aligned}
    $$
    """)
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

    data = ac.available_aircrafts(data_dir, ac_type="Propeller")[:8]

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
        rho_array,
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
    Pa0 = active_selection["Pa0"] * 1e3  # Watts
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
        K,
        MTOM,
        OEM,
        Pa0,
        S,
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
def _(Pa0, atmos, beta, h_array, h_slider, meshgrid_n, np):
    # Define variables, this cell runs every time the altitude slider is run
    h_selected = int(h_slider.value * 1e3)  # meters
    step_h = h_array[1] - h_array[0]
    idx_h_selected = int((h_selected - h_array[0]) / step_h)

    a_selected = atmos.a(h_selected)

    sigma_selected = atmos.rhoratio(h_selected)

    rho_selected = atmos.rho(h_selected)

    power_scalar = Pa0 * sigma_selected**beta

    power_available = np.repeat(power_scalar, meshgrid_n)
    return (
        h_selected,
        idx_h_selected,
        power_available,
        power_scalar,
        rho_selected,
    )


@app.cell
def _(
    CL_E,
    CL_P,
    CL_array,
    CLmax,
    S,
    W_selected,
    dT_array,
    drag_curve,
    drag_yrange,
    idx_h_selected,
    mach_trace,
    np,
    plot_utils,
    power_available,
    power_scalar,
    power_yrange,
    rho_selected,
    stall_trace,
    velocity_stall_harray,
):
    # Computation only cell, indexing happens in another cell
    velocity_CLarray = np.sqrt(2 * W_selected / (rho_selected * S * CL_array))
    velocity_CL_E = velocity_CLarray[-1] * np.sqrt(CLmax / CL_E)
    velocity_CL_P = velocity_CLarray[-1] * np.sqrt(CLmax / CL_P)

    thrust_available = power_scalar / velocity_CLarray
    power_required = drag_curve * velocity_CLarray / 1e3

    constraint = drag_curve / thrust_available

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
        thrust_available,
        power_required,
        power_available / 1e3,
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
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[CL_array[35]],
                y=[constraint[35]],
                z=[inv_velocity_surface[0, 35]],
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

    fig_initial.update_layout(
        scene=dict(
            xaxis=dict(
                title="C<sub>L</sub> (-)",
                range=[xy_lowerbound, active_selection["CLmax_ld"]],
            ),
            yaxis=dict(title="δ<sub>T</sub> (-)", range=[xy_lowerbound, 1]),
            zaxis=dict(title="V<sup> -1</sup> (s/m)"),
        ),
    )

    # Set the camera to show the end of both axes
    camera = dict(eye=dict(x=-1.35, y=-1.35, z=1.35))

    # Add title at the end, before mo.output.clear()
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
        & \quad g_1 = \delta_T P_{a0}\sigma^\beta - \frac{W^{3/2}}{\sigma^{1/2}}\sqrt{\frac{2}{\rho_0S}} \left(C_{D_0}C_L^{-3/2} + KC_L^{1/2}\right) = 0 \\
        & \quad h_1 = C_L - C_{L_\mathrm{max}} \le 0 \\
        & \quad h_2 = \delta_T - 1 \le 0 \\
    \end{aligned}
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    In the interactive graph below, select a simplified propeller aircraft of your choice and experiment in finding an optimum by changing the control variables, $C_L$ and $\delta_T$. The design point is marked in white in the 3D velocity surface.
    """)
    return


@app.cell
def _(ac_table):
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
    & + \lambda_1 \left[\delta_T P_{a0}\sigma^\beta - \frac{W^{3/2}}{\sigma^{1/2}}\sqrt{\frac{2}{\rho_0S}} \left(C_{D_0}C_L^{-3/2} + KC_L^{1/2}\right)\right] + \\
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

    1. $\displaystyle \frac{\partial \mathcal{L}}{\partial C_L} = \frac{1}{2}\sqrt{\rho_0\frac{S}{2}\frac{\sigma}{W}}C_L^{-1/2} - \lambda_1 \frac{W^{3/2}}{\sigma^{1/2}}\sqrt{\frac{2}{\rho_0S}} \left(-\frac{3}{2}C_{D_0}C_L^{-5/2} + \frac{1}{2}KC_L^{-1/2}\right) + \mu_1 = 0$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1 P_{a0}\sigma^\beta + \mu_2 = 0$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $\displaystyle \delta_T P_{a0}\sigma^\beta - \frac{W^{3/2}}{\sigma^{1/2}}\sqrt{\frac{2}{\rho_0S}} \left(C_{D_0}C_L^{-3/2} + KC_L^{1/2}\right) = 0$
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
    CL_maxthrust_selected,
    CLmax,
    E_array,
    OptimumGridView,
    Pa0,
    W_selected,
    active_selection,
    beta,
    configTraces,
    dT_array,
    drag_curve,
    drag_maxliftThrust_selected,
    drag_yrange,
    h_maxliftThrust,
    h_maxthrust_array,
    h_selected,
    mach_trace,
    maxliftThrust_multiplier,
    np,
    plot_utils,
    power_maxthrust_selected,
    power_required_maxliftThrust,
    power_yrange,
    range_performance_diagrams,
    sigma_maxliftThrust,
    stall_trace,
    tab_value,
    thrust_vector_maxliftThrust,
    title_keys,
    true_maxliftThrust,
    true_maxthrust,
    velocity_CL_E,
    velocity_CL_P,
    velocity_maxliftThrust_CLarray,
    velocity_maxliftThrust_selected,
    velocity_maxthrust_harray,
    velocity_maxthrust_selected,
):
    if tab_value == title_keys[0]:
        # Interior graphics
        pass

    elif tab_value == title_keys[1]:
        # maxlift graphics
        pass

    elif tab_value == title_keys[2]:
        # Maxthrust graphics
        figure_optimum = OptimumGridView(
            configTraces,
            h_selected,
            (velocity_maxthrust_harray, velocity_maxthrust_selected),
            (np.nan, power_maxthrust_selected),
            (h_maxthrust_array, true_maxthrust, CL_maxthrust_selected, true_maxthrust),
            f"Thrust-limited minimum power for {active_selection.full_name}",
        )
        figure_optimum.update_axes_ranges(range_performance_diagrams)

    elif tab_value == title_keys[3]:
        constraint_maxliftThrust = (
            W_selected
            / E_array
            / Pa0
            / sigma_maxliftThrust**beta
            * velocity_maxliftThrust_selected
        )

        inv_velocity_surface_maxliftThrust = np.broadcast_to(
            (1 / velocity_maxliftThrust_CLarray)[np.newaxis, :],  # Shape: (101, 1)
            (len(CL_array), len(dT_array)),  # Target shape: (101, 101)
        )

        min_colorbar_maxliftThrust = np.min(inv_velocity_surface_maxliftThrust)
        max_colorbar_maxliftThrust = np.max(inv_velocity_surface_maxliftThrust)
        zcolorbar_maxliftThrust = (
            min_colorbar_maxliftThrust,
            max_colorbar_maxliftThrust,
        )

        # Create graphic traces
        configTraces_maxliftThrust = plot_utils.ConfigTraces(
            CL_array,
            dT_array,
            constraint_maxliftThrust,
            drag_curve,
            thrust_vector_maxliftThrust,
            power_required_maxliftThrust,
            thrust_vector_maxliftThrust * velocity_maxliftThrust_CLarray / 1e3,
            inv_velocity_surface_maxliftThrust,
            velocity_maxliftThrust_CLarray,
            velocity_CL_P * maxliftThrust_multiplier,
            velocity_CL_E * maxliftThrust_multiplier,
            velocity_maxliftThrust_selected,
            velocity_maxliftThrust_selected,
            (drag_yrange, power_yrange / 1e3, CLmax),
            zcolorbar_maxliftThrust,
            mach_trace,
            stall_trace,
        )

        # Maxliftthrust graphics
        figure_optimum = OptimumGridView(
            configTraces_maxliftThrust,
            h_selected,
            (velocity_maxliftThrust_CLarray, velocity_maxliftThrust_selected),
            (np.nan, drag_maxliftThrust_selected * velocity_maxliftThrust_selected),
            (h_maxliftThrust, 1, true_maxliftThrust, np.nan),
            f"Thrust-lift limited minimum drag for {active_selection.full_name}",
            equality=True,
        )

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
    \lambda_1 = -\frac{\mu_2}{P_{a0}\sigma^\beta} \lt 0
    $$

    Stationarity condition (1) then becomes:

    $$
    \begin{align}
    \lambda_1 &= \frac{\frac{\rho_0S}{2}\frac{\sigma^{1/2}}{W^{3/2}}C_L^{2}}{kC_L^2-3C_{D_0}} \lt 0 \quad \mathrm{for} \quad C_L \lt \sqrt{\frac{3C_{D_0}}{K}} = \sqrt{3}C_{L_E} = C_{L_P} \nonumber
    \end{align}
    $$

    This shows that maximum speed is obtained, intuitively, on the positive (right-hand side) branch of the performance diagram.

    The loosest condition is $C_{L_P} \lt C_{L_{}\mathrm{max}}$.

    The corresponding optimum value of the $C_L$ is obtained by solving the primal feasibiliy condition (3), having $\delta_T = 1$:

    $$
    P_{a0}\sigma^\beta - \frac{W^{3/2}}{\sigma^{1/2}}\sqrt{\frac{2}{\rho_0S}} \left(C_{D_0}C_L^{-3/2} + KC_L^{1/2}\right) = 0
    $$

    where it is impractical to obtain analytic solutions. The previous function and the conditions where it intercepts the y = 0 axis can therefore be studied graphically, as a function of $C_L$ (for different values of $W$ and $\sigma$) on the performance diagram.

    The operational condition is also found numerically by setting the numerical soltuion $C_L^*\lt C_{L_P}$
    The conditions
    Thus the optimal values are:

    $$
    \delta_T^* = 1, \quad C_L^* = \mathrm{numerically \: solved}, \quad \text{for} \:\:\frac{W^{1/2}}{\sigma^{\beta+1/2}} \lt  \mathrm{numerically \: solved}, \quad\text{if}\:\: C_{L_\mathrm{max}} \gt \sqrt{\frac{C_{D_0}}{K}}
    $$

    """),
            variables_stack,
            figure_optimum.figure,
        ]
    ).callout()
    return


@app.cell
def _(CD0, K, Pa0, S, atmos, beta, np):
    def maxthrust_solver(W, h):
        sigma = atmos.rhoratio(h)

        function = lambda CL: (
            Pa0 * sigma**beta
            - W**1.5
            / (sigma**0.5)
            * np.sqrt(2 / atmos.rho0 / S)
            * (CD0 + K * CL**2)
            / (CL ** (3 / 2))
        )

        return function

    def maxthrust_condition(CD0, K, CLstar, h_selected, h_array, CLmax):
        # condition = (CLmax > np.sqrt(CD0 / K)) & (CLstar < np.sqrt(3 * CD0 / K))

        mask_CL = ~np.isnan(CLstar)

        hopt_array = h_array[mask_CL]

        CL_optimum = CLstar[mask_CL]

        CL_selected = CL_optimum[np.isclose(hopt_array, h_selected)]

        CL_selected = CL_selected.item() if CL_selected.size == 1 else np.nan

        cond = 1 if hopt_array.min() <= h_selected <= hopt_array.max() else np.nan

        return hopt_array, CL_optimum, CL_selected, cond

    return maxthrust_condition, maxthrust_solver


@app.cell
def _(W_selected, h_array, maxthrust_solver, np, root_scalar):
    CL_maxthrust_star = []

    for h in h_array:
        func = maxthrust_solver(W_selected, h)
        CL_sol = root_scalar(func, x0=0.04).root
        CL_maxthrust_star.append(CL_sol)

    CL_maxthrust_star = np.array(CL_maxthrust_star)
    return (CL_maxthrust_star,)


@app.cell
def _(
    CD0,
    CL_maxthrust_star,
    CLmax,
    K,
    S,
    W_selected,
    atmos,
    h_array,
    h_selected,
    maxthrust_condition,
    np,
):
    h_maxthrust_array, CLopt_maxthrust, CL_maxthrust_selected, true_maxthrust = (
        maxthrust_condition(CD0, K, CL_maxthrust_star, h_selected, h_array, CLmax)
    )

    velocity_maxthrust_harray = np.sqrt(
        2 * W_selected / (atmos.rho(h_maxthrust_array) * CLopt_maxthrust * S)
    )
    velocity_maxthrust_selected = np.sqrt(
        2 * W_selected / (atmos.rho(h_selected) * CL_maxthrust_selected * S)
    )

    dTopt_maxthrust = 1 * true_maxthrust

    power_maxthrust_selected = (
        W_selected
        * (CD0 + K * CL_maxthrust_selected**2)
        / CL_maxthrust_selected
        * velocity_maxthrust_selected
    )
    return (
        CL_maxthrust_selected,
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
    \lambda_1 = -\frac{\mu_2}{P_{a0}\sigma^\beta} < 0
    $$

    From stationary condition (1):

    $$
    \mu_1 =\lambda_1 \frac{W^{3/2}}{\sigma^{1/2}}\sqrt{\frac{2}{\rho_0S}} \left(-\frac{3}{2}C_{D_0}C_{L_\mathrm{max}}^{-5/2} + \frac{1}{2}KC_{L_\mathrm{max}}^{-1/2}\right) -  \frac{1}{2}\sqrt{\rho_0\frac{S}{2}\frac{\sigma}{W}}C_{L_\mathrm{max}}^{-1/2}\gt 0
    $$

    $$
    \mathrm{for}\quad \quad\frac{\rho_0\frac{S}{2}\frac{\sigma^{1/2}}{W^{3/2}}C_{L_\mathrm{max}}^2}{KC_{L_\mathrm{max}}^2 - 3C_{D_0}} \lt \lambda_1 \lt 0 \quad \Leftrightarrow \quad C_{L_\mathrm{max}} \lt \sqrt{\frac{3C_{D_0}}{K}} = \sqrt{3}C_{L_E} = C_{L_P}
    $$


    In order for this case to occur, the aircraft has to be designed to stall at a higher speed than the one for minimum power, in the same conditions of weight and altitude. $C_{L_\mathrm{max}}$ becomes the limiting $C_L$ when maximizing speed, as it is not possible to lower it even more towards C_{L_P}.

    $$
    C_L^* = C_{L_\mathrm{max}}, \quad \delta_T^*=1, \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}} = P_{a0}E_S\sqrt{\frac{\rho_0 S}{2}C_{L_{\mathrm{max}}}}, \quad \mathrm{if} \quad C_{L_\mathrm{max}} \lt C_{L_P}
    $$
    """),
            variables_stack,
            figure_optimum.figure,
        ]
    ).callout()
    return


@app.cell
def _(atmos, np):
    def maxliftThrust_condition(W, Pa0, E_S, beta, CL_E, CL_P, S, CLmax):
        sigma_maxliftThrust = (
            W**1.5 / Pa0 / E_S / (np.sqrt(atmos.rho0 * S * CLmax / 2))
        ) ** (1 / (beta + 0.5))
        h_maxliftThrust_selected = atmos.altitude(sigma_maxliftThrust)

        if CLmax > CL_P:
            return h_maxliftThrust_selected, sigma_maxliftThrust, np.nan

        condition = True

        return (
            h_maxliftThrust_selected,
            sigma_maxliftThrust,
            condition,
        )

    return (maxliftThrust_condition,)


@app.cell
def _(
    CL_E,
    CL_P,
    CLmax,
    E_S,
    Pa0,
    S,
    W_selected,
    atmos,
    beta,
    drag_curve,
    maxliftThrust_condition,
    meshgrid_n,
    np,
    rho_selected,
    velocity_CLarray,
):
    # Max lift Max thrust
    h_maxliftThrust, sigma_maxliftThrust, true_maxliftThrust = maxliftThrust_condition(
        W_selected, Pa0, E_S, beta, CL_E, CL_P, S, CLmax
    )

    maxliftThrust_multiplier = np.sqrt(
        rho_selected / (atmos.rho0 * sigma_maxliftThrust)
    )

    power_available_maxliftThrust = (
        np.repeat(Pa0 * sigma_maxliftThrust**beta, meshgrid_n) / 1e3
    )

    velocity_maxliftThrust_CLarray = velocity_CLarray * maxliftThrust_multiplier
    velocity_maxliftThrust_selected = velocity_maxliftThrust_CLarray[-1]
    thrust_vector_maxliftThrust = (
        power_available_maxliftThrust / velocity_maxliftThrust_CLarray * 1e3
    )
    power_required_maxliftThrust = drag_curve * velocity_maxliftThrust_CLarray / 1e3

    drag_maxliftThrust_selected = W_selected / E_S
    return (
        drag_maxliftThrust_selected,
        h_maxliftThrust,
        maxliftThrust_multiplier,
        power_required_maxliftThrust,
        sigma_maxliftThrust,
        thrust_vector_maxliftThrust,
        true_maxliftThrust,
        velocity_maxliftThrust_CLarray,
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
    true_maxliftThrust,
    velocity_maxliftThrust_selected,
    velocity_maxthrust_harray,
    velocity_stall_harray,
):
    flight_envelope = plot_utils.create_final_flightenvelope(
        velocity_stall_harray,
        a_harray,
        h_array,
        (np.nan, np.nan, False),
        (
            np.concat((h_maxthrust_array, [h_maxliftThrust])),
            np.concat(
                (
                    velocity_maxthrust_harray,
                    [velocity_maxliftThrust_selected * true_maxliftThrust],
                )
            ),
            True,
        ),
        (np.nan, np.nan, False),
        (h_maxliftThrust, velocity_maxliftThrust_selected * true_maxliftThrust, False),
    )

    mo.vstack([mass_stack, flight_envelope])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Name | Condition | $C_L^*$ | $\delta_T^*$ | $V^*$ |
    |:-|:----------|:-------:|:------------:|:------|
    |Thrust and Lift-limited    | $\displaystyle \frac{W^{3/2}}{\sigma^{\beta+1/2}} = P_{a0}E_S\sqrt{\frac{\rho_0 S}{2}C_{L_{\mathrm{max}}}}$ | $C_{L_\mathrm{max}}$ | $1$ | $\displaystyle V_s =\sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}$ |
    |Thrust-limited    | $\displaystyle \mathrm{numerical}$ | $\displaystyle \mathrm{numerical}$ | $1$ | $\displaystyle \mathrm{numerical}$ |
    """)
    return


@app.cell
def _():
    _defaults.nav_footer(
        before_file="MaxSpeed_Jet.py",
        before_title="Maximum Speed Simplified Jet",
        above_file="MaxSpeed.py",
        above_title="Maximum Speed Homepage",
        above_before=False,
    )
    return


if __name__ == "__main__":
    app.run()
