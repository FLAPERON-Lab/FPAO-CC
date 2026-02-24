import marimo

__generated_with = "0.20.2"
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


@app.cell
def _():
    # Set navbar on the right
    _defaults.set_sidebar()
    return


@app.cell
def _():
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
def _(a_0, ac_table, dT_array, data, meshgrid_n, xy_lowerbound):
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
def _(CD0, CLmax, E_array, K, MTOM, OEM, S, a_0, h_array, m_slider, rho_array):
    # Define variables, this cell runs every time the mass slider is run
    W_selected = (OEM + (MTOM - OEM) * m_slider.value) * atmos.g0  # Netwons
    drag_curve = W_selected / E_array

    velocity_stall_harray = np.sqrt(2 * W_selected / (rho_array * S * CLmax))

    # Visual computations
    stall_trace = plot_utils.create_stall_trace(h_array, velocity_stall_harray)

    CL_a0 = OEM * atmos.g0 * 2 / (atmos.rho0 * S * a_0**2)

    drag_yrange = 1 * OEM * atmos.g0 * (CD0 + K * CL_a0**2) / CL_a0
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
def _(Ta0, beta, h_array, h_slider, meshgrid_n):
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

    velocity_surface = np.broadcast_to(
        velocity_CLarray[np.newaxis, :],  # Shape: (101, 1)
        (len(CL_array), len(dT_array)),  # Target shape: (101, 101)
    )

    constraint = drag_curve / Ta0 / (sigma_selected**beta)

    min_colorbar = np.min(velocity_CLarray)
    max_colorbar = min_colorbar * 2
    zcolorbar = (min_colorbar, max_colorbar)

    range_performance_diagrams = (drag_yrange, power_yrange, CLmax)

    # Create graphic traces
    configTraces = plot_utils.ConfigTraces(
        CL_array,
        dT_array,
        constraint,
        drag_curve,
        thrust_vector,
        power_required,
        power_available,
        velocity_surface,
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
        max_colorbar,
        min_colorbar,
        range_performance_diagrams,
        velocity_CL_E,
        velocity_CL_P,
        velocity_CLarray,
        velocity_surface,
    )


@app.cell
def _(idx_CL_selected, velocity_CLarray):
    velocity_selected = velocity_CLarray[idx_CL_selected]
    return (velocity_selected,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Minimum airspeed: simplified jet aircraft

    $$
    \begin{aligned}
        \min_{C_L, \delta_T}
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
    a_0,
    active_selection,
    constraint,
    dT_array,
    dT_slider,
    max_colorbar,
    min_colorbar,
    velocity_selected,
    velocity_surface,
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
                z=velocity_surface,
                opacity=0.9,
                name="Velocity",
                colorscale="viridis",
                cmax=max_colorbar,
                cmin=min_colorbar,
                colorbar={"title": "Velocity (m/s)"},
            ),
            go.Scatter3d(
                x=CL_array,
                y=constraint,
                z=velocity_surface[0],
                opacity=1,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[CL_array[-15]],
                y=[constraint[-15]],
                z=[velocity_surface[0, -15]],
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
                z=[velocity_selected],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=3,
                    color="white",
                    symbol="circle",
                ),
                name="Design Point",
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub> : %{y}<br>V: %{z}<extra>%{fullData.name}</extra>",
            ),
        ]
    )
    camera = dict(eye=dict(x=1.35, y=1.35, z=1.35))

    fig_initial.update_layout(
        scene=dict(
            xaxis=dict(
                title="C<sub>L</sub> (-)",
                range=[xy_lowerbound, active_selection["CLmax_ld"]],
            ),
            yaxis=dict(title="δ<sub>T</sub> (-)", range=[xy_lowerbound, 1]),
            zaxis=dict(title="V (m/s)", range=[0, a_0]),
        ),
    )

    fig_initial.update_layout(
        scene_camera=camera,
        title={
            "text": f"Minimum airspeed domain for {active_selection.full_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        },
    )

    mo.output.clear()
    return (fig_initial,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## KKT formulation
    To be reconducted in the standard KKT analysis format, the objective function is expressed in terms of the controls by direct elimination of $c_1^\mathrm{eq}$.
    Also, minimizing $V$ is equivalent to minimizing $V^2$, because the square power function is monotonically increasing.
    Therefore, to simplify the calculations, the problem is rewritten as follows:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $$
    \begin{aligned}
        \min_{C_L, \delta_T}
        & \quad V^2 = \frac{2W}{\rho S C_L} \\
        \text{subject to}
        & \quad g_1 = \frac{T}{W} - \frac{1}{E}  =\frac{\delta_T T_{a0}\sigma^\beta}{W} - \frac{C_{D_0} + K C_L^2}{C_L} = 0 \\
        & \quad h_1 = C_L - C_{L_\mathrm{max}} \le 0 \\
        & \quad h_2 = -C_L \le 0 \\
        & \quad h_3 = \delta_T - 1 \le 0 \\
        & \quad h_4 = -\delta_T \le 0 \\
    \end{aligned}
    $$
    """)
    return


@app.cell
def _():
    mo.md(r"""
    In the interactive graph below, select a simplified jet aircraft of your choice and experiment in finding an optimum by changing the control variables, $C_L$ and $\delta_T$. The design point is marked in white in the 3D velocity surface.
    """)
    return


@app.cell(hide_code=True)
def _(ac_table):
    # Database cell (1)
    ac_table
    return


@app.cell(hide_code=True)
def _(CL_slider, dT_slider):
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Lagrangian function and KKT conditions

    The Lagrangian function combines the objective function with equality constraints using Lagrange multipliers ($\lambda_i$) and inequality constraints using KKT multipliers ($\mu_j$).

    $$
    \begin{aligned}
    \mathcal{L}(C_L, \delta_T, \lambda_1, \mu_1, \mu_2, \mu_3, \mu_4) =
    \quad \frac{2W}{\rho S C_L}
    & + \\
    & + \lambda_1 \left[\frac{\delta_T T_{a0}\sigma^\beta}{W} - \frac{C_{D_0} + K C_L^2}{C_L}\right] + \\
    & + \mu_1 (C_L - C_{L_\mathrm{max}}) + \\
    & + \mu_2 (-C_L) + \\
    & + \mu_3 (\delta_T - 1) +\\
    & + \mu_4 (-\delta_T)
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The multipliers $\lambda_1, \mu_1, \mu_2, \mu_3, \mu_4$ have to meet the following conditions for an optimal solution of the optimization problem $(C_L^*, \delta_T^*)$ to exist.

    **A. Stationarity ($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \frac{\partial \mathcal{L}}{\partial C_L} = -\frac{2W}{\rho S C_L^2} + \lambda_1 \left(\frac{C_{D_0}- KC_L^2}{C_L^2}\right) + \mu_1 - \mu_2 = 0$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1 \frac{T_{a0}\sigma^\beta}{W} + \mu_3 - \mu_4 = 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $\displaystyle \frac{\delta_T T_{a0}\sigma^\beta}{W} - \frac{C_{D_0} + K C_L^2}{C_L} = 0$
    4.  $C_L - C_{L_\mathrm{max}} \le 0$
    5.  $-C_L \le 0$
    6.  $\delta_T - 1 \le 0$
    7.  $-\delta_T \le 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    8.  $\mu_1, \mu_2, \mu_3, \mu_4 \ge 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraint have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

    9.  $\mu_1 (C_L - C_{L_\mathrm{max}}) = 0$
    10. $\mu_2 (-C_L) = 0$
    11. $\mu_3 (\delta_T - 1) = 0$
    12. $\mu_4 (-\delta_T) = 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## KKT analysis

    We can now proceed to systematically examine the conditions where various inequality constraints are active or inactive.
    ### _Interior solutions_

    Assuming that that $0 < C_L < C_{L_\mathrm{max}}$ and $0 < \delta_T < 1$ is equivalent to consider all inequality constraints as inactive.

    Therefore: $\mu_1,\mu_2,\mu_3,\mu_4=0$.

    From stationarity condition (2): $\lambda_1 = 0$.

    It can now be seen that stationarity condition (1) is never verified.

    It can be concluded that the minimum speed cannot be achieved in the interior of the domain.
    The minimum must lie on at least one of the boundaries defined by $C_L = C_{L_\mathrm{max}}$ or $\delta_T = 1$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### _Lower boundary solutions_
    The case where $C_L=0$ and the case where $\delta_T=0$ can be immediately discarded because of the primal feasibility conditions.
    This means that $\mu_2=\mu_4=0$ in all cases.

    We can then proceed with the analysis of the cases where the boundaries $C_L = C_{L_\mathrm{max}}$ and $\delta_T = 1$ are active in any of the three possible combinations.
    """)
    return


@app.cell
def _():
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
    CLopt_maxlift,
    E_array,
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
            (h_maxthrust_array, dTopt_maxthrust, CL_maxthrust_selected, true_maxthrust),
            f"Thrust-limited minimum power for {active_selection.full_name}",
        )

    elif tab_value == title_keys[3]:
        velocity_surface_maxliftThrust = np.broadcast_to(
            velocity_CLarray_maxliftThrust[np.newaxis, :],  # Shape: (101, 1)
            (len(CL_array), len(dT_array)),  # Target shape: (101, 101)
        )

        min_colorbar_maxliftThrust = np.min(velocity_surface_maxliftThrust)
        max_colorbar_maxliftThrust = min_colorbar_maxliftThrust * 2
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
            velocity_surface_maxliftThrust,
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
def _(tab_value, title_keys):
    if tab_value != title_keys[0]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ### _Interior solutions_ 

    Assuming that that $0 < C_L < C_{L_\mathrm{max}}$ and $0 < \delta_T < 1$ is equivalent to consider all inequality constraints as inactive.

    Therefore: $\mu_1,\mu_2,\mu_3,\mu_4=0$. 

    From stationarity condition (2): $\lambda_1 = 0$.

    It can now be seen that stationarity condition (1) is never verified.

    It can be concluded that the minimum speed cannot be achieved in the interior of the domain. 
    The minimum must lie on at least one of the boundaries defined by $C_L = C_{L_\mathrm{max}}$ or $\delta_T = 1$.
    """)
        ]
    ).callout()
    return


@app.cell
def _(figure_optimum, tab_value, title_keys, variables_stack):
    if tab_value != title_keys[2]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ### _Thrust-limited minimum airspeed_

    $\delta_T=1 \quad \Rightarrow \quad \mu_3 > 0$

    $C_L < C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 = 0$

    From stationarity condition (2): 

    $$
    \lambda_1 = -\mu_3\frac{W}{T_{a0}\sigma^\beta} \quad \Rightarrow \quad \lambda_1 < 0
    $$

    Stationarity condition (1) then becomes:

    $$
    \frac{2T_{a0}\sigma^\beta}{\rho S C_L^2} + \mu_3\left( \frac{C_{D_0}-KC_L^2}{C_L^2}\right) = 0
    \quad \text{and } \quad 
    \mu_3>0 
    \quad \Rightarrow \quad 
    C_L^* > \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$

    and implies that the thrust-limited minimum airspeed is obtained strictly on the left branch of the drag performance diagram, at a lift coefficient strictly higher than the one for maximum aerodynamic efficiency.

    The corresponding optimum value of the $C_L$ is obtained by solving the primal feasibility condition (3) and taking the highest of the two solutions:

    $$
    C_L^* = \frac{T_{a0}\sigma^\beta}{2KW} \left[1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}\right]
    $$

    It has still to be verified that $C_L^* < C_{L_\mathrm{max}}$, which depends on the numerical values of the design parameters, and on the current values of the weight and altitude.

    First, this optimum value of the lift coefficient is achievable for 

    $$
    1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2 \ge 0
    \quad \Rightarrow \quad 
    \frac{W}{\sigma^\beta} \le  T_{a0} E_\mathrm{max}
    $$

    The limit equality can be used to calculate the corresponding limit altitude at which the minimum speed is limited by thrust, for a given weight. This is called the _theoretical ceiling_.

    Second, the optimum value is lower than $C_{L_\mathrm{max}}$ if

    $$
    \frac{W}{\sigma^\beta} > T_{a0} E_\mathrm{S}
    $$

    This concludes the analysis for the minimum airspeed of a simplified jet aircraft in the thrust-limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = \frac{T_{a0}\sigma^\beta}{2KW} \left[1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}\right]}, \quad \boxed{\delta_T^*=1}, \quad \text{for} \quad C_L^* > \sqrt{\frac{C_{D_0}}{K}}\quad \text{and} \quad \frac{W}{\sigma^\beta} > T_{a0} E_\mathrm{S}
    $$

    If the conditions stated above are satisfied, the objective function $V$ takes the value: 

    $$
    V_{\mathrm{min}}^* = \sqrt{\frac{4KW^2/\rho S T_{a0}\sigma^\beta}{1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}}}
    = V_s \sqrt{\frac{2KWC_{L_\mathrm{max}}/T_{a0}\sigma^\beta}{1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}}}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.
    """),
            variables_stack,
            figure_optimum.figure,
        ]
    ).callout()
    return


@app.function
def maxthrust_condition(W, h_selected, K, E_max, E_S, h_array, Ta0, beta, sigma_array):
    B = W / E_max / Ta0
    max_sigma = (W / E_S / Ta0) ** (1 / beta)

    if max_sigma >= 1:
        return np.array([np.nan]), 1, np.nan, False

    dT_optimum = 1
    min_h = atmos.altitude(max_sigma)

    hopt_array = h_array[(h_array > min_h)]

    sigma_optimum = sigma_array[np.isin(h_array, hopt_array)]

    A = Ta0 * sigma_optimum**beta / (2 * K * W)
    B = (B / (sigma_optimum**beta)) ** 2

    CL_optimum = A * (1 + np.sqrt(1 - B, where=(B < 1), out=np.full_like(B, np.nan)))

    mask_CL = ~np.isnan(CL_optimum)

    hopt_array = hopt_array[mask_CL]
    CL_optimum = CL_optimum[mask_CL]
    CL_selected = CL_optimum[np.isclose(hopt_array, h_selected)]

    CL_selected = CL_selected.item() if CL_selected.size == 1 else np.nan

    cond = 1 if min_h <= h_selected else np.nan
    return (
        hopt_array,
        dT_optimum,
        CL_optimum,
        CL_selected,
        cond,
    )


@app.cell
def _(
    CD0,
    E_S,
    E_max,
    K,
    S,
    Ta0,
    W_selected,
    beta,
    h_array,
    h_selected,
    rho_selected,
    sigma_array,
):
    # Maxthrust computations
    (
        h_maxthrust_array,
        dTopt_maxthrust,
        CLopt_maxthrust,
        CL_maxthrust_selected,
        true_maxthrust,
    ) = maxthrust_condition(W_selected, h_selected, K, E_max, E_S, h_array, Ta0, beta, sigma_array)

    velocity_maxthrust_harray = np.sqrt(2 * W_selected / (atmos.rho(h_maxthrust_array) * S * CLopt_maxthrust))

    velocity_maxthrust_selected = (
        np.sqrt(2 * W_selected / (rho_selected * S * CL_maxthrust_selected)) * true_maxthrust
    )

    power_maxthrust_selected = (
        W_selected * (CD0 + K * CL_maxthrust_selected**2) / CL_maxthrust_selected * velocity_maxthrust_selected
    )
    return (
        CL_maxthrust_selected,
        dTopt_maxthrust,
        h_maxthrust_array,
        power_maxthrust_selected,
        true_maxthrust,
        velocity_maxthrust_harray,
        velocity_maxthrust_selected,
    )


@app.cell
def _(figure_optimum, tab_value, title_keys, variables_stack):
    if tab_value != title_keys[1]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ###_Lift-limited minimum airspeed_

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 > 0$ 

    $0 < \delta_T < 1 \quad \Rightarrow \quad \mu_3 = 0$.

    From stationarity condition (2): $\lambda_1 = 0$.

    From stationarity condition (1): $\displaystyle \mu_1 = \frac{2W}{\rho S C_{L_\mathrm{max}}^2}>0$, which does not depend on the value of $\delta_T$, and is always verified.

    The corresponding value of the throttle is calculated from the primal feasibility condition (3):

    $$
    \delta_T^*
    = \frac{W}{T_{a0}\sigma^\beta} \frac{C_{D_0} + K C^2_{L_\mathrm{max}}}{C_{L_\mathrm{max}}} 
    = \frac{W}{T_{a0}\sigma^\beta} \frac{1}{E_S} 
    $$

    This is valid only if the calculated $\delta_T^*$ is strictly lower than the maximum allowed, which corresponds to:

    $$
    \frac{W}{\sigma^\beta} < T_{a0} E_S
    $$

    The limit equality can be used to calculate the corresponding limit altitude at which the minimum speed is limited by lift, for a given weight.

    The corresponding minimum airspeed is called the _stall speed_.

    $$
    V^* = \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}
    $$

    This concludes the analysis for the minimum airspeed of a simplified jet aircraft in the lift-limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = C_{L_\mathrm{max}}}, \quad \boxed{\delta_T^*= \frac{W}{T_{a0}\sigma^\beta} \frac{1}{E_S}}, \quad \text{for} \quad \frac{W}{\sigma^\beta} < T_{a0} E_S
    $$

    If the conditions stated above are satisfied, the objective function $V$ takes the value: 

    $$
    V_{\mathrm{min}}^* = \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.
    """),
            variables_stack,
            figure_optimum.figure,
        ]
    ).callout()
    return


@app.function
def maxlift_condition(
    W,
    h_selected,
    CLmax,
    E_S,
    Ta0,
    beta,
    h_array,
    min_sigma,
    sigma_selected,
):
    sigma_optimum = (W / E_S / Ta0) ** (1 / beta)
    dT = W / E_S / Ta0 / (sigma_selected**beta)

    if sigma_optimum <= min_sigma:
        return np.array([np.nan]), dT, np.nan, np.nan

    maximum_hopt = atmos.altitude(sigma_optimum)

    hopt_array = h_array[h_array < maximum_hopt]

    h_min = hopt_array.min()
    h_max = hopt_array.max()
    cond = 1 if h_min <= h_selected <= h_max else np.nan

    return hopt_array, dT, CLmax, cond


@app.cell
def _(
    CLmax,
    E_S,
    Ta0,
    W_selected,
    beta,
    h_array,
    h_selected,
    min_sigma,
    rho_selected,
    sigma_selected,
    velocity_CLarray,
):
    # Maxlift condition
    h_maxlift_array, dTopt_maxlift, CLopt_maxlift, true_maxlift = maxlift_condition(
        W_selected,
        h_selected,
        CLmax,
        E_S,
        Ta0,
        beta,
        h_array,
        min_sigma,
        sigma_selected,
    )

    velocity_maxlift_selected = velocity_CLarray[-1] * true_maxlift
    velocity_maxlift_harray = velocity_CLarray[-1] * np.sqrt(rho_selected / atmos.rho(h_maxlift_array))

    power_maxlift_selected = W_selected / E_S * velocity_maxlift_selected
    return (
        CLopt_maxlift,
        dTopt_maxlift,
        h_maxlift_array,
        power_maxlift_selected,
        true_maxlift,
        velocity_maxlift_harray,
        velocity_maxlift_selected,
    )


@app.cell
def _(figure_optimum, tab_value, title_keys, variables_stack):
    if tab_value != title_keys[3]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ### _Thrust- and lift-limited minimum speed_

    $\delta_T = 1 \quad \Rightarrow \quad \mu_3 > 0$

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 > 0$.

    From the stationary conditions (2):

    $$
    \lambda_1 = -\frac{\mu_3}{T_{a0}\sigma^\beta} \quad \Rightarrow \quad \lambda_1 < 0
    $$

    From stationary condition (1): 

    $$
    \mu_1 = \frac{2W}{\rho S C_{L_\mathrm{max}}^2} + \mu_3\frac{W}{T_{a0}\sigma^\beta}\left(\frac{C_{D_0} - K C_{L_\mathrm{max}}^2}{C_{L_\mathrm{max}}^2}\right) > 0 
    \quad \text{if } \quad
    C_{L_\mathrm{max}} < \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$

    In other words, this condition is reached only if the aircraft is designed in such a way that its maximum lift coefficient is lower than the one for maximum aerodynamic efficiency. 
    It is obvious then that, for the same combination of weight and altitude, its stall speed will be higher than the speed for maximum efficiency (and minimum drag), which would then be unreachable for the aircraft in Steady Level Flight.
    This is, of course, an undesired situation to be in, and should not be resulting out of good aerodynamic design.

    The primal feasibility equation (3) returns the expression of the condition where the minimum speed is limited by both thrust and lift capabilities of the aircraft.

    $$
    \frac{W}{\sigma^\beta} = T_{a0} E_S
    $$

    The corresponding value of the airspeed is once again

    $$
    V^* = \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}
    $$

    This concludes the analysis for the minimum airspeed of a simplified jet aircraft in the lift-thrust limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = C_{L_\mathrm{max}}}, \quad \boxed{\delta_T^*= 1}, \quad \text{for} \quad \frac{W}{\sigma^\beta} = T_{a0} E_S
    $$

    If the conditions stated above are satisfied, the objective function $V$ takes the value: 

    $$
    V_{\mathrm{min}}^* = \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.
    """),
            variables_stack,
            figure_optimum.figure,
        ]
    ).callout()
    return


@app.function
def maxliftThrust_condition(W, Ta0, E_S, beta, min_sigma, CLmax):
    sigma_maxliftThrust = (W / Ta0 / E_S) ** (1 / beta)
    h_maxliftThrust_selected = atmos.altitude(sigma_maxliftThrust)

    if sigma_maxliftThrust >= min_sigma:
        return h_maxliftThrust_selected, sigma_maxliftThrust, np.nan, 1, np.nan

    condition = True

    return (
        h_maxliftThrust_selected,
        sigma_maxliftThrust,
        CLmax,
        1,
        condition,
    )


@app.cell
def _(
    CLmax,
    E_S,
    Ta0,
    W_selected,
    beta,
    drag_curve,
    meshgrid_n,
    min_sigma,
    rho_selected,
    velocity_CLarray,
):
    (
        h_maxliftThrust,
        sigma_maxliftThrust,
        CLopt_maxliftThrust,
        dTopt_maxliftThrust,
        true_maxliftThrust,
    ) = maxliftThrust_condition(W_selected, Ta0, E_S, beta, CLmax, min_sigma)

    maxliftThrust_multiplier = np.sqrt(rho_selected / atmos.rho(h_maxliftThrust))
    velocity_maxliftThrust_selected = velocity_CLarray[-1] * maxliftThrust_multiplier
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
def _():
    mo.md(r"""
    ## Final flight envelope
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Now after deriving all the optima for each condition we can summarize the flight envelopes in one graph, as shown below. Experiment with the weight of the aircrarft to understand how the theoretical ceiling for minimum speed moves in the graph.
    """)
    return


@app.cell
def _(
    a_harray,
    h_array,
    h_maxliftThrust,
    h_maxlift_array,
    h_maxthrust_array,
    mass_stack,
    velocity_maxliftThrust_selected,
    velocity_maxlift_harray,
    velocity_maxthrust_harray,
    velocity_stall_harray,
):
    flight_envelope = plot_utils.create_final_flightenvelope(
        velocity_stall_harray,
        a_harray,
        h_array,
        (np.nan, np.nan, False),
        (
            np.concat((h_maxlift_array, [h_maxliftThrust], h_maxthrust_array)),
            np.concat(
                (
                    velocity_maxlift_harray,
                    [velocity_maxliftThrust_selected],
                    velocity_maxthrust_harray,
                )
            ),
            True,
        ),
        (np.nan, np.nan, False),
        (h_maxliftThrust, velocity_maxliftThrust_selected, False),
    )

    mo.vstack([mass_stack, flight_envelope])
    return


@app.cell
def _():
    mo.md(r"""
    ## Summary
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    | Name | Condition | $C_L^*$ | $\delta_T^*$ | $V^*$ |
    |:- |:----------|:-------:|:------------:|:------|
    |Interior-optima|\ |\ |\ |\ ||
    |Lift-limited    | $\displaystyle \frac{W}{\sigma^\beta} < T_{a0} E_S$ | $C_{L_\mathrm{max}}$ | $\displaystyle \frac{W}{T_{a0}\sigma^\beta} \frac{1}{E_S}$ | $\displaystyle V_s = \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}$ |
    |Thrust and Lift-limited    | $\displaystyle \frac{W}{\sigma^\beta} =  T_{a0} E_S$, $C_{L_\mathrm{max}} < \sqrt{\frac{C_{D_0}}{K}}$ | $C_{L_\mathrm{max}}$ | $1$ | $\displaystyle V_s =\sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}$ |
    |Thrust-limited    | $\displaystyle T_{a0} E_\mathrm{S} < \frac{W}{\sigma^\beta} \le  T_{a0} E_\mathrm{max}$ | $\displaystyle \frac{T_{a0}\sigma^\beta}{2KW} \left[1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}\right]$ | $1$ | $\displaystyle V_s \sqrt{\frac{2KWC_{L_\mathrm{max}}/T_{a0}\sigma^\beta}{1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}}}$ |
    """
    ).center()
    return


@app.cell
def _():
    _defaults.nav_footer(
        after_file="MinSpeed_Prop_KKT.py",
        after_title="Minimum Speed Simplified Propeller",
        above_file="MinSpeed.py",
        above_title="Minimum Speed Homepage",
        above_before=True,
    )
    return


if __name__ == "__main__":
    app.run()
