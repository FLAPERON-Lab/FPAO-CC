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
    data_dir = str(
        mo.notebook_location().parent.parent / "data" / "AircraftDB_Standard.csv"
    )


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
    CL_grid, dT_grid = np.meshgrid(CL_array, dT_array)

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
        CL_grid,
        CL_slider,
        CLmax,
        E_P,
        E_S,
        E_array,
        K,
        MTOM,
        OEM,
        Pa0,
        S,
        active_selection,
        beta,
        dT_grid,
    )


@app.cell
def _():
    return


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
    CL_grid,
    CLmax,
    E_array,
    K,
    MTOM,
    OEM,
    Pa0,
    S,
    a_0,
    atmos,
    beta,
    dT_grid,
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

    sigma_surface = (
        W_selected ** (1.5)
        / (dT_grid * Pa0)
        * np.sqrt(2 / atmos.rho0 / S)
        * (CD0 + K * CL_grid**2)
        / (CL_grid**1.5)
    ) ** (1 / (beta + 0.5))

    return (
        W_selected,
        drag_curve,
        drag_yrange,
        power_yrange,
        sigma_surface,
        stall_trace,
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

    power_scalar = Pa0 * sigma_selected**beta / 1e3

    power_vector = np.repeat(power_scalar, meshgrid_n)
    return power_scalar, power_vector, rho_selected, sigma_selected


@app.cell
def _(
    CL_E,
    CL_P,
    CL_array,
    CLmax,
    S,
    W_selected,
    drag_curve,
    drag_yrange,
    np,
    power_scalar,
    power_yrange,
    rho_selected,
):
    # Computation only cell, indexing happens in another cell
    velocity_CLarray = np.sqrt(2 * W_selected / (rho_selected * S * CL_array))
    velocity_CL_E = velocity_CLarray[-1] * np.sqrt(CLmax / CL_E)
    velocity_CL_P = velocity_CLarray[-1] * np.sqrt(CLmax / CL_P)

    thrust_available = power_scalar / velocity_CLarray * 1e3
    power_required = drag_curve * velocity_CLarray / 1e3

    range_performance_diagrams = (drag_yrange, power_yrange, CLmax, 400)
    return velocity_CL_E, velocity_CL_P, velocity_CLarray


@app.cell
def _(dT_array, dT_slider):
    step_dT = dT_array[2] - dT_array[1]
    dT_selected = float(dT_slider.value)
    idx_dT_selected = int((dT_selected - dT_array[0]) / step_dT)
    return (idx_dT_selected,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Maximum Altitude: simplified propeller aircraft

    $$
    \begin{aligned}
        \min_{C_L, \delta_T}
        & \quad h \\
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here, h does not appear explicitely but we can transform the problem formulation in a convenient way, by knowing $\rho(h)$ is a monotonically decreasing function of h, as shown in the graph below.

    $$
    \begin{aligned}
        \min_{C_L, \delta_T}
        & \quad h  \qquad \Longleftrightarrow \qquad \max_{C_L, \delta_T} \quad \sigma = \frac{\rho(h)}{\rho_0}\\
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Moreover, since density is always positive, and $\beta$ as well, we can say, because $\sigma^\beta$ is a monotically increasing function of $\sigma$, minimizing $\sigma^\beta$ minimizes $\sigma$ which is maximizing $h$.

    $$
    \min_{C_L, \delta_T} \sigma  \quad \Longleftrightarrow \quad \min_{C_L, \delta_T} \quad \sigma^\beta \quad \Longleftrightarrow \quad \min_{C_L, \delta_T} \quad \sigma^{\beta + 1/2}
    $$

    We can thus now susbitute the horizontal equilibrium equation in the objective function directly, and then also substitute the expression of $V$ rom the vertical equilibrium, constraint.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## KKT formulation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The KKT formulation can now be written:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$
    \begin{aligned}
        \min_{C_L, \delta_T}
        & \quad \sigma^{\beta + 1/2} = \frac{W^{3/2}}{\delta_T P_{a0}}\sqrt{\frac{2}{\rho_0 S}}\left(\frac{C_{D_0} + K C_L^2}{C_L^{3/2}}\right)\\
        \text{subject to}
        & \quad h_1 = C_L - C_{L_\mathrm{max}} \le 0 \\
        & \quad h_2 = \delta_T - 1 \le 0 \\
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The lower bounds for the lift coefficient ($C_L = 0$), and for $\delta_T$ have already been excluded as they cannot comply with the vertical and horizontal constraints respectively.

    As it can be noted, the problem is now formulated to have only inequality constraints due to the bounds on the decision variables. In other words, it is an unconstrained optimization problem in a partially bounded domain.
    """)
    return


@app.cell(hide_code=True)
def _(ac_table):
    # Database cell
    ac_table
    return


@app.cell
def _(
    CL_array,
    CL_slider,
    active_selection,
    beta,
    dT_array,
    dT_slider,
    go,
    idx_CL_selected,
    idx_dT_selected,
    mo,
    np,
    sigma_surface,
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
                z=sigma_surface**beta,
                opacity=0.9,
                name="σ<sup>β</sup>",
                colorscale="viridis",
                cmin=np.min(sigma_surface),
                cmax=1,
                colorbar={"title": "σ<sup>β</sup> (-)"},
            ),
            go.Scatter3d(
                x=[CL_slider.value],
                y=[dT_slider.value],
                z=[sigma_surface[idx_dT_selected, idx_CL_selected] ** beta],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=3,
                    color="white",
                    symbol="circle",
                ),
                name="Design Point",
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub> : %{y}<br>σ<sup>β</sup>: %{z}<extra>%{fullData.name}</extra>",
            ),
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[5],  # dummy point to render the graph correctly
                mode="markers",
                showlegend=False,
                marker=dict(
                    color="rgba(0, 0, 0, 0)",
                ),
            ),
        ]
    )

    # Set the camera to show the end of both axes
    camera = dict(eye=dict(x=1.35, y=1.35, z=1.35))

    fig_initial.update_layout(
        scene=dict(
            xaxis=dict(
                title="C<sub>L</sub> (-)",
                range=[xy_lowerbound, active_selection["CLmax_ld"]],
            ),
            yaxis=dict(title="δ<sub>T</sub> (-)", range=[xy_lowerbound, 1]),
            zaxis=dict(title="σ<sup>β</sup> (-)", range=[0, 1]),
        ),
    )

    fig_initial.update_layout(
        scene_camera=camera,
        title={
            "text": f"Maximum altitude domain for {active_selection.full_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        },
    )

    mo.output.clear()
    return (fig_initial,)


@app.cell(hide_code=True)
def _(CL_slider, dT_slider, mo):
    mo.md(rf"""
    Here you can modify the control variables to understand how it affects the design: {mo.hstack([dT_slider, CL_slider])}
    """)
    return


@app.cell
def _(mass_stack):
    mass_stack
    return


@app.cell
def _(fig_initial):
    fig_initial
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Lagrangian function and KKT conditions

    The Lagrangian function combines the objective function with the inequality constraints using KKT multipliers ($\mu_j$).

    $$
    \begin{aligned}
    \mathcal{L}(C_L, \delta_T, \mu_1, \mu_2) = & \sigma^{\beta +1/2} + \mu_1 (C_L - C_{L_\mathrm{max}}) +\mu_2 (\delta_T - 1)\\
    =&\left[\frac{W^{3/2}}{\delta_T P_{a0}}\sqrt{\frac{2}{\rho_0 S}}\left(\frac{C_{D_0} + K C_L^2}{C_L^{3/2}}\right)\right] +\\
    & + \mu_1 \left(C_L - C_{L_\mathrm{max}}\right) + \\
    & + \mu_2 (\delta_T - 1) \\
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A necessary condition for an optimal solution of the optimization problem $(C_L^*, \delta_T^*)$ to exist, the multipliers $\lambda_1, \mu_1, \mu_2$ have to meet the following conditions:

    **A. Stationarity ($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \frac{\partial \mathcal{L}}{\partial C_L} = \frac{W^{3/2}}{\delta_T P_{a0}}\sqrt{\frac{2}{\rho_0 S}}\left(-\frac{3}{2}C_{D_0}C_L^{-5/2} + \frac{1}{2}KC_L^{-1/2}\right) + \mu_1= 0$

    3.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = - \frac{W^{3/2}}{\delta_T^2 P_{a0}}\sqrt{\frac{2}{\rho_0 S}}\left(\frac{C_{D_0} + K C_L^2}{C_L^{3/2}}\right) + \mu_2= 0$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $C_L - C_{L_\mathrm{max}} \le 0$
    4.  $\delta_T - 1 \le 0$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    5.  $\mu_1, \mu_2\ge 0$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraint have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

    6.  $\mu_1 (C_L - C_{L_\mathrm{max}}) = 0$
    7. $\mu_3 (\delta_T - 1) = 0$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## KKT analysis

    We can now proceed to systematically examine the conditions where various inequality constraints are active or inactive.
    """)
    return


@app.cell
def _(mo):
    titles_dict = {
        "### Interior & Lift limited solutions": "",
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
    CL_E,
    CL_P,
    CL_array,
    CLmax,
    OptimumGridView,
    active_selection,
    dT_array,
    drag_curve,
    drag_yrange,
    h_maxliftThrust,
    h_maxthrust,
    mach_trace,
    maxliftThrust_multiplier,
    maxthrust_multiplier,
    np,
    plot_utils,
    power_maxliftThrust_selected,
    power_maxthrust_selected,
    power_required_maxliftThrust,
    power_required_maxthrust,
    power_yrange,
    sigma_surface,
    stall_trace,
    tab_value,
    thrust_vector_maxliftThrust,
    thrust_vector_maxthrust,
    title_keys,
    true_maxliftThrust,
    true_maxthrust,
    velocity_CL_E,
    velocity_CL_P,
    velocity_maxliftThrust_CLarray,
    velocity_maxliftThrust_selected,
    velocity_maxthrust_CLarray,
    velocity_maxthrust_selected,
):
    if tab_value == title_keys[1]:
        configTraces_maxthrust = plot_utils.ConfigTraces(
            CL_array,
            dT_array,
            [np.nan],
            drag_curve,
            thrust_vector_maxthrust,
            power_required_maxthrust,
            thrust_vector_maxthrust * velocity_maxthrust_CLarray / 1e3,
            sigma_surface,
            velocity_maxthrust_CLarray,
            velocity_CL_P * maxthrust_multiplier,
            velocity_CL_E * maxthrust_multiplier,
            velocity_maxthrust_CLarray[-1],
            velocity_maxthrust_CLarray[-1],
            (drag_yrange, power_yrange, CLmax),
            (np.min(sigma_surface), 1),
            mach_trace,
            stall_trace,
        )

        # maxthrust graphics
        figure_optimum = OptimumGridView(
            configTraces_maxthrust,
            h_maxthrust,
            (velocity_maxthrust_CLarray, velocity_maxthrust_selected),
            (np.nan, power_maxthrust_selected),
            (h_maxthrust, 1, CL_P, true_maxthrust),
            f"Thrust-limited minimum power for {active_selection.full_name}",
            equality=True,
        )
    elif tab_value == title_keys[2]:
        configTraces_maxliftThrust = plot_utils.ConfigTraces(
            CL_array,
            dT_array,
            [np.nan],
            drag_curve,
            thrust_vector_maxliftThrust,
            power_required_maxliftThrust,
            thrust_vector_maxliftThrust * velocity_maxliftThrust_CLarray / 1e3,
            sigma_surface,
            velocity_maxliftThrust_CLarray,
            velocity_CL_P * maxliftThrust_multiplier,
            velocity_CL_E * maxliftThrust_multiplier,
            velocity_maxliftThrust_CLarray[-1],
            velocity_maxliftThrust_CLarray[-1],
            (drag_yrange, power_yrange, CLmax),
            (np.min(sigma_surface), 1),
            mach_trace,
            stall_trace,
        )

        # maxliftThrust graphics
        figure_optimum = OptimumGridView(
            configTraces_maxliftThrust,
            h_maxliftThrust,
            (velocity_maxliftThrust_CLarray, velocity_maxliftThrust_selected),
            (np.nan, power_maxliftThrust_selected),
            (h_maxliftThrust, 1, CL_E, true_maxliftThrust),
            f"Thrust-lift limited minimum power for {active_selection.full_name}",
            equality=True,
        )
    return (figure_optimum,)


@app.cell
def _(mo, tab_value, title_keys):
    if tab_value != title_keys[0]:
        mo.stop(True)
    mo.md(r"""
    ### _Interior solutions_

    Assuming that that $C_L < C_{L_\mathrm{max}}$ and $\delta_T < 1$ is equivalent to consider all inequality constraints as inactive.

    Therefore: $\mu_1,\mu_2=0$.

    It is clear from stationarity condition 2, that the equation cannot be solved for any value of $\delta_T$.

    It can be concluded that the maximum speed cannot be achieved in the interior of the domain.
    The minimum must lie on at least one of the boundaries defined by $C_L = C_{L_\mathrm{max}}$ or $\delta_T = 1$.

    Moreover, the stationarity condition 2 can be solved for a value of $\delta_T$ only when $\mu_2 \neq 0$, this means it also pointless to investigate the _max-lift condition_ as we would have $\mu_2 = 0$ again.
    """).callout()
    return


@app.cell
def _(figure_optimum, mass_stack, mo, tab_value, title_keys):
    if tab_value != title_keys[1]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ### _Thrust-limited minimum airspeed_

    $C_L < C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 = 0$

    $\delta_T=1 \quad \Rightarrow \quad \mu_2 > 0$

    From stationarity condition (1):

    $$
    C_L^*= \sqrt{\frac{3C_{D_0}}{K}}=C_{L_P}
    $$

    while stationarity condition (2) is always satisfied given $\delta_T = 1$.

    This condition is achievable only if $C_L^* \lt C_{L_\mathrm{max}}$, meaning that it stalls at lower speed than the airpseed for minimum power in steady level flight, for the same weight and altitude, and is therefore able to fly on the induced brach of the power performance diagram.

    The corresponding altitude is given by the density ratio:

    $$
    \displaystyle \sigma^* = \left(\frac{W^{3/2}}{P_{a0}E_{P}}\sqrt{\frac{2}{\rho_0 SC_{L_P}}}\right)^{\frac{1}{\beta+ 1/2}}
    $$

    which depends on the weight. We call this the "theoretical ceiling", by inspecting the equation for the density ratio, the lower the weight, the lower $\sigma$, and thus the higher the altitude $h$ of the ceiling.

    The operational condition is given by:

    $$
    \frac{W^{3/2}}{\sigma^{*^{\beta+1/2}}} = P_{a0}E_P \sqrt{\frac{1}{2}\rho_0 S C_{L_P}}
    $$
    """),
            mass_stack,
            figure_optimum.figure,
        ]
    ).callout()
    return


@app.cell
def _(atmos, np):
    def maxthrust_condition(
        W,
        S,
        Pa0,
        beta,
        CL_P,
        E_P,
        CLmax,
        sigma_min,
    ):
        sigma_optimum = (
            W ** (1.5) / Pa0 / E_P / (np.sqrt(atmos.rho0 * S * CL_P / 2))
        ) ** (1 / (beta + 0.5))

        dT = 1

        h_optimum = atmos.altitude(sigma_optimum)

        if CLmax < CL_P or sigma_optimum < sigma_min:
            return (
                h_optimum,
                dT,
                np.nan,
            )

        cond = 1

        return h_optimum, sigma_optimum, cond

    return (maxthrust_condition,)


@app.cell
def _(
    CL_P,
    CLmax,
    E_P,
    Pa0,
    S,
    W_selected,
    atmos,
    beta,
    drag_curve,
    maxthrust_condition,
    min_sigma,
    np,
    power_vector,
    rho_selected,
    sigma_selected,
    velocity_CL_P,
    velocity_CLarray,
):
    # Max lift Max thrust
    h_maxthrust, sigma_maxthrust, true_maxthrust = maxthrust_condition(
        W_selected,
        S,
        Pa0,
        beta,
        CL_P,
        E_P,
        CLmax,
        min_sigma,
    )

    maxthrust_multiplier = np.sqrt(rho_selected / (atmos.rho0 * sigma_maxthrust))

    velocity_maxthrust_CLarray = velocity_CLarray * maxthrust_multiplier
    velocity_maxthrust_selected = velocity_CL_P * maxthrust_multiplier
    thrust_vector_maxthrust = (
        (power_vector * (sigma_maxthrust / sigma_selected) ** beta)
        / velocity_maxthrust_CLarray
    ) * 1e3

    power_required_maxthrust = drag_curve * velocity_maxthrust_CLarray / 1e3
    power_maxthrust_selected = W_selected / E_P * velocity_maxthrust_selected
    return (
        h_maxthrust,
        maxthrust_multiplier,
        power_maxthrust_selected,
        power_required_maxthrust,
        thrust_vector_maxthrust,
        true_maxthrust,
        velocity_maxthrust_CLarray,
        velocity_maxthrust_selected,
    )


@app.cell
def _(figure_optimum, mass_stack, mo, tab_value, title_keys):
    if tab_value != title_keys[2]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ### _Thrust- and lift-limited minimum speed_

    $\delta_T = 1 \quad \Rightarrow \quad \mu_3 > 0$

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 > 0$.

    From the stationary conditions (1):

    $$
    \mu_1 = \frac{W^{3/2}}{\delta_T P_{a0}}\sqrt{\frac{2}{\rho_0 S}}\left(\frac{3}{2}C_{D_0}C_L^{-5/2} - \frac{1}{2}KC_L^{-1/2}\right) \gt 0 \quad \Longleftrightarrow \quad C_{L_\mathrm{max}} \lt \sqrt{\frac{3C_{D_0}}{K}} = C_{L_{E}}
    $$

    In this case the aircraft stalls at a higher speed than the one for minimum required power in steady level flight, for the same altitude and weight.

    The corresponding altitude is given by the density ratio:

    $$
    \displaystyle \sigma^* = \left(\frac{W^{3/2}}{P_{a0}E_{S}}\sqrt{\frac{2}{\rho_0 SC_{L_\mathrm{max}}}}\right)^{\frac{1}{\beta+ 1/2}}
    $$

    While the operational condition is given by:

    $$
    \frac{W^{3/2}}{\sigma^{*^{\beta+1/2}}} = P_{a0}E_S \sqrt{\frac{1}{2}\rho_0 S C_{L_\mathrm{max}}}
    $$
    """),
            mass_stack,
            figure_optimum.figure,
        ]
    ).callout()
    return


@app.cell
def _(CL_P, atmos, np):
    def maxliftThrust_condition(
        W,
        S,
        Pa0,
        beta,
        CLmax,
        E_S,
        sigma_min,
    ):
        sigma_optimum = (
            W ** (1.5) / Pa0 / E_S / (np.sqrt(atmos.rho0 * S * CLmax / 2))
        ) ** (1 / (beta + 0.5))

        dT = 1

        h_optimum = atmos.altitude(sigma_optimum)

        if CLmax > CL_P or sigma_optimum < sigma_min:
            return (
                h_optimum,
                dT,
                np.nan,
            )

        cond = 1

        return h_optimum, sigma_optimum, cond

    return (maxliftThrust_condition,)


@app.cell
def _(
    CLmax,
    E_P,
    E_S,
    Pa0,
    S,
    W_selected,
    atmos,
    beta,
    drag_curve,
    maxliftThrust_condition,
    min_sigma,
    np,
    power_vector,
    rho_selected,
    sigma_selected,
    velocity_CL_P,
    velocity_CLarray,
):
    # Max lift Max thrust
    h_maxliftThrust, sigma_maxliftThrust, true_maxliftThrust = maxliftThrust_condition(
        W_selected,
        S,
        Pa0,
        beta,
        CLmax,
        E_S,
        min_sigma,
    )

    maxliftThrust_multiplier = np.sqrt(
        rho_selected / (atmos.rho0 * sigma_maxliftThrust)
    )

    velocity_maxliftThrust_CLarray = velocity_CLarray * maxliftThrust_multiplier
    velocity_maxliftThrust_selected = velocity_CL_P * maxliftThrust_multiplier
    thrust_vector_maxliftThrust = (
        (power_vector * (sigma_maxliftThrust / sigma_selected) ** beta)
        / velocity_maxliftThrust_CLarray
    ) * 1e3

    power_required_maxliftThrust = drag_curve * velocity_maxliftThrust_CLarray / 1e3
    power_maxliftThrust_selected = W_selected / E_P * velocity_maxliftThrust_selected
    return (
        h_maxliftThrust,
        maxliftThrust_multiplier,
        power_maxliftThrust_selected,
        power_required_maxliftThrust,
        thrust_vector_maxliftThrust,
        true_maxliftThrust,
        velocity_maxliftThrust_CLarray,
        velocity_maxliftThrust_selected,
    )


@app.cell
def _(mo):
    mo.md(r"""
    Now after deriving all the optima for each condition we can summarize the flight envelopes in one graph, as shown below. Experiment with the weight of the aircrarft to understand how the theoretical ceiling for minimum power moves in the graph.
    """)
    return


@app.cell
def _():
    _defaults.nav_footer(
        before_file="MaxAltitude_Jet.py",
        before_title="Maximum Simplified Jet",
        above_file="MinSpeed.py",
        above_title="Maximum Altitude Homepage",
        above_before=False,
    )
    return


if __name__ == "__main__":
    app.run()
