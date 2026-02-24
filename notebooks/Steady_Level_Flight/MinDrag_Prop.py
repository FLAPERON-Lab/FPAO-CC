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
    data_dir = str(mo.notebook_location().parent / "public" / "AircraftDB_Standard.csv")


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
        E_max,
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
    CL_array,
    CLmax,
    E_array,
    K,
    MTOM,
    OEM,
    S,
    a_0,
    atmos,
    dT_array,
    h_array,
    m_slider,
    np,
    plot_utils,
    rho_array,
):
    # Define variables, this cell runs every time the mass slider is run
    W_selected = (OEM + (MTOM - OEM) * m_slider.value) * atmos.g0  # Netwons
    drag_curve = W_selected / E_array

    drag_surface = np.broadcast_to(
        drag_curve[np.newaxis, :],  # Shape: (101, 1)
        (len(CL_array), len(dT_array)),  # Target shape: (101, 101)
    )

    velocity_stall_harray = np.sqrt(2 * W_selected / (rho_array * S * CLmax))

    # Visual computations
    stall_trace = plot_utils.create_stall_trace(h_array, velocity_stall_harray)

    CL_a0 = W_selected * 2 / (atmos.rho0 * S * a_0**2)

    drag_yrange = 0.2 * W_selected * (CD0 + K * CL_a0**2) / CL_a0
    power_yrange = 0.5 * drag_yrange * a_0 / 1e3

    min_colorbar = np.min(drag_curve)
    max_colorbar = min_colorbar * 2
    zcolorbar = (min_colorbar, max_colorbar)
    return (
        W_selected,
        drag_curve,
        drag_surface,
        drag_yrange,
        max_colorbar,
        min_colorbar,
        power_yrange,
        stall_trace,
        velocity_stall_harray,
        zcolorbar,
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

    power_available = np.repeat(power_scalar, meshgrid_n)
    return (
        h_selected,
        idx_h_selected,
        power_available,
        power_scalar,
        rho_selected,
        sigma_selected,
    )


@app.cell
def _(
    CL_E,
    CL_P,
    CL_array,
    CLmax,
    E_array,
    S,
    W_selected,
    dT_array,
    drag_curve,
    drag_surface,
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
    zcolorbar,
):
    # Computation only cell, indexing happens in another cell
    velocity_CLarray = np.sqrt(2 * W_selected / (rho_selected * S * CL_array))
    velocity_CL_E = velocity_CLarray[-1] * np.sqrt(CLmax / CL_E)
    velocity_CL_P = velocity_CLarray[-1] * np.sqrt(CLmax / CL_P)

    thrust_vector = power_scalar / velocity_CLarray * 1e3
    power_required = drag_curve * velocity_CLarray / 1e3

    constraint = W_selected / E_array / thrust_vector

    range_performance_diagrams = (drag_yrange, power_yrange, CLmax, 250)

    # Create graphic traces
    configTraces = plot_utils.ConfigTraces(
        CL_array,
        dT_array,
        constraint,
        drag_curve,
        thrust_vector,
        power_required,
        power_available,
        drag_surface,
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
        range_performance_diagrams,
        velocity_CL_E,
        velocity_CL_P,
        velocity_CLarray,
    )


@app.cell
def _(drag_curve, idx_CL_selected):
    drag_selected = drag_curve[idx_CL_selected]
    return (drag_selected,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Minimum drag: simplified piston propeller aircraft

    $$
    \begin{aligned}
        \min_{C_L, \delta_T}
        & \quad D = \frac{1}{2}\rho V^2S\left(C_{D_0} + K C_L^2\right) \\
        \text{subject to}
        & \quad c_1^\mathrm{eq} = L-W = \frac{1}{2}\rho V^2 S C_L - W = 0 \\
        & \quad c_2^\mathrm{eq} = T-D = \delta_T T_a(V,h) - \frac{1}{2} \rho V^2 S (C_{D_0}+K C_L^2) =0 \\
        \text{for }
        & \quad C_L \in [0, C_{L_\mathrm{max}}] \\
        & \quad \delta_T \in [0, 1] \\
        \text{with }
        & \quad T_a(V,h) = \frac{P_a(h)}{V} =\frac{P_{a0}\sigma^\beta}{V} \\
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## KKT formulation

    As shown in the simplified jet case, we express $V$ from $c_1^\mathrm{eq}$ and substitute it out of the entire problem to eliminate it. The KKT formulation thus becomes:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$
    \begin{aligned}
    \min_{C_L, \delta_T}
    & \quad D = W \frac{C_{D_0} + K C_L^2}{C_L} = \frac{W}{E} \\
    \text{subject to}
    & \quad g_1 = \frac{T}{W} - \frac{1}{E}  = \delta_T P_{a0}\sigma^\beta\sqrt{\frac{\rho S C_L}{2W}} - W\frac{C_{D_0} +K C_L^2}{C_L} = 0 \\
    & \quad h_1 = C_L - C_{L_\mathrm{max}} \le 0 \\
    & \quad h_2 = -C_L \le 0 \\
    & \quad h_3 = \delta_T - 1 \le 0 \\
    & \quad h_4 = -\delta_T \le 0 \\
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below you can see the graph of the domain $0 \leq C_L \leq C_{L_{\mathrm{max}}}$ and $0 \leq \delta_T \leq 1$, with the surface $D$ and the contraint $g_1$ in red. Choose a simplified jet aircraft of your liking in the database below.
    """)
    return


@app.cell
def _(ac_table, mo):
    mo.lazy(ac_table)
    return


@app.cell
def _(CL_slider, dT_slider, mo):
    mo.md(rf"""
    Here you can modify the control variables to understand how it affects the design: {mo.hstack([dT_slider, CL_slider])}
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
    drag_selected,
    drag_surface,
    go,
    max_colorbar,
    min_colorbar,
    mo,
    xy_lowerbound,
):
    # Create go.Figure() object
    fig_initial = go.Figure()

    # Minimum velocity surface
    fig_initial.add_traces(
        [
            go.Surface(
                x=CL_array,
                y=dT_array,
                z=drag_surface,
                opacity=0.9,
                name="Drag",
                colorscale="viridis",
                cmin=min_colorbar,
                cmax=max_colorbar,
                colorbar={"title": "Drag (N)"},
            ),
            go.Scatter3d(
                x=CL_array,
                y=constraint,
                z=drag_surface[0],
                opacity=1,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
                name="g1 constraint",
                textfont=dict(size=14, family="Arial"),
            ),
            go.Scatter3d(
                x=[CL_array[-25]],
                y=[constraint[-25]],
                z=[drag_surface[0, -25] + 2e3],
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
                z=[drag_selected],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=3,
                    color="white",
                    symbol="circle",
                ),
                name="Design Point",
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub> : %{y}<br>D: %{z}<extra>%{fullData.name}</extra>",
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
            zaxis=dict(title="D (N)", range=[0, max_colorbar]),
        ),
    )
    # Set the camera to show the end of both axes
    camera = dict(eye=dict(x=1.35, y=1.35, z=1.35))

    fig_initial.update_layout(
        scene_camera=camera,
        title={
            "text": f"Minimum Drag domain for {active_selection.full_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        },
    )

    mo.output.clear()
    return (fig_initial,)


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell
def _(fig_initial):
    fig_initial
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Lagrangian function and KKT conditions

    The Lagrangian function combines the objective function with equality constraints using Lagrange multipliers ($\lambda_i$) and inequality constraints using KKT multipliers ($\mu_j$).

    $$
    \begin{aligned}
    \mathcal{L}(C_L, \delta_T, \lambda_1, \mu_1, \mu_2, \mu_3, \mu_4) =
    \quad W\frac{C_{D_0} + K C_L^2}{C_L}
    & + \\
    & + \lambda_1 \left[\delta_T P_{a0}\sigma^\beta \sqrt{\frac{\rho S C_L}{2W}} - W\frac{C_{D_0} +K C_L^2}{C_L}\right] + \\
    & + \mu_1 (C_L - C_{L_\mathrm{max}}) + \\
    & + \mu_2 (-C_L) + \\
    & + \mu_3 (\delta_T - 1) +\\
    & + \mu_4 (-\delta_T)
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The multipliers $\lambda_1, \mu_1, \mu_2, \mu_3, \mu_4$ have to meet the following conditions for an optimal solution of the optimization problem $(C_L^*, \delta_T^*)$ to exist.

    **A. Stationarity conditions($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \begin{aligned}\frac{\partial \mathcal{L}}{\partial C_L} & = W \frac{K C_L^2 - C_{D_0}}{C_L^2} + \lambda_1 \left( \frac{1}{2} \delta_T P_{a0}\sigma^\beta \sqrt{\frac{\rho S}{2WC_L}} - W \frac{K C_L^2 - C_{D_0}}{C_L^2} \right) + \mu_1 - \mu_2 \\
    & = W\frac{K C_L^2 - C_{D_0}}{C_L^2} (1 -\lambda_1) +  \frac{1}{2} \lambda_1\delta_T P_{a0}\sigma^\beta \sqrt{\frac{\rho S}{2WC_L}} +\mu_1 - \mu_2 = 0 \end{aligned}$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1 P_{a0} \sigma^\beta \sqrt{\frac{\rho S C_L}{2W}}+\mu_3-\mu_4 = 0$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $\displaystyle \delta_T P_{a0}\sigma^\beta\sqrt{\frac{\rho S C_L}{2W}} - W\frac{C_{D_0} +K C_L^2}{C_L} = 0$
    4.  $C_L - C_{L_\mathrm{max}} \le 0$
    5.  $-C_L \le 0$
    6.  $\delta_T - 1 \le 0$
    7.  $-\delta_T \le 0$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    8.  $\mu_1, \mu_2, \mu_3, \mu_4 \ge 0$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraints have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

    9.  $\mu_1 (C_L - C_{L_\mathrm{max}}) = 0$
    10. $\mu_2 (-C_L) = 0$
    11. $\mu_3 (\delta_T - 1) = 0$
    12. $\mu_4 (-\delta_T) = 0$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It is evident that $\mu_2$ and $\mu_4$ can never be active, as we would have an unfeasible situation ($C_L = \delta_T = 0$). In other words, strictly for aircraft flight: $C_L \gt 0$ and $\delta_T \gt 0$. Therefore, we can simplify the analysis by setting these two KKT multipliers to zero:

    $$
    \begin{aligned}
    \mu_2 = \mu_4 = 0
    \end{aligned}
    $$

    We can now rewrite the new conditions to simplify the problem. We will refer to these simplified conditions for the entire notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Simplified conditions**

    1. $\displaystyle W\frac{K C_L^2 - C_{D_0}}{C_L^2} (1 -\lambda_1) +  \frac{1}{2} \lambda_1\delta_T P_{a0}\sigma^\beta \sqrt{\frac{\rho S}{2WC_L}} +\mu_1 = 0$
    2. $\displaystyle \lambda_1 P_{a0} \sigma^\beta \sqrt{\frac{\rho S C_L}{2W}}+\mu_3 = 0$
    3. $\displaystyle \delta_T P_{a0}\sigma^\beta\sqrt{\frac{\rho S C_L}{2W}} - W\frac{C_{D_0} +K C_L^2}{C_L} = 0$
    4. $C_L - C_{L_\mathrm{max}} \le 0$
    5. $\delta_T - 1 \le 0$
    6. $\mu_1, \mu_3 \ge 0$
    7. $\mu_1 (C_L - C_{L_\mathrm{max}}) = 0$
    8. $\mu_3 (\delta_T - 1) = 0$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## KKT analysis

    We can now systematically examine the conditions where various inequality constraints are active or inactive.
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
    CLopt_interior,
    CLopt_maxlift,
    E_array,
    OptimumGridView,
    Pa0,
    W_selected,
    active_selection,
    beta,
    configTraces,
    dT_array,
    dTopt_interior,
    dTopt_maxlift,
    drag_curve,
    drag_maxliftThrust_harray,
    drag_maxliftThrust_selected,
    drag_maxthrust_harray,
    drag_maxthrust_selected,
    drag_surface,
    drag_yrange,
    h_interior_array,
    h_maxliftThrust,
    h_maxlift_array,
    h_maxthrust_array,
    h_selected,
    mach_trace,
    maxliftThrust_multiplier,
    np,
    plot_utils,
    power_interior_harray,
    power_interior_selected,
    power_maxlift_harray,
    power_maxlift_selected,
    power_required_maxliftThrust,
    power_yrange,
    range_performance_diagrams,
    sigma_maxliftThrust,
    stall_trace,
    tab_value,
    thrust_vector_maxliftThrust,
    title_keys,
    true_interior,
    true_maxlift,
    true_maxliftThrust,
    true_maxthrust,
    velocity_CL_E,
    velocity_CL_P,
    velocity_interior_harray,
    velocity_interior_selected,
    velocity_maxliftThrust_CLarray,
    velocity_maxliftThrust_selected,
    velocity_maxlift_harray,
    velocity_maxlift_selected,
    velocity_maxthrust_harray,
    velocity_maxthrust_selected,
    zcolorbar,
):
    if tab_value == title_keys[0]:
        # Interior graphics
        figure_optimum = OptimumGridView(
            configTraces,
            h_selected,
            (velocity_interior_harray, velocity_interior_selected),
            (power_interior_harray, power_interior_selected),
            (h_interior_array, dTopt_interior, CLopt_interior, true_interior),
            f"Interior minimum power for {active_selection.full_name}",
        )

    elif tab_value == title_keys[1]:
        # maxlift graphics
        figure_optimum = OptimumGridView(
            configTraces,
            h_selected,
            (velocity_maxlift_harray, velocity_maxlift_selected),
            (power_maxlift_harray, power_maxlift_selected),
            (h_maxlift_array, dTopt_maxlift, CLopt_maxlift, true_maxlift),
            f"Lift-limited minimum power for {active_selection.full_name}",
        )

    elif tab_value == title_keys[2]:
        # Maxthrust graphics
        figure_optimum = OptimumGridView(
            configTraces,
            h_selected,
            (velocity_maxthrust_harray, velocity_maxthrust_selected),
            (drag_maxthrust_harray, drag_maxthrust_selected),
            (h_maxthrust_array, true_maxthrust, CL_maxthrust_selected, true_maxthrust),
            f"Thrust-limited minimum power for {active_selection.full_name}",
        )

    elif tab_value == title_keys[3]:
        constraint_maxliftThrust = (
            W_selected
            / E_array
            / Pa0
            / sigma_maxliftThrust**beta
            * velocity_maxliftThrust_selected
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
            drag_surface,
            velocity_maxliftThrust_CLarray,
            velocity_CL_P * maxliftThrust_multiplier,
            velocity_CL_E * maxliftThrust_multiplier,
            velocity_maxliftThrust_selected,
            velocity_maxliftThrust_selected,
            (drag_yrange, power_yrange / 1e3, CLmax),
            zcolorbar,
            mach_trace,
            stall_trace,
        )

        # Maxliftthrust graphics
        figure_optimum = OptimumGridView(
            configTraces_maxliftThrust,
            h_selected,
            (velocity_maxliftThrust_CLarray, velocity_maxliftThrust_selected),
            (drag_maxliftThrust_harray, drag_maxliftThrust_selected),
            (h_maxliftThrust, 1, true_maxliftThrust, np.nan),
            f"Thrust-lift limited minimum drag for {active_selection.full_name}",
            equality=True,
        )

    figure_optimum.update_axes_ranges(range_performance_diagrams)
    return (figure_optimum,)


@app.cell(hide_code=True)
def _(figure_optimum, mo, tab_value, title_keys, variables_stack):
    if tab_value != title_keys[0]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""

    ### _Interior solutions_

    Assuming that that $0 < C_L < C_{L_\mathrm{max}}$ and $0 < \delta_T < 1$ is equivalent to consider all inequality constraints as inactive.

    Therefore: $\mu_1, \mu_3 =0$.

    From stationarity condition (2): $\lambda_1 = 0$.

    From stationarity condition (1), it is possible to obtain the value of $C_L^*$ for minimum drag.

    $$
    C_L^* = \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$

    Notice how the optimal $C_L^*$ has the **same value** for maximum aerodynamic efficiency, or maximum $C_L/C_D$, for
    $0\lt C_L \lt  C_{L_{max}}$ and $0 \lt \delta_T \lt 1$, as shown in [aerodynamic efficiency](/?file=Steady_Level_Flight/MinDrag.py).

    The corresponding airspeed is

    $$
    \displaystyle V^* = V_E = \sqrt{\frac{2}{\rho}\frac{W}{S}\frac{1}{C_{L_E}}} = \sqrt{\frac{2}{\rho}\frac{W}{S}}\sqrt[4]{\frac{K}{C_{D_0}}}
    $$

    The corresponding $\delta_T^*$ is found by solving the primal feasibility constraint (3) and using $C_L = C_L^*$.


    $$
    \delta_T^* = \frac{W}{E_\mathrm{max}}\frac{V_E}{P_{a0}\sigma^\beta} = \frac{W^{3/2}}{\sigma^{\beta+1/2}}\frac{1}{E_\mathrm{max}P_{a0}}\sqrt{\frac{2}{\rho_0 S C_{L_E}}}
    $$

    This value is compliant with the primal feasibility constraint (5) for:

    $$
    \delta_T^* < 1 \quad \Leftrightarrow
    \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}}
    \lt
    P_{a0}E_\mathrm{max}\sqrt{\frac{\rho_0 S C_{L_E}}{2}}
    $$

    The corresponding minimum drag is found by first computing $C_D^*=C_{D_E}$:

    $$
    C_D^* = C_{D_E} = 2C_{D_0} \\
    D_{\mathrm{min}}^* =  \frac{1}{2}\rho {V_E}^2 S C_{D_E}=
    2W\sqrt{KC_{D_0}}=\frac{W}{E_\mathrm{max}}
    $$

    This is the same result as in the simplified jet analysis!

    This concludes the analysis for the minimum drag of a simplified propeller aircraft in the domain's interior. Below is a summary of the optima:

    $$
    \boxed{C_L^* = \sqrt{\frac{C_{D_0}}{K}}}, \quad \boxed{\delta_T^*=\frac{W^{3/2}}{\sigma^{\beta+1/2}}\frac{1}{E_\mathrm{max}P_{a0}}\sqrt{\frac{2}{\rho_0 S C_{L_E}}}}, \quad \text{for} \quad C_L^* \lt C_{L_\mathrm{max}}\quad \text{and} \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}}
    \lt
    P_{a0}E_\mathrm{max}\sqrt{\frac{\rho_0 S C_{L_E}}{2}}
    $$

    With the optimal value for minimum drag:

    $$
    D_{\mathrm{min}}^* = 2W\sqrt{KC_{D_0}}=\frac{W}{E_\mathrm{max}}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.

    """),
            variables_stack,
            figure_optimum.figure,
        ]
    ).callout()
    return


@app.cell
def _(atmos, np):
    def interior_condition(
        W,
        h_selected,
        S,
        Pa0,
        beta,
        CL_E,
        E_max,
        velocity_CLE,
        power_available,  # input scalar
        CLmax,
        min_sigma,
        sigma_selected,
        h_array,
    ):
        sigma_interior = (
            W ** (1.5) / Pa0 / E_max / (np.sqrt(atmos.rho0 * S * CL_E / 2))
        ) ** (1 / (beta + 0.5))

        dT_interior = W / E_max * velocity_CLE / power_available / 1e3

        if CL_E > CLmax or sigma_interior <= min_sigma:
            return np.array([np.nan]), dT_interior, np.nan, np.nan

        h_interior = atmos.altitude(sigma_interior)

        h_interior_array = h_array[h_array < h_interior]

        h_min = h_interior_array.min()
        h_max = h_interior_array.max()
        cond = 1 if h_min <= h_selected <= h_max else np.nan
        return h_interior_array, dT_interior, CL_E, cond

    return (interior_condition,)


@app.cell
def _(
    CL_E,
    CLmax,
    E_max,
    Pa0,
    S,
    W_selected,
    atmos,
    beta,
    h_array,
    h_selected,
    interior_condition,
    min_sigma,
    np,
    power_scalar,
    sigma_selected,
    velocity_CL_E,
):
    h_interior_array, dTopt_interior, CLopt_interior, true_interior = (
        interior_condition(
            W_selected,
            h_selected,
            S,
            Pa0,
            beta,
            CL_E,
            E_max,
            velocity_CL_E,
            power_scalar,  # input scalar
            CLmax,
            min_sigma,
            sigma_selected,
            h_array,
        )
    )

    velocity_interior_selected = velocity_CL_E * true_interior
    velocity_interior_harray = np.sqrt(
        2 * W_selected / (CL_E * S * atmos.rho(h_interior_array))
    )

    power_interior_harray = W_selected / E_max * velocity_interior_harray
    power_interior_selected = W_selected / E_max * velocity_interior_selected
    return (
        CLopt_interior,
        dTopt_interior,
        h_interior_array,
        power_interior_harray,
        power_interior_selected,
        true_interior,
        velocity_interior_harray,
        velocity_interior_selected,
    )


@app.cell
def _(figure_optimum, mo, tab_value, title_keys, variables_stack):
    if tab_value != title_keys[1]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ###_Lift-limited minimum drag (stall boundary)_

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 > 0$

    $\delta_T < 1 \quad \Rightarrow \quad \mu_3 = 0$

    From stationarity condition (2): $\lambda_1 = 0$.

    From stationarity condition (1): $\displaystyle \mu_1 = W\frac{C_{D_0} - KC_{L_\mathrm{max}}^2}{C_{L_\mathrm{max}}^2} \gt 0$, which results in:

    $$
    C_{L_\mathrm{max}} < \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$

    This means the aircraft is able to achieve minimum drag at its maximum lift coefficient only if the maximum lift coefficient is lower than the one for maximum aerodynamic efficiency.
    In this case, the aircraft would stall at higher speeds than the one for maximum aerodynamic efficiency, and would therefore only be able to fly on the right branch of the drag performance diagram.

    The corresponding throttle is obtained from the primal feasibility constraint (3):

    $$
    \delta_T^* = \frac{W}{E_S}\frac{V_S}{P_{a0}\sigma^\beta} = \frac{W^{3/2}}{\sigma^{\beta+1/2}}\frac{1}{P_{a0}E_S}\sqrt{\frac{2}{\rho_0 S C_{L_\mathrm{max}}}}
    $$

    By setting $\delta_T^* \lt 1$ obtain the operational condition for which minimum drag can be achieved:


    $$
    \frac{W^{3/2}}{\sigma^{\beta+1/2}} \lt P_{a0}E_S \sqrt{\frac{\rho_0SC_{L_\mathrm{max}}}{2}}
    $$

    Finally, the value for the objective function $D$ can be found:

    $$
    D^*_\mathrm{min} = \frac{1}{2}\rho V^2 S C_D = \frac{1}{2}\rho \left(\frac{W}{S}\frac{2}{\rho}\frac{1}{C_{L_\mathrm{max}}}\right)S (C_{D_0} + KC_{L_\mathrm{max}}^2) =  \frac{W}{E_S}
    $$

    This concludes the analysis for the minimum drag of a simplified propeller aircraft in the lift-limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = C_{L_\mathrm{max}}}, \quad \boxed{\delta_T^*=\frac{W^{3/2}}{\sigma^{\beta+1/2}}\frac{1}{E_SP_{a0}}\sqrt{\frac{2}{\rho_0 S C_{L_\mathrm{max}}}}}, \quad \text{for} \quad C_L^* \lt \sqrt{\frac{C_{D_0}}{K}} \quad \text{and} \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}}\lt P_{a0}E_S\sqrt{\frac{\rho_0 S C_{L_\mathrm{max}}}{2}}
    $$

    With the optimal value for minimum drag:

    $$
    D_{\mathrm{min}}^* =\frac{W}{E_S}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.
    """),
            variables_stack,
            figure_optimum.figure,
        ]
    ).callout()
    return


@app.cell
def _(atmos, np):
    def maxlift_condition(
        W,
        h_selected,
        S,
        Pa0,
        beta,
        CL_E,
        E_S,
        velocity_stall,
        power_available,  # input scalar
        CLmax,
        min_sigma,
        sigma_selected,
        h_array,
    ):
        sigma_maxlift = (
            W ** (1.5) / Pa0 / E_S / (np.sqrt(atmos.rho0 * S * CLmax / 2))
        ) ** (1 / (beta + 0.5))

        dT_maxlift = W / E_S * velocity_stall / power_available / 1e3

        if CL_E <= CLmax or sigma_maxlift <= min_sigma:
            return np.array([np.nan]), dT_maxlift, np.nan, np.nan

        h_maxlift = atmos.altitude(sigma_maxlift)
        h_maxlift_array = h_array[h_array < h_maxlift]

        h_min = h_maxlift_array.min()
        h_max = h_maxlift_array.max()
        cond = 1 if h_min <= h_selected <= h_max else np.nan
        return h_maxlift_array, dT_maxlift, CLmax, cond

    return (maxlift_condition,)


@app.cell
def _(
    CL_E,
    CLmax,
    E_S,
    E_max,
    Pa0,
    S,
    W_selected,
    atmos,
    beta,
    h_array,
    h_selected,
    maxlift_condition,
    min_sigma,
    np,
    power_available,
    sigma_selected,
    velocity_CL_E,
    velocity_stall_harray,
):
    h_maxlift_array, dTopt_maxlift, CLopt_maxlift, true_maxlift = maxlift_condition(
        W_selected,
        h_selected,
        S,
        Pa0,
        beta,
        CL_E,
        E_S,
        velocity_stall_harray,
        power_available,  # input scalar
        CLmax,
        min_sigma,
        sigma_selected,
        h_array,
    )

    velocity_maxlift_selected = velocity_CL_E * true_maxlift
    velocity_maxlift_harray = np.sqrt(
        2 * W_selected / (CL_E * S * atmos.rho(h_maxlift_array))
    )

    power_maxlift_harray = W_selected / E_max * velocity_maxlift_harray
    power_maxlift_selected = W_selected / E_max * velocity_maxlift_selected
    return (
        CLopt_maxlift,
        dTopt_maxlift,
        h_maxlift_array,
        power_maxlift_harray,
        power_maxlift_selected,
        true_maxlift,
        velocity_maxlift_harray,
        velocity_maxlift_selected,
    )


@app.cell
def _(figure_optimum, mo, tab_value, title_keys, variables_stack):
    if tab_value != title_keys[2]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ###_Thrust-limited minimum drag_

    $C_L \lt C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 = 0$

    $\delta_T = 1 \quad \Rightarrow \quad \mu_3 \gt 0$

    From stationarity condition (2):

    $$
    \mu_3 = -\lambda_1 P_{a0}\sigma^\beta \sqrt{\frac{\rho S C_L}{2W}} \gt 0 \quad \Rightarrow \quad \lambda_1 \lt 0
    $$

    Thus, $\lambda_1 \lt 0$ and $(\lambda_1 - 1) \lt 0$. From stationarity condition (1):

    $$
    \frac{\lambda_1 - 1}{\lambda_1} = \frac{C_L^2}{KC_L^2 - C_{D_0}}\frac{P_{a0}\sigma^\beta}{2 W}\sqrt{\frac{\rho S}{2WC_L}} > 0 \quad \Rightarrow \quad
    KC_L^2-C_{D_0} \gt 0, \quad \Rightarrow \quad C_L\gt \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$

    This means that $C_L^*$ is bounded by both $C_{L_E}$ and $C_{L_\mathrm{max}}$:


    $$
    C_{L_E} \lt C_L^* \lt C_{L_\mathrm{max}}
    $$

    The actual value of $C_L^*$ can be found from the primal feasibility constraint (3), with $\delta_T^* = 1$.
    Unfortunately, solving this cannot be done analytically, and a closed-form expression for $C_L^*$ cannot be found.
    Thus we proceed to a numerical solution for $C_L^*$, which also yields the operational condition:

    $$
    \text{solve numerically for } C_L^* :\quad  P_{a0}\sigma^\beta \sqrt{\frac{\rho S C_L}{2W}} - W \frac{C_{D_0} + K C_L^2}{C_L} = 0
    $$

    Thus the results from the thrust-limited analysis yield:

    $$
    \boxed{\delta_T = 1}, \quad \boxed{C_L^* = \text{numerical}}, \quad {\frac{W^{3/2}}{\sigma^{\beta + 1/2}} \gt \text{numerical}}, \quad {C_{L_E}\lt C_L \lt C_{L_\mathrm{max}}}
    $$
    """),
            variables_stack,
            figure_optimum.figure,
        ]
    ).callout()
    return


@app.cell
def _(
    CD0,
    K,
    Pa0,
    S,
    W_selected,
    atmos,
    beta,
    h_array,
    h_selected,
    np,
    sigma_array,
):
    def maxthrust_solver(W, h):
        sigma = atmos.rhoratio(h)
        C1 = Pa0 * sigma**beta * np.sqrt(atmos.rho(h) * S / (2 * W))

        # define H(s) and its derivative
        def H(s):
            # H(s) = C1 * s^(3/2) - W * (CD0 + K * s^2)
            return C1 * s**1.5 - W * (CD0 + K * s**2)

        def dHds(s):
            # dH/ds = (3/2)*C1*s^(1/2) - 2*W*K*s
            return 1.5 * C1 * np.sqrt(s) - 2 * W * K * s

        return H, dHds

    def maxthrust_condition(CD0, K, CLstar, CLmax):
        # condition = (
        #     K * CLstar**2
        #     - Pa0 * sigma ** (beta + 0.5) / (2 * W_selected**1.5) * np.sqrt(atmos.rho0 * S / 2) * CLstar**1.5
        #     - CD0
        #     < 0
        # ) & (CLstar < CLmax)

        sigma_min = (
            (2 * W_selected**1.5)
            * (K * CLstar**2 - CD0)
            * np.sqrt(2 / (atmos.rho0 * S))
            / (Pa0 * CLstar**1.5) ** (1 / (beta + 0.5))
        )

        h_maxthrust_array = h_array[(sigma_array < sigma_min)]

        CLopt = CLstar[(sigma_array < sigma_min) & (CLstar < CLmax)]

        cond = 1 if h_selected in h_maxthrust_array else np.nan

        return h_maxthrust_array, CLopt, cond

    return maxthrust_condition, maxthrust_solver


@app.cell
def _(CL_E, CLmax, W_selected, h_array, maxthrust_solver, np, root_scalar):
    CL_maxthrust_star = []

    for h in h_array:
        H, dHds = maxthrust_solver(W_selected, h)

        # Newton’s method — requires derivative, and one initial guess
        sol = root_scalar(
            H,
            fprime=dHds,
            bracket=[0, CLmax],
            x0=CL_E + 0.01,  # initial guess (should be near the root)
            method="newton",
            # xtol=1e-6,
            # rtol=1e-6,
            maxiter=1000,
        )

        s_root = sol.root

        CL_maxthrust_star.append(s_root)

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
    h_selected,
    maxthrust_condition,
    np,
):
    h_maxthrust_array, CLopt_maxthrust, true_maxthrust = maxthrust_condition(
        CD0, K, CL_maxthrust_star, CLmax
    )

    CL_maxthrust_selected = (
        CLopt_maxthrust[h_selected == h_maxthrust_array][0]
        if h_maxthrust_array.size > 0 and np.any(h_selected == h_maxthrust_array)
        else np.nan
    )

    velocity_maxthrust_harray = np.sqrt(
        2 * W_selected / (atmos.rho(h_maxthrust_array) * CLopt_maxthrust * S)
    )
    velocity_maxthrust_selected = np.sqrt(
        2 * W_selected / (atmos.rho(h_selected) * CL_maxthrust_selected * S)
    )

    dTopt_maxthrust = 1

    drag_maxthrust_harray = (
        W_selected * (CD0 + K * CLopt_maxthrust**2) / CLopt_maxthrust
    )

    drag_maxthrust_selected = (
        W_selected * (CD0 + K * CL_maxthrust_selected**2) / CL_maxthrust_selected
    )

    CLopt_maxthrust_selected = CL_maxthrust_selected
    dTopt_maxthrust_selected = 1
    return (
        CL_maxthrust_selected,
        drag_maxthrust_harray,
        drag_maxthrust_selected,
        h_maxthrust_array,
        true_maxthrust,
        velocity_maxthrust_harray,
        velocity_maxthrust_selected,
    )


@app.cell
def _(figure_optimum, mo, tab_value, title_keys, variables_stack):
    if tab_value != title_keys[3]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    _Thrust-lift limited minimum drag_


    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 \gt 0$

    $\delta_T = 1 \quad \Rightarrow \quad \mu_3 \gt 0$

    From stationarity condition (2) obtain:

    $$
    \mu_3 = -\lambda_1 P_{a0}\sigma^\beta \sqrt{\frac{\rho S C_{L_\mathrm{max}}}{2W}} \gt 0 \quad \Rightarrow \quad \lambda_1 \lt 0
    $$

    From stationarity condition (1):

    $$
    \mu_1 = W \frac{K C_{L_\mathrm{max}}^2 - C_{D_0}}{C_{L_\mathrm{max}}^2} (\lambda_1 - 1) - \frac{1}{2} \lambda_1 P_{a0}\sigma^\beta \sqrt{\frac{\rho S}{2 W C_{L_\mathrm{max}}}}   \gt 0
    $$

    Isolating $\lambda_1$ and simplifying results in:

    $$
    \lambda_1 > \displaystyle\frac{1}{1 - \displaystyle\frac{P_{a0}\sigma^{\beta+1/2}}{2W^{3/2}}\sqrt{\frac{\rho_0 S}{2}}\frac{C_{L_\mathrm{max}}^{3/2}}{K C_{L_\mathrm{max}}^2 - C_{D_0}}}
    % \left(-\frac{1}{2} A C_{L_\mathrm{max}} ^{-1/2} + W \frac{KC_{L_\mathrm{max}} ^2 - C_{D_0}}{C_{L_\mathrm{max}}^2} \right) - W \frac{KC_{L_\mathrm{max}} ^2 - C_{D_0}}{C_{L_\mathrm{max}}^2} \gt 0
    $$

    Remembering that it must be $\lambda_1<0$, the condition for this optimum to exist is found in terms of weight and altitude by imposing that the previous denominator is negative. We obtain:

    $$
    \frac{W^{3/2}}{\sigma^{\beta+1/2}} < \frac{P_{a0}}{2}\sqrt{\frac{\rho_0 S}{2}}\frac{C_{L_\mathrm{max}}^{3/2}}{K C_{L_\mathrm{max}}^2 - C_{D_0}}
    $$

    where we note that it must be:

    $$
    K C_{L_\mathrm{max}}^2 - C_{D_0} > 0 \quad \Rightarrow \quad C_{L_\mathrm{max}} > \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$


    From the primal feasibility condition (3):

    $$
    \frac{W^{3/2}}{\sigma^{\beta+1/2}} = P_{a0}\sqrt{\frac{\rho_0 S}{2}}\frac{C_{L_\mathrm{max}}^{3/2}}{C_{D_0} + K C_{L_\mathrm{max}}^2}
    $$

    Substituting this into the previous inequality yields:

    $$
    \frac{3C_{D_0}-KC_{L_\mathrm{max}}^2}{KC_{L_\mathrm{max}}^2 - C_{D_0}} > 0
    $$

    which is then verified for:

    $$
    C_{L_\mathrm{max}} < \sqrt{\frac{3 C_{D_0}}{K}} = C_{L_P}
    $$

    Thus, taking the intersection of the conditions on $C_L^*$ obtain:

    $$
    \sqrt{\frac{C_{D_0}}{K}} \lt C_L^* \lt \sqrt{\frac{3C_{D_0}}{K}}
    $$

    Finally, the value of the objective function $D$ can be calculated:

    $$
    D_{\mathrm{min}}^* = \frac{W}{E_S}
    $$

    This concludes the analysis for the minimum drag of a simplified propeller aircraft in the thrust-lift limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = C_{L_\mathrm{max}}}, \quad \boxed{\delta_T^*=1}, \quad \text{for} \quad \sqrt{\frac{C_{D_0}}{K}} \lt C_L^* \lt \sqrt{\frac{3C_{D_0}}{K}} \quad \text{and} \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}}= P_{a0}E_S\sqrt{\frac{\rho_0 S C_{L_\mathrm{max}}}{2}}
    $$

    With the optimal value for minimum drag:

    $$
    D_{\mathrm{min}}^* =\frac{W}{E_S}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.

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

        if CLmax > CL_P or CLmax < CL_E:
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

    drag_maxliftThrust_harray = drag_curve
    drag_maxliftThrust_selected = W_selected / E_S
    return (
        drag_maxliftThrust_harray,
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
    Now that we have derived all the optima for each condition, we can summarize the flight envelopes in one graph, as shown below. Experiment with the weight of the aircraft to understand how the theoretical ceiling for minimum power moves in the graph.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Final flight envelope

    This concludes the minimum drag derivation, find below the flight envelope showing the operational conditions where the simplified propeller aircraft can fly at minimum drag. The graph below combines all the solutions explored in this notebook.
    """)
    return


@app.cell
def _(mass_stack):
    mass_stack
    return


@app.cell
def _(
    a_harray,
    h_array,
    h_interior_array,
    h_maxliftThrust,
    h_maxlift_array,
    h_maxthrust_array,
    np,
    plot_utils,
    true_maxliftThrust,
    velocity_interior_harray,
    velocity_maxliftThrust_selected,
    velocity_maxlift_harray,
    velocity_maxthrust_harray,
    velocity_stall_harray,
):
    plot_utils.create_final_flightenvelope(
        velocity_stall_harray,
        a_harray,
        h_array,
        (
            np.concat((h_interior_array, h_maxthrust_array)),
            np.concat((velocity_interior_harray, velocity_maxthrust_harray)),
            True,
        ),
        (h_maxlift_array, velocity_maxlift_harray, True),
        (h_maxthrust_array, velocity_maxthrust_harray, True),
        (h_maxliftThrust, velocity_maxliftThrust_selected * true_maxliftThrust, False),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary
    """)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    | Name | Condition | $C_L^*$ | $\delta_T^*$ | $D^*$ |
    |:-|:-------|:-------:|:------------:|:-------|
    |Interior-optima    | $\displaystyle   C_L^* \lt C_{L_\mathrm{max}} \quad \text{and} \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}} \lt P_{a0}E_\mathrm{max}\sqrt{\frac{\rho_0 S C_{L_E}}{2}}$ | $\sqrt{\frac{C_{D_0}}{K}}$ | $\displaystyle \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}}\frac{1}{P_{a0}E_\mathrm{max}}\sqrt{\frac{2}{\rho_0 S C_{L_E}}}$ | $\displaystyle 2W\sqrt{KC_{D_0}}=\frac{W}{E_\mathrm{max}}$ |
    |Lift-limited    |  $\displaystyle C_L^* \lt \sqrt{\frac{C_{D_0}}{K}} \quad \text{and} \quad\frac{W^{3/2}}{\sigma^{\beta+1/2}}\lt P_{a0}E_S\sqrt{\frac{\rho_0 S C_{L_\mathrm{max}}}{2}}$ | $C_{L_\mathrm{max}}$ | $\displaystyle \displaystyle \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}}\frac{1}{P_{a0}E_S}\sqrt{\frac{2}{\rho_0 S C_{L_\mathrm{max}}}}$ | $\displaystyle \frac{W}{E_S}$|
    |Thrust-limited    | || $1$| |
    |Thrust-lift limited    |   $\displaystyle   \sqrt{\frac{C_{D_0}}{K}} \lt C_L^* \lt \sqrt{\frac{3C_{D_0}}{K}} \quad \text{and} \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}}=P_{a0}E_S\sqrt{\frac{\rho_0 S C_{L_\mathrm{max}}}{2}}$ | $C_{L_\mathrm{max}}$ | $1$ | $\displaystyle \frac{W}{E_S}$|
    """
    ).center()
    return


if __name__ == "__main__":
    app.run()
