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


@app.cell
def _():
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
    drag_surface,
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
    zcolorbar,
):
    # Computation only cell, indexing happens in another cell
    velocity_CLarray = np.sqrt(2 * W_selected / (rho_selected * S * CL_array))
    velocity_CL_E = velocity_CLarray[-1] * np.sqrt(CLmax / CL_E)
    velocity_CL_P = velocity_CLarray[-1] * np.sqrt(CLmax / CL_P)

    power_available = thrust_scalar * velocity_CLarray / 1e3
    power_required = drag_curve * velocity_CLarray / 1e3

    constraint = drag_curve / Ta0 / (sigma_selected**beta)

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
    # Minimum drag: simplified jet aircraft

    $$
    \begin{aligned}
    \min_{C_L, \delta_T}
    & \quad D = \frac{1}{2}\rho V^2S\left(C_{D_0} + K C_L^2\right) \\
    \text{subject to}
    & \quad c_1^\mathrm{eq} = L-W = \frac{1}{2}\rho V^2 S C_L - W = 0\\
    & \quad c_2^ \mathrm{eq} = T-D = \delta_T T_a(V,h) - \frac{1}{2} \rho V^2 S (C_{D_0}+K C_L^2) =0 \\
    \text{for}
    & \quad C_L \in [0, C_{L_\mathrm{max}}] \\
    & \quad \delta_T \in [0, 1] \\
    \text{with}
    & \quad T_a(V,h) = T_a(h) = T_{a0}\sigma^\beta \\
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## KKT formulation
    To be reconducted in the standard KKT analysis format, the objective function is expressed in terms of the controls by directly eliminating $c_1^\mathrm{eq}$. The velocity $V$ can be described as:

    $$
    V = \sqrt{\frac{2}{\rho}\frac{W}{S}\frac{1}{C_L}}
    $$


    The KKT formulation thus becomes:
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
    & \quad g_1 = \frac{T}{W} - \frac{1}{E}  =\frac{\delta_T T_{a0}\sigma^\beta}{W} - \frac{C_{D_0} + K C_L^2}{C_L} = 0 \\
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


@app.cell(hide_code=True)
def _(ac_table):
    ac_table
    return


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell(hide_code=True)
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
                cmax=max_colorbar,
                cmin=min_colorbar,
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
            ),
            go.Scatter3d(
                x=[CL_array[-15]],
                y=[constraint[-15]],
                z=[drag_surface[0, -15] + 7e3],
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
    camera = dict(eye=dict(x=1.35, y=1.35, z=1.35))

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

    fig_initial.update_layout(
        scene_camera=camera,
        title={
            "text": f"Minimum drag domain for {active_selection.full_name}",
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
def _(mo):
    mo.md(r"""
    The multipliers $\lambda_1, \mu_1, \mu_2, \mu_3, \mu_4$ have to meet the following conditions for an optimal solution of the optimization problem $(C_L^*, \delta_T^*)$ to exist.

    **A. Stationarity conditions($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \begin{aligned}\frac{\partial \mathcal{L}}{\partial C_L} = W \frac{K C_L^2 - C_{D_0}}{C_L^2} - \lambda_1W\left(\frac{K C_L^2 - C_{D_0}}{C_L^2}\right) + \mu_1 - \mu_2 = W\frac{K C_L^2 - C_{D_0}}{C_L^2} (1 -\lambda_1) +\mu_1 - \mu_2 = 0 \end{aligned}$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1\frac{T_{a0}\sigma^\beta}{W}+\mu_3-\mu_4 = 0$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
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
def _(mo):
    mo.md(r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    8.  $\mu_1, \mu_2, \mu_3, \mu_4 \ge 0$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraint have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

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

    1. $\displaystyle W\frac{K C_L^2 - C_{D_0}}{C_L^2} (1 -\lambda_1) +\mu_1 = 0$
    2. $\displaystyle \lambda_1\frac{T_{a0}\sigma^\beta}{W}+\mu_3 = 0$
    3. $\displaystyle \frac{\delta_T T_{a0}\sigma^\beta}{W} - \frac{C_{D_0} + K C_L^2}{C_L} = 0$
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
    CL_E,
    CL_array,
    CLmax,
    CLopt_interior,
    CLopt_maxlift,
    OptimumGridView,
    active_selection,
    configTraces,
    constraint_maxliftThrust,
    constraint_maxthrust,
    dT_array,
    dTopt_interior,
    dTopt_maxlift,
    drag_curve,
    drag_surface,
    drag_yrange,
    h_interior_array,
    h_maxliftThrust,
    h_maxlift_array,
    h_maxthrust,
    h_selected,
    mach_trace,
    maxliftThrust_multiplier,
    maxthrust_multiplier,
    plot_utils,
    power_interior_harray,
    power_interior_selected,
    power_maxliftThrust_harray,
    power_maxliftThrust_selected,
    power_maxlift_harray,
    power_maxlift_selected,
    power_maxthrust_harray,
    power_maxthrust_selected,
    power_yrange,
    range_performance_diagrams,
    stall_trace,
    tab_value,
    thrust_vector_maxliftThrust,
    thrust_vector_maxthrust,
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
    velocity_maxthrust_CLarray,
    velocity_maxthrust_selected,
    velocity_stall_maxliftThrust,
    velocity_stall_maxthrust,
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
        configTraces_maxthrust = plot_utils.ConfigTraces(
            CL_array,
            dT_array,
            constraint_maxthrust,
            drag_curve,
            thrust_vector_maxthrust,
            power_maxthrust_harray,
            thrust_vector_maxthrust * velocity_maxthrust_CLarray / 1e3,
            drag_surface,
            velocity_maxthrust_CLarray,
            velocity_CL_P * maxthrust_multiplier,
            velocity_CL_E * maxthrust_multiplier,
            velocity_maxthrust_selected,
            velocity_stall_maxthrust,
            (drag_yrange, power_yrange / 1e3, CLmax),
            zcolorbar,
            mach_trace,
            stall_trace,
        )

        # maxthrust graphics
        figure_optimum = OptimumGridView(
            configTraces_maxthrust,
            h_maxthrust,
            (velocity_maxthrust_CLarray, velocity_maxthrust_selected),
            (power_maxthrust_harray, power_maxthrust_selected),
            (h_maxthrust, 1 * true_maxthrust, CL_E, true_maxthrust),
            f"Thrust-lift limited minimum drag for {active_selection.full_name}",
            equality=True,
        )
    elif tab_value == title_keys[3]:
        configTraces_maxliftThrust = plot_utils.ConfigTraces(
            CL_array,
            dT_array,
            constraint_maxliftThrust,
            drag_curve,
            thrust_vector_maxliftThrust,
            power_maxliftThrust_harray,
            thrust_vector_maxliftThrust * velocity_maxliftThrust_CLarray / 1e3,
            drag_surface,
            velocity_maxliftThrust_CLarray,
            velocity_CL_P * maxliftThrust_multiplier,
            velocity_CL_E * maxliftThrust_multiplier,
            velocity_maxliftThrust_selected,
            velocity_stall_maxliftThrust,
            (drag_yrange, power_yrange / 1e3, CLmax),
            zcolorbar,
            mach_trace,
            stall_trace,
        )

        # maxliftThrust graphics
        figure_optimum = OptimumGridView(
            configTraces_maxliftThrust,
            h_maxliftThrust,
            (velocity_maxliftThrust_CLarray, velocity_maxliftThrust_selected),
            (power_maxliftThrust_harray, power_maxliftThrust_selected),
            (h_maxliftThrust, 1 * true_maxliftThrust, CLmax, true_maxliftThrust),
            f"Thrust-lift limited minimum drag for {active_selection.full_name}",
            equality=True,
        )

    figure_optimum.update_axes_ranges(range_performance_diagrams)
    return (figure_optimum,)


@app.cell
def _(figure_optimum, mo, tab_value, title_keys, variables_stack):
    if tab_value != title_keys[0]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ### _Interior optimum for minimum drag_ 

    Assuming that that $0 < C_L^* < C_{L_\mathrm{max}}$ and $0 < \delta_T^* < 1$ is equivalent to consider all inequality constraints as inactive.

    Therefore: $\mu_1, \mu_3 =0$. 

    From stationarity condition (2): $\lambda_1 = 0$.

    From stationarity condition (1), it is possible to obtain the value of $C_L^*$ for minimum drag.

    $$
    C_L^* = \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$

    Notice how the optimal $C_L^*$ has the **same value** for maximum aerodynamic efficiency (maximum $C_L /C_D$), for 
    $0\lt C_L \lt  C_{L_\mathrm{max}}$ and $0 \lt \delta_T \lt 1$, as shown in [Aerodynamic Efficiency](/?file=Steady_Level_Flight/MinDrag.py).

    The corresponding $\delta_T^*$ is found by solving the primal feasibility constraint (3) and using $C_L = C_L^*$, as calculated above.


    $$
    \delta_T^* = \frac{2W}{T_{a0}\sigma^\beta}\frac{C_{D_0} + K C_L^2}{C_L} = \frac{2W}{T_{a0}\sigma^\beta}\sqrt{C_{D_0}K}
    $$

    This value is compliant with the primal feasibility constraint (5) for: 

    $$
    \delta_T^* = \frac{2W}{T_{a0}\sigma^\beta}\sqrt{C_{D_0}K} \lt 1 \quad \Leftrightarrow \quad \frac{W}{\sigma^\beta} \lt \frac{T_{a0}}{2\sqrt{C_{D_0}K}} = \frac{W}{\sigma^\beta} \lt T_{a0}E_{max}$$

    Which tells us that it is possible to achieve this optimal condition only when the combination of aircraft weight and altitude respect the above inequality.

    The corresponding minimum drag is found by first computing $V^*$ and $C_D^*$: 

    $$
    V = \sqrt{\frac{W}{S}\frac{2}{\rho}\frac{1}{C_L^*}} \quad \text{and} \quad  C_D^* = 
    2C_{D_0}$$

    $$
    D_{\mathrm{min}}^* =  \frac{1}{2}\rho {V^*}^2 S C_D= 
    2W\sqrt{KC_{D_0}}=\frac{W}{E_\mathrm{max}}
    $$

    We can now rewrite $\delta_T^*$ in terms of $D_\mathrm{min}$:

    $$
    \delta_T^*=\frac{2W}{T_{a0}\sigma^\beta}\sqrt{C_{D_0}K}=\frac{D_\mathrm{min}^*}{T_{a0}\sigma^{\beta}}
    $$

    This concludes the analysis for the minimum drag of a simplified jet aircraft in the domain's interior. Below is a summary of the optima:

    $$
    \boxed{C_L^* = \sqrt{\frac{C_{D_0}}{K}}}, \quad \boxed{\delta_T^*=\frac{2W}{T_{a0}\sigma^\beta}\sqrt{C_{D_0}K}=\frac{D_\mathrm{min}^*}{T_{a0}\sigma^{\beta}}}, \quad \text{for} \quad C_L^* \lt C_{L_\mathrm{max}}\quad \text{and} \quad \frac{W}{\sigma^\beta} \lt T_{a0}E_\mathrm{max}
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
        Ta0,
        beta,
        CL_E,
        CLmax,
        E_max,
        min_sigma,
        sigma_selected,
        h_array,
    ):
        sigma_interior = (W / (E_max * Ta0)) ** (1 / beta)

        dT_interior = W / E_max / Ta0 / (sigma_selected**beta)

        if CLmax < CL_E:
            return np.array([np.nan]), np.nan, np.nan, np.nan

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
    Ta0,
    W_selected,
    atmos,
    beta,
    h_array,
    h_selected,
    interior_condition,
    min_sigma,
    np,
    rho_selected,
    sigma_selected,
    velocity_CL_E,
):
    # Interior computation
    h_interior_array, dTopt_interior, CLopt_interior, true_interior = (
        interior_condition(
            W_selected,
            h_selected,
            Ta0,
            beta,
            CL_E,
            CLmax,
            E_max,
            min_sigma,
            sigma_selected,
            h_array,
        )
    )

    velocity_interior_selected = velocity_CL_E * true_interior
    velocity_interior_harray = velocity_CL_E * np.sqrt(
        rho_selected / atmos.rho(h_interior_array)
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

    From stationarity condition (1): $\displaystyle \mu_1 = W\frac{C_{D_0} - KC_{L_\mathrm{max}}^2}{C_{L_\mathrm{max}}^2} \gt 0$, which results in: $\displaystyle  C_{L_\mathrm{max}} < \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}$

    This means that, in order for aerodynamic drag to have a minimum at $C_L = C_{L_\mathrm{max}}$, the aircraft must have been designed to have a higher lift coefficient for maximum efficiency than its stall lift coefficient.

    In other words, the aircraft would only be able to fly on the right branch of the performance diagram, and the stall speed would be higher than the speed for maximum efficiency, therefore representing the speed for minimum drag.

    In the rare occasion this condition would be verified, the corresponding throttle could be once again calculated from stationarity condition (3):

    $$
    \displaystyle \delta_T^* = \frac{C_{D_\mathrm{max}}}{C_{L_\mathrm{max}}}\frac{W}{T_{a0}\sigma^\beta} = \frac{W}{E_S T_{a0}\sigma^\beta}
    $$

    This value is compliant with the primal feasibility constraint if:

    $$
    \delta_T^* < 1 \Leftrightarrow \frac{W}{\sigma^\beta} < T_{a0}E_S
    $$

    which gives us the conditions to achieve minimum drag in terms of aircraft weight and altitude.

    The value of the objective function, minimum drag, is calculated straightforwardly as:

    $$
    D_{\mathrm{min}}^* =  \frac{1}{2}\rho V_s^2 S C_{D_s} = \frac{W}{E_s}
    $$

    This is a higher value than the unconstrained one, and therefore operating in this scenario should be avoided if minimum drag is a goal.

    This concludes the analysis for the minimum drag of a simplified jet aircraft in the lift-limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = C_{L_\mathrm{max}}}, \quad \boxed{\delta_T^*=\frac{W}{E_S T_{a0}\sigma^\beta}}, \quad \text{for} \quad C_{L}^* < \sqrt{\frac{C_{D_0}}{K}} \quad \text{and} \quad \frac{W}{\sigma^\beta} \lt T_{a0}E_{S}
    $$

    With the optimal value for minimum drag:

    $$
    D_{\mathrm{min}}^* = \frac{W}{E_S}
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
        Ta0,
        beta,
        CL_E,
        CLmax,
        E_S,
        min_sigma,
        sigma_selected,
        h_array,
    ):
        sigma_interior = (W / (E_S * Ta0)) ** (1 / beta)

        dT_interior = W / E_S / Ta0 / (sigma_selected**beta)

        if CLmax >= CL_E:
            return np.array([np.nan]), np.nan, np.nan, np.nan

        h_interior = atmos.altitude(sigma_interior)
        h_interior_array = h_array[h_array < h_interior]

        h_min = h_interior_array.min()
        h_max = h_interior_array.max()
        cond = 1 if h_min <= h_selected <= h_max else np.nan
        return h_interior_array, dT_interior, CL_E, cond

    return (maxlift_condition,)


@app.cell
def _(
    CL_P,
    CLmax,
    E_S,
    Ta0,
    W_selected,
    atmos,
    beta,
    h_array,
    h_selected,
    maxlift_condition,
    min_sigma,
    np,
    rho_selected,
    sigma_selected,
    velocity_CLarray,
):
    # Maxlift condition
    h_maxlift_array, dTopt_maxlift, CLopt_maxlift, true_maxlift = maxlift_condition(
        W_selected,
        h_selected,
        CL_P,
        CLmax,
        E_S,
        Ta0,
        beta,
        h_array,
        min_sigma,
        sigma_selected,
    )

    velocity_maxlift_selected = velocity_CLarray[-1] * true_maxlift
    velocity_maxlift_harray = velocity_CLarray[-1] * np.sqrt(
        rho_selected / atmos.rho(h_maxlift_array)
    )

    power_maxlift_harray = W_selected / E_S * velocity_maxlift_harray
    power_maxlift_selected = W_selected / E_S * velocity_maxlift_selected
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

    From stationarity condition (2), obtain:

    $$
    \lambda_1 = -\frac{\mu_3}{T_{a0}\sigma^{\beta}} \lt 0
    $$

    and from stationarity condition (1):

    $$
    \displaystyle \left(\frac{KC_L^2 - C_{D_0}}{C_L^2}\right)(1-\lambda_1) = 0 \Rightarrow \frac{KC_L^2 - C_{D_0}}{C_L^2} = 0
    $$

    Which yields the following optima:

    $$
    C_L^* = \sqrt{\frac{C_{D_0}}{K}} = C_{L_E} \quad  \text{and} \quad \delta_T^* = 1
    $$

    This optimum is continuous with the interior optimum, thus yielding the same result for $D_\mathrm{min}$:

    $$
    D_\mathrm{min}^* =  \frac{1}{2}\rho {V^*}^2 S C_D = 2W\sqrt{KC_{D_0}}=\frac{W}{E_\mathrm{max}}
    $$

    The operational condition is found from (3), with $\delta_T = 1$, obtaining:

    $$
    \frac{W}{\sigma^\beta} = T_{a0}E_{\mathrm{max}}
    $$

    with:

    $$
    C_D^* = 2C_{D_0}, \quad V^* = \sqrt{\frac{W}{S}\frac{2}{\rho}\frac{1}{C_L^*}}, \quad \delta_T^*=1, \quad \frac{W}{\sigma^\beta} = T_{a0}E_{\mathrm{max}}
    $$

    This concludes the analysis for the minimum drag of a simplified jet aircraft in the thrust-limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = \sqrt{\frac{C_{D_0}}{K}}}, \quad \boxed{\delta_T^*=1}, \quad \text{for} \quad C_{L}^* < C_{L_\mathrm{max}} \quad \text{and} \quad \frac{W}{\sigma^\beta} = T_{a0}E_\mathrm{max}
    $$

    With the following value for the objective function:

    $$
    D_{\mathrm{min}}^* = \frac{W}{E_\mathrm{max}}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.

    """),
            variables_stack,
            figure_optimum.figure,
        ]
    ).callout()
    return


@app.cell
def _(atmos, min_sigma, np):
    def maxthrust_condition(W, Ta0, E_max, beta, CL_E, CLmax):
        sigma_maxthrust = (W / Ta0 / E_max) ** (1 / beta)
        h_maxthrust_selected = atmos.altitude(sigma_maxthrust)

        if CLmax < CL_E and sigma_maxthrust > min_sigma:
            return h_maxthrust_selected, sigma_maxthrust, np.nan

        condition = True

        return (
            h_maxthrust_selected,
            sigma_maxthrust,
            condition,
        )

    return (maxthrust_condition,)


@app.cell
def _(
    CL_E,
    CLmax,
    E_S,
    E_max,
    S,
    Ta0,
    W_selected,
    atmos,
    beta,
    drag_curve,
    h_selected,
    maxthrust_condition,
    np,
    rho_selected,
    sigma_selected,
    thrust_vector,
    velocity_CL_E,
    velocity_CLarray,
):
    # Max lift Max thrust
    h_maxthrust, sigma_maxthrust, true_maxthrust = maxthrust_condition(
        W_selected, Ta0, E_max, beta, CL_E, CLmax
    )

    maxthrust_multiplier = np.sqrt(rho_selected / (atmos.rho0 * sigma_maxthrust))

    constraint_maxthrust = drag_curve / Ta0 / (sigma_maxthrust**beta)

    thrust_vector_maxthrust = thrust_vector * (sigma_maxthrust / sigma_selected) ** beta
    velocity_maxthrust_CLarray = velocity_CLarray * maxthrust_multiplier
    velocity_maxthrust_selected = velocity_CL_E * maxthrust_multiplier

    velocity_stall_maxthrust = (
        np.sqrt(2 * W_selected / (atmos.rho(h_selected) * S * CLmax))
        * maxthrust_multiplier
    )

    power_maxthrust_harray = drag_curve * velocity_maxthrust_CLarray / 1e3
    power_maxthrust_selected = W_selected / E_S * velocity_maxthrust_selected
    return (
        constraint_maxthrust,
        h_maxthrust,
        maxthrust_multiplier,
        power_maxthrust_harray,
        power_maxthrust_selected,
        thrust_vector_maxthrust,
        true_maxthrust,
        velocity_maxthrust_CLarray,
        velocity_maxthrust_selected,
        velocity_stall_maxthrust,
    )


@app.cell
def _(figure_optimum, mo, tab_value, title_keys, variables_stack):
    if tab_value != title_keys[3]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ###_Thrust- and lift- limited minimum drag_

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 \gt 0$

    $\delta_T = 1 \quad \Rightarrow \quad \mu_3 \gt 0$

    from stationarity condition (2):

    $$
    \lambda_1= -\frac{\mu_3 }{T_{a0}\sigma^{\beta}} \lt 0
    $$

    and from stationarity condition (1):

    $$
    \displaystyle \mu_1 = W \left( \frac{C_{D_0} - KC_{L_\mathrm{max}}^2}{C_{L_\mathrm{max}}^2}\right)(1 - \lambda_1) \gt 0
    $$

    The two conditions above need to be verified for a minimum to exist when both boundaries are active,  at the same time:

    $$
    C_{L_\mathrm{max}} \lt \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$

    and not equality
    The same considerations hold for the case of the lift-limited analysis, with the only difference that now $\delta_T^* = 1$
    In fact, once again, the aircraft would have to stall at a higher speed than the one for minimum drag. Continuing with primal feasibility condition (3), obtain the operational condition:


    $$
    \frac{W}{\sigma^\beta} = T_{a0}E_S
    $$

    This concludes the analysis for the minimum drag of a simplified jet aircraft in the thrust-lift limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = C_{L_\mathrm{max}}}, \quad \boxed{\delta_T^*=1}, \quad \text{for} \quad C_{L}^* \lt \sqrt{\frac{C_{D_0}}{K}} \quad \text{and} \quad \frac{W}{\sigma^\beta} = T_{a0}E_S
    $$

    With the following value for the objective function:

    $$
    D_{\mathrm{min}}^* = \frac{W}{E_\mathrm{S}}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.
    """),
            variables_stack,
            figure_optimum.figure,
        ]
    ).callout()
    return


@app.cell
def _(atmos, min_sigma, np):
    def maxliftThrust_condition(W, Ta0, E_S, beta, CL_E, CLmax):
        sigma_maxliftThrust = (W / Ta0 / E_S) ** (1 / beta)
        h_maxliftThrust_selected = atmos.altitude(sigma_maxliftThrust)

        if CLmax > CL_E or sigma_maxliftThrust > min_sigma:
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
    CLmax,
    E_S,
    E_max,
    S,
    Ta0,
    W_selected,
    atmos,
    beta,
    drag_curve,
    h_selected,
    maxliftThrust_condition,
    np,
    rho_selected,
    sigma_selected,
    thrust_vector,
    velocity_CLarray,
):
    # Max lift Max thrust
    h_maxliftThrust, sigma_maxliftThrust, true_maxliftThrust = maxliftThrust_condition(
        W_selected, Ta0, E_max, beta, CL_E, CLmax
    )

    maxliftThrust_multiplier = np.sqrt(
        rho_selected / (atmos.rho0 * sigma_maxliftThrust)
    )

    constraint_maxliftThrust = drag_curve / Ta0 / (sigma_maxliftThrust**beta)

    drag_maxliftThrust_selected = W_selected / E_S

    thrust_vector_maxliftThrust = (
        thrust_vector * (sigma_maxliftThrust / sigma_selected) ** beta
    )
    velocity_maxliftThrust_CLarray = velocity_CLarray * maxliftThrust_multiplier
    velocity_maxliftThrust_selected = (
        velocity_maxliftThrust_CLarray[-1] * true_maxliftThrust
    )

    velocity_stall_maxliftThrust = (
        np.sqrt(2 * W_selected / (atmos.rho(h_selected) * S * CLmax))
        * maxliftThrust_multiplier
    )

    power_maxliftThrust_harray = drag_curve * velocity_maxliftThrust_CLarray / 1e3
    power_maxliftThrust_selected = W_selected / E_max * velocity_maxliftThrust_selected
    return (
        constraint_maxliftThrust,
        h_maxliftThrust,
        maxliftThrust_multiplier,
        power_maxliftThrust_harray,
        power_maxliftThrust_selected,
        thrust_vector_maxliftThrust,
        true_maxliftThrust,
        velocity_maxliftThrust_CLarray,
        velocity_maxliftThrust_selected,
        velocity_stall_maxliftThrust,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Final flight envelope
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Now after deriving all the optima for each condition we can summarize the flight envelopes in one graph, as shown below. Experiment with the weight of the aircrarft to understand how the theoretical ceiling for minimum power moves in the graph.
    """)
    return


@app.cell
def _(
    a_harray,
    h_array,
    h_interior_array,
    h_maxliftThrust,
    h_maxlift_array,
    h_maxthrust,
    mass_stack,
    mo,
    np,
    plot_utils,
    velocity_interior_harray,
    velocity_maxliftThrust_selected,
    velocity_maxlift_harray,
    velocity_maxthrust_selected,
    velocity_stall_harray,
):
    flight_envelope = plot_utils.create_final_flightenvelope(
        velocity_stall_harray,
        a_harray,
        h_array,
        (
            np.concat((h_interior_array, [h_maxthrust])),
            np.concat((velocity_interior_harray, [velocity_maxthrust_selected])),
            True,
        ),
        (
            np.concat((h_maxlift_array, [h_maxliftThrust])),
            np.concat((velocity_maxlift_harray, [velocity_maxliftThrust_selected])),
            True,
        ),
        (h_maxthrust, velocity_maxthrust_selected, False),
        (h_maxliftThrust, velocity_maxliftThrust_selected, False),
    )

    mo.vstack([mass_stack, flight_envelope])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    | Name | Condition | $C_L^*$ | $\delta_T^*$ | $P^*$ |
    |:-|:-------|:-------:|:------------:|:-------|
    |Interior-optima    | $\displaystyle C_L^* \lt C_{L_\mathrm{max}} \quad \text{and} \quad \frac{W}{\sigma^\beta} \lt \frac{\sqrt{3}}{2}E_\mathrm{max}T_{a0}$ | $\sqrt{\frac{3C_{D_0}}{K}}$ | $\displaystyle \frac{W}{E_{\mathrm{P}}}\frac{1}{T_{a0}\sigma^\beta}$ | $\displaystyle 2W\sqrt{KC_{D_0}}=\frac{W}{E_\mathrm{max}}$ |
    |Lift-limited    |  $\displaystyle C_L^* \lt \sqrt{\frac{C_{D_0}}{K}}\quad \text{and}\quad \frac{W}{\sigma^\beta} < T_{a0} E_\mathrm{S}$ | $C_{L_\mathrm{max}}$ | $\displaystyle \frac{W}{E_S T_{a0} \sigma^\beta}$ | $\displaystyle \frac{W}{E_S}$|
    |Thrust-limited    | $\displaystyle C_L^* \lt C_{L_\mathrm{max}} \quad \text{and}\quad  \frac{W}{\sigma^\beta} = T_{a0} E_\mathrm{max}$ | $\displaystyle \sqrt{\frac{C_{D_0}}{K}}$ | $1$ | $\displaystyle \frac{W^{3/2}}{\sigma^{1/2}}\left(\frac{C_{D_0}+ K C_L^{*2}}{C_L^{*}}\right)\sqrt{\frac{2}{\rho_0 S C_L^*}}$ |
    |Thrust-lift limited    |  $\displaystyle C_L^* \lt \sqrt{\frac{C_{D_0}}{K}}\quad \text{and}\quad  \frac{W}{\sigma^\beta} = T_{a0} E_\mathrm{S}$ | $C_{L_\mathrm{max}}$ | $1$ | $\displaystyle \frac{W}{E_S}$|
    """
    ).center()
    return


@app.cell
def _():
    _defaults.nav_footer(
        after_file="MinDrag_Prop.py",
        after_title="Minimum Drag Simplified Propeller",
        above_file="MinDrag.py",
        above_title="Minimum Drag Homepage",
        above_before=True,
    )
    return


if __name__ == "__main__":
    app.run()
