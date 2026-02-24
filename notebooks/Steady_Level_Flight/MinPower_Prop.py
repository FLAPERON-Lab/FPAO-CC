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
def _(Pa0, beta, h_array, h_slider, meshgrid_n):
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
    drag_yrange,
    idx_h_selected,
    mach_trace,
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

    velocity_stall_selected = velocity_CLarray[-1]
    thrust_vector = power_scalar / velocity_CLarray * 1e3
    power_required = drag_curve * velocity_CLarray / 1e3

    power_surface = np.broadcast_to(
        power_required[np.newaxis, :],  # Shape: (101, 1)
        (len(CL_array), len(dT_array)),  # Target shape: (101, 101)
    )
    min_colorbar = np.min(power_required)
    max_colorbar = min_colorbar * 1.5
    zcolorbar = (min_colorbar, max_colorbar)

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
        power_surface,
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
        power_required,
        power_surface,
        range_performance_diagrams,
        velocity_CL_E,
        velocity_CL_P,
        velocity_CLarray,
        velocity_stall_selected,
    )


@app.cell
def _(idx_CL_selected, power_required):
    power_required_selected = power_required[idx_CL_selected]
    return (power_required_selected,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Minimum Power Required: simplified propeller aircraft

    $$
    \begin{aligned}
        \min_{C_L, \delta_T}
        & \quad P = DV = \frac{1}{2}\rho V^2S(C_{D_0}+KC_L^2)V \\
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
def _(
    CL_array,
    CL_slider,
    active_selection,
    constraint,
    dT_array,
    dT_slider,
    max_colorbar,
    min_colorbar,
    power_required_selected,
    power_surface,
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
                z=power_surface,
                opacity=0.9,
                name="Power",
                colorscale="viridis",
                cmax=max_colorbar,
                cmin=min_colorbar,
                colorbar={"title": "Power (kW)"},
            ),
            go.Scatter3d(
                x=CL_array,
                y=constraint,
                z=power_surface[0],
                opacity=1,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[CL_array[-15]],
                y=[constraint[-15]],
                z=[power_surface[0, -15] + 250],
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
                z=[power_required_selected],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=3,
                    color="white",
                    symbol="circle",
                ),
                name="Design Point",
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub> : %{y}<br>P: %{z}<extra>%{fullData.name}</extra>",
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
            zaxis=dict(title="P (kW)", range=[0, max_colorbar]),
        ),
    )

    fig_initial.update_layout(
        scene_camera=camera,
        title={
            "text": f"Minimum power domain for {active_selection.full_name}",
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
    To be reconducted in the standard KKT analysis format, the objective function is expressed in terms of the controls by direct elimination of $c_1^\mathrm{eq}$. The velocity $V$ can be expressed as:

    $$
    V = \sqrt{\frac{2}{\rho}\frac{W}{S}\frac{1}{C_L}}
    $$

    Moreover, in previous analyses we found $\delta_T=C_L=0$ does not correspond to a sensible solution, thus we can write:

    $$
    0\lt \delta_T \le 1 \quad \text{and} \quad  0\lt C_L\le C_{L_{\mathrm{max}}}
    $$

    Notice the open interval in the lower bounds.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The KKT formulation can now be written:

    $$
    \begin{aligned}
        \min_{C_L, \delta_T}
        & \quad P = DV = W \left(\frac{C_{D_0} +K C_L^2}{C_L}\right)\sqrt{\frac{2}{\rho}\frac{W}{S}\frac{1}{C_L}}=\sqrt{\frac{2W^3}{\rho S}}\left(\frac{C_{D_0}+K C_L^2}{C_L^{3/2}}\right) = \sqrt{\frac{2W^3}{\rho S}}\left(C_{D_0} C_L^{-3/2}+K C_L^{1/2}\right)\\
        \text{subject to}
        & \quad g_1 = T - W\, \frac{1}{E}  =\frac{\delta_T P_{a0}\sigma^\beta}{V} - W\frac{C_{D_0} + K C_L^2}{C_L} = 0 \quad \Rightarrow \quad \delta_T P_{a0}\sigma^\beta - \sqrt{\frac{2W^{3}}{\rho S}} \left(C_{D_0} C_L^{-3/2}+K C_L^{1/2}\right) = 0\\
        & \quad h_1 = C_L - C_{L_\mathrm{max}} \le 0 \\
        & \quad h_2 = \delta_T - 1 \le 0 \\
    \end{aligned}
    $$

    Below you can see the graph of the domain $0 \lt C_L \leq C_{L_{\mathrm{max}}}$ and $0 \lt \delta_T \leq 1$, with the surface $P$ and the constraint $g_1$ in red. Choose a simplified jet aircraft of your liking in the database below.
    """)
    return


@app.cell(hide_code=True)
def _(ac_table):
    ac_table
    return


@app.cell(hide_code=True)
def _(CL_slider, dT_slider):
    mo.md(rf"""
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
    \mathcal{L}(C_L, \delta_T, \lambda_1, \mu_1, \mu_2) = & P + \lambda_1 \left[T - D\right]+ \mu_1 (C_L - C_{L_\mathrm{max}}) +\mu_2 (\delta_T - 1)\\
    =&\quad \sqrt{\frac{2W^3}{\rho S}}\left(C_{D_0} C_L^{-3/2}+K C_L^{1/2}\right)(1 - \lambda_1) +\\
    & + \lambda_1 \delta_T P_{a0}\sigma^\beta \\
    & + \mu_1 (C_L - C_{L_\mathrm{max}}) + \\
    & + \mu_2 (\delta_T - 1) +\\
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The multipliers $\lambda_1, \mu_1, \mu_2$ have to meet the following conditions for an optimal solution of the optimization problem $(C_L^*, \delta_T^*)$ to exist.

    **A. Stationarity ($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \frac{\partial \mathcal{L}}{\partial C_L} = \sqrt{\frac{2W^3}{\rho S}}\left(-\frac{3}{2}C_{D_0}C_L^{-5/2} + \frac{1}{2} K C_L^{-1/2}\right)(1-\lambda_1) + \mu_1= 0$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1 P_{a0}\sigma^\beta+ \mu_2= 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $\displaystyle \delta_T P_{a0}\sigma^\beta - \sqrt{\frac{2W^{3}}{\rho S}} \left(C_{D_0} C_L^{-3/2}+K C_L^{1/2}\right) = 0$
    4.  $C_L - C_{L_\mathrm{max}} \le 0$
    5.  $\delta_T - 1 \le 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    6.  $\mu_1, \mu_2 \ge 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraint have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

    7.  $\mu_1 (C_L - C_{L_\mathrm{max}}) = 0$
    8. $\mu_2 (\delta_T - 1) = 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## KKT analysis

    We can now proceed to examine systematically the conditions where various inequality constraints are active
    or inactive.
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
    CLmax,
    CLopt_interior,
    CLopt_maxlift,
    CLopt_maxliftThrust,
    CLopt_maxthrust,
    active_selection,
    configTraces,
    constraint_maxliftThrust,
    constraint_maxthrust,
    dT_array,
    dTopt_interior,
    dTopt_maxlift,
    drag_curve,
    drag_yrange,
    h_interior_array,
    h_maxliftThrust,
    h_maxlift_array,
    h_maxthrust,
    h_selected,
    mach_trace,
    maxliftThrust_multiplier_selected,
    maxthrust_multiplier_selected,
    power_available_maxliftThrust,
    power_available_maxthrust,
    power_interior_harray,
    power_interior_selected,
    power_maxliftThrust_selected,
    power_maxlift_harray,
    power_maxlift_selected,
    power_maxthrust_selected,
    power_required_maxliftThrust,
    power_required_maxthrust,
    power_yrange,
    range_performance_diagrams,
    stall_trace,
    tab_value,
    thrust_available_maxliftThrust,
    thrust_available_maxthrust,
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
    velocity_stall_maxliftThrust_selected,
    velocity_stall_maxthrust_selected,
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
        power_surface_maxthrust = np.broadcast_to(
            power_required_maxthrust[np.newaxis, :],  # Shape: (101, 1)
            (len(CL_array), len(dT_array)),  # Target shape: (101, 101)
        )

        min_colorbar_maxthrust = np.min(power_required_maxthrust)
        max_colorbar_maxthrust = min_colorbar_maxthrust * 1.5
        zcolorbar_maxthrust = (min_colorbar_maxthrust, max_colorbar_maxthrust)

        configTraces_maxthrust = plot_utils.ConfigTraces(
            CL_array,
            dT_array,
            constraint_maxthrust,
            drag_curve,
            thrust_available_maxthrust,
            power_available_maxthrust,
            power_required_maxthrust,
            power_surface_maxthrust,
            velocity_maxthrust_CLarray,
            velocity_CL_P * maxthrust_multiplier_selected,
            velocity_CL_E * maxthrust_multiplier_selected,
            velocity_maxthrust_selected,
            velocity_stall_maxthrust_selected,
            (drag_yrange, power_yrange / 1e3, CLmax),
            zcolorbar_maxthrust,
            mach_trace,
            stall_trace,
        )

        # maxthrust graphics
        figure_optimum = OptimumGridView(
            configTraces_maxthrust,
            h_maxthrust,
            (velocity_maxthrust_CLarray, velocity_maxthrust_selected),
            (np.nan, power_maxthrust_selected * 1e3),
            (h_maxthrust, 1 * true_maxthrust, CLopt_maxthrust, true_maxthrust),
            f"Thrust-lift limited minimum drag for {active_selection.full_name}",
            equality=True,
        )
    elif tab_value == title_keys[3]:
        power_surface_maxliftThrust = np.broadcast_to(
            power_required_maxliftThrust[np.newaxis, :],  # Shape: (101, 1)
            (len(CL_array), len(dT_array)),  # Target shape: (101, 101)
        )

        min_colorbar_maxliftThrust = np.min(power_required_maxliftThrust)
        max_colorbar_maxliftThrust = min_colorbar_maxliftThrust * 1.5
        zcolorbar_maxliftThrust = (
            min_colorbar_maxliftThrust,
            max_colorbar_maxliftThrust,
        )

        configTraces_maxliftThrust = plot_utils.ConfigTraces(
            CL_array,
            dT_array,
            constraint_maxliftThrust,
            drag_curve,
            thrust_available_maxliftThrust,
            power_available_maxliftThrust,
            power_required_maxliftThrust,
            power_surface_maxliftThrust,
            velocity_maxliftThrust_CLarray,
            velocity_CL_P * maxliftThrust_multiplier_selected,
            velocity_CL_E * maxliftThrust_multiplier_selected,
            velocity_maxliftThrust_selected,
            velocity_stall_maxliftThrust_selected,
            (drag_yrange, power_yrange / 1e3, CLmax),
            zcolorbar_maxliftThrust,
            mach_trace,
            stall_trace,
        )

        # maxliftThrust graphics
        figure_optimum = OptimumGridView(
            configTraces_maxliftThrust,
            h_maxliftThrust,
            (velocity_maxliftThrust_CLarray, velocity_maxliftThrust_selected),
            (np.nan, power_maxliftThrust_selected * 1e3),
            (h_maxliftThrust, 1, CLopt_maxliftThrust, true_maxliftThrust),
            f"Thrust-lift limited minimum drag for {active_selection.full_name}",
            equality=True,
        )

    figure_optimum.update_axes_ranges(range_performance_diagrams)
    return (figure_optimum,)


@app.cell
def _(figure_optimum, tab_value, title_keys, variables_stack):
    if tab_value != title_keys[0]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ### _Interior solutions_ 

    In this case: $C_L \lt C_{L_{\mathrm{max}}}$, $\delta_T \lt 1$, $\mu_1=\mu_2= 0$

    from stationarity condition (2): $\lambda_1 = 0$

    from stationarity condition (1): 

    $$
    -\frac{3}{2}C_{D_0} C_L^{-5/2}+\frac{1}{2}KC_L^{-1/2}= 0 \quad \Rightarrow \quad KC_L^2 = 3C_{D_0} \quad \Rightarrow \quad C_L^* = \sqrt{\frac{3C_{D_0}}{K}} = \sqrt{3}C_{L_E} = C_{L_P}
    $$

    Before finding the corresponding $\delta_T$ value find the velocity associated with $C_{L_P}$:

    $$
    V_P = \sqrt{\frac{2W}{\rho S}}\sqrt[4]{\frac{K}{3C_{D_0}}}
    $$


    The optimum $\delta_T$ value is obtained from primal feasibility constraint (3). Using the velocity for minimum power, we find:

    $$
    \delta_T^* = \frac{W}{E_P}\frac{V_P}{P_{a0}\sigma^\beta}
    $$

    Where: $\displaystyle E_{\mathrm{P}} = \frac{\sqrt{3}}{2}E_{\mathrm{max}}$

    This is valid for:  

    $$
    \delta_T^*\lt 1 \quad \Leftrightarrow\quad  \frac{W^{3/2}}{\sigma^{\beta+1/2}} \lt P_{a0} E_P \sqrt{\frac{\rho_0SC_{L_P}}{2}} \;  = P_{a0} E_P \sqrt{\frac{\rho_0 S}{2}} \sqrt[4]{\frac{3 C_{D_0}}{K}}
    $$

    The value of the objective function, power, is calculated as:

    $$
    P^*_{\mathrm{min}} = DV = 4 \sqrt{\frac{2W^3}{S\sigma\rho_0}}\sqrt[4]{\frac{C_{D_0} K^3}{27}}
    $$

    This concludes the analysis for the minimum power of a simplified propeller aircraft in the interior case. Below is a summary of the optima:

    $$
    \boxed{C_L^* =  \sqrt{\frac{3C_{D_0}}{K}}}, \quad \boxed{\delta_T^* = \frac{W}{E_P}\frac{V_P}{P_{a0}\sigma^\beta}}, \quad \text{with} \quad V_P = \sqrt{\frac{2W}{\rho S}}\sqrt[4]{\frac{K}{3C_{D_0}}} \quad \text{for} \quad C_{L}^* \lt C_{L_\mathrm{max}} \quad \text{and}\quad \frac{W^{3/2}}{\sigma^{\beta+1/2}} \lt P_{a0} E_P \sqrt{\frac{\rho_0 S}{2}} \sqrt[4]{\frac{3 C_{D_0}}{K}}
    $$

    With the optimal value for minimum power: 

    $$
    P^*_{\mathrm{min}} = DV = 4 \sqrt{\frac{2W^3}{S\sigma\rho_0}}\sqrt[4]{\frac{C_{D_0} K^3}{27}}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.
    """),
            variables_stack,
            figure_optimum.figure,
        ]
    ).callout()
    return


@app.function
def interior_condition(
    W,
    h_selected,
    S,
    Pa0,
    beta,
    CL_P,
    E_P,
    velocity_CLP,
    power_available,  # input scalar
    CLmax,
    sigma_selected,
    h_array,
):
    sigma_interior = (
        W ** (1.5) / Pa0 / E_P / (np.sqrt(atmos.rho0 * S * CL_P / 2))
    ) ** (1 / (beta + 0.5))

    dT_interior = W / E_P * velocity_CLP / power_available / 1e3

    if CL_P > CLmax:
        return np.array([np.nan]), dT_interior, np.nan, np.nan

    h_interior = atmos.altitude(sigma_interior)

    h_interior_array = h_array[h_array < h_interior]

    h_min = h_interior_array.min()
    h_max = h_interior_array.max()
    cond = 1 if h_min <= h_selected <= h_max else np.nan
    return h_interior_array, dT_interior, CL_P, cond


@app.cell
def _(
    CL_P,
    CLmax,
    E_P,
    Pa0,
    S,
    W_selected,
    beta,
    h_array,
    h_selected,
    power_scalar,
    sigma_selected,
    velocity_CL_P,
):
    h_interior_array, dTopt_interior, CLopt_interior, true_interior = (
        interior_condition(
            W_selected,
            h_selected,
            S,
            Pa0,
            beta,
            CL_P,
            E_P,
            velocity_CL_P,
            power_scalar,  # input scalar
            CLmax,
            sigma_selected,
            h_array,
        )
    )

    velocity_interior_selected = velocity_CL_P * true_interior
    velocity_interior_harray = np.sqrt(
        2 * W_selected / (CL_P * S * atmos.rho(h_interior_array))
    )

    power_interior_harray = W_selected / E_P * velocity_interior_harray
    power_interior_selected = W_selected / E_P * velocity_interior_selected
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
def _(figure_optimum, tab_value, title_keys, variables_stack):
    if tab_value != title_keys[1]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ### _Lift-limited solutions (stall)_

    In this case: $C_L = C_{L_{\mathrm{max}}}$, $\delta_T \lt 1$, $\mu_1 \gt 0$, $\mu_2= 0$

    from stationarity condition (2): $\lambda_1 = 0$

    from stationarity condition (1):

    $$
    \mu_1 = \sqrt{\frac{2W^3}{\rho S}}\left(\frac{3}{2}C_{D_0}C_{L_{\mathrm{max}}}^{-5/2} - \frac{1}{2} K C_{L_{\mathrm{max}}}^{-1/2}\right) \gt 0
    $$

    $$
    \Rightarrow 3C_{D_0}C_{L_{\mathrm{max}}}^{-2} - K \gt 0 \quad  \Rightarrow \quad C_{L_{\mathrm{max}}} \lt \sqrt{\frac{3C_{D_0}}{K}} = \sqrt{3}C_{L_E} = C_{L_P}
    $$

    This means that, if we design an aircraft such that its $C_{L_P}$ is lower than its stall lift coefficient, then the minimum power required will be obtained at stall, because the aircraft is not able to fly at $C_{L_P}$ in steady level flight.
    In other words, an aircraft so designed would only be able to fly on the right branch of the power performance diagram, because the stall speed would be higher than the speed for minimum power. Therefore, the effective minimum power flyable in steady level flight would be obtained at the stall speed.

    We can now calculate the optimal $\delta_T^*$. As before, define the velocity at which the aircraft is flying for a cleaner solution. Note that $C_L = C_{L_{\mathrm{max}}}$ thus the aircraft is flying at stall speed $V_S$: 

    $$
    V_S= \sqrt{\frac{2W}{\rho S C_{L_{\mathrm{max}}}}}
    $$

    The correrponding $\delta_T^*$, found from the primal feasibility constraint (3): 

    $$
    \delta_T^* = \frac{W}{E_S}\frac{V_S}{P_{a0}\sigma^\beta}
    $$

    This is valid for: 

    $$
    \delta_T^*\lt 1 \Leftrightarrow \frac{W^{3/2}}{\sigma^{\beta+1/2}} \lt  \; P_{a0} \,E_S\sqrt{\frac{1}{2}\rho_0SC_{L_{\mathrm{max}}}}
    $$

    The value of the objective function, power, is calculated as:

    $$
    P^*_{\mathrm{min}} = DV = \frac{W}{E_S}V_S
    $$

    This concludes the analysis for the minimum power of a simplified propeller aircraft in the lift-limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = C_{L_{\mathrm{max}}}}, \quad \boxed{\delta_T^* = \frac{W}{E_S}\frac{V_S}{P_{a0}\sigma^\beta}}, \quad \text{for} \quad C_{L}^* \lt \sqrt{\frac{3C_{D_0}}{K}} \quad \text{and}\quad \frac{W^{3/2}}{\sigma^{\beta+1/2}} \lt  \; P_{a0} \,E_S\sqrt{\frac{1}{2}\rho_0SC_{L_{\mathrm{max}}}}
    $$

    With the optimal value for minimum power: 

    $$
    P^*_{\mathrm{min}}= \frac{W^{3/2}}{\sigma^{1/2}E_S}\sqrt{\frac{2}{\rho_0 S C_{L_\mathrm{max}}}}
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
    S,
    Pa0,
    beta,
    CL_P,
    E_S,
    velocity_stall,
    power_available,  # input scalar
    CLmax,
    sigma_selected,
    h_array,
):
    sigma_bound = (W ** (1.5) / Pa0 / E_S / (np.sqrt(atmos.rho0 * S * CLmax / 2))) ** (
        1 / (beta + 0.5)
    )

    dT = W / E_S * velocity_stall / power_available / 1e3

    if CLmax > CL_P:
        return np.array([np.nan]), dT, np.nan, np.nan

    h_maxlift = atmos.altitude(sigma_selected)

    h_array = h_array[h_array < h_maxlift]

    h_min = h_array.min()
    h_max = h_array.max()
    cond = 1 if h_min <= h_selected <= h_max else np.nan
    return h_array, dT, CLmax, cond


@app.cell
def _(
    CL_P,
    CLmax,
    E_S,
    Pa0,
    S,
    W_selected,
    beta,
    drag_curve,
    h_array,
    h_selected,
    power_available,
    rho_array,
    sigma_selected,
    velocity_stall_harray,
    velocity_stall_selected,
):
    h_maxlift_array, dTopt_maxlift, CLopt_maxlift, true_maxlift = maxlift_condition(
        W_selected,
        h_selected,
        S,
        Pa0,
        beta,
        CL_P,
        E_S,
        velocity_stall_selected,
        power_available,  # input scalar
        CLmax,
        sigma_selected,
        h_array,
    )

    maxlift_multiplier = np.sqrt(rho_array / atmos.rho(h_maxlift_array))

    velocity_maxlift_selected = velocity_stall_selected * true_maxlift
    velocity_maxlift_harray = velocity_stall_harray * maxlift_multiplier

    power_maxlift_harray = drag_curve[-1] * velocity_maxlift_harray
    power_maxlift_selected = drag_curve[-1] * velocity_maxlift_selected
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
def _(figure_optimum, tab_value, title_keys, variables_stack):
    if tab_value != title_keys[2]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ### _Thrust limited solutions_

    In this case: $C_L \lt C_{L_{\mathrm{max}}}$, $\delta_T = 1$, $\mu_1 = 0$, $\mu_2 \gt 0$

    from stationarity condition (2): $\displaystyle \lambda_1 = -\frac{\mu_2}{P_{a0}\sigma^\beta} \quad \Rightarrow \quad \lambda_1 \lt 0$

    Thus, from stationarity condition (1): 

    $$
    \sqrt{\frac{2W^3}{\rho S}}\left(-\frac{3}{2}C_{D_0}C_L^{-5/2} + \frac{1}{2} K C_L^{-1/2}\right)(1-\lambda_1) = 0 \quad \text{with } \quad 1-\lambda_1 \gt 0
    $$

    $$
    \Rightarrow -3C_{D_0}C_L^{-2} + K = 0
    $$


    $$
    C_L^* = \sqrt{\frac{3C_{D_0}}{K}} = \sqrt{3}C_{L_E} = C_{L_P}
    $$

    The condition for which this is true is found using the primal feasibility constraint (3). 

    $$
    \frac{W^{3/2}}{\sigma^{\beta+1/2}} = P_{a0} E_P \sqrt{\frac{\rho_0SC_{L_P}}{2}} \;  = P_{a0} E_P \sqrt{\frac{\rho_0 S}{2}} \sqrt[4]{\frac{3 C_{D_0}}{K}}
    $$

    This can be compared with what we found in the interior of the domain, showing the thrust-limited case represents the limit case of the interior optima.

    The value of the objective function, power, is calculated as:

    $$
    P^*_{\mathrm{min}} = DV = \frac{W}{E_P}\sqrt{\frac{W}{S}\frac{2}{\rho}\frac{1}{C_{L_P}}} = \frac{W^{3/2}}{E_P\sigma^{1/2}}\sqrt[4]{\frac{K}{3C_{D_0}}}\sqrt{\frac{2}{\rho_0S}}
    $$

    This concludes the analysis for the minimum power of a simplified propeller aircraft in the thrust-limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = \sqrt{\frac{3C_{D_0}}{K}}}, \quad \boxed{\delta_T^* =1}, \quad \text{for} \quad C_{L}^* \lt C_{L_\mathrm{max}} \quad \text{and}\quad \frac{W^{3/2}}{\sigma^{\beta+1/2}} = P_{a0} E_P \sqrt{\frac{\rho_0 S}{2}} \sqrt[4]{\frac{3 C_{D_0}}{K}}
    $$

    With the optimal value for minimum power: 

    $$
    P^*_{\mathrm{min}}= \frac{W^{3/2}}{E_P\sigma^{1/2}}\sqrt[4]{\frac{K}{3C_{D_0}}}\sqrt{\frac{2}{\rho_0S}}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.
    """),
            variables_stack,
            figure_optimum.figure,
        ]
    ).callout()
    return


@app.function
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
    sigma_optimum = (W ** (1.5) / Pa0 / E_P / (np.sqrt(atmos.rho0 * S * CL_P / 2))) ** (
        1 / (beta + 0.5)
    )

    dT = 1

    if CL_P >= CLmax or sigma_optimum < sigma_min:
        return np.array([np.nan]), dT, np.nan, np.nan, np.nan

    h_optimum = atmos.altitude(sigma_optimum)

    cond = 1

    return h_optimum, dT, CL_P, sigma_optimum, cond


@app.cell
def _(
    CL_P,
    CLmax,
    E_P,
    E_array,
    Pa0,
    S,
    W_selected,
    beta,
    drag_curve,
    meshgrid_n,
    min_sigma,
    rho_selected,
    velocity_CL_P,
    velocity_CLarray,
    velocity_stall_harray,
):
    # Max thrust
    h_maxthrust, dTopt_maxthrust, CLopt_maxthrust, sigma_maxthrust, true_maxthrust = (
        maxthrust_condition(W_selected, S, Pa0, beta, CL_P, E_P, CLmax, min_sigma)
    )

    maxthrust_multiplier_selected = np.sqrt(rho_selected / atmos.rho(h_maxthrust))

    velocity_maxthrust_CLarray = velocity_CLarray * maxthrust_multiplier_selected
    velocity_stall_maxthrust = velocity_stall_harray * maxthrust_multiplier_selected
    velocity_stall_maxthrust_selected = (
        velocity_CLarray[-1] * maxthrust_multiplier_selected
    )
    velocity_maxthrust_selected = velocity_CL_P * maxthrust_multiplier_selected

    power_required_maxthrust = drag_curve * velocity_maxthrust_CLarray / 1e3
    power_available_maxthrust = np.repeat(Pa0 * sigma_maxthrust**beta, meshgrid_n) / 1e3
    thrust_available_maxthrust = (
        power_available_maxthrust / velocity_maxthrust_CLarray * 1e3
    )

    constraint_maxthrust = W_selected / E_array / thrust_available_maxthrust

    power_maxthrust_selected = W_selected / E_P * velocity_maxthrust_selected / 1e3
    return (
        CLopt_maxthrust,
        constraint_maxthrust,
        h_maxthrust,
        maxthrust_multiplier_selected,
        power_available_maxthrust,
        power_maxthrust_selected,
        power_required_maxthrust,
        thrust_available_maxthrust,
        true_maxthrust,
        velocity_maxthrust_CLarray,
        velocity_maxthrust_selected,
        velocity_stall_maxthrust_selected,
    )


@app.cell
def _(figure_optimum, mass_stack):
    mo.vstack(
        [
            mo.md(r"""
    ### _Lift- and thrust- limited optimum_

    In this case: $C_L = C_{L_{\mathrm{max}}}$, $\delta_T = 1$, $\mu_1 \gt 0$, $\mu_2 \gt 0$

    from stationarity condition (2): $\displaystyle \lambda_1 = -\frac{\mu_2}{P_{a0}\sigma^\beta} \quad \Rightarrow \quad \lambda_1 \lt 0$

    Thus, from stationarity condition (1), since $1-\lambda_1 \gt 0$: 

    $$
    \mu_1 = \sqrt{\frac{2W^3}{\rho S}}\left(\frac{3}{2}C_{D_0}C_{L_{\mathrm{max}}}^{-5/2} - \frac{1}{2} K C_{L_{\mathrm{max}}}^{-1/2}\right)(1-\lambda_1)\gt 0
    $$

    $$
    \Rightarrow \quad  3 C_{D_0}C_{L_{\mathrm{max}}}^{-2} - K \gt 0
    $$


    $$
    \Rightarrow \quad  C_{L_{\mathrm{max}}} \lt \sqrt{\frac{3C_{D_0}}{K}} = \sqrt{3}C_{L_E} = C_{L_P}
    $$

    which shows once again that the necessary condition to obtain minimum power in stall conditions and maximum throttle. If it were otherwise ($C_{L_{\mathrm{max}}} > C_{L_P}$), it would be impossible to minimise power at stall and maximum thrust as the aircraft would reach the unconstrained minimum power before stalling.

    The condition for which this is true is found using the primal feasibility constraint (3). 

    $$
    \frac{W^{3/2}}{\sigma^{\beta+1/2}} = P_{a0}E_S \sqrt{\frac{1}{2}\rho_0SC_{L_{\mathrm{max}}}}
    $$

    The value of the objective function, power, is calculated as:

    $$
    P^*_{\mathrm{min}} = DV = \frac{W}{E_S}V_S
    $$

    This concludes the analysis for the minimum power of a simplified propeller aircraft in the lift-thrust limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = C_{L_\mathrm{max}}}, \quad \boxed{\delta_T^* =1}, \quad \text{for} \quad C_{L_{\mathrm{max}}} \lt \sqrt{\frac{3C_{D_0}}{K}} \quad \text{and}\quad \frac{W^{3/2}}{\sigma^{\beta+1/2}} = P_{a0} E_S \sqrt{\frac{\rho_0 S C_{L_\mathrm{max}}}{2}}
    $$

    With the optimal value for minimum power: 

    $$
    P^*_{\mathrm{min}}= \frac{W^{3/2}}{\sigma^{1/2}E_S}\sqrt{\frac{2}{\rho_0 S C_{L_\mathrm{max}}}}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.
    """),
            mass_stack,
            figure_optimum.figure,
        ]
    ).callout()
    return


@app.function
def maxliftThrust_condition(
    W,
    S,
    Pa0,
    beta,
    CL_P,
    E_S,
    CLmax,
    sigma_min,
):
    sigma_optimum = (
        W ** (1.5) / Pa0 / E_S / (np.sqrt(atmos.rho0 * S * CLmax / 2))
    ) ** (1 / (beta + 0.5))

    dT = 1

    h_optimum = atmos.altitude(sigma_optimum)

    if CLmax > CL_P or sigma_optimum < sigma_min:
        return h_optimum, dT, np.nan, sigma_optimum, np.nan

    cond = 1

    return h_optimum, dT, CLmax, sigma_optimum, cond


@app.cell
def _(
    CL_P,
    CLmax,
    E_P,
    E_S,
    E_array,
    Pa0,
    S,
    W_selected,
    beta,
    drag_curve,
    meshgrid_n,
    min_sigma,
    rho_selected,
    velocity_CL_P,
    velocity_CLarray,
    velocity_stall_harray,
):
    # Max lift Max thrust
    (
        h_maxliftThrust,
        dTopt_maxliftThrust,
        CLopt_maxliftThrust,
        sigma_maxliftThrust,
        true_maxliftThrust,
    ) = maxliftThrust_condition(W_selected, S, Pa0, beta, CL_P, E_S, CLmax, min_sigma)

    maxliftThrust_multiplier_selected = np.sqrt(
        rho_selected / atmos.rho(h_maxliftThrust)
    )

    velocity_maxliftThrust_CLarray = (
        velocity_CLarray * maxliftThrust_multiplier_selected
    )
    velocity_stall_maxliftThrust = (
        velocity_stall_harray * maxliftThrust_multiplier_selected
    )
    velocity_stall_maxliftThrust_selected = (
        velocity_CLarray[-1] * maxliftThrust_multiplier_selected
    )
    velocity_maxliftThrust_selected = velocity_CL_P * maxliftThrust_multiplier_selected

    power_required_maxliftThrust = drag_curve * velocity_maxliftThrust_CLarray / 1e3
    power_available_maxliftThrust = (
        np.repeat(Pa0 * sigma_maxliftThrust**beta, meshgrid_n) / 1e3
    )
    thrust_available_maxliftThrust = (
        power_available_maxliftThrust / velocity_maxliftThrust_CLarray * 1e3
    )

    constraint_maxliftThrust = W_selected / E_array / thrust_available_maxliftThrust

    power_maxliftThrust_selected = (
        W_selected / E_P * velocity_maxliftThrust_selected / 1e3
    )
    return (
        CLopt_maxliftThrust,
        constraint_maxliftThrust,
        h_maxliftThrust,
        maxliftThrust_multiplier_selected,
        power_available_maxliftThrust,
        power_maxliftThrust_selected,
        power_required_maxliftThrust,
        thrust_available_maxliftThrust,
        true_maxliftThrust,
        velocity_maxliftThrust_CLarray,
        velocity_maxliftThrust_selected,
        velocity_stall_maxliftThrust_selected,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Final flight Envelope
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
    h_interior_array,
    h_maxliftThrust,
    h_maxlift_array,
    h_maxthrust,
    mass_stack,
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
def _():
    mo.md(r"""
    ## Summary
    """)
    return


@app.cell
def _():
    mo.md(
        r"""
    | Name | Condition | $C_L^*$ | $\delta_T^*$ | $P^*$ |
    |:-|:-------|:-------:|:------------:|:-------|
    |Interior-optima    | $\displaystyle \quad C_L^* \lt C_{L_\mathrm{max}} \quad \text{and} \quad\frac{W^{3/2}}{\sigma^{\beta+1/2}} \lt P_{a0} E_P \sqrt{\frac{\rho_0 S}{2}} \sqrt[4]{\frac{3 C_{D_0}}{K}}$ | $\displaystyle \sqrt{\frac{3C_{D_0}}{K}}$ | $\displaystyle \frac{W}{E_P}\frac{V_P}{P_{a0}\sigma^\beta}$ | $\displaystyle 4 \sqrt{\frac{2W^3}{S\sigma\rho_0}}\sqrt[4]{\frac{C_{D_0} K^3}{27}}$ |
    |Lift-limited    |  $\displaystyle C_{L_\mathrm{max}} \lt \sqrt{\frac{3C_{D_0}}{K}} \quad \text{and}\quad \frac{W^{3/2}}{\sigma^{\beta+1/2}} \lt  \; P_{a0} \,E_S\sqrt{\frac{\rho_0SC_{L_{\mathrm{max}}}}{2}}$ | $C_{L_\mathrm{max}}$ | $\displaystyle \frac{W}{E_S}\frac{V_S}{P_{a0}\sigma^\beta}$ | $\displaystyle \frac{W^{3/2}}{\sigma^{1/2}E_S}\sqrt{\frac{2}{\rho_0 S C_{L_\mathrm{max}}}}$|
    |Thrust-limited    | $\displaystyle C_L^* \lt C_{L_\mathrm{max}} \quad \text{and}\quad \frac{W^{3/2}}{\sigma^{\beta+1/2}} = P_{a0} E_P \sqrt{\frac{\rho_0 S}{2}} \sqrt[4]{\frac{3 C_{D_0}}{K}}$ | $\displaystyle \sqrt{\frac{3C_{D_0}}{K}}$ | $1$ | $\displaystyle \frac{W^{3/2}}{E_P\sigma^{1/2}}\sqrt[4]{\frac{K}{3C_{D_0}}}\sqrt{\frac{2}{\rho_0S}}$ |
    |Thrust-lift limited    |  $\displaystyle \quad C_{L_{\mathrm{max}}} \lt \sqrt{\frac{3C_{D_0}}{K}} \quad \text{and} \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}} = P_{a0} E_S \sqrt{\frac{\rho_0 S C_{L_\mathrm{max}}}{2}}$ | $C_{L_\mathrm{max}}$ | $1$ | $\displaystyle \frac{W^{3/2}}{\sigma^{1/2}E_S}\sqrt{\frac{2}{\rho_0 S C_{L_\mathrm{max}}}}$|
    """
    ).center()
    return


@app.cell
def _():
    _defaults.nav_footer(
        before_file="MinPower_Prop.py",
        before_title="Minimum Power Simplified Jet",
        above_file="MinPower.py",
        above_title="Minimum Power Homepage",
        above_before=False,
    )
    return


if __name__ == "__main__":
    app.run()
