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

    V_slider = mo.ui.slider(
        0,
        a_0,
        step=10,
        label=r"$V$",
        value=0,
    )

    variables_stack = mo.hstack([mass_stack, h_slider])

    rho_array = atmos.rho(h_array)
    sigma_array = atmos.rhoratio(h_array)
    min_sigma = atmos.rhoratio(atmos.hmax)
    a_harray = atmos.a(h_array)

    # Visual computations
    mach_trace = plot_utils.create_mach_trace(h_array, a_harray)
    return (
        V_slider,
        a_0,
        ac_table,
        dT_array,
        dT_slider,
        data,
        h_array,
        h_slider,
        m_slider,
        mach_trace,
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
    return


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
    a_0,
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

    thrust_vector = power_scalar / velocity_CLarray * 1e3
    power_required = drag_curve * velocity_CLarray / 1e3

    constraint = W_selected / E_array / thrust_vector

    range_performance_diagrams = (drag_yrange, power_yrange, CLmax, 250)

    velocity_surface = np.broadcast_to(
        velocity_CLarray[np.newaxis, :],  # Shape: (101, 1)
        (len(CL_array), len(dT_array)),  # Target shape: (101, 101)
    )

    min_colorbar = np.min(velocity_CLarray)
    max_colorbar = a_0
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
        velocity_CLarray,
        velocity_surface,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Minimum airspeed: simplified piston propeller aircraft

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
        & \quad T_a(V,h) =  \frac{P_a(h)}{V} =  \frac{P_{a0}\sigma^\beta}{V} \\
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We could approach the solution of this problem in the same way we have approched the one for simplified jets: obtain the expression of $V$ from $c_1^\mathrm{eq}$, substitute it out of the whole problem, then proceed with deriving with respec to $C_L$ and $\delta_T$.
    In the case of propeller airplanes, this results in the following expression of the horizontal equilibrium contraint, which is unhandy to take derivatives with respect to $C_L$:

    $$
    \delta_T  \frac{P_{a0}\sigma^\beta}{V} - \frac{1}{2} \rho V^2 S \left( C_{D_0} + K C_L^2 \right) = 0
    \quad \Leftrightarrow \quad
    \delta_T  P_{a0}\sigma^\beta - \frac{1}{2} \rho S \left(\frac{2W}{\rho S C_L} \right)^{3/2}\left( C_{D_0} + K C_L^2 \right) = 0
    $$

    Instead, in this case, it is more convenient to reformulate the problem by eliminating $C_L$ instead of $V$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Problem reformulation

    From the vertical equilibrium equation:

    $$
    C_L = \frac{2W}{\rho V^2 S}
    $$

    The horizontal equilibrium equation then becomes:

    $$
    \delta_T  P_{a0}\sigma^\beta - \frac{1}{2} \rho V^3 S \left( C_{D_0} + \frac{4KW^2}{\rho^2 S^2  V^4}\right) = 0
    \quad \Leftrightarrow \quad
    \delta_T  P_{a0}\sigma^\beta - \frac{1}{2} \rho S V^3 C_{D_0} - \frac{2KW^2}{\rho S V} = 0
    $$

    The bounds on $C_L$ can be rewritten as the following inequality constraint:

    $$
    0 \le \frac{2W}{\rho V^2 S} \le C_{L_\mathrm{max}}
    $$

    where the left one is always verified, and the right one is equivalent to:

    $$
    V \ge \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}} = V_s
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## KKT Formulation

    $$
    \begin{aligned}
        \min_{V, \delta_T}
        & \quad V \\
        \text{subject to}
        & \quad g_1 = \delta_T  P_{a0}\sigma^\beta - \frac{1}{2} \rho S V^3 C_{D_0} - \frac{2KW^2}{\rho S V} = 0 \\
        & \quad h_1 = \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}} - V \le 0 \\
        & \quad h_2 = -\delta_T \le 0 \\
        & \quad h_3 = \delta_T - 1 \le 0 \\
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(ac_table):
    # Database cell (1)

    ac_table
    return


@app.cell
def _(
    CL_array,
    V_slider,
    a_0,
    active_selection,
    constraint,
    dT_slider,
    max_colorbar,
    min_colorbar,
    velocity_CLarray,
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
                y=velocity_CLarray,
                z=velocity_surface.T,
                opacity=0.9,
                name="Velocity",
                colorscale="viridis",
                cmin=min_colorbar,
                cmax=max_colorbar,
                colorbar={"title": "Velocity (m/s)"},
            ),
            go.Scatter3d(
                x=constraint,
                y=velocity_CLarray,
                z=velocity_surface[:, 0],
                opacity=0.7,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.1)", width=10),
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[constraint[10]],
                y=[velocity_CLarray[10]],
                z=[velocity_surface[10, 0] + 10],
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
                x=[dT_slider.value],
                y=[V_slider.value],
                z=[V_slider.value if V_slider.value > velocity_CLarray[-1] else np.nan],
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

    camera = dict(eye=dict(x=1.25, y=-1.25, z=1.25))

    fig_initial.update_layout(
        scene=dict(
            yaxis=dict(
                title=r"C<sub>L</sub> (-)",
                range=[xy_lowerbound, a_0],
            ),
            xaxis=dict(title="δ<sub>T</sub> (-)", range=[xy_lowerbound, 1]),
            zaxis=dict(title="V (m/s)", range=[0, a_0 + 15]),
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
def _(V_slider, dT_slider):
    mo.md(f"""
    Here you can modify the control variables to understand how it affects the design: {mo.hstack([V_slider, dT_slider])}
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

    The Lagrangian function combines the objective function with eqaulity constraints using Lagrange multipliers ($\lambda_i$) and inequality constraints using KKT multipliers ($\mu_j$).

    $$
    \begin{aligned}
    \mathcal{L}(V, \delta_T, \lambda_1, \mu_1, \mu_2, \mu_3) =
    \quad \frac{2W}{\rho S C_L}
    & + \\
    & + \lambda_1 \left(\delta_T  P_{a0}\sigma^\beta - \frac{1}{2} \rho S V^3 C_{D_0} - \frac{2KW^2}{\rho S V}\right) + \\
    & + \mu_1 \left( \frac{2W}{\rho S C_{L_\mathrm{max}}} - V \right) + \\
    & + \mu_2 (-\delta_T) + \\
    & + \mu_3 (\delta_T - 1) +\\
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **A. Stationarity conditions($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \frac{\partial \mathcal{L}}{\partial V} = 1 + \lambda_1 \left( \frac{2KW^2}{\rho S V^2} - \frac{3}{2}\rho V^2SC_{D_0} \right) -\mu_1 = 0$
    2. $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1  P_{a0}\sigma^\beta - \mu_2 + \mu_3 = 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $\displaystyle \delta_T  P_{a0}\sigma^\beta - \frac{1}{2} \rho S V^3 C_{D_0} - \frac{2KW^2}{\rho S V} = 0$
    4.  $\displaystyle \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}} - V \le 0$
    5.  $-\delta_T \le 0$
    6.  $\delta_T - 1 \le 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    8.  $\mu_1, \mu_2, \mu_3 \ge 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraint have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

    9.  $\displaystyle \mu_1\left( \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}} - V \right) = 0$
    10. $\mu_2 (\delta_T) = 0$
    11. $\mu_3 (\delta_T - 1) = 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## KKT Analysis

    We can now proceed to systematically examine the conditions where various inequality constraints are active or inactive.

    ### _Interior solutions_

    If all inequality constraints as inactive, $\mu_1,\mu_2,\mu_3=0$.
    From stationarity condition 2: $\lambda_1=0$. And from stationarity condition 2: $1=0$.
    Therefore, once again, optimal solutions lie on some boundary.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### _Idle thrust boundary active_

    In this case: $\mu_2 > 0, \delta_T=0, \mu_1=\mu_3=0$

    It is easy to see that the primal feasibility constraint 3, in other words the horizontal equilibrium, can never be verified.
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
    CLopt_maxlift,
    active_selection,
    configTraces,
    dTopt_maxlift,
    h_maxlift_array,
    h_selected,
    power_maxlift_harray,
    power_maxlift_selected,
    range_performance_diagrams,
    tab_value,
    title_keys,
    true_maxlift,
    velocity_maxlift_harray,
    velocity_maxlift_selected,
):
    if tab_value == title_keys[1]:
        # maxlift graphics
        figure_optimum = OptimumGridView(
            configTraces,
            h_selected,
            (velocity_maxlift_harray, velocity_maxlift_selected),
            (power_maxlift_harray, power_maxlift_selected),
            (h_maxlift_array, dTopt_maxlift, CLopt_maxlift, true_maxlift),
            f"Lift-limited minimum velocity for {active_selection.full_name}",
        )

        figure_optimum.update_axes_ranges(range_performance_diagrams)
    return (figure_optimum,)


@app.cell
def _(figure_optimum, tab_value, title_keys, variables_stack):
    if tab_value == title_keys[1]:
        mo.vstack(
            [
                mo.md(r"""
    ### _Stall boundary active_

    In this case: $\mu_1 > 0, V=V_s, \mu_2=\mu_3=0$

    From stationarity conditions: $\lambda_1=0 \Rightarrow \mu_1=0$, which is acceptable.

    The minimum airspeed is of course the stall speed, which seems trivial as a result of how we have reformulated the problem.

    $$
    V^* = V_s = \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}
    $$

    The corresponding optimum lift coefficient is $C_L^* = C_{L_\mathrm{max}}$ and the throttle setting is: 

    $$
    \delta_T^* = \frac{ \displaystyle \frac{1}{2}\rho V^3_s S C_{D_0} + \frac{2KW^2}{\rho S V_s} }{ P_{a0} \sigma^\beta} = 
    \frac{W V_s / E_S}{P_{a0}\sigma^\beta}
    $$

    The condition to achieve this is given by $0 \le \delta_T^* \le 1$, where only the right-hand side is relevant.
    This tells that the required power at stall speed has to be less or equal to the available power at stall speed, and is equivalent to either of the two following conditions:

    $$
    \frac{W^{3/2}}{\sigma^{\beta+1/2}} \le  P_{a0} E_S \sqrt{\frac{1}{2}\rho_0 S C_{L_\mathrm{max}}}
    \quad \Leftrightarrow \quad
    \frac{W}{\sigma^{\beta+1/2}} \le \frac{ P_{a0} E_S}{V_{s0}}
    $$
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
    E_S,
    velocity_stall,
    power_available,  # input scalar
    CLmax,
    min_sigma,
    sigma_selected,
    h_array,
):
    sigma_limit = (W ** (1.5) / Pa0 / E_S / (np.sqrt(atmos.rho0 * S * CLmax / 2))) ** (1 / (beta + 0.5))

    dTopt = W / E_S / power_available[0] / 1e3 * np.sqrt(2 * W / (atmos.rho(h_selected) * S * CLmax))

    if sigma_limit <= min_sigma:
        return np.array([np.nan]), dTopt, np.nan, np.nan

    h_bound = atmos.altitude(sigma_limit)
    h_maxlift_array = h_array[h_array < h_bound]

    h_min = h_maxlift_array.min()
    h_max = h_maxlift_array.max()
    cond = 1 if h_min <= h_selected <= h_max else np.nan
    return h_maxlift_array, dTopt, CLmax, cond


@app.cell
def _(
    CLmax,
    E_S,
    Pa0,
    S,
    W_selected,
    beta,
    h_array,
    h_selected,
    min_sigma,
    power_available,
    sigma_selected,
    velocity_CLarray,
    velocity_stall_harray,
):
    h_maxlift_array, dTopt_maxlift, CLopt_maxlift, true_maxlift = maxlift_condition(
        W_selected,
        h_selected,
        S,
        Pa0,
        beta,
        E_S,
        velocity_stall_harray,
        power_available,  # input scalar
        CLmax,
        min_sigma,
        sigma_selected,
        h_array,
    )

    velocity_maxlift_selected = velocity_CLarray[-1] * true_maxlift
    velocity_maxlift_harray = np.sqrt(2 * W_selected / (CLmax * S * atmos.rho(h_maxlift_array)))

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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### _Max throttle boundary active_

    In this case: $\mu_3 > 0, \delta_T=1, \mu_1=\mu_2=0$

    From the stationarity conditions and the complementary slack conditions:

    $$
    \mu_3 = -\lambda_1  P_{a0}\sigma^\beta\\
    1 + \lambda_1 \left( \frac{2KW^2}{\rho S V^2} - \frac{3}{2}\rho V^2SC_{D_0} \right) = 0
    $$

    Therefore, $\mu_3 > 0$ if $\lambda_1 < 0$, and the latter is true when:

    $$
    \frac{3}{2}\rho V^2SC_{D_0} - \frac{2KW^2}{\rho S V^2} < 0
    \quad \Leftrightarrow \quad 3 C_{D_0} - K C_L^2 < 0
    \quad \Leftrightarrow \quad C_L > \sqrt{\frac{3 C_{D_0}}{K}} = \sqrt{3} C_{L_E} = C_{L_P}
    $$

    This means that minimum speed is achieved at max throttle when flying on the induced branch of the power curve, that is with a lift coefficient that is higher than the one for minimum required power ($C_{L_P}$) and lower than $C_{L_\mathrm{max}}$) of course. This means that minimum speed is achieved at max throttle when flying on the induced branch of the power curve, that is with a lift coefficient that is higher than the one for minimum required power ($C_{L_P}$) and lower than $C_{L_\mathrm{max}}$).
    The corresponding aircraft design condition, which basically ensures the existence of the induced branch of the power curve, is therefore:

    $$
    C_{L_P} < C_L \le C_{L_\mathrm{max}}
    \quad \Rightarrow \quad
    C_{L_\mathrm{max}} > \sqrt{\frac{3 C_{D_0}}{K}}
    $$

    The corresponding minimum speed is obtained by solving the following equation:

    $$
    V^* :  P_{a0}\sigma^\beta - \frac{1}{2} \rho S V^3 C_{D_0} - \frac{2KW^2}{\rho S V} = 0
    $$

    which cannot be solved analytically.
    The solution is valid only if $V^* > V_s$.

    The corresponding throttle is $\delta_T^*=1$ and the optimum lift coefficient is: $C_L^* = \frac{2W}{\rho S V^{*2}}$
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### _Max throttle and stall boundaries active_

    In this case: $V=V_s, \delta_T=1, \mu_1 > 0, \mu_2 > 0, \mu_3 > 0$

    From the stationarity conditions and the complementary slack conditions:

    $$
    \mu_3 = -\lambda_1  P_{a0}\sigma^\beta > 0 \\
    \mu_1 = 1 + \lambda_1 \left[ \frac{1}{2}\rho V_s^2 S \left( K C_{L_\mathrm{max}}^2 - 3 C_{D_0}\right)\right] > 0
    $$

    It follows that:

    $$
    \frac{1}{\frac{1}{2}\rho V_s^2 S \left( K C_{L_\mathrm{max}}^2 - 3 C_{D_0}\right)} \le \lambda_1 < 0
    $$

    which corresponds to $C_{L_\mathrm{max}} > \sqrt{\frac{3 C_{D_0}}{K}} = C_{L_P}$.

    The condition in which this optimum is achieved is given by the horizontal equilibrium constraint, which states that the required power has to be equal to the available power in stall conditions and at max throttle. This results in the following equation:

    $$
    \frac{W^{3/2}}{\sigma^{\beta+1/2}} =  P_{a0} E_S \sqrt{\frac{1}{2}\rho_0 S C_{L_\mathrm{max}}}
    \quad \Leftrightarrow \quad
    \frac{W}{\sigma^{\beta+1/2}} = \frac{ P_{a0} E_S}{V_{s0}}
    $$
    """)
    return


@app.function
def maxlift_thrust_altitude(W, beta, Pa0, E_S, S, CLmax):
    sigma_exp = W**1.5 / Pa0 / E_S / np.sqrt(0.5 * atmos.rho0 * S * CLmax)

    sigma = sigma_exp ** (1 / (beta + 0.5))

    h = atmos.altitude(sigma)
    return np.where(h > 0, h, np.nan)


@app.cell
def _():
    mo.md(r"""
    Now after deriving all the optima for each condition we can summarize the flight envelopes in one graph, as shown below. Experiment with the weight of the aircrarft to understand how the theoretical ceiling for minimum speed moves in the graph.
    """)
    return


@app.cell
def _():
    _defaults.nav_footer(
        before_file="MinSpeed_Jet_KKT.py",
        before_title="Minimum Speed Simplified Jet",
        above_file="MinSpeed.py",
        above_title="Minimum Speed Homepage",
        above_before=False,
    )
    return


if __name__ == "__main__":
    app.run()
