import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # Initialization code that runs before all other cells
    import marimo as mo

    # Import dependencies
    from core import _defaults
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np
    from core import atmos
    from core import aircraft as ac
    from scipy.optimize import fsolve
    from core.aircraft import (
        velocity,
        power,
        drag,
        endurance,
    )

    # Set local/online filepath
    _defaults.FILEURL = _defaults.get_url()

    # Plotly dark mode template
    _defaults.set_plotly_template()

    # Data directory
    data_dir = str(mo.notebook_location() / "public" / "AircraftDB_Standard.csv")
    return (
        ac,
        atmos,
        data_dir,
        drag,
        endurance,
        fsolve,
        go,
        make_subplots,
        mo,
        np,
        power,
        velocity,
    )


@app.cell
def _():
    # Set navbar on the right
    _defaults.set_sidebar()
    return


@app.cell(hide_code=True)
def _(ac_table, data, mo):
    # Interactive elements (1)

    # Handle deselected row from table
    if ac_table.value is not None and ac_table.value.any().any():
        active_selection = ac_table.value.iloc[0]
    else:
        active_selection = data.iloc[0]

    # Interactive CL and \delta_T sliders
    CL_slider = mo.ui.slider(
        start=0,
        stop=active_selection["CLmax_ld"],
        step=0.2,
        label=r"$C_L$",
        value=0.5,
    )

    dT_slider = mo.ui.slider(start=0, stop=1, step=0.1, label=r"$\delta_T$", value=0.5)

    m_slider = mo.ui.slider(start=0, stop=1, step=0.1, label=r"", show_value=True)

    h_slider = h_slider = mo.ui.slider(
        start=0,
        stop=20,
        label=r"Altitude (km)",
        value=10,
        show_value=True,
    )

    # Create stacks
    mass_stack = mo.hstack(
        [mo.md("**OEW**"), m_slider, mo.md("**MTOW**")],
        align="start",
        justify="start",
    )

    variables_stack = mo.hstack([mass_stack, h_slider])
    return (
        CL_slider,
        active_selection,
        dT_slider,
        h_slider,
        m_slider,
        mass_stack,
        variables_stack,
    )


@app.cell(hide_code=True)
def _(
    active_selection,
    atmos,
    endurance,
    h_slider,
    m_slider,
    mo,
    np,
    velocity,
):
    # Variables declared
    meshgrid_n = 101
    xy_lowerbound = -0.1

    CL_array = np.linspace(0, active_selection["CLmax_ld"], meshgrid_n)  # -
    dT_array = np.linspace(0, 1, meshgrid_n)  # -
    h_array = np.linspace(0, 20e3, meshgrid_n)  # meters
    # Retrieve selected values
    # Compute selected weight
    W_selected = (
        active_selection["OEM"]
        + (active_selection["MTOM"] - active_selection["OEM"]) * m_slider.value
    ) * atmos.g0  # Netwons

    h_selected = int(h_slider.value * 1e3)  # meters
    step = h_array[1] - h_array[0]  # here it's 200
    idx_selected = int((h_selected - h_array[0]) / step)

    a = atmos.a(h_selected)
    a_harray = atmos.a(h_array)
    CD0 = active_selection["CD0"]
    S = active_selection["S"]
    K = active_selection["K"]
    CLmax = active_selection["CLmax_ld"]
    Pa0 = active_selection["Pa0"] * 1e3  # Watts
    beta = active_selection["beta"]
    CL_P = np.sqrt(3 * CD0 / K)
    CL_E = np.sqrt(CD0 / K)
    E_max = endurance(K, CD0, "max")
    E_P = (np.sqrt(3) / 2) * E_max
    E_S = CLmax / (CD0 + K * CLmax**2)
    velocity_stall_harray = velocity(W_selected, h_array, CLmax, S)
    velocity_stall_selected = float(velocity(W_selected, h_selected, CLmax, S))
    V_array = np.linspace(velocity_stall_selected, a, meshgrid_n)
    V_slider = mo.ui.slider(
        velocity_stall_selected,
        a,
        step=10,
        label=r"$V$",
        value=velocity_stall_selected,
    )
    return (
        CD0,
        CL_E,
        CL_P,
        CL_array,
        CLmax,
        E_S,
        K,
        Pa0,
        S,
        V_slider,
        W_selected,
        a,
        a_harray,
        beta,
        dT_array,
        h_array,
        h_selected,
        idx_selected,
        velocity_stall_harray,
        velocity_stall_selected,
        xy_lowerbound,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We could approach the solution of this problem in the same way we have approched the one for simplified jets: obtain the expression of $V$ from $c_1^\mathrm{eq}$, substitute it out of the whole problem, then proceed with deriving with respec to $C_L$ and $\delta_T$.
    In the case of propeller airplanes, this results in the following expression of the horizontal equilibrium contraint, which is unhandy to take derivatives with respect to $C_L$:

    $$
    \delta_T  \frac{P_{a0}\sigma^\beta}{V} - \frac{1}{2} \rho V^2 S \left( C_{D_0} + K C_L^2 \right) = 0
    \quad \Leftrightarrow \quad
    \delta_T  P_{a0}\sigma^\beta - \frac{1}{2} \rho S \left(\frac{2W}{\rho S C_L} \right)^{3/2}\left( C_{D_0} + K C_L^2 \right) = 0
    $$ 

    Instead, in this case, it is more convenient to reformulate the problem by eliminating $C_L$ instead of $V$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell(hide_code=True)
def _(ac, data_dir, mo):
    # Database cell (1)

    data = ac.available_aircrafts(data_dir, ac_type="Propeller")

    ac_table = mo.ui.table(
        data=data,
        pagination=True,
        show_column_summaries=False,
        selection="single",
        initial_selection=[0],
        page_size=4,
        show_data_types=False,
    )

    ac_table
    return ac_table, data


@app.cell(hide_code=True)
def _(atmos, np):
    def horizontal_constraint_minspeed(W, h, CD0, K, V, S, Pa0, beta):
        """Return dT array"""
        rho = atmos.rho(h)
        sigma = atmos.rhoratio(h)
        numerator = 0.5 * rho * S * CD0 * V**4 + 2 * K * W**2 / rho / S
        denominator = Pa0 * sigma**beta * V
        deltaT = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(V),
            where=V != 0,
        )
        return deltaT

    return (horizontal_constraint_minspeed,)


@app.cell
def _(
    CD0,
    CL_array,
    K,
    S,
    constraint,
    drag,
    h_selected,
    np,
    power,
    velocity_CLarray,
):
    power_curve = np.where(
        ~np.isnan(constraint),
        power(h_selected, S, CD0, K, CL_array, velocity_CLarray),
        np.nan,
    )
    drag_curve = drag(
        h_selected,
        S,
        CD0,
        K,
        CL_array,
        velocity_CLarray,
    )
    return drag_curve, power_curve


@app.cell(hide_code=True)
def _(
    CD0,
    CL_E,
    CL_P,
    CL_array,
    CL_slider,
    K,
    Pa0,
    S,
    W_selected,
    beta,
    dT_array,
    h_selected,
    horizontal_constraint_minspeed,
    np,
    velocity,
):
    # Computation cell (1)
    velocity_CLarray = velocity(W_selected, h_selected, CL_array, S, cap=True)

    velocity_user_selected = velocity(
        W_selected, h_selected, CL_slider.value, S, cap=False
    )

    constraint = horizontal_constraint_minspeed(
        W_selected, h_selected, CD0, K, velocity_CLarray, S, Pa0, beta
    )

    velocity_surface = np.tile(velocity_CLarray, (len(dT_array), 1)).T

    velocity_CL_E = float(velocity(W_selected, h_selected, CL_E, S, False))
    velocity_CL_P = float(velocity(W_selected, h_selected, CL_P, S, False))
    return constraint, velocity_CL_P, velocity_CLarray, velocity_surface


@app.cell(hide_code=True)
def _(
    V_slider,
    a,
    active_selection,
    constraint,
    dT_array,
    dT_slider,
    go,
    mo,
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
                x=dT_array,
                y=velocity_CLarray,
                z=velocity_surface,
                opacity=0.9,
                name="Velocity",
                colorscale="cividis",
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
                x=[constraint[-10]],
                y=[velocity_CLarray[-10]],
                z=[velocity_surface[-10, 0] + 10],
                opacity=1,
                textposition="middle left",
                mode="markers+text",
                text=["g<sub>1</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[dT_slider.value],
                y=[V_slider.value],
                z=[V_slider.value],  # Slightly elevate to show the full marker
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

    fig_initial.update_layout(
        scene=dict(
            yaxis=dict(
                title="V (m/s)",
                range=[xy_lowerbound, a],
            ),
            xaxis=dict(title="δ<sub>T</sub> (-)", range=[xy_lowerbound, 1]),
            zaxis=dict(title="V (m/s)"),
        ),
        title_text=active_selection["full_name"],
        title_x=0.5,
    )

    mo.output.clear()
    return (fig_initial,)


@app.cell(hide_code=True)
def _(V_slider, dT_slider, mo):
    mo.md(
        f"""Here you can modify the control variables to understand how it affects the design: {mo.hstack([V_slider, dT_slider])}"""
    )
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
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **A. Stationarity conditions($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \frac{\partial \mathcal{L}}{\partial V} = 1 + \lambda_1 \left( \frac{2KW^2}{\rho S V^2} - \frac{3}{2}\rho V^2SC_{D_0} \right) -\mu_1 = 0$
    2. $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1  P_{a0}\sigma^\beta - \mu_2 + \mu_3 = 0$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $\displaystyle \delta_T  P_{a0}\sigma^\beta - \frac{1}{2} \rho S V^3 C_{D_0} - \frac{2KW^2}{\rho S V} = 0$
    4.  $\displaystyle \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}} - V \le 0$
    5.  $-\delta_T \le 0$
    6.  $\delta_T - 1 \le 0$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    8.  $\mu_1, \mu_2, \mu_3 \ge 0$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraint have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

    9.  $\displaystyle \mu_1\left( \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}} - V \right) = 0$
    10. $\mu_2 (\delta_T) = 0$
    11. $\mu_3 (\delta_T - 1) = 0$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## KKT Analysis

    We can now proceed to systematically examine the conditions where various inequality constraints are active or inactive.

    ### _Interior solutions_ 

    If all inequality constraints as inactive, $\mu_1,\mu_2,\mu_3=0$. 
    From stationarity condition 2: $\lambda_1=0$. And from stationarity condition 2: $1=0$.
    Therefore, once again, optimal solutions lie on some boundary.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell(hide_code=True)
def _(atmos, np):
    def maxlift_condition(W, h, beta, S, Pa0, CLmax, E_S, CL_P):
        sigma = atmos.rhoratio(h)
        condition = ((W**1.5) / (sigma ** (beta + 0.5))) < (
            np.sqrt(0.5 * atmos.rho0 * S * CLmax) * Pa0 * E_S
        )
        return condition

    return (maxlift_condition,)


@app.cell(hide_code=True)
def _(
    CD0,
    CL_P,
    CLmax,
    E_S,
    K,
    Pa0,
    S,
    W_selected,
    beta,
    h_array,
    horizontal_constraint_minspeed,
    idx_selected,
    maxlift_condition,
    np,
    power,
    velocity,
):
    maxlift_mask = maxlift_condition(
        W_selected, h_array, beta, S, Pa0, CLmax, E_S, CL_P
    )

    CLopt_maxlift = np.where(maxlift_mask, CLmax, np.nan)

    velocity_maxlift_harray = velocity(W_selected, h_array, CLopt_maxlift, S)

    dTopt_maxlift = horizontal_constraint_minspeed(
        W_selected, h_array, CD0, K, velocity_maxlift_harray, S, Pa0, beta
    )

    CLopt_maxlift_selected = CLopt_maxlift[idx_selected]
    dTopt_maxlift_selected = dTopt_maxlift[idx_selected]

    velocity_maxlift_selected = velocity_maxlift_harray[idx_selected]

    power_maxlift_harray = power(
        h_array, S, CD0, K, CLopt_maxlift, velocity_maxlift_harray
    )
    power_maxlift_selected = power_maxlift_harray[idx_selected]
    return (
        CLopt_maxlift,
        dTopt_maxlift,
        dTopt_maxlift_selected,
        velocity_maxlift_harray,
        velocity_maxlift_selected,
    )


@app.cell(hide_code=True)
def _(
    CLopt_maxlift,
    a,
    a_harray,
    active_selection,
    atmos,
    constraint,
    dT_array,
    dTopt_maxlift,
    dTopt_maxlift_selected,
    go,
    h_array,
    h_selected,
    make_subplots,
    mo,
    np,
    velocity_CLarray,
    velocity_maxlift_harray,
    velocity_maxlift_selected,
    velocity_stall_harray,
    velocity_surface,
    xy_lowerbound,
):
    fig_maxlift_optimum = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]]
    )

    # Traces on the 3D plot
    fig_maxlift_optimum.add_traces(
        [
            go.Surface(
                x=dT_array,
                y=velocity_CLarray,
                z=velocity_surface,
                opacity=0.9,
                name="Velocity",
                colorscale="cividis",
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
                x=[constraint[-10]],
                y=[velocity_CLarray[-10]],
                z=[velocity_surface[-10, 0] + 10],
                opacity=1,
                textposition="middle left",
                mode="markers+text",
                text=["g<sub>1</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[dTopt_maxlift_selected],
                y=[velocity_maxlift_selected],
                z=[velocity_maxlift_selected],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=3,
                    color="white",
                    symbol="circle",
                ),
                name="Max Lift Optimum",
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub> : %{y}<br>P: %{z}<extra>%{fullData.name}</extra>",
            ),
            go.Scatter3d(
                x=[dTopt_maxlift_selected, xy_lowerbound],
                y=[velocity_maxlift_selected, velocity_maxlift_selected],
                z=[
                    velocity_maxlift_selected,
                    velocity_maxlift_selected,
                ],
                mode="lines",
                showlegend=False,
                line=dict(color="grey", width=2),
            ),
            go.Scatter3d(
                x=[dTopt_maxlift_selected, dTopt_maxlift_selected],
                y=[xy_lowerbound, velocity_maxlift_selected],
                z=[
                    velocity_maxlift_selected,
                    velocity_maxlift_selected,
                ],
                mode="lines",
                showlegend=False,
                line=dict(color="grey", width=2),
            ),
            go.Scatter3d(
                x=dTopt_maxlift,
                y=np.ones(len(velocity_maxlift_harray)) * xy_lowerbound,
                z=np.tile(velocity_maxlift_harray, len(CLopt_maxlift)),
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(129, 216, 208, 1)", width=8),
            ),
            go.Scatter3d(
                x=np.ones(len(CLopt_maxlift)) * xy_lowerbound,
                y=velocity_maxlift_harray,
                z=np.tile(velocity_maxlift_harray, len(CLopt_maxlift)),
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(129, 216, 208, 1)", width=8),
            ),
        ],
        cols=1,
        rows=1,
    )

    # Traces on the flight envelope
    fig_maxlift_optimum.add_traces(
        [
            go.Scatter(
                x=velocity_stall_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(width=1, color="rgba(255, 0, 0, 1)", dash="dash"),
                name="V<sub>stall</sub>",
                showlegend=False,
            ),
            go.Scatter(
                x=[velocity_stall_harray[-8]],
                y=[h_array[-8] / 1e3],
                mode="markers+text",
                marker=dict(size=1, color="rgba(255, 0, 0, 0)"),
                text=["V<sub>stall</sub>"],
                hoverinfo="skip",
                textposition="top left",
                showlegend=False,
            ),
            go.Scatter(
                x=a_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(color="rgba(255, 180, 90, 1)", width=2, dash="dash"),
                name="M1.0",
                showlegend=False,
            ),
            go.Scatter(
                x=[a_harray[-8] - 5],
                y=[h_array[-8] / 1e3],
                mode="markers+text",
                marker=dict(size=1, color="rgba(0, 0, 0, 0.0)"),
                text=["M1.0"],
                hoverinfo="skip",
                textposition="top left",
                showlegend=False,
            ),
            go.Scatter(
                x=velocity_maxlift_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(width=3, color="rgba(129, 216, 208, 1)"),
                showlegend=False,
                name="P_min",
            ),
            go.Scatter(
                x=[velocity_maxlift_selected],
                y=[h_selected / 1e3],
                mode="markers+text",
                marker=dict(size=5, color="white"),
                name="Max Lift Optimum",
                showlegend=False,
            ),
        ],
        cols=2,
        rows=1,
    )

    fig_maxlift_optimum.update_layout(
        scene=dict(
            yaxis=dict(
                title="V (m/s)",
                range=[xy_lowerbound, a],
            ),
            xaxis=dict(title="δ<sub>T</sub> (-)", range=[xy_lowerbound, 1]),
            zaxis=dict(title="V (m/s)"),
        ),
        xaxis=dict(
            title="V (m/s)",
            range=[xy_lowerbound, atmos.a(0) + 15],
            showgrid=True,
            gridcolor="#515151",
            gridwidth=1,
        ),
        yaxis=dict(
            title="h (km)",
            range=[xy_lowerbound, 20],
            showgrid=True,
            gridcolor="#515151",
            gridwidth=1,
        ),
        title_text=active_selection["full_name"],
        title_x=0.5,
    )

    mo.output.clear()
    return (fig_maxlift_optimum,)


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell
def _(fig_maxlift_optimum):
    fig_maxlift_optimum
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### _Idle thrust boundary active_

    In this case: $\mu_2 > 0, \delta_T=0, \mu_1=\mu_3=0$

    It is easy to see that the primal feasibility constraint 3, in other words the horizontal equilibrium, can never be verified.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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

    This means that minimum speed is achieved at max throttle when flying on the induced branch of the power curve, that is with a lift coefficient that is higher than the one for minimum required power ($C_{L_P}$) and lower than $C_{L_\mathrm{max}}$) of course.
    """
    )
    return


@app.cell(hide_code=True)
def _(CLmax, S, W_selected, h_selected, np, velocity, velocity_CLarray):
    CL_ticks = np.arange(0, CLmax + 1, 1)[1:-2]
    CL_ticks = np.append(CL_ticks, CLmax)
    text_cl_ticks = [str(tick) for tick in CL_ticks[:-1]]
    text_cl_ticks.append(r"$C_{L_\mathrm{max}}$")

    velocity_cl_line = np.append(
        velocity(W_selected, h_selected, CLmax, S) - 10,
        max(velocity_CLarray),
    )

    velocity_cl_array = velocity(W_selected, h_selected, np.array(CL_ticks), S)
    return text_cl_ticks, velocity_cl_array, velocity_cl_line


@app.cell(hide_code=True)
def _(
    a,
    active_selection,
    drag_curve,
    go,
    mo,
    power_curve,
    text_cl_ticks,
    velocity_CL_P,
    velocity_CLarray,
    velocity_cl_array,
    velocity_cl_line,
    velocity_stall_selected,
):
    fig_thrust_limited = go.Figure()

    # Power curve vs CL
    fig_thrust_limited.add_traces(
        [
            go.Scatter(x=velocity_CLarray, y=power_curve, name="Power"),
            go.Scatter(
                x=velocity_CLarray,
                y=drag_curve,
                name="Drag",
                yaxis="y2",
            ),
            go.Scatter(
                x=velocity_cl_line,
                y=[max(drag_curve) * 0.1 for i in range(len(velocity_cl_line))],
                showlegend=False,
                mode="lines",
                hoverinfo=None,
                line=dict(color="rgba(80, 103, 132, 0.35)"),
                yaxis="y2",
            ),
            go.Scatter(
                x=[min(velocity_cl_line)],
                y=[max(drag_curve) * 0.1],
                showlegend=False,
                mode="markers+text",
                line=dict(color="rgba(80, 103, 132, 0.35)"),
                hoverinfo=None,
                marker=dict(
                    symbol="arrow",
                    size=16,
                    angle=270,
                ),
                text=r"$C_L$",
                textposition="middle left",
                yaxis="y2",
            ),
            go.Scatter(
                x=velocity_cl_array,
                y=[max(drag_curve) * 0.08 for i in range(len(velocity_cl_array))],
                mode="markers+text",
                yaxis="y2",
                marker=dict(color="rgba(0, 0 ,0 ,0)"),
                text=text_cl_ticks,
                textposition="bottom center",
                showlegend=False,
            ),
            go.Scatter(
                x=velocity_cl_array,
                y=[max(drag_curve) * 0.1 for i in range(len(velocity_cl_array))],
                mode="markers",
                yaxis="y2",
                marker=dict(color="rgba(255, 255 ,255 ,1)", symbol="142"),
                showlegend=False,
            ),
        ]
    )

    # Add CL_P and CL_E curves
    fig_thrust_limited.add_vline(
        x=velocity_CL_P,
        line_dash="dot",
        annotation=dict(text="$C_{L_P}$", xshift=10, yshift=-10),
        line=dict(color="white"),
    )
    fig_thrust_limited.add_vrect(
        x0=velocity_stall_selected,
        x1=velocity_CL_P,
        fillcolor="green",
        opacity=0.25,
        line_width=0,
    )

    fig_thrust_limited.add_vline(
        x=velocity_stall_selected,
        line_dash="dot",
        annotation=dict(text=r"$V_{\mathrm{stall}}$", xshift=-50, yshift=-10),
        line=dict(color="rgba(255, 0, 0, 0.3)"),
    )

    # Axes configuration
    fig_thrust_limited.update_layout(
        legend=dict(
            x=0.01,  # Left edge
            y=1,  # Top edge
            xanchor="auto",
            yanchor="auto",
            bgcolor="rgba(0, 0, 0, 0.0)",  # Semi-transparent background
        ),
        xaxis=dict(title="Velocity (m/s)", range=[0, a]),
        yaxis=dict(title="Power (W)"),
        yaxis2=dict(title="Drag (N)", overlaying="y", side="right"),
        title_text=active_selection["full_name"],
        title_x=0.5,
    )
    mo.output.clear()
    return (fig_thrust_limited,)


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell
def _(fig_thrust_limited):
    fig_thrust_limited
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This means that minimum speed is achieved at max throttle when flying on the induced branch of the power curve, that is with a lift coefficient that is higher than the one for minimum required power ($C_{L_P}$) and lower than $C_{L_\mathrm{max}}$). 
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
    """
    )
    return


@app.cell
def _(
    CD0,
    CL_P,
    CLmax,
    K,
    Pa0,
    S,
    W_selected,
    atmos,
    beta,
    fsolve,
    h_array,
    horizontal_constraint_minspeed,
    idx_selected,
    np,
):
    velocity_maxthrust_harray = []
    for h in h_array:
        func = (
            lambda V: Pa0 * (atmos.rhoratio(h) ** beta)
            - 0.5 * atmos.rho(h) * S * CD0 * V**3
            - 2 * K * W_selected**2 / atmos.rho(h) / S / V
        )
        V_sol = fsolve(func, x0=250, xtol=1e-12, maxfev=20000)[0]
        velocity_maxthrust_harray.append(V_sol)

    # usage
    velocity_maxthrust_harray = np.asarray(velocity_maxthrust_harray)

    CL_maxthrust_harray = (
        2 * W_selected / atmos.rho(h_array) / S / velocity_maxthrust_harray**2
    )

    maxthrust_mask = (CL_maxthrust_harray < CL_P) & (CL_maxthrust_harray < CLmax)

    dT_maxthrust_check = (
        horizontal_constraint_minspeed(
            W_selected, h_array, CD0, K, velocity_maxthrust_harray, S, Pa0, beta
        )
        < 1.01
    )

    dTopt_maxthrust = np.where(dT_maxthrust_check, 1, np.nan)
    dTopt_maxthrust_selected = dTopt_maxthrust[idx_selected]
    velocity_maxthrust_harray = np.where(
        dT_maxthrust_check, velocity_maxthrust_harray, np.nan
    )
    velocity_maxthrust_selected = velocity_maxthrust_harray[idx_selected]
    return (
        dTopt_maxthrust,
        dTopt_maxthrust_selected,
        velocity_maxthrust_harray,
        velocity_maxthrust_selected,
    )


@app.cell(hide_code=True)
def _(
    a_harray,
    active_selection,
    atmos,
    constraint,
    dT_array,
    dTopt_maxthrust,
    dTopt_maxthrust_selected,
    go,
    h_array,
    h_selected,
    make_subplots,
    maxlift_thrust_h,
    mo,
    np,
    velocity_CLarray,
    velocity_maxthrust_harray,
    velocity_maxthrust_selected,
    velocity_stall_harray,
    velocity_surface,
    xy_lowerbound,
):
    # To finish
    fig_maxthrust_optimum = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]]
    )

    # Traces on the 3D plot, first four are template
    fig_maxthrust_optimum.add_traces(
        [
            go.Surface(
                x=dT_array,
                y=velocity_CLarray,
                z=velocity_surface,
                opacity=0.9,
                name="Velocity",
                colorscale="cividis",
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
                x=[constraint[-10]],
                y=[velocity_CLarray[-10]],
                z=[velocity_surface[-10, 0] + 10],
                opacity=1,
                textposition="middle left",
                mode="markers+text",
                text=["g<sub>1</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[dTopt_maxthrust_selected],
                y=[velocity_maxthrust_selected],
                z=[velocity_maxthrust_selected],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=3,
                    color="white",
                    symbol="circle",
                ),
                name="Max Lift Optimum",
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub> : %{y}<br>P: %{z}<extra>%{fullData.name}</extra>",
            ),
            go.Scatter3d(
                x=[dTopt_maxthrust_selected, xy_lowerbound],
                y=[velocity_maxthrust_selected, velocity_maxthrust_selected],
                z=[
                    velocity_maxthrust_selected,
                    velocity_maxthrust_selected,
                ],
                mode="lines",
                showlegend=False,
                line=dict(color="grey", width=2),
            ),
            go.Scatter3d(
                x=[dTopt_maxthrust_selected, dTopt_maxthrust_selected],
                y=[xy_lowerbound, velocity_maxthrust_selected],
                z=[
                    velocity_maxthrust_selected,
                    velocity_maxthrust_selected,
                ],
                mode="lines",
                showlegend=False,
                line=dict(color="grey", width=2),
            ),
            go.Scatter3d(
                x=dTopt_maxthrust,
                y=np.ones(len(velocity_maxthrust_harray)) * xy_lowerbound,
                z=np.tile(velocity_maxthrust_harray, len(velocity_maxthrust_harray)),
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(129, 216, 208, 1)", width=8),
            ),
            go.Scatter3d(
                x=np.ones(len(velocity_maxthrust_harray)) * xy_lowerbound,
                y=velocity_maxthrust_harray,
                z=np.tile(velocity_maxthrust_harray, len(velocity_maxthrust_harray)),
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(129, 216, 208, 1)", width=8),
            ),
        ],
        cols=1,
        rows=1,
    )

    # Traces on the flight envelope, first four traces are template
    fig_maxthrust_optimum.add_traces(
        [
            go.Scatter(
                x=velocity_stall_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(width=1, color="rgba(255, 0, 0, 1)", dash="dash"),
                name="V<sub>stall</sub>",
                showlegend=False,
            ),
            go.Scatter(
                x=[velocity_stall_harray[-8]],
                y=[h_array[-8] / 1e3],
                mode="markers+text",
                marker=dict(size=1, color="rgba(255, 0, 0, 0)"),
                text=["V<sub>stall</sub>"],
                hoverinfo="skip",
                textposition="top left",
                showlegend=False,
            ),
            go.Scatter(
                x=a_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(color="rgba(255, 180, 90, 1)", width=2, dash="dash"),
                name="M1.0",
                showlegend=False,
            ),
            go.Scatter(
                x=[a_harray[-8] - 5],
                y=[h_array[-8] / 1e3],
                mode="markers+text",
                marker=dict(size=1, color="rgba(0, 0, 0, 0.0)"),
                text=["M1.0"],
                hoverinfo="skip",
                textposition="top left",
                showlegend=False,
            ),
            go.Scatter(
                x=velocity_maxthrust_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(width=3, color="rgba(129, 216, 208, 1)"),
                showlegend=False,
                name="P_min",
            ),
            go.Scatter(
                x=[velocity_maxthrust_selected],
                y=[h_selected / 1e3],
                mode="markers+text",
                marker=dict(size=5, color="white"),
                name="Max Lift Optimum",
                showlegend=False,
            ),
        ],
        cols=2,
        rows=1,
    )

    fig_maxthrust_optimum.update_layout(
        scene=dict(
            yaxis=dict(
                title="V (m/s)",
                range=[xy_lowerbound, atmos.a(maxlift_thrust_h)],
            ),
            xaxis=dict(title="δ<sub>T</sub> (-)", range=[xy_lowerbound, 1]),
            zaxis=dict(title="V (m/s)"),
        ),
        xaxis=dict(
            title="V (m/s)",
            range=[xy_lowerbound, atmos.a(0) + 15],
            showgrid=True,
            gridcolor="#515151",
            gridwidth=1,
        ),
        yaxis=dict(
            title="h (km)",
            range=[xy_lowerbound, 20],
            showgrid=True,
            gridcolor="#515151",
            gridwidth=1,
        ),
        title_text=active_selection["full_name"],
        title_x=0.5,
    )

    mo.output.clear()
    return (fig_maxthrust_optimum,)


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell
def _(fig_maxthrust_optimum):
    fig_maxthrust_optimum
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### _Max throttle and stall boundaries active_

    In this case: $V=V_s, \delta_T=1, \mu_1 > 0, \mu_2 > 0, \mu_3 > 0$

    From the stationarity conditions and the complementary slack conditions:

    $$
    \mu_3 = -\lambda_1  P_{a0}\sigma^\beta > 0 \\
    \mu_1 = 1 + \lambda_1 \left[ \frac{1}{2}\rho V_s^2 S \left( K C_{L_\mathrm{max}}^2 - 3 C_{D_0}\right)\right] > 0
    $$

    It follows that:

    $$
    - \frac{1}{\frac{1}{2}\rho V_s^2 S \left( K C_{L_\mathrm{max}}^2 - 3 C_{D_0}\right)} \le \lambda_1 < 0
    $$

    which corresponds to $C_{L_\mathrm{max}} > \sqrt{\frac{3 C_{D_0}}{K}} = C_{L_P}$.

    The condition in which this optimum is achieved is given by the horizontal equilibrium constraint, which states that the required power has to be equal to the available power in stall conditions and at max throttle. This results in the following equation:

    $$
    \frac{W^{3/2}}{\sigma^{\beta+1/2}} =  P_{a0} E_S \sqrt{\frac{1}{2}\rho_0 S C_{L_\mathrm{max}}}
    \quad \Leftrightarrow \quad
    \frac{W}{\sigma^{\beta+1/2}} = \frac{ P_{a0} E_S}{V_{s0}}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(atmos, np):
    def maxlift_thrust_altitude(W, beta, Pa0, E_S, S, CLmax):
        sigma_exp = W**1.5 / Pa0 / E_S / np.sqrt(0.5 * atmos.rho0 * S * CLmax)

        sigma = sigma_exp ** (1 / (beta + 0.5))

        h = atmos.altitude(sigma)
        return np.where(h > 0, h, np.nan)

    return (maxlift_thrust_altitude,)


@app.cell(hide_code=True)
def _(
    CD0,
    CL_array,
    CLmax,
    E_S,
    K,
    Pa0,
    S,
    W_selected,
    beta,
    drag,
    horizontal_constraint_minspeed,
    maxlift_thrust_altitude,
    np,
    velocity,
):
    maxlift_thrust_h = maxlift_thrust_altitude(W_selected, beta, Pa0, E_S, S, CLmax)

    CLopt_maxlift_thrust = CLmax

    velocity_maxlift_thrust_selected = velocity(
        W_selected, maxlift_thrust_h, CLopt_maxlift_thrust, S, cap=True
    )

    velocity_CLarray_maxlift_thrust_h = velocity(
        W_selected, maxlift_thrust_h, CL_array, S, cap=True
    )

    dTopt_maxlift_thrust = 1

    drag_maxlift_thrust_h_curve = drag(
        maxlift_thrust_h, S, CD0, K, CL_array, velocity_CLarray_maxlift_thrust_h
    )

    constraint_maxlift_thrust = horizontal_constraint_minspeed(
        W_selected,
        maxlift_thrust_h,
        CD0,
        K,
        velocity_CLarray_maxlift_thrust_h,
        S,
        Pa0,
        beta,
    )

    velocity_maxlift_thrust_surface = np.tile(
        velocity_CLarray_maxlift_thrust_h, (len(CL_array), 1)
    ).T
    return (
        constraint_maxlift_thrust,
        dTopt_maxlift_thrust,
        maxlift_thrust_h,
        velocity_CLarray_maxlift_thrust_h,
        velocity_maxlift_thrust_selected,
        velocity_maxlift_thrust_surface,
    )


@app.cell(hide_code=True)
def _(
    a_harray,
    active_selection,
    atmos,
    constraint_maxlift_thrust,
    dT_array,
    dTopt_maxlift_thrust,
    go,
    h_array,
    make_subplots,
    maxlift_thrust_h,
    mo,
    velocity_CLarray_maxlift_thrust_h,
    velocity_maxlift_thrust_selected,
    velocity_maxlift_thrust_surface,
    velocity_stall_harray,
    xy_lowerbound,
):
    fig_maxlift_thrust_optimum = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]]
    )

    # Traces on the 3D plot, first four are template
    fig_maxlift_thrust_optimum.add_traces(
        [
            go.Surface(
                x=dT_array,
                y=velocity_CLarray_maxlift_thrust_h,
                z=velocity_maxlift_thrust_surface,
                opacity=0.9,
                name="Velocity",
                colorscale="cividis",
            ),
            go.Scatter3d(
                x=constraint_maxlift_thrust,
                y=velocity_CLarray_maxlift_thrust_h,
                z=velocity_maxlift_thrust_surface[:, 0],
                opacity=0.7,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.1)", width=10),
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[constraint_maxlift_thrust[-10]],
                y=[velocity_CLarray_maxlift_thrust_h[-10]],
                z=[velocity_maxlift_thrust_surface[-10, 0] + 10],
                opacity=1,
                textposition="middle left",
                mode="markers+text",
                text=["g<sub>1</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[dTopt_maxlift_thrust],
                y=[velocity_maxlift_thrust_selected],
                z=[
                    velocity_maxlift_thrust_selected
                ],  # Slightly elevate to show the full marker
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=3,
                    color="white",
                    symbol="circle",
                ),
                name="maxlift Optimum",
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub> : %{y}<br>V: %{z}<extra>%{fullData.name}</extra>",
            ),
        ],
        cols=1,
        rows=1,
    )

    # Traces on the flight envelope, first four traces are template
    fig_maxlift_thrust_optimum.add_traces(
        [
            go.Scatter(
                x=velocity_stall_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(width=1, color="rgba(255, 0, 0, 1)", dash="dash"),
                name="V<sub>stall</sub>",
                showlegend=False,
            ),
            go.Scatter(
                x=[velocity_stall_harray[-8]],
                y=[h_array[-8] / 1e3],
                mode="markers+text",
                marker=dict(size=1, color="rgba(255, 0, 0, 0)"),
                text=["V<sub>stall</sub>"],
                hoverinfo="skip",
                textposition="top left",
                showlegend=False,
            ),
            go.Scatter(
                x=a_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(color="rgba(255, 180, 90, 1)", width=2, dash="dash"),
                name="M1.0",
                showlegend=False,
            ),
            go.Scatter(
                x=[a_harray[-8] - 5],
                y=[h_array[-8] / 1e3],
                mode="markers+text",
                marker=dict(size=1, color="rgba(0, 0, 0, 0.0)"),
                text=["M1.0"],
                hoverinfo="skip",
                textposition="top left",
                showlegend=False,
            ),
            go.Scatter(
                x=[velocity_maxlift_thrust_selected],
                y=[maxlift_thrust_h / 1e3],
                mode="markers",
                line=dict(width=3, color="rgba(129, 216, 208, 1)"),
                showlegend=False,
                name="V_min",
            ),
        ],
        cols=2,
        rows=1,
    )

    fig_maxlift_thrust_optimum.update_layout(
        scene=dict(
            yaxis=dict(
                title="V (m/s)",
                range=[xy_lowerbound, atmos.a(maxlift_thrust_h)],
            ),
            xaxis=dict(title="δ<sub>T</sub> (-)", range=[xy_lowerbound, 1]),
            zaxis=dict(title="V (m/s)"),
        ),
        xaxis=dict(
            title="V (m/s)",
            range=[xy_lowerbound, atmos.a(0) + 15],
            showgrid=True,
            gridcolor="#515151",
            gridwidth=1,
        ),
        yaxis=dict(
            title="h (km)",
            range=[xy_lowerbound, 20],
            showgrid=True,
            gridcolor="#515151",
            gridwidth=1,
        ),
        title_text=active_selection["full_name"],
        title_x=0.5,
    )

    mo.output.clear()
    return (fig_maxlift_thrust_optimum,)


@app.cell(hide_code=True)
def _(mass_stack):
    mass_stack
    return


@app.cell(hide_code=True)
def _(fig_maxlift_thrust_optimum):
    fig_maxlift_thrust_optimum
    return


@app.cell(hide_code=True)
def _(
    a_harray,
    active_selection,
    atmos,
    go,
    h_array,
    maxlift_thrust_h,
    mo,
    velocity_maxlift_harray,
    velocity_maxlift_thrust_selected,
    velocity_maxthrust_harray,
    velocity_stall_harray,
    xy_lowerbound,
):
    fig_final_flightenv = go.Figure()

    fig_final_flightenv.add_traces(
        [
            go.Scatter(
                x=velocity_stall_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(width=1, color="rgba(255, 0, 0, 1)", dash="dash"),
                name="V<sub>stall</sub>",
                showlegend=False,
            ),
            go.Scatter(
                x=[velocity_stall_harray[-8]],
                y=[h_array[-8] / 1e3],
                mode="markers+text",
                marker=dict(size=1, color="rgba(255, 0, 0, 0)"),
                text=["V<sub>stall</sub>"],
                hoverinfo="skip",
                textposition="top left",
                showlegend=False,
            ),
            go.Scatter(
                x=a_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(color="rgba(255, 180, 90, 1)", width=2, dash="dash"),
                name="M1.0",
                showlegend=False,
            ),
            go.Scatter(
                x=[a_harray[-8] - 5],
                y=[h_array[-8] / 1e3],
                mode="markers+text",
                marker=dict(size=1, color="rgba(0, 0, 0, 0.0)"),
                text=["M1.0"],
                hoverinfo="skip",
                textposition="top left",
                showlegend=False,
            ),
            go.Scatter(
                x=velocity_maxlift_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(width=3, color="rgba(129, 216, 208, 1)"),
                showlegend=False,
                name="P_min interior",
            ),
            go.Scatter(
                x=[velocity_maxlift_thrust_selected],
                y=[maxlift_thrust_h / 1e3],
                mode="markers",
                marker=dict(size=7, color="rgba(129, 216, 208, 1)"),
                name="Max Thrust Optimum",
                showlegend=False,
            ),
            go.Scatter(
                x=velocity_maxthrust_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(width=3, color="rgba(129, 216, 208, 1)"),
                showlegend=False,
                name="V_min",
            ),
        ],
    )

    fig_final_flightenv.update_layout(
        xaxis=dict(
            title="V (m/s)",
            range=[xy_lowerbound, atmos.a(0) + 15],
            showgrid=True,
            gridcolor="#515151",
            gridwidth=1,
        ),
        yaxis=dict(
            title="h (km)",
            range=[xy_lowerbound, 20],
            showgrid=True,
            gridcolor="#515151",
            gridwidth=1,
        ),
        title_text=active_selection["full_name"],
        title_x=0.5,
    )

    mo.output.clear()
    return (fig_final_flightenv,)


@app.cell
def _(mo):
    mo.md(r"""Summarizing all the flight envelopes derived so far we obtain:""")
    return


@app.cell
def _(mass_stack):
    mass_stack
    return


@app.cell
def _(fig_final_flightenv):
    fig_final_flightenv
    return


if __name__ == "__main__":
    app.run()
