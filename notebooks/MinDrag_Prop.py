import marimo

__generated_with = "0.15.2"
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
    from scipy.optimize import root_scalar
    from core import atmos
    from core import aircraft as ac
    from core.aircraft import (
        velocity,
        horizontal_constraint,
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
        go,
        make_subplots,
        mo,
        np,
        root_scalar,
        velocity,
    )


@app.cell
def _():
    # Set navbar on the right
    _defaults.set_sidebar()
    return


@app.cell
def _(CL_slider, active_selection, atmos, endurance, h_slider, m_slider, np):
    # Variables declared
    meshgrid_n = 101
    xy_lowerbound = -0.1

    CL_array = np.linspace(0, active_selection["CLmax_ld"], meshgrid_n)  # -
    CL_array = np.where(CL_array < 5e-3, np.nan, CL_array)
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
    OEM = active_selection["OEM"]
    MTOM = active_selection["MTOM"]
    CL_P = np.sqrt(3 * CD0 / K)
    CL_E = np.sqrt(CD0 / K)
    E_max = endurance(K, CD0, "max")
    E_P = (np.sqrt(3) / 2) * E_max
    E_S = CLmax / (CD0 + K * CLmax)
    idx_CL_selected = int(CL_slider.value / 0.5)
    return (
        CD0,
        CL_E,
        CL_P,
        CL_array,
        CLmax,
        E_max,
        K,
        Pa0,
        S,
        W_selected,
        a_harray,
        beta,
        dT_array,
        h_array,
        h_selected,
        idx_selected,
        xy_lowerbound,
    )


@app.cell
def _(
    CL_E,
    CL_P,
    CL_array,
    CLmax,
    S,
    W_selected,
    h_array,
    h_selected,
    velocity,
):
    velocity_CL_E = float(velocity(W_selected, h_selected, CL_E, S, False))
    velocity_CL_P = float(velocity(W_selected, h_selected, CL_P, S, False))
    velocity_stall_selected = float(velocity(W_selected, h_selected, CLmax, S))
    velocity_CLarray = velocity(W_selected, h_selected, CL_array, S, False)
    velocity_stall_harray = velocity(W_selected, h_array, CLmax, S)
    return velocity_CLarray, velocity_stall_harray


@app.cell
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

    dT_slider = mo.ui.slider(
        start=0, stop=1, step=0.1, label=r"$\delta_T$", value=0.5
    )

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
def _(mo):
    mo.md(
        r"""
    # Minimum drag: simplified piston propeller aircraft

    The derivation for simplified piston propeller shows the same results as for simplified jet.

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
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## KKT formulation

    AS previously shown, we express $V$ from $c_1^\mathrm{eq}$ and substitute it out of the entire problem to eliminate it. The KKT formulation thus becomes:
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell(hide_code=True)
def _(ac, data_dir, mo):
    # Database cell
    data = ac.available_aircrafts(data_dir, ac_type="Propeller")[:8]

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


@app.cell
def _(CL_slider, dT_slider, mo):
    mo.md(
        rf"""Here you can modify the control variables to understand how it affects the design: {mo.hstack([dT_slider, CL_slider])}"""
    )
    return


@app.cell
def _(atmos, np):
    def g1_constraint(W, h, Pa0, beta, S, CD0, K, CL):
        numerator = W / CL * (CD0 + K * CL**2)

        sigma = atmos.rhoratio(h)
        rho = atmos.rho(h)

        denominator = Pa0 * sigma**beta * np.sqrt(S / W * rho / 2 * CL)

        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(CL),
            where=CL != 0,
        )
    return (g1_constraint,)


@app.cell
def _(
    CD0,
    CL_array,
    CL_slider,
    K,
    Pa0,
    S,
    W_selected,
    beta,
    drag,
    g1_constraint,
    h_selected,
    np,
    velocity,
    velocity_CLarray,
):
    variables_stackvelocity_user_selected = velocity(
        W_selected, h_selected, CL_slider.value, S, cap=False
    )

    drag_curve = drag(
        h_selected,
        S,
        CD0,
        K,
        CL_array,
        velocity_CLarray,
    )

    constraint = g1_constraint(
        W_selected, h_selected, Pa0, beta, S, CD0, K, CL_array
    )

    drag_surface = np.tile(drag_curve, (len(CL_array), 1))

    drag_selected = drag(
        h_selected,
        S,
        CD0,
        K,
        CL_slider.value,
        velocity(W_selected, h_selected, CL_slider.value, S),
    )
    return constraint, drag_curve, drag_selected, drag_surface


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
                colorscale="cividis",
            ),
            go.Scatter3d(
                x=CL_array,
                y=constraint,
                z=drag_surface[0],
                opacity=0.7,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.1)", width=10),
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[CL_array[-15]],
                y=[constraint[-15]],
                z=[drag_surface[0, -15]],
                opacity=1,
                textposition="middle left",
                mode="markers+text",
                text=["g<sub>1</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="g1 constraint",
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
            zaxis=dict(title="D (N)"),
        ),
        title_text=active_selection["full_name"],
        title_x=0.5,
    )

    mo.output.clear()
    return (fig_initial,)


@app.cell
def _(fig_initial):
    fig_initial
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    A necessary condition for an optimal solution of the optimization problem $(C_L^*, \delta_T^*)$ to exist, the multipliers $\lambda_1, \mu_1, \mu_2, \mu_3, \mu_4$ have to meet the following conditions:

    **A. Stationarity conditions($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \begin{aligned}\frac{\partial \mathcal{L}}{\partial C_L} & = W \frac{K C_L^2 - C_{D_0}}{C_L^2} + \lambda_1 \left( \frac{1}{2} \delta_T P_{a0}\sigma^\beta \sqrt{\frac{\rho S}{2WC_L}} - W \frac{K C_L^2 - C_{D_0}}{C_L^2} \right) + \mu_1 - \mu_2 \\
    & = W\frac{K C_L^2 - C_{D_0}}{C_L^2} (1 -\lambda_1) +  \frac{1}{2} \lambda_1\delta_T P_{a0}\sigma^\beta \sqrt{\frac{\rho S}{2WC_L}} +\mu_1 - \mu_2 = 0 \end{aligned}$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1 P_{a0} \sigma^\beta \sqrt{\frac{\rho S C_L}{2W}}+\mu_3-\mu_4 = 0$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $\displaystyle \delta_T P_{a0}\sigma^\beta\sqrt{\frac{\rho S C_L}{2W}} - W\frac{C_{D_0} +K C_L^2}{C_L} = 0$
    4.  $C_L - C_{L_\mathrm{max}} \le 0$
    5.  $-C_L \le 0$
    6.  $\delta_T - 1 \le 0$
    7.  $-\delta_T \le 0$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    8.  $\mu_1, \mu_2, \mu_3, \mu_4 \ge 0$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraint have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

    9.  $\mu_1 (C_L - C_{L_\mathrm{max}}) = 0$
    10. $\mu_2 (-C_L) = 0$
    11. $\mu_3 (\delta_T - 1) = 0$
    12. $\mu_4 (-\delta_T) = 0$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    As seen in previous analyses, it is evident that $\mu_2$ and $\mu_4$ can never be active, as we would have an unfeasible situation ($C_L = \delta_T = 0$). In other words, for aircraft flight: $C_L \gt 0 \wedge \delta_T \gt 0$, stricly. Therefore we can simplify the analysis by setting these two KKT multipliers to zero: 

    $$
    \begin{aligned}
    \mu_2 = \mu_4 = 0
    \end{aligned}
    $$

    We can now rewrite the new conditions to simplify the problem.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Simplified conditions**

    1. $\displaystyle W\frac{K C_L^2 - C_{D_0}}{C_L^2} (1 -\lambda_1) +  \frac{1}{2} \lambda_1\delta_T P_{a0}\sigma^\beta \sqrt{\frac{\rho S}{2WC_L}} +\mu_1 = 0$
    2. $\displaystyle \lambda_1 P_{a0} \sigma^\beta \sqrt{\frac{\rho S C_L}{2W}}+\mu_3 = 0$
    3. $\displaystyle \delta_T P_{a0}\sigma^\beta\sqrt{\frac{\rho S C_L}{2W}} - W\frac{C_{D_0} +K C_L^2}{C_L} = 0$
    4. $C_L - C_{L_\mathrm{max}} \le 0$
    5. $\delta_T - 1 \le 0$
    6. $\mu_1, \mu_3 \ge 0$
    7. $\mu_1 (C_L - C_{L_\mathrm{max}}) = 0$
    8. $\mu_3 (\delta_T - 1) = 0$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## KKT analysis

    We can now proceed to systematically examine the conditions where various inequality constraints are active or inactive.

    ### _Interior solutions_ 

    Assuming that that $0 < C_L < C_{L_\mathrm{max}}$ and $0 < \delta_T < 1$ is equivalent to consider all inequality constraints as inactive.

    Therefore: $\mu_1, \mu_3 =0$. 

    From stationarity condition (2): $\lambda_1 = 0$.

    From stationarity condition (1) it is possible to obtain the value of $C_L^*$ for minimum drag.

    $$
    C_L^* = \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$

    Notice how the optimal $C_L^*$ has the **same value** for maximum aerodynamic efficiency, or maximum $C_L/C_D$, for 
    $0\lt C_L \lt  C_{L_{max}} \wedge 0 \lt \delta_T \lt 1$, as shown in [aerodynamic efficiency](/?file=AerodynamicEfficiency.py).
    """
    )
    return


@app.cell
def _(atmos, np):
    def interior_condition(W, h, beta, S, Pa0, CD0, K, CLmax, CL_E):
        sigma = atmos.rhoratio(h)
        rho = atmos.rho(h)
        condition = (
            (W**1.5 / (sigma**beta) / (rho**0.5))
            < np.sqrt(S / 2) * (Pa0 / (2 * (CD0**0.25) * (K**0.75)))
        ) & (CL_E < CLmax)

        return condition
    return (interior_condition,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The corresponding airspeed is

    $$
    V^* = V_E = \sqrt{\frac{2}{\rho}\frac{W}{S}\frac{1}{C_{L_E}}} = \sqrt{\frac{2}{\rho}\frac{W}{S}}\sqrt[4]{\frac{K}{C_{D_0}}}
    $$

    The corresponding $\delta_T^*$ is found by solving the primal feasibility constraint (3) and using $C_L = C_L^*$.


    $$
    \delta_T^* = \frac{W}{E_\mathrm{max}}\frac{V_E}{P_{a0}\sigma^\beta} = \frac{W^{3/2}}{\sigma^{\beta+1/2}}\frac{1}{E_\mathrm{max}P_{a0}}\sqrt{\frac{2}{\rho_0 S C_{L_E}}}
    $$

    This value is compliant with the primal feasibility constraint (5) for: 

    $$
    \delta_T^* < 1 \Leftrightarrow 
    \frac{W^{3/2}}{\sigma^{\beta+1/2}}
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
    """
    )
    return


@app.cell
def _(
    CD0,
    CL_E,
    CLmax,
    E_max,
    K,
    Pa0,
    S,
    W_selected,
    atmos,
    beta,
    h_array,
    h_selected,
    idx_selected,
    interior_condition,
    np,
    velocity,
):
    interior_mask = interior_condition(
        W_selected, h_array, beta, S, Pa0, CD0, K, CLmax, CL_E
    )

    CLopt_interior = np.where(interior_mask, CL_E, np.nan)

    velocity_interior_harray = velocity(
        W_selected, h_array, CLopt_interior, S, cap=False
    )

    drag_interior_harray = np.where(interior_mask, W_selected / E_max, np.nan)

    dTopt_interior = (
        W_selected
        / Pa0
        / (atmos.rhoratio(h_array) ** beta)
        * 2
        * (CD0**0.25)
        * (K**0.75)
        / np.sqrt(atmos.rho(h_selected) * S / 2 / W_selected)
    )

    CLopt_interior_selected = CLopt_interior[idx_selected]
    dTopt_interior_selected = dTopt_interior[idx_selected]

    velocity_interior_selected = velocity_interior_harray[idx_selected]

    drag_interior_selected = drag_interior_harray[idx_selected]
    return (
        CLopt_interior,
        CLopt_interior_selected,
        dTopt_interior,
        dTopt_interior_selected,
        drag_interior_harray,
        drag_interior_selected,
        velocity_interior_harray,
        velocity_interior_selected,
    )


@app.cell
def _(
    CL_array,
    CLopt_interior,
    CLopt_interior_selected,
    a_harray,
    active_selection,
    constraint,
    dT_array,
    dTopt_interior,
    dTopt_interior_selected,
    drag_curve,
    drag_interior_harray,
    drag_interior_selected,
    drag_surface,
    go,
    h_array,
    h_selected,
    make_subplots,
    mo,
    np,
    velocity_interior_harray,
    velocity_interior_selected,
    velocity_stall_harray,
    xy_lowerbound,
):
    fig_interior_optimum = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]]
    )

    # Traces on the 3D plot, first four are template
    fig_interior_optimum.add_traces(
        [
            go.Surface(
                x=CL_array,
                y=dT_array,
                z=drag_surface,
                opacity=0.9,
                name="Drag",
                colorscale="cividis",
            ),
            go.Scatter3d(
                x=CL_array,
                y=constraint,
                z=drag_curve,
                opacity=0.7,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.1)", width=10),
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[CL_array[-15]],
                y=[constraint[-15]],
                z=[drag_surface[0, -15]],
                opacity=1,
                textposition="middle left",
                mode="markers+text",
                text=["g<sub>1</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[CLopt_interior_selected],
                y=[dTopt_interior_selected],
                z=[drag_interior_selected],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=3,
                    color="white",
                    symbol="circle",
                ),
                name="Interior Optimum",
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub> : %{y}<br>D: %{z}<extra>%{fullData.name}</extra>",
            ),
            go.Scatter3d(
                x=[CLopt_interior_selected, CLopt_interior_selected],
                y=[dTopt_interior_selected, xy_lowerbound],
                z=[
                    drag_interior_selected,
                    drag_interior_selected,
                ],
                mode="lines",
                showlegend=False,
                line=dict(color="grey", width=2),
            ),
            go.Scatter3d(
                x=[xy_lowerbound, CLopt_interior_selected],
                y=[dTopt_interior_selected, dTopt_interior_selected],
                z=[
                    drag_interior_selected,
                    drag_interior_selected,
                ],
                mode="lines",
                showlegend=False,
                line=dict(color="grey", width=2),
            ),
            go.Scatter3d(
                x=CLopt_interior,
                y=np.ones(len(dT_array)) * xy_lowerbound,
                z=np.tile(drag_interior_harray, len(CLopt_interior)),
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(129, 216, 208, 1)", width=8),
            ),
            go.Scatter3d(
                x=np.ones(len(CLopt_interior)) * xy_lowerbound,
                y=dTopt_interior,
                z=np.tile(drag_interior_harray, len(CLopt_interior)),
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(129, 216, 208, 1)", width=8),
            ),
        ],
        cols=1,
        rows=1,
    )

    # Traces on the flight envelope, first four traces are template
    fig_interior_optimum.add_traces(
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
                x=[a_harray[-8]],
                y=[h_array[-8] / 1e3],
                mode="markers+text",
                marker=dict(size=1, color="rgba(0, 0, 0, 0.0)"),
                text=["M1.0"],
                hoverinfo="skip",
                textposition="top left",
                showlegend=False,
            ),
            go.Scatter(
                x=velocity_interior_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(width=3, color="rgba(129, 216, 208, 1)"),
                showlegend=False,
                name="D_min",
            ),
            go.Scatter(
                x=[velocity_interior_selected],
                y=[h_selected / 1e3],
                mode="markers+text",
                marker=dict(size=5, color="white"),
                name="Interior Optimum",
                showlegend=False,
            ),
        ],
        cols=2,
        rows=1,
    )

    fig_interior_optimum.update_layout(
        scene=dict(
            xaxis=dict(
                title="C<sub>L</sub> (-)",
                range=[xy_lowerbound, active_selection["CLmax_ld"]],
            ),
            yaxis=dict(title="δ<sub>T</sub> (-)", range=[xy_lowerbound, 1]),
            zaxis=dict(
                title="D (N)",
            ),
        ),
        xaxis=dict(
            title="V (m/s)",
            range=[
                xy_lowerbound,
                max(a_harray.max(), np.nanmax(velocity_interior_harray)) + 15,
            ],
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
    return (fig_interior_optimum,)


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell
def _(fig_interior_optimum):
    fig_interior_optimum
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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

    Thus, minimum drag for a simplified propeller, in a lift-limited scenario, can be achieved on the following conditions: 

    $$
    \boxed{C_L^* = C_{L_\mathrm{max}}}, \quad \boxed{\delta_T^* =  \frac{W}{E_S}\frac{V_S}{P_{a0}\sigma^\beta}}, \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}} \lt P_{a0}E_S \sqrt{\frac{\rho_0SC_{L_\mathrm{max}}}{2}}, \quad C_{L_\mathrm{max}} \lt \sqrt{\frac{C_{D_0}}{K}}
    $$
    """
    )
    return


@app.cell
def _(atmos, np):
    def maxlift_condition(W, h, beta, S, Pa0, CD0, K, CLmax):
        sigma = atmos.rhoratio(h)
        E_S = CLmax / (CD0 + K * CLmax**2)
        condition = (
            W**1.5 / (sigma ** (beta + 0.5))
            < (np.sqrt(2) * Pa0 * E_S * np.sqrt(atmos.rho0 * S * CLmax) / 2)
        ) & (CLmax < np.sqrt(CD0 / K))

        return condition
    return (maxlift_condition,)


@app.cell
def _(
    CD0,
    CL_E,
    CLmax,
    E_max,
    K,
    Pa0,
    S,
    W_selected,
    atmos,
    beta,
    h_array,
    h_selected,
    idx_selected,
    maxlift_condition,
    np,
    velocity,
):
    maxlift_mask = maxlift_condition(
        W_selected, h_array, beta, S, Pa0, CD0, K, CLmax
    )

    CLopt_maxlift = np.where(maxlift_mask, CL_E, np.nan)

    velocity_maxlift_harray = velocity(
        W_selected, h_array, CLopt_maxlift, S, cap=False
    )

    drag_maxlift_harray = np.where(maxlift_mask, W_selected / E_max, np.nan)

    dTopt_maxlift = (
        W_selected
        / Pa0
        / (atmos.rhoratio(h_array) ** beta)
        * 2
        * (CD0**0.25)
        * (K**0.75)
        / np.sqrt(atmos.rho(h_selected) * S / 2 / W_selected)
    )

    CLopt_maxlift_selected = CLopt_maxlift[idx_selected]
    dTopt_maxlift_selected = dTopt_maxlift[idx_selected]

    velocity_maxlift_selected = velocity_maxlift_harray[idx_selected]

    drag_maxlift_selected = drag_maxlift_harray[idx_selected]
    return (
        CLopt_maxlift,
        CLopt_maxlift_selected,
        dTopt_maxlift,
        dTopt_maxlift_selected,
        drag_maxlift_harray,
        drag_maxlift_selected,
        velocity_maxlift_harray,
        velocity_maxlift_selected,
    )


@app.cell
def _(
    CL_array,
    CLopt_maxlift,
    CLopt_maxlift_selected,
    a_harray,
    active_selection,
    constraint,
    dT_array,
    dTopt_maxlift,
    dTopt_maxlift_selected,
    drag_curve,
    drag_maxlift_harray,
    drag_maxlift_selected,
    drag_surface,
    go,
    h_array,
    h_selected,
    make_subplots,
    mo,
    np,
    velocity_maxlift_harray,
    velocity_maxlift_selected,
    velocity_stall_harray,
    xy_lowerbound,
):
    fig_maxlift_optimum = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]]
    )

    # Traces on the 3D plot, first four are template
    fig_maxlift_optimum.add_traces(
        [
            go.Surface(
                x=CL_array,
                y=dT_array,
                z=drag_surface,
                opacity=0.9,
                name="Drag",
                colorscale="cividis",
            ),
            go.Scatter3d(
                x=CL_array,
                y=constraint,
                z=drag_curve,
                opacity=0.7,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.1)", width=10),
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[CL_array[-15]],
                y=[constraint[-15]],
                z=[drag_surface[0, -15]],
                opacity=1,
                textposition="middle left",
                mode="markers+text",
                text=["g<sub>1</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[CLopt_maxlift_selected],
                y=[dTopt_maxlift_selected],
                z=[drag_maxlift_selected],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=3,
                    color="white",
                    symbol="circle",
                ),
                name="maxlift Optimum",
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub> : %{y}<br>D: %{z}<extra>%{fullData.name}</extra>",
            ),
            go.Scatter3d(
                x=[CLopt_maxlift_selected, CLopt_maxlift_selected],
                y=[dTopt_maxlift_selected, xy_lowerbound],
                z=[
                    drag_maxlift_selected,
                    drag_maxlift_selected,
                ],
                mode="lines",
                showlegend=False,
                line=dict(color="grey", width=2),
            ),
            go.Scatter3d(
                x=[xy_lowerbound, CLopt_maxlift_selected],
                y=[dTopt_maxlift_selected, dTopt_maxlift_selected],
                z=[
                    drag_maxlift_selected,
                    drag_maxlift_selected,
                ],
                mode="lines",
                showlegend=False,
                line=dict(color="grey", width=2),
            ),
            go.Scatter3d(
                x=CLopt_maxlift,
                y=np.ones(len(dT_array)) * xy_lowerbound,
                z=np.tile(drag_maxlift_harray, len(CLopt_maxlift)),
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(129, 216, 208, 1)", width=8),
            ),
            go.Scatter3d(
                x=np.ones(len(CLopt_maxlift)) * xy_lowerbound,
                y=dTopt_maxlift,
                z=np.tile(drag_maxlift_harray, len(CLopt_maxlift)),
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(129, 216, 208, 1)", width=8),
            ),
        ],
        cols=1,
        rows=1,
    )

    # Traces on the flight envelope, first four traces are template
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
                x=[a_harray[-8]],
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
                name="D_min",
            ),
            go.Scatter(
                x=[velocity_maxlift_selected],
                y=[h_selected / 1e3],
                mode="markers+text",
                marker=dict(size=5, color="white"),
                name="maxlift Optimum",
                showlegend=False,
            ),
        ],
        cols=2,
        rows=1,
    )

    fig_maxlift_optimum.update_layout(
        scene=dict(
            xaxis=dict(
                title="C<sub>L</sub> (-)",
                range=[xy_lowerbound, active_selection["CLmax_ld"]],
            ),
            yaxis=dict(title="δ<sub>T</sub> (-)", range=[xy_lowerbound, 1]),
            zaxis=dict(
                title="D (N)",
            ),
        ),
        xaxis=dict(
            title="V (m/s)",
            range=[
                xy_lowerbound,
                max(a_harray.max(), np.nanmax(velocity_maxlift_harray)) + 15,
            ],
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
    """
    )
    return


@app.cell
def _(CD0, K, Pa0, S, atmos, beta, np):
    def maxthrust_solver(W, h):
        sigma = atmos.rhoratio(h)
        A = np.sqrt(atmos.rho(h) * S / (2 * W))
        C1 = Pa0 * sigma**beta * A / W

        # define H(s) = C1*s^3 - CD0 - K*s^4
        def H(s):
            return C1 * s**3 - CD0 - K * s**4

        return H


    def maxthrust_condition(CD0, K, CLstar, CLmax):
        condition = (CLstar > np.sqrt(CD0 / K)) & (CLstar < CLmax)

        return condition
    return maxthrust_condition, maxthrust_solver


@app.cell
def _(CL_E, CLmax, W_selected, h_array, maxthrust_solver, root_scalar):
    CL_maxthrust_star = []

    for h in h_array:
        H = maxthrust_solver(W_selected, h)

        # now solve in s-space (s = sqrt(CL))
        sol = root_scalar(H, x0=CL_E, x1=CLmax, method="secant")
        s_root = sol.root

        CL_sol = s_root**2
        CL_maxthrust_star.append(CL_sol)
    return (CL_maxthrust_star,)


@app.cell
def _(
    CD0,
    CL_maxthrust_star,
    CLmax,
    K,
    S,
    W_selected,
    drag,
    h_array,
    idx_selected,
    maxthrust_condition,
    np,
    velocity,
):
    maxthrust_mask = maxthrust_condition(CD0, K, CL_maxthrust_star, CLmax)

    CLopt_maxthrust = np.where(maxthrust_mask, CL_maxthrust_star, np.nan)

    velocity_maxthrust_harray = velocity(
        W_selected, h_array, CLopt_maxthrust, S, cap=False
    )

    dTopt_maxthrust = np.where(
        maxthrust_mask,
        1,
        np.nan,
    )

    drag_maxthrust_harray = drag(
        h_array, S, CD0, K, CLopt_maxthrust, velocity_maxthrust_harray
    )

    drag_maxthrust_selected = drag_maxthrust_harray[idx_selected]

    CLopt_maxthrust_selected = CLopt_maxthrust[idx_selected]
    dTopt_maxthrust_selected = dTopt_maxthrust[idx_selected]

    velocity_maxthrust_selected = velocity_maxthrust_harray[idx_selected]
    return (
        CLopt_maxthrust,
        CLopt_maxthrust_selected,
        dTopt_maxthrust,
        dTopt_maxthrust_selected,
        drag_maxthrust_harray,
        drag_maxthrust_selected,
        velocity_maxthrust_harray,
        velocity_maxthrust_selected,
    )


@app.cell
def _(
    CL_array,
    CLopt_maxthrust,
    CLopt_maxthrust_selected,
    a_harray,
    active_selection,
    constraint,
    dT_array,
    dTopt_maxthrust,
    dTopt_maxthrust_selected,
    drag_curve,
    drag_maxthrust_harray,
    drag_maxthrust_selected,
    drag_surface,
    go,
    h_array,
    h_selected,
    make_subplots,
    mo,
    np,
    velocity_maxthrust_harray,
    velocity_maxthrust_selected,
    velocity_stall_harray,
    xy_lowerbound,
):
    fig_maxthrust_optimum = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]]
    )

    # Traces on the 3D plot, first four are template
    fig_maxthrust_optimum.add_traces(
        [
            go.Surface(
                x=CL_array,
                y=dT_array,
                z=drag_surface,
                opacity=0.9,
                name="Drag",
                colorscale="cividis",
            ),
            go.Scatter3d(
                x=CL_array,
                y=constraint,
                z=drag_curve,
                opacity=0.7,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.1)", width=10),
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[CL_array[-15]],
                y=[constraint[-15]],
                z=[drag_surface[0, -15]],
                opacity=1,
                textposition="middle left",
                mode="markers+text",
                text=["g<sub>1</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[CLopt_maxthrust_selected],
                y=[dTopt_maxthrust_selected],
                z=[drag_maxthrust_selected],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=3,
                    color="white",
                    symbol="circle",
                ),
                name="maxthrust Optimum",
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub> : %{y}<br>D: %{z}<extra>%{fullData.name}</extra>",
            ),
            go.Scatter3d(
                x=[CLopt_maxthrust_selected, CLopt_maxthrust_selected],
                y=[dTopt_maxthrust_selected, xy_lowerbound],
                z=[
                    drag_maxthrust_selected,
                    drag_maxthrust_selected,
                ],
                mode="lines",
                showlegend=False,
                line=dict(color="grey", width=2),
            ),
            go.Scatter3d(
                x=[xy_lowerbound, CLopt_maxthrust_selected],
                y=[dTopt_maxthrust_selected, dTopt_maxthrust_selected],
                z=[
                    drag_maxthrust_selected,
                    drag_maxthrust_selected,
                ],
                mode="lines",
                showlegend=False,
                line=dict(color="grey", width=2),
            ),
            go.Scatter3d(
                x=CLopt_maxthrust,
                y=np.ones(len(dT_array)) * xy_lowerbound,
                z=np.tile(drag_maxthrust_harray, len(CLopt_maxthrust)),
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(129, 216, 208, 1)", width=8),
            ),
            go.Scatter3d(
                x=np.ones(len(CLopt_maxthrust)) * xy_lowerbound,
                y=dTopt_maxthrust,
                z=np.tile(drag_maxthrust_harray, len(CLopt_maxthrust)),
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
                x=[a_harray[-8]],
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
                name="D_min",
            ),
            go.Scatter(
                x=[velocity_maxthrust_selected],
                y=[h_selected / 1e3],
                mode="markers+text",
                marker=dict(size=5, color="white"),
                name="maxthrust Optimum",
                showlegend=False,
            ),
        ],
        cols=2,
        rows=1,
    )

    fig_maxthrust_optimum.update_layout(
        scene=dict(
            xaxis=dict(
                title="C<sub>L</sub> (-)",
                range=[xy_lowerbound, active_selection["CLmax_ld"]],
            ),
            yaxis=dict(title="δ<sub>T</sub> (-)", range=[xy_lowerbound, 1]),
            zaxis=dict(
                title="D (N)",
            ),
        ),
        xaxis=dict(
            title="V (m/s)",
            range=[xy_lowerbound, np.nanmax(velocity_maxthrust_harray) + 15],
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


@app.cell
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
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

    By substituting this into the previous inequality yields:

    $$
    \frac{3C_{D_0}-KC_{L_\mathrm{max}}^2}{KC_{L_\mathrm{max}}^2 - C_{D_0}} > 0
    $$

    which is then verified for:

    $$
    C_{L_\mathrm{max}} < \sqrt{\frac{3 C_{D_0}}{K}} = C_{L_P}  
    $$

    This concludes the analysis for max-throttle, lift-limited minimum drag for simplified propeller aircraft.
    Below the summarised results:

    $$
    \boxed{\delta_T^* = 1}, \quad \boxed{C_L^* = C_{L_\mathrm{max}}}, \quad \boxed{\frac{W^{3/2}}{\sigma^{\beta+1/2}} = P_{a0} E_S \sqrt{\frac{\rho_0 S C_{L_\mathrm{max}}}{2}}}, \quad \boxed{\sqrt{\frac{C_{D_0}}{K}} < C_{L_\mathrm{max}} < \sqrt{\frac{3 C_{D_0}}{K}}}
    $$

    Show these results below in the flight envelope.
    """
    )
    return


@app.cell
def _(atmos, np):
    def maxlift_thrust_altitude(W, beta, CLmax, CD0, K, Pa0, S):
        E_S = CLmax / (CD0 + K * CLmax**2)

        num = W ** (3 / 2)

        den = Pa0 * np.sqrt(atmos.rho0 * S * CLmax / 2) * E_S

        sigma_exp = num / den

        sigma = sigma_exp ** (1 / (beta + 0.5))

        h = atmos.altitude(sigma)

        return np.where((CLmax > np.sqrt(CD0 / K)) & (h > 0), h, np.nan)
    return (maxlift_thrust_altitude,)


@app.cell
def _(
    CD0,
    CL_array,
    CLmax,
    K,
    Pa0,
    S,
    W_selected,
    beta,
    drag,
    g1_constraint,
    maxlift_thrust_altitude,
    np,
    velocity,
):
    maxlift_thrust_h = maxlift_thrust_altitude(
        W_selected, beta, CLmax, CD0, K, Pa0, S
    )

    CLopt_maxlift_thrust = CLmax

    velocity_maxlift_thrust_selected = velocity(
        W_selected, maxlift_thrust_h, CLopt_maxlift_thrust, S, cap=False
    )

    velocity_CLarray_maxlift_thrust_h = velocity(
        W_selected, maxlift_thrust_h, CL_array, S, cap=False
    )

    dTopt_maxlift_thrust = 1

    drag_maxlift_thrust_h_curve = drag(
        maxlift_thrust_h, S, CD0, K, CL_array, velocity_CLarray_maxlift_thrust_h
    )

    drag_maxlift_thrust_selected = drag(
        maxlift_thrust_h,
        S,
        CD0,
        K,
        CLopt_maxlift_thrust,
        velocity_maxlift_thrust_selected,
    )

    maxlift_thrust_constraint = g1_constraint(
        W_selected, maxlift_thrust_h, Pa0, beta, S, CD0, K, CL_array
    )

    drag_maxlift_thrust_h_surface = np.tile(
        drag_maxlift_thrust_h_curve, (len(CL_array), 1)
    )
    return (
        CLopt_maxlift_thrust,
        dTopt_maxlift_thrust,
        drag_maxlift_thrust_h_surface,
        drag_maxlift_thrust_selected,
        maxlift_thrust_constraint,
        maxlift_thrust_h,
        velocity_maxlift_thrust_selected,
    )


@app.cell
def _(
    CL_array,
    CLopt_maxlift_thrust,
    a_harray,
    active_selection,
    atmos,
    dT_array,
    dTopt_maxlift_thrust,
    drag_maxlift_thrust_h_surface,
    drag_maxlift_thrust_selected,
    go,
    h_array,
    make_subplots,
    maxlift_thrust_constraint,
    maxlift_thrust_h,
    mo,
    velocity_maxlift_thrust_selected,
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
                x=CL_array,
                y=dT_array,
                z=drag_maxlift_thrust_h_surface,
                opacity=0.9,
                name="1/Velocity",
                colorscale="cividis",
            ),
            go.Scatter3d(
                x=CL_array,
                y=maxlift_thrust_constraint,
                z=drag_maxlift_thrust_h_surface[0],
                opacity=0.7,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.1)", width=10),
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[CL_array[50]],
                y=[maxlift_thrust_constraint[50]],
                z=[drag_maxlift_thrust_h_surface[0, 50]],
                opacity=1,
                textposition="middle left",
                mode="markers+text",
                text=["g<sub>1</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[CLopt_maxlift_thrust],
                y=[dTopt_maxlift_thrust],
                z=[drag_maxlift_thrust_selected],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=3,
                    color="white",
                    symbol="circle",
                ),
                name="maxlift Optimum",
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub> : %{y}<br>1/V: %{z}<extra>%{fullData.name}</extra>",
            ),
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[10],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=3,
                    color="rgba(0, 0, 0, 0)",
                    symbol="circle",
                ),
                name="maxlift Optimum",
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub> : %{y}<br>1/V: %{z}<extra>%{fullData.name}</extra>",
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
                name="V_",
            ),
        ],
        cols=2,
        rows=1,
    )

    fig_maxlift_thrust_optimum.update_layout(
        scene=dict(
            xaxis=dict(
                title="C<sub>L</sub> (-)",
                range=[xy_lowerbound, active_selection["CLmax_ld"]],
            ),
            yaxis=dict(title="δ<sub>T</sub> (-)", range=[xy_lowerbound, 1]),
            zaxis=dict(
                title="V (m/s)",
            ),
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


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell
def _(fig_maxlift_thrust_optimum):
    fig_maxlift_thrust_optimum
    return


@app.cell
def _(mo):
    mo.md(
        r"""Summarizing all the above derivations in one single flight envelope obtain:"""
    )
    return


@app.cell
def _(
    h_array,
    maxlift_thrust_h,
    np,
    velocity_interior_harray,
    velocity_maxlift_thrust_selected,
    velocity_maxthrust_harray,
):
    # merged velocity: prefer V_interior, otherwise v_maxthrust
    V = np.where(
        np.isnan(velocity_interior_harray),
        velocity_maxthrust_harray,
        velocity_interior_harray,
    )

    # remove NaNs from merged velocities
    mask = ~np.isnan(V)
    h_clean = h_array[mask]
    V_clean = V[mask]

    # sort original points by altitude
    sort_idx = np.argsort(h_clean)
    h_sorted = h_clean[sort_idx]
    V_sorted = V_clean[sort_idx]

    final_envelope_h = np.append(h_sorted, maxlift_thrust_h)
    final_envelope_velocity = np.append(V_sorted, velocity_maxlift_thrust_selected)
    return final_envelope_h, final_envelope_velocity


@app.cell
def _(
    a_harray,
    active_selection,
    final_envelope_h,
    final_envelope_velocity,
    go,
    h_array,
    mo,
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
                x=final_envelope_velocity,
                y=final_envelope_h / 1e3,
                mode="lines",
                line=dict(width=3, color="rgba(129, 216, 208, 1)"),
                showlegend=False,
                name="P_min interior",
            ),
        ],
    )

    fig_final_flightenv.update_layout(
        xaxis=dict(
            title="V (m/s)",
            range=[xy_lowerbound, velocity_maxthrust_harray.max() + 15],
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
    mo.md(
        r"""This concludes the minimum drag derivation, find below the flight envelope showing the operational conditions where the simplified propeller aircraft can fly at minimum drag. The graph below simply concatenates all the solutions explored in this notebook!"""
    )
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
