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
    from core import atmos
    from core import aircraft as ac
    from core.aircraft import endurance, velocity
    from scipy.optimize import root_scalar

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
def _(mo):
    mo.md(
        r"""
    # Maximum airspeed: simplfied jet aircraft

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
    """
    )
    return


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


@app.cell
def _(active_selection, atmos, endurance, h_slider, m_slider, np, velocity):
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
    return (
        CD0,
        CL_array,
        CLmax,
        E_S,
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
        velocity_stall_harray,
        xy_lowerbound,
    )


@app.cell
def _(atmos, np):
    def g1_constraint(W, h, S, CL, K, CD0, Pa0, beta):
        """
        Returns the delta_T values that correspond to the constraint
        """
        sigma = atmos.rhoratio(h)

        # Mask CL = 0 before using it in powers
        mask = CL != 0
        result = np.full_like(CL, np.inf, dtype=float)  # default inf where CL=0

        if np.any(mask):
            CL_safe = CL[mask]
            numerator = (
                W**1.5
                / np.sqrt(sigma)
                * np.sqrt(2 / (atmos.rho0 * S))
                * (CD0 * (CL_safe ** (-1.5)) + K * (CL_safe**0.5))
            )
            denominator = Pa0 * (sigma**beta)
            result[mask] = numerator / denominator

        return result
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
    g1_constraint,
    h_selected,
    np,
    velocity,
):
    # Computation cell (1)
    velocity_CLarray = velocity(W_selected, h_selected, CL_array, S, cap=False)

    velocity_user_selected = velocity(
        W_selected, h_selected, CL_slider.value, S, cap=False
    )

    delta_T_g1 = g1_constraint(
        W_selected, h_selected, S, CL_array, K, CD0, Pa0, beta
    )


    velocity_surface = np.tile(velocity_CLarray, (len(CL_array), 1))
    return delta_T_g1, velocity_surface, velocity_user_selected


@app.cell
def _(
    CL_array,
    CL_slider,
    active_selection,
    dT_array,
    dT_slider,
    delta_T_g1,
    go,
    mo,
    velocity_surface,
    velocity_user_selected,
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
                z=1 / velocity_surface,
                opacity=0.9,
                name="Velocity",
                colorscale="cividis",
                showscale=False,
            ),
            go.Scatter3d(
                x=CL_array,
                y=delta_T_g1,
                z=1 / velocity_surface[0],
                opacity=0.7,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.1)", width=10),
                name="C2 constraint",
            ),
            go.Scatter3d(
                x=[CL_array[50]],
                y=[delta_T_g1[50]],
                z=[1 / velocity_surface[0, 50]],
                opacity=1,
                textposition="middle left",
                mode="markers+text",
                text=["c<sub>2</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="C2 constraint",
            ),
            go.Scatter3d(
                x=[CL_slider.value],
                y=[dT_slider.value],
                z=[
                    1 / velocity_user_selected
                ],  # Slightly elevate to show the full marker
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
            xaxis=dict(
                title="C<sub>L</sub> (-)",
                range=[xy_lowerbound, active_selection["CLmax_ld"]],
            ),
            yaxis=dict(title="δ<sub>T</sub> (-)", range=[xy_lowerbound, 1]),
            zaxis=dict(title="V<sup> -1</sup> (s/m)"),
        ),
        title_text=active_selection["full_name"],
        title_x=0.5,
    )

    mo.output.clear()
    return (fig_initial,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## KKT formulation
    To be reconducted in the standard KKT analysis format, the objective function is expressed in terms of the controls by direct elimination of $c_1^\mathrm{eq}$.
    Also, maximizing $V$ is equivalent to minimizing its inverse, $1/V$.
    Therefore, to simplify the calculations, the problem is rewritten as follows:
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""In the interactive graph below, select a simplified propeller aircraft of your choice and experiment in finding an optimum by changing the control variables, $C_L$ and $\delta_T$. The design point is marked in white in the 3D velocity surface."""
    )
    return


@app.cell
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


@app.cell
def _(CL_slider, dT_slider, mo):
    mo.md(
        f"""Here you can modify the control variables to understand how it affects the design: {mo.hstack([dT_slider, CL_slider])}"""
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


@app.cell
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    A necessary condition for an optimal solution of the optimization problem $(C_L^*, \delta_T^*)$ to exist, the multipliers $\lambda_1, \mu_1, \mu_2$ have to meet the following conditions:

    **A. Stationarity ($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \frac{\partial \mathcal{L}}{\partial C_L} = \frac{1}{2}\sqrt{\rho_0\frac{S}{2}\frac{\sigma}{W}}C_L^{-1/2} - \lambda_1 \frac{W^{3/2}}{\sigma^{1/2}}\sqrt{\frac{2}{\rho_0S}} \left(-\frac{3}{2}C_{D_0}C_L^{-5/2} + \frac{1}{2}KC_L^{-1/2}\right) + \mu_1 = 0$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1 P_{a0}\sigma^\beta + \mu_2 = 0$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $\displaystyle \delta_T P_{a0}\sigma^\beta - \frac{W^{3/2}}{\sigma^{1/2}}\sqrt{\frac{2}{\rho_0S}} \left(C_{D_0}C_L^{-3/2} + KC_L^{1/2}\right) = 0$
    4.  $C_L - C_{L_\mathrm{max}} \le 0$
    5.  $\delta_T - 1 \le 0$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    6.  $\mu_1, \mu_2\ge 0$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraint have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

    7.  $\mu_1 (C_L - C_{L_\mathrm{max}}) = 0$
    8. $\mu_2 (\delta_T - 1) = 0$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## KKT analysis

    We can now proceed to systematically examine the conditions where various inequality constraints are active or inactive.

    ### _Interior solutions_ 

    Assuming that that $C_L < C_{L_\mathrm{max}}$ and $\delta_T < 1$ is equivalent to consider all inequality constraints as inactive.

    Therefore: $\mu_1,\mu_2=0$. 

    From stationarity condition (2): $\lambda_1 = 0$.

    It can now be seen that stationarity condition (1) is never verified.

    It can be concluded that the maximum speed cannot be achieved in the interior of the domain. The maximum speed must lie on at least one of the boundaries defined by $C_L = C_{L_\mathrm{max}}$ or $\delta_T = 1$.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell
def _(CD0, K, Pa0, S, atmos, beta, np):
    def maxthrust_solver(W, h):
        sigma = atmos.rhoratio(h)

        function = lambda CL: Pa0 * sigma**beta - W**1.5 / (sigma**0.5) * np.sqrt(
            2 / atmos.rho0 / S
        ) * (CD0 + K * CL**2) / (CL ** (3 / 2))

        return function


    def maxthrust_condition(CD0, K, CLstar, CLmax):
        condition = (CLmax > np.sqrt(CD0 / K)) & (CLstar < np.sqrt(3 * CD0 / K))

        return condition
    return maxthrust_condition, maxthrust_solver


@app.cell
def _(W_selected, h_array, maxthrust_solver, root_scalar):
    CL_maxthrust_star = []

    for h in h_array:
        func = maxthrust_solver(W_selected, h)
        CL_sol = root_scalar(func, x0=0.04).root
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

    CLopt_maxthrust_selected = CLopt_maxthrust[idx_selected]
    dTopt_maxthrust_selected = dTopt_maxthrust[idx_selected]

    velocity_maxthrust_selected = velocity_maxthrust_harray[idx_selected]
    return (
        CLopt_maxthrust,
        CLopt_maxthrust_selected,
        dTopt_maxthrust_selected,
        velocity_maxthrust_harray,
        velocity_maxthrust_selected,
    )


@app.cell(hide_code=True)
def _(
    CL_array,
    CLopt_maxthrust,
    CLopt_maxthrust_selected,
    a_harray,
    active_selection,
    dT_array,
    dTopt_maxthrust_selected,
    delta_T_g1,
    go,
    h_array,
    h_selected,
    make_subplots,
    mo,
    np,
    velocity_maxthrust_harray,
    velocity_maxthrust_selected,
    velocity_stall_harray,
    velocity_surface,
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
                z=1 / velocity_surface,
                opacity=0.9,
                name="1/Velocity",
                colorscale="cividis",
                showscale=False,
            ),
            go.Scatter3d(
                x=CL_array,
                y=delta_T_g1,
                z=1 / velocity_surface[0],
                opacity=0.7,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.1)", width=10),
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[CL_array[50]],
                y=[delta_T_g1[50]],
                z=[1 / velocity_surface[0, 50]],
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
                z=[1 / velocity_maxthrust_selected],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=3,
                    color="white",
                    symbol="circle",
                ),
                name="maxthrust Optimum",
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub> : %{y}<br>1/V: %{z}<extra>%{fullData.name}</extra>",
            ),
            go.Scatter3d(
                x=[CLopt_maxthrust_selected, CLopt_maxthrust_selected],
                y=[dTopt_maxthrust_selected, xy_lowerbound],
                z=[
                    1 / velocity_maxthrust_selected,
                    1 / velocity_maxthrust_selected,
                ],
                mode="lines",
                showlegend=False,
                line=dict(color="grey", width=2),
            ),
            go.Scatter3d(
                x=CLopt_maxthrust,
                y=np.ones(len(dT_array)) * xy_lowerbound,
                z=np.tile(1 / velocity_maxthrust_harray, len(CLopt_maxthrust)),
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
                name="V_max",
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
                title="V (m/s)",
            ),
        ),
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
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell
def _(atmos, np):
    def maxlift_thrust_altitude(W, beta, CLmax, CD0, K, Pa0, E_S, S):
        sigma_exp = W**1.5 / Pa0 / E_S / np.sqrt(atmos.rho0 * S * CLmax / 2)

        sigma = sigma_exp ** (1 / (beta))

        h = atmos.altitude(sigma)

        return np.where((h > 0) & (CLmax < np.sqrt(3 * CD0 / K)), h, np.nan)
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
    g1_constraint,
    maxlift_thrust_altitude,
    np,
    velocity,
):
    maxlift_thrust_h = maxlift_thrust_altitude(
        W_selected, beta, CLmax, CD0, K, Pa0, E_S, S
    )

    CLopt_maxlift_thrust = CLmax

    velocity_maxlift_thrust_selected = velocity(
        W_selected, maxlift_thrust_h, CLopt_maxlift_thrust, S, cap=True
    )

    velocity_CLarray_maxlift_thrust_h = velocity(
        W_selected, maxlift_thrust_h, CL_array, S, cap=True
    )

    dTopt_maxlift_thrust = 1


    constraint_maxlift_thrust = g1_constraint(
        W_selected, maxlift_thrust_h, S, CL_array, K, CD0, Pa0, beta
    )

    velocity_maxlift_thrust_surface = np.tile(
        velocity_CLarray_maxlift_thrust_h, (len(CL_array), 1)
    )
    return (
        CLopt_maxlift_thrust,
        constraint_maxlift_thrust,
        dTopt_maxlift_thrust,
        maxlift_thrust_h,
        velocity_maxlift_thrust_selected,
        velocity_maxlift_thrust_surface,
    )


@app.cell
def _(
    CL_array,
    CLopt_maxlift_thrust,
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
                x=CL_array,
                y=dT_array,
                z=1 / velocity_maxlift_thrust_surface,
                opacity=0.9,
                name="1/Velocity",
                colorscale="cividis",
            ),
            go.Scatter3d(
                x=CL_array,
                y=constraint_maxlift_thrust,
                z=1 / velocity_maxlift_thrust_surface[0],
                opacity=0.7,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.1)", width=10),
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[CL_array[50]],
                y=[constraint_maxlift_thrust[50]],
                z=[1 / velocity_maxlift_thrust_surface[0, 50]],
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
                z=[1 / velocity_maxlift_thrust_selected],
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
def _(mass_stack):
    mass_stack
    return


@app.cell
def _(fig_maxlift_thrust_optimum):
    fig_maxlift_thrust_optimum
    return


@app.cell
def _(mo):
    mo.md(
        r"""Now after deriving all the optima for each condition we can summarize the flight envelopes in one graph, as shown below. Experiment with the weight of the aircrarft to understand how the theoretical ceiling for maximum speed moves in the graph."""
    )
    return


@app.cell
def _(velocity_maxthrust_harray):
    # Merge lines to have a continuous line showing up in the final flight envelope

    final_velocity_flightenvelope = velocity_maxthrust_harray
    return (final_velocity_flightenvelope,)


@app.cell
def _(velocity_maxlift_thrust_selected):
    velocity_maxlift_thrust_selected
    return


@app.cell
def _(
    a_harray,
    active_selection,
    final_velocity_flightenvelope,
    go,
    h_array,
    maxlift_thrust_h,
    mo,
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
                x=final_velocity_flightenvelope,
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
def _(mass_stack):
    mass_stack
    return


@app.cell
def _(fig_final_flightenv):
    fig_final_flightenv
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Summary

    | Name | Condition | $C_L^*$ | $\delta_T^*$ | $V^*$ |
    |:-|:----------|:-------:|:------------:|:------|
    |Thrust and Lift-limited    | $\displaystyle \frac{W^{3/2}}{\sigma^{\beta+1/2}} = P_{a0}E_S\sqrt{\frac{\rho_0 S}{2}C_{L_{\mathrm{max}}}}$ | $C_{L_\mathrm{max}}$ | $1$ | $\displaystyle V_s =\sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}$ |
    |Thrust-limited    | $\displaystyle \mathrm{numerical}$ | $\displaystyle \mathrm{numerical}$ | $1$ | $\displaystyle \mathrm{numerical}$ |
    """
    )
    return


if __name__ == "__main__":
    app.run()
