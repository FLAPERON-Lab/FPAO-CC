import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")

with app.setup:
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


    def CL_from_horizontal_constraint(W, h, S, CD0, K, Ta0, beta, ac_type):
        E_max = endurance(K, CD0, "max")
        sigma = atmos.rhoratio(h)

        plus_solution = np.full_like(sigma, np.nan, dtype=float)
        minus_solution = np.full_like(sigma, np.nan, dtype=float)

        if ac_type == "jet":
            # validity condition
            condition = (W / (sigma**beta)) < (Ta0 * E_max)

            # compute safe argument for sqrt
            arg = 1 - (W / (Ta0 * sigma**beta * E_max)) ** 2
            arg = np.where(arg >= 0, arg, np.nan)  # mask negatives

            root = np.sqrt(arg)
            multiplier = Ta0 * sigma**beta / (2 * K * W)

            plus_solution = np.where(condition, multiplier * (1 + root), np.nan)
            minus_solution = np.where(condition, multiplier * (1 - root), np.nan)

        return [plus_solution, minus_solution]


    def optimum_maxthrust(W, h, S, CD0, K, Ta0, beta, ac_type):
        sigma = atmos.rhoratio(h)  # array if h is array
        E_max = endurance(K, CD0, "max")

        # elementwise logical condition
        condition = ((W / sigma**beta) < (Ta0 * E_max)) & (
            (W / sigma**beta) > (np.sqrt(3) * (Ta0 * E_max) / 2)
        )

        CL_star = CL_from_horizontal_constraint(
            W, h, S, CD0, K, Ta0, beta, ac_type
        )[0]

        deltaT_out = np.where(condition, 1, np.nan)
        CL_out = np.where(condition, CL_star, np.nan)

        return CL_out, deltaT_out


@app.cell
def _():
    # Set navbar on the right
    _defaults.set_sidebar()
    return


@app.cell(hide_code=True)
def _(active_selection, h_slider, m_slider):
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
    Ta0 = active_selection["Ta0"] * 1e3  # Watts
    beta = active_selection["beta"]
    OEM = active_selection["OEM"]
    MTOM = active_selection["MTOM"]
    CL_P = np.sqrt(3 * CD0 / K)
    CL_E = np.sqrt(CD0 / K)
    E_max = endurance(K, CD0, "max")
    E_P = (np.sqrt(3) / 2) * E_max
    E_S = CLmax / (CD0 + K * CLmax)
    return (
        CD0,
        CL_E,
        CL_P,
        CL_array,
        CLmax,
        E_max,
        K,
        S,
        Ta0,
        W_selected,
        a,
        a_harray,
        beta,
        dT_array,
        h_array,
        h_selected,
        idx_selected,
        xy_lowerbound,
    )


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    # Minimum Power Required: simplified jet aircraft

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
        & \quad T_a(V,h) = T_a(h) = T_{a0}\sigma^\beta \\
    \end{aligned}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(ac_table, data):
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
def _(CL_E, CL_P, CL_array, CLmax, S, W_selected, h_array, h_selected):
    velocity_CL_E = float(velocity(W_selected, h_selected, CL_E, S, False))
    velocity_CL_P = float(velocity(W_selected, h_selected, CL_P, S, False))
    velocity_stall_selected = float(velocity(W_selected, h_selected, CLmax, S))
    velocity_CLarray = velocity(W_selected, h_selected, CL_array, S)
    velocity_stall_harray = velocity(W_selected, h_array, CLmax, S)
    return (
        velocity_CL_E,
        velocity_CL_P,
        velocity_CLarray,
        velocity_stall_harray,
        velocity_stall_selected,
    )


@app.cell
def _(
    CD0,
    CL_array,
    CL_slider,
    K,
    S,
    Ta0,
    W_selected,
    beta,
    h_selected,
    velocity_CLarray,
):
    velocity_user_selected = velocity(W_selected, h_selected, CL_slider.value, S)
    power_user_selected = power(
        h_selected,
        S,
        CD0,
        K,
        CL_slider.value,
        velocity_user_selected,
    )

    drag_curve = drag(
        h_selected,
        S,
        CD0,
        K,
        CL_array,
        velocity_CLarray,
    )


    constraint = horizontal_constraint(
        W_selected,
        h_selected,
        CD0,
        K,
        CL_array,
        Ta0,
        beta,
        V=velocity_CLarray,
        S=S,
        D=drag_curve,
        type="jet",
    )


    power_curve = np.where(
        ~np.isnan(constraint),
        power(h_selected, S, CD0, K, CL_array, velocity_CLarray),
        np.nan,
    )


    power_surface = np.tile(power_curve, (len(CL_array), 1))
    return (
        constraint,
        drag_curve,
        power_curve,
        power_surface,
        power_user_selected,
    )


@app.cell(hide_code=True)
def _(
    CL_array,
    CL_slider,
    active_selection,
    constraint,
    dT_array,
    dT_slider,
    power_surface,
    power_user_selected,
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
                z=power_surface / 1e3,
                opacity=0.9,
                name="Power",
                colorscale="cividis",
            ),
            go.Scatter3d(
                x=CL_array,
                y=constraint,
                z=power_surface[0] / 1e3,
                opacity=0.7,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.1)", width=10),
                name="C2 constraint",
            ),
            go.Scatter3d(
                x=[CL_array[-15]],
                y=[constraint[-15]],
                z=[power_surface[0, -15] / 1e3 + 450],
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
                    power_user_selected / 1e3 + 100
                ],  # Slightly elevate to show the full marker
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

    fig_initial.update_layout(
        scene=dict(
            xaxis=dict(
                title="C<sub>L</sub> (-)",
                range=[xy_lowerbound, active_selection["CLmax_ld"]],
            ),
            yaxis=dict(title="δ<sub>T</sub> (-)", range=[xy_lowerbound, 1]),
            zaxis=dict(title="P (kW)"),
        ),
        title_text=active_selection["full_name"],
        title_x=0.5,
    )

    mo.output.clear()
    return (fig_initial,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## KKT formulation
    To be reconducted in the standard KKT analysis format, the objective function is expressed in terms of the controls by direct elimination of $c_1^\mathrm{eq}$. The velocity $V$ can be expressed as: 

    $$
    V = \sqrt{\frac{2}{\rho}\frac{W}{S}\frac{1}{C_L}}
    $$

    Moreover, in previous analyses we found $\delta_T=C_L=0$ does not correspond to sensible solution, thus we can write:

    $$
    0\lt \delta_T \le 1 \quad \land \quad  0\lt C_L\le C_{L_{\mathrm{max}}}
    $$

    Notice the open interval in the lower bounds.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""The KKT formulation can now be written:""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    $$
    \begin{aligned}
        \min_{C_L, \delta_T} 
        & \quad P = DV = W \left(\frac{C_{D_0} +K C_L^2}{C_L}\right)\sqrt{\frac{2}{\rho}\frac{W}{S}\frac{1}{C_L}}=\sqrt{\frac{2W^3}{\rho S}}\left(\frac{C_{D_0}+K C_L^2}{C_L^{3/2}}\right) = \sqrt{\frac{2W^3}{\rho S}}\left(C_{D_0} C_L^{-3/2}+K C_L^{1/2}\right)\\
        \text{subject to} 
        & \quad g_1 = T - \frac{W}{E}  =\delta_T T_{a0}\sigma^\beta - W\frac{C_{D_0} + K C_L^2}{C_L} = 0 \\
        & \quad h_1 = C_L - C_{L_\mathrm{max}} \le 0 \\
        & \quad h_2 = \delta_T - 1 \le 0 \\
    \end{aligned}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Lagrangian function and KKT conditions

    The Lagrangian function combines the objective function with eqaulity constraints using Lagrange multipliers ($\lambda_i$) and inequality constraints using KKT multipliers ($\mu_j$).

    $$
    \begin{aligned}
    \mathcal{L}(C_L, \delta_T, \lambda_1, \mu_1, \mu_2) = & P + \lambda_1 \left[T - D\right]+ \mu_1 (C_L - C_{L_\mathrm{max}}) +\mu_2 (\delta_T - 1)\\ 
    =&\quad \sqrt{\frac{2W^3}{\rho S}}\left(C_{D_0} C_L^{-3/2}+K C_L^{1/2}\right) +\\
    & + \lambda_1 \left[\delta_T T_{a0}\sigma^\beta - W\frac{C_{D_0} + K C_L^2}{C_L}\right] + \\
    & + \mu_1 (C_L - C_{L_\mathrm{max}}) + \\
    & + \mu_2 (\delta_T - 1) +\\
    \end{aligned}
    $$
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""In the interactive graph below, select a simplified jet aircraft of your choice and experiment in finding an optimum by changing the control variables, $C_L$ and $\delta_T$. The design point is marked in white in the 3D power surface."""
    )
    return


@app.cell(hide_code=True)
def _():
    # Database cell
    data = ac.available_aircrafts(data_dir, ac_type="Jet")[:8]

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
def _(CL_slider, dT_slider):
    mo.md(
        f"""Here you can modify the control variables to understand how it affects the design: {mo.hstack([dT_slider, CL_slider])}"""
    )
    return


@app.cell(hide_code=True)
def _(variables_stack):
    variables_stack
    return


@app.cell(hide_code=True)
def _(fig_initial):
    fig_initial
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    A necessary condition for an optimal solution of the optimization problem $(C_L^*, \delta_T^*)$ to exist, the multipliers $\lambda_1, \mu_1, \mu_2$ have to meet the following conditions:

    **A. Stationarity ($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \frac{\partial \mathcal{L}}{\partial C_L} = \sqrt{\frac{2W^3}{\rho S}}\left(-\frac{3}{2}C_{D_0}C_L^{-5/2} + \frac{1}{2} K C_L^{-1/2}\right) - \lambda_1 W \left(\frac{KC_L^2 -C_{D_0}}{C_L^2}\right) + \mu_1= 0$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1 T_{a0}\sigma^\beta+ \mu_2= 0$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $\displaystyle \delta_T T_{a0}\sigma^\beta - W \frac{C_{D_0} + K C_L^2}{C_L} = 0$
    4.  $C_L - C_{L_\mathrm{max}} \le 0$
    5.  $\delta_T - 1 \le 0$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    6.  $\mu_1, \mu_2 \ge 0$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraint have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

    7.  $\mu_1 (C_L - C_{L_\mathrm{max}}) = 0$
    8. $\mu_3 (\delta_T - 1) = 0$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## KKT analysis

    We can now proceed to systematically examine the conditions where various inequality constraints are active 
    or inactive.

    ### _Interior solutions_ 

    In this case: $C_L \lt C_{L_{\mathrm{max}}}$, $\delta_T \lt 1$, $\mu_1=\mu_2= 0$

    from stationarity condition (2): $\lambda_1 = 0$

    from stationarity condition (1): 

    $$
    -\frac{3}{2}C_{D_0} C_L^{-5/2}+\frac{1}{2}KC_L^{-1/2}= 0 \quad \Rightarrow \quad KC_L^2 = 3C_{D_0} \quad \Rightarrow \quad C_L^* = \sqrt{\frac{3C_{D_0}}{K}} = \sqrt{3}C_{L_E} = C_{L_P}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    The corresponding $\delta_T$ value is obtained from primal feasibility constraint (3): 

    $$
    \delta_T^* = \frac{W}{T_{a0}\sigma^\beta} \left(\frac{C_{D_0}+K \cdot 3C_{D_0}/K}{\sqrt{3C_{D_0}/K}}\right) = \frac{W}{T_{a0}\sigma^\beta}\sqrt{\frac{16C_{D_0}K}{3}} = \sqrt{\frac{4}{3}}\frac{W}{E_{\mathrm{max}}}\frac{1}{T_{a0}\sigma^\beta} = \frac{W}{E_{\mathrm{P}}}\frac{1}{T_{a0}\sigma^\beta}
    $$

    Where: $\displaystyle E_{\mathrm{P}} = \frac{\sqrt{3}}{2}E_{\mathrm{max}}$

    This is valid for:  

    $$
    \delta_T^*\lt 1 \Leftrightarrow \frac{W}{\sigma^\beta} \lt E_{\mathrm{P}}T_{a0} = \frac{\sqrt{3}}{2}E_{\mathrm{max}}T_{a0}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    Finally, the optimum for the interior of the domain is thus:

    $$
    \boxed{C_L^* = \sqrt{\frac{3C_{D_0}}{K}} = \sqrt{3}C_{L_E}} \quad \land \quad \boxed{\delta_T^* = \frac{2}{\sqrt{3}}\frac{W}{E_{\mathrm{max}}T_{a0}\sigma^\beta}} \qquad \qquad \forall \quad \frac{W}{\sigma^\beta} \lt \frac{\sqrt{3}}{2}E_{\mathrm{max}}T_{a0}
    $$
    """
    )
    return


@app.function
def interior_condition(W, h, E_max, Ta0, beta):
    sigma = atmos.rhoratio(h)
    condition = (W / (sigma**beta)) < ((np.sqrt(3) / 2) * E_max * Ta0)

    return condition


@app.cell
def _(CD0, CL_P, E_max, K, S, Ta0, W_selected, beta, h_array, idx_selected):
    interior_mask = interior_condition(W_selected, h_array, E_max, Ta0, beta)

    CLopt_interior = np.where(interior_mask, CL_P, np.nan)

    velocity_interior_harray = velocity(W_selected, h_array, CLopt_interior, S)

    dTopt_interior = np.where(
        interior_mask,
        (
            2
            / np.sqrt(3)
            * W_selected
            / E_max
            / Ta0
            / (atmos.rhoratio(h_array) ** beta)
        ),
        np.nan,
    )

    CLopt_interior_selected = CLopt_interior[idx_selected]
    dTopt_interior_selected = dTopt_interior[idx_selected]

    velocity_interior_selected = velocity_interior_harray[idx_selected]

    power_interior_harray = power(
        h_array, S, CD0, K, CLopt_interior, velocity_interior_harray
    )
    power_interior_selected = power_interior_harray[idx_selected]
    return (
        CLopt_interior,
        CLopt_interior_selected,
        dTopt_interior,
        dTopt_interior_selected,
        power_interior_harray,
        power_interior_selected,
        velocity_interior_harray,
        velocity_interior_selected,
    )


@app.cell(hide_code=True)
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
    h_array,
    h_selected,
    power_interior_harray,
    power_interior_selected,
    power_surface,
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
                z=power_surface / 1e3,
                opacity=0.9,
                name="Power",
                colorscale="cividis",
            ),
            go.Scatter3d(
                x=CL_array,
                y=constraint,
                z=power_surface[0] / 1e3,
                opacity=0.7,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.1)", width=10),
                name="C2 constraint",
            ),
            go.Scatter3d(
                x=[CL_array[-15]],
                y=[constraint[-15]],
                z=[power_surface[0, -15] / 1e3 + 250],
                opacity=1,
                textposition="middle left",
                mode="markers+text",
                text=["c<sub>2</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="C2 constraint",
            ),
            go.Scatter3d(
                x=[CLopt_interior_selected],
                y=[dTopt_interior_selected],
                z=[
                    power_interior_selected / 1e3 + 100
                ],  # Slightly elevate to show the full marker
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=3,
                    color="white",
                    symbol="circle",
                ),
                name="Interior Optimum",
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub> : %{y}<br>P: %{z}<extra>%{fullData.name}</extra>",
            ),
            go.Scatter3d(
                x=[CLopt_interior_selected, CLopt_interior_selected],
                y=[dTopt_interior_selected, xy_lowerbound],
                z=[
                    power_interior_selected / 1e3 + 100,
                    power_interior_selected / 1e3 + 100,
                ],
                mode="lines",
                showlegend=False,
                line=dict(color="grey", width=2),
            ),
            go.Scatter3d(
                x=[xy_lowerbound, CLopt_interior_selected],
                y=[dTopt_interior_selected, dTopt_interior_selected],
                z=[
                    power_interior_selected / 1e3 + 100,
                    power_interior_selected / 1e3 + 100,
                ],
                mode="lines",
                showlegend=False,
                line=dict(color="grey", width=2),
            ),
            go.Scatter3d(
                x=CLopt_interior,
                y=np.ones(len(dT_array)) * xy_lowerbound,
                z=np.tile(power_interior_harray / 1e3 + 100, len(CLopt_interior)),
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(129, 216, 208, 1)", width=8),
            ),
            go.Scatter3d(
                x=np.ones(len(CLopt_interior)) * xy_lowerbound,
                y=dTopt_interior,
                z=np.tile(power_interior_harray / 1e3 + 100, len(CLopt_interior)),
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
                x=velocity_interior_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(width=3, color="rgba(129, 216, 208, 1)"),
                showlegend=False,
                name="P_min",
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
                title="P (kW)",
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
def _():
    mo.md(
        r"""Notice how $C_{L_P}$ (minimum power) $\gt$ $C_{L_E}$ (minimum drag) but $E_\mathrm{P} \lt E_{\mathrm{max}}$ ($E = C_L/C_D$) because the drag coefficient increases more rapidly than $C_L$, as $C_D \propto C_L^2$. Thus the range of $W/\sigma^\beta$ for which it is possible to fly at minimum power is smaller ($\sqrt{3}/2\lt 1$) than the one for which it is possible to fly at minimum drag."""
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### _Lift limited solutions (stall)_

    In this case: $C_L = C_{L_{\mathrm{max}}}$, $\delta_T \lt 1$, $\mu_1 \gt 0$, $\mu_2= 0$

    from stationarity condition (2): $\lambda_1 = 0$

    from stationarity condition (1):

    $$
    \mu_1 = - \left.\frac{\partial P}{\partial C_L} \right|_{C_{L_\mathrm{max}}} =- \sqrt{\frac{2W^3}{\rho S}}\left(-\frac{3}{2}C_{D_0}C_{L_\mathrm{max}}^{-5/2} + \frac{1}{2} K C_{L_\mathrm{max}}^{-1/2}\right) \gt 0
    $$ 

    This inequality is saying that the required power should decrease for an increase in $C_L$ starting from $C_{L_\mathrm{max}}$. In other words, $P_r$ should decrease for a decrease in speed from the stall speed. Equivalently, $P_r$ should increase for a n increase in speed from the stall speed. This is clearly impossible, by the taking the shape of the power curve on the performance diagram.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    As a matter of fact, by substitution from the stationarity constraint (1):

    $$ \frac{3}{2}C_{D_0}C_{L_\mathrm{max}}^{-5/2} + \frac{1}{2} K C_{L_\mathrm{max}}^{-1/2} \lt 0 
    $$

    $$
    \Rightarrow -3C_{D_0}+KC_{L_\mathrm{max}}^{2} \lt 0 \Rightarrow C_{L_\mathrm{max}} \lt \sqrt{\frac{3C_{D_0}}{K}} = C_{L_P}
    $$

    We thus find that the maximum $C_L$ must be smaller than the lift coefficient for minimum power, this is clearly impossible by definition of $C_{L_\mathrm{max}}$.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### _Thrust-limited optimum_


    In this case: $C_L \lt C_{L_{\mathrm{max}}}$, $\delta_T = 1$, $\mu_1= 0$, $\mu_2 > 0$

    from stationarity condition (2): $\mu_2= -\lambda_1 T_{a0}\sigma^ \beta \gt 0 \Rightarrow \lambda_1 \lt 0$

    from stationarity condition (1): 

    $$
    \frac{\partial P}{\partial C_L} = \lambda_1 \frac{\partial D}{\partial C_L}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    This tells us that the required power and drag change in opposite directions with respect to the change in $C_L$. If one decreaes, then the other one has to increase, given that $\lambda_1 \lt 0$.
    This can only happen in the range of $C_L$ between $C_{L_P}$ and $C_{L_E}$, since they represent the minimum power and maximum aerodynamic efficiency (alternatively minimum drag) respectively. 

    This is clearer in the performance diagram:
    """
    )
    return


@app.cell
def _(CLmax):
    CL_ticks = np.arange(0, CLmax + 1, 1)[1:-2]
    CL_ticks = np.append(CL_ticks, CLmax)
    text_cl_ticks = [str(tick) for tick in CL_ticks[:-1]]
    text_cl_ticks.append(r"$C_{L_\mathrm{max}}$")
    return CL_ticks, text_cl_ticks


@app.cell
def _(CL_ticks, CLmax, S, W_selected, h_selected, velocity_CLarray):
    velocity_cl_line = np.append(
        velocity(W_selected, h_selected, CLmax, S) - 10,
        max(velocity_CLarray),
    )


    velocity_cl_array = velocity(W_selected, h_selected, np.array(CL_ticks), S)
    return velocity_cl_array, velocity_cl_line


@app.cell(hide_code=True)
def _(
    a,
    active_selection,
    drag_curve,
    power_curve,
    text_cl_ticks,
    velocity_CL_E,
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
        x=velocity_CL_E,
        line_dash="dot",
        annotation=dict(text="$C_{L_E}$", xshift=10, yshift=-10),
        line=dict(color="white"),
    )
    fig_thrust_limited.add_vline(
        x=velocity_CL_P,
        line_dash="dot",
        annotation=dict(text="$C_{L_P}$", xshift=10, yshift=-10),
        line=dict(color="white"),
    )
    fig_thrust_limited.add_vrect(
        x0=velocity_CL_P,
        x1=velocity_CL_E,
        fillcolor="green",
        opacity=0.25,
        line_width=0,
    )

    fig_thrust_limited.add_vline(
        x=velocity_stall_selected,
        line_dash="dot",
        annotation=dict(text=r"$V_{\mathrm{stall}}$", xshift=10, yshift=-10),
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
def _():
    mo.md(
        r"""
    This condition is given by:

    $$
    C_{L_E}\lt C_L\lt C_{L_P} \quad \Leftrightarrow \quad \boxed{\sqrt{\frac{C_{D_0}}{K}}\lt C_L \lt \sqrt{3} \sqrt{\frac{C_{D_0}}{K}}}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    The corresponding $C_L$ is given by primal feasibility constraint (3): 

    $$
    T_{a0} \sigma^\beta - W \left(\frac{C_{D_0}+KC_L^2}{C_L}\right)=0
    $$

    Yielding the following quadratic equation:

    $$
    K C_L^2 - \frac{T_{a0}\sigma^\beta}{W}C_L+C_{D_0} = 0 \quad \Rightarrow \quad C_L = \frac{T_{a0}\sigma^\beta}{2KW}\left[1 \pm\sqrt{1- \left(\frac{W}{T_{a0}\sigma^\beta E_{\mathrm{max}}}\right)^2}\right]
    $$

    where the relevant solution is given by the "${+}$" sign, on the left branch of the drag curve in the performance diagram: 

    $$
    \Rightarrow C_L^* = \frac{T_{a0}\sigma^\beta}{2KW}\left[1 +\sqrt{1- \left(\frac{W}{T_{a0}\sigma^\beta E_{\mathrm{max}}}\right)^2}\right]
    $$

    The solution is valid as long as: $\sqrt{\frac{C_{D_0}}{K}}\lt C_L^* \lt \sqrt{3} \sqrt{\frac{C_{D_0}}{K}}$.
    """
    )
    return


@app.cell
def _(CD0, K, S, Ta0, W_selected, beta, h_selected):
    plus_cl_solution, minus_cl_solution = CL_from_horizontal_constraint(
        W_selected, h_selected, S, CD0, K, Ta0, beta, "jet"
    )

    velocity_plus_solution = velocity(
        W_selected, h_selected, float(plus_cl_solution), S, cap=False
    )
    velocity_minus_solution = velocity(
        W_selected, h_selected, float(minus_cl_solution), S, cap=False
    )
    return plus_cl_solution, velocity_minus_solution, velocity_plus_solution


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    For its existence, the square root must be zero or positive, thus: 

    $$
    1 - \left(\frac{W}{T_{a0}\sigma^\beta E_{\mathrm{max}}}\right)^2 \ge 0 \quad \Rightarrow \quad \frac{W}{\sigma^\beta}\le T_{a0}E_{\mathrm{max}}
    $$

    as already seen in multiple occasions.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""Try to find whether there is a combination of altitude and weight for which the solution of the quadratic equation with the "+" sign falls within the bounds of $C_{L_P}$ and $C_{L_E}$, denoted by the green area in the graph below. Be careful, this is not always possible and will define the flight envelope where minimum power can be achieved."""
    )
    return


@app.cell(hide_code=True)
def _(
    a,
    active_selection,
    drag_curve,
    plus_cl_solution,
    power_curve,
    velocity_CL_E,
    velocity_CL_P,
    velocity_CLarray,
    velocity_minus_solution,
    velocity_plus_solution,
):
    fig_performance_cl_eq = go.Figure()

    fig_performance_cl_eq.add_traces(
        [
            go.Scatter(x=velocity_CLarray, y=power_curve, name="Power"),
            go.Scatter(
                x=velocity_CLarray,
                y=drag_curve,
                name="Drag",
                yaxis="y2",
            ),
        ]
    )

    if (
        plus_cl_solution is not None
        and not np.isnan(velocity_CL_E)
        and not np.isnan(velocity_CL_P)
    ):
        fig_performance_cl_eq.add_vline(
            x=velocity_plus_solution,
            line_dash="dot",
            annotation=dict(text="$C_{L}^{*+}$", xshift=10, yshift=-10),
            line=dict(color="white"),
        )
        fig_performance_cl_eq.add_vline(
            x=velocity_minus_solution,
            line_dash="dot",
            annotation=dict(text="$C_{L}^{*-}$", xshift=10, yshift=-10),
            line=dict(color="white"),
        )
        fig_performance_cl_eq.add_vline(
            x=velocity_CL_E,
            line_dash="dot",
            annotation=dict(text="$C_{L_E}$", xshift=10, yshift=-10),
            line=dict(color="white"),
        )
        fig_performance_cl_eq.add_vline(
            x=velocity_CL_P,
            line_dash="dot",
            annotation=dict(text="$C_{L_P}$", xshift=10, yshift=-10),
            line=dict(color="white"),
        )
        fig_performance_cl_eq.add_vrect(
            x0=velocity_CL_P,
            x1=velocity_CL_E,
            fillcolor="green",
            opacity=0.25,
            line_width=0,
        )

    # Axes configuration
    fig_performance_cl_eq.update_layout(
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
    return (fig_performance_cl_eq,)


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell
def _(fig_performance_cl_eq):
    fig_performance_cl_eq
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    In order for $C_L^* \gt \sqrt{\frac{C_{D_0}}{K}}$ it must be:

    $$
    1 - \left(\frac{W}{T_{a0}\sigma^\beta E_{\mathrm{max}}}\right)^2  \gt \left(\frac{W}{T_{a0}\sigma^\beta E_{\mathrm{max}}} - 1\right)^2
    $$

    which then simplifies to: 

    $$
    \frac{W}{\sigma^\beta}\lt T_{a0}E_{\mathrm{max}}
    $$

    which can be compared to the domain imposed by the square root directly. The strongest condition, or the lower upper bound, for $W/\sigma^\beta$ is given by: 


    $$
    \frac{W}{\sigma^\beta}\lt T_{a0}E_{\mathrm{max}}
    $$

    Similarly, for the upper bound, 

    $$
    C_L^* \lt \sqrt{3} \sqrt{\frac{C_{D_0}}{K}}\quad \Rightarrow \quad \frac{W}{\sigma^\beta} \gt  \frac{\sqrt{3}}{2} T_{a0} E_{\mathrm{max}}
    $$

    Which combined with the previous condition and together with the reults yields: 

    $$
    \boxed{\delta_T^* = 1} \qquad \land \qquad \boxed{C_L^* = \frac{T_{a0}\sigma^\beta}{2KW}\left[1 +\sqrt{1- \left(\frac{W}{T_{a0}\sigma^\beta E_{\mathrm{max}}}\right)^2}\right]} \qquad \mathrm{for} \qquad \frac{\sqrt{3}}{2} T_{a0} E_{\mathrm{max}} \lt \frac{W}{\sigma^\beta} \lt T_{a0} E_{\mathrm{max}}
    $$
    """
    )
    return


@app.function
def maxthrust_condition(W, h, E_max, beta, Ta0):
    sigma = atmos.rhoratio(h)
    condition = ((W / (sigma**beta)) < (Ta0 * E_max)) & (
        (W / (sigma**beta)) > (np.sqrt(3) * (Ta0 * E_max) / 2)
    )
    return condition


@app.cell(hide_code=True)
def _(CD0, E_max, K, S, Ta0, W_selected, beta, h_array, idx_selected):
    maxthrust_mask = maxthrust_condition(W_selected, h_array, E_max, beta, Ta0)

    CLopt_maxthrust = np.where(
        maxthrust_mask,
        CL_from_horizontal_constraint(
            W_selected, h_array, S, CD0, K, Ta0, beta, ac_type="jet"
        )[0],
        np.nan,
    )

    velocity_maxthrust_harray = velocity(W_selected, h_array, CLopt_maxthrust, S)

    dTopt_maxthrust = np.where(
        maxthrust_mask,
        1,
        np.nan,
    )

    CLopt_maxthrust_selected = CLopt_maxthrust[idx_selected]
    dTopt_maxthrust_selected = dTopt_maxthrust[idx_selected]

    velocity_maxthrust_selected = velocity_maxthrust_harray[idx_selected]

    power_maxthrust_harray = power(
        h_array, S, CD0, K, CLopt_maxthrust, velocity_maxthrust_harray
    )
    power_maxthrust_selected = power_maxthrust_harray[idx_selected]
    return (
        CLopt_maxthrust,
        CLopt_maxthrust_selected,
        dTopt_maxthrust,
        dTopt_maxthrust_selected,
        power_maxthrust_harray,
        power_maxthrust_selected,
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
    constraint,
    dT_array,
    dTopt_maxthrust,
    dTopt_maxthrust_selected,
    h_array,
    h_selected,
    power_maxthrust_harray,
    power_maxthrust_selected,
    power_surface,
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
                z=power_surface / 1e3,
                opacity=0.9,
                name="Power",
                colorscale="cividis",
            ),
            go.Scatter3d(
                x=CL_array,
                y=constraint,
                z=power_surface[0] / 1e3,
                opacity=0.7,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.1)", width=10),
                name="C2 constraint",
            ),
            go.Scatter3d(
                x=[CL_array[-15]],
                y=[constraint[-15]],
                z=[power_surface[0, -15] / 1e3 + 250],
                opacity=1,
                textposition="middle left",
                mode="markers+text",
                text=["c<sub>2</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="C2 constraint",
            ),
            go.Scatter3d(
                x=[CLopt_maxthrust_selected],
                y=[dTopt_maxthrust_selected],
                z=[
                    power_maxthrust_selected / 1e3 + 100
                ],  # Slightly elevate to show the full marker
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=3,
                    color="white",
                    symbol="circle",
                ),
                name="maxthrust Optimum",
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub> : %{y}<br>P: %{z}<extra>%{fullData.name}</extra>",
            ),
            go.Scatter3d(
                x=[CLopt_maxthrust_selected, CLopt_maxthrust_selected],
                y=[dTopt_maxthrust_selected, xy_lowerbound],
                z=[
                    power_maxthrust_selected / 1e3 + 100,
                    power_maxthrust_selected / 1e3 + 100,
                ],
                mode="lines",
                showlegend=False,
                line=dict(color="grey", width=2),
            ),
            go.Scatter3d(
                x=[xy_lowerbound, CLopt_maxthrust_selected],
                y=[dTopt_maxthrust_selected, dTopt_maxthrust_selected],
                z=[
                    power_maxthrust_selected / 1e3 + 100,
                    power_maxthrust_selected / 1e3 + 100,
                ],
                mode="lines",
                showlegend=False,
                line=dict(color="grey", width=2),
            ),
            go.Scatter3d(
                x=CLopt_maxthrust,
                y=np.ones(len(dT_array)) * xy_lowerbound,
                z=np.tile(power_maxthrust_harray / 1e3 + 100, len(CLopt_maxthrust)),
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(129, 216, 208, 1)", width=8),
            ),
            go.Scatter3d(
                x=np.ones(len(CLopt_maxthrust)) * xy_lowerbound,
                y=dTopt_maxthrust,
                z=np.tile(power_maxthrust_harray / 1e3 + 100, len(CLopt_maxthrust)),
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
                title="P (kW)",
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
def _():
    mo.md(
        r"""
    ### _Lift- and thrust- limited optimum_

    In this case: $C_L = C_{L_{\mathrm{max}}}$, $\delta_T = 1$, $\mu_1 \gt 0$, $\mu_2 \gt 0$

    from stationarity condition (2): $\lambda_1 \lt 0$

    from stationarity condition (1):

    $$
    \mu_1 = - \left.\frac{\partial P}{\partial C_L} \right|_{C_{L_\mathrm{max}}} + \lambda_1 \left.\frac{\partial D}{\partial C_L} \right|_{C_{L_\mathrm{max}}} \gt 0
    $$

    which becomes:

    $$
    \sqrt{\frac{2W^3}{\rho S}}\left(-\frac{3}{2}C_{D_0}C_{L_{\mathrm{max}}}^{-5/2} + \frac{1}{2} K C_{L_{\mathrm{max}}}^{-1/2}\right) - \lambda_1 W \left(\frac{KC_{L_{\mathrm{max}}}^2 -C_{D_0}}{C_{L_{\mathrm{max}}}^2}\right) \gt 0
    $$

    $$
    \Rightarrow \sqrt{\frac{2W}{\rho S}}\left(-\frac{3}{2}C_{D_0}C_{L_{\mathrm{max}}}^{-5/2} + \frac{1}{2} K C_{L_{\mathrm{max}}}^{-1/2}\right) - \lambda_1 \left(\frac{KC_{L_{\mathrm{max}}}^2 -C_{D_0}}{C_{L_{\mathrm{max}}}^2}\right) \gt 0
    $$

    $$
    \Rightarrow \lambda_1 > \sqrt{\frac{2W}{\rho S}} \frac{-\frac{3}{2}C_{D_0}C_{L_{\mathrm{max}}}^{-1/2} + \frac{1}{2} K C_{L_{\mathrm{max}}}^{3/2}}{KC_{L_{\mathrm{max}}}^2 -C_{D_0}}
    $$

    $$
    \Rightarrow 0 \lt \lambda_1 \lt \sqrt{\frac{2W}{\rho S}} \frac{-\frac{3}{2}C_{D_0}C_{L_{\mathrm{max}}}^{-1/2} + \frac{1}{2} K C_{L_{\mathrm{max}}}^{3/2}}{KC_{L_{\mathrm{max}}}^2 -C_{D_0}}
    $$

    This is the true only if:

    $$
    \Rightarrow \frac{-{3}C_{D_0}C_{L_{\mathrm{max}}}^{-1/2} + K C_{L_{\mathrm{max}}}^{3/2}}{KC_{L_{\mathrm{max}}}^2 -C_{D_0}} \lt 0 
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    This yields two solutions. The first one, by assuming $KC_{L_{\mathrm{max}}}^2 -C_{D_0} \lt 0$, yields:

    $$
    C_{L_\mathrm{max}} \lt C_{L_E}\quad \land \quad C_{L_\mathrm{max}} \gt \sqrt{3}C_{L_E} \quad \mathrm{impossible} \; \forall \; C_{L_\mathrm{max}} \in \R
    $$

    The second one results in a suitable solution, by taking $KC_{L_{\mathrm{max}}}^2 -C_{D_0} \gt 0$.

    $$
    C_{L_E} \lt C_{L_{\mathrm{max}}} \lt \sqrt{3}C_{L_E}
    $$

    Opposite to what one might think, the condition $C_{L_{\mathrm{max}}} \lt \sqrt{3}C_{L_E}$ is plausible as this is a design choice. $C_{L_{\mathrm{max}}}$, $C_{D_0}$, and $K$ are in fact all independent with one other.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    Now continuing with the primal feasibility condition (3):

    $$
    T_{a0}\sigma^\beta = W \frac{C_{D_0} + K C_{L_{\mathrm{max}}}^2}{C_{L_{\mathrm{max}}}} = W E_S \Leftrightarrow C_{L_{\mathrm{max}}}^2 - \frac{T_{a0}\sigma^\beta}{KW}C_{L_{\mathrm{max}}}+\frac{C_{D_0}}{K} = 0
    $$

    The solution to the quadratic equation is:

    $$
    \Rightarrow C_L^* = \frac{T_{a0}\sigma^\beta}{2KW}\left[1 \pm\sqrt{1- \left(\frac{W}{T_{a0}\sigma^\beta E_{\mathrm{max}}}\right)^2}\right]
    $$

    which has to belong to the interval $(C_{L_E},\sqrt{3}C_{L_E})$. Therefore we select the solution with the positive sign above, yielding: 


    $$
    C_{L_{\mathrm{max}}} \gt C_{L_E} \Leftrightarrow \frac{W}{\sigma^\beta} \lt \frac{T_{a0}}{2\sqrt{C_{D_0}K}} = T_{a0}E_{\mathrm{max}}
    $$

    $$
    C_{L_{\mathrm{max}}} \lt \sqrt{3}C_{L_E} \Leftrightarrow \frac{W}{\sigma^\beta} \gt \frac{\sqrt{3}}{8}T_{a0}E_{\mathrm{max}}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    This analysis yields the following:

    $$
    \boxed{C_L^* = C_{L_{\mathrm{max}}}} \quad \land \quad \boxed{\delta_T^* = 1} \quad \mathrm{if} \quad C_{L_E} \lt C_{L_{\mathrm{max}}} \lt \sqrt{3}C_{L_E} \quad \mathrm{and \: for} \quad \frac{\sqrt{3}}{8}T_{a0}E_{\mathrm{max}} \lt \frac{W}{\sigma^\beta} \lt T_{a0}E_{\mathrm{max}}
    $$
    """
    )
    return


@app.cell
def _(CD0, CLmax, K):
    def maxlift_thrust_optimum(W, h, Ta0, beta, E_max):
        sigma = atmos.rhoratio(h)  # array if h is array
        # elementwise logical condition
        condition = (
            ((W / sigma**beta) < ((np.sqrt(3) / 8) * Ta0 * E_max))
            & ((W / sigma**beta) > (Ta0 * E_max))
            & (CLmax > np.sqrt(CD0 / K))
            & (CLmax < np.sqrt(3 * CD0 / K))
        )

        return condition
    return (maxlift_thrust_optimum,)


@app.cell
def _(
    CD0,
    CLmax,
    E_max,
    K,
    S,
    Ta0,
    W_selected,
    beta,
    h_array,
    idx_selected,
    maxlift_thrust_optimum,
):
    maxlift_thrust_mask = maxlift_thrust_optimum(
        W_selected, h_array, E_max, beta, Ta0
    )

    CLopt_maxlift_thrust = np.where(
        maxlift_thrust_mask,
        CLmax,
        np.nan,
    )

    velocity_maxlift_thrust_harray = velocity(
        W_selected, h_array, CLopt_maxlift_thrust, S
    )

    dTopt_maxlift_thrust = np.where(maxlift_thrust_mask, 1, np.nan)

    CLopt_maxlift_thrust_selected = CLopt_maxlift_thrust[idx_selected]
    dTopt_maxlift_thrust_selected = dTopt_maxlift_thrust[idx_selected]

    velocity_maxlift_thrust_selected = velocity_maxlift_thrust_harray[idx_selected]

    power_maxlift_thrust_harray = power(
        h_array, S, CD0, K, CLopt_maxlift_thrust, velocity_maxlift_thrust_harray
    )
    power_maxlift_thrust_selected = power_maxlift_thrust_harray[idx_selected]
    return (
        CLopt_maxlift_thrust,
        CLopt_maxlift_thrust_selected,
        dTopt_maxlift_thrust,
        dTopt_maxlift_thrust_selected,
        power_maxlift_thrust_harray,
        power_maxlift_thrust_selected,
        velocity_maxlift_thrust_harray,
        velocity_maxlift_thrust_selected,
    )


@app.cell
def _(
    CL_array,
    CLopt_maxlift_thrust,
    CLopt_maxlift_thrust_selected,
    a_harray,
    active_selection,
    constraint,
    dT_array,
    dTopt_maxlift_thrust,
    dTopt_maxlift_thrust_selected,
    h_array,
    h_selected,
    power_maxlift_thrust_harray,
    power_maxlift_thrust_selected,
    power_surface,
    velocity_maxlift_thrust_harray,
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
                z=power_surface / 1e3,
                opacity=0.9,
                name="Power",
                colorscale="cividis",
            ),
            go.Scatter3d(
                x=CL_array,
                y=constraint,
                z=power_surface[0] / 1e3,
                opacity=0.7,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.1)", width=10),
                name="C2 constraint",
            ),
            go.Scatter3d(
                x=[CL_array[-15]],
                y=[constraint[-15]],
                z=[power_surface[0, -15] / 1e3 + 250],
                opacity=1,
                textposition="middle left",
                mode="markers+text",
                text=["c<sub>2</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="C2 constraint",
            ),
            go.Scatter3d(
                x=[CLopt_maxlift_thrust_selected],
                y=[dTopt_maxlift_thrust_selected],
                z=[
                    power_maxlift_thrust_selected / 1e3 + 100
                ],  # Slightly elevate to show the full marker
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=3,
                    color="white",
                    symbol="circle",
                ),
                name="maxlift_thrust Optimum",
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub> : %{y}<br>P: %{z}<extra>%{fullData.name}</extra>",
            ),
            go.Scatter3d(
                x=[CLopt_maxlift_thrust_selected, CLopt_maxlift_thrust_selected],
                y=[dTopt_maxlift_thrust_selected, xy_lowerbound],
                z=[
                    power_maxlift_thrust_selected / 1e3 + 100,
                    power_maxlift_thrust_selected / 1e3 + 100,
                ],
                mode="lines",
                showlegend=False,
                line=dict(color="grey", width=2),
            ),
            go.Scatter3d(
                x=[xy_lowerbound, CLopt_maxlift_thrust_selected],
                y=[dTopt_maxlift_thrust_selected, dTopt_maxlift_thrust_selected],
                z=[
                    power_maxlift_thrust_selected / 1e3 + 100,
                    power_maxlift_thrust_selected / 1e3 + 100,
                ],
                mode="lines",
                showlegend=False,
                line=dict(color="grey", width=2),
            ),
            go.Scatter3d(
                x=CLopt_maxlift_thrust,
                y=np.ones(len(dT_array)) * xy_lowerbound,
                z=np.tile(
                    power_maxlift_thrust_harray / 1e3 + 100,
                    len(CLopt_maxlift_thrust),
                ),
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(129, 216, 208, 1)", width=8),
            ),
            go.Scatter3d(
                x=np.ones(len(CLopt_maxlift_thrust)) * xy_lowerbound,
                y=dTopt_maxlift_thrust,
                z=np.tile(
                    power_maxlift_thrust_harray / 1e3 + 100,
                    len(CLopt_maxlift_thrust),
                ),
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(129, 216, 208, 1)", width=8),
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
                x=velocity_maxlift_thrust_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(width=3, color="rgba(129, 216, 208, 1)"),
                showlegend=False,
                name="P_min",
            ),
            go.Scatter(
                x=[velocity_maxlift_thrust_selected],
                y=[h_selected / 1e3],
                mode="markers+text",
                marker=dict(size=5, color="white"),
                name="maxlift_thrust Optimum",
                showlegend=False,
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
                title="P (kW)",
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


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""Now after deriving all the optima for each condition we can summarize the flight envelopes in one graph, as shown below. Experiment with the weight of the aircrarft to understand how the theoretical ceiling for minimum power moves in the graph."""
    )
    return


@app.cell
def _(velocity_interior_harray, velocity_maxthrust_harray):
    final_velocity_flightenvelope = np.where(
        np.isnan(velocity_interior_harray),
        velocity_maxthrust_harray,
        velocity_interior_harray,
    )
    return (final_velocity_flightenvelope,)


@app.cell
def _(
    a_harray,
    active_selection,
    final_velocity_flightenvelope,
    h_array,
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
def _(mass_stack):
    mass_stack
    return


@app.cell
def _(fig_final_flightenv):
    fig_final_flightenv
    return


if __name__ == "__main__":
    app.run()
