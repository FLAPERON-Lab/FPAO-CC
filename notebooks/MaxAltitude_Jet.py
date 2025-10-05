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
    import numpy as np
    from core import atmos
    from core import aircraft as ac
    from core.aircraft import velocity

    # Set local/online filepath
    _defaults.FILEURL = _defaults.get_url()

    # Plotly dark mode template
    _defaults.set_plotly_template()

    # Data directory
    data_dir = str(mo.notebook_location() / "public" / "AircraftDB_Standard.csv")
    return ac, atmos, data_dir, go, make_subplots, mo, np, velocity


@app.cell
def _():
    # Set navbar on the right
    _defaults.set_sidebar()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Maximum Altitude: simplified jet aircraft

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
        & \quad T_a(V,h) = T_a(h) = T_{a0}\sigma^\beta \\
    \end{aligned}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Here, h does not appear explicitely but we can transform the problem formulation in a convenient way, by knowing $\rho(h)$ is a monotonically decreasing function of h, as shown in the graph below.

    $$
    \begin{aligned}
        \min_{C_L, \delta_T} 
        & \quad h  \qquad \Longleftrightarrow \qquad \max_{C_L, \delta_T} \quad \sigma = \frac{\rho(h)}{\rho_0}\\
    \end{aligned}
    $$
    """
    )
    return


@app.cell
def _(atmos, np):
    # Insert graph of density decreasing with increasing altitude.
    meshgrid = 101
    xy_lowerbound = -0.15

    h_array = np.linspace(0, 20e3, meshgrid)

    rhoratio_harray = atmos.rhoratio(h_array)
    return h_array, meshgrid, rhoratio_harray, xy_lowerbound


@app.cell
def _(go, h_array, rhoratio_harray):
    figure_height_relation = go.Figure()

    figure_height_relation.add_traces(
        [go.Scatter(x=h_array * 1e-3, y=rhoratio_harray, name=r"$\sigma")]
    )

    figure_height_relation.update_layout(
        yaxis=dict(title=r"$\sigma \quad \mathrm{(-)}$", showgrid=True),
        xaxis=dict(title=r"$h \quad\text{(km)}$", showgrid=True),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Moreover, since density is always positive, and $\beta$ as well, we can say, because $\sigma^\beta$ is a monotically increasing function of $\sigma$, minimizing $\sigma^\beta$ minimizes $\sigma$ which is maximizing $h$.

    $$
    \min_{C_L, \delta_T} \sigma  \qquad \Longleftrightarrow \qquad \max_{C_L, \delta_T} \quad \sigma^\beta
    $$

    We can thus now susbitute the horizontal equilibrium equation in the objective function directly, and then also substitute the expression of $V$ rom the vertical equilibrium, constraint.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## KKT formulation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The KKT formulation can now be written:""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    \begin{aligned}
        \min_{C_L, \delta_T} 
        & \quad \sigma^\beta = \left[\frac{W}{\delta_T T_{a0}}\left(\frac{C_{D_0} + K C_L^2}{C_L}\right)\right]\\
        \text{subject to}
        & \quad h_1 = C_L - C_{L_\mathrm{max}} \le 0 \\
        & \quad h_2 = \delta_T - 1 \le 0 \\
    \end{aligned}
    $$
    """
    )
    return


@app.cell
def _(CD0, CL_grid, K, Ta0, W_selected, beta, dT_grid, np):
    rhoratio_surface = (
        W_selected
        / (dT_grid * Ta0)
        * ((CD0 + K * CL_grid**2)
        / CL_grid)
    )**(1/beta)

    min_colorbar = np.nanmin(rhoratio_surface ** beta)
    max_colorbar = 1
    return max_colorbar, min_colorbar, rhoratio_surface


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The lower bounds for the lift coefficient ($C_L = 0$), and for $\delta_T$ have already been excluded as they cannot comply with the vertical and horizontal constraints respectively.

    As it can be noted, the problem is now formulated to have only inequality constraints due to the bounds on the decision variables. In other words, it is an unconstrained optimization problem in a partially bounded domain.
    """
    )
    return


@app.cell(hide_code=True)
def _(ac, data_dir, mo):
    # Database cell

    data = ac.available_aircrafts(data_dir, ac_type="Jet")

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
def _(
    CL_array,
    CL_slider,
    active_selection,
    beta,
    dT_array,
    dT_slider,
    go,
    idx_CL,
    idx_dT,
    max_colorbar,
    min_colorbar,
    mo,
    rhoratio_surface,
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
                z=rhoratio_surface ** beta,
                opacity=0.9,
                name="σ<sup>β</sup>",
                colorscale="viridis",
                cmin=min_colorbar,
                cmax=max_colorbar,
                colorbar={"title": "σ (-)"},
            ),
            go.Scatter3d(
                x=[CL_slider.value],
                y=[dT_slider.value],
                z=[rhoratio_surface[idx_dT, idx_CL] ** beta],
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

    # Create stacks
    mass_stack = mo.hstack(
        [mo.md("**OEW**"), m_slider, mo.md("**MTOW**")],
        align="start",
        justify="start",
    )

    h_slider = h_slider = mo.ui.slider(
        start=0,
        stop=20,
        label=r"Altitude (km)",
        value=10,
        show_value=True,
    )
    return CL_slider, active_selection, dT_slider, m_slider, mass_stack


@app.cell
def _(
    CL_slider,
    W_selected,
    active_selection,
    atmos,
    dT_slider,
    h_array,
    meshgrid,
    np,
    velocity,
):
    CLmax = active_selection["CLmax_ld"]
    Ta0 = active_selection["Ta0"] * 1e3
    CD0 = active_selection["CD0"]
    K = active_selection["K"]
    beta = active_selection["beta"]
    OEW = active_selection["OEM"] * atmos.g0
    MTOW = active_selection["MTOM"] * atmos.g0
    S = active_selection["S"]
    E_max = 1 / np.sqrt(4 * CD0 * K)
    E_S = (CD0 + K * CLmax**2) / CLmax
    CL_E = np.sqrt(CD0 / K)

    dT_array = np.linspace(0.0, 1, meshgrid)

    dT_array[0] = dT_array[1]

    CL_array = np.linspace(0.0, CLmax, meshgrid)

    CL_array[0] = CL_array[1]

    CL_grid, dT_grid = np.meshgrid(CL_array, dT_array)

    idx_dT = int(round(dT_slider.value / 1 * (meshgrid - 1)))
    idx_CL = int(round(CL_slider.value / CLmax * (meshgrid - 1)))

    a_harray = atmos.a(h_array)
    velocity_stall_harray = velocity(W_selected, h_array, CLmax, S)
    return (
        CD0,
        CL_array,
        CL_grid,
        CLmax,
        E_S,
        E_max,
        K,
        S,
        Ta0,
        a_harray,
        beta,
        dT_array,
        dT_grid,
        idx_CL,
        idx_dT,
        velocity_stall_harray,
    )


@app.cell
def _(active_selection, atmos, m_slider):
    W_selected = (
        active_selection["OEM"]
        + (active_selection["MTOM"] - active_selection["OEM"]) * m_slider.value
    ) * atmos.g0  # Netwons
    return (W_selected,)


@app.cell(hide_code=True)
def _(CL_slider, dT_slider, mo):
    mo.md(rf"""Here you can modify the control variables to understand how it affects the design: {mo.hstack([dT_slider, CL_slider])}""")
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
    mo.md(
        r"""
    ### Lagrangian function and KKT conditions

    The Lagrangian function combines the objective function with the inequality constraints using KKT multipliers ($\mu_j$).

    $$
    \begin{aligned}
    \mathcal{L}(C_L, \delta_T, \mu_1, \mu_2) = & \sigma^\beta + \mu_1 (C_L - C_{L_\mathrm{max}}) +\mu_2 (\delta_T - 1)\\ 
    =&\left[\frac{W}{\delta_T T_{a0}}\left(\frac{C_{D_0} + K C_L^2}{C_L}\right)\right] +\\
    & + \mu_1 \left(C_L - C_{L_\mathrm{max}}\right) + \\
    & + \mu_2 (\delta_T - 1) \\
    \end{aligned}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    A necessary condition for an optimal solution of the optimization problem $(C_L^*, \delta_T^*)$ to exist, the multipliers $\lambda_1, \mu_1, \mu_2$ have to meet the following conditions:

    **A. Stationarity ($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \frac{\partial \mathcal{L}}{\partial C_L} = \frac{W}{\delta_T T_{a0}}\left(\frac{K C_L^2 - C_{D_0}}{C_L^2}\right) + \mu_1= 0$

    3.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = - \frac{W}{\delta_T^2 T_{a0}}\left(\frac{C_{D_0} + K C_L^2}{C_L}\right) + \mu_2= 0$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $C_L - C_{L_\mathrm{max}} \le 0$
    4.  $\delta_T - 1 \le 0$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    5.  $\mu_1, \mu_2\ge 0$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraint have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

    6.  $\mu_1 (C_L - C_{L_\mathrm{max}}) = 0$
    7. $\mu_3 (\delta_T - 1) = 0$
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

    Assuming that that $C_L < C_{L_\mathrm{max}}$ and $\delta_T < 1$ is equivalent to consider all inequality constraints as inactive.

    Therefore: $\mu_1,\mu_2=0$. 

    It is clear from stationarity condition 2, that the equation cannot be solved for any value of $\delta_T$.

    It can be concluded that the maximum speed cannot be achieved in the interior of the domain. 
    The minimum must lie on at least one of the boundaries defined by $C_L = C_{L_\mathrm{max}}$ or $\delta_T = 1$.

    Moreover, the stationarity condition 2 can be solved for a value of $\delta_T$ only when $\mu_2 \neq 0$, this means it also pointless to investigate the _max-lift condition_ as we would have $\mu_2 = 0$ again.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### _Thrust-limited minimum airspeed_

    $C_L < C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 = 0$

    $\delta_T=1 \quad \Rightarrow \quad \mu_2 > 0$

    From stationarity condition (1): 

    $$
    C_L^*= \sqrt{\frac{C_{D_0}}{K}}=C_{L_E}
    $$

    while stationarity condition (2) is always satisfied given $\delta_T = 1$.

    This condition is achievable only if $C_L^* \lt C_{L_\mathrm{max}}$ mening the aircraft is able to fly on the induced branch of the drag performance diagram.

    The corresponding altitude is given by the density ratio: 

    $$
    \displaystyle \sigma^* = \left(\frac{W}{T_{a0}E_{max}}\right)^{\frac{1}{\beta}}
    $$

    which depends on the weight. We call this the "theoretical ceiling", by inspecting the equation for the density ratio, the lower the weight, the lower $\sigma$, and thus the higher the altitude $h$ of the ceiling.

    The operational condition is given by:

    $$
    \frac{W}{\sigma^{*^\beta}} = T_{a0}E_{\mathrm{max}}
    $$
    """
    )
    return


@app.cell
def _(atmos, np):
    def maxthrust_altitude(W, beta, Ta0, E_max, CL_star, CLmax):
        """
        Returns either the optimum weight or the optimum height
        """

        sigma_exp = W / Ta0 / E_max

        out = sigma_exp ** (1 / beta)

        return np.where(
            (CL_star < CLmax) & (out > atmos.rhoratio(20e3)), out, np.nan
        )
    return (maxthrust_altitude,)


@app.cell
def _(
    CD0,
    CL_array,
    CL_grid,
    CLmax,
    E_max,
    K,
    S,
    Ta0,
    W_selected,
    atmos,
    beta,
    dT_grid,
    maxthrust_altitude,
    np,
    velocity,
):
    CLopt_maxthrust = np.sqrt(CD0 / K)

    maxthrust_sigma = maxthrust_altitude(
        W_selected, beta, Ta0, E_max, CLopt_maxthrust, CLmax
    )

    maxthrust_h = atmos.altitude(maxthrust_sigma)

    velocity_maxthrust_selected = velocity(
        W_selected, maxthrust_h, CLopt_maxthrust, S, cap=False
    )

    velocity_CLarray_maxthrust_h = velocity(
        W_selected, maxthrust_h, CL_array, S, cap=False
    )

    dTopt_maxthrust = 1

    rhoratio_surface_maxthrust = (
        W_selected
        / (dT_grid * Ta0)
        * ((CD0 + K * CL_grid**2)
        / CL_grid)
    )**(1/beta)

    if np.isnan(maxthrust_sigma):
        CLopt_maxthrust = np.nan

    max_colorbar_maxthrust = 1
    min_colorbar_maxthrust = np.nanmin(rhoratio_surface_maxthrust ** beta)
    return (
        CLopt_maxthrust,
        dTopt_maxthrust,
        max_colorbar_maxthrust,
        maxthrust_h,
        maxthrust_sigma,
        min_colorbar_maxthrust,
        rhoratio_surface_maxthrust,
        velocity_maxthrust_selected,
    )


@app.cell
def _(
    CL_array,
    CLopt_maxthrust,
    a_harray,
    active_selection,
    atmos,
    beta,
    dT_array,
    dTopt_maxthrust,
    go,
    h_array,
    make_subplots,
    max_colorbar_maxthrust,
    maxthrust_h,
    maxthrust_sigma,
    min_colorbar_maxthrust,
    mo,
    rhoratio_surface_maxthrust,
    velocity_maxthrust_selected,
    velocity_stall_harray,
    xy_lowerbound,
):
    fig_maxthrust_optimum = make_subplots(
        rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}]]
    )

    # Traces on the 3D plot, first four are template
    fig_maxthrust_optimum.add_traces(
        [
            go.Heatmap(
                x=CL_array,
                y=dT_array,
                z=rhoratio_surface_maxthrust ** beta,
                opacity=0.9,
                name="σ<sup>β</sup> (-)",
                colorscale="viridis",
                zsmooth="best",
                zmin=min_colorbar_maxthrust,
                zmax=max_colorbar_maxthrust,
                colorbar={"title": "σ<sup>β</sup> (-)"},
            ),
            go.Scatter(
                x=[CLopt_maxthrust],
                y=[dTopt_maxthrust],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=10,
                    color="#FFFFFF",
                    symbol="circle",
                ),
                name="σ<sup>β</sup><sub>min</sub>",
                customdata=[maxthrust_sigma**beta],
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub>: 1 <br>σ<sup>β</sup>: %{customdata}<extra></extra>",
            ),
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                showlegend=False,
                marker=dict(
                    color="rgba(0, 0, 0, 0)",
                ),
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
                x=[velocity_maxthrust_selected],
                y=[maxthrust_h / 1e3],
                mode="markers",
                marker=dict(size=10, color="#FFFFFF"),
                showlegend=False,
                name="σ<sup>β</sup><sub>min</sub>",
            ),
        ],
        cols=2,
        rows=1,
    )

    fig_maxthrust_optimum.update_xaxes(
        title_text=r"$C_L\:(\text{-})$",
        range=[xy_lowerbound, active_selection["CLmax_ld"] + 0.05],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=1,
    )
    fig_maxthrust_optimum.update_yaxes(
        title_text=r"$\delta_T \:(\text{-})$",
        range=[xy_lowerbound, 1 + 0.05],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=1,
    )

    # Second subplot: V vs h
    fig_maxthrust_optimum.update_xaxes(
        title_text=r"$V \: \text{(m/s)}$",
        range=[xy_lowerbound, max(atmos.a(0), velocity_maxthrust_selected) + 15],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=2,
    )
    fig_maxthrust_optimum.update_yaxes(
        title_text=r"$h \: 	\text{(km)}$",
        range=[xy_lowerbound, 20],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=2,
    )

    fig_maxthrust_optimum.update_layout(
        title={
            "text": f"Thrust-limited maximum altitude for {active_selection.full_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        }
    )

    mo.output.clear()
    return (fig_maxthrust_optimum,)


@app.cell
def _(mass_stack):
    mass_stack
    return


@app.cell
def _(fig_maxthrust_optimum):
    fig_maxthrust_optimum
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### _Thrust- and lift-limited minimum speed_

    $\delta_T = 1 \quad \Rightarrow \quad \mu_3 > 0$

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 > 0$.

    From the stationary conditions (1):

    $$
    \mu_1 = \frac{W}{T_{a0}}\left(\frac{C_{D_0} - K C_{L_\mathrm{max}}^2} {C_{L_\mathrm{max}}^2}  \right) \gt 0 \quad \Longleftrightarrow \quad C_{L_\mathrm{max}} \lt \sqrt{\frac{C_{D_0}}{K}} = C_{L_{E}}
    $$

    In this case the aircraft stalls at a higher speed than the one for $E_{\mathrm{max}}$, and therefore $C_{L_\mathrm{max}}$ constraints the performance. 

    The corresponding altitude is given by the density ratio: 

    $$
    \displaystyle \sigma^* = \left(\frac{W}{T_{a0}E_{S}}\right)^{\frac{1}{\beta}}
    $$

    While the operational condition is given by:

    $$
    \frac{W}{\sigma^{*^\beta}} = T_{a0}E_{\mathrm{S}}
    $$
    """
    )
    return


@app.cell
def _(atmos, np):
    def maxlift_thrust_altitude(W, beta, Ta0, E_S, CD0, K, CLmax):
        """
        Returns either the optimum weight or the optimum height
        """

        sigma_exp = W / Ta0 / E_S

        out = sigma_exp ** (1 / beta)

        return np.where(
            (CLmax < np.sqrt(CD0 / K)) & (out > atmos.rhoratio(20e3)), out, np.nan
        )
    return (maxlift_thrust_altitude,)


@app.cell
def _(
    CD0,
    CL_array,
    CL_grid,
    CLmax,
    E_S,
    K,
    S,
    Ta0,
    W_selected,
    atmos,
    beta,
    dT_grid,
    maxlift_thrust_altitude,
    np,
    velocity,
):
    CLopt_maxlift_thrust = np.sqrt(CD0 / K)

    maxlift_thrust_sigma = maxlift_thrust_altitude(
        W_selected, beta, Ta0, E_S, CD0, K, CLmax
    )

    maxlift_thrust_h = atmos.altitude(maxlift_thrust_sigma)

    velocity_maxlift_thrust_selected = velocity(
        W_selected, maxlift_thrust_h, CLopt_maxlift_thrust, S, cap=False
    )

    velocity_CLarray_maxlift_thrust_h = velocity(
        W_selected, maxlift_thrust_h, CL_array, S, cap=False
    )

    dTopt_maxlift_thrust = 1

    rhoratio_surface_maxlift_thrust = (
        W_selected
        / (dT_grid * Ta0)
        * (CD0 + K * CL_grid**2)
        / CL_grid
    )**(1/beta)

    if np.isnan(maxlift_thrust_sigma):
        CLopt_maxlift_thrust = np.nan

    min_colorbar_maxlift_thrust = np.nanmin(rhoratio_surface_maxlift_thrust ** beta)

    max_colorbar_maxlift_thrust = 1
    return (
        CLopt_maxlift_thrust,
        dTopt_maxlift_thrust,
        max_colorbar_maxlift_thrust,
        maxlift_thrust_h,
        maxlift_thrust_sigma,
        min_colorbar_maxlift_thrust,
        rhoratio_surface_maxlift_thrust,
        velocity_maxlift_thrust_selected,
    )


@app.cell
def _(
    CL_array,
    CLopt_maxlift_thrust,
    a_harray,
    active_selection,
    atmos,
    beta,
    dT_array,
    dTopt_maxlift_thrust,
    go,
    h_array,
    make_subplots,
    max_colorbar_maxlift_thrust,
    maxlift_thrust_h,
    maxlift_thrust_sigma,
    min_colorbar_maxlift_thrust,
    mo,
    rhoratio_surface_maxlift_thrust,
    velocity_maxlift_thrust_selected,
    velocity_stall_harray,
    xy_lowerbound,
):
    fig_maxlift_thrust_optimum = make_subplots(
        rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}]]
    )

    # Traces on the 3D plot, first four are template
    fig_maxlift_thrust_optimum.add_traces(
        [
            go.Heatmap(
                x=CL_array,
                y=dT_array,
                z=rhoratio_surface_maxlift_thrust ** beta,
                opacity=0.9,
                name="σ<sup>β</sup> (-)",
                colorscale="viridis",
                zsmooth='best',
                zmin=min_colorbar_maxlift_thrust,
                zmax=max_colorbar_maxlift_thrust,
                colorbar={'title' : "σ<sup>β</sup> (-)"}
            ),
            go.Scatter(
                x=[CLopt_maxlift_thrust],
                y=[dTopt_maxlift_thrust],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=10,
                    color="#FFFFFF",
                    symbol="circle",
                ),
                name="σ<sup>β</sup><sub>min</sub>",
                customdata=[maxlift_thrust_sigma**beta],
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub>: 1 <br>σ<sup>β</sup>: %{customdata}<extra></extra>",
    ),
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                showlegend=False,
                marker=dict(
                    color="rgba(0, 0, 0, 0)",
                ),
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
                marker=dict(size=10, color="#FFFFFF"),
                showlegend=False,
                name="σ<sup>β</sup><sub>min</sub>",
            ),
        ],
        cols=2,
        rows=1,
    )

    fig_maxlift_thrust_optimum.update_xaxes(
        title_text=r"$C_L\:(\text{-})$",
        range=[xy_lowerbound, active_selection["CLmax_ld"] + 0.05],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=1,
    )
    fig_maxlift_thrust_optimum.update_yaxes(
        title_text=r"$\delta_T \:(\text{-})$",
        range=[xy_lowerbound, 1 + 0.05],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=1,
    )

    # Second subplot: V vs h
    fig_maxlift_thrust_optimum.update_xaxes(
        title_text=r"$V \: \text{(m/s)}$",
        range=[xy_lowerbound, atmos.a(0) + 15],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=2,
    )
    fig_maxlift_thrust_optimum.update_yaxes(
        title_text=r"$h \: 	\text{(km)}$",
        range=[xy_lowerbound, 20],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=2,
    )

    fig_maxlift_thrust_optimum.update_layout(
        title={
            "text": f"Thrust-lift limited maximum altitude for {active_selection.full_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        }
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
    mo.md(r"""Now after deriving all the optima for each condition we can summarize the flight envelopes in one graph, as shown below. Experiment with the weight of the aircrarft to understand how the theoretical ceiling for minimum power moves in the graph.""")
    return


@app.cell
def _(
    a_harray,
    active_selection,
    atmos,
    go,
    h_array,
    maxlift_thrust_h,
    maxthrust_h,
    mo,
    velocity_maxlift_thrust_selected,
    velocity_maxthrust_selected,
    velocity_stall_harray,
    xy_lowerbound,
):
    # Initial Figure
    fig_final_flightenv = go.Figure()

    # Traces on the flight envelope, first four traces are template
    fig_final_flightenv.add_traces(
        [   go.Scatter(
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
                x=[velocity_maxthrust_selected],
                y=[maxthrust_h / 1e3],
                mode="markers",
                marker=dict(size=10, color="rgb(232,158,184)"),
                showlegend=False,
                name="σ<sup>β</sup><sub>min</sub>",
            ),
            go.Scatter(
                x=[velocity_maxlift_thrust_selected],
                y=[maxlift_thrust_h / 1e3],
                mode="markers",
                marker=dict(size=10, color="rgb(232,158,184)"),
                showlegend=False,
                name="σ<sup>β</sup><sub>min</sub>",
            ),
        ],
    )

    fig_final_flightenv.update_layout(
        xaxis=dict(
            title=r"$V \: \text{(m/s)}$",
            range=[xy_lowerbound, max(atmos.a(0), max(velocity_maxlift_thrust_selected, velocity_maxthrust_selected)) + 15],
            showgrid=True,
            gridcolor="#515151",
            gridwidth=1,
        ),
        yaxis=dict(
            title=r"$h \: 	\text{(km)}$",
            range=[xy_lowerbound, 20],
            showgrid=True,
            gridcolor="#515151",
            gridwidth=1,
        ),
    )
    fig_final_flightenv.update_layout(
        title={
            "text": f"Flight envelope for maximum altitude for {active_selection.full_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        }
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
