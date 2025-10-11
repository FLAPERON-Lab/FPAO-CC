import marimo

__generated_with = "0.16.5"
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
    from core.aircraft import (
        velocity,
        horizontal_constraint,
        drag,
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
        horizontal_constraint,
        make_subplots,
        mo,
        np,
        velocity,
    )


@app.cell
def _():
    _defaults.set_sidebar()
    return


@app.cell
def _(CL_slider, active_selection, atmos, endurance, h_slider, m_slider, np):
    # Variables declared
    meshgrid_n = 101
    xy_lowerbound = -0.1

    CL_array = np.linspace(0, active_selection["CLmax_ld"], meshgrid_n)  # -
    CL_array[0] = CL_array[1]
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
    idx_CL_selected = int(CL_slider.value / 0.5)
    return (
        CD0,
        CL_E,
        CL_P,
        CL_array,
        CLmax,
        E_S,
        E_max,
        K,
        S,
        Ta0,
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
    np,
    velocity,
):
    velocity_CL_E = float(velocity(W_selected, h_selected, CL_E, S, False))
    velocity_CL_P = float(velocity(W_selected, h_selected, CL_P, S, False))
    velocity_stall_selected = float(velocity(W_selected, h_selected, CLmax, S))
    velocity_CLarray = velocity(W_selected, h_selected, CL_array, S, False)
    velocity_CLarray = np.where(
        np.isnan(velocity_CLarray), np.nanmax(velocity_CLarray), velocity_CLarray
    )

    velocity_CLarray_capped = velocity(W_selected, h_selected, CL_array, S, True)
    velocity_stall_harray = velocity(W_selected, h_array, CLmax, S)
    return velocity_CLarray, velocity_CLarray_capped, velocity_stall_harray


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
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## KKT formulation

    To simplify the optimization form, we express $V$ from $c_1^\mathrm{eq}$ and substitute it out of the entire problem to eliminate it. The KKT formulation thus becomes:
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
    & \quad g_1 = \frac{T}{W} - \frac{1}{E}  =\frac{\delta_T T_{a0}\sigma^\beta}{W} - \frac{C_{D_0} + K C_L^2}{C_L} = 0 \\
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
def _(mo):
    mo.md(r"""Below you can see the graph of the domain $0 \lt C_L \lt C_{L_{\mathrm{max}}}$ and $0 \lt \delta_T \lt 1$, with the surface $D$ and the contraint $g_1$ in red. Choose a simplified jet aircraft of your liking in the database below.""")
    return


@app.cell(hide_code=True)
def _(ac, data_dir, mo):
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


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell(hide_code=True)
def _(
    CD0,
    CL_array,
    CL_slider,
    K,
    S,
    Ta0,
    W_selected,
    beta,
    drag,
    h_selected,
    horizontal_constraint,
    np,
    velocity,
    velocity_CLarray,
    velocity_CLarray_capped,
):
    velocity_user_selected = velocity(
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

    drag_curve = np.where(np.isnan(drag_curve), np.nanmax(drag_curve), drag_curve)

    drag_curve_a_capped = drag(
        h_selected,
        S,
        CD0,
        K,
        CL_array,
        velocity_CLarray_capped,
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

    drag_surface = np.tile(drag_curve, (len(CL_array), 1))

    drag_selected = drag(
        h_selected,
        S,
        CD0,
        K,
        CL_slider.value,
        velocity(W_selected, h_selected, CL_slider.value, S),
    )

    min_colorbar = np.nanmin(drag_curve)
    max_colorbar = np.nanmin(drag_curve) * 3
    return constraint, drag_selected, drag_surface, max_colorbar, min_colorbar


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
                x=[CL_array[-65]],
                y=[constraint[-65]],
                z=[drag_surface[0, -65] + 7e3],
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
    mo.md(rf"""Here you can modify the control variables to understand how it affects the design: {mo.hstack([dT_slider, CL_slider])}""")
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
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The multipliers $\lambda_1, \mu_1, \mu_2, \mu_3, \mu_4$ have to meet the following conditions for an optimal solution of the optimization problem $(C_L^*, \delta_T^*)$ to exist.

    **A. Stationarity conditions($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \begin{aligned}\frac{\partial \mathcal{L}}{\partial C_L} = W \frac{K C_L^2 - C_{D_0}}{C_L^2} - \lambda_1W\left(\frac{K C_L^2 - C_{D_0}}{C_L^2}\right) + \mu_1 - \mu_2 = W\frac{K C_L^2 - C_{D_0}}{C_L^2} (1 -\lambda_1) +\mu_1 - \mu_2 = 0 \end{aligned}$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1\frac{T_{a0}\sigma^\beta}{W}+\mu_3-\mu_4 = 0$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $\displaystyle \frac{\delta_T T_{a0}\sigma^\beta}{W} - \frac{C_{D_0} + K C_L^2}{C_L} = 0$
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
    It is evident that $\mu_2$ and $\mu_4$ can never be active, as we would have an unfeasible situation ($C_L = \delta_T = 0$). In other words, strictly for aircraft flight: $C_L \gt 0$ and $\delta_T \gt 0$. Therefore, we can simplify the analysis by setting these two KKT multipliers to zero: 

    $$
    \begin{aligned}
    \mu_2 = \mu_4 = 0
    \end{aligned}
    $$

    We can now rewrite the new conditions to simplify the problem. We will refer to these simplified conditions for the entire notebook.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Simplified conditions**

    1. $\displaystyle W\frac{K C_L^2 - C_{D_0}}{C_L^2} (1 -\lambda_1) +\mu_1 = 0$
    2. $\displaystyle \lambda_1\frac{T_{a0}\sigma^\beta}{W}+\mu_3 = 0$
    3. $\displaystyle \frac{\delta_T T_{a0}\sigma^\beta}{W} - \frac{C_{D_0} + K C_L^2}{C_L} = 0$
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

    We can now systematically examine the conditions where various inequality constraints are active or inactive.

    ### _Interior optimum for minimum drag_ 

    Assuming that that $0 < C_L^* < C_{L_\mathrm{max}}$ and $0 < \delta_T^* < 1$ is equivalent to consider all inequality constraints as inactive.

    Therefore: $\mu_1, \mu_3 =0$. 

    From stationarity condition (2): $\lambda_1 = 0$.

    From stationarity condition (1), it is possible to obtain the value of $C_L^*$ for minimum drag.

    $$
    C_L^* = \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$

    Notice how the optimal $C_L^*$ has the **same value** for maximum aerodynamic efficiency (maximum $C_L /C_D$), for 
    $0\lt C_L \lt  C_{L_\mathrm{max}}$ and $0 \lt \delta_T \lt 1$, as shown in [Aerodynamic Efficiency](/?file=AerodynamicEfficiency.py).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    D_{\mathrm{min}}^* =  \frac{1}{2}\rho {V^*}^2 S C_D= 
    2W\sqrt{KC_{D_0}}=\frac{W}{E_\mathrm{max}}
    $$

    We can now rewrite $\delta_T^*$ in terms of $D_\mathrm{min}$:

    $$
    \delta_T^*=\frac{2W}{T_{a0}\sigma^\beta}\sqrt{C_{D_0}K}=\frac{D_\mathrm{min}^*}{T_{a0}\sigma^{\beta}}
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    This concludes the analysis for the minimum drag of a simplified jet aircraft in the domain's interior. Below is a summary of the optima:

    $$
    \boxed{C_L^* = \sqrt{\frac{C_{D_0}}{K}}}, \quad \boxed{\delta_T^*=\frac{2W}{T_{a0}\sigma^\beta}\sqrt{C_{D_0}K}=\frac{D_\mathrm{min}^*}{T_{a0}\sigma^{\beta}}}, \quad \text{for} \quad C_L^* \lt C_{L_\mathrm{max}}\quad \text{and} \quad \frac{W}{\sigma^\beta} \lt T_{a0}E_\mathrm{max}
    $$

    With the optimal value for minimum drag: 

    $$
    D_{\mathrm{min}}^* = 2W\sqrt{KC_{D_0}}=\frac{W}{E_\mathrm{max}}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.
    """
    )
    return


@app.cell
def _(atmos):
    def interior_condition(W, h, E_max, Ta0, beta, CLmax, CL_E):
        sigma = atmos.rhoratio(h)
        condition = ((W / (sigma**beta)) < (E_max * Ta0)) & (CL_E < CLmax)

        return condition
    return (interior_condition,)


@app.cell
def _(
    CL_E,
    CLmax,
    E_max,
    S,
    Ta0,
    W_selected,
    atmos,
    beta,
    h_array,
    idx_selected,
    interior_condition,
    np,
    velocity,
):
    interior_mask = interior_condition(
        W_selected, h_array, E_max, Ta0, beta, CLmax, CL_E
    )

    CLopt_interior = np.where(interior_mask, CL_E, np.nan)

    velocity_interior_harray = velocity(
        W_selected, h_array, CLopt_interior, S, cap=False
    )

    drag_interior_harray = np.where(interior_mask, W_selected / E_max, np.nan)

    dTopt_interior = drag_interior_harray / Ta0 / (atmos.rhoratio(h_array) ** beta)

    CLopt_interior_selected = CLopt_interior[idx_selected]
    dTopt_interior_selected = dTopt_interior[idx_selected]

    velocity_interior_selected = velocity_interior_harray[idx_selected]

    drag_interior_selected = drag_interior_harray[idx_selected]
    return (
        CLopt_interior_selected,
        dTopt_interior_selected,
        drag_interior_selected,
        velocity_interior_harray,
        velocity_interior_selected,
    )


@app.cell
def _(
    CL_array,
    CLopt_interior_selected,
    a_harray,
    active_selection,
    atmos,
    constraint,
    dT_array,
    dTopt_interior_selected,
    drag_interior_selected,
    drag_surface,
    go,
    h_array,
    h_selected,
    make_subplots,
    max_colorbar,
    min_colorbar,
    mo,
    np,
    velocity_interior_harray,
    velocity_interior_selected,
    velocity_stall_harray,
    xy_lowerbound,
):
    fig_interior_optimum = make_subplots(
        rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}]]
    )

    # Traces on the 3D plot, first four are template
    fig_interior_optimum.add_traces(
        [
            go.Heatmap(
                x=CL_array,
                y=dT_array,
                z=drag_surface,
                opacity=0.9,
                name="Drag",
                colorscale="viridis",
                zsmooth="best",
                zmin=min_colorbar,
                zmax=max_colorbar,
                colorbar={"title": "Drag (N)"},
            ),
            go.Scatter(
                x=CL_array,
                y=constraint,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
                name="g1 constraint",
            ),
            go.Scatter(
                x=[CL_array[-15]],
                y=[constraint[-15] + 0.07],
                opacity=1,
                textposition="middle left",
                mode="markers+text",
                text=["g<sub>1</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="g1 constraint",
                textfont=dict(size=14, family="Arial"),
            ),
            go.Scatter(
                x=[CLopt_interior_selected],
                y=[dTopt_interior_selected],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=10,
                    color="#FFFFFF",
                    symbol="circle",
                ),
                name="D<sub>min</sub>",
                customdata=[drag_interior_selected],
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub>: 1 <br>D: %{customdata}<extra></extra>",
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
                line=dict(width=3, color="rgb(232,158,184)"),
                showlegend=False,
                name="D<sub>min</sub>",
            ),
            go.Scatter(
                x=[velocity_interior_selected],
                y=[h_selected / 1e3],
                mode="markers+text",
                marker=dict(size=10, color="#FFFFFF"),
                name="D<sub>min</sub>",
                showlegend=False,
            ),
        ],
        cols=2,
        rows=1,
    )

    fig_interior_optimum.update_xaxes(
        title_text=r"$C_L\:(\text{-})$",
        range=[xy_lowerbound, active_selection["CLmax_ld"] + 0.05],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=1,
    )
    fig_interior_optimum.update_yaxes(
        title_text=r"$\delta_T \:(\text{-})$",
        range=[xy_lowerbound, 1 + 0.05],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=1,
    )

    # Second subplot: V vs h
    fig_interior_optimum.update_xaxes(
        title_text=r"$V \: \text{(m/s)}$",
        range=[
            xy_lowerbound,
            max(atmos.a(0), np.nanmax(velocity_interior_harray)) + 15,
        ],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=2,
    )
    fig_interior_optimum.update_yaxes(
        title_text=r"$h \: 	\text{(km)}$",
        range=[xy_lowerbound, 20],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=2,
    )

    fig_interior_optimum.update_layout(
        title={
            "text": f"Interior minimum drag for {active_selection.full_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        }
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

    The value of the objective function, minimum drag, is calculated in a straightforward way as:

    $$
    D_{min}^* =  \frac{1}{2}\rho V_s^2 S C_{D_s} = \frac{W}{E_s}
    $$

    This is a higher value than the unconstrained one, and therefore operating in this scenario should be avoided if minimum drag is a goal.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    This concludes the analysis for the minimum drag of a simplified jet aircraft in the lift-limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = C_{L_\mathrm{max}}}, \quad \boxed{\delta_T^*=\frac{W}{E_S T_{a0}\sigma^\beta}}, \quad \text{for} \quad C_{L}^* < \sqrt{\frac{C_{D_0}}{K}} \quad \text{and} \quad \frac{W}{\sigma^\beta} \lt T_{a0}E_{S}
    $$

    With the optimal value for minimum drag: 

    $$
    D_{\mathrm{min}}^* = \frac{W}{E_S}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.
    """
    )
    return


@app.cell
def _(atmos, np):
    def maxlift_condition(W, h, E_S, beta, Ta0, CLmax, CD0, K):
        sigma = atmos.rhoratio(h)
        condition = ((W / (sigma**beta)) < (Ta0 * E_S)) & (
            CLmax < np.sqrt(3 * CD0 / K)
        )
        return condition
    return (maxlift_condition,)


@app.cell
def _(
    CD0,
    CL_E,
    CLmax,
    E_S,
    E_max,
    K,
    S,
    Ta0,
    W_selected,
    atmos,
    beta,
    h_array,
    idx_selected,
    maxlift_condition,
    np,
    velocity,
):
    maxlift_mask = maxlift_condition(
        W_selected, h_array, E_S, beta, Ta0, CLmax, CD0, K
    )

    CLopt_maxlift = np.where(maxlift_mask, CL_E, np.nan)

    velocity_maxlift_harray = velocity(
        W_selected, h_array, CLopt_maxlift, S, cap=False
    )

    drag_maxlift_harray = np.where(maxlift_mask, W_selected / E_max, np.nan)

    dTopt_maxlift = drag_maxlift_harray / Ta0 / (atmos.rhoratio(h_array) ** beta)

    CLopt_maxlift_selected = CLopt_maxlift[idx_selected]
    dTopt_maxlift_selected = dTopt_maxlift[idx_selected]

    velocity_maxlift_selected = velocity_maxlift_harray[idx_selected]

    drag_maxlift_selected = drag_maxlift_harray[idx_selected]
    return (
        CLopt_maxlift_selected,
        dTopt_maxlift_selected,
        drag_maxlift_selected,
        velocity_maxlift_harray,
        velocity_maxlift_selected,
    )


@app.cell
def _(
    CL_array,
    CLopt_maxlift_selected,
    a_harray,
    active_selection,
    atmos,
    constraint,
    dT_array,
    dTopt_maxlift_selected,
    drag_maxlift_selected,
    drag_surface,
    go,
    h_array,
    h_selected,
    make_subplots,
    max_colorbar,
    min_colorbar,
    mo,
    np,
    velocity_maxlift_harray,
    velocity_maxlift_selected,
    velocity_stall_harray,
    xy_lowerbound,
):
    fig_maxlift_optimum = make_subplots(
        rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}]]
    )

    # Traces on the 3D plot, first four are template
    fig_maxlift_optimum.add_traces(
        [
            go.Heatmap(
                x=CL_array,
                y=dT_array,
                z=drag_surface,
                opacity=0.9,
                name="Drag",
                colorscale="viridis",
                zsmooth="best",
                zmin=min_colorbar,
                zmax=max_colorbar,
                colorbar={"title": "Drag (N)"},
            ),
            go.Scatter(
                x=CL_array,
                y=constraint,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
                name="g1 constraint",
            ),
            go.Scatter(
                x=[CL_array[-15]],
                y=[constraint[-15] + 0.07],
                textposition="middle left",
                mode="markers+text",
                text=["g<sub>1</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="g1 constraint",
                textfont=dict(size=14, family="Arial"),
            ),
            go.Scatter(
                x=[CLopt_maxlift_selected],
                y=[dTopt_maxlift_selected],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=10,
                    color="#FFFFFF",
                    symbol="circle",
                ),
                name="D<sub>min</sub>",
                customdata=[drag_maxlift_selected],
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub>: 1 <br>D: %{customdata}<extra></extra>",
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
                line=dict(width=3, color="rgb(232,158,184)"),
                showlegend=False,
                name="D<sub>min</sub>",
            ),
            go.Scatter(
                x=[velocity_maxlift_selected],
                y=[h_selected / 1e3],
                mode="markers+text",
                marker=dict(size=10, color="#FFFFFF"),
                name="D<sub>min</sub>",
                showlegend=False,
            ),
        ],
        cols=2,
        rows=1,
    )
    fig_maxlift_optimum.update_xaxes(
        title_text=r"$C_L\:(\text{-})$",
        range=[xy_lowerbound, active_selection["CLmax_ld"] + 0.05],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=1,
    )
    fig_maxlift_optimum.update_yaxes(
        title_text=r"$\delta_T \:(\text{-})$",
        range=[xy_lowerbound, 1 + 0.05],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=1,
    )

    # Second subplot: V vs h
    fig_maxlift_optimum.update_xaxes(
        title_text=r"$V \: \text{(m/s)}$",
        range=[
            xy_lowerbound,
            max(atmos.a(0), np.nanmax(velocity_maxlift_harray)) + 15,
        ],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=2,
    )
    fig_maxlift_optimum.update_yaxes(
        title_text=r"$h \: 	\text{(km)}$",
        range=[xy_lowerbound, 20],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=2,
    )

    fig_maxlift_optimum.update_layout(
        title={
            "text": f"Lift-limited minimum drag for {active_selection.full_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        }
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


@app.cell
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The operational condition is found from (3), with $\delta_T = 1$, obtaining:

    $$
    \frac{W}{\sigma^\beta} = T_{a0}E_{\mathrm{max}}
    $$

    with:

    $$
    C_D^* = 2C_{D_0}, \quad V^* = \sqrt{\frac{W}{S}\frac{2}{\rho}\frac{1}{C_L^*}}, \quad \delta_T^*=1, \quad \frac{W}{\sigma^\beta} = T_{a0}E_{\mathrm{max}}
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    This concludes the analysis for the minimum drag of a simplified jet aircraft in the thrust-limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = \sqrt{\frac{C_{D_0}}{K}}}, \quad \boxed{\delta_T^*=1}, \quad \text{for} \quad C_{L}^* < C_{L_\mathrm{max}} \quad \text{and} \quad \frac{W}{\sigma^\beta} = T_{a0}E_\mathrm{max}
    $$

    With the following value for the objective function: 

    $$
    D_{\mathrm{min}}^* = \frac{W}{E_\mathrm{max}}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.
    """
    )
    return


@app.cell
def _(atmos, np):
    def maxthrust_altitude(W, beta, Ta0, S, CD0, K, CLmax, E_max):
        sigma_exp = (W) / Ta0 / E_max

        sigma = sigma_exp ** (1 / (beta))

        h = atmos.altitude(sigma)
        return np.where(
            ((h > 0) & (h < 20e3) & ((np.sqrt(CD0 / K)) < CLmax)), h, np.nan
        )
    return (maxthrust_altitude,)


@app.cell
def _(
    CD0,
    CL_array,
    CLmax,
    E_max,
    K,
    S,
    Ta0,
    W_selected,
    beta,
    drag,
    horizontal_constraint,
    maxthrust_altitude,
    np,
    velocity,
):
    maxthrust_h = maxthrust_altitude(W_selected, beta, Ta0, S, CD0, K, CLmax, E_max)

    CLopt_maxthrust_selected = np.sqrt(CD0 / K)

    velocity_maxthrust_selected = velocity(
        W_selected, maxthrust_h, CLopt_maxthrust_selected, S, cap=False
    )

    velocity_CLarray_maxthrust_h = velocity(
        W_selected, maxthrust_h, CL_array, S, cap=False
    )

    dTopt_maxthrust_selected = 1

    drag_maxthrust_h_curve = drag(
        maxthrust_h, S, CD0, K, CL_array, velocity_CLarray_maxthrust_h
    )

    constraint_maxthrust = horizontal_constraint(
        W_selected, maxthrust_h, CD0, K, CL_array, Ta0, beta, type="jet"
    )

    drag_maxthrust_surface = np.tile(drag_maxthrust_h_curve, (len(CL_array), 1))

    drag_maxthrust_selected = drag(
        maxthrust_h,
        S,
        CD0,
        K,
        CLopt_maxthrust_selected,
        velocity_maxthrust_selected,
    )

    if np.all(np.isnan(drag_maxthrust_surface)):
        drag_maxthrust_surface[0, 0] = 1e-10
        CLopt_maxthrust_selected = np.nan
    return (
        CLopt_maxthrust_selected,
        constraint_maxthrust,
        dTopt_maxthrust_selected,
        drag_maxthrust_selected,
        drag_maxthrust_surface,
        maxthrust_h,
        velocity_maxthrust_selected,
    )


@app.cell
def _(
    CL_array,
    CLopt_maxthrust_selected,
    a_harray,
    active_selection,
    atmos,
    constraint_maxthrust,
    dT_array,
    dTopt_maxthrust_selected,
    drag_maxthrust_selected,
    drag_maxthrust_surface,
    go,
    h_array,
    make_subplots,
    maxthrust_h,
    mo,
    np,
    velocity_maxthrust_selected,
    velocity_stall_harray,
    xy_lowerbound,
):
    fig_maxthrust_optimum = make_subplots(
        rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}]]
    )

    # Traces on the 3D plot
    fig_maxthrust_optimum.add_traces(
        [
            go.Heatmap(
                x=CL_array,
                y=dT_array,
                z=drag_maxthrust_surface,
                opacity=0.9,
                name="Drag",
                colorscale="viridis",
                zsmooth="best",
                zmin=np.nanmin(drag_maxthrust_surface),
                zmax=np.nanmin(drag_maxthrust_surface) * 3,
                colorbar={"title": "Drag (N)"},
            ),
            go.Scatter(
                x=CL_array,
                y=constraint_maxthrust,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
                name="g1 constraint",
            ),
            go.Scatter(
                x=[CL_array[-15]],
                y=[constraint_maxthrust[-15] + 0.07],
                textposition="middle left",
                mode="markers+text",
                text=["g<sub>1</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="g1 constraint",
                textfont=dict(size=14, family="Arial"),
            ),
            go.Scatter(
                x=[CLopt_maxthrust_selected],
                y=[dTopt_maxthrust_selected],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=10,
                    color="#FFFFFF",
                    symbol="circle",
                ),
                name="...<sub>min</sub>",
                customdata=[drag_maxthrust_selected],
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub>: 1 <br>D: %{customdata}<extra></extra>",
            ),
            go.Scatter(  # dummy point to avoid plotly plotting a weirdly shaped box
                x=[0],
                y=[0],
                opacity=0.7,
                mode="markers",
                showlegend=False,
                line=dict(color="rgba(0, 0, 0, 0.0)", width=10),
            ),
        ],
        cols=1,
        rows=1,
    )

    # Traces on the flight envelope
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
                name="D<sub>min</sub>",
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
            "text": f"Thrust-limited minimum drag for {active_selection.full_name}",
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


@app.cell
def _(mo):
    mo.md(
        r"""
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

    The same considerations hold for the case of the lift-limited analysis, with the only difference that now $\delta_T^* = 1$
    In fact, once again, the aircraft would have to stall at a higher speed than the one for minimum drag. Continuing with primal feasibility condition (3), obtain the operational condition: 


    $$
    \frac{W}{\sigma^\beta} = T_{a0}E_S
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    This concludes the analysis for the minimum drag of a simplified jet aircraft in the thrust-lift limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = C_{L_\mathrm{max}}}, \quad \boxed{\delta_T^*=1}, \quad \text{for} \quad C_{L}^* \lt \sqrt{\frac{C_{D_0}}{K}} \quad \text{and} \quad \frac{W}{\sigma^\beta} = T_{a0}E_S
    $$

    With the following value for the objective function: 

    $$
    D_{\mathrm{min}}^* = \frac{W}{E_\mathrm{S}}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.
    """
    )
    return


@app.cell
def _(atmos, np):
    def maxlift_thrust_altitude(W, beta, Ta0, S, CD0, K, CLmax, E_S):
        sigma_exp = (W) / Ta0 / E_S

        sigma = sigma_exp ** (1 / (beta))

        h = atmos.altitude(sigma)
        return np.where(
            ((h > 0) & (h < 20e3) & ((np.sqrt(CD0 / K)) > CLmax)), h, np.nan
        )
    return (maxlift_thrust_altitude,)


@app.cell
def _(
    CD0,
    CL_array,
    CLmax,
    E_S,
    K,
    S,
    Ta0,
    W_selected,
    beta,
    drag,
    horizontal_constraint,
    maxlift_thrust_altitude,
    np,
    velocity,
):
    maxlift_thrust_h = maxlift_thrust_altitude(
        W_selected, beta, Ta0, S, CD0, K, CLmax, E_S
    )

    CLopt_maxlift_thrust_selected = np.sqrt(CD0 / K)

    velocity_maxlift_thrust_selected = velocity(
        W_selected, maxlift_thrust_h, CLopt_maxlift_thrust_selected, S, cap=False
    )

    velocity_CLarray_maxlift_thrust_h = velocity(
        W_selected, maxlift_thrust_h, CL_array, S, cap=False
    )

    dTopt_maxlift_thrust_selected = 1

    drag_maxlift_thrust_h_curve = drag(
        maxlift_thrust_h, S, CD0, K, CL_array, velocity_CLarray_maxlift_thrust_h
    )

    constraint_maxlift_thrust = horizontal_constraint(
        W_selected, maxlift_thrust_h, CD0, K, CL_array, Ta0, beta, type="jet"
    )

    drag_maxlift_thrust_surface = np.tile(
        drag_maxlift_thrust_h_curve, (len(CL_array), 1)
    )

    drag_maxlift_thrust_selected = drag(
        maxlift_thrust_h,
        S,
        CD0,
        K,
        CLopt_maxlift_thrust_selected,
        velocity_maxlift_thrust_selected,
    )

    if np.all(np.isnan(drag_maxlift_thrust_surface)):
        drag_maxlift_thrust_surface[0, 0] = 1e-10
        CLopt_maxlift_thrust_selected = np.nan
    return (
        CLopt_maxlift_thrust_selected,
        constraint_maxlift_thrust,
        dTopt_maxlift_thrust_selected,
        drag_maxlift_thrust_selected,
        drag_maxlift_thrust_surface,
        maxlift_thrust_h,
        velocity_maxlift_thrust_selected,
    )


@app.cell
def _(
    CL_array,
    CLopt_maxlift_thrust_selected,
    a_harray,
    active_selection,
    atmos,
    constraint_maxlift_thrust,
    dT_array,
    dTopt_maxlift_thrust_selected,
    drag_maxlift_thrust_selected,
    drag_maxlift_thrust_surface,
    go,
    h_array,
    make_subplots,
    maxlift_thrust_h,
    mo,
    np,
    velocity_maxlift_thrust_selected,
    velocity_stall_harray,
    xy_lowerbound,
):
    fig_maxlift_thrust_optimum = make_subplots(
        rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}]]
    )

    # Traces on the 3D plot
    fig_maxlift_thrust_optimum.add_traces(
        [
            go.Heatmap(
                x=CL_array,
                y=dT_array,
                z=drag_maxlift_thrust_surface,
                opacity=0.9,
                name="drag",
                colorscale="viridis",
                zsmooth="best",
                zmin=np.nanmin(drag_maxlift_thrust_surface),
                zmax=np.nanmin(drag_maxlift_thrust_surface) * 3,
                colorbar={"title": "Drag (N)"},
            ),
            go.Scatter(
                x=CL_array,
                y=constraint_maxlift_thrust,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
                name="g1 constraint",
            ),
            go.Scatter(
                x=[CL_array[-15]],
                y=[constraint_maxlift_thrust[-15] + 0.07],
                textposition="middle left",
                mode="markers+text",
                text=["g<sub>1</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="g1 constraint",
                textfont=dict(size=14, family="Arial"),
            ),
            go.Scatter(
                x=[CLopt_maxlift_thrust_selected],
                y=[dTopt_maxlift_thrust_selected],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=10,
                    color="#FFFFFF",
                    symbol="circle",
                ),
                name="D<sub>min</sub>",
                customdata=[drag_maxlift_thrust_selected],
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub>: 1 <br>D: %{customdata}<extra></extra>",
            ),
            go.Scatter(  # dummy point to avoid plotly plotting a weirdly shaped box
                x=[0],
                y=[0],
                opacity=0.7,
                mode="markers",
                showlegend=False,
                line=dict(color="rgba(0, 0, 0, 0.0)", width=10),
            ),
        ],
        cols=1,
        rows=1,
    )

    # Traces on the flight envelope
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
                name="D<sub>min</sub>",
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
        range=[
            xy_lowerbound,
            max(atmos.a(0), velocity_maxlift_thrust_selected) + 15,
        ],
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
            "text": f"Thrust-lift limited minimum drag for {active_selection.full_name}",
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
    mo.md(r"""## Final flight envelope""")
    return


@app.cell
def _(mo):
    mo.md(r"""Now after deriving all the optima for each condition we can summarize the flight envelopes in one graph, as shown below. Experiment with the weight of the aircrarft to understand how the theoretical ceiling for minimum power moves in the graph.""")
    return


@app.cell
def _(
    h_array,
    maxlift_thrust_h,
    maxthrust_h,
    np,
    velocity_interior_harray,
    velocity_maxlift_harray,
    velocity_maxlift_thrust_selected,
    velocity_maxthrust_selected,
):
    V = np.where(
        np.isnan(velocity_interior_harray),
        velocity_maxlift_harray,
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

    final_envelope_h = np.append(h_sorted, [maxthrust_h, maxlift_thrust_h])
    final_envelope_velocity = np.append(
        V_sorted, [velocity_maxthrust_selected, velocity_maxlift_thrust_selected]
    )
    return (final_envelope_velocity,)


@app.cell
def _(
    a_harray,
    active_selection,
    atmos,
    final_envelope_velocity,
    go,
    h_array,
    maxlift_thrust_h,
    maxthrust_h,
    mo,
    np,
    velocity_maxlift_thrust_selected,
    velocity_maxthrust_selected,
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
                y=h_array / 1e3,
                mode="lines",
                line=dict(width=3, color="rgb(232,158,184)"),
                showlegend=False,
                name="D<sub>min</sub>",
            ),
            go.Scatter(
                x=[velocity_maxthrust_selected],
                y=[maxthrust_h / 1e3],
                mode="markers",
                marker=dict(size=10, color="rgb(232,158,184)"),
                showlegend=False,
                name="D<sub>min</sub>",
            ),
            go.Scatter(
                x=[velocity_maxlift_thrust_selected],
                y=[maxlift_thrust_h / 1e3],
                mode="markers",
                marker=dict(size=10, color="rgb(232,158,184)"),
                showlegend=False,
                name="D<sub>min</sub>",
            ),
        ],
    )

    fig_final_flightenv.update_layout(
        xaxis=dict(
            title=r"$V \: \text{(m/s)}$",
            range=[
                xy_lowerbound,
                max(atmos.a(0), np.nanmax(final_envelope_velocity)) + 15,
            ],
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
            "text": f"Flight envelope for minimum drag for {active_selection.full_name}",
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


@app.cell
def _(mo):
    mo.md(r"""## Summary""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    | Name | Condition | $C_L^*$ | $\delta_T^*$ | $D^*$ |
    |:-|:-------|:-------:|:------------:|:-------|
    |Interior-optima    | $\displaystyle \frac{W}{\sigma^\beta} < T_{a0} E_\mathrm{max} \quad \text{and} \quad C_L^* \lt C_{L_\mathrm{max}}$ | $\sqrt{\frac{C_{D_0}}{K}}$ | $\displaystyle \frac{2W}{T_{a0}\sigma^\beta}\sqrt{C_{D_0}K}$ | $\displaystyle 2W\sqrt{KC_{D_0}}=\frac{W}{E_\mathrm{max}}$ |
    |Lift-limited    |  $\displaystyle \frac{W}{\sigma^\beta} < T_{a0} E_\mathrm{S} \quad \text{and} \quad C_L^* \lt \sqrt{\frac{C_{D_0}}{K}}$ | $C_{L_\mathrm{max}}$ | $\displaystyle \frac{W}{E_S T_{a0} \sigma^\beta}$ | $\displaystyle \frac{W}{E_S}$|
    |Thrust-limited    | $\displaystyle \frac{W}{\sigma^\beta} = T_{a0} E_\mathrm{max} \quad \text{and} \quad C_L^* \lt C_{L_\mathrm{max}}$ | $\displaystyle \sqrt{\frac{C_{D_0}}{K}}$ | $1$ | $\displaystyle 2W\sqrt{KC_{D_0}}=\frac{W}{E_{\mathrm{max}}}$ |
    |Thrust-lift limited    |  $\displaystyle \frac{W}{\sigma^\beta} = T_{a0} E_\mathrm{S} \quad \text{and} \quad C_L^* \lt \sqrt{\frac{C_{D_0}}{K}}$ | $C_{L_\mathrm{max}}$ | $1$ | $\displaystyle \frac{W}{E_S}$|
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
