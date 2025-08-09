import marimo

__generated_with = "0.14.15"
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

    # Set local/online filepath
    _defaults.FILEURL = _defaults.get_url()

    # Plotly dark mode template
    _defaults.set_plotly_template()

    # Set navbar on the right
    _defaults.set_sidebar()

    # Data directory
    data_dir = str(mo.notebook_location() / "public" / "AircraftDB_Standard.csv")
    return ac, atmos, data_dir, go, make_subplots, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Minimum airspeed: simplfied jet aircraft

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
        & \quad T_a(V,h) = T_a(h) = T_{a0}\sigma^\beta \\
    \end{aligned}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(ac, data_dir, mo):
    # Database cell (1)

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
        variables_stack,
    )


@app.cell
def _(active_selection, atmos, np):
    # Functions definition for computation Figure (1)

    # Compute velocity as a function of C_L
    def velocity(C_L, W, h):
        S = active_selection["S"]
        numerator = 2 * W  # scalar or array
        denominator = atmos.rho(h) * S * C_L
        vel = np.sqrt(
            np.divide(
                numerator,
                denominator,
                out=np.zeros_like(denominator),
                where=C_L != 0,
            )
        )

        return np.where(vel > atmos.a(h), np.nan, vel)


    def c2_eq(C_L, W, h):
        S = active_selection["S"]
        CD0 = active_selection["CD0"]
        K = active_selection["K"]
        Ta0 = active_selection["Ta0"]
        beta = active_selection["beta"]

        # Sigma ratio from rhoratio
        sigma = atmos.rhoratio(h)

        return np.divide(
            W * (CD0 + K * C_L**2) / (Ta0 * 10**3 * sigma**beta),
            C_L,
            out=np.zeros_like(C_L),
            where=C_L != 0,
        )
    return c2_eq, velocity


@app.cell
def _(active_selection, atmos, h_slider, m_slider, np):
    # Variables declared
    meshgrid_n = 100

    C_Larray = np.linspace(0, active_selection["CLmax_ld"], meshgrid_n)
    dTarray = np.linspace(0, 1, meshgrid_n)

    # Retrieve selected values
    # Compute selected weight
    W_selected = (
        active_selection["OEM"]
        + (active_selection["MTOM"] - active_selection["OEM"]) * m_slider.value
    ) * atmos.g0  # Netwons

    h_selected = int(h_slider.value * 1e3)  # meters

    a = atmos.a(h_selected)
    return C_Larray, W_selected, a, dTarray, h_selected, meshgrid_n


@app.cell
def _(C_Larray, W_selected, a, c2_eq, h_selected, np, velocity):
    # Computation cell (1)

    # Calculate the c2_eq constraint curve
    c2_constraint = c2_eq(C_Larray, W_selected, h_selected)

    # Cut off due to the domain of dT
    c2_constraint = np.where(c2_constraint > 1.1, np.nan, c2_constraint)

    velocity_surface = np.tile(
        velocity(C_Larray, W_selected, h_selected), (len(C_Larray), 1)
    )

    # Handle unrealistic values above Mach 1
    velocity_surface = np.where(velocity_surface > a, np.nan, velocity_surface)
    return c2_constraint, velocity_surface


@app.cell
def _(
    CL_slider,
    C_Larray,
    W_selected,
    a,
    active_selection,
    c2_constraint,
    dT_slider,
    dTarray,
    go,
    h_selected,
    mo,
    np,
    velocity,
    velocity_surface,
):
    # Figure cell (1.0)

    # Create go.Figure() object
    fig1 = go.Figure()

    xy_lowerbound = -0.1

    # Minimum velocity surface
    fig1.add_traces(
        [
            go.Surface(
                x=C_Larray,
                y=dTarray,
                z=velocity_surface,
                opacity=0.9,
                name="V_min",
                colorscale="viridis",
            ),
            go.Scatter3d(
                x=C_Larray,
                y=c2_constraint,
                z=velocity_surface[0],
                opacity=0.45,
                mode="lines",
                showlegend=False,
                line=dict(color="red", width=10),
                name="c2_constraint",
            ),
            go.Scatter3d(
                x=[C_Larray[15]],
                y=[c2_constraint[15]],
                z=[velocity_surface[0, 15]],
                opacity=1,
                textposition="middle left",
                mode="markers+text",
                text=["c<sub>2</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="c2_constraint",
            ),
            go.Scatter3d(
                x=np.linspace(xy_lowerbound, active_selection["CLmax_ld"]),
                y=np.ones(len(dTarray)) * xy_lowerbound,
                z=np.ones(C_Larray.shape) * a,
                mode="lines",
                showlegend=False,
                line=dict(color="orange", width=8, dash="dash"),
                name="M1.0",
                text=[
                    f"M1.0 V = {round(V, 2)} (m/s)"
                    for V in np.ones(C_Larray.shape) * a
                ],
                hoverinfo="text",
            ),
            go.Scatter3d(
                x=[np.linspace(xy_lowerbound, active_selection["CLmax_ld"])[-20]],
                y=[(np.ones(len(dTarray)) * xy_lowerbound)[-20]],
                z=[(np.ones(C_Larray.shape) * a)[-20]],
                mode="markers+text",
                text=["M1.0"],
                textposition="top center",
                showlegend=False,
                marker=dict(size=1, color="rgba(0, 0, 0, 0.0)"),
                name="",
                hoverinfo="skip",
            ),
            go.Scatter3d(
                x=np.ones(len(C_Larray)) * xy_lowerbound,
                y=np.linspace(xy_lowerbound, 1),
                z=np.ones(C_Larray.shape) * a,
                mode="lines",
                showlegend=False,
                line=dict(color="orange", width=8, dash="dash"),
                name="M1.0",
                text=[
                    f"M1.0 V = {round(V, 2)} (m/s)"
                    for V in np.ones(C_Larray.shape) * a
                ],
                hoverinfo="text",
            ),
            go.Scatter3d(
                x=[CL_slider.value],
                y=[dT_slider.value],
                z=[
                    velocity(CL_slider.value, W_selected, h_selected) + 5
                ],  # Slightly elevate to show the full marker
                mode="markers",
                showlegend=False,
                marker=dict(size=5, color="cyan"),
                name="design_point",
                hovertemplate="x: %{x}<br>y: %{y}<extra>%{fullData.name}</extra>",
            ),
        ]
    )

    fig1.update_layout(
        scene=dict(
            xaxis=dict(
                title="C<sub>L</sub> (-)",
                range=[xy_lowerbound, active_selection["CLmax_ld"]],
            ),
            yaxis=dict(title="δ<sub>T</sub> (-)", range=[xy_lowerbound, 1]),
            zaxis=dict(title="V (m/s)", range=[0, a + 15]),
        ),
        title_text=active_selection["full_name"],
        title_x=0.5,
    )

    mo.output.clear()
    return fig1, xy_lowerbound


@app.cell(hide_code=True)
def _(CL_slider, dT_slider, mo):
    mo.md(
        f"""Here you can modify the control variables to understand how it affects the design: {mo.hstack([dT_slider, CL_slider])}"""
    )
    return


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell(hide_code=True)
def _(fig1):
    fig1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## KKT formulation
    To be reconducted in the standard KKT analysis format, the objective function is expressed in terms of the controls by direct elimination of $c_1^\mathrm{eq}$.
    Also, minimizing $V$ is equivalent to minimizing $V^2$, because the square power function is monotonically increasing.
    Therefore, to simplify the calculations, the problem is rewritten as follows:
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
        & \quad V^2 = \frac{2W}{\rho S C_L} \\
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
    mo.md(
        r"""
    ### Lagrangian function and KKT conditions

    The Lagrangian function combines the objective function with eqaulity constraints using Lagrange multipliers ($\lambda_i$) and inequality constraints using KKT multipliers ($\mu_j$).

    $$
    \begin{aligned}
    \mathcal{L}(C_L, \delta_T, \lambda_1, \mu_1, \mu_2, \mu_3, \mu_4) = 
    \quad \frac{2W}{\rho S C_L}
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
    A necessary condition for an optimal solution of the optimization problem $(C_L^*, \delta_T^*)$ to exist, the multipliers $\lambda_1, \mu_1, \mu_2, \mu_3, \mu_4$ have to meet the following conditions:

    **A. Stationarity ($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \frac{\partial \mathcal{L}}{\partial C_L} = -\frac{2W}{\rho S C_L^2} + \lambda_1 \left(\frac{C_{D_0}- KC_L^2}{C_L^2}\right) + \mu_1 - \mu_2 = 0$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1 \frac{T_{a0}\sigma^\beta}{W} + \mu_3 - \mu_4 = 0$
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
    ## KKT analysis

    We can now proceed to systematically examine the conditions where various inequality constraints are active or inactive.

    ### _Interior solutions_ 

    Assuming that that $0 < C_L < C_{L_\mathrm{max}}$ and $0 < \delta_T < 1$ is equivalent to consider all inequality constraints as inactive.

    Therefore: $\mu_1,\mu_2,\mu_3,\mu_4=0$. 

    From stationarity condition (2): $\lambda_1 = 0$.

    It can now be seen that stationarity condition (1) is never verified.

    It can be concluded that the minimum speed cannot be achieved in the interior of the domain. 
    The minimum must lie on at least one of the boundaries defined by $C_L = C_{L_\mathrm{max}}$ or $\delta_T = 1$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### _Lower boundary solutions_
    The case where $C_L=0$ and the case where $\delta_T=0$ can be immediately discaded because of the primal feasibility conditions.
    This means that $\mu_2=\mu_4=0$ in all cases.

    We can then proceed with the analysis of the cases where the boundaries $C_L = C_{L_\mathrm{max}}$ and $\delta_T = 1$ are active in any of the three possible combinations.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### _Thrust-limited minimum airspeed_

    $\delta_T=1 \quad \Rightarrow \quad \mu_3 > 0$

    $C_L < C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 = 0$

    From stationarity condition (2): 

    $$
    \lambda_1 = -\mu_3\frac{W}{T_{a0}\sigma^\beta} \quad \Rightarrow \quad \lambda_1 < 0
    $$

    Stationarity condition (1) then becomes:

    $$
    \frac{2T_{a0}\sigma^\beta}{\rho S C_L^2} + \mu_3\left( \frac{C_{D_0}-KC_L^2}{C_L^2}\right) = 0
    \quad \text{and } \quad 
    \mu_3>0 
    \quad \Rightarrow \quad 
    C_L > \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$

    and implies that the thrust-limited minimum airspeed is obtained strictly on the left branch of the drag performance diagram, at a lift-coefficient strictly higher than the one for maximum aerodynamic efficiency.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The corresponding optimum value of the $C_L$ is obtained by solving the primal feasibiliy condition (3) and taking the highest of the two solutions:

    $$
    C_L^* = \frac{T_{a0}\sigma^\beta}{2KW} \left[1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}\right]
    $$

    It has still to be verified that $C_L^* < C_{L_\mathrm{max}}$, which depends on the numerical values of the design parameters, and on the current values of the weight and altitude.

    This optimum value of the lift-coefficient is achievable for 

    $$
    1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2 \ge 0
    \quad \Rightarrow \quad \frac{W}{\sigma^\beta} \le  T_{a0} E_\mathrm{max}
    $$

    The limit equality can be used to calculate the corresponding limit altitude at which the minimum speed is limited by thrust, for a given weight. This is called the _theoretcal ceiling_.

    The corresponding minimum airspeed is:

    $$
    V^* = 
    \sqrt{\frac{4KW^2/\rho S T_{a0}\sigma^\beta}{1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}}}
    = V_s \sqrt{\frac{2KWC_{L_\mathrm{max}}/T_{a0}\sigma^\beta}{1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}}}
    $$
    """
    )
    return


@app.cell
def _(ac_table):
    ac_table
    return


@app.cell
def _(atmos, np):
    # Functions definition for computation Figure (2)
    def CL_maxthrust(W, h, K, CD0, Ta0, beta, CLmax):
        E_max = np.sqrt(1 / (4 * K * CD0))

        sigma = atmos.rhoratio(h)

        C_Lopt = (
            Ta0
            * sigma**beta
            / (2 * K * W)
            * (1 + np.sqrt(1 - (W / (E_max * Ta0 * sigma**beta)) ** 2))
        )
        condition = (
            (C_Lopt < CLmax)
            & (C_Lopt > np.sqrt(CD0 / K))
            & ((W / (sigma**beta)) <= (Ta0 * E_max))
        )

        C_Lopt = np.where(condition, C_Lopt, np.nan)

        return C_Lopt
    return (CL_maxthrust,)


@app.cell
def _(
    CL_maxthrust,
    W_selected,
    active_selection,
    h_selected,
    meshgrid_n,
    np,
    velocity,
):
    # Computation cell (2)
    h_array = np.linspace(0, 20e3, meshgrid_n)

    C_Loptimal_maxthrust = CL_maxthrust(
        W_selected,
        h_array,
        active_selection["K"],
        active_selection["CD0"],
        active_selection["Ta0"] * 1e3,
        active_selection["beta"],
        active_selection["CLmax_ld"],
    )

    min_vel_maxthrust = velocity(C_Loptimal_maxthrust, W_selected, h_array)

    V_stall_maxthrust = velocity(
        active_selection["CLmax_ld"],
        W_selected,
        h_array[np.where(~np.isnan(min_vel_maxthrust))[0][0]],
    )

    CL_opt_selected = CL_maxthrust(
        W_selected,
        h_selected,
        active_selection["K"],
        active_selection["CD0"],
        active_selection["Ta0"] * 1e3,
        active_selection["beta"],
        active_selection["CLmax_ld"],
    )

    v_min_design = velocity(
        CL_opt_selected,
        W_selected,
        h_selected,
    )
    return (
        CL_opt_selected,
        C_Loptimal_maxthrust,
        V_stall_maxthrust,
        h_array,
        min_vel_maxthrust,
        v_min_design,
    )


@app.cell
def _(
    CL_opt_selected,
    C_Larray,
    C_Loptimal_maxthrust,
    V_stall_maxthrust,
    a,
    active_selection,
    atmos,
    c2_constraint,
    dTarray,
    go,
    h_array,
    h_selected,
    make_subplots,
    min_vel_maxthrust,
    mo,
    np,
    v_min_design,
    velocity_surface,
    xy_lowerbound,
):
    # Figure cell (2.0)

    # Create go.Figure() object
    fig2 = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]]
    )

    # Traces on the 3D plot
    fig2.add_traces(
        [
            go.Scatter3d(
                x=[CL_opt_selected, CL_opt_selected],
                y=[1, xy_lowerbound],
                z=[v_min_design, v_min_design],
                mode="lines",
                showlegend=False,
                line=dict(color="grey", width=2),
            ),
            go.Surface(
                x=C_Larray,
                y=dTarray,
                z=velocity_surface,
                opacity=0.9,
                name="V_min",
                colorscale="viridis",
            ),
            go.Scatter3d(
                x=C_Loptimal_maxthrust,
                y=np.ones(len(dTarray)) * xy_lowerbound,
                z=np.tile(min_vel_maxthrust, len(C_Loptimal_maxthrust)),
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(144, 238, 144, 1)", width=8),
            ),
            go.Scatter3d(
                x=C_Larray,
                y=c2_constraint,
                z=velocity_surface[0],
                opacity=0.45,
                mode="lines",
                showlegend=False,
                line=dict(color="red", width=10),
                name="c2_constraint",
            ),
            go.Scatter3d(
                x=[C_Larray[15]],
                y=[c2_constraint[15]],
                z=[velocity_surface[0, 15]],
                opacity=1,
                textposition="middle left",
                mode="markers+text",
                text=["c<sub>2</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="c2_constraint",
            ),
            go.Scatter3d(
                x=np.linspace(xy_lowerbound, active_selection["CLmax_ld"]),
                y=np.ones(len(dTarray)) * xy_lowerbound,
                z=np.ones(C_Larray.shape) * a,
                mode="lines",
                showlegend=False,
                line=dict(color="orange", width=8, dash="dash"),
                name="M1.0",
                text=[
                    f"M1.0 V = {round(V, 2)} (m/s)"
                    for V in np.ones(C_Larray.shape) * a
                ],
                hoverinfo="text",
            ),
            go.Scatter3d(
                x=[np.linspace(xy_lowerbound, active_selection["CLmax_ld"])[-20]],
                y=[(np.ones(len(dTarray)) * xy_lowerbound)[-20]],
                z=[(np.ones(C_Larray.shape) * a)[-20]],
                mode="markers+text",
                text=["M1.0"],
                textposition="top center",
                showlegend=False,
                marker=dict(size=1, color="rgba(0, 0, 0, 0.0)"),
                name="",
                hoverinfo="skip",
            ),
            go.Scatter3d(
                x=np.ones(len(C_Larray)) * xy_lowerbound,
                y=np.linspace(xy_lowerbound, 1),
                z=np.ones(C_Larray.shape) * a,
                mode="lines",
                showlegend=False,
                line=dict(color="orange", width=8, dash="dash"),
                name="M1.0",
                text=[
                    f"M1.0 V = {round(V, 2)} (m/s)"
                    for V in np.ones(C_Larray.shape) * a
                ],
                hoverinfo="text",
            ),
            go.Scatter3d(
                x=[CL_opt_selected],
                y=[1],
                z=[v_min_design],  # Slightly elevate to show the full marker
                mode="markers",
                showlegend=False,
                marker=dict(size=5, color="cyan"),
                name="design_point",
                hovertemplate="x: %{x}<br>y: %{y}<extra>%{fullData.name}</extra>",
            ),
        ],
        cols=1,
        rows=1,
    )

    # Traces on the flight envelope
    fig2.add_traces(
        [
            go.Scatter(
                x=min_vel_maxthrust,
                y=h_array / 1e3,
                mode="lines",
                line_color="rgba(144, 238, 144, 1)",
                line=dict(width=3),
                showlegend=False,
                name="V_min",
            ),
            go.Scatter(
                x=[
                    min_vel_maxthrust[np.where(~np.isnan(min_vel_maxthrust))[0][10]]
                ],
                y=[
                    h_array[np.where(~np.isnan(min_vel_maxthrust))[0][10]] / 1e3
                    + 0.5
                ],
                mode="markers+text",
                text=[
                    "V<sub>min</sub>"
                ],  # , δ<sub>T</sub> = 1, C<sub>L</sub> < C<sub>L<sub>max</sub></sub>
                hoverinfo="skip",
                marker=dict(size=1, color="rgba(0, 0, 0, 0.0)"),
                name="",
                showlegend=False,
                textposition="top left",
            ),
            go.Scatter(
                x=atmos.a(h_array),
                y=h_array / 1e3,
                showlegend=False,
                mode="lines",
                line=dict(color="orange", width=2, dash="dash"),
                name="M1.0",
            ),
            go.Scatter(
                x=[atmos.a(h_array[-8]) - 5],
                y=[h_array[-8] / 1e3],
                mode="markers+text",
                text=["M1.0"],
                hoverinfo="skip",
                marker=dict(size=1, color="rgba(0, 0, 0, 0.0)"),
                name="",
                showlegend=False,
                textposition="top left",
            ),
            go.Scatter(
                x=[V_stall_maxthrust],
                y=[h_array[np.where(~np.isnan(min_vel_maxthrust))[0][0]] / 1e3],
                mode="markers+text",
                marker=dict(size=4, color="rgba(255, 0, 0, 1.0)"),
                text=["C<sub>L</sub> = C<sub>L<sub>max</sub></sub>"],
                showlegend=False,
                name="V_min",
                textposition="top left",
            ),
            go.Scatter(
                x=[v_min_design],
                y=[h_selected / 1e3],
                mode="markers+text",
                marker=dict(size=5, color="cyan"),
                showlegend=False,
            ),
        ],
        cols=2,
        rows=1,
    )

    fig2.update_layout(
        scene=dict(
            xaxis=dict(
                title="C<sub>L</sub> (-)",
                range=[xy_lowerbound, active_selection["CLmax_ld"]],
            ),
            yaxis=dict(title="δ<sub>T</sub> (-)", range=[xy_lowerbound, 1]),
            zaxis=dict(title="V (m/s)", range=[0, a + 15]),
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
    return (fig2,)


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell
def _(fig2):
    fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ###_Lift-limited minimum airspeed_

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 > 0$ 

    $0 < \delta_T < 1 \quad \Rightarrow \quad \mu_3 = 0$.

    From stationarity condition (2): $\lambda_1 = 0$.

    From stationarity condition (1): $\mu_1 = \frac{2W}{\rho S C_{L_\mathrm{max}}^2}>0$, which does not depend on the value of $\delta_T$, and is always verified.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The corresponding value of the throttle is calculated from the primal feasibility condition (3):

    $$
    \delta_T 
    = \frac{W}{T_{a0}\sigma^\beta} \frac{C_{D_0} + K C^2_{L_\mathrm{max}}}{C_{L_\mathrm{max}}} 
    = \frac{W}{T_{a0}\sigma^\beta} \frac{1}{E_S} 
    $$

    This is valid only if the calculated $\delta_T$ is strictly lower than the maximum, which corresponds to:

    $$
    \frac{W}{\sigma^\beta} < T_{a0} E_S
    $$

    The limit equality can be used to calculate the corresponding limit altitude at which the minimum speed is limited by lift, for a given weight.

    The corresponding minimum airspeed is called the _stall speed_.

    $$
    V^* = \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}
    $$
    """
    )
    return


@app.cell
def _(atmos, np):
    def CL_liftlimited(W, h, K, CD0, Ta0, beta, CLmax):
        E_s = CLmax / (CD0 + K * CLmax**2)

        sigma = atmos.rhoratio(h)

        C_Lopt = CLmax
        condition = (W / (sigma**beta)) < (Ta0 * E_s)

        C_Lopt = np.where(condition, C_Lopt, np.nan)

        return C_Lopt
    return (CL_liftlimited,)


@app.cell
def _(CL_liftlimited, W_selected, active_selection, h_array, velocity):
    # Computation cell (3)
    CL_optimal_liftlimited = CL_liftlimited(
        W_selected,
        h_array,
        active_selection["K"],
        active_selection["CD0"],
        active_selection["Ta0"] * 1e3,
        active_selection["beta"],
        active_selection["CLmax_ld"],
    )

    minvelocity_liftlim = velocity(CL_optimal_liftlimited, W_selected, h_array)
    return (minvelocity_liftlim,)


@app.cell
def _(
    CL_opt_selected,
    C_Larray,
    C_Loptimal_maxthrust,
    V_stall_maxthrust,
    a,
    active_selection,
    atmos,
    c2_constraint,
    dTarray,
    go,
    h_array,
    h_selected,
    make_subplots,
    minvelocity_liftlim,
    mo,
    np,
    v_min_design,
    velocity_surface,
    xy_lowerbound,
):
    # Figure cell (3.0)

    # Create go.Figure() object
    fig3 = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]]
    )

    # Traces on the 3D plot
    fig3.add_traces(
        [
            go.Scatter3d(
                x=[CL_opt_selected, CL_opt_selected],
                y=[1, xy_lowerbound],
                z=[v_min_design, v_min_design],
                mode="lines",
                showlegend=False,
                line=dict(color="grey", width=2),
            ),
            go.Surface(
                x=C_Larray,
                y=dTarray,
                z=velocity_surface,
                opacity=0.9,
                name="V_min",
                colorscale="viridis",
            ),
            go.Scatter3d(
                x=C_Loptimal_maxthrust,
                y=np.ones(len(dTarray)) * xy_lowerbound,
                z=np.tile(minvelocity_liftlim, len(C_Loptimal_maxthrust)),
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(144, 238, 144, 1)", width=8),
            ),
            go.Scatter3d(
                x=C_Larray,
                y=c2_constraint,
                z=velocity_surface[0],
                opacity=0.45,
                mode="lines",
                showlegend=False,
                line=dict(color="red", width=10),
                name="c2_constraint",
            ),
            go.Scatter3d(
                x=[C_Larray[15]],
                y=[c2_constraint[15]],
                z=[velocity_surface[0, 15]],
                opacity=1,
                textposition="middle left",
                mode="markers+text",
                text=["c<sub>2</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="c2_constraint",
            ),
            go.Scatter3d(
                x=np.linspace(xy_lowerbound, active_selection["CLmax_ld"]),
                y=np.ones(len(dTarray)) * xy_lowerbound,
                z=np.ones(C_Larray.shape) * a,
                mode="lines",
                showlegend=False,
                line=dict(color="orange", width=8, dash="dash"),
                name="M1.0",
                text=[
                    f"M1.0 V = {round(V, 2)} (m/s)"
                    for V in np.ones(C_Larray.shape) * a
                ],
                hoverinfo="text",
            ),
            go.Scatter3d(
                x=[np.linspace(xy_lowerbound, active_selection["CLmax_ld"])[-20]],
                y=[(np.ones(len(dTarray)) * xy_lowerbound)[-20]],
                z=[(np.ones(C_Larray.shape) * a)[-20]],
                mode="markers+text",
                text=["M1.0"],
                textposition="top center",
                showlegend=False,
                marker=dict(size=1, color="rgba(0, 0, 0, 0.0)"),
                name="",
                hoverinfo="skip",
            ),
            go.Scatter3d(
                x=np.ones(len(C_Larray)) * xy_lowerbound,
                y=np.linspace(xy_lowerbound, 1),
                z=np.ones(C_Larray.shape) * a,
                mode="lines",
                showlegend=False,
                line=dict(color="orange", width=8, dash="dash"),
                name="M1.0",
                text=[
                    f"M1.0 V = {round(V, 2)} (m/s)"
                    for V in np.ones(C_Larray.shape) * a
                ],
                hoverinfo="text",
            ),
            go.Scatter3d(
                x=[CL_opt_selected],
                y=[1],
                z=[v_min_design],  # Slightly elevate to show the full marker
                mode="markers",
                showlegend=False,
                marker=dict(size=5, color="cyan"),
                name="design_point",
                hovertemplate="x: %{x}<br>y: %{y}<extra>%{fullData.name}</extra>",
            ),
        ],
        cols=1,
        rows=1,
    )

    # Traces on the flight envelope
    fig3.add_traces(
        [
            go.Scatter(
                x=minvelocity_liftlim,
                y=h_array / 1e3,
                mode="lines",
                line_color="rgba(144, 238, 144, 1)",
                line=dict(width=3),
                showlegend=False,
                name="V_min",
            ),
            go.Scatter(
                x=[
                    minvelocity_liftlim[np.where(~np.isnan(minvelocity_liftlim))[0][10]]
                ],
                y=[
                    h_array[np.where(~np.isnan(minvelocity_liftlim))[0][10]] / 1e3
                    + 0.5
                ],
                mode="markers+text",
                text=[
                    "V<sub>min</sub>"
                ],  # , δ<sub>T</sub> = 1, C<sub>L</sub> < C<sub>L<sub>max</sub></sub>
                hoverinfo="skip",
                marker=dict(size=1, color="rgba(0, 0, 0, 0.0)"),
                name="",
                showlegend=False,
                textposition="top left",
            ),
            go.Scatter(
                x=atmos.a(h_array),
                y=h_array / 1e3,
                showlegend=False,
                mode="lines",
                line=dict(color="orange", width=2, dash="dash"),
                name="M1.0",
            ),
            go.Scatter(
                x=[atmos.a(h_array[-8]) - 5],
                y=[h_array[-8] / 1e3],
                mode="markers+text",
                text=["M1.0"],
                hoverinfo="skip",
                marker=dict(size=1, color="rgba(0, 0, 0, 0.0)"),
                name="",
                showlegend=False,
                textposition="top left",
            ),
            go.Scatter(
                x=[V_stall_maxthrust],
                y=[h_array[np.where(~np.isnan(minvelocity_liftlim))[0][-1]] / 1e3],
                mode="markers+text",
                marker=dict(size=4, color="rgba(255, 0, 0, 1.0)"),
                text=["C<sub>L</sub> = C<sub>L<sub>max</sub></sub>"],
                showlegend=False,
                name="V_min",
                textposition="top left",
            ),
            go.Scatter(
                x=[v_min_design],
                y=[h_selected / 1e3],
                mode="markers+text",
                marker=dict(size=5, color="cyan"),
                showlegend=False,
            ),
        ],
        cols=2,
        rows=1,
    )

    fig3.update_layout(
        scene=dict(
            xaxis=dict(
                title="C<sub>L</sub> (-)",
                range=[xy_lowerbound, active_selection["CLmax_ld"]],
            ),
            yaxis=dict(title="δ<sub>T</sub> (-)", range=[xy_lowerbound, 1]),
            zaxis=dict(title="V (m/s)", range=[0, a + 15]),
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
    return (fig3,)


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell
def _(fig3):
    fig3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### _Thrust- and lift-limited minimum speed_

    $\delta_T = 1 \quad \Rightarrow \quad \mu_3 > 0$

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 > 0$.

    From the stationary conditions (2):

    $$
    \lambda_1 = -\frac{\mu_3}{T_{a0}\sigma^\beta} \quad \Rightarrow \quad \lambda_1 < 0
    $$

    From stationary condition (1): 

    $$
    \mu_1 = \frac{2W}{\rho S C_{L_\mathrm{max}}^2} + \mu_3\frac{W}{T_{a0}\sigma^\beta}\left(\frac{C_{D_0} - K C_{L_\mathrm{max}}^2}{C_{L_\mathrm{max}}^2}\right) > 0 \quad \text{always}
    $$

    The primal feasibility equaiton (3) returns the expression of the condition where the minimum speed is limited by both thrust and lift capabilities of the aircraft.

    $$
    \frac{W}{\sigma^\beta} = T_{a0} E_S
    $$

    The corresponding value of the airspeed is once again

    $$
    V^* = \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Summary

    | Name | Condition | $C_L^*$ | $\delta_T^*$ | $V^*$ |
    |:-|:----------|:-------:|:------------:|:------|
    |Lift-limited    | $\displaystyle \frac{W}{\sigma^\beta} < T_{a0} E_S$ | $C_{L_\mathrm{max}}$ | $\displaystyle \frac{W}{T_{a0}\sigma^\beta} \frac{1}{E_S}$ | $\displaystyle V_s = \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}$ |
    |Thrust and Lift-limited    | $\displaystyle \frac{W}{\sigma^\beta} =  T_{a0} E_S$ | $C_{L_\mathrm{max}}$ | $1$ | $\displaystyle V_s =\sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}$ |
    |Thrust-limited    | $\displaystyle \frac{W}{\sigma^\beta} \le  T_{a0} E_\mathrm{max}$ | $\displaystyle \frac{T_{a0}\sigma^\beta}{2KW} \left[1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}\right]$ | $1$ | $\displaystyle V_s \sqrt{\frac{2KWC_{L_\mathrm{max}}/T_{a0}\sigma^\beta}{1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}}}$ |
    """
    )
    return


if __name__ == "__main__":
    app.run()
