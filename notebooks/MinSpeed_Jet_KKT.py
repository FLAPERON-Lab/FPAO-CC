import marimo

__generated_with = "0.14.15"
app = marimo.App(width="medium")


@app.cell
def _():
    # Initialization code that runs before all other cells
    import marimo as mo

    # Import dependencies
    from core import _defaults

    _defaults.FILEURL = _defaults.get_url()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import numpy as np
    from core import atmos
    from core import aircraft as ac

    # Plotly dark mode template
    _defaults.set_plotly_template()

    # Set navbar on the right
    _defaults.set_sidebar()
    return ac, atmos, go, make_subplots, mo, np


@app.cell
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
def _(ac, mo):
    # Database cell
    data = ac.available_aircrafts().round(decimals=4)

    cols_4dec = [
        "CD0",
        "K",
        "beta",
        "CLmax_cl",
        "CLmax_to",
        "CLmax_ld",
        "cT",
        "cP",
        "MMO",
    ]

    data[cols_4dec] = data[cols_4dec].round(4)

    other_cols = data.columns.difference(cols_4dec)
    data[other_cols] = data[other_cols].round(1)

    ac_table = mo.ui.table(
        data=data,
        pagination=True,
        freeze_columns_left=["full_name"],
        show_column_summaries=False,
        selection="single",
        initial_selection=[0],
        page_size=4,
    )

    ac_table
    return (ac_table,)


@app.cell(hide_code=True)
def _(ac, ac_table, atmos, mo, np):
    # Computation cell
    if ac_table.value is not None and ac_table.value.any().any():
        CL_maxld = float(ac_table.value.CLmax_ld.values[0])
    else:
        CL_maxld = 3

    CL_slider = mo.ui.slider(
        start=0, stop=CL_maxld, step=0.1, label=r"$C_L$", value=0.5
    )

    dT_slider = mo.ui.slider(start=0, stop=1, step=0.05, label=r"$\delta_T$", value=0.5)

    aircraft_list = []
    h = 0  # m

    rho = atmos.rho(0)
    sigma = atmos.rhoratio(h)

    if ac_table.value is not None and ac_table.value.any().any():
        aircraft_list = ac_table.value["ID"]

    fleet = {ID: ac.Aircraft(ac_ID=ID) for ID in aircraft_list}

    dTs = np.linspace(1e-4, 1, 300)

    for index, (id, obj) in enumerate(fleet.items()):
        # Compute the constraint line c2
        CLs = np.linspace(1e-4, CL_maxld, 300)
        cd0 = float(obj.ac_data.CD0.values[0])
        k = float(obj.ac_data.K.values[0])
        beta = float(obj.ac_data.beta.values[0])
        W = float(obj.ac_data.MTOM.values[0])
        Ta0 = float(obj.ac_data.Ta0.values[0])
        S = float(obj.ac_data.S.values[0])
        CLmax = float(obj.ac_data.CLmax_ld.values[0])
        c2_dT = W * (cd0 + k * CLs**2) / CLs / (Ta0 * 10**3 * sigma**beta)

        V = np.sqrt(2 * W / (rho * S * CLs))

        V = np.tile(V, (len(CLs), 1))

        V = np.where(V > 350, np.nan, V)

    V_func = lambda x: np.sqrt(2 * W / (rho * S * x))
    return CL_maxld, CL_slider, CLs, V, V_func, c2_dT, dT_slider, dTs


@app.cell(hide_code=True)
def _(
    CL_maxld,
    CL_slider,
    CLs,
    V,
    V_func,
    ac_table,
    c2_dT,
    dT_slider,
    dTs,
    go,
    make_subplots,
    mo,
    np,
):
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "scene"}]])

    # Title
    title_text = ""
    if ac_table.value is not None and ac_table.value.any().any():
        title_text = str(ac_table.value.full_name.values[0])

    # Left subplot (2D)
    fig.add_trace(
        go.Scatter(
            x=[float(CL_slider.value)],
            y=[float(dT_slider.value)],
            mode="markers",
            marker=dict(color="#EF553B", size=15, line=dict(width=4, color="Grey")),
            name="Design Point",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=CLs,
            y=c2_dT,
            mode="lines",
            name="c2^eq constraint",
            showlegend=False,
            line=dict(color="#AAFF00"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Contour(
            x=CLs,
            y=dTs,
            z=V,
            opacity=0.9,
            name="V_min contour",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Surface
    fig.add_trace(
        go.Surface(
            x=CLs,
            y=dTs,
            z=V,
            opacity=0.9,
            showscale=False,
            name="V_min",
            colorscale="cividis",
        ),
        row=1,
        col=2,
    )

    z_wall = np.stack([np.zeros_like(CLs), np.ones_like(CLs) * np.nanmax(V)])
    # # Surface
    fig.add_trace(
        go.Surface(
            x=np.tile(CLs, (2, 1)),
            y=np.tile(c2_dT, (2, 1)),
            z=z_wall,
            showscale=False,
            opacity=0.3,
            surfacecolor=np.ones_like(z_wall),  # constant color
            colorscale=[[0, "#AAFF00"], [1, "#AAFF00"]],
            name="c2^eq constraint",
        ),
        row=1,
        col=2,
    )

    # Slider marker
    fig.add_trace(
        go.Scatter3d(
            x=[float(CL_slider.value), float(CL_slider.value)],
            y=[float(dT_slider.value), float(dT_slider.value)],
            # z=[0, np.nanmax(V)],
            z=[V_func(float(CL_slider.value)) + 1],
            mode="markers",
            showlegend=False,
            # line=dict(color="#EF553B", width=5),
            name="Design Point",
            scene="scene2",
        ),
        row=1,
        col=2,
    )

    # Layout
    fig.update_layout(
        xaxis=dict(title="C<sub>L</sub> (-)", range=[-0.1, CL_maxld]),
        yaxis=dict(title="δ<sub>T</sub> (-)", range=[-0.1, 1]),
        scene=dict(
            xaxis=dict(title="C<sub>L</sub> (-)", range=[-0.1, CL_maxld]),
            yaxis=dict(title="δ<sub>T</sub> (-)", range=[-0.1, 1]),
            zaxis=dict(title="V (m/s)", range=[0, 350]),
        ),
        title_text=title_text,
        title_x=0.5,
    )
    mo.output.clear()
    return (fig,)


@app.cell
def _(CL_slider, ac_table, dT_slider, fig, mo):
    if ac_table.value is not None and ac_table.value.any().any():
        output = mo.vstack([fig, mo.hstack([CL_slider, dT_slider])])
    else:
        output = fig

    output
    return


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
def _(mo):
    mo.md(
        r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    8.  $\mu_1, \mu_2, \mu_3, \mu_4 \ge 0$
    """
    )
    return


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
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
