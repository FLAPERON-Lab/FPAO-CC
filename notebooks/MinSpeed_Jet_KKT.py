import marimo

__generated_with = "0.14.15"
app = marimo.App(width="medium")


@app.cell
def _():
    # Initialization code that runs before all other cells
    import marimo as mo

    # Import dependencies
    from core import _defaults
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
    return ac, atmos, data_dir, go, mo, np


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
    return CL_slider, active_selection, dT_slider


@app.cell
def _(active_selection, atmos, np):
    # Computation cell (1)
    meshgrid_n = 100

    C_Larray = np.linspace(0, active_selection["CLmax_ld"], meshgrid_n)
    dTarray = np.linspace(0, 1, meshgrid_n)


    # Compute velocity as a function of C_L
    def velocity(C_L):
        W = active_selection["MTOM"] * atmos.g0
        S = active_selection["S"]

        return np.sqrt(
            np.divide(
                2 * W,
                atmos.rho0 * S * C_L,
                out=np.zeros_like(C_L),
                where=C_L != 0,
            )
        )


    def c2_eq(C_L):
        W = active_selection["MTOM"] * atmos.g0
        S = active_selection["S"]
        CD0 = active_selection["CD0"]
        K = active_selection["K"]
        Ta0 = active_selection["Ta0"]
        beta = active_selection["beta"]

        # Sigma ratio from rhoratio
        sigma = atmos.rhoratio(0)

        return np.divide(
            W * (CD0 + K * C_L**2) / (Ta0 * 10**3 * sigma**beta),
            C_L,
            out=np.zeros_like(C_L),
            where=C_L != 0,
        )


    c2_constraint = c2_eq(C_Larray)

    # Cut off due to the domain of dT
    c2_constraint = np.where(c2_constraint > 1, np.nan, c2_constraint)

    velocity_surface = np.tile(velocity(C_Larray), (len(C_Larray), 1))

    # Handle unrealistic values of 350+ m/s, above Mach 1.
    velocity_surface = np.where(velocity_surface > 350, np.nan, velocity_surface)
    return C_Larray, c2_constraint, dTarray, velocity, velocity_surface


@app.cell
def _(
    CL_slider,
    C_Larray,
    c2_constraint,
    dT_slider,
    dTarray,
    go,
    mo,
    velocity,
    velocity_surface,
):
    # Figure cell (1.0)
    # Create go.Figure() object
    fig = go.Figure()

    # Design point
    fig.add_traces(
        go.Scatter3d(
            x=[CL_slider.value],
            y=[dT_slider.value],
            z=[velocity(CL_slider.value)],
            mode="markers",
            showlegend=False,
            marker=dict(size=5, color="red"),
        )
    )

    # Minimum velocity surface
    fig.add_traces(
        go.Surface(
            x=C_Larray,
            y=dTarray,
            z=velocity_surface,
            opacity=0.9,
            name="V_min",
            colorscale="Plotly3",
        ),
    )

    # c2_eq constraint curve
    fig.add_trace(
        go.Scatter3d(
            x=C_Larray,
            y=c2_constraint,
            z=velocity_surface[0],
            opacity=0.45,
            mode="lines",
            showlegend=False,
            line=dict(color="red", width=10),
            name="c2_constraint",
        )
    )


    mo.output.clear()
    return (fig,)


@app.cell
def _(CL_slider, dT_slider, mo):
    mo.hstack([dT_slider, CL_slider])
    return


@app.cell(hide_code=True)
def _(fig):
    fig
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
