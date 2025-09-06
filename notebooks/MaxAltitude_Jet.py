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
    from core.aircraft import velocity, horizontal_constraint, power, drag

    # Set local/online filepath
    _defaults.FILEURL = _defaults.get_url()

    # Plotly dark mode template
    _defaults.set_plotly_template()

    # Data directory
    data_dir = str(mo.notebook_location() / "public" / "AircraftDB_Standard.csv")
    return (mo,)


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
def _():
    # Insert graph of density decreasing with increasing altitude.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Moreover, since density is always positive, and $\beta$ as well, we can say, because $\sigma^\beta$ is a monotically increasing function of $\sigma$, minimizing $\sigma^\beta$ minimizes $\sigma$ which is maximizing $h$.

    $$
    \min_{C_L, \delta_T} \sigma  \qquad \Longleftrightarrow \qquad \max_{C_L, \delta_T} \quad \sigma^\beta
    $$

    We can thus now susbitute the horizontal equilibrium equation in the objective function directyl, and then also substitute the expression of $V$ rom the vertical equilibrium, constraint.
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


@app.cell
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

    Moreover, the stationarity condition 2 can be solved for a value of $\delta_T$ only when $\mu_2 \neq 0$, this means it also pointless to investigate the max-lift condition as we would have $\mu_2 = 0$ again.
    """
    )
    return


@app.cell
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
def _():
    # Not sure about the stall at higher speed than the one for $E_{\mathrm[max}}$
    return


if __name__ == "__main__":
    app.run()
