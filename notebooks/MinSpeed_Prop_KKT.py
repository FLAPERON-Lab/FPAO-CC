import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    # Initialization code that runs before all other cells
    import marimo as mo
    from core import _defaults

    _defaults.FILEURL = _defaults.get_url()

    _defaults.set_plotly_template()
    return (mo,)


@app.cell
def _():
    _defaults.set_sidebar()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Minimum airspeed: simplfied piston propeller aircraft

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
        & \quad T_a(V,h) = \eta \frac{P_a(h)}{V} = \eta \frac{P_{a0}\sigma^\beta}{V} \\
    \end{aligned}
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    We could approahc the solution of this problem in the same way we have approched the one for simplified jets: obtain the expression of $V$ from $c_1^\mathrm{eq}$, substitute it out of the whole problem, then proceed with deriving with respec to $C_L$ and $\delta_T$.
    In the case of propeller airplanes, this results in the following expression of the horizontal equilibrium contraint, which is unhandy to take derivatives with respect to $C_L$:

    $$
    \delta_T \eta \frac{P_{a0}\sigma^\beta}{V} - \frac{1}{2} \rho V^2 S \left( C_{D_0} + K C_L^2 \right) = 0
    \quad \Leftrightarrow \quad
    \delta_T \eta P_{a0}\sigma^\beta - \frac{1}{2} \rho S \left(\frac{2W}{\rho S C_L} \right)^{3/2}\left( C_{D_0} + K C_L^2 \right) = 0
    $$ 

    Instead, in this case, it is more convenient to reformulate the problem by eliminating $C_L$ instead of $V$.
    """
    )
    return


@app.cell
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
    \delta_T \eta P_{a0}\sigma^\beta - \frac{1}{2} \rho V^3 S \left( C_{D_0} + \frac{4KW^2}{\rho^2 S^2  V^4}\right) = 0
    \quad \Leftrightarrow \quad
    \delta_T \eta P_{a0}\sigma^\beta - \frac{1}{2} \rho S V^3 C_{D_0} - \frac{2KW^2}{\rho S V} = 0
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


@app.cell
def _(mo):
    mo.md(
        r"""
    ## KKT Formulation

    $$
    \begin{aligned}
        \min_{V, \delta_T} 
        & \quad V \\
        \text{subject to} 
        & \quad g_1 = \delta_T \eta P_{a0}\sigma^\beta - \frac{1}{2} \rho S V^3 C_{D_0} - \frac{2KW^2}{\rho S V} = 0 \\
        & \quad h_1 = \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}} - V \le 0 \\
        & \quad h_2 = -\delta_T \le 0 \\
        & \quad h_3 = \delta_T - 1 \le 0 \\
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
    \mathcal{L}(V, \delta_T, \lambda_1, \mu_1, \mu_2, \mu_3) = 
    \quad \frac{2W}{\rho S C_L}
    & + \\
    & + \lambda_1 \left(\delta_T \eta P_{a0}\sigma^\beta - \frac{1}{2} \rho S V^3 C_{D_0} - \frac{2KW^2}{\rho S V}\right) + \\
    & + \mu_1 \left( \frac{2W}{\rho S C_{L_\mathrm{max}}} - V \right) + \\
    & + \mu_2 (-\delta_T) + \\
    & + \mu_3 (\delta_T - 1) +\\
    \end{aligned}
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **A. Stationarity conditions($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \frac{\partial \mathcal{L}}{\partial V} = 1 + \lambda_1 \left( \frac{2KW^2}{\rho S V^2} - \frac{3}{2}\rho V^2SC_{D_0} \right) -\mu_1 = 0$
    2. $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1 \eta P_{a0}\sigma^\beta - \mu_2 + \mu_3 = 0$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $\displaystyle \delta_T \eta P_{a0}\sigma^\beta - \frac{1}{2} \rho S V^3 C_{D_0} - \frac{2KW^2}{\rho S V} = 0$
    4.  $\displaystyle \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}} - V \le 0$
    5.  $-\delta_T \le 0$
    6.  $\delta_T - 1 \le 0$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    8.  $\mu_1, \mu_2, \mu_3 \ge 0$
    """
    )
    return


@app.cell
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


@app.cell
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


@app.cell
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
    \delta_T^* = \frac{ \displaystyle \frac{1}{2}\rho V^3_s S C_{D_0} + \frac{2KW^2}{\rho S V_s} }{\eta P_{a0} \sigma^\beta}
    $$

    The condition to achieve this is given by $0 \le \delta_T^* \le 1$, where only the right-hand side is relevant.
    This tells that the required power at stall speed has to be less or equal to the available power at stall speed, and is equivalent to either of the two following conditions:

    $$
    \frac{W^{3/2}}{\sigma^{\beta+1/2}} \le \eta P_{a0} E_S \sqrt{\frac{1}{2}\rho_0 S C_{L_\mathrm{max}}}
    \quad \Leftrightarrow \quad
    \frac{W}{\sigma^{\beta+1/2}} \le \frac{\eta P_{a0} E_S}{V_{s0}}
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### _Idle thrust boundary active_

    In this case: $\mu_2 > 0, \delta_T=0, \mu_1=\mu_3=0$

    It is easy to see that the primal feasibility constraint 3, in other words the horizontal equilibrium, can never be verified.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### _Max throttle boundary active_

    In this case: $\mu_3 > 0, \delta_T=1, \mu_1=\mu_2=0$

    From the stationarity conditions and the complementary slack conditions: 

    $$
    \mu_3 = -\lambda_1 \eta P_{a0}\sigma^\beta\\
    1 + \lambda_1 \left( \frac{2KW^2}{\rho S V^2} - \frac{3}{2}\rho V^2SC_{D_0} \right) = 0 
    $$

    Therefore, $\mu_3 > 0$ if $\lambda_1 < 0$, and the latter is true when:

    $$
    \frac{3}{2}\rho V^2SC_{D_0} - \frac{2KW^2}{\rho S V^2} < 0
    \quad \Leftrightarrow \quad 3 C_{D_0} - K C_L^2 < 0 
    \quad \Leftrightarrow \quad C_L > \sqrt{\frac{3 C_{D_0}}{K}} = \sqrt{3} C_{L_E} = C_{L_P}
    $$

    This means that minimum speed is achieved at max throttle when flying on the induced branch of the power curve, that is with a lift coefficient that is higher than the one for minimum required power ($C_{L_P}$) and lower than $C_{L_\mathrm{max}}$) of course. 

    The corresponding minimum speed is obtained by solving the following equation: 

    $$
    V^* : \eta P_{a0}\sigma^\beta - \frac{1}{2} \rho S V^3 C_{D_0} - \frac{2KW^2}{\rho S V} = 0 
    $$

    which cannot be solved analytically.
    The solution is valid only if $V^* > V_s$.

    The corresponding throttle is $\delta_T^*=1$ and the optimum lift coefficient is: $C_L^* = \frac{2W}{\rho S V^{*2}}$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### _Max throttle and stall boundaries active_

    In this case: $V=V_s, \delta_T=1, \mu_1 > 0, \mu_2 > 0, \mu_3 > 0$

    From the stationarity conditions and the complementary slack conditions:

    $$
    \mu_3 = -\lambda_1 \eta P_{a0}\sigma^\beta > 0 \\
    \mu_1 = 1 + \lambda_1 \left[ \frac{1}{2}\rho V_s^2 S \left( K C_{L_\mathrm{max}}^2 - 3 C_{D_0}\right)\right] > 0
    $$

    It follows that:

    $$
    - \frac{1}{\frac{1}{2}\rho V_s^2 S \left( K C_{L_\mathrm{max}}^2 - 3 C_{D_0}\right)} \le \lambda_1 \le 0
    $$

    which corresponds to $C_{L_\mathrm{max}} \ge \sqrt{\frac{3 C_{D_0}}{K}} = C_{L_P}$, and is always verified, by defintion of $C_{L_\mathrm{max}}$.

    The condition in which this optimum is achieved is given by the horizontal equilibrium constraint, which states that the required power has to be equal to the available power in stall conditions and at max throttle. This results in the following equation:

    $$
    \frac{W^{3/2}}{\sigma^{\beta+1/2}} = \eta P_{a0} E_S \sqrt{\frac{1}{2}\rho_0 S C_{L_\mathrm{max}}}
    \quad \Leftrightarrow \quad
    \frac{W}{\sigma^{\beta+1/2}} = \frac{\eta P_{a0} E_S}{V_{s0}}
    $$
    """
    )
    return


if __name__ == "__main__":
    app.run()
