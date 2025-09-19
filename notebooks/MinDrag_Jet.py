import marimo

__generated_with = "0.15.2"
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


@app.cell
def _(mo):
    mo.md(
        r"""
    ## KKT formulation

    Similarly to the derivation for minimum speed in a simplified jet, we express $V$ from $c_1^\mathrm{eq}$ and substitute it out of the entire problem to eliminate it. The KKT formulation thus becomes:
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


@app.cell
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


@app.cell
def _(mo):
    mo.md(
        r"""
    A necessary condition for an optimal solution of the optimization problem $(C_L^*, \delta_T^*)$ to exist, the multipliers $\lambda_1, \mu_1, \mu_2, \mu_3, \mu_4$ have to meet the following conditions:

    **A. Stationarity conditions($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \begin{aligned}\frac{\partial \mathcal{L}}{\partial C_L} = W \frac{K C_L^2 - C_{D_0}}{C_L^2} - \lambda_1W\left(\frac{K C_L^2 - C_{D_0}}{C_L^2}\right) + \mu_1 - \mu_2 = W\frac{K C_L^2 - C_{D_0}}{C_L^2} (1 -\lambda_1) +\mu_1 - \mu_2 = 0 \end{aligned}$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1\frac{T_{a0}\sigma^\beta}{W}+\mu_3-\mu_4 = 0$
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


@app.cell
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


@app.cell
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
    $0\lt C_L \lt  C_{L_{max}} \wedge 0 \lt \delta_T \lt 1$, as shown in ... (insert link of the page).
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The corresponding $\delta_T^*$ is found by solving the primal feasibility constraint (3) and using $C_L = C_L^*$.


    $$
    \delta_T^* = \frac{2W}{T_{a0}\sigma^\beta}\frac{C_{D_0} + K C_L^2}{C_L} = \frac{2W}{T_{a0}\sigma^\beta}\sqrt{C_{D_0}K}
    $$

    This value is compliant with the primal feasibility constraint(5) for: 

    $$
    \delta_T^* = \frac{2W}{T_{a0}\sigma^\beta}\sqrt{C_{D_0}K} \lt 1 \Leftrightarrow \frac{W}{\sigma^\beta} \lt \frac{T_{a0}}{2\sqrt{C_{D_0}K}} = \frac{W}{\sigma^\beta} \lt T_{a0}E_{max}$$

    The corresponding minimum drag is found by first computing $V^*$ and $C_D^*$: 

    $$
    V = \sqrt{\frac{W}{S}\frac{2}{\rho}\frac{1}{C_L^*}} \quad \land \quad  C_D^* = 
    2C_{D_0}$$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    $$
    D_{\mathrm{min}}^* =  \frac{1}{2}\rho {V^*}^2 S = 
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
    ###_Lift-limited minimum drag (stall boundary)_

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 > 0$ 

    $\delta_T < 1 \quad \Rightarrow \quad \mu_3 = 0$ 

    From stationarity condition (2): $\lambda_1 = 0$.

    From stationarity condition (1): $\displaystyle \mu_1 = W\frac{C_{D_0} - KC_{L_{max}}^2}{C_{L_{max}}^2} \gt 0$, which results to:

    $\sqrt{\frac{C_{D_0}}{K}} = C_{L_E} \gt C_{L_{max}}$ which is impossible by definition of $C_{L_{max}}$. 

    We conclude it is impossible to minimize drag in level flight, at $C_L = C_{L_{max}}$ with $\delta_T < 1$.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ###_Thrust-limited minimum drag_


    $C_L \lt C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 = 0$ 

    $\delta_T = 1 \quad \Rightarrow \quad \mu_3 \gt 0$ 

    From stationarity condition (2) obtain: 

    $$
    \lambda_1 = -\frac{\mu_3}{T_{a0}\sigma^{\beta}} \lt 0
    $$

    and from stationarity condition (1): 

    $$
    \displaystyle \left(\frac{KC_L^2 - C_{D_0}}{C_L^2}\right)(1-\lambda_1) = 0 \Rightarrow (1-\lambda_1) \gt 0
    $$

    $$
    \frac{KC_L^2 - C_{D_0}}{C_L^2} = 0
    $$

    Which yields the folowing optima:

    $$ 
    C_L^* = \sqrt{C_{D_0}}{K} = C_{L_E} \quad  \land \quad \delta_T^* = 1 
    $$

    This optimum is continuous with the interior optimum, thus yielding the same result for $D_{min}$:

    $$
    D_{min}^* =  \frac{1}{2}\rho {V^*}^2 S = 2W\sqrt{KC_{D_0}}=\frac{W}{E_{max}}
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    with:

    $$
    C_D^* = 2C_{D_0}, \quad V^* = \sqrt{\frac{W}{S}\frac{2}{\rho}\frac{1}{C_L^*}}, \quad \delta_T^*=\frac{2W}{T_{a0}\sigma^\beta}\sqrt{C_{D_0}K}=\frac{D_{min}^*}{T_{a0}\sigma^{\beta}}
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ###_Thrust- and stall- limited minimum drag_

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 \gt 0$ 

    $\delta_T = 1 \quad \Rightarrow \quad \mu_3 \gt 0$ 

    from stationarity condition (2): 

    $$
    \lambda_1= -\frac{\mu_3 }{T_{a0}\sigma^{\beta}} \lt 0
    $$

    from stationarity condition (1): 

    $$
    \displaystyle W (\lambda_1 - 1) \left( \frac{KC_{L_\mathrm{max}}^2 - C_{D_0}}{C_{L_\mathrm{max}}^2}\right) \gt 0 \Rightarrow (\lambda_1 - 1) \lt 0
    $$

    yielding: 

    $$
    C_{L_\mathrm{max}} \lt \sqrt{\frac{C_{D_0}}{K}} = C_{L_E} \Rightarrow \text{impossible}
    $$ 


    Once again, drag cannot be minimized in stall conditions.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Summary

    | Name | Condition | $C_L^*$ | $\delta_T^*$ | $D^*$ |
    |:-|:-------|:-------:|:------------:|:-------|
    |Interior-optima    | $\displaystyle \frac{W}{\sigma^\beta} < T_{a0} E_\mathrm{max}$ | $\sqrt{\frac{C_{D_0}}{K}}$ | $\displaystyle \frac{2W}{T_{a0}\sigma^\beta}\sqrt{C_{D_0}K}$ | $\displaystyle 2W\sqrt{KC_{D_0}}=\frac{W}{E_{max}}$ |
    |Thrust-limited    | $\displaystyle \frac{W}{\sigma^\beta} \le  T_{a0} E_\mathrm{max}$ | $\displaystyle \sqrt{\frac{C_{D_0}}{K}}$ | $1$ | $2W\sqrt{KC_{D_0}}=\frac{W}{E_{max}}$ |
    """
    )
    return


@app.cell
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


@app.cell
def _(mo):
    mo.md(
        r"""
    ## KKT formulation

    AS previously shown, we express $V$ from $c_1^\mathrm{eq}$ and substitute it out of the entire problem to eliminate it. The KKT formulation thus becomes:
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
    & \quad D = W \frac{C_{D_0} + K C_L^2}{C_L} = \frac{W}{E} \\
    \text{subject to} 
    & \quad g_1 = \frac{T}{W} - \frac{1}{E}  =\delta_T P_{a0}\sigma^\beta\sqrt{\frac{S}{W}\frac{\rho}{2}C_L} - W\frac{C_{D_0} +K C_L^2}{C_L} = 0 \quad \Rightarrow \quad A = \sqrt{\frac{\rho S}{2W}}\\
    & \quad \;\; \; = \delta_T P_{a0}\sigma^\beta A \sqrt{C_L} - W\frac{C_{D_0} +K C_L^2}{C_L} = 0\\
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

    The Lagrangian function combines the objective function with equality constraints using Lagrange multipliers ($\lambda_i$) and inequality constraints using KKT multipliers ($\mu_j$).

    $$
    \begin{aligned}
    \mathcal{L}(C_L, \delta_T, \lambda_1, \mu_1, \mu_2, \mu_3, \mu_4) = 
    \quad W\frac{C_{D_0} + K C_L^2}{C_L}
    & + \\
    & + \lambda_1 \left[\delta_T P_{a0}\sigma^\beta A \sqrt{C_L} - W\frac{C_{D_0} +K C_L^2}{C_L}\right] + \\
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

    **A. Stationarity conditions($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \begin{aligned}\frac{\partial \mathcal{L}}{\partial C_L} & = W \frac{K C_L^2 - C_{D_0}}{C_L^2} - \lambda_1W\left(\frac{K C_L^2 - C_{D_0}}{C_L^2}\right) +\frac{1}{2} \lambda_1 \delta_T P_{a0}\sigma^\beta A \frac{1}{\sqrt{C_L}}+ \mu_1 - \mu_2 \\
    & = W\frac{K C_L^2 - C_{D_0}}{C_L^2} (1 -\lambda_1) +  \frac{1}{2} \lambda_1\delta_T P_{a0}\sigma^\beta A \frac{1}{\sqrt{C_L}} +\mu_1 - \mu_2 = 0 \end{aligned}$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1 P_{a0} \sigma^\beta A \sqrt{C_L}+\mu_3-\mu_4 = 0$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $\displaystyle \frac{\delta_T P_{a0}\sigma^\beta A \sqrt{C_L}}{W} - \frac{C_{D_0} +K C_L^2}{C_L} = 0$
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


@app.cell
def _(mo):
    mo.md(
        r"""
    **Simplified conditions**

    1. $\displaystyle W\frac{K C_L^2 - C_{D_0}}{C_L^2} (1 -\lambda_1) +  \frac{1}{2} \lambda_1\delta_T P_{a0}\sigma^\beta A \frac{1}{\sqrt{C_L}} +\mu_1 = 0$
    2. $\displaystyle \lambda_1 P_{a0} \sigma^\beta A \sqrt{C_L}+\mu_3 = 0$
    3. $\displaystyle \frac{\delta_T P_{a0}\sigma^\beta A \sqrt{C_L}}{W} - \frac{C_{D_0} +K C_L^2}{C_L} = 0$
    4. $C_L - C_{L_\mathrm{max}} \le 0$
    5. $\delta_T - 1 \le 0$
    6. $\mu_1, \mu_3 \ge 0$
    7. $\mu_1 (C_L - C_{L_\mathrm{max}}) = 0$
    8. $\mu_3 (\delta_T - 1) = 0$
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

    Therefore: $\mu_1, \mu_3 =0$. 

    From stationarity condition (2): $\lambda_1 = 0$.

    From stationarity condition (1) it is possible to obtain the value of $C_L^*$ for minimum drag.

    $$
    C_L^* = \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$

    Notice how the optimal $C_L^*$ has the **same value** for maximum aerodynamic efficiency, or maximum $C_L/C_D$, for 
    $0\lt C_L \lt  C_{L_{max}} \wedge 0 \lt \delta_T \lt 1$, as shown in ... (insert link of the page).
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The corresponding $\delta_T^*$ is found by solving the primal feasibility constraint (3) and using $C_L = C_L^*$.


    $$
    \delta_T^* = \frac{W}{P_{a0}\sigma^\beta}\frac{1}{A}2C_{D_0}^{1/4}K^{3/4}
    $$

    This value is compliant with the primal feasibility constraint (5) for: 

    $$
    \delta_T^* =  \frac{W}{P_{a0}\sigma^\beta}\frac{1}{A}2C_{D_0}^{1/4}K^{3/4} <1 \Leftrightarrow \frac{W}{A\sigma^\beta} \lt \frac{P_{a0}}{2C_{D_0}^{1/4}K^{3/4}} \Leftrightarrow \sqrt{\frac{W^3}{\rho}}\frac{1}{\sigma^\beta} \lt \sqrt{\frac{S}{2}}\frac{P_{a0}}{2C_{D_0}^{1/4}K^{3/4}}
    $$

    The corresponding minimum drag is found by first computing $V^*$ and $C_D^*$: 

    $$
    V = \sqrt{\frac{W}{S}\frac{2}{\rho}\frac{1}{C_L^*}} \quad \land \quad  C_D^* = 
    2C_{D_0}$$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    $$
    D_{\mathrm{min}}^* =  \frac{1}{2}\rho {V^*}^2 S = 
    2W\sqrt{KC_{D_0}}=\frac{W}{E_\mathrm{max}}
    $$

    This is the same result as in the simplified jet analysis!
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ###_Lift-limited minimum drag (stall boundary)_

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 > 0$ 

    $\delta_T < 1 \quad \Rightarrow \quad \mu_3 = 0$ 

    From stationarity condition (2): $\lambda_1 = 0$.

    From stationarity condition (1): $\displaystyle \mu_1 = W\frac{C_{D_0} - KC_{L_{max}}^2}{C_{L_{max}}^2} \gt 0$, which results to:

    $\sqrt{\frac{C_{D_0}}{K}} = C_{L_E} \gt C_{L_{max}}$ which is impossible by definition of $C_{L_{max}}$. 

    We conclude it is impossible to minimize drag in level flight, at $C_L = C_{L_{max}}$ with $\delta_T < 1$.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ###_Thrust-limited minimum drag_


    $C_L \lt C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 = 0$ 

    $\delta_T = 1 \quad \Rightarrow \quad \mu_3 \gt 0$ 

    From stationarity condition (2) obtain:
    """
    )
    return


if __name__ == "__main__":
    app.run()
