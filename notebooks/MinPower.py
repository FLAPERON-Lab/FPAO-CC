import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    from core import _defaults

    _defaults.FILEURL = _defaults.get_url()

    _defaults.set_plotly_template()


@app.cell
def _():
    _defaults.set_sidebar()
    return


@app.cell
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


@app.cell
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


@app.cell
def _():
    mo.md(
        r"""
    The KKT formulation can now be written: 

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


@app.cell
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
        r"""
    A necessary condition for an optimal solution of the optimization problem $(C_L^*, \delta_T^*)$ to exist, the multipliers $\lambda_1, \mu_1, \mu_2$ have to meet the following conditions:

    **A. Stationarity ($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \frac{\partial \mathcal{L}}{\partial C_L} = \sqrt{\frac{2W^3}{\rho S}}\left(-\frac{3}{2}C_{D_0}C_L^{-5/2} + \frac{1}{2} K C_L^{-1/2}\right) - \lambda_1 W \left(\frac{KC_L^2 -C_{D_0}}{C_L^2}\right) + \mu_1= 0$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1 T_{a0}\sigma^\beta+ \mu_2= 0$
    """
    )
    return


@app.cell
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


@app.cell
def _():
    mo.md(
        r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    6.  $\mu_1, \mu_2 \ge 0$
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraint have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

    7.  $\mu_1 (C_L - C_{L_\mathrm{max}}) = 0$
    8. $\mu_3 (\delta_T - 1) = 0$
    """
    )
    return


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
def _():
    mo.md(r"""Notice how $C_{L_P}$ (minimum power) $\gt$ $C_{L_E}$ (minimum drag) but $E_\mathrm{P} \lt E_{\mathrm{max}}$ ($E = C_L/C_D$) because drag increases more rapidly than $C_L$. Thus the range of $W/\sigma^\beta$ for which it is possible to fly at minimum power is smaller ($\sqrt{3}/2\lt 1$) than the one for which it is possible to fly at minimum drag.""")
    return


@app.cell
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

    This inequality is saying that the reuiqred power should decrease for an increase in $C_L$ starting from $C_{L_\mathrm{max}}$. In other words, $P_r$ should decrease for a decrease in speed from the stall speed. Equivalently, $P_r$ should increase for a n increase in speed from the stall speed. This is clearly impossible, by the taking the shape of the power curve on the performance diagram.

    - [ ] insert power diagram curve. 

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


@app.cell
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


@app.cell
def _():
    mo.md(
        r"""
    This tells us that the required power and drag change in opposite directions with respectot the change in $C_L$. If one decreaes, then the other one has to increase. 
    This can only happen in the range of $C_L$ between $C_{L_P}$ and $C_{L_E}$. On the performance diagram:

    - [ ] Plot performance diagram as seen on paper. not sure I understand how the C_{D_0} and K disappear under here.

    This condition is given by:

    $$
    C_{L_E}\lt C_L\lt C_{L_P} \quad \Leftrightarrow \quad \sqrt{\frac{C_{D_0}}{K}}\lt C_L \lt \sqrt{3} \sqrt{\frac{C_{D_0}}{K}}\quad \Leftrightarrow  \quad \boxed{1 \lt C_L \lt \sqrt{3}}
    $$

    Interistingly, it does not depend on any design parametes, in the assumptions made so far.
    """
    )
    return


@app.cell
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

    where the relevant solution is given by the "${+}$" sign, on the left branch of the drag curve: 

    $$
    \Rightarrow C_L^* = \frac{T_{a0}\sigma^\beta}{2KW}\left[1 +\sqrt{1- \left(\frac{W}{T_{a0}\sigma^\beta E_{\mathrm{max}}}\right)^2}\right]
    $$

    The solution is valid as long as: $1\lt C_L^* \lt \sqrt{3}$
    For its existence, the square root must be zero or positive, thus: 

    $$
    1 - \left(\frac{W}{T_{a0}\sigma^\beta E_{\mathrm{max}}}\right)^2 \ge 0 \quad \Rightarrow \quad \frac{W}{\sigma^\beta}\le T_{a0}E_{\mathrm{max}}
    $$

    as already seen in multiple occasions.
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    In order for $C_L^* \gt 1$ it must be:

    $$
    1 - \left(\frac{W}{T_{a0}\sigma^\beta E_{\mathrm{max}}}\right)^2  \gt \left(\frac{2KW}{T_{a0}\sigma^\beta} - 1\right)^2
    $$

    which then simplifies to: 

    $$
    \frac{W}{\sigma^\beta}\lt\frac{T_{a0}}{C_{D_0}+K}
    $$

    which can be compared to the previous one directly, when the latter is expressed as:

    $$
    \frac{W}{\sigma^\beta}\le T_{a0}E_{\mathrm{max}} = \frac{T_{a0}}{2\sqrt{KC_{D_0}}}
    $$

    - [ ] why is 9t a closed interval here?

    As it can be seen, 

    $$
    C_{D_0} + K \ge 2\sqrt{KC_{D_0}} \quad \Leftrightarrow \quad C_{D_0} - 2\sqrt{KC_{D_0}} + K \ge 0 \quad \Leftrightarrow \quad (\sqrt{C_{D_0}} - \sqrt{K})^2 \ge 0 \quad \mathrm{always}
    $$

    Therefore the strongest condition, or the lower upper bound, for $W/\sigma^\beta$ is given by: 


    $$
    \frac{W}{\sigma^\beta}\lt\frac{T_{a0}}{C_{D_0}+K}
    $$

    Similarly, 

    $$
    C_L^* \lt \sqrt{3} \quad \mathrm{for} \quad \frac{W}{\sigma^\beta} \gt \frac{\sqrt{3}\; T_{a0}}{3K+C_{D_0}}
    $$

    Which combined with the previous condition and together with the reults yields: 

    $$
    \boxed{\delta_T^* = 1} \qquad \land \qquad \boxed{C_L^* = \frac{T_{a0}\sigma^\beta}{2KW}\left[1 +\sqrt{1- \left(\frac{W}{T_{a0}\sigma^\beta E_{\mathrm{max}}}\right)^2}\right]} \qquad \forall \qquad \frac{T_{a0}}{\frac{C_{D_0}}{\sqrt{3}} + \sqrt{3}K} \lt \frac{W}{\sigma^\beta} \lt \frac{T_{a0}}{C_{D_0}+K}
    $$
    """
    )
    return


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
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


if __name__ == "__main__":
    app.run()
