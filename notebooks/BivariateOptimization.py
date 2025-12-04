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
    from core import aircraft as ac

    # Set local/online filepath
    _defaults.FILEURL = _defaults.get_url()

    # Plotly dark mode template
    _defaults.set_plotly_template()

    # Data directory
    data_dir = str(mo.notebook_location() / "public" / "AircraftDB_Standard.csv")


@app.cell
def _():
    # Set navbar on the right
    _defaults.set_sidebar()
    return


@app.cell
def _():
    mo.md(
        r"""
    # Bivariate optimization

    In this case, we are going to assume that the drag coefficient $C_D$ is a function of the lift coefficient $C_L$ and additionally also of the Mach number $M$.

    The simplest model that captures the phenomena related to these two parameters somewhat realistically is expressed by the following expression

    $$C_D(M, C_L) = C_{D_0}(M, C_L) + K_1(M, C_L)C_L + K_2(M, C_L)C_L^2$$

    where $C_{D_0}$, $K_1$, and $K_2$ are themselves functions of both the Mach number $M$ and the lift coefficient $C_L$.

    In particular:

    - $C_{D_0}(M, C_L) = [0.045 -0.059052 M + 0.025 M^2 + 0.005426 e^{12.942 (M - M_\mathrm{dd})}] + 
             [0.06 + 0.1 e^{2 (M - M_\mathrm{dd})}] (0.4 - 0.05 M)^2$
    - $K_1(M, C_L) = -2 [0.06 + 0.1 e^{2 (M - M_\mathrm{dd})}] (0.4 - 0.05 M)$
    - $K_2(M, C_L) =  0.06 + 0.1 e^{2 (M - M_\mathrm{dd})}$
    - $M_\mathrm{dd}(C_L) = 0.82 - 0.17 C_L$
    """
    )
    return


@app.cell
def _():
    mo.md(r"""TODO: plot CD in 3d, contour plots, and slices E(CL) for various Mach numbers -> recall previous notebook""")
    return


@app.cell
def _():
    mo.md(
        r"""
    This mathematical model requires many more parameters than the uncompressible parabolic drag polar, which requires only two. 
    We are departing significantly from the latter by adding effects related to 

    - strong increase of zero-lift drag $C_{D_0}$ with Mach number
    - small decrease of drag up to transonic Mach numbers for low lift coefficients ($K_1 < 0$)
    - drag divergence at high subsonic Mach numbers and/or high lift coefficients ($M_dd$ decreases with $C_L$)

    The latter makes it so that $C_{D_0}$, $K_1$ and $K_2$ on $C_L$ depend not only on the Mach number, but also on the lift coefficient.
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    The aerodynamic efficiency is therefore also a function of both the Mach number and the lift coefficient:

    $$ E = E(M, C_L) = \frac{C_L}{C_D(M, C_L)} $$

    Its defition can be expressed in a way to highlight its main components as:

    $$ E = \frac{C_L}{C_D} = \frac{C_L}{C_{D_0}(M, C_L) + K_1(M, C_L)C_L + K_2(M, C_L)C_L^2} $$
    """
    )
    return


@app.cell
def _():
    mo.md(r"""% TODO: plot E in 3d, contour plots, and slices E(CL) for various Mach numbers -> recall previous notebook""")
    return


@app.cell
def _():
    mo.md(
        r"""
    In this and in the following notebooks, we are going to leave the numerical values explicit in order not to clutter the analytical derivation. 
    Numbers can be simplified by operating on them, symbols would have to be combined in more and more complicated expressions all the way down to a solution. 
    In these notebooks we are still going to carry out the solution to the probem analytically, but you can imagine how the necessity to implement numerical methods to solve flight performance optimization problems becomes urgent quite fast.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Optimization

    The optimization problem can then be formulated as:
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Find the maximum aerodynamic efficiency by changing the lift coefficient within certain limits

    $$
    \begin{aligned}
        \max_{M, C_L} 
        & \quad E = \frac{C_L}{C_D} = \frac{C_L}{C_{D_0}(M, C_L) + K_1(M, C_L)C_L + K_2(M, C_L)C_L^2} \\
        % \text{subject to} 
        % & \quad \bm{c}_\mathrm{eq}(\bm{x},\bm{u}; \bm{p}) = 0 \\
        % & \quad \bm{c_\mathrm{ineq}}(\bm{x},\bm{u}; \bm{p}) \le 0 \\
        \text{for } 
        & \quad M \in [0, 1] \\
        & \quad C_L \in [0, 0.9]
    \end{aligned}
    $$

    """).callout().center().style({"text-align": "center"})
    return


@app.cell
def _():
    mo.md(
        r"""
    **Note**

    We are looking for the maximum of the objective function in a compact domain where all possible combinations of $M$ and $C_L$ can be evaluated.
    This is mathematically possible because the objective function is composed of elementary functions that are well defined in this domain. 
    It is also aerodynamically meaningful, as you can imagine obtaining the values of the objective function by means of CFD simulations or wind tunnel tests, where you can systematically vary both $M$ and $C_L$ independently from each other.

    On the other hand, this does not make a lot of sense from the Flight Performance point of view. 
    Aircraft cannot fly at very high lift coefficient and Mach numbers due to excessive loads on their structure (assuming they would have enough power to even get to those conditions).
    Because of this, such combinations of parameters are usually not even investigated during applied aircraft aerodynamic analyses (through CFD or wind tunnel tests), as they would cost time and money, and the aerodynamic field would be very complex while ultimately not relevant for flight.

    In any case, in this notebook we are going to pretend that the domain is a rectangle in the $M-C_L$ plane, and we are going to solve the optimization problem with a purely aerodynamic performance metric, focusing on the mathematical methodology.
    """
    ).callout(kind="warn").center()
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    Similarly to what we have seen in the previous notebook, $E$ is a non-monotonic continuous function of $M$ and $C_L$, and therefore it is guaranteeed by the Extreme Value Theorem to have a maximum (and also a minimum) in the compact domain.
    In order to find it, we need to evaluate the objective function in its stationary points and on the boundary of its domain, and then compare the values obtained in these points.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Stationary values

    We can find the stationary points of the objective function by equating its gradient to zero:

    $$ \nabla E = \left( \frac{\partial E}{\partial M}, \frac{\partial E}{\partial C_L} \right) = (0, 0) $$

    The partial derivatives are:

    $$ 
    \frac{\partial E}{\partial M} 
    =
    -\frac{C_L}{C_D^2} \frac{\partial C_D}{\partial M} 
    =
    - C_L \frac{\frac{\partial C_{D_0}}{\partial M} + \frac{\partial K_1}{\partial M}C_L + \frac{\partial K_2}{\partial M}C_L^2}{\left( C_{D_0} + K_1 C_L + K_2 C_L^2 \right)^2} = 0 
    $$

    $$ 
    \frac{\partial E}{\partial C_L}
    = 
    \frac{1}{C_D} - \frac{C_L}{C_D^2} \frac{\partial C_D}{\partial C_L} 
    =
    \frac{C_{D_0}  + K_1 C_L + K_2 C_L^2 - C_L \left( \frac{\partial C_{D_0}}{\partial C_L} + \frac{\partial K_1}{\partial C_L}C_L + K_1  + \frac{\partial K_2}{\partial C_L}C_L^2 + 2K_2 C_L \right)}{\left( C_{D_0}  + K_1 C_L + K_2 C_L^2 \right)^2} = 0
    $$

    After calculating the expressions of the derivatives, setting $\frac{\partial E}{\partial M} = 0$ and $\frac{\partial E}{\partial C_L} = 0$ leads to a complex system of nonlinear equations in $M$ and $C_L$, which cannot be solved analytically to obtain a closed-form solution.

    By solving this system of equations numerically, we find that the objective function has only one stationary point within the domain. Its coordinates and the corresponding value of the aerodynamic efficiency are given by: 

    $$ M^* = 0.64, \quad C_L^* = 0.50 \quad \Rightarrow \quad E^* = 23.24 $$
    """
    )
    return


@app.cell
def _():
    mo.md(r"""% TODO: is it the case to show a code snippet of how to solve this problem using some python function like fsolve?""")
    return


@app.cell
def _():
    mo.md(
        r"""
    ## Boundary values

    The domain boundary is now constituted by four edges, and we need to evaluate the objective function on each of them.

    Because the domain is a rectangle, either $M$ or $C_L$ are constant on each edge, and the problem reduces to a univariate optimization problem similar to the one we have seen in the previous notebook.

    **Edge 1**: $M = 0$, $C_L \in [0, 0.9]$

    The expression for aerodynamic efficiency becomes dependent only on $C_L$ where the coefficients are evaluated at $M=0$.

    $$  E(0, C_L) = \frac{C_L}{C_{D_0}(0, C_L) + K_1(0, C_L)C_L + K_2(0, C_L)C_L^2} $$

    The boundary maximum on this edge occurs at $C_L^\dagger = 0.82$ and is equal to $E^\dagger = 13.64$.


    **Edge 2**: $M = 1$, $C_L \in [0, 0.9]$

    The expression for aerodynamic efficiency becomes dependent only on $C_L$ where the coefficients are evaluated at $M=1$.

    $$ E(1, C_L) = \frac{C_L}{C_{D_0}(1, C_L) + K_1(1, C_L)C_L + K_2(1, C_L)C_L^2} $$

    The boundary maximum on this edge occurs at $C_L^\dagger = 0.44$ and is equal to $E^\dagger = 2.76$.


    **Edge 3**: $C_L = 0$, $M \in [0, 1]$

    The expression for aerodynamic efficiency becomes dependent only on $M$ where the coefficients are evaluated at $C_L=0$.

    $$ E(M, 0) = \frac{0}{C_{D_0}(M, 0) + K_1(M, 0) \cdot 0 + K_2(M, 0) \cdot 0^2} = 0 $$

    For all values of $M$, the aerodynamic efficiency is zero since $C_L = 0$.


    **Edge 4**: $C_L = 0.9$, $M \in [0, 1]$

    The expression for aerodynamic efficiency becomes dependent only on $M$ where the coefficients are evaluated at $C_L=0.9$.

    $$ E(M, 0.9) = \frac{0.9}{C_{D_0}(M, 0.9) + K_1(M, 0.9) \cdot 0.9 + K_2(M, 0.9) \cdot 0.9^2} $$

    The boundary maximum on this edge occurs at $M^\dagger = 0.39$ and is equal to $E^\dagger = 15.62$.
    """
    )
    return


@app.cell
def _():
    mo.md(r"""% TODO: show plots of these 1d functions?""")
    return


@app.cell
def _():
    mo.md(
        r"""
    ## Conclusion
    By comparing the extreme values of the objective function at the stationary point and on the boundary of the domain, we find that the maximum aerodynamic efficiency within the defined domain occurs at the stationary point:

    $$ 
    E_{\mathrm{max}} = E^* = 23.24 
    \quad \text{for} \quad
    M = M^* = 0.64, \quad C_L = C_L^* = 0.50
    $$
    """
    )
    return


@app.cell
def _():
    _defaults.nav_footer(
        "UnivariateOptimization.py",
        "Univariate Optimization",
        "EqualityConstraints.py",
        "Equality Constraints",
    )
    return


if __name__ == "__main__":
    app.run()
