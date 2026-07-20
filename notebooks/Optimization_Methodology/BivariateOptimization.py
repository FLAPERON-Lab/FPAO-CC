# SPDX-FileCopyrightText: 2026 Carmine Varriale <C.varriale@tudelft.nl>
# SPDX-FileCopyrightText: 2026 Federico Angioni <F.angioni@student.tudelft.nl>
# SPDX-FileCopyrightText: 2026 Maarten van Hoven <M.B.vanHoven@tudelft.nl>
#
# SPDX-License-Identifier: Apache-2.0
import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path.cwd()))

    # Initialization code that runs before all other cells
    import marimo as mo

    # Import dependencies
    from core import _defaults
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np
    from core import aircraft as ac
    from scipy.optimize import fsolve

    # Set local/online filepath
    _defaults.FILEURL = _defaults.get_url()

    # Plotly dark mode template
    _defaults.set_plotly_template()

    # Data directory
    data_dir = str(
        mo.notebook_location().parent.parent / "data" / "AircraftDB_Standard.csv"
    )


@app.cell
def _():
    # Set navbar on the right
    _defaults.set_sidebar()
    return


@app.cell
def _():
    mo.md(r"""
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
    """)
    return


@app.cell
def _(CD_func):
    # Variables declared
    meshgrid_n = 101
    xy_lowerbound = -0.1

    CL_buffer = 1
    # Handle deselected row from table
    CLmax = 0.9  # active_selection["CLmax_ld"]
    M_range = np.linspace(0, 1, meshgrid_n)
    CL_range = np.linspace(0, CLmax, meshgrid_n)
    # Create meshgrid
    M_grid, CL_grid = np.meshgrid(M_range, CL_range)

    # Evaluate CD on the grid
    CD_grid = CD_func(M_grid, CL_grid)

    CL_array = np.linspace(0, CLmax + CL_buffer, meshgrid_n)
    return CD_grid, CL_grid, CL_range, CLmax, M_range


@app.cell
def _():
    mo.md(r"""
    This mathematical model requires many more parameters than the uncompressible parabolic drag polar, which requires only two.
    We are departing significantly from the latter by adding effects related to

    - strong increase of zero-lift drag $C_{D_0}$ with Mach number
    - small decrease of drag up to transonic Mach numbers for low lift coefficients ($K_1 < 0$)
    - drag divergence at high subsonic Mach numbers and/or high lift coefficients ($M_dd$ decreases with $C_L$)

    The latter makes it so that $C_{D_0}$, $K_1$ and $K_2$ on $C_L$ depend not only on the Mach number, but also on the lift coefficient.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    The aerodynamic efficiency is therefore also a function of both the Mach number and the lift coefficient:

    $$ E = E(M, C_L) = \frac{C_L}{C_D(M, C_L)} $$

    It's definition can be expressed in a way to highlight its main components as:

    $$ E = \frac{C_L}{C_D} = \frac{C_L}{C_{D_0}(M, C_L) + K_1(M, C_L)C_L + K_2(M, C_L)C_L^2} $$
    """)
    return


@app.cell
def _(CD_grid, CL_grid, CL_range, M_range):
    figure_surface = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"type": "scene"}]],
    )

    figure_surface.add_trace(
        go.Surface(
            x=M_range,
            y=CL_range,
            z=CL_grid / CD_grid,
            opacity=0.9,
            colorscale="viridis",
            colorbar={"title": "E (-)", "len": 0.5, "y": 0.5},
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    figure_surface.update_scenes(
        xaxis_title="M (-)",
        yaxis_title=r"C<sub>L</sub> (-)",
        zaxis_title=r"E (-)",
        row=1,
        col=1,
    )

    figure_surface.update_layout(
        title_text="Aerodynamic Efficiency Surface",
        title_font_size=25,
        title_x=0.5,
        height=600,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Below you can find a heatmap and slices of the aerodynamic efficiency surface above formulated. Try to find yourself what the points could be candidates to maximize the aerodynamic efficiency function!
    """)
    return


@app.cell
def _(CLmax):
    M_slider = mo.ui.slider(start=0, stop=1, step=0.05, label="$M$")
    CL_slider = mo.ui.slider(0, CLmax, step=0.05, label="$C_L$")
    mo.hstack([M_slider, CL_slider])
    return CL_slider, M_slider


@app.cell
def _(CD_func, CD_grid, CL_grid, CL_range, CL_slider, M_range, M_slider):
    figure_CD = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [
                {"type": "xy", "colspan": 2},
                None,
            ],  # row 1: 2D heatmap spanning 2 columns
            [{"type": "xy"}, {"type": "xy"}],  # row 2: 2D scatter plots
        ],
        row_heights=[0.5, 0.5],
    )

    figure_CD.add_trace(
        go.Heatmap(
            x=M_range,
            y=CL_range,
            z=CL_grid / CD_grid,
            zsmooth="fast",
            colorscale="viridis",
            opacity=0.9,
            showlegend=False,
            showscale=False,
        ),
        row=1,
        col=1,
    )

    # Contour lines
    figure_CD.add_traces(
        [
            go.Contour(
                x=M_range,
                y=CL_range,
                z=CL_grid / CD_grid,
                showlegend=False,
                contours=dict(
                    showlines=True,
                    coloring="none",  # <- important: lines only
                ),
                line=dict(color="black", width=1),
                showscale=False,  # don't add a second colorbar
            ),
            go.Scatter(
                x=[M_slider.value, M_slider.value],
                y=[0.0, np.max(CL_range)],
                line=dict(color="red", width=3, dash="dot"),
                showlegend=False,
            ),
            go.Scatter(
                x=[0.0, 1.0],
                y=[CL_slider.value, CL_slider.value],
                line=dict(color="red", width=3, dash="dot"),
                showlegend=False,
            ),
        ],
        rows=1,
        cols=1,
    )

    figure_CD.add_traces(
        [
            go.Scatter(
                x=CL_range,
                y=CL_range / CD_func(M_slider.value, CL_range),
                name=r"$E$",
                showlegend=False,
            ),
            go.Scatter(
                x=[CL_slider.value, CL_slider.value],
                y=[0.0, CL_slider.value / CD_func(M_slider.value, CL_slider.value)],
                line=dict(color="red", width=3, dash="dot"),
                showlegend=False,
            ),
        ],
        cols=1,
        rows=2,
    )

    figure_CD.add_traces(
        [
            go.Scatter(
                x=M_range,
                y=CL_slider.value / CD_func(M_range, CL_slider.value),
                name=r"$E$",
                showlegend=False,
            ),
            go.Scatter(
                x=[M_slider.value, M_slider.value],
                y=[0.0, CL_slider.value / CD_func(M_slider.value, CL_slider.value)],
                line=dict(color="red", width=3, dash="dot"),
                showlegend=False,
            ),
        ],
        cols=2,
        rows=2,
    )

    # Calculate max efficiency for y-axis limits
    max_E = np.max(CL_grid / CD_grid)

    figure_CD.update_xaxes(title_text=r"$M \; (-)$", col=1, row=1)
    figure_CD.update_yaxes(title_text=r"$C_L \; (-)$", range=[0, 0.9], col=1, row=1)
    figure_CD.update_xaxes(title_text=r"$C_L \; (-)$", range=[0, 0.9], col=1, row=2)
    figure_CD.update_yaxes(title_text=r"$E \; (-)$", range=[0, max_E], col=1, row=2)
    figure_CD.update_yaxes(title_text=r"$E \; (-)$", range=[0, max_E], col=2, row=2)
    figure_CD.update_xaxes(title_text=r"$M \; (-)$", col=2, row=2)

    figure_CD.update_layout(
        title_text="Heatmap with Slices for Aerodynamic Efficiency",
        title_font_size=25,
        title_x=0.5,
        height=700,
    )
    return


@app.cell
def _():
    mo.md(r"""
    In this and in the following notebooks, we are going to leave the numerical values explicit in order not to clutter the analytical derivation.
    Numbers can be simplified by operating on them, symbols would have to be combined in more and more complicated expressions all the way down to a solution.
    In these notebooks we are still going to carry out the solution to the probem analytically, but you can imagine how the necessity to implement numerical methods to solve flight performance optimization problems becomes urgent quite fast.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Optimization

    The optimization problem can then be formulated as:
    """)
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
    mo.md(r"""
    Similarly to what we have seen in the previous notebook, $E$ is a non-monotonic continuous function of $M$ and $C_L$, and therefore it is guaranteeed by the Extreme Value Theorem to have a maximum (and also a minimum) in the compact domain.
    In order to find it, we need to evaluate the objective function in its stationary points and on the boundary of its domain, and then compare the values obtained in these points.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Stationary values

    We can find the stationary points of the objective function by equating its gradient to zero:

    $$
    \nabla E = \left( \frac{\partial E}{\partial M}, \frac{\partial E}{\partial C_L} \right) = (0, 0)
    $$

    The partial derivatives are:

    $$
    \frac{\partial E}{\partial M}
    = -\frac{C_L}{C_D^2} \frac{\partial C_D}{\partial M}
    = - C_L \frac{\frac{\partial C_{D_0}}{\partial M} + \frac{\partial K_1}{\partial M}C_L + \frac{\partial K_2}{\partial M}C_L^2}{\left( C_{D_0} + K_1 C_L + K_2 C_L^2 \right)^2} = 0
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
    """)
    return


@app.cell
def _():
    def M_dd_func(CL):
        return 0.82 - 0.17 * CL

    def CD_func(M, CL):
        M_dd_val = M_dd_func(CL)
        exp_12 = np.exp(12.942 * (M - M_dd_val))
        exp_2 = np.exp(2 * (M - M_dd_val))

        CD0 = (0.045 - 0.059052 * M + 0.025 * M**2 + 0.005426 * exp_12) + (
            0.06 + 0.1 * exp_2
        ) * (0.4 - 0.05 * M) ** 2
        K1 = -2 * (0.06 + 0.1 * exp_2) * (0.4 - 0.05 * M)
        K2 = 0.06 + 0.1 * exp_2

        return CD0 + K1 * CL + K2 * CL**2

    def E_func(M, CL):
        return CL / CD_func(M, CL)

    def gradient_E_numerical(x):
        M, CL = x
        h = 1e-8
        dE_dM = (E_func(M + h, CL) - E_func(M - h, CL)) / (2 * h)
        dE_dCL = (E_func(M, CL + h) - E_func(M, CL - h)) / (2 * h)
        return [dE_dM, dE_dCL]

    # Solve for stationary point
    x0 = [0.6, 0.4]
    solution, info, ier, msg = fsolve(gradient_E_numerical, x0, full_output=True)
    M_star, CL_star = solution

    print("Stationary Point Found:")
    print(f"  M*  = {M_star:.2f}")
    print(f"  CL* = {CL_star:.2f}")
    print(f"  E*  = {E_func(M_star, CL_star):.2f}")

    mo.show_code()
    return CD_func, CL_star, E_func, M_star


@app.cell
def _(CL_star, E_func, M_star):
    with mo.redirect_stdout():
        print("Stationary Point Found:")
        print(f"  M*  = {M_star:.2f}")
        print(f"  CL* = {CL_star:.2f}")
        print(f"  E*  = {E_func(M_star, CL_star):.2f}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
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
    """)
    return


@app.cell(hide_code=True)
def _(CD_func, CL_range, CLmax, M_range):
    # Create 2x2 subplot
    fig_edges = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Edge 1: M = 0",
            "Edge 2: M = 1",
            "Edge 3: C<sub>L</sub> = 0",
            "Edge 4: C<sub>L</sub> = 0.9",
        ),
    )

    # Edge 1: M = 0, CL varies
    E_edge1 = CL_range / CD_func(0, CL_range)
    fig_edges.add_trace(
        go.Scatter(
            x=CL_range,
            y=E_edge1,
            mode="lines",
            name="E(0, C<sub>L</sub>)",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )
    # Mark the maximum
    CL_max_edge1 = 0.82
    E_max_edge1 = 13.64
    fig_edges.add_trace(
        go.Scatter(
            x=[CL_max_edge1],
            y=[E_max_edge1],
            mode="markers",
            marker=dict(color="red", size=10, symbol="star"),
            name=f"Max: C<sub>L</sub>={CL_max_edge1}, E={E_max_edge1}",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Edge 2: M = 1, CL varies
    E_edge2 = CL_range / CD_func(1, CL_range)
    fig_edges.add_trace(
        go.Scatter(
            x=CL_range,
            y=E_edge2,
            mode="lines",
            name="E(1, C<sub>L</sub>)",
            line=dict(color="green"),
        ),
        row=1,
        col=2,
    )
    # Mark the maximum
    CL_max_edge2 = 0.44
    E_max_edge2 = 2.76
    fig_edges.add_trace(
        go.Scatter(
            x=[CL_max_edge2],
            y=[E_max_edge2],
            mode="markers",
            marker=dict(color="red", size=10, symbol="star"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Edge 3: CL = 0, M varies (E = 0 everywhere)
    E_edge3 = np.zeros_like(M_range)
    fig_edges.add_trace(
        go.Scatter(
            x=M_range,
            y=E_edge3,
            mode="lines",
            name="E(M, 0) = 0",
            line=dict(color="orange"),
        ),
        row=2,
        col=1,
    )

    # Edge 4: CL = 0.9, M varies
    E_edge4 = CLmax / CD_func(M_range, CLmax)
    fig_edges.add_trace(
        go.Scatter(
            x=M_range,
            y=E_edge4,
            mode="lines",
            name="E(M, 0.9)",
            line=dict(color="purple"),
        ),
        row=2,
        col=2,
    )
    # Mark the maximum
    M_max_edge4 = 0.39
    E_max_edge4 = 15.62
    fig_edges.add_trace(
        go.Scatter(
            x=[M_max_edge4],
            y=[E_max_edge4],
            mode="markers",
            marker=dict(color="red", size=10, symbol="star"),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # Update axes labels
    # Update axes labels
    fig_edges.update_xaxes(title_text=r"$C_L \; (-)$", row=1, col=1)
    fig_edges.update_yaxes(title_text=r"$E \; (-)$", row=1, col=1)

    fig_edges.update_xaxes(title_text=r"$C_L \; (-)$", row=1, col=2)
    fig_edges.update_yaxes(title_text=r"$E \; (-)$", row=1, col=2)

    fig_edges.update_xaxes(title_text=r"$M \; (-)$", row=2, col=1)
    fig_edges.update_yaxes(title_text=r"$E \; (-)$", row=2, col=1)

    fig_edges.update_xaxes(title_text=r"$M \; (-)$", row=2, col=2)
    fig_edges.update_yaxes(title_text=r"$E \; (-)$", row=2, col=2)
    # Update layout
    fig_edges.update_layout(
        title_text="Aerodynamic Efficiency on Domain Boundaries",
        title_x=0.5,
        height=600,
        showlegend=False,
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## Conclusion
    By comparing the extreme values of the objective function at the stationary point and on the boundary of the domain, we find that the maximum aerodynamic efficiency within the defined domain occurs at the stationary point:

    $$
    E_{\mathrm{max}} = E^* = 23.24
    \quad \text{for} \quad
    M = M^* = 0.64, \quad C_L = C_L^* = 0.50
    $$
    """)
    return


@app.cell
def _():
    _defaults.nav_footer(
        "Optimization_Methodology/UnivariateOptimization.py",
        "Univariate Optimization",
        "Optimization_Methodology/EqualityConstraints.py",
        "Equality Constraints",
    )
    return


if __name__ == "__main__":
    app.run()
