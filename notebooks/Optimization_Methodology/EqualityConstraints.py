import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo

    # Import dependencies
    from core import _defaults
    from core import atmos
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
    data_dir = str(mo.notebook_location().parent / "public" / "AircraftDB_Standard.csv")


@app.cell
def _():
    # Set navbar on the right
    _defaults.set_sidebar()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Equality Constraints

    In the previous notebook, we sought to maximize the aerodynamic efficiency $E(M, C_L)$ over a rectangular domain with no additional restrictions beyond simple bounds on the decision variables.

    In many practical flight performance problems, however, we need to respect additional constraints that relate the decision variables to each other or to fixed parameters. These constraints arise from physical laws, operational requirements, or design specifications.

    In this notebook, we focus on how to include and treat _equality_ constraints in an optimization problem.
    It is the first step moving from the optimization of an "aerodynamic performance metric" to the optimization of a "flight mechanics performance" metric.

    We will continue using aerodynamic efficiency as an objective function, with the same drag coefficient model from the bivariate optimization case:

    $$ E(M, C_L) = \frac{C_L}{C_D(M, C_L)} $$

    $$C_D(M, C_L) = C_{D_0}(M, C_L) + K_1(M, C_L)C_L + K_2(M, C_L)C_L^2$$
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

    return (CD_func,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Simple Equality Constraints

    Let's first consider two simple types of equality constraints that commonly appear in flight performance analysis.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fixed Lift Coefficient

    Consider the case where the lift coefficient must remain at a specific value:

    $$ C_L = C_{L_\mathrm{given}} \quad \text{or equivalently} \quad C_L - C_{L_\mathrm{given}} = 0 $$

    With our simple models, this corresponds to a given angle of attack.
    This constraint could be relevant for various operational scenarios. For example, during aerial refueling, the receiving aircraft must maintain a fixed attitude to stay in the correct position relative to the tanker. In horizontal flight, maintaining a constant attitude (pitch angle) corresponds to maintaining a constant angle of attack and therefore a constant $C_L$ as well.

    The optimization problem can be written as:

    $$
    \begin{aligned}
        \max_{M, C_L}
        & \quad E(M, C_L) = \frac{C_L}{C_D(M, C_L)} \\
        \text{subject to}
        & \quad C_L - C_{L_\mathrm{given}} = 0 \\
        \text{for }
        & \quad M \in [0, 1]
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fixed Mach Number

    Another common constraint is flying at a specific Mach number:

    $$ M = M_\mathrm{given} \quad \text{or equivalently} \quad M - M_\mathrm{given} = 0 $$

    This might represent, for instance, the highest Mach number before encountering drag divergence or other undesired aerodynamic phenomena, or a specific cruise Mach number required by air traffic control or mission profile.

    The optimization problem can be written as:

    $$
    \begin{aligned}
        \max_{M, C_L}
        & \quad E(M, C_L) = \frac{C_L}{C_D(M, C_L)} \\
        \text{subject to}
        & \quad M - M_\mathrm{given} = 0 \\
        \text{for }
        & \quad C_L \in [0, 0.9]
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Solution Method: Substitution

    For these simple equality constraints, the method to solve the optimization problem is straightforward. The constraint equation explicitly specifies the value of one decision variable, allowing us to substitute it directly into the objective function.

    **For fixed $C_L$**: We substitute $C_L = C_{L_\mathrm{given}}$ into $E(M, C_L)$ to get:

    $$ E(M) = E(M, C_{L_\mathrm{given}}) = \frac{C_{L_\mathrm{given}}}{C_D(M, C_{L_\mathrm{given}})} $$

    This reduces the constrained optimization problem to an **unconstrained univariate optimization** problem with $M$ being the only decision variable:

    $$
    \begin{aligned}
        \max_{M}
        & \quad E(M) = \frac{C_{L_\mathrm{given}}}{C_D(M, C_{L_\mathrm{given}})} \\
        \text{for }
        & \quad M \in [0, 1]
    \end{aligned}
    $$

    **For fixed $M$**: We substitute $M = M_\mathrm{given}$ into $E(M, C_L)$ to get:

    $$ E(C_L) = E(M_\mathrm{given}, C_L) = \frac{C_L}{C_D(M_\mathrm{given}, C_L)} $$

    This similarly reduces to an **unconstrained univariate optimization** problem with $C_L$ being the only decision variable:

    $$
    \begin{aligned}
        \max_{C_L}
        & \quad E(C_L) = \frac{C_L}{C_D(M_\mathrm{given}, C_L)} \\
        \text{for }
        & \quad C_L \in [0, 0.9]
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Geometric interpretation

    Graphically, this means we are looking for the maximum of the objective function along the constraint curve (which is a straight line in these cases).
    We are only interested in the values of the function that lie on this slice of the domain.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    TODO: transform the charts below to interactive (AI did the first draft). Similar to Bivariate, but now the constraint lines must be treated differently. For example: for the case of given CL, use a numerical input to ask users to fix the value of the CL constraint. Then use a slider to let them change the Mach number to find the optimum. Same, with inverted role, for the case of given Mach.

    If possible, plot the contour line corresponding to the current value of the maximum, this will show the tangency condition discussed soon
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

    E_grid = CL_grid / CD_grid

    CL_array = np.linspace(0, CLmax + CL_buffer, meshgrid_n)

    M_slider_eq = mo.ui.slider(start=0, stop=1, step=0.05, label="$M$")
    CL_slider_eq = mo.ui.slider(0, CLmax, step=0.05, label="$C_L$")
    return CL_range, CL_slider_eq, E_grid, M_range, M_slider_eq


@app.cell
def _(CD_func, CL_range, CL_slider_eq, M_range, M_slider_eq):
    # Slices of E along constraints
    # For fixed CL
    CL_selected_eq = CL_slider_eq.value
    E_fixed_CL = CL_selected_eq / CD_func(M_range, CL_selected_eq)
    M_opt_CL = M_range[np.argmax(E_fixed_CL)]
    E_max_CL = np.max(E_fixed_CL)

    # For fixed M
    M_selected_eq = M_slider_eq.value
    E_fixed_M = CL_range / CD_func(M_selected_eq, CL_range)
    CL_opt_M = CL_range[np.argmax(E_fixed_M)]
    E_max_M = np.max(E_fixed_M)
    return CL_opt_M, CL_selected_eq, E_max_CL, E_max_M, M_opt_CL, M_selected_eq


@app.cell
def _():
    # Create separate sliders for fig_simple
    CL_constraint_left = mo.ui.slider(
        0, 0.9, step=0.05, label="$C_L$ (constraint)", value=0.6
    )
    M_position_left = mo.ui.slider(0, 1, step=0.05, label="$M$ (position)", value=0.5)
    M_constraint_right = mo.ui.slider(
        0, 1, step=0.05, label="$M$ (constraint)", value=0.7
    )
    CL_position_right = mo.ui.slider(
        0, 0.9, step=0.05, label="$C_L$ (position)", value=0.5
    )
    return (
        CL_constraint_left,
        CL_position_right,
        M_constraint_right,
        M_position_left,
    )


@app.cell
def _(
    CD_func,
    CL_constraint_left,
    CL_position_right,
    CL_range,
    E_grid,
    M_constraint_right,
    M_position_left,
    M_range,
):
    fig_simple = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Fixed C<sub>L</sub> = {CL_constraint_left.value:.2f}",
            f"Fixed M = {M_constraint_right.value:.2f}",
        ),
    )

    # Left plot: Fixed CL (constraint), varying M
    fig_simple.add_trace(
        go.Contour(
            x=M_range,
            y=CL_range,
            z=E_grid,
            colorscale="viridis",
            contours=dict(
                showlines=True,
                coloring="heatmap",
            ),
            showscale=False,
        ),
        row=1,
        col=1,
    )

    # Calculate maximum efficiency along left constraint line
    E_along_left_constraint = CL_constraint_left.value / CD_func(
        M_range, CL_constraint_left.value
    )
    E_max_left = np.max(E_along_left_constraint)

    # Add red contour line at maximum efficiency for left plot
    fig_simple.add_trace(
        go.Contour(
            x=M_range,
            y=CL_range,
            z=E_grid,
            contours=dict(
                start=E_max_left,
                end=E_max_left,
                size=0.1,
                showlabels=False,
                coloring="none",
            ),
            line=dict(color="red", width=2),
            showscale=False,
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    # Constraint line on left (horizontal CL line)
    fig_simple.add_trace(
        go.Scatter(
            x=M_range,
            y=[CL_constraint_left.value] * len(M_range),
            mode="lines",
            line=dict(color="red", dash="dot"),
            name="Constraint",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Optimal point on left (star at M_position_left)
    E_at_point_left = CL_constraint_left.value / CD_func(
        M_position_left.value, CL_constraint_left.value
    )
    fig_simple.add_trace(
        go.Scatter(
            x=[M_position_left.value],
            y=[CL_constraint_left.value],
            mode="markers",
            marker=dict(color="yellow", size=12, symbol="star"),
            name="Optimum",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Right plot: Fixed M (constraint), varying CL
    fig_simple.add_trace(
        go.Contour(
            x=M_range,
            y=CL_range,
            z=E_grid,
            colorscale="viridis",
            contours=dict(
                showlines=True,
                coloring="heatmap",
            ),
            colorbar=dict(title="E (-)", len=0.9, x=1.02, y=0.4, yanchor="middle"),
            showscale=True,
        ),
        row=1,
        col=2,
    )

    # Calculate maximum efficiency along right constraint line
    E_along_right_constraint = CL_range / CD_func(M_constraint_right.value, CL_range)
    E_max_right = np.max(E_along_right_constraint)

    # Add red contour line at maximum efficiency for right plot
    fig_simple.add_trace(
        go.Contour(
            x=M_range,
            y=CL_range,
            z=E_grid,
            contours=dict(
                start=E_max_right,
                end=E_max_right,
                size=0.1,
                showlabels=False,
                coloring="none",
            ),
            line=dict(color="red", width=2),
            showscale=False,
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )

    # Constraint line on right (vertical M line)
    fig_simple.add_trace(
        go.Scatter(
            x=[M_constraint_right.value] * len(CL_range),
            y=CL_range,
            mode="lines",
            line=dict(color="red", dash="dot"),
            name="Constraint",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Optimal point on right (star at CL_position_right)
    E_at_point_right = CL_position_right.value / CD_func(
        M_constraint_right.value, CL_position_right.value
    )
    fig_simple.add_trace(
        go.Scatter(
            x=[M_constraint_right.value],
            y=[CL_position_right.value],
            mode="markers",
            marker=dict(color="yellow", size=12, symbol="star"),
            name="Optimum",
            showlegend=True,
        ),
        row=1,
        col=2,
    )

    # Calculate max efficiency for axis limits
    max_E = np.max(E_grid)

    fig_simple.update_xaxes(title_text=r"$M \; (-)$", row=1, col=1)
    fig_simple.update_yaxes(title_text=r"$C_L \; (-)$", range=[0, 0.9], row=1, col=1)
    fig_simple.update_xaxes(title_text=r"$M \; (-)$", row=1, col=2)
    fig_simple.update_yaxes(title_text=r"$C_L \; (-)$", range=[0, 0.9], row=1, col=2)

    fig_simple.update_layout(
        title_text="Simple Equality Constraints",
        title_x=0.5,
        height=500,
        showlegend=True,
    )

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack([CL_constraint_left, M_position_left]),
                    mo.vstack([M_constraint_right, CL_position_right]),
                ],
                justify="center",
            ),
            fig_simple,
        ]
    )
    return


@app.cell
def _(
    CD_func,
    CL_constraint_left,
    CL_position_right,
    CL_range,
    M_constraint_right,
    M_position_left,
    M_range,
):
    fig_slices = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Fixed C<sub>L</sub> = {CL_constraint_left.value:.2f}",
            f"Fixed M = {M_constraint_right.value:.2f}",
        ),
    )

    # Left plot: E vs M for fixed CL
    E_along_CL = CL_constraint_left.value / CD_func(M_range, CL_constraint_left.value)
    E_max_along_CL = np.max(E_along_CL)

    fig_slices.add_trace(
        go.Scatter(
            x=M_range,
            y=E_along_CL,
            mode="lines",
            line=dict(color="blue", width=2),
            name="E(M, C<sub>L</sub>)",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Max efficiency line on left
    fig_slices.add_trace(
        go.Scatter(
            x=[M_range[0], M_range[-1]],
            y=[E_max_along_CL, E_max_along_CL],
            mode="lines",
            line=dict(color="red", width=2),
            name="Max E",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Current point on left
    E_marker_left = CL_constraint_left.value / CD_func(
        M_position_left.value, CL_constraint_left.value
    )
    fig_slices.add_trace(
        go.Scatter(
            x=[M_position_left.value],
            y=[E_marker_left],
            mode="markers",
            marker=dict(color="yellow", size=10, symbol="star"),
            name="Current point",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Right plot: E vs CL for fixed M
    E_along_M = CL_range / CD_func(M_constraint_right.value, CL_range)
    E_max_along_M = np.max(E_along_M)

    fig_slices.add_trace(
        go.Scatter(
            x=CL_range,
            y=E_along_M,
            mode="lines",
            line=dict(color="green", width=2),
            name="E(M, C<sub>L</sub>)",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Max efficiency line on right
    fig_slices.add_trace(
        go.Scatter(
            x=[CL_range[0], CL_range[-1]],
            y=[E_max_along_M, E_max_along_M],
            mode="lines",
            line=dict(color="red", width=2),
            name="Max E",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Current point on right
    E_marker_right = CL_position_right.value / CD_func(
        M_constraint_right.value, CL_position_right.value
    )
    fig_slices.add_trace(
        go.Scatter(
            x=[CL_position_right.value],
            y=[E_marker_right],
            mode="markers",
            marker=dict(color="yellow", size=10, symbol="star"),
            name="Current point",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig_slices.update_xaxes(title_text=r"$M \; (-)$", row=1, col=1)
    fig_slices.update_yaxes(title_text=r"$E \; (-)$", row=1, col=1)
    fig_slices.update_xaxes(title_text=r"$C_L \; (-)$", row=1, col=2)
    fig_slices.update_yaxes(title_text=r"$E \; (-)$", row=1, col=2)

    fig_slices.update_layout(
        title_text="Simple Equality Constraints - 1D Slices",
        title_font_size=25,
        title_x=0.5,
        height=400,
        showlegend=False,
    )

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack([CL_constraint_left, M_position_left]),
                    mo.vstack([M_constraint_right, CL_position_right]),
                ],
                justify="center",
            ),
            fig_slices,
        ]
    )
    return


@app.cell(hide_code=True)
def _(CL_opt_M, CL_selected_eq, E_max_CL, E_max_M, M_opt_CL, M_selected_eq):
    mo.md(f"""
    **Results:**

    - **Fixed $C_L = {CL_selected_eq}$**: Optimal Mach number $M^* = {M_opt_CL:.3f}$, with $E_{{\\mathrm{{max}}}} = {E_max_CL:.2f}$
    - **Fixed $M = {M_selected_eq}$**: Optimal lift coefficient $C_L^* = {CL_opt_M:.3f}$, with $E_{{\\mathrm{{max}}}} = {E_max_M:.2f}$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    As you can see, the constrained optimum lies at the point where a contour line of the objective function is tangent to the constraint line.
    In the case of only one equality constraint, this can be geometrically explained as follows:

    - The gradient of the objective function $\nabla E$ points in the direction of maximum increase of the function
    - A contour line of the objective function is perpendicular to the gradient at every point
    - For every point on the constraint curve, the gradient can have a component tangent to the constraint curve $\nabla_\parallel E$ and a component perpendicular to the constraint curve $\nabla_\perp E$.
        - if $\nabla_\parallel E \ne 0$ at a point on the constraint, the value of the objective function can still increase along the constraint curve; hence, that point cannot be the constrained optimum
        - if $\nabla_\parallel E = 0$ and $\nabla_\perp E \ne 0$ at a point on the constraint, the value of the objective function can only increase away from the constraint; hence, that point could potentially be a constrained optimum
        - if $\nabla_\parallel E = 0$ and $\nabla_\perp E = 0$ at a point on the constraint, the value of the objective function cannot increase in any direction; hence, that point could potentially be an interior optimum that lies on the constraint

    In particular, when a point on the constraint is the constrained optimum, the gradient of the objective function $\nabla E$ is  perpendicular to the constraint at the constrained optimum.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Complex Equality Constraint: Vertical Equilibrium

    Now let's consider a more realistic and complex constraint: the vertical equilibrium equation for steady level flight, according to which the lift force must equal the weight of the aircraft.

    $$ L = \frac{1}{2} \rho V^2 S C_L = W $$

    This can be rearranged in terms of Mach number as

    $$ C_L M^2 = \frac{2W}{\gamma p S }  \quad \text{or equivalently} \quad \gamma p S C_L M^2 - 2W = 0 $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The optimization problem now becomes:

    $$
    \begin{aligned}
        \max_{M, C_L}
        & \quad E(M, C_L) = \frac{C_L}{C_D(M, C_L)} \\
        \text{subject to}
        & \quad \gamma p S C_L M^2 - 2W = 0 \\
        \text{for }
        & \quad M \in [0, 1] \\
        & \quad C_L \in [0, 0.9]
    \end{aligned}
    $$

    This constraint is more complex than the previous ones because it relates both decision variables to each other through a nonlinear equation.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To solve this problem, we could of course still use substitution, in principle.
    The constraint can be manipulated to express one variable in terms of the other:

    $$ C_L = \frac{2W}{\gamma p S M^2} \quad \text{or} \quad M = \sqrt{\frac{2W}{\gamma p S C_L}} $$

    However, substituting either expression would significantly complicate the objective function:

    - If we substitute $C_L = \frac{2W}{\gamma p S M^2}$, we get:

    $$ E(M) = \frac{2W}{\gamma p S M^2C_D \left(M, \frac{2W}{\gamma p S M^2}\right) } $$

    - If we substitute $M = \sqrt{\frac{2W}{\gamma p S C_L}}$, we get:

    $$ E(C_L) = \frac{C_L}{C_D\left(\sqrt{\frac{2W}{\gamma p S C_L}}, C_L\right)} $$

    Both resulting functions are quite complex, and their algebraic manipulation is hard to manage, which means it can lend itself to errors.
    Imagine then the complexity of adding more constraints: it quickly becomes completely impossible.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Solution Method: Lagrange Multipliers

    A more elegant and systematic approach for handling equality constraints is the method of Lagrange multipliers.
    This method allows us to solve constrained optimization problems without explicit substitution.

    Let us rewrite the previous problem in more general form as

    $$
    \begin{aligned}
        \max_{M, C_L}
        & \quad \mathcal{J}(M, C_L)\\
        \text{subject to}
        & \quad g(M, C_L) = 0 \\
        \text{for }
        & \quad M \in [0, 1] \\
        & \quad C_L \in [0, 0.9]
    \end{aligned}
    $$

    where $\mathcal{J}(M,C_L)$ is the objective function (in our case the aerodynamic efficiency) and $g(M, C_L) = \gamma p S C_L M^2 - 2W$ is the constraint function, and therefore $g(M, C_L) = 0$ is the feasible region.



    The key idea is to introduce a new "multiplier" variable $\lambda$ and form the Lagrangian function:

    $$ \mathcal{L}(M, C_L, \lambda) = \mathcal{J}(M, C_L) + \lambda g(M, C_L) $$

    The Lagrangian function is a linear combination of the objective function and the equality constraints, and is a function of the decision variables and the multipliers (one for each constraint equation).

    The "Lagrange Multipliers theorem" is a necessary condition for a constrained optimum, and can be stated as follows:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    *Let $\mathcal{J}: \mathbb{R}^n \to \mathbb{R}$ and $g: \mathbb{R}^n \to \mathbb{R}$ be continuously differentiable functions. If $\mathbf{x}^* \in \mathbb{R}^n$ is a local extremum of $\mathcal{J}(\mathbf{x})$ subject to the constraint $g(\mathbf{x}) = 0$, and if $\nabla g(\mathbf{x}^*) \neq \mathbf{0}$, then there exists a scalar $\lambda^* \in \mathbb{R}$ such that*

    $$ \nabla \mathcal{J}(\mathbf{x}^*) + \lambda^* \nabla g(\mathbf{x}^*) = \mathbf{0} $$

    *Equivalently, defining the Lagrangian $\mathcal{L}(\mathbf{x}, \lambda) = \mathcal{J}(\mathbf{x}) + \lambda g(\mathbf{x})$, the necessary conditions for $(\mathbf{x}^*, \lambda^*)$ to be a constrained extremum are:*

    $$
    \begin{aligned}
        \nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}^*, \lambda^*) &= \mathbf{0} \\
        \frac{\partial \mathcal{L}}{\partial \lambda}(\mathbf{x}^*, \lambda^*) &= g(\mathbf{x}^*) = 0
    \end{aligned}
    $$

    *where $\nabla_{\mathbf{x}} \mathcal{L}$ denotes the gradient with respect to the decision variables $\mathbf{x}$, and $\lambda^*$ is called the Lagrange multiplier.*
    """).callout()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Recall now the tangency interpretation we have observed before in the case of only one equality constraint.
    At the constrained optimum, the gradient of the objective function must be perpendicular to the constraint curve.
    But the constraint function $g$ is constant (always equal to 0) on the constraint curve itself.
    This means that the gradient of the constraint function is also perpendicular to the constraint curve, for every point on the constraint.

    Therefore, at the constrained optimum, the gradient of the objective function must be parallel to the gradient of the constraint function. In other words:

    $$ \nabla \mathcal{J}^* = -\lambda \nabla g^*
    \quad \Leftrightarrow \quad
    \nabla \mathcal{J}^* + \lambda \nabla g^* = 0
    \quad \Leftrightarrow \quad
    \nabla \mathcal{L}^* = 0
    $$

    In summary, the necessary conditions for a point in the domain to be a constrained optimum is that the gradient of the Lagrangian function with respect to all variables and multipliers is zero in that point.

    The constrained optimum of the original problem is therefore a stationary point for the Lagrangian, which helps transforming the constrained optimization problem into an unconstrained one.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In our example with aerodynamic efficiency, the maximum efficiency in vertical equilibrium is found by solving the following equations for $C_L^*, M^*$, and $\lambda^*$:

    $$
    \begin{aligned}
        \frac{\partial \mathcal{L}}{\partial M} &= \frac{\partial E}{\partial M} + \lambda \frac{\partial g}{\partial M} = 0 \\
        \frac{\partial \mathcal{L}}{\partial C_L} &= \frac{\partial E}{\partial C_L} + \lambda \frac{\partial g}{\partial C_L} = 0 \\
        \frac{\partial \mathcal{L}}{\partial \lambda} &= g(M, C_L) = 0
    \end{aligned}
    $$

    The expression of the gradient of the objective function has already been derived (or, at least, expanded) in the previous notebook, for the unconstrained optimization problem.
    The expression of the gradient of the constraint function is very easy to derive:

    $$ \frac{\partial g}{\partial M} = 2\gamma p S C_L M $$

    $$ \frac{\partial g}{\partial C_L} = \gamma p S M^2 $$


    Lastly, the derivative of the Lagrangian with respect to the multiplier is the constraint function itself, and imposing that it must be equal to zero is equivalent to stating that the constraint must be respected.
    This is also called the "feasibility condition".

    $$ \frac{\partial \mathcal{L}}{\partial \lambda} = g(M, C_L) = 0
    \quad \Leftrightarrow \quad
    C_LM^2 = \frac{2W}{\gamma p S} $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The constraint curve position depends on flight parameters like weight and altitude. You can explore how the constraint changes with these parameters using the sliders below.
    """)
    return


@app.cell
def _():
    # Create interactive sliders for weight and altitude
    W_slider = mo.ui.slider(
        start=30000,
        stop=100000,
        step=5000,
        label="Weight W (N)",
        value=50000,
    )
    altitude_slider = mo.ui.slider(
        start=0,
        stop=20000,
        step=500,
        label="Altitude h (m)",
        value=0,
    )
    return W_slider, altitude_slider


@app.cell
def _(CD_func, W_slider, altitude_slider):
    # Standard atmosphere parameters
    gamma = 1.4
    R = 287  # J/(kg·K)
    S = 100  # m^2 (example wing area)

    # Get slider values
    W = W_slider.value
    h = altitude_slider.value

    # Pressure variation with altitude (barometric formula)
    p = atmos.p(h)

    # Generate constraint curve
    M_constraint = np.linspace(0.001, 1.0, 120)
    CL_constraint = (2 * W) / (gamma * p * S * M_constraint**2)

    # Clip CL values to stay within plot range [0, 0.9]
    CL_constraint_clipped = np.clip(CL_constraint, 0, 2.0)

    # Find valid points for optimization (where CL <= 0.9)
    valid_idx = CL_constraint <= 0.9
    M_constraint_valid = M_constraint[valid_idx]
    CL_constraint_valid = CL_constraint[valid_idx]

    # Evaluate E along constraint (only on valid portion)
    if len(M_constraint_valid) > 0:
        E_constraint = CL_constraint_valid / CD_func(
            M_constraint_valid, CL_constraint_valid
        )

        # Find optimum along constraint
        opt_idx = np.argmax(E_constraint)
        M_opt_eq = M_constraint_valid[opt_idx]
        CL_opt_eq = CL_constraint_valid[opt_idx]
        E_opt_eq = E_constraint[opt_idx]
    else:
        # No valid points on constraint (all CL > 0.9)
        M_opt_eq = M_constraint[0]
        CL_opt_eq = 0.9
        E_opt_eq = 0.9 / CD_func(M_constraint[0], 0.9)
    return (
        CL_constraint_clipped,
        CL_opt_eq,
        E_constraint,
        E_opt_eq,
        M_constraint,
        M_constraint_valid,
        M_opt_eq,
    )


@app.cell
def _(
    CL_constraint_clipped,
    CL_opt_eq,
    CL_range,
    E_grid,
    E_opt_eq,
    M_constraint,
    M_opt_eq,
    M_range,
    W_slider,
    altitude_slider,
):
    fig_equilibrium = go.Figure()

    # Contour plot of E
    fig_equilibrium.add_trace(
        go.Contour(
            x=M_range,
            y=CL_range,
            z=E_grid,
            colorscale="viridis",
            contours=dict(
                showlines=True,
                coloring="heatmap",
            ),
            colorbar=dict(title="E (-)", len=0.9, x=1.02, y=0.43, yanchor="middle"),
            showscale=True,
        )
    )

    # Constraint curve (clipped to visible range)
    fig_equilibrium.add_trace(
        go.Scatter(
            x=M_constraint,
            y=CL_constraint_clipped,
            mode="lines",
            line=dict(color="red", width=3, dash="dot"),
            name="Vertical Equilibrium",
        )
    )

    # Optimal point
    fig_equilibrium.add_trace(
        go.Scatter(
            x=[M_opt_eq],
            y=[CL_opt_eq],
            mode="markers+text",
            marker=dict(color="yellow", size=12, symbol="star"),
            name="Optimum",
            text=[f"E={E_opt_eq:.2f}"],
            textposition="top right",
        )
    )

    fig_equilibrium.update_xaxes(title_text=r"$M \; (-)$")
    fig_equilibrium.update_yaxes(title_text=r"$C_L \; (-)$", range=[0, 0.9])

    fig_equilibrium.update_layout(
        title_text="Constrained Optimization: Vertical Equilibrium",
        title_font_size=25,
        title_x=0.5,
        height=500,
        showlegend=True,
    )

    mo.vstack(
        [
            mo.hstack([W_slider, altitude_slider]),
            fig_equilibrium,
        ]
    )
    return


@app.cell(hide_code=True)
def _(CL_opt_eq, E_opt_eq, M_opt_eq, W_slider, altitude_slider):
    mo.md(f"""
    **Result:**

    For the vertical equilibrium constraint with weight W = {W_slider.value:,.0f} N and altitude h = {altitude_slider.value:,.0f} m, the optimal condition is:
    - Mach number: $M^* = {M_opt_eq:.3f}$
    - Lift coefficient: $C_L^* = {CL_opt_eq:.3f}$
    - Maximum aerodynamic efficiency: $E_{{\\mathrm{{max}}}} = {E_opt_eq:.2f}$

    Note how the optimal point lies on the constraint curve at the location where a contour line of $E$ is tangent to the curve. As you adjust weight and altitude, observe how the constraint curve and optimum shift.
    """)
    return


@app.cell(hide_code=True)
def _(
    E_constraint,
    E_opt_eq,
    M_constraint_valid,
    M_opt_eq,
    W_slider,
    altitude_slider,
):
    fig_E_constraint = go.Figure()

    fig_E_constraint.add_trace(
        go.Scatter(
            x=M_constraint_valid,
            y=E_constraint,
            mode="lines",
            line=dict(color="blue", width=2),
            name="E along constraint",
        )
    )

    fig_E_constraint.add_trace(
        go.Scatter(
            x=[M_opt_eq],
            y=[E_opt_eq],
            mode="markers",
            marker=dict(color="yellow", size=12, symbol="star"),
            name="Maximum",
        )
    )

    fig_E_constraint.update_xaxes(title_text=r"$M \; (-)$")
    fig_E_constraint.update_yaxes(title_text=r"$E \; (-)$")

    fig_E_constraint.update_layout(
        title_text="Aerodynamic Efficiency Along Equilibrium Constraint",
        title_x=0.5,
        title_font_size=25,
        height=400,
    )

    mo.vstack(
        [
            mo.hstack([W_slider, altitude_slider]),
            fig_E_constraint,
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Multiple Equality Constraints

    The Lagrange multiplier method naturally extends to problems with multiple equality constraints.

    Consider the general constrained optimization problem:

    $$
    \begin{aligned}
        \max_{\mathbf{x}}
        & \quad \mathcal{J}(\mathbf{x})\\
        \text{subject to}
        & \quad g_i(\mathbf{x}) = 0, \quad i = 1, 2, \ldots, m \\
        \text{for }
        & \quad \mathbf{x} \in \mathcal{D} \subseteq \mathbb{R}^n
    \end{aligned}
    $$

    where $\mathbf{x} = (x_1, x_2, \ldots, x_n)$ are the decision variables, $\mathcal{J}: \mathbb{R}^n \to \mathbb{R}$ is the objective function, and $g_i: \mathbb{R}^n \to \mathbb{R}$ are $m$ constraint functions with $m < n$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    *Let $\mathcal{J}: \mathbb{R}^n \to \mathbb{R}$ and $g_i: \mathbb{R}^n \to \mathbb{R}$ for $i = 1, 2, \ldots, m$ be continuously differentiable functions. If $\mathbf{x}^* \in \mathbb{R}^n$ is a local extremum of $\mathcal{J}(\mathbf{x})$ subject to the constraints $g_i(\mathbf{x}) = 0$ for $i = 1, 2, \ldots, m$, and if the constraint gradients $\nabla g_1(\mathbf{x}^*), \nabla g_2(\mathbf{x}^*), \ldots, \nabla g_m(\mathbf{x}^*)$ are linearly independent, then there exist scalars $\lambda_1^*, \lambda_2^*, \ldots, \lambda_m^* \in \mathbb{R}$ such that*

    $$ \nabla \mathcal{J}(\mathbf{x}^*) + \sum_{i=1}^{m} \lambda_i^* \nabla g_i(\mathbf{x}^*) = \mathbf{0} $$

    *Equivalently, defining the Lagrangian function*

    $$ \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}) = \mathcal{J}(\mathbf{x}) + \sum_{i=1}^{m} \lambda_i g_i(\mathbf{x}) $$

    *where $\boldsymbol{\lambda} = (\lambda_1, \lambda_2, \ldots, \lambda_m)$, the necessary conditions for $(\mathbf{x}^*, \boldsymbol{\lambda}^*)$ to be a constrained extremum are:*

    $$
    \begin{aligned}
        \nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}^*, \boldsymbol{\lambda}^*) &= \mathbf{0} \\
        \frac{\partial \mathcal{L}}{\partial \lambda_i}(\mathbf{x}^*, \boldsymbol{\lambda}^*) &= g_i(\mathbf{x}^*) = 0, \quad i = 1, 2, \ldots, m
    \end{aligned}
    $$
    """).callout()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The condition that $\nabla g_1(\mathbf{x}^*), \ldots, \nabla g_m(\mathbf{x}^*)$ are linearly independent is sometimes called the "Linear Independence Constraint Qualification (LICQ)".


    With multiple equality constraints, a geometric interpretation becomes a bit less immediate.

    Each constraint $g_i(\mathbf{x}) = 0$ defines a hyper-surface in $\mathbb{R}^n$, and the gradient $\nabla g_i(\mathbf{x})$ is perpendicular to that surface at every point.
    The feasible region is the intersection of all constraint hyper-surfaces and is typically a manifold of dimension $n - m$ (assuming the constraints are independent):

    $$ \mathcal{F} = \{\mathbf{x} \in \mathbb{R}^n : g_i(\mathbf{x}) = 0 \text{ for all } i = 1, 2, \ldots, m\} $$

    At a constrained optimum $\mathbf{x}^*$:

    - The gradient of the objective function $\nabla \mathcal{J}(\mathbf{x}^*)$ must be perpendicular to the feasible region $\mathcal{F}$ at $\mathbf{x}^*$, otherwise the objective function would increase within the feasbile region itself
    - The tangent space to $\mathcal{F}$ at $\mathbf{x}^*$ is the intersection of the tangent spaces to each individual constraint surface
    - The normal space to $\mathcal{F}$ at $\mathbf{x}^*$ is the span of all constraint gradients $\{\nabla g_1(\mathbf{x}^*), \nabla g_2(\mathbf{x}^*), \ldots, \nabla g_m(\mathbf{x}^*)\}$, as the gradient of a constraint function is perpendicular to the feasible region defined by that constraint

    In light of this, the gradient of the objective function at the optimum $\nabla \mathcal{J}(\mathbf{x}^*)$ is only allowed to lie in the space spanned by all constraint gradients.
    This is mathematically equivalent to saying that it must be a linear combination of the constraint gradients and the Lagrange multipliers:

    $$ \nabla \mathcal{J}(\mathbf{x}^*) = -\sum_{i=1}^{m} \lambda_i^* \nabla g_i(\mathbf{x}^*) $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Advantages

    The method of Lagrange multipliers offers several advantages over substitution:

    1. We don't need to solve the constraint for one variable in terms of the others and then take long compound derivatives, which can be algebraically complex or even impossible for some constraints.

    2. The method is systematic and generalizable to any number of equality constraints

    3. The Lagrange multiplier $\lambda$ can be interpretated as the sensitivity of the objective function to the constraint. Specifically, it indicates how much the objective function would change if the constraint was relaxed slightly.

    In the following notebook, we will explore how the method of Lagrange multipliers extends to inequality constraints through the Karush-Kuhn-Tucker (KKT) conditions.
    """)
    return


@app.cell
def _():
    _defaults.nav_footer(
        "BivariateOptimization.py",
        "Bivariate Optimization",
        "InequalityConstraints.py",
        "Inequality Constraints",
    )
    return


if __name__ == "__main__":
    app.run()
