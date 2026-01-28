import marimo

__generated_with = "0.19.4"
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


@app.function
def CD(M, CL):
    """
    Evaluate the drag coefficient C_D(M, C_L)

    Parameters
    ----------
    M : float or ndarray
        Mach number
    CL : float or ndarray
        Lift coefficient

    Returns
    -------
    CD : float or ndarray
        Drag coefficient
    """

    # Drag-divergence Mach number
    M_dd = 0.82 - 0.17 * CL

    # Common term
    A = 0.06 + 0.1 * np.exp(2.0 * (M - M_dd))

    # C_D0
    CD0 = 0.045 - 0.06 * M + 0.025 * M**2 + 0.005 * np.exp(13 * (M - M_dd)) + A * (0.4 - 0.05 * M) ** 2

    # K1 and K2
    K1 = -2.0 * A * (0.4 - 0.05 * M)
    K2 = A

    # Total drag coefficient
    CD = CD0 + K1 * CL + K2 * CL**2

    return CD


@app.cell
def _():
    mo.md(r"""
    ## Simple Equality Constraints

    Let's first consider two simple types of equality constraints that commonly appear in flight performance analysis.
    """)
    return


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
def _():
    mo.md(r"""
    ## Geometric interpretation

    Graphically, this means we are looking for the maximum of the objective function along the constraint curve (which is a straight line in these cases).
    We are only interested in the values of the function that lie on this slice of the domain.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    TODO: transform the charts below to interactive (AI did the first draft). Similar to Bivariate, but now the constraint lines must be treated differently. For example: for the case of given CL, use a numerical input to ask users to fix the value of the CL constraint. Then use a slider to let them change the Mach number to find the optimum. Same, with inverted role, for the case of given Mach.

    If possible, plot the contour line corresponding to the current value of the maximum, this will show the tangency condition discussed soon
    """)
    return


@app.cell
def _():
    # Constraint values
    CL_given = 0.6
    M_given = 0.7
    return CL_given, M_given


@app.cell
def _():
    # Create meshgrid for contour plots
    meshgrid_n = 101
    M_range = np.linspace(0, 1, meshgrid_n)
    CL_range = np.linspace(0, 0.9, meshgrid_n)
    M_grid, CL_grid = np.meshgrid(M_range, CL_range)

    # Evaluate E on the grid
    E_grid = CL_grid / CD(M_grid, CL_grid)
    return CL_range, E_grid, M_range


@app.cell
def _(CL_given, CL_opt_M, CL_range, E_grid, M_given, M_opt_CL, M_range):
    fig_simple = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"Fixed C<sub>L</sub> = {CL_given}", f"Fixed M = {M_given}"),
    )

    # Left plot: Fixed CL
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
            colorbar=dict(title="E (-)", x=0.45),
            showscale=True,
        ),
        row=1,
        col=1,
    )

    # Constraint line
    fig_simple.add_trace(
        go.Scatter(
            x=M_range,
            y=[CL_given] * len(M_range),
            mode="lines",
            line=dict(color="red", width=3, dash="dash"),
            name="Constraint",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Optimal point
    fig_simple.add_trace(
        go.Scatter(
            x=[M_opt_CL],
            y=[CL_given],
            mode="markers",
            marker=dict(color="yellow", size=12, symbol="star"),
            name="Optimum",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Right plot: Fixed M
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
            colorbar=dict(title="E (-)", x=1.02),
            showscale=True,
        ),
        row=1,
        col=2,
    )

    # Constraint line
    fig_simple.add_trace(
        go.Scatter(
            x=[M_given] * len(CL_range),
            y=CL_range,
            mode="lines",
            line=dict(color="red", width=3, dash="dash"),
            name="Constraint",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Optimal point
    fig_simple.add_trace(
        go.Scatter(
            x=[M_given],
            y=[CL_opt_M],
            mode="markers",
            marker=dict(color="yellow", size=12, symbol="star"),
            name="Optimum",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig_simple.update_xaxes(title_text="M (-)", row=1, col=1)
    fig_simple.update_yaxes(title_text="C<sub>L</sub> (-)", row=1, col=1)
    fig_simple.update_xaxes(title_text="M (-)", row=1, col=2)
    fig_simple.update_yaxes(title_text="C<sub>L</sub> (-)", row=1, col=2)

    fig_simple.update_layout(
        title_text="Simple Equality Constraints",
        title_x=0.5,
        height=500,
    )

    mo.output.clear()
    fig_simple
    return


@app.cell
def _(CL_given, CL_range, M_given, M_range):
    # Slices of E along constraints
    # For fixed CL
    E_fixed_CL = CL_given / CD(M_range, CL_given)
    M_opt_CL = M_range[np.argmax(E_fixed_CL)]
    E_max_CL = np.max(E_fixed_CL)

    # For fixed M
    E_fixed_M = CL_range / CD(M_given, CL_range)
    CL_opt_M = CL_range[np.argmax(E_fixed_M)]
    E_max_M = np.max(E_fixed_M)
    return CL_opt_M, E_fixed_CL, E_fixed_M, E_max_CL, E_max_M, M_opt_CL


@app.cell
def _(CL_given, CL_opt_M, E_max_CL, E_max_M, M_given, M_opt_CL):
    mo.md(f"""
    **Results:**

    - **Fixed $C_L = {CL_given}$**: Optimal Mach number $M^* = {M_opt_CL:.3f}$, with $E_{{\\mathrm{{max}}}} = {E_max_CL:.2f}$
    - **Fixed $M = {M_given}$**: Optimal lift coefficient $C_L^* = {CL_opt_M:.3f}$, with $E_{{\\mathrm{{max}}}} = {E_max_M:.2f}$
    """)
    return


@app.cell
def _(
    CL_given,
    CL_opt_M,
    CL_range,
    E_fixed_CL,
    E_fixed_M,
    E_max_CL,
    E_max_M,
    M_given,
    M_opt_CL,
    M_range,
):
    fig_slices = make_subplots(rows=1, cols=2)

    # Left: E vs M for fixed CL
    fig_slices.add_trace(
        go.Scatter(
            x=M_range,
            y=E_fixed_CL,
            mode="lines",
            line=dict(color="blue", width=2),
            name=f"E(M, C<sub>L</sub>={CL_given})",
        ),
        row=1,
        col=1,
    )

    fig_slices.add_trace(
        go.Scatter(
            x=[M_opt_CL],
            y=[E_max_CL],
            mode="markers",
            marker=dict(color="yellow", size=12, symbol="star"),
            name="Maximum",
        ),
        row=1,
        col=1,
    )

    # Right: E vs CL for fixed M
    fig_slices.add_trace(
        go.Scatter(
            x=CL_range,
            y=E_fixed_M,
            mode="lines",
            line=dict(color="green", width=2),
            name=f"E(M={M_given}, C<sub>L</sub>)",
            showlegend=True,
        ),
        row=1,
        col=2,
    )

    fig_slices.add_trace(
        go.Scatter(
            x=[CL_opt_M],
            y=[E_max_M],
            mode="markers",
            marker=dict(color="yellow", size=12, symbol="star"),
            name="Maximum",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig_slices.update_xaxes(title_text="M (-)", row=1, col=1)
    fig_slices.update_yaxes(title_text="E (-)", row=1, col=1)
    fig_slices.update_xaxes(title_text="C<sub>L</sub> (-)", row=1, col=2)
    fig_slices.update_yaxes(title_text="E (-)", row=1, col=2)

    fig_slices.update_layout(
        title_text="Objective Function Along Constraint Lines",
        title_x=0.5,
        height=400,
    )

    mo.output.clear()
    fig_slices
    return


@app.cell
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


@app.cell
def _():
    mo.md(r"""
    ## Complex Equality Constraint: Vertical Equilibrium

    Now let's consider a more realistic and complex constraint: the vertical equilibrium equation for steady level flight, according to which the lift force must equal the weight of the aircraft.

    $$ L = \frac{1}{2} \rho V^2 S C_L = W $$

    This can be rearranged in terms of Mach number as

    $$ C_L M^2 = \frac{2W}{\gamma p S }  \quad \text{or equivalently} \quad \gamma p S C_L M^2 - 2W = 0 $$
    """)
    return


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
def _():
    mo.md(r"""
    TODO: verify and check the following charts (removing the static results). Optionally, discuss that the value of the constraint depends on other flight parameters like weight and altitude (foreshadow what happens in the following notebooks) and add sliders accordingly. For convenience, you can write the constraint by isolating the $W/p$ term so that the constraint curve moves up if weight or altitude increase, and moves down if weight or altitude decrease.
    """)
    return


@app.cell
def _():
    # Example parameters for vertical equilibrium
    # Assume sea level standard atmosphere
    rho = 1.225  # kg/m^3
    gamma = 1.4
    R = 287  # J/(kg·K)
    T = 288.15  # K
    S = 100  # m^2 (example wing area)
    W = 50000  # N (example weight)

    A_eq = 0.5 * rho * gamma * R * T * S
    B_eq = W

    # Generate constraint curve
    M_constraint = np.linspace(0.3, 1.0, 100)
    CL_constraint = B_eq / (A_eq * M_constraint**2)

    # Filter to keep only valid CL values
    valid_idx = CL_constraint <= 0.9
    M_constraint = M_constraint[valid_idx]
    CL_constraint = CL_constraint[valid_idx]

    # Evaluate E along constraint
    E_constraint = CL_constraint / CD(M_constraint, CL_constraint)

    # Find optimum along constraint
    opt_idx = np.argmax(E_constraint)
    M_opt_eq = M_constraint[opt_idx]
    CL_opt_eq = CL_constraint[opt_idx]
    E_opt_eq = E_constraint[opt_idx]
    return (
        CL_constraint,
        CL_opt_eq,
        E_constraint,
        E_opt_eq,
        M_constraint,
        M_opt_eq,
    )


@app.cell
def _(
    CL_constraint,
    CL_opt_eq,
    CL_range,
    E_grid,
    E_opt_eq,
    M_constraint,
    M_opt_eq,
    M_range,
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
            colorbar=dict(title="E (-)"),
            showscale=True,
        )
    )

    # Constraint curve
    fig_equilibrium.add_trace(
        go.Scatter(
            x=M_constraint,
            y=CL_constraint,
            mode="lines",
            line=dict(color="red", width=3, dash="dash"),
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

    fig_equilibrium.update_xaxes(title_text="M (-)")
    fig_equilibrium.update_yaxes(title_text="C<sub>L</sub> (-)")

    fig_equilibrium.update_layout(
        title_text="Constrained Optimization: Vertical Equilibrium",
        title_x=0.5,
        height=500,
    )

    mo.output.clear()
    fig_equilibrium
    return


@app.cell
def _(CL_opt_eq, E_opt_eq, M_opt_eq):
    mo.md(f"""
    **Result:**

    For the vertical equilibrium constraint with the given parameters, the optimal condition is:
    - Mach number: $M^* = {M_opt_eq:.3f}$
    - Lift coefficient: $C_L^* = {CL_opt_eq:.3f}$
    - Maximum aerodynamic efficiency: $E_{{\\mathrm{{max}}}} = {E_opt_eq:.2f}$

    Note how the optimal point lies on the constraint curve at the location where a contour line of $E$ is tangent to the curve.
    """)
    return


@app.cell
def _(E_constraint, E_opt_eq, M_constraint, M_opt_eq):
    fig_E_constraint = go.Figure()

    fig_E_constraint.add_trace(
        go.Scatter(
            x=M_constraint,
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

    fig_E_constraint.update_xaxes(title_text="M (-)")
    fig_E_constraint.update_yaxes(title_text="E (-)")

    fig_E_constraint.update_layout(
        title_text="Aerodynamic Efficiency Along Equilibrium Constraint",
        title_x=0.5,
        height=400,
    )

    mo.output.clear()
    fig_E_constraint
    return


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
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
