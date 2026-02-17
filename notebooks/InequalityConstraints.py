import marimo

__generated_with = "0.19.11"
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
    from core import atmos
    from scipy.optimize import minimize

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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Inequality Constraints

    In the previous notebooks, we explored optimization with simple bounds and equality constraints. In practical flight performance problems, we often also encounter inequality constraints that define feasible regions rather than exact relationships between decision variable and other parameters.

    Inequality constraints express requirements like "the speed must not exceed a limit" or "the load factor must remain below a safety threshold". Unlike equality constraints that force variables to lie on a specific curve or surface, inequality constraints allow variables to lie anywhere within a permitted region.

    In this notebook, we introduce the Karush-Kuhn-Tucker (KKT) conditions, which generalize the method of Lagrangian multipliers by  providing necessary conditions for optimality in the presence of inequality constraints.

    We'll apply these conditions to our familiar aerodynamic efficiency maximization problem, hence we continue using the same drag coefficient model:

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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## From bounds to inequalities

    In our previous optimization problems, we used simple bounds on the decision variables:

    $$
    \begin{aligned}
        M &\in [0, 1] \\
        C_L &\in [0, 0.9]
    \end{aligned}
    $$

    These bounds can be reformulated as inequality constraints in standard form:

    $$
    \begin{aligned}
        h_1(M, C_L) &= -M \le 0 \\
        h_2(M, C_L) &= M - 1 \le 0 \\
        h_3(M, C_L) &= -C_L \le 0 \\
        h_4(M, C_L) &= C_L - 0.9 \le 0
    \end{aligned}
    $$

    Note the convention: inequality constraints are written in the form $h_i(\bm{x}) \le 0$.

    This reformulation may seem pedantic for simple bounds, but it becomes essential when we introduce more complex constraints that couple multiple variables.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Additional Operational Constraint: Maximum Operating Speed

    In practical applications, aircraft are subject to many other physical and operational constraints, which may be expressed as  equalities or inequalities.

    As an example, we will look at the case in which the aircraft may also be subject to a maximum operating speed $V_\mathrm{MO}$ beyond which structural loads, flutter, or other aerodynamic phenomena impede the correct functioning of the system.
    The value of this airspeed is typically specified in the aircraft operating manual.

    In light of the relationship $V = M a$, where $a$ is the speed of sound at the flight altitude, an inequality constraint on $V_\mathrm{MO}$ can be expressed as function of the Mach number as:

    $$ V \le V_\mathrm{MO}
    \quad \Longleftrightarrow \quad
    Ma \le V_\mathrm{MO}
    \quad \Longleftrightarrow \quad
    M \le \frac{V_\mathrm{MO}}{a} $$

    In standard notation:

    $$ h_5(M) = M - \frac{V_\mathrm{MO}}{a(h)} \le 0 $$

    For our **purely didactic** example, we'll use $V_\mathrm{MO} = 171$ m/s at cruise altitude (approximately 11000 m). The speed of sound at this altitude is approximately 295 m/s, giving an equivalent $M_\mathrm{MO} \approx 0.58$.
    This is very low to be realistic, but enables us to make comparisons without changing the objective function.

    We will explore the optimization problem both _with_ and _without_ this constraint to illustrate the difference between interior and boundary optima.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Problem Formulation

    The optimization problem can now be stated as:

    $$
    \begin{aligned}
        \max_{M, C_L}
        & \quad E(M, C_L) = \frac{C_L}{C_D(M, C_L)} \\
        \text{subject to}
        & \quad h_1 = -M \le 0 \\
        & \quad h_2 = M - 1 \le 0 \\
        & \quad h_3 = -C_L \le 0 \\
        & \quad h_4 = C_L - 0.9 \le 0 \\
        & \quad h_5 = M - \frac{V_\mathrm{MO}}{a} \le 0 \quad \text{(optional)}
    \end{aligned}
    $$

    It is also customary, in optimization theory, to express every optimization problem as a minimization problem.
    This will allows us to formulate the KKT conditions in a standard way, which we can reuse for any optimization problem formulated in this canonic form.

    In this case, the optimization problem can be formulated as a minimization one by negating the objective function:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $$
    \begin{aligned}
        \min_{M, C_L} 
        & \quad \mathcal{J}(M, C_L) = -E(M, C_L) = -\frac{C_L}{C_D(M, C_L)} \\
        \text{subject to} 
        & \quad h_1 = -M \le 0 \\
        & \quad h_2 = M - 1 \le 0 \\
        & \quad h_3 = -C_L \le 0 \\
        & \quad h_4 = C_L - 0.9 \le 0 \\
        & \quad h_5 = M - \frac{V_\mathrm{MO}}{a} \le 0 \quad \text{(optional)}
    \end{aligned}
    $$
    """).callout().center().style({"text-align": "center"})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Karush-Kuhn-Tucker (KKT) Conditions

    The Karush-Kuhn-Tucker (KKT) conditions provide necessary conditions for a point $\bm{x}^* = (M^*, C_L^*)$ to be a local optimum of a constrained optimization problem with inequality constraints.

    For the general problem

    $$
    \begin{aligned}
        \min_{\bm{x}}
        & \quad \mathcal{J}(\bm{x}) \\
        \text{subject to}
        & \quad h_i(\bm{x}) \le 0, \quad i = 1, \ldots, m
    \end{aligned}
    $$

    and similarly to the method of Lagrange multipliers for equality constraints, we construct a Lagrangian function by combining the objective with the constraints using KKT multipliers $\mu_i$:

    $$ \mathcal{L}(\bm{x}, \bm{\mu}) = \mathcal{J}(\bm{x}) + \sum_{i=1}^{m} \mu_i h_i(\bm{x}) $$

    The key difference from equality constraints is that the multipliers $\mu_i$ must be **non-negative** ($\mu_i \ge 0$).
    This reflects the asymmetric nature of inequality constraints: they only restrict the feasible region to one "side" of the region that they represent.

    Before looking at the mathematical form of the KKT conditions, let's try to understand why they take their specific form by considering again a geometry interpretation of the problem.
    Only the following cases can occur:

    1. **Interior optimum** — If the optimum lies strictly inside the feasible region, the solution does not verify the equality part of the constraints. In other words, the constraints don't "feel" the solution. In this case:
        - The gradient of the objective vanishes at the optimum: $\nabla \mathcal{J}(\bm{x}^*) = \bm{0}$
        - $h_i(\bm{x}^*) < 0 \quad \forall i$ (constraints are inactive)
        - The multipliers are non-active inequality constraints are zero, indicating that the optimum is not affected by the constraints: $\mu_i^* = 0$

    2. **Boundary optimum** — If the optimum lies on the boundary of the inequality constraints, these behave effectively as equality constraints. In this case:
        - The gradient of the objecitve function $\nabla \mathcal{J}(\bm{x}^*)$ at the optimum cannot point into the infeasible region (otherwise we could improve by moving away from the boundary); it is therefore a combination of the gradients of the constraints $\nabla h_i(\bm{x}^*)$, and points towards the feasible region
        - $h_i(\bm{x}^*) = 0 \quad \forall i (contraints are active)
        - The multipliers of the active inequality constraints are positive to ensure the correct feasible "side": $\mu_i^* > 0$

    3. **Mixed situation** — In the vast majority of cases, some inequality constraints are active while others are inactive.
        - $h_i(\bm{x}^*) = 0$ with $\mu_i^* > 0$ for some $i$
        - $h_i(\bm{x}^*) < 0$ with $\mu_i^* = 0$ otherwise

    These geometric observations lead to four formal conditions that must hold simultaneously at an optimum.
    They therefore can be stated as a necessary condition for a constrained optimum in the presence of inequality constraints,and are formalized in the following theorem:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r""" 
    *Let $\mathcal{J}: \mathbb{R}^n \to \mathbb{R}$ and $h_i: \mathbb{R}^n \to \mathbb{R}$ for $i = 1, 2, \ldots, m$ be continuously differentiable functions. Consider the optimization problem*

    $$
    \begin{aligned}
        \min_{\bm{x} \in \mathbb{R}^n}
        & \quad \mathcal{J}(\bm{x}) \\
        \text{subject to}
        & \quad h_i(\bm{x}) \le 0, \quad i = 1, 2, \ldots, m
    \end{aligned}
    $$

    *If $\bm{x}^* \in \mathbb{R}^n$ is a local minimum and the gradients of active constraints are linearly independent, then there exist Lagrange multipliers $\mu_1^*, \mu_2^*, \ldots, \mu_m^* \in \mathbb{R}$ such that the following conditions are satisfied:*

    1. **Stationarity:** $\displaystyle \nabla \mathcal{J}(\bm{x}^*) + \sum_{i=1}^{m} \mu_i^* \nabla h_i(\bm{x}^*) = \bm{0}$

    2. **Primal Feasibility:** $h_i(\bm{x}^*) \le 0, \quad \forall i = 1, \ldots, m$

    3. **Dual Feasibility:** $\mu_i^* \ge 0, \quad \forall i = 1, \ldots, m$

    4. **Complementary Slackness:** $\mu_i^* h_i(\bm{x}^*) = 0, \quad \forall i = 1, \ldots, m$

    *These four conditions are collectively known as the Karush-Kuhn-Tucker (KKT) conditions.*
    """).callout()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The four KKT conditions are disccussed as follows:

    ### 1. Stationarity

    The gradient of the Lagrangian function must vanish at the optimum:

    $$ \nabla \mathcal{J}(\bm{x}^*) + \sum_{i=1}^{m} \mu_i^* \nabla h_i(\bm{x}^*) = \bm{0} $$

    This condition states that, at the optimum, the gradient of the objective function must be a linear combination of the gradients of the active constraints.
    The active constraints will contribute to pushing the gradient in the right direction because their multipliers are positive.
    The inactive constraints will not play any role because their multipliers are zero.
    If all inequality constraints are inactive, the gradient of the objective function must vanish itself, which corresponds to the necessary condition for an interior optimum.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 2. Primal Feasibility

    All inequality constraints must be satisfied:

    $$ h_i(\bm{x}^*) \le 0, \quad i = 1, \ldots, m $$

    This simply ensures that the candidate solution lies within the feasible region.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 3. Dual Feasibility

    All KKT multipliers must be non-negative:

    $$ \mu_i^* \ge 0, \quad i = 1, \ldots, m $$

    This condition is specific to inequality constraints. It ensures that constraints can only "push" the solution in the direction that minimizes the objective.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 4. Complementary Slackness

    For each constraint, either the constraint is active or its multiplier is zero:

    $$ \mu_i^* h_i(\bm{x}^*) = 0, \quad i = 1, \ldots, m $$

    and therefore:
    - If $\mu_i^* > 0$, then $h_i(\bm{x}^*) = 0$: the constraint is active
    - If $h_i(\bm{x}^*) < 0$, then $\mu_i^* = 0$: the constraint is inactive
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Application to Aerodynamic Efficiency

    Let's apply the KKT conditions to our problem expressed in standard form as a minimization problem:

    $$ \mathcal{J}(M, C_L) = -E(M, C_L) = -\frac{C_L}{C_D(M, C_L)} $$

    The expression of the gradient of the objective function is reported below.
    Note that this is the same as the one derived in the [BivariateOptimization.py](?file=BivariateOptimization.py) notebook.

    $$ \nabla \mathcal{J} = \left( \frac{\partial \mathcal{J}}{\partial M}, \frac{\partial \mathcal{J}}{\partial C_L} \right) = \left( \frac{C_L}{C_D^2} \frac{\partial C_D}{\partial M}, -\frac{1}{C_D} + \frac{C_L}{C_D^2} \frac{\partial C_D}{\partial C_L} \right) $$

    In the [EqualityConstraints.py](?file=EqualityConstraints.py) notebook we saw that using the the substitution method to remove equality constraints by injecting them into the objective function can make such derivation much more complicated.
    In this case, it is completely impossible to substitute expressions from the constraints because we are dealing with inequalities.
    To obtain equalities from the constraints we would have to explore all possible combinations of inequalities being active and inactive, reducing the problem to a set of optimization problems with equality constraints.
    This systematic process is enacted in an elegant way by the KKT conditions.

    The constraint gradients for the simple bounds are:

    $$
    \begin{aligned}
        \nabla h_1 &= (-1, 0) \\
        \nabla h_2 &= (1, 0) \\
        \nabla h_3 &= (0, -1) \\
        \nabla h_4 &= (0, 1)
    \end{aligned}
    $$

    For simplicity, we focus on these four fundamental constraints in our analytical discussion.
    The additional operational constraint $h_5$ (maximum speed limit) will be included in the numerical solution that follows.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The stationarity condition is:

    $$
    \left\{
    \begin{aligned}
        \frac{C_L^*}{(C_D^*)^2} \frac{\partial C_D}{\partial M}\bigg|_{(M^*, C_L^*)} &= \mu_1^* - \mu_2^* - \mu_5^* \\
        -\frac{1}{C_D^*} + \frac{C_L^*}{(C_D^*)^2} \frac{\partial C_D}{\partial C_L}\bigg|_{(M^*, C_L^*)} &= \mu_3^* - \mu_4^*
    \end{aligned}
    \right.
    $$

    The primal feasibility conditions are:

    $$
    \left\{
    \begin{aligned}
        h_1(M^*, C_L^*) &= -M^* \le 0 \\
        h_2(M^*, C_L^*) &= M^* - 1 \le 0 \\
        h_3(M^*, C_L^*) &= -C_L^* \le 0 \\
        h_4(M^*, C_L^*) &= C_L^* - 0.9 \le 0 \\
        h_5(M^*, C_L^*) &= M^* - \frac{V_\mathrm{MO}}{a} \le 0
    \end{aligned}
    \right.
    $$

    The dual feasibility conditions are:

    $$
    \mu_i^* \ge 0, \quad i = 1, \ldots, 5
    $$

    The complementary slackness conditions are:

    $$
    \left\{
    \begin{aligned}
        \mu_1^* h_1(M^*, C_L^*) &= 0 \\
        \mu_2^* h_2(M^*, C_L^*) &= 0 \\
        \mu_3^* h_3(M^*, C_L^*) &= 0 \\
        \mu_4^* h_4(M^*, C_L^*) &= 0 \\
        \mu_5^* h_5(M^*, C_L^*) &= 0
    \end{aligned}
    \right.
    $$

    To solve these conditions analytically, we would need to:
    1. Enumerate all possible combinations of active/inactive constraints (32 cases for 5 constraints)
    2. For each case, simplify the equations depending on which multipliers are zero and which constraints are active
    3. Solve the system of resulting equations
    4. Check if the solution satisfies primal feasibility, dual feasibility, and complementary slackness
    5. Compare objective values to find the global optimum

    Often we can use physical and engineering insight to exclude the analysis of some of these cases a priori, and therefore streamline the solution process.

    For now, given the rather complex nature of the $C_D(M, C_L)$ model, it would be tedious to do the entire derivation and analysis explicitly here. Instead, we will just show the results obtained from a numerical optimization algorithm, which systematically searches for points satisfying all four KKT conditions.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Numerical Solutions

    We now solve the two variants of the optimization problem to illustrate the concepts of interior and boundary optima on an active inequality constraint.
    """)
    return


@app.cell
def _():
    # Define flight conditions
    h_cruise = 11000.0  # Cruise altitude [m]
    V_MO = 171.0  # Maximum operating speed [m/s] (chosen to give M ≈ 0.58)

    # Get atmospheric properties at cruise altitude
    a_cruise = atmos.a(h_cruise)  # Speed of sound at cruise altitude [m/s]

    # Convert V_MO to equivalent M_MO for constraints
    M_MO = V_MO / a_cruise
    return M_MO, V_MO, a_cruise


@app.cell
def _():
    # Define the objective function (negative of E for minimization)
    def objective(x):
        M, CL = x
        return -CL / CD(M, CL)


    # Initial guess
    x0 = np.array([0.5, 0.5])
    return objective, x0


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Inactive contraints, interior optimum

    $$
    \begin{aligned}
        \min_{M, C_L}
        & \quad \mathcal{J}(M, C_L) = -\frac{C_L}{C_D(M, C_L)} \\
        \text{subject to}
        & \quad h_1 = -M \le 0 \\
        & \quad h_2 = M - 1 \le 0 \\
        & \quad h_3 = -C_L \le 0 \\
        & \quad h_4 = C_L - 0.9 \le 0
    \end{aligned}
    $$

    We expect an interior optimum where all constraints are inactive and the gradient vanishes.
    This is entirely equivalent to the problem solved in the [BivariateOptimization.py](?file=BivariateOptimization.py) notebook.
    """)
    return


@app.cell
def _():
    tol = 1e-6
    return (tol,)


@app.cell
def _(objective, tol, x0):
    # Solve WITHOUT constraint h5 (interior optimum)
    def constraints_ineq_simple(x):
        M, CL = x
        return np.array(
            [
                -M,  # h1: M >= 0
                M - 1.0,  # h2: M <= 1
                -CL,  # h3: CL >= 0
                CL - 0.9,  # h4: CL <= 0.9
            ]
        )


    result_simple = minimize(
        objective,
        x0,
        method="SLSQP",
        constraints={"type": "ineq", "fun": lambda x: -constraints_ineq_simple(x)},
        options={"disp": False, "ftol": 1e-9},
    )

    M_opt_simple = result_simple.x[0]
    CL_opt_simple = result_simple.x[1]
    E_opt_simple = -result_simple.fun

    # Evaluate constraints at optimum
    h_opt_simple = constraints_ineq_simple(result_simple.x)

    # Determine which constraints are active
    active_constraints_simple = np.abs(h_opt_simple) < tol
    return (
        CL_opt_simple,
        E_opt_simple,
        M_opt_simple,
        active_constraints_simple,
        h_opt_simple,
    )


@app.cell
def _(
    CL_opt_simple,
    E_opt_simple,
    M_opt_simple,
    active_constraints_simple,
    h_opt_simple,
):
    constraint_names_simple = [
        r"$h_1: M \ge 0$",
        r"$h_2: M \le 1$",
        r"$h_3: C_L \ge 0$",
        r"$h_4: C_L \le 0.9$",
    ]

    results_text_simple = f"""
    **Results (Case 1):**

    - Optimal Mach number: $M^* = {M_opt_simple:.4f}$
    - Optimal lift coefficient: $C_L^* = {CL_opt_simple:.4f}$
    - Maximum aerodynamic efficiency: $E_{{\\mathrm{{max}}}} = {E_opt_simple:.4f}$

    **Constraint Status:**
    """

    for i_simple, (name_simple, h_val_simple, is_active_simple) in enumerate(
        zip(constraint_names_simple, h_opt_simple, active_constraints_simple)
    ):
        status_simple = "**ACTIVE**" if is_active_simple else "inactive"
        results_text_simple += f"\n- {name_simple}: $h_{i_simple + 1} = {h_val_simple:.6f}$ — {status_simple}"

    results_text_simple += (
        "\n\n**All constraints are inactive** — the optimum lies strictly in the interior of the feasible region."
    )

    mo.md(results_text_simple)
    return


@app.cell
def _():
    # Create meshgrid for visualization
    meshgrid_n = 201
    M_range = np.linspace(0, 1, meshgrid_n)
    CL_range = np.linspace(0, 0.9, meshgrid_n)
    M_grid, CL_grid = np.meshgrid(M_range, CL_range)

    # Evaluate E on the grid
    E_grid = CL_grid / CD(M_grid, CL_grid)
    return CL_grid, CL_range, E_grid, M_grid, M_range


@app.cell
def _(CL_opt_simple, CL_range, E_grid, M_opt_simple, M_range):
    # Visualization for Case 1
    fig_case1 = go.Figure()

    fig_case1.add_trace(
        go.Contour(
            x=M_range,
            y=CL_range,
            z=E_grid,
            colorscale="viridis",
            contours=dict(showlines=True, coloring="heatmap"),
                    colorbar=dict(title="E (-)", len=0.95, x=1.02, y=0.44, yanchor="middle"),

            hovertemplate="M: %{x:.3f}<br>C<sub>L</sub>: %{y:.3f}<br>E: %{z:.2f}<extra></extra>",
        )
    )

    fig_case1.add_trace(
        go.Scatter(
            x=[M_opt_simple],
            y=[CL_opt_simple],
            mode="markers+text",
            marker=dict(
                color="cyan",
                size=15,
                symbol="circle",
                line=dict(color="black", width=2),
            ),
            text=["Optimum"],
            textposition="top center",
            textfont=dict(size=12, color="cyan"),
            name=f"M*={M_opt_simple:.3f}, C<sub>L</sub>*={CL_opt_simple:.3f}",
            showlegend=True,
        )
    )

    fig_case1.update_xaxes(title_text="M (-)", range=[0, 1])
    fig_case1.update_yaxes(title_text="C<sub>L</sub> (-)", range=[0, 0.9])
    fig_case1.update_layout(
        title_x=0.5,
        height=500,
        hovermode="closest",
    )

    mo.output.clear()
    fig_case1
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Active constraint, boundary optimum

    $$
    \begin{aligned}
        \min_{M, C_L}
        & \quad \mathcal{J}(M, C_L) = -\frac{C_L}{C_D(M, C_L)} \\
        \text{subject to}
        & \quad h_1 = -M \le 0 \\
        & \quad h_2 = M - 1 \le 0 \\
        & \quad h_3 = -C_L \le 0 \\
        & \quad h_4 = C_L - 0.9 \le 0 \\
        & \quad h_5 = M - \frac{V_\mathrm{MO}}{a} \le 0
    \end{aligned}
    $$

    We expect a boundary optimum where constraint $h_5$ becomes active and restricts the solution.
    """)
    return


@app.cell
def _(M_MO, objective, tol, x0):
    # Solve WITH constraint h5 (boundary optimum)
    def constraints_ineq_full(x):
        M, CL = x
        return np.array(
            [
                -M,  # h1: M >= 0
                M - 1.0,  # h2: M <= 1
                -CL,  # h3: CL >= 0
                CL - 0.9,  # h4: CL <= 0.9
                M - M_MO,  # h5: M <= M_MO
            ]
        )


    result_full = minimize(
        objective,
        x0,
        method="SLSQP",
        constraints={"type": "ineq", "fun": lambda x: -constraints_ineq_full(x)},
        options={"disp": False, "ftol": 1e-9},
    )

    M_opt_full = result_full.x[0]
    CL_opt_full = result_full.x[1]
    E_opt_full = -result_full.fun

    # Evaluate constraints at optimum
    h_opt_full = constraints_ineq_full(result_full.x)

    # Determine which constraints are active
    active_constraints_full = np.abs(h_opt_full) < tol
    return (
        CL_opt_full,
        E_opt_full,
        M_opt_full,
        active_constraints_full,
        h_opt_full,
    )


@app.cell
def _(
    CL_opt_full,
    E_opt_full,
    M_MO,
    M_opt_full,
    V_MO,
    active_constraints_full,
    h_opt_full,
):
    constraint_names_full = [
        r"$h_1: M \ge 0$",
        r"$h_2: M \le 1$",
        r"$h_3: C_L \ge 0$",
        r"$h_4: C_L \le 0.9$",
        f"$h_5: V \\le {V_MO:.0f}$ m/s $(M \\le {M_MO:.3f})$",
    ]

    results_text_full = f"""
    **Results (Case 2):**

    - Optimal Mach number: $M^* = {M_opt_full:.4f}$
    - Optimal lift coefficient: $C_L^* = {CL_opt_full:.4f}$
    - Maximum aerodynamic efficiency: $E_{{\\mathrm{{max}}}} = {E_opt_full:.4f}$

    **Constraint Status:**
    """

    for i, (name, h_val, is_active) in enumerate(zip(constraint_names_full, h_opt_full, active_constraints_full)):
        status = "**ACTIVE (binding)**" if is_active else "inactive"
        results_text_full += f"\n- {name}: $h_{i + 1} = {h_val:.6f}$ — {status}"

    results_text_full += "\n\n**Constraint $h_5$ is active** — the optimum lies on the boundary imposed by the maximum operating speed."

    mo.md(results_text_full)
    return


@app.cell
def _(CL_opt_full, CL_range, E_grid, M_MO, M_grid, M_opt_full, M_range, V_MO):
    # Visualization for Case 2
    fig_case2 = go.Figure()

    # Show contour plot over entire domain
    fig_case2.add_trace(
        go.Contour(
            x=M_range,
            y=CL_range,
            z=E_grid,
            colorscale="viridis",
            contours=dict(
                showlines=True,
                coloring="heatmap",
                start=np.nanmin(E_grid),
                end=np.nanmax(E_grid),
                size=0.5,
            ),
            colorbar=dict(title="E (-)", len=0.9, x=1.02, y=0.43, yanchor="middle"),
            hovertemplate="M: %{x:.3f}<br>C<sub>L</sub>: %{y:.3f}<br>E: %{z:.2f}<extra></extra>",
            opacity=0.4,  # Make entire domain semi-transparent
        )
    )

    # Overlay feasible region with full brightness
    E_grid_feasible = np.where(M_grid <= M_MO, E_grid, np.nan)
    fig_case2.add_trace(
        go.Contour(
            x=M_range,
            y=CL_range,
            z=E_grid_feasible,
            colorscale="viridis",
            contours=dict(
                showlines=True,
                coloring="heatmap",
                start=np.nanmin(E_grid),
                end=np.nanmax(E_grid),
                size=0.5,
            ),
            showscale=False,
            hovertemplate="M: %{x:.3f}<br>C<sub>L</sub>: %{y:.3f}<br>E: %{z:.2f}<extra></extra>",
            opacity=1.0,  # Full brightness for feasible region
        )
    )

    # Add constraint h5 boundary line
    fig_case2.add_trace(
        go.Scatter(
            x=[M_MO, M_MO],
            y=[0, 0.9],
            mode="lines",
            line=dict(color="red", width=4),
            name=f"h<sub>5</sub>: V = {V_MO:.0f} m/s",
            showlegend=True,
        )
    )

    # Highlight the constrained optimum
    fig_case2.add_trace(
        go.Scatter(
            x=[M_opt_full],
            y=[CL_opt_full],
            mode="markers+text",
            marker=dict(
                color="cyan",
                size=15,
                symbol="circle",
                line=dict(color="black", width=2),
            ),
            text=["Optimum"],
            textposition="top center",
            textfont=dict(size=12, color="cyan"),
            name=f"M*={M_opt_full:.3f}, C<sub>L</sub>*={CL_opt_full:.3f}",
            showlegend=True,
        )
    )

    fig_case2.update_xaxes(title_text="M (-)", range=[0, 1])
    fig_case2.update_yaxes(title_text="C<sub>L</sub> (-)", range=[0, 0.9])
    fig_case2.update_layout(
        title_x=0.5,
        height=500,
        hovermode="closest",
    )

    mo.output.clear()
    fig_case2
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Comparison

    In the first case, all inequality constraints are inactive, all multipliers are null ($\mu_i^* = 0$) and the gradient of the objective function vanishes

    In the second case, the $h_5$ constraint is active, its multiplier is strictly positive $\mu_5^* > 0$ and the gradient of the objective function is balanced by the gradient of the constraint itself.
    The optimum lies once again at the tangency point between the active inequality constraint and the contour line of the objective function where it lies.

    From a flight performance point of view, the operational speed limit forces the aircraft to operate at lower efficiency than it could achieve without this restriction.

    You can use the slider and checkbox below to explore how the maximum operating speed constraint affects the optimal solution.
    """)
    return


@app.cell
def _():
    V_MO_slider = mo.ui.slider(
        start=150.0,
        stop=300.0,
        step=5.0,
        value=171.0,
        label=r"$V_\mathrm{MO}$ (m/s)",
        show_value=True,
    )

    include_h5_checkbox = mo.ui.checkbox(value=True, label="Include $h_5$ constraint")

    mo.hstack([V_MO_slider, include_h5_checkbox], justify="start")
    return V_MO_slider, include_h5_checkbox


@app.cell
def _(
    CL_grid,
    E_grid,
    M_grid,
    V_MO_slider,
    a_cruise,
    include_h5_checkbox,
    objective,
    x0,
):
    # Solve optimization for current slider value
    V_MO_interactive = V_MO_slider.value
    M_MO_interactive = V_MO_interactive / a_cruise
    include_h5 = include_h5_checkbox.value


    def constraints_ineq_interactive(x):
        M, CL = x
        if include_h5:
            return np.array(
                [
                    -M,
                    M - 1.0,
                    -CL,
                    CL - 0.9,
                    M - M_MO_interactive,
                ]
            )
        else:
            return np.array(
                [
                    -M,
                    M - 1.0,
                    -CL,
                    CL - 0.9,
                ]
            )


    result_interactive = minimize(
        objective,
        x0,
        method="SLSQP",
        constraints={"type": "ineq", "fun": lambda x: -constraints_ineq_interactive(x)},
        options={"disp": False, "ftol": 1e-9},
    )

    M_opt_interactive = result_interactive.x[0]
    CL_opt_interactive = result_interactive.x[1]
    E_opt_interactive = -result_interactive.fun

    # Update feasibility mask
    if include_h5:
        feasible_mask_interactive = (
            (M_grid >= 0) & (M_grid <= 1) & (CL_grid >= 0) & (CL_grid <= 0.9) & (M_grid <= M_MO_interactive)
        )
    else:
        feasible_mask_interactive = (M_grid >= 0) & (M_grid <= 1) & (CL_grid >= 0) & (CL_grid <= 0.9)

    E_grid_feasible_interactive = np.where(feasible_mask_interactive, E_grid, np.nan)
    return (
        CL_opt_interactive,
        E_grid_feasible_interactive,
        E_opt_interactive,
        M_MO_interactive,
        M_opt_interactive,
        V_MO_interactive,
        include_h5,
    )


@app.cell
def _(
    CL_opt_interactive,
    CL_range,
    E_grid_feasible_interactive,
    E_opt_interactive,
    M_MO_interactive,
    M_opt_interactive,
    M_range,
    V_MO_interactive,
    include_h5,
):
    fig_interactive = go.Figure()

    # Add contour plot
    fig_interactive.add_trace(
        go.Contour(
            x=M_range,
            y=CL_range,
            z=E_grid_feasible_interactive,
            colorscale="viridis",
            contours=dict(
                showlines=True,
                coloring="heatmap",
            ),
            colorbar=dict(title="E (-)", len=0.8, x=1.02, y=0.37, yanchor="middle"),

            hovertemplate="M: %{x:.3f}<br>C<sub>L</sub>: %{y:.3f}<br>E: %{z:.2f}<extra></extra>",
        )
    )

    # Add V_MO constraint line only if h5 is included
    if include_h5:
        fig_interactive.add_trace(
            go.Scatter(
                x=[M_MO_interactive, M_MO_interactive],
                y=[0, 0.9],
                mode="lines",
                line=dict(color="red", width=3, dash="dash"),
                name=f"h<sub>5</sub>: V = {V_MO_interactive:.0f} m/s (M = {M_MO_interactive:.3f})",
            )
        )

    # Add optimal point
    point_color = "yellow" if include_h5 else "cyan"
    point_symbol = "star" if include_h5 else "circle"
    optimum_type = "Boundary" if include_h5 else "Interior"

    fig_interactive.add_trace(
        go.Scatter(
            x=[M_opt_interactive],
            y=[CL_opt_interactive],
            mode="markers",
            marker=dict(
                color=point_color,
                size=15,
                symbol=point_symbol,
                line=dict(color="black", width=2),
            ),
            name=f"{optimum_type} Optimum<br>M={M_opt_interactive:.3f}<br>C<sub>L</sub>={CL_opt_interactive:.3f}<br>E={E_opt_interactive:.2f}",
        )
    )

    fig_interactive.update_xaxes(title_text="M (-)", range=[0, 1])
    fig_interactive.update_yaxes(title_text="C<sub>L</sub> (-)", range=[0, 0.9])

    title_suffix = (
        f"with h<sub>5</sub> (V<sub>MO</sub> = {V_MO_interactive:.0f} m/s)"
        if include_h5
        else "without h<sub>5</sub>"
    )
    fig_interactive.update_layout(
        title_text=f"Interactive Feasible Region ({title_suffix})",
        title_x=0.5,
        height=600,
        hovermode="closest",
    )

    mo.output.clear()
    fig_interactive
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Conclusion

    It may seem trivial to have developed this entire KKT framework for an additional inequality constraint $h_5$ that is such a simple function of $M$ alone, represented by a vertical line in the $(M, C_L)$ plane.

    The real power of the methodology becomes evident when dealing with more complex constraints that couple multiple variables.
    For example, aircraft are typically subject to a **buffet boundary** that limits the maximum usable lift coefficient as a function of Mach number.
    It can for example be expressed like this:

    $$C_{L,\text{buffet}}(M) = C_{L,\text{max}} \left(\frac{1 - M}{1 - M_\mathrm{ref}}\right)^p$$

    where $p \approx 0.5$ is an empirical exponent.
    This constraint would appear as a curved boundary in the $(M, C_L)$ space, and the KKT conditions would elegantly handle its interaction with the objective function regardless of its complexity.


    In subsequent notebooks, we'll build on these concepts to formulate and solve complete flight performance optimization problems, incorporating both inequality constraints to represent the bounds of the feasible region and equality constraints to represent the condition of straight and level flight at constant speed.

    We are going to manipulate and analyse those problems for different objective functions, and discuss the flight conditions (in terms of aircraft weight and altitude) that determine if the optimum lies in the interior or the boundary of the domain.
    In order to make analytical derivations and discuss the role of important flight paramters, we are going to keep the aero-propulsive models simple.
    """)
    return


@app.cell
def _():
    _defaults.nav_footer(
        "EqualityConstraints.py",
        "Equality Constraints",
        "MinDrag.py",
        "Minimum Drag",
    )
    return


if __name__ == "__main__":
    app.run()
