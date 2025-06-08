import marimo

__generated_with = "0.13.8"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import _defaults

    _defaults.set_plotly_template()


@app.cell
def _():
    _defaults.set_sidebar()
    return


@app.cell
def _():
    mo.md(r"""# Problem formulation""")
    return


@app.cell
def _():
    mo.md(
        r"""
    During flight, the aircraft can be commanded to achieve the best performance made available by the capabilities of its aero-propulsive systems, or that is allowed by regulations, safety, or other contraints.

    This means making appropriate use of the available flight controls $\bm{u}$ in order to maximize or minimize a certain _performance metric_ $J$, which depends on the mission phase and/or chosen flight strategy, while complying with constraints $\bm{c}$ introduced by different sources

    The FPAO problem is here formalized using either of the two following mathematical notations:
    """
    )
    return


@app.cell
def _():
    equations_min = mo.md(r"""
        $$
        \begin{aligned}
            \min_{\bm{u}} 
            & \quad J(\bm{x},\bm{u}; \bm{p}) \\
            \text{subject to} 
            & \quad \bm{c}_\mathrm{eq}(\bm{x},\bm{u}; \bm{p}) = 0 \\
            & \quad \bm{c_\mathrm{ineq}}(\bm{x},\bm{u}; \bm{p}) \le 0 \\
            \text{for } 
            & \quad \bm{u} \in [\bm{u}_\mathrm{lb}, \bm{u}_\mathrm{ub}]
        \end{aligned}
        $$
        """)

    equations_max = mo.md(r"""
        $$
        \begin{aligned}
            \max_{\bm{u}} 
            & \quad J(\bm{x},\bm{u}; \bm{p}) \\
            \text{subject to} 
            & \quad \bm{c}_\mathrm{eq}(\bm{x},\bm{u}; \bm{p}) = 0 \\
            & \quad \bm{c_\mathrm{ineq}}(\bm{x},\bm{u}; \bm{p}) \le 0 \\
            \text{for } 
            & \quad \bm{u} \in [\bm{u}_\mathrm{lb}, \bm{u}_\mathrm{ub}]
        \end{aligned}
        $$
        """)

    tabs = mo.ui.tabs(
        {
            r"$J$": mo.md(
                r"""$J$ is the objective function, which we want to minimize or maximize.              
                It is a function of the state variables $\bm{x}$ and the control variables $\bm{u}$.              
                In FPAO, it can take multiple forms depending on the mission phase, and flight strategy.  
                It is also referred to as the _performance metric_.
                """
            ),
            r"$\bm{u}$": mo.md(
                r"""$\bm{u}$ is the vector of decision variables, also called inputs or controls.              
                These are the variables that we can manipulate to find the optimal values of the objective function $J$.  
                They appear as independent variables in the model equations.              
                The evolution of the states $\bm{x}$ and of other parameters depend on them."""
            ),
            r"$\bm{x}$": mo.md(
                r"""$\bm{x}$ is the vector of state variables, also called states.  
                They are the minimum set of variables to characterize the dynamic system at a certain point in time.  
                This is called the state of the system."""
            ),
            r"$\bm{p}$": mo.md(
                r"""$\bm{p}$ is the vector of parameters.  
                These characterize the dynamic system, but are not altered within the optimization process.  
                They are constant for a given optimization, but their value could change for different optimization problems (different constraints or objective functions, for example).
                The discuss of optimization results as a function of the parameters is referred to as "parametric optimization". """
            ),
            r"$\bm{c}_\mathrm{eq}$": mo.md(
                r"""$\bm{c}_\mathrm{eq}$ are the vectors of equality constraints.  
                These are the equalities that must be satisfied for the solution to be valid.  
                They can represent physical laws, or other relationships between the variables.  
                In general, they are implicit equations in the states and controls, but usually they take the form of very simple explicit relationship."""
            ),
            r"$\bm{c}_\mathrm{ineq}$": mo.md(
                r"""$\bm{c}_\mathrm{ineq}$ are the vectors of inequality constraints.  
                These are the inequalities that must be satisfied for the solution to be valid.  
                They can represent physical laws, or other relationships between the variables.  
                In general, they are implicit equations in the states and controls, but usually they take the form of very simple explicit relationship."""
            ),
            r"$\bm{u}_\mathrm{lb}$": mo.md(
                r"""$\bm{u}_\mathrm{lb}$ are the lower bounds on the control variables $\bm{u}$.    
                They define the lower limit of the feasible space for the optimization problem."""
            ),
            r"$\bm{u}_\mathrm{ub}$": mo.md(
                r"""$\bm{u}_\mathrm{ub}$ are the upper bounds on the control variables $\bm{u}$.  
                They define the upper limit of the feasible space for the optimization problem."""
            ),
        }
    )

    mo.vstack(
        [
            mo.hstack(
                [
                    equations_min,
                    equations_max,
                ],
                justify="center",
                align="center",
                widths="equal",
                gap=0,
            )
            .style({"width": "50%"})
            .center(),
            tabs.center().style({"justify": "stretch", "text-align": "center"}),
        ],
        justify="center",
        align="stretch",
    )
    return


@app.cell
def _():
    mo.md(
        r"""For a _point performance_ optimization problem, the objective of FPAO is to find the _feasible_ value of the controls $\bm{u}$ that optimize a given flight performance metric $J$ while complying to the physical and operational constraints $\bm{c}_\mathrm{eq}$ and $\bm{c}_\mathrm{ineq}$.
    
        The solution of an FPAO problem should be analized as a function of the aircraft design and flight parameters $\bm{p}$."""
    ).callout(kind="success").style(
        {"width": "75%", "text-align": "center"}
    ).center()
    return


@app.cell
def _():
    mo.md(
        r"""
    Common examples of performance metrics in FPAO are: aerodynamic efficiency, instantaneous rate of climb, instantaneous angle of climb, instantaneous turn radius, instantaneous load factor, time required to turn at constant speed.

    They are analyzed in the following notebooks, after a discussion on controls and contraints
    """
    )
    return


@app.cell
def _():
    _defaults.nav_footer(
        "AircraftCustom.py.py",
        "Custom Aircraft Models",
        "FlightConsrtaints.py",
        "Flight Consrtaints",
    )
    return


if __name__ == "__main__":
    app.run()
