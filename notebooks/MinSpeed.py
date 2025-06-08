import marimo

__generated_with = "0.13.15"
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
    mo.md(r"""# Minimum airspeed""")
    return


@app.cell
def _():
    mo.md(r"""## Unconstrained optimization problem""")
    return


@app.cell
def _():
    mo.callout(
        mo.md(
            r"""
        Find the minimum airspeed by changing the lift coefficient and throttle within certain limits:

    $$
    \begin{aligned}
        \min_{C_L, \delta_T} 
        & \quad V \\
        % \text{subject to} 
        % & \quad \bm{c}_\mathrm{eq}(\bm{x},\bm{u}; \bm{p}) = 0 \\
        % & \quad \bm{c_\mathrm{ineq}}(\bm{x},\bm{u}; \bm{p}) \le 0 \\
        \text{for } 
        & \quad C_L \in [0, C_{L_\mathrm{max}}] \\
        & \quad \delta_T \in [0, 1]
    \end{aligned}
    $$
        """
        )
    ).center().style({"text-align": "center"})
    return


@app.cell
def _():
    mo.md(
        r"""
    This problem is ill posed, and it does not make sense to solve it.

    There is no functional relation between the objective function $V$ and the controls $C_L, \delta_T$.
    In other words, there is no equation that specifies how $V$ can change with respect to the controls.
    It does not make sense to optimize Flight Performance if the flight dynamics is not controlled.

    For example, the minimum airspeed achievable could be 0, if the aircraft is standing still on the runway.
    It could even be negative, if someone is pushing the aircraft back, or there is tailwind.

    A relation must be introduced with constraint equatios, starting from the EoMS.
    These will define the problem properly.
    """
    )
    return


@app.cell
def _():
    mo.md(r"""- [ ] Plot a 2D chart with CL on x axis and dT on y axis, and a 3D chart with also V on Z axis (with nothing plotted on it). There is a selection menu for only one aircraft at a time, which is useless (but that's the point). Two sliders allow to pick a value of Cl and dT. The chart shows only the one point in the domain corresponding to the chosen values.""")
    return


@app.cell
def _():
    mo.md(r"""## Constrained optimization problem""")
    return


@app.cell
def _():
    mo.callout(
        mo.md(r"""
        Find the minimum airspeed that can be maintained in Steady Level Flight by changing the lift coefficient and throttle within certain limits

    $$
    \begin{aligned}
        \min_{C_L, \delta_T} 
        & \quad V \\
        \text{subject to} 
        & \quad c_1^\mathrm{eq} = L-W = \frac{1}{2}\rho V^2 S C_L - W = 0 \\
        & \quad c_2^\mathrm{eq} = T-D = \delta_T T_a(V,h) - \frac{1}{2} \rho V^2 S (C_{D_0}+K C_L^2) =0 \\
        \text{for } 
        & \quad C_L \in [0, C_{L_\mathrm{max}}] \\
        & \quad \delta_T \in [0, 1]
    \end{aligned}
    $$
        """)
    ).center().style({"text-align": "center"})
    return


@app.cell
def _():
    mo.md(
        r"""
    The introduction of the constraints for vertical ($c_1^\mathrm{eq}$) and horizontal equilibrium ($c_2^\mathrm{eq}$) restricts the scope to only a certain type of optimal speeds we are looking for. 

    The constraint equations introduce a functional dependency between the objective function and the controls.
    We are going to use them to reformulate the problem in order to analyse its properties.

    Before that, we notice that the expression of $c_2^\mathrm{eq}$ depends on the type of powertrain of the aircraft, and therefore we must proceed diffently for each powertrain architecture.

    1. [Simplified Jet -  Monotonicity Analysis](/?file=MinSpeed_Jet_MonoAn.py)
    1. [Simplified Jet -  Karush-Kuhn-Tucker Analyis](/?file=MinSpeed_Jet_KKT.py)
    1. [Simplified Piston-Prop -  Monotonicity Analysis](/?file=MinSpeed_Prop_MonoAn.py)
    1. [Simplified Piston-Prop -  Karush-Kuhn-Tucker Analysis](/?file=MinSpeed_Prop_KKT.py)
    """
    )
    return


@app.cell
def _():
    _defaults.nav_footer(
        "AerodynamicEfficiency.py", "Aerodynamic Efficiency", "", ""
    )
    return


if __name__ == "__main__":
    app.run()
