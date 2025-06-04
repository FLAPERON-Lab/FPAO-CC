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
    mo.md(r"""# Aerodynamic Efficiency""")
    return


@app.cell
def _():
    mo.md(
        r"""
    ## Unconstrained optimization

    $$
    \begin{aligned}
        \max_{C_L} 
        & \quad E=\frac{C_L}{C_D}=\frac{C_L}{C_{D_0}+KC_L^2} \\
        % \text{subject to} 
        % & \quad \bm{c}_\mathrm{eq}(\bm{x},\bm{u}; \bm{p}) = 0 \\
        % & \quad \bm{c_\mathrm{ineq}}(\bm{x},\bm{u}; \bm{p}) \le 0 \\
        \text{for } 
        & \quad C_L \in [0, C_{L_\mathrm{max}}]
    \end{aligned}
    $$

    Solution is a constant that depends only on design and aerodyanmic parameters,w hich for us are constant.

    Its existence is guaranteed by Weierstrass theorem for a contiunoues function in a compact set. The necessary condition to find a stationary point is given by equating its gradient with respect to the decision variable to zero. A sufficient condition to prove its actually a maximum is to show that the Hessian is negative at the stationary point.
    The function is actually concave in the domain

    - [ ] Plot E vs CL, with its maximum
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    ## Constrained optimization: steady symmetric horizontal flight (cruise flight)

    $$
    \begin{aligned}
        \max_{C_L} 
        & \quad J(C_L)=E(C_L)=\frac{C_L}{C_D}=\frac{C_L}{C_{D_0}+KC_L^2} \\
        \text{subject to} 
        & \quad \bm{c}_\mathrm{eq1}(C_L) = L-W = 0 \quad \Leftrightarrow \quad \frac{1}{2}\rho V^2 S C_L -W = 0 \\
        & \quad \bm{c}_\mathrm{eq2}(\delta_T,C_L) = T-D = 0 \quad \Leftrightarrow \quad \delta_T T_a - \frac{1}{2}\rho V^2 S (C_{D_0}+KC_L^2)  = 0 \\
        % & \quad \bm{c_\mathrm{ineq}}(\bm{x},\bm{u}; \bm{p}) \le 0 \\
        \text{for } 
        & \quad C_L \in [0, C_{L_\mathrm{max}}] \\
        & \quad \delta_T \in [0, 1]
    \end{aligned}
    $$

    The second constraint is useless because J is only a function of CL, so we can remove it
    It can be used to calculate the thrust and throttle a posteriori.

    The choice of CL for maximum aerodynamic efficiency also depends on other flight parameters: altitude, speed and aircraft weight. 
    Need to specify the parameters of interest

    ### Pilot perspective: assigned W, h and V

    The CL is uniquely identified by the constraints, you have no more decision variables to optimize the aerodynamic efficiency.
    It is up to the operators to make sure to prescribe a pair of h and V which corresponds to CLopt for a given weight.
    The values of W, h and V have to correspond to a feasible CL within the bounds.

    If the W is higher, then either h is ... or V is ... or both.

    - [ ] Plot E vs CL and c1 vs Cl for everything fixed. add sliders for h, W and V.

    ### Pilot perspective: assigned W, h
    The pilot can change both CL and V to maximize E at a given altitude.

    max_{CL,V} E 

    Raises the question: what are the bounds for V? -> Vmin and Vmax
    - [ ] Start another notebook before called "Flight Envelope". Present this problem as a min and max problem at fixed altitude

    #### Solution method 1: direct substitution and change of variables CL->V

    #### Solution method 2: Lagrangian function with equality constraints


    ### Pilot perspective: assigned W, V
    The pilot can change both CL and h to maximize E at a given speed.

    max_{CL,h} E 

    Raises the question: what are the bounds for h? -> theoretical ceiling
    - [ ] In the flight envelope notebook, present this as a max problem 

    ### Pilot perspective: assigned W
    The pilot can change CL, V and h to maximize E at a given speed.

    max_{CL,h,V} E 


    ### Scheduler perspective: assigned V and h
    The scheduler can decide how much to load a (Cargo) aircraft to perform best at the assigned cruise altitude and speed.

    max_{CL,W} E
    """
    )
    return


@app.cell
def _():
    _defaults.nav_footer("FlightControls.py", "Flight Controls", "MinimumSpeed.py", "Minimum Speed")
    return


if __name__ == "__main__":
    app.run()
