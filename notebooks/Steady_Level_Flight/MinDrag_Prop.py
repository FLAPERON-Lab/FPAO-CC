# SPDX-FileCopyrightText: 2026 Carmine Varriale <C.varriale@tudelft.nl>
# SPDX-FileCopyrightText: 2026 Federico Angioni <F.angioni@student.tudelft.nl>
# SPDX-FileCopyrightText: 2026 Maarten van Hoven <M.B.vanHoven@tudelft.nl>
#
# SPDX-License-Identifier: Apache-2.0
import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path.cwd()))

    # Initialization code that runs before all other cells
    import marimo as mo

    # Import dependencies
    from core import _defaults
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np
    from core import atmos
    from core.aircraft import (
        available_aircrafts,
        AircraftBase,
        ModelSimplifiedProp,
        OptimumCondition,
    )
    from scipy.optimize import root_scalar

    from core import plot_utils
    # from core.plot_utils import OptimumGridView

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
    # Define constants, this cell runs once and is not dependent in any way on any interactive element (not even the ac database)
    data = available_aircrafts(data_dir, ac_type="Propeller")
    ac_table = plot_utils.InteractiveElements.init_table(data)

    labels = {"Title": "Drag (N)", "Symbol": "D", "hover_name": "D<sub>min</sub>"}
    return ac_table, data


@app.cell
def _(ac_table, data):
    # Define constants dependent on the ac database. This runs every time another aircraft is selected
    if ac_table.value is not None and ac_table.value.any().any():
        active_selection = ac_table.value.iloc[0]
    else:
        active_selection = data.iloc[0]

    aircraft = AircraftBase(active_selection)

    label = "Min Speed (m/s)"

    initialControls = plot_utils.InteractiveElements(aircraft, initial=True)
    initialModel = ModelSimplifiedProp(aircraft)

    initial_mass_slider = initialControls.mass_slider
    initial_altitude_slider = initialControls.altitude_slider
    initial_CL_slider = initialControls.CL_slider
    initial_dT_slider = initialControls.dT_slider

    initial_mass_stack, initial_variables_stack = initialControls.init_layout(
        initial_mass_slider, initial_altitude_slider
    )
    return (
        aircraft,
        initialControls,
        initialModel,
        initial_CL_slider,
        initial_altitude_slider,
        initial_dT_slider,
        initial_mass_slider,
        initial_variables_stack,
    )


@app.cell
def _(initialControls, initialModel, initial_mass_slider):
    W_selected_initial = initialControls.sense_mass(initial_mass_slider)

    initialModel.update_mass_dependency(W_selected_initial)
    return (W_selected_initial,)


@app.cell
def _(initialControls, initialModel, initial_altitude_slider):
    h_selected_initial = initialControls.sense_altitude(initial_altitude_slider)

    initialModel.update_altitude_dependency(h_selected_initial)
    return (h_selected_initial,)


@app.cell
def _(W_selected_initial, h_selected_initial, initialModel):
    initialModel.update_context(W_selected_initial, h_selected_initial)
    return


@app.cell
def _(W_selected_initial, h_selected_initial, initialModel, initial_CL_slider):
    _ = h_selected_initial, W_selected_initial

    initialSurface = np.broadcast_to(
        initialModel.drag_curve[np.newaxis, :],
        (plot_utils.meshgrid_n, plot_utils.meshgrid_n),
    )

    selected_value = initialModel.compute_drag(
        W_selected_initial, initial_CL_slider.value
    )

    plot_options_initial = {
        "surface": initialSurface,
        "title": "Minimum drag",
        "axes": {"z": {"label": "D (N)"}},
    }
    return plot_options_initial, selected_value


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Minimum drag: simplified piston propeller aircraft

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
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## KKT formulation

    As shown in the simplified jet case, we express $V$ from $c_1^\mathrm{eq}$ and substitute it out of the entire problem to eliminate it. The KKT formulation thus becomes:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $$
    \begin{aligned}
    \min_{C_L, \delta_T}
    & \quad D = W \frac{C_{D_0} + K C_L^2}{C_L} = \frac{W}{E} \\
    \text{subject to}
    & \quad g_1 = \frac{T}{W} - \frac{1}{E}  = \delta_T P_{a0}\sigma^\beta\sqrt{\frac{\rho S C_L}{2W}} - W\frac{C_{D_0} +K C_L^2}{C_L} = 0 \\
    & \quad h_1 = C_L - C_{L_\mathrm{max}} \le 0 \\
    & \quad h_2 = -C_L \le 0 \\
    & \quad h_3 = \delta_T - 1 \le 0 \\
    & \quad h_4 = -\delta_T \le 0 \\
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Below you can see the graph of the domain $0 \leq C_L \leq C_{L_{\mathrm{max}}}$ and $0 \leq \delta_T \leq 1$, with the surface $D$ and the contraint $g_1$ in red. Choose a simplified jet aircraft of your liking in the database below.
    """)
    return


@app.cell(hide_code=True)
def _(ac_table):
    # Database cell (1)
    ac_table
    return


@app.cell(hide_code=True)
def _(
    initialModel,
    initial_CL_slider,
    initial_dT_slider,
    initial_variables_stack,
    plot_options_initial,
    selected_value,
):
    mo.md(f"""
    Here you can modify the control variables to understand how it affects the design: {mo.vstack([mo.hstack([initial_dT_slider, initial_CL_slider]), initial_variables_stack, initialModel.plot_initial(plot_options_initial, [initial_CL_slider.value, initial_dT_slider.value, selected_value]).figure])}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Lagrangian function and KKT conditions

    The Lagrangian function combines the objective function with equality constraints using Lagrange multipliers ($\lambda_i$) and inequality constraints using KKT multipliers ($\mu_j$).

    $$
    \begin{aligned}
    \mathcal{L}(C_L, \delta_T, \lambda_1, \mu_1, \mu_2, \mu_3, \mu_4) =
    \quad W\frac{C_{D_0} + K C_L^2}{C_L}
    & + \\
    & + \lambda_1 \left[\delta_T P_{a0}\sigma^\beta \sqrt{\frac{\rho S C_L}{2W}} - W\frac{C_{D_0} +K C_L^2}{C_L}\right] + \\
    & + \mu_1 (C_L - C_{L_\mathrm{max}}) + \\
    & + \mu_2 (-C_L) + \\
    & + \mu_3 (\delta_T - 1) +\\
    & + \mu_4 (-\delta_T)
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The multipliers $\lambda_1, \mu_1, \mu_2, \mu_3, \mu_4$ have to meet the following conditions for an optimal solution of the optimization problem $(C_L^*, \delta_T^*)$ to exist.

    **A. Stationarity conditions($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \begin{aligned}\frac{\partial \mathcal{L}}{\partial C_L} & = W \frac{K C_L^2 - C_{D_0}}{C_L^2} + \lambda_1 \left( \frac{1}{2} \delta_T P_{a0}\sigma^\beta \sqrt{\frac{\rho S}{2WC_L}} - W \frac{K C_L^2 - C_{D_0}}{C_L^2} \right) + \mu_1 - \mu_2 \\
    & = W\frac{K C_L^2 - C_{D_0}}{C_L^2} (1 -\lambda_1) +  \frac{1}{2} \lambda_1\delta_T P_{a0}\sigma^\beta \sqrt{\frac{\rho S}{2WC_L}} +\mu_1 - \mu_2 = 0 \end{aligned}$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1 P_{a0} \sigma^\beta \sqrt{\frac{\rho S C_L}{2W}}+\mu_3-\mu_4 = 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $\displaystyle \delta_T P_{a0}\sigma^\beta\sqrt{\frac{\rho S C_L}{2W}} - W\frac{C_{D_0} +K C_L^2}{C_L} = 0$
    4.  $C_L - C_{L_\mathrm{max}} \le 0$
    5.  $-C_L \le 0$
    6.  $\delta_T - 1 \le 0$
    7.  $-\delta_T \le 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    8.  $\mu_1, \mu_2, \mu_3, \mu_4 \ge 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraints have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

    9.  $\mu_1 (C_L - C_{L_\mathrm{max}}) = 0$
    10. $\mu_2 (-C_L) = 0$
    11. $\mu_3 (\delta_T - 1) = 0$
    12. $\mu_4 (-\delta_T) = 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    It is evident that $\mu_2$ and $\mu_4$ can never be active, as we would have an unfeasible situation ($C_L = \delta_T = 0$). In other words, strictly for aircraft flight: $C_L \gt 0$ and $\delta_T \gt 0$. Therefore, we can simplify the analysis by setting these two KKT multipliers to zero:

    $$
    \begin{aligned}
    \mu_2 = \mu_4 = 0
    \end{aligned}
    $$

    We can now rewrite the new conditions to simplify the problem. We will refer to these simplified conditions for the entire notebook.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Simplified conditions**

    1. $\displaystyle W\frac{K C_L^2 - C_{D_0}}{C_L^2} (1 -\lambda_1) +  \frac{1}{2} \lambda_1\delta_T P_{a0}\sigma^\beta \sqrt{\frac{\rho S}{2WC_L}} +\mu_1 = 0$
    2. $\displaystyle \lambda_1 P_{a0} \sigma^\beta \sqrt{\frac{\rho S C_L}{2W}}+\mu_3 = 0$
    3. $\displaystyle \delta_T P_{a0}\sigma^\beta\sqrt{\frac{\rho S C_L}{2W}} - W\frac{C_{D_0} +K C_L^2}{C_L} = 0$
    4. $C_L - C_{L_\mathrm{max}} \le 0$
    5. $\delta_T - 1 \le 0$
    6. $\mu_1, \mu_3 \ge 0$
    7. $\mu_1 (C_L - C_{L_\mathrm{max}}) = 0$
    8. $\mu_3 (\delta_T - 1) = 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## KKT analysis

    We can now systematically examine the conditions where various inequality constraints are active or inactive.
    """)
    return


@app.cell
def _(aircraft):
    analysisControls = plot_utils.InteractiveElements(aircraft)

    mass_slider_analysis = analysisControls.mass_slider
    altitude_slider_analysis = analysisControls.altitude_slider

    mass_stack_analysis, variables_stack_analysis = analysisControls.init_layout(
        mass_slider_analysis, altitude_slider_analysis
    )

    analysisModel = ModelSimplifiedProp(aircraft)
    tab_view, title_keys = analysisControls.init_analysis_tabs()
    tab = analysisControls.tab
    return (
        altitude_slider_analysis,
        analysisControls,
        analysisModel,
        mass_slider_analysis,
        mass_stack_analysis,
        tab,
        tab_view,
        title_keys,
        variables_stack_analysis,
    )


@app.cell
def _(analysisControls, analysisModel, mass_slider_analysis):
    W_selected_analysis = analysisControls.sense_mass(mass_slider_analysis)

    analysisModel.update_mass_dependency(W_selected_analysis)
    return (W_selected_analysis,)


@app.cell
def _(altitude_slider_analysis, analysisControls, analysisModel):
    h_selected_analysis = analysisControls.sense_altitude(altitude_slider_analysis)

    analysisModel.update_altitude_dependency(h_selected_analysis)
    return (h_selected_analysis,)


@app.cell
def _(W_selected_analysis, analysisModel, h_selected_analysis):
    analysisModel.update_context(W_selected_analysis, h_selected_analysis)
    return


@app.cell
def _(tab_view):
    tab_view
    return


@app.cell
def _(W_selected_analysis, analysisModel, h_selected_analysis, tab):
    tab_value = tab.value

    _ = h_selected_analysis, W_selected_analysis

    surface = np.broadcast_to(
        analysisModel.drag_curve[np.newaxis, :],
        (plot_utils.meshgrid_n, plot_utils.meshgrid_n),
    )

    plot_options_analysis = {
        "surface": surface,
    }
    return plot_options_analysis, tab_value


@app.cell(hide_code=True)
def _(
    InteriorCondition,
    W_selected_analysis,
    analysisModel,
    h_selected_analysis,
    plot_options_analysis,
    tab_value,
    title_keys,
    variables_stack_analysis,
):
    if tab_value != title_keys[0]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""

    ### _Interior solutions_

    Assuming that that $0 < C_L < C_{L_\mathrm{max}}$ and $0 < \delta_T < 1$ is equivalent to consider all inequality constraints as inactive.

    Therefore: $\mu_1, \mu_3 =0$.

    From stationarity condition (2): $\lambda_1 = 0$.

    From stationarity condition (1), it is possible to obtain the value of $C_L^*$ for minimum drag.

    $$
    C_L^* = \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$

    Notice how the optimal $C_L^*$ has the **same value** for maximum aerodynamic efficiency, or maximum $C_L/C_D$, for
    $0\lt C_L \lt  C_{L_{max}}$ and $0 \lt \delta_T \lt 1$, as shown in [aerodynamic efficiency](/?file=Steady_Level_Flight/MinDrag.py).

    The corresponding airspeed is

    $$
    \displaystyle V^* = V_E = \sqrt{\frac{2}{\rho}\frac{W}{S}\frac{1}{C_{L_E}}} = \sqrt{\frac{2}{\rho}\frac{W}{S}}\sqrt[4]{\frac{K}{C_{D_0}}}
    $$

    The corresponding $\delta_T^*$ is found by solving the primal feasibility constraint (3) and using $C_L = C_L^*$.


    $$
    \delta_T^* = \frac{W}{E_\mathrm{max}}\frac{V_E}{P_{a0}\sigma^\beta} = \frac{W^{3/2}}{\sigma^{\beta+1/2}}\frac{1}{E_\mathrm{max}P_{a0}}\sqrt{\frac{2}{\rho_0 S C_{L_E}}}
    $$

    This value is compliant with the primal feasibility constraint (5) for:

    $$
    \delta_T^* < 1 \quad \Leftrightarrow
    \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}}
    \lt
    P_{a0}E_\mathrm{max}\sqrt{\frac{\rho_0 S C_{L_E}}{2}}
    $$

    The corresponding minimum drag is found by first computing $C_D^*=C_{D_E}$:

    $$
    C_D^* = C_{D_E} = 2C_{D_0} \\
    D_{\mathrm{min}}^* =  \frac{1}{2}\rho {V_E}^2 S C_{D_E}=
    2W\sqrt{KC_{D_0}}=\frac{W}{E_\mathrm{max}}
    $$

    This is the same result as in the simplified jet analysis!

    This concludes the analysis for the minimum drag of a simplified propeller aircraft in the domain's interior. Below is a summary of the optima:

    $$
    \boxed{C_L^* = \sqrt{\frac{C_{D_0}}{K}}}, \quad \boxed{\delta_T^*=\frac{W^{3/2}}{\sigma^{\beta+1/2}}\frac{1}{E_\mathrm{max}P_{a0}}\sqrt{\frac{2}{\rho_0 S C_{L_E}}}}, \quad \text{for} \quad C_L^* \lt C_{L_\mathrm{max}}\quad \text{and} \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}}
    \lt
    P_{a0}E_\mathrm{max}\sqrt{\frac{\rho_0 S C_{L_E}}{2}}
    $$

    With the optimal value for minimum drag:

    $$
    D_{\mathrm{min}}^* = 2W\sqrt{KC_{D_0}}=\frac{W}{E_\mathrm{max}}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.

    """),
            variables_stack_analysis,
            analysisModel.plot_grid(
                (
                    InteriorCondition(
                        W_selected_analysis, h_selected_analysis, analysisModel
                    ),
                ),
                plot_options_analysis,
            ).figure,
        ]
    ).callout()
    return


@app.cell
def _(aircraft):
    class InteriorCondition(OptimumCondition):
        def __init__(self, W, h, Model):
            velocity = Model.compute_velocity(W, h, Model.aircraft.CL_E)

            self.dTopt = W / Model.aircraft.E_max / (Model.compute_thrust(h, velocity))

            self.condition = (
                W
                < Model.compute_thrust(
                    Model.aircraft.h_array,
                    Model.compute_velocity(
                        W, Model.aircraft.h_array, Model.aircraft.CL_E
                    ),
                )
                * aircraft.E_max
            ) & (Model.aircraft.CL_E < Model.aircraft.CLmax)

            self.CLopt = self.CLopt_selected = Model.aircraft.CL_E

            self.compute_optimal(W, h, Model)

    return (InteriorCondition,)


@app.cell
def _(
    MaxliftCondition,
    W_selected_analysis,
    analysisModel,
    h_selected_analysis,
    plot_options_analysis,
    tab_value,
    title_keys,
    variables_stack_analysis,
):
    if tab_value != title_keys[1]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ###_Lift-limited minimum drag (stall boundary)_

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 > 0$

    $\delta_T < 1 \quad \Rightarrow \quad \mu_3 = 0$

    From stationarity condition (2): $\lambda_1 = 0$.

    From stationarity condition (1): $\displaystyle \mu_1 = W\frac{C_{D_0} - KC_{L_\mathrm{max}}^2}{C_{L_\mathrm{max}}^2} \gt 0$, which results in:

    $$
    C_{L_\mathrm{max}} < \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$

    This means the aircraft is able to achieve minimum drag at its maximum lift coefficient only if the maximum lift coefficient is lower than the one for maximum aerodynamic efficiency.
    In this case, the aircraft would stall at higher speeds than the one for maximum aerodynamic efficiency, and would therefore only be able to fly on the right branch of the drag performance diagram.

    The corresponding throttle is obtained from the primal feasibility constraint (3):

    $$
    \delta_T^* = \frac{W}{E_S}\frac{V_S}{P_{a0}\sigma^\beta} = \frac{W^{3/2}}{\sigma^{\beta+1/2}}\frac{1}{P_{a0}E_S}\sqrt{\frac{2}{\rho_0 S C_{L_\mathrm{max}}}}
    $$

    By setting $\delta_T^* \lt 1$ obtain the operational condition for which minimum drag can be achieved:


    $$
    \frac{W^{3/2}}{\sigma^{\beta+1/2}} \lt P_{a0}E_S \sqrt{\frac{\rho_0SC_{L_\mathrm{max}}}{2}}
    $$

    Finally, the value for the objective function $D$ can be found:

    $$
    D^*_\mathrm{min} = \frac{1}{2}\rho V^2 S C_D = \frac{1}{2}\rho \left(\frac{W}{S}\frac{2}{\rho}\frac{1}{C_{L_\mathrm{max}}}\right)S (C_{D_0} + KC_{L_\mathrm{max}}^2) =  \frac{W}{E_S}
    $$

    This concludes the analysis for the minimum drag of a simplified propeller aircraft in the lift-limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = C_{L_\mathrm{max}}}, \quad \boxed{\delta_T^*=\frac{W^{3/2}}{\sigma^{\beta+1/2}}\frac{1}{E_SP_{a0}}\sqrt{\frac{2}{\rho_0 S C_{L_\mathrm{max}}}}}, \quad \text{for} \quad C_L^* \lt \sqrt{\frac{C_{D_0}}{K}} \quad \text{and} \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}}\lt P_{a0}E_S\sqrt{\frac{\rho_0 S C_{L_\mathrm{max}}}{2}}
    $$

    With the optimal value for minimum drag:

    $$
    D_{\mathrm{min}}^* =\frac{W}{E_S}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.
    """),
            variables_stack_analysis,
            analysisModel.plot_grid(
                (
                    MaxliftCondition(
                        W_selected_analysis, h_selected_analysis, analysisModel
                    ),
                ),
                plot_options_analysis,
            ).figure,
        ]
    ).callout()
    return


@app.cell
def _(aircraft):
    class MaxliftCondition(OptimumCondition):
        def __init__(self, W, h, Model):
            velocity = Model.compute_velocity(W, h, Model.aircraft.CLmax)

            self.dTopt = W / Model.aircraft.E_S / (Model.compute_thrust(h, velocity))

            self.condition = (
                W
                < Model.compute_thrust(
                    Model.aircraft.h_array,
                    Model.compute_velocity(
                        W, Model.aircraft.h_array, Model.aircraft.CLmax
                    ),
                )
                * aircraft.E_S
            ) & (Model.aircraft.CLmax < Model.aircraft.CL_E)

            self.CLopt = self.CLopt_selected = Model.aircraft.CLmax

            self.compute_optimal(W, h, Model)

    return (MaxliftCondition,)


@app.cell
def _(
    MaxThrustCondition,
    W_selected_analysis,
    analysisModel,
    h_selected_analysis,
    plot_options_analysis,
    tab_value,
    title_keys,
    variables_stack_analysis,
):
    if tab_value != title_keys[2]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ###_Thrust-limited minimum drag_

    $C_L \lt C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 = 0$

    $\delta_T = 1 \quad \Rightarrow \quad \mu_3 \gt 0$

    From stationarity condition (2):

    $$
    \mu_3 = -\lambda_1 P_{a0}\sigma^\beta \sqrt{\frac{\rho S C_L}{2W}} \gt 0 \quad \Rightarrow \quad \lambda_1 \lt 0
    $$

    Thus, $\lambda_1 \lt 0$ and $(\lambda_1 - 1) \lt 0$. From stationarity condition (1):

    $$
    \frac{\lambda_1 - 1}{\lambda_1} = \frac{C_L^2}{KC_L^2 - C_{D_0}}\frac{P_{a0}\sigma^\beta}{2 W}\sqrt{\frac{\rho S}{2WC_L}} > 0 \quad \Rightarrow \quad
    KC_L^2-C_{D_0} \gt 0, \quad \Rightarrow \quad C_L\gt \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$

    This means that $C_L^*$ is bounded by both $C_{L_E}$ and $C_{L_\mathrm{max}}$:


    $$
    C_{L_E} \lt C_L^* \lt C_{L_\mathrm{max}}
    $$

    The actual value of $C_L^*$ can be found from the primal feasibility constraint (3), with $\delta_T^* = 1$.
    Unfortunately, solving this cannot be done analytically, and a closed-form expression for $C_L^*$ cannot be found.
    Thus we proceed to a numerical solution for $C_L^*$, which also yields the operational condition:

    $$
    \text{solve numerically for } C_L^* :\quad  P_{a0}\sigma^\beta \sqrt{\frac{\rho S C_L}{2W}} - W \frac{C_{D_0} + K C_L^2}{C_L} = 0
    $$

    Thus the results from the thrust-limited analysis yield:

    $$
    \boxed{\delta_T = 1}, \quad \boxed{C_L^* = \text{numerical}}, \quad {\frac{W^{3/2}}{\sigma^{\beta + 1/2}} \gt \text{numerical}}, \quad {C_{L_E}\lt C_L \lt C_{L_\mathrm{max}}}
    $$
    """),
            variables_stack_analysis,
            analysisModel.plot_grid(
                (
                    MaxThrustCondition(
                        W_selected_analysis, h_selected_analysis, analysisModel
                    ),
                ),
                plot_options_analysis,
            ).figure,
        ]
    ).callout()
    return


@app.cell
def _():
    def maxthrust_solver(W, h, Model):
        sigma = atmos.rhoratio(h)
        C1 = (
            Model.aircraft.Pa0
            * 1e3
            * sigma**Model.aircraft.beta
            * np.sqrt(atmos.rho(h) * Model.aircraft.S / (2 * W))
        )

        # define H(s) and its derivative
        def H(s):
            # H(s) = C1 * s^(3/2) - W * (CD0 + K * s^2)
            return C1 * s**1.5 - W * (Model.aircraft.CD0 + Model.aircraft.K * s**2)

        def dHds(s):
            # dH/ds = (3/2)*C1*s^(1/2) - 2*W*K*s
            return 1.5 * C1 * np.sqrt(s) - 2 * W * Model.aircraft.K * s

        return H, dHds

    class MaxThrustCondition(OptimumCondition):
        def __init__(self, W, h, Model):
            CL_maxthrust_star = []

            for hi in Model.aircraft.h_array:
                H, dHds = maxthrust_solver(W, hi, Model)

                sol = root_scalar(
                    H,
                    fprime=dHds,
                    x0=Model.aircraft.CL_E,
                    method="newton",
                    xtol=1e-12,
                    rtol=1e-12,
                    maxiter=1000,
                )

                s_root = sol.root

                CL_maxthrust_star.append(s_root)

            CL_maxthrust_star = np.array(CL_maxthrust_star)
            mask = (CL_maxthrust_star > Model.aircraft.CL_E) & (
                CL_maxthrust_star < Model.aircraft.CLmax
            )

            velocity = Model.compute_velocity(W, h, Model.aircraft.CLmax)

            self.dTopt = 1

            self.condition = mask

            self.CLopt = CL_maxthrust_star[mask]

            self.hopt_array = Model.aircraft.h_array[mask]

            # Safe single-point lookup
            h_idx = np.where(np.isclose(Model.aircraft.h_array, h))[0]

            if len(h_idx) > 0 and mask[h_idx[0]]:
                idx_in_masked = np.where(np.isclose(self.hopt_array, h))[0][0]
                self.CLopt_selected = self.CLopt[idx_in_masked]
            else:
                self.CLopt_selected = np.nan

            self.compute_optimal(W, h, Model, True)

    return (MaxThrustCondition,)


@app.cell
def _(
    W_selected_analysis,
    analysisModel,
    mass_stack_analysis,
    tab_value,
    title_keys,
):
    if tab_value != title_keys[3]:
        mo.stop(True)

    MaxLiftThrust = (MaxLiftThrustCondition(W_selected_analysis, analysisModel),)
    surface_MaxLiftThrust = np.broadcast_to(
        analysisModel.drag_curve[np.newaxis, :],
        (plot_utils.meshgrid_n, plot_utils.meshgrid_n),
    )

    plot_options_analysis_MaxLiftThrust = {
        "surface": surface_MaxLiftThrust,
    }

    mo.vstack(
        [
            mo.md(r"""
    _Thrust-lift limited minimum drag_


    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 \gt 0$

    $\delta_T = 1 \quad \Rightarrow \quad \mu_3 \gt 0$

    From stationarity condition (2) obtain:

    $$
    \mu_3 = -\lambda_1 P_{a0}\sigma^\beta \sqrt{\frac{\rho S C_{L_\mathrm{max}}}{2W}} \gt 0 \quad \Rightarrow \quad \lambda_1 \lt 0
    $$

    From stationarity condition (1):

    $$
    \mu_1 = W \frac{K C_{L_\mathrm{max}}^2 - C_{D_0}}{C_{L_\mathrm{max}}^2} (\lambda_1 - 1) - \frac{1}{2} \lambda_1 P_{a0}\sigma^\beta \sqrt{\frac{\rho S}{2 W C_{L_\mathrm{max}}}}   \gt 0
    $$

    Isolating $\lambda_1$ and simplifying results in:

    $$
    \lambda_1 > \displaystyle\frac{1}{1 - \displaystyle\frac{P_{a0}\sigma^{\beta+1/2}}{2W^{3/2}}\sqrt{\frac{\rho_0 S}{2}}\frac{C_{L_\mathrm{max}}^{3/2}}{K C_{L_\mathrm{max}}^2 - C_{D_0}}}
    % \left(-\frac{1}{2} A C_{L_\mathrm{max}} ^{-1/2} + W \frac{KC_{L_\mathrm{max}} ^2 - C_{D_0}}{C_{L_\mathrm{max}}^2} \right) - W \frac{KC_{L_\mathrm{max}} ^2 - C_{D_0}}{C_{L_\mathrm{max}}^2} \gt 0
    $$

    Remembering that it must be $\lambda_1<0$, the condition for this optimum to exist is found in terms of weight and altitude by imposing that the previous denominator is negative. We obtain:

    $$
    \frac{W^{3/2}}{\sigma^{\beta+1/2}} < \frac{P_{a0}}{2}\sqrt{\frac{\rho_0 S}{2}}\frac{C_{L_\mathrm{max}}^{3/2}}{K C_{L_\mathrm{max}}^2 - C_{D_0}}
    $$

    where we note that it must be:

    $$
    K C_{L_\mathrm{max}}^2 - C_{D_0} > 0 \quad \Rightarrow \quad C_{L_\mathrm{max}} > \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$


    From the primal feasibility condition (3):

    $$
    \frac{W^{3/2}}{\sigma^{\beta+1/2}} = P_{a0}\sqrt{\frac{\rho_0 S}{2}}\frac{C_{L_\mathrm{max}}^{3/2}}{C_{D_0} + K C_{L_\mathrm{max}}^2}
    $$

    Substituting this into the previous inequality yields:

    $$
    \frac{3C_{D_0}-KC_{L_\mathrm{max}}^2}{KC_{L_\mathrm{max}}^2 - C_{D_0}} > 0
    $$

    which is then verified for:

    $$
    C_{L_\mathrm{max}} < \sqrt{\frac{3 C_{D_0}}{K}} = C_{L_P}
    $$

    Thus, taking the intersection of the conditions on $C_L^*$ obtain:

    $$
    \sqrt{\frac{C_{D_0}}{K}} \lt C_L^* \lt \sqrt{\frac{3C_{D_0}}{K}}
    $$

    Finally, the value of the objective function $D$ can be calculated:

    $$
    D_{\mathrm{min}}^* = \frac{W}{E_S}
    $$

    This concludes the analysis for the minimum drag of a simplified propeller aircraft in the thrust-lift limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = C_{L_\mathrm{max}}}, \quad \boxed{\delta_T^*=1}, \quad \text{for} \quad \sqrt{\frac{C_{D_0}}{K}} \lt C_L^* \lt \sqrt{\frac{3C_{D_0}}{K}} \quad \text{and} \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}}= P_{a0}E_S\sqrt{\frac{\rho_0 S C_{L_\mathrm{max}}}{2}}
    $$

    With the optimal value for minimum drag:

    $$
    D_{\mathrm{min}}^* =\frac{W}{E_S}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.

    """),
            mass_stack_analysis,
            analysisModel.plot_grid(
                MaxLiftThrust, plot_options_analysis_MaxLiftThrust
            ).figure,
        ]
    ).callout()
    return


@app.class_definition
class MaxLiftThrustCondition(OptimumCondition):
    def __init__(self, W, Model, modifyModel=True):
        sigma_opt = (
            W**1.5
            / (Model.aircraft.Pa0 * 1e3)
            / Model.aircraft.E_S
            / (np.sqrt(atmos.rho0 * Model.aircraft.S * Model.aircraft.CLmax / 2))
        ) ** (1 / (Model.aircraft.beta + 0.5))
        h_optimum = (
            atmos.altitude(sigma_opt) if sigma_opt > atmos.rhoratio(atmos.hmax) else 0.0
        )

        if modifyModel:
            Model.update_altitude_dependency(h_optimum)
            Model.update_context(W, h_optimum)

        self.dTopt = 1

        self.hopt_array = np.array([h_optimum])
        self.condition = (
            (Model.aircraft.CLmax < Model.aircraft.CL_P)
            & (Model.aircraft.CLmax > Model.aircraft.CL_E)
            & (sigma_opt < atmos.rhoratio(atmos.hmax))
        )

        self.CLopt = self.CLopt_selected = (
            Model.aircraft.CLmax if self.condition else np.nan
        )

        self.compute_optimal(W, h_optimum, Model, True)

        self.cond = (
            1
            if ((sigma_opt > atmos.rhoratio(atmos.hmax)) and self.condition)
            else np.nan
        )

        self.V_selected = (
            Model.compute_velocity(W, h_optimum, self.CLopt_selected) * self.cond
        )

        self.CLopt_selected = self.CLopt_selected * self.cond


@app.cell
def _():
    mo.md(r"""
    Now that we have derived all the optima for each condition, we can summarize the flight envelopes in one graph, as shown below. Experiment with the weight of the aircraft to understand how the theoretical ceiling for minimum power moves in the graph.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Final flight envelope

    This concludes the minimum drag derivation, find below the flight envelope showing the operational conditions where the simplified propeller aircraft can fly at minimum drag. The graph below combines all the solutions explored in this notebook.
    """)
    return


@app.cell
def _(aircraft):
    envelopeControls = plot_utils.InteractiveElements(aircraft)

    mass_slider_envelope = envelopeControls.mass_slider
    altitude_slider_envelope = envelopeControls.altitude_slider

    mass_stack_envelope, variables_stack_envelope = envelopeControls.init_layout(
        mass_slider_envelope, altitude_slider_envelope
    )

    envelopeModel = ModelSimplifiedProp(aircraft)
    return (
        altitude_slider_envelope,
        envelopeControls,
        envelopeModel,
        mass_slider_envelope,
        variables_stack_envelope,
    )


@app.cell
def _(envelopeControls, envelopeModel, mass_slider_envelope):
    W_selected_envelope = envelopeControls.sense_mass(mass_slider_envelope)

    envelopeModel.update_mass_dependency(W_selected_envelope)
    return (W_selected_envelope,)


@app.cell
def _(altitude_slider_envelope, envelopeControls, envelopeModel):
    h_selected_envelope = envelopeControls.sense_altitude(altitude_slider_envelope)

    envelopeModel.update_altitude_dependency(h_selected_envelope)
    return (h_selected_envelope,)


@app.cell
def _(W_selected_envelope, envelopeModel, h_selected_envelope):
    envelopeModel.update_context(W_selected_envelope, h_selected_envelope)
    return


@app.cell
def _(W_selected_envelope, envelopeModel, h_selected_envelope):
    _ = h_selected_envelope, W_selected_envelope

    envelopeSurface = np.broadcast_to(
        envelopeModel.drag_curve[np.newaxis, :],
        (plot_utils.meshgrid_n, plot_utils.meshgrid_n),
    )
    plot_options_envelope = {
        "surface": envelopeSurface,
    }
    return (plot_options_envelope,)


@app.cell
def _(W_selected_envelope, envelopeModel):
    MaxLiftThrustEnvelope = MaxLiftThrustCondition(
        W_selected_envelope, envelopeModel, False
    )
    return (MaxLiftThrustEnvelope,)


@app.cell
def _(MaxLiftThrustEnvelope):
    equality_trace = plot_utils.add_equality((MaxLiftThrustEnvelope,))
    return (equality_trace,)


@app.cell
def _(
    InteriorCondition,
    MaxThrustCondition,
    MaxliftCondition,
    W_selected_envelope,
    envelopeModel,
    equality_trace,
    h_selected_envelope,
    plot_options_envelope,
    variables_stack_envelope,
):
    mo.vstack(
        [
            variables_stack_envelope,
            envelopeModel.plot_grid(
                (
                    InteriorCondition(
                        W_selected_envelope, h_selected_envelope, envelopeModel
                    ),
                    MaxThrustCondition(
                        W_selected_envelope, h_selected_envelope, envelopeModel
                    ),
                    MaxliftCondition(
                        W_selected_envelope, h_selected_envelope, envelopeModel
                    ),
                ),
                plot_options_envelope,
            ).figure.add_traces(equality_trace),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## Summary
    """)
    return


@app.cell
def _():
    mo.md(
        r"""
    | Name | Condition | $C_L^*$ | $\delta_T^*$ | $D^*$ |
    |:-|:-------|:-------:|:------------:|:-------|
    |Interior-optima    | $\displaystyle   C_L^* \lt C_{L_\mathrm{max}} \quad \text{and} \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}} \lt P_{a0}E_\mathrm{max}\sqrt{\frac{\rho_0 S C_{L_E}}{2}}$ | $\sqrt{\frac{C_{D_0}}{K}}$ | $\displaystyle \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}}\frac{1}{P_{a0}E_\mathrm{max}}\sqrt{\frac{2}{\rho_0 S C_{L_E}}}$ | $\displaystyle 2W\sqrt{KC_{D_0}}=\frac{W}{E_\mathrm{max}}$ |
    |Lift-limited    |  $\displaystyle C_L^* \lt \sqrt{\frac{C_{D_0}}{K}} \quad \text{and} \quad\frac{W^{3/2}}{\sigma^{\beta+1/2}}\lt P_{a0}E_S\sqrt{\frac{\rho_0 S C_{L_\mathrm{max}}}{2}}$ | $C_{L_\mathrm{max}}$ | $\displaystyle \displaystyle \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}}\frac{1}{P_{a0}E_S}\sqrt{\frac{2}{\rho_0 S C_{L_\mathrm{max}}}}$ | $\displaystyle \frac{W}{E_S}$|
    |Thrust-limited    | || $1$| |
    |Thrust-lift limited    |   $\displaystyle   \sqrt{\frac{C_{D_0}}{K}} \lt C_L^* \lt \sqrt{\frac{3C_{D_0}}{K}} \quad \text{and} \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}}=P_{a0}E_S\sqrt{\frac{\rho_0 S C_{L_\mathrm{max}}}{2}}$ | $C_{L_\mathrm{max}}$ | $1$ | $\displaystyle \frac{W}{E_S}$|
    """
    ).center()
    return


@app.cell
def _():
    _defaults.nav_footer(
        before_file="Steady_Level_Flight/MinDrag_Jet.py",
        before_title="Minimum Drag Simplified Jet",
        above_file="Steady_Level_Flight/MinDrag.py",
        above_title="Minimum Drag Homepage",
        above_before=False,
    )
    return


if __name__ == "__main__":
    app.run()
