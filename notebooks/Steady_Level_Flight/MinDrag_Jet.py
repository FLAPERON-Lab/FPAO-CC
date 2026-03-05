# SPDX-FileCopyrightText: 2026 Carmine Varriale <C.varriale@tudelft.nl>
# SPDX-FileCopyrightText: 2026 Federico Angioni <F.angioni@student.tudelft.nl>
# SPDX-FileCopyrightText: 2026 Maarten van Hoven <M.B.vanHoven@tudelft.nl>
#
# SPDX-License-Identifier: Apache-2.0

import marimo

__generated_with = "0.17.8"
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
    from core import aircraft as ac
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
    data = ac.available_aircrafts(data_dir, ac_type="Jet")
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

    aircraft = ac.AircraftBase(active_selection)

    label = "Min Speed (m/s)"

    initialControls = plot_utils.InteractiveElements(aircraft, initial=True)
    initialModel = ac.ModelSimplifiedJet(aircraft)

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
    # Minimum drag: simplified jet aircraft

    $$
    \begin{aligned}
    \min_{C_L, \delta_T}
    & \quad D = \frac{1}{2}\rho V^2S\left(C_{D_0} + K C_L^2\right) \\
    \text{subject to}
    & \quad c_1^\mathrm{eq} = L-W = \frac{1}{2}\rho V^2 S C_L - W = 0\\
    & \quad c_2^ \mathrm{eq} = T-D = \delta_T T_a(V,h) - \frac{1}{2} \rho V^2 S (C_{D_0}+K C_L^2) =0 \\
    \text{for}
    & \quad C_L \in [0, C_{L_\mathrm{max}}] \\
    & \quad \delta_T \in [0, 1] \\
    \text{with}
    & \quad T_a(V,h) = T_a(h) = T_{a0}\sigma^\beta \\
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## KKT formulation
    To be reconducted in the standard KKT analysis format, the objective function is expressed in terms of the controls by directly eliminating $c_1^\mathrm{eq}$. The velocity $V$ can be described as:

    $$
    V = \sqrt{\frac{2}{\rho}\frac{W}{S}\frac{1}{C_L}}
    $$


    The KKT formulation thus becomes:
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
    & \quad g_1 = \frac{T}{W} - \frac{1}{E}  =\frac{\delta_T T_{a0}\sigma^\beta}{W} - \frac{C_{D_0} + K C_L^2}{C_L} = 0 \\
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
    & + \lambda_1 \left[\frac{\delta_T T_{a0}\sigma^\beta}{W} - \frac{C_{D_0} + K C_L^2}{C_L}\right] + \\
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

    1. $\displaystyle \begin{aligned}\frac{\partial \mathcal{L}}{\partial C_L} = W \frac{K C_L^2 - C_{D_0}}{C_L^2} - \lambda_1W\left(\frac{K C_L^2 - C_{D_0}}{C_L^2}\right) + \mu_1 - \mu_2 = W\frac{K C_L^2 - C_{D_0}}{C_L^2} (1 -\lambda_1) +\mu_1 - \mu_2 = 0 \end{aligned}$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1\frac{T_{a0}\sigma^\beta}{W}+\mu_3-\mu_4 = 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $\displaystyle \frac{\delta_T T_{a0}\sigma^\beta}{W} - \frac{C_{D_0} + K C_L^2}{C_L} = 0$
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
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraint have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

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

    1. $\displaystyle W\frac{K C_L^2 - C_{D_0}}{C_L^2} (1 -\lambda_1) +\mu_1 = 0$
    2. $\displaystyle \lambda_1\frac{T_{a0}\sigma^\beta}{W}+\mu_3 = 0$
    3. $\displaystyle \frac{\delta_T T_{a0}\sigma^\beta}{W} - \frac{C_{D_0} + K C_L^2}{C_L} = 0$
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

    analysisModel = ac.ModelSimplifiedJet(aircraft)
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

    plot_options_analysis = {"surface": surface}
    return plot_options_analysis, tab_value


@app.cell
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
    ### _Interior optimum for minimum drag_ 

    Assuming that that $0 < C_L^* < C_{L_\mathrm{max}}$ and $0 < \delta_T^* < 1$ is equivalent to consider all inequality constraints as inactive.

    Therefore: $\mu_1, \mu_3 =0$. 

    From stationarity condition (2): $\lambda_1 = 0$.

    From stationarity condition (1), it is possible to obtain the value of $C_L^*$ for minimum drag.

    $$
    C_L^* = \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$

    Notice how the optimal $C_L^*$ has the **same value** for maximum aerodynamic efficiency (maximum $C_L /C_D$), for 
    $0\lt C_L \lt  C_{L_\mathrm{max}}$ and $0 \lt \delta_T \lt 1$, as shown in [Aerodynamic Efficiency](/?file=Steady_Level_Flight/MinDrag.py).

    The corresponding $\delta_T^*$ is found by solving the primal feasibility constraint (3) and using $C_L = C_L^*$, as calculated above.


    $$
    \delta_T^* = \frac{2W}{T_{a0}\sigma^\beta}\frac{C_{D_0} + K C_L^2}{C_L} = \frac{2W}{T_{a0}\sigma^\beta}\sqrt{C_{D_0}K}
    $$

    This value is compliant with the primal feasibility constraint (5) for: 

    $$
    \delta_T^* = \frac{2W}{T_{a0}\sigma^\beta}\sqrt{C_{D_0}K} \lt 1 \quad \Leftrightarrow \quad \frac{W}{\sigma^\beta} \lt \frac{T_{a0}}{2\sqrt{C_{D_0}K}} = \frac{W}{\sigma^\beta} \lt T_{a0}E_{max}$$

    Which tells us that it is possible to achieve this optimal condition only when the combination of aircraft weight and altitude respect the above inequality.

    The corresponding minimum drag is found by first computing $V^*$ and $C_D^*$: 

    $$
    V = \sqrt{\frac{W}{S}\frac{2}{\rho}\frac{1}{C_L^*}} \quad \text{and} \quad  C_D^* = 
    2C_{D_0}$$

    $$
    D_{\mathrm{min}}^* =  \frac{1}{2}\rho {V^*}^2 S C_D= 
    2W\sqrt{KC_{D_0}}=\frac{W}{E_\mathrm{max}}
    $$

    We can now rewrite $\delta_T^*$ in terms of $D_\mathrm{min}$:

    $$
    \delta_T^*=\frac{2W}{T_{a0}\sigma^\beta}\sqrt{C_{D_0}K}=\frac{D_\mathrm{min}^*}{T_{a0}\sigma^{\beta}}
    $$

    This concludes the analysis for the minimum drag of a simplified jet aircraft in the domain's interior. Below is a summary of the optima:

    $$
    \boxed{C_L^* = \sqrt{\frac{C_{D_0}}{K}}}, \quad \boxed{\delta_T^*=\frac{2W}{T_{a0}\sigma^\beta}\sqrt{C_{D_0}K}=\frac{D_\mathrm{min}^*}{T_{a0}\sigma^{\beta}}}, \quad \text{for} \quad C_L^* \lt C_{L_\mathrm{max}}\quad \text{and} \quad \frac{W}{\sigma^\beta} \lt T_{a0}E_\mathrm{max}
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
def _(aircraft, analysisModel):
    class InteriorCondition(ac.OptimumCondition):
        def __init__(self, W, h, Model):
            self.CLopt = self.CLopt_selected = Model.aircraft.CL_E
            self.dTopt = (
                W
                / Model.aircraft.E_max
                / (Model.aircraft.Ta0 * 1e3)
                / (Model.rhoratio_selected**Model.aircraft.beta)
            )

            self.condition = (
                W
                < analysisModel.compute_thrust(analysisModel.aircraft.h_array)
                * aircraft.E_max
            ) & (analysisModel.aircraft.CL_E < analysisModel.aircraft.CLmax)

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

    From stationarity condition (1): $\displaystyle \mu_1 = W\frac{C_{D_0} - KC_{L_\mathrm{max}}^2}{C_{L_\mathrm{max}}^2} \gt 0$, which results in: $\displaystyle  C_{L_\mathrm{max}} < \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}$

    This means that, in order for aerodynamic drag to have a minimum at $C_L = C_{L_\mathrm{max}}$, the aircraft must have been designed to have a higher lift coefficient for maximum efficiency than its stall lift coefficient.

    In other words, the aircraft would only be able to fly on the right branch of the performance diagram, and the stall speed would be higher than the speed for maximum efficiency, therefore representing the speed for minimum drag.

    In the rare occasion this condition would be verified, the corresponding throttle could be once again calculated from stationarity condition (3):

    $$
    \displaystyle \delta_T^* = \frac{C_{D_\mathrm{max}}}{C_{L_\mathrm{max}}}\frac{W}{T_{a0}\sigma^\beta} = \frac{W}{E_S T_{a0}\sigma^\beta}
    $$

    This value is compliant with the primal feasibility constraint if:

    $$
    \delta_T^* < 1 \Leftrightarrow \frac{W}{\sigma^\beta} < T_{a0}E_S
    $$

    which gives us the conditions to achieve minimum drag in terms of aircraft weight and altitude.

    The value of the objective function, minimum drag, is calculated straightforwardly as:

    $$
    D_{\mathrm{min}}^* =  \frac{1}{2}\rho V_s^2 S C_{D_s} = \frac{W}{E_s}
    $$

    This is a higher value than the unconstrained one, and therefore operating in this scenario should be avoided if minimum drag is a goal.

    This concludes the analysis for the minimum drag of a simplified jet aircraft in the lift-limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = C_{L_\mathrm{max}}}, \quad \boxed{\delta_T^*=\frac{W}{E_S T_{a0}\sigma^\beta}}, \quad \text{for} \quad C_{L}^* < \sqrt{\frac{C_{D_0}}{K}} \quad \text{and} \quad \frac{W}{\sigma^\beta} \lt T_{a0}E_{S}
    $$

    With the optimal value for minimum drag:

    $$
    D_{\mathrm{min}}^* = \frac{W}{E_S}
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
def _(aircraft, analysisModel):
    class MaxliftCondition(ac.OptimumCondition):
        def __init__(self, W, h, Model):
            self.CLopt = self.CLopt_selected = Model.aircraft.CLmax
            self.dTopt = (
                W
                / Model.aircraft.E_S
                / (Model.aircraft.Ta0 * 1e3)
                / (Model.rhoratio_selected**Model.aircraft.beta)
            )

            self.condition = (
                W
                < analysisModel.compute_thrust(analysisModel.aircraft.h_array)
                * aircraft.E_S
            ) & (analysisModel.aircraft.CLmax < analysisModel.aircraft.CL_E)

            self.compute_optimal(W, h, Model)

    return (MaxliftCondition,)


@app.cell
def _(
    W_selected_analysis,
    analysisModel,
    mass_stack_analysis,
    tab_value,
    title_keys,
):
    if tab_value != title_keys[2]:
        mo.stop(True)

    MaxThrust = (MaxThrustCondition(W_selected_analysis, analysisModel),)
    surface_MaxThrust = np.broadcast_to(
        analysisModel.drag_curve[np.newaxis, :],
        (plot_utils.meshgrid_n, plot_utils.meshgrid_n),
    )

    plot_options_analysis_MaxThrust = {
        "surface": surface_MaxThrust,
    }

    mo.vstack(
        [
            mo.md(r"""
    ###_Thrust-limited minimum drag_


    $C_L \lt C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 = 0$

    $\delta_T = 1 \quad \Rightarrow \quad \mu_3 \gt 0$

    From stationarity condition (2), obtain:

    $$
    \lambda_1 = -\frac{\mu_3}{T_{a0}\sigma^{\beta}} \lt 0
    $$

    and from stationarity condition (1):

    $$
    \displaystyle \left(\frac{KC_L^2 - C_{D_0}}{C_L^2}\right)(1-\lambda_1) = 0 \Rightarrow \frac{KC_L^2 - C_{D_0}}{C_L^2} = 0
    $$

    Which yields the following optima:

    $$
    C_L^* = \sqrt{\frac{C_{D_0}}{K}} = C_{L_E} \quad  \text{and} \quad \delta_T^* = 1
    $$

    This optimum is continuous with the interior optimum, thus yielding the same result for $D_\mathrm{min}$:

    $$
    D_\mathrm{min}^* =  \frac{1}{2}\rho {V^*}^2 S C_D = 2W\sqrt{KC_{D_0}}=\frac{W}{E_\mathrm{max}}
    $$

    The operational condition is found from (3), with $\delta_T = 1$, obtaining:

    $$
    \frac{W}{\sigma^\beta} = T_{a0}E_{\mathrm{max}}
    $$

    with:

    $$
    C_D^* = 2C_{D_0}, \quad V^* = \sqrt{\frac{W}{S}\frac{2}{\rho}\frac{1}{C_L^*}}, \quad \delta_T^*=1, \quad \frac{W}{\sigma^\beta} = T_{a0}E_{\mathrm{max}}
    $$

    This concludes the analysis for the minimum drag of a simplified jet aircraft in the thrust-limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = \sqrt{\frac{C_{D_0}}{K}}}, \quad \boxed{\delta_T^*=1}, \quad \text{for} \quad C_{L}^* < C_{L_\mathrm{max}} \quad \text{and} \quad \frac{W}{\sigma^\beta} = T_{a0}E_\mathrm{max}
    $$

    With the following value for the objective function:

    $$
    D_{\mathrm{min}}^* = \frac{W}{E_\mathrm{max}}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.

    """),
            mass_stack_analysis,
            analysisModel.plot_grid(MaxThrust, plot_options_analysis_MaxThrust).figure,
        ]
    ).callout()
    return


@app.class_definition
class MaxThrustCondition(ac.OptimumCondition):
    def __init__(self, W, Model, modifyModel=True):
        sigma_opt = (W / (Model.aircraft.Ta0 * 1e3) / Model.aircraft.E_max) ** (
            1 / Model.aircraft.beta
        )
        h_optimum = (
            atmos.altitude(sigma_opt)
            if sigma_opt > atmos.rhoratio(atmos.hmax)
            else 20000.0
        )

        if modifyModel:
            Model.update_altitude_dependency(h_optimum)
            Model.update_context(W, h_optimum)

        self.dTopt = 1

        self.hopt_array = np.array([h_optimum])
        self.condition = (Model.aircraft.CL_E < Model.aircraft.CLmax) & (
            sigma_opt > atmos.rhoratio(atmos.hmax)
        )

        self.CLopt = self.CLopt_selected = (
            Model.aircraft.CL_E if self.condition else np.nan
        )

        self.compute_optimal(W, h_optimum, Model, True)

        self.cond = 1 if (sigma_opt > atmos.rhoratio(atmos.hmax)) else np.nan

        self.V_selected = (
            Model.compute_velocity(W, h_optimum, self.CLopt_selected) * self.cond
        )

        self.CLopt_selected = self.CLopt_selected * self.cond


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
    ###_Thrust- and lift- limited minimum drag_

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 \gt 0$

    $\delta_T = 1 \quad \Rightarrow \quad \mu_3 \gt 0$

    from stationarity condition (2):

    $$
    \lambda_1= -\frac{\mu_3 }{T_{a0}\sigma^{\beta}} \lt 0
    $$

    and from stationarity condition (1):

    $$
    \displaystyle \mu_1 = W \left( \frac{C_{D_0} - KC_{L_\mathrm{max}}^2}{C_{L_\mathrm{max}}^2}\right)(1 - \lambda_1) \gt 0
    $$

    The two conditions above need to be verified for a minimum to exist when both boundaries are active,  at the same time:

    $$
    C_{L_\mathrm{max}} \lt \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$

    and not equality
    The same considerations hold for the case of the lift-limited analysis, with the only difference that now $\delta_T^* = 1$
    In fact, once again, the aircraft would have to stall at a higher speed than the one for minimum drag. Continuing with primal feasibility condition (3), obtain the operational condition:


    $$
    \frac{W}{\sigma^\beta} = T_{a0}E_S
    $$

    This concludes the analysis for the minimum drag of a simplified jet aircraft in the thrust-lift limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = C_{L_\mathrm{max}}}, \quad \boxed{\delta_T^*=1}, \quad \text{for} \quad C_{L}^* \lt \sqrt{\frac{C_{D_0}}{K}} \quad \text{and} \quad \frac{W}{\sigma^\beta} = T_{a0}E_S
    $$

    With the following value for the objective function:

    $$
    D_{\mathrm{min}}^* = \frac{W}{E_\mathrm{S}}
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
class MaxLiftThrustCondition(ac.OptimumCondition):
    def __init__(self, W, Model, modifyModel=True):
        sigma_opt = (W / (Model.aircraft.Ta0 * 1e3) / Model.aircraft.E_S) ** (
            1 / Model.aircraft.beta
        )
        h_optimum = (
            atmos.altitude(sigma_opt) if sigma_opt > atmos.rhoratio(atmos.hmax) else 0.0
        )

        if modifyModel:
            Model.update_altitude_dependency(h_optimum)
            Model.update_context(W, h_optimum)

        self.dTopt = 1

        self.hopt_array = np.array([h_optimum])
        self.condition = (Model.aircraft.CLmax < Model.aircraft.CL_E) & (
            sigma_opt < atmos.rhoratio(atmos.hmax)
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
    ## Final flight envelope
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Now after deriving all the optima for each condition we can summarize the flight envelopes in one graph, as shown below. Experiment with the weight of the aircrarft to understand how the theoretical ceiling for minimum power moves in the graph.
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

    envelopeModel = ac.ModelSimplifiedJet(aircraft)
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

    plot_options_envelope = {"surface": envelopeSurface}
    return (plot_options_envelope,)


@app.cell
def _(W_selected_envelope, envelopeModel):
    MaxLiftThrustEnvelope = MaxLiftThrustCondition(
        W_selected_envelope, envelopeModel, False
    )
    MaxThrustEnvelope = MaxThrustCondition(W_selected_envelope, envelopeModel, False)
    return MaxLiftThrustEnvelope, MaxThrustEnvelope


@app.cell
def _(MaxLiftThrustEnvelope, MaxThrustEnvelope):
    equality_trace = plot_utils.add_equality((MaxLiftThrustEnvelope, MaxThrustEnvelope))
    return (equality_trace,)


@app.cell
def _(
    InteriorCondition,
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


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    | Name | Condition | $C_L^*$ | $\delta_T^*$ | $P^*$ |
    |:-|:-------|:-------:|:------------:|:-------|
    |Interior-optima    | $\displaystyle C_L^* \lt C_{L_\mathrm{max}} \quad \text{and} \quad \frac{W}{\sigma^\beta} \lt \frac{\sqrt{3}}{2}E_\mathrm{max}T_{a0}$ | $\sqrt{\frac{3C_{D_0}}{K}}$ | $\displaystyle \frac{W}{E_{\mathrm{P}}}\frac{1}{T_{a0}\sigma^\beta}$ | $\displaystyle 2W\sqrt{KC_{D_0}}=\frac{W}{E_\mathrm{max}}$ |
    |Lift-limited    |  $\displaystyle C_L^* \lt \sqrt{\frac{C_{D_0}}{K}}\quad \text{and}\quad \frac{W}{\sigma^\beta} < T_{a0} E_\mathrm{S}$ | $C_{L_\mathrm{max}}$ | $\displaystyle \frac{W}{E_S T_{a0} \sigma^\beta}$ | $\displaystyle \frac{W}{E_S}$|
    |Thrust-limited    | $\displaystyle C_L^* \lt C_{L_\mathrm{max}} \quad \text{and}\quad  \frac{W}{\sigma^\beta} = T_{a0} E_\mathrm{max}$ | $\displaystyle \sqrt{\frac{C_{D_0}}{K}}$ | $1$ | $\displaystyle \frac{W^{3/2}}{\sigma^{1/2}}\left(\frac{C_{D_0}+ K C_L^{*2}}{C_L^{*}}\right)\sqrt{\frac{2}{\rho_0 S C_L^*}}$ |
    |Thrust-lift limited    |  $\displaystyle C_L^* \lt \sqrt{\frac{C_{D_0}}{K}}\quad \text{and}\quad  \frac{W}{\sigma^\beta} = T_{a0} E_\mathrm{S}$ | $C_{L_\mathrm{max}}$ | $1$ | $\displaystyle \frac{W}{E_S}$|
    """
    ).center()
    return


@app.cell
def _():
    _defaults.nav_footer(
        after_file="Steady_Level_Flight/MinDrag_Prop.py",
        after_title="Minimum Drag Simplified Propeller",
        above_file="Steady_Level_Flight/MinDrag.py",
        above_title="Minimum Drag Homepage",
        above_before=True,
    )
    return


if __name__ == "__main__":
    app.run()
