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
    from core import aircraft as ac
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
    data = ac.available_aircrafts(data_dir, ac_type="Propeller")
    ac_table = plot_utils.InteractiveElements.init_table(data)
    return ac_table, data


@app.cell
def _(ac_table, data):
    # Define constants dependent on the ac database. This runs every time another aircraft is selected
    if ac_table.value is not None and ac_table.value.any().any():
        active_selection = ac_table.value.iloc[0]
    else:
        active_selection = data.iloc[0]

    aircraft = ac.AircraftBase(active_selection)

    initialControls = plot_utils.InteractiveElements(aircraft, initial=True)
    initialModel = ac.ModelSimplifiedProp(aircraft)

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
    _ = h_selected_initial, W_selected_initial, initial_CL_slider

    initialSurface = np.broadcast_to(
        initialModel.power_required[np.newaxis, :],
        (plot_utils.meshgrid_n, plot_utils.meshgrid_n),
    )

    selected_value = float(
        initialModel.compute_drag(W_selected_initial, initial_CL_slider.value)
        * initialModel.compute_velocity(
            W_selected_initial, h_selected_initial, initial_CL_slider.value
        ),
    )

    plot_options_initial = {
        "surface": initialSurface,
        "title": "Minimum power",
        "axes": {"z": {"label": "P (kW)"}},
    }
    return plot_options_initial, selected_value


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Minimum Power Required: simplified propeller aircraft

    $$
    \begin{aligned}
        \min_{C_L, \delta_T}
        & \quad P = DV = \frac{1}{2}\rho V^2S(C_{D_0}+KC_L^2)V \\
        \text{subject to}
        & \quad c_1^\mathrm{eq} = L-W = \frac{1}{2}\rho V^2 S C_L - W = 0 \\
        & \quad c_2^\mathrm{eq} = T-D = \delta_T T_a(V,h) - \frac{1}{2} \rho V^2 S (C_{D_0}+K C_L^2) =0 \\
        \text{for }
        & \quad C_L \in [0, C_{L_\mathrm{max}}] \\
        & \quad \delta_T \in [0, 1] \\
        \text{with }
        & \quad T_a(V,h) = \frac{P_a(h)}{V} = \frac{P_{a0}\sigma^\beta}{V} \\
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## KKT formulation
    To be reconducted in the standard KKT analysis format, the objective function is expressed in terms of the controls by direct elimination of $c_1^\mathrm{eq}$. The velocity $V$ can be expressed as:

    $$
    V = \sqrt{\frac{2}{\rho}\frac{W}{S}\frac{1}{C_L}}
    $$

    Moreover, in previous analyses we found $\delta_T=C_L=0$ does not correspond to a sensible solution, thus we can write:

    $$
    0\lt \delta_T \le 1 \quad \text{and} \quad  0\lt C_L\le C_{L_{\mathrm{max}}}
    $$

    Notice the open interval in the lower bounds.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The KKT formulation can now be written:

    $$
    \begin{aligned}
        \min_{C_L, \delta_T}
        & \quad P = DV = W \left(\frac{C_{D_0} +K C_L^2}{C_L}\right)\sqrt{\frac{2}{\rho}\frac{W}{S}\frac{1}{C_L}}=\sqrt{\frac{2W^3}{\rho S}}\left(\frac{C_{D_0}+K C_L^2}{C_L^{3/2}}\right) = \sqrt{\frac{2W^3}{\rho S}}\left(C_{D_0} C_L^{-3/2}+K C_L^{1/2}\right)\\
        \text{subject to}
        & \quad g_1 = T - W\, \frac{1}{E}  =\frac{\delta_T P_{a0}\sigma^\beta}{V} - W\frac{C_{D_0} + K C_L^2}{C_L} = 0 \quad \Rightarrow \quad \delta_T P_{a0}\sigma^\beta - \sqrt{\frac{2W^{3}}{\rho S}} \left(C_{D_0} C_L^{-3/2}+K C_L^{1/2}\right) = 0\\
        & \quad h_1 = C_L - C_{L_\mathrm{max}} \le 0 \\
        & \quad h_2 = \delta_T - 1 \le 0 \\
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


@app.cell
def _(initial_CL_slider, initial_dT_slider, selected_value):
    [initial_CL_slider.value, initial_dT_slider.value, selected_value]
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
    \mathcal{L}(C_L, \delta_T, \lambda_1, \mu_1, \mu_2) = & P + \lambda_1 \left[T - D\right]+ \mu_1 (C_L - C_{L_\mathrm{max}}) +\mu_2 (\delta_T - 1)\\
    =&\quad \sqrt{\frac{2W^3}{\rho S}}\left(C_{D_0} C_L^{-3/2}+K C_L^{1/2}\right)(1 - \lambda_1) +\\
    & + \lambda_1 \delta_T P_{a0}\sigma^\beta \\
    & + \mu_1 (C_L - C_{L_\mathrm{max}}) + \\
    & + \mu_2 (\delta_T - 1) +\\
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The multipliers $\lambda_1, \mu_1, \mu_2$ have to meet the following conditions for an optimal solution of the optimization problem $(C_L^*, \delta_T^*)$ to exist.

    **A. Stationarity ($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \frac{\partial \mathcal{L}}{\partial C_L} = \sqrt{\frac{2W^3}{\rho S}}\left(-\frac{3}{2}C_{D_0}C_L^{-5/2} + \frac{1}{2} K C_L^{-1/2}\right)(1-\lambda_1) + \mu_1= 0$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1 P_{a0}\sigma^\beta+ \mu_2= 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $\displaystyle \delta_T P_{a0}\sigma^\beta - \sqrt{\frac{2W^{3}}{\rho S}} \left(C_{D_0} C_L^{-3/2}+K C_L^{1/2}\right) = 0$
    4.  $C_L - C_{L_\mathrm{max}} \le 0$
    5.  $\delta_T - 1 \le 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    6.  $\mu_1, \mu_2 \ge 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraint have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

    7.  $\mu_1 (C_L - C_{L_\mathrm{max}}) = 0$
    8. $\mu_2 (\delta_T - 1) = 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## KKT analysis

    We can now proceed to examine systematically the conditions where various inequality constraints are active
    or inactive.
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

    analysisModel = ac.ModelSimplifiedProp(aircraft)
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
        analysisModel.power_required[np.newaxis, :],
        (plot_utils.meshgrid_n, plot_utils.meshgrid_n),
    )

    plot_options_analysis = {
        "surface": surface,
    }
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
    ### _Interior solutions_ 

    In this case: $C_L \lt C_{L_{\mathrm{max}}}$, $\delta_T \lt 1$, $\mu_1=\mu_2= 0$

    from stationarity condition (2): $\lambda_1 = 0$

    from stationarity condition (1): 

    $$
    -\frac{3}{2}C_{D_0} C_L^{-5/2}+\frac{1}{2}KC_L^{-1/2}= 0 \quad \Rightarrow \quad KC_L^2 = 3C_{D_0} \quad \Rightarrow \quad C_L^* = \sqrt{\frac{3C_{D_0}}{K}} = \sqrt{3}C_{L_E} = C_{L_P}
    $$

    Before finding the corresponding $\delta_T$ value find the velocity associated with $C_{L_P}$:

    $$
    V_P = \sqrt{\frac{2W}{\rho S}}\sqrt[4]{\frac{K}{3C_{D_0}}}
    $$


    The optimum $\delta_T$ value is obtained from primal feasibility constraint (3). Using the velocity for minimum power, we find:

    $$
    \delta_T^* = \frac{W}{E_P}\frac{V_P}{P_{a0}\sigma^\beta}
    $$

    Where: $\displaystyle E_{\mathrm{P}} = \frac{\sqrt{3}}{2}E_{\mathrm{max}}$

    This is valid for:  

    $$
    \delta_T^*\lt 1 \quad \Leftrightarrow\quad  \frac{W^{3/2}}{\sigma^{\beta+1/2}} \lt P_{a0} E_P \sqrt{\frac{\rho_0SC_{L_P}}{2}} \;  = P_{a0} E_P \sqrt{\frac{\rho_0 S}{2}} \sqrt[4]{\frac{3 C_{D_0}}{K}}
    $$

    The value of the objective function, power, is calculated as:

    $$
    P^*_{\mathrm{min}} = DV = 4 \sqrt{\frac{2W^3}{S\sigma\rho_0}}\sqrt[4]{\frac{C_{D_0} K^3}{27}}
    $$

    This concludes the analysis for the minimum power of a simplified propeller aircraft in the interior case. Below is a summary of the optima:

    $$
    \boxed{C_L^* =  \sqrt{\frac{3C_{D_0}}{K}}}, \quad \boxed{\delta_T^* = \frac{W}{E_P}\frac{V_P}{P_{a0}\sigma^\beta}}, \quad \text{with} \quad V_P = \sqrt{\frac{2W}{\rho S}}\sqrt[4]{\frac{K}{3C_{D_0}}} \quad \text{for} \quad C_{L}^* \lt C_{L_\mathrm{max}} \quad \text{and}\quad \frac{W^{3/2}}{\sigma^{\beta+1/2}} \lt P_{a0} E_P \sqrt{\frac{\rho_0 S}{2}} \sqrt[4]{\frac{3 C_{D_0}}{K}}
    $$

    With the optimal value for minimum power: 

    $$
    P^*_{\mathrm{min}} = DV = 4 \sqrt{\frac{2W^3}{S\sigma\rho_0}}\sqrt[4]{\frac{C_{D_0} K^3}{27}}
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
    class InteriorCondition(ac.OptimumCondition):
        def __init__(self, W, h, Model):
            velocity = Model.compute_velocity(W, h, Model.aircraft.CL_P)

            self.dTopt = W / Model.aircraft.E_P / (Model.compute_thrust(h, velocity))

            self.condition = (
                W
                < Model.compute_thrust(
                    Model.aircraft.h_array,
                    Model.compute_velocity(
                        W, Model.aircraft.h_array, Model.aircraft.CL_P
                    ),
                )
                * aircraft.E_P
            ) & (Model.aircraft.CL_P < Model.aircraft.CLmax)

            self.CLopt = self.CLopt_selected = Model.aircraft.CL_P

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
    ### _Lift-limited solutions (stall)_

    In this case: $C_L = C_{L_{\mathrm{max}}}$, $\delta_T \lt 1$, $\mu_1 \gt 0$, $\mu_2= 0$

    from stationarity condition (2): $\lambda_1 = 0$

    from stationarity condition (1):

    $$
    \mu_1 = \sqrt{\frac{2W^3}{\rho S}}\left(\frac{3}{2}C_{D_0}C_{L_{\mathrm{max}}}^{-5/2} - \frac{1}{2} K C_{L_{\mathrm{max}}}^{-1/2}\right) \gt 0
    $$

    $$
    \Rightarrow 3C_{D_0}C_{L_{\mathrm{max}}}^{-2} - K \gt 0 \quad  \Rightarrow \quad C_{L_{\mathrm{max}}} \lt \sqrt{\frac{3C_{D_0}}{K}} = \sqrt{3}C_{L_E} = C_{L_P}
    $$

    This means that, if we design an aircraft such that its $C_{L_P}$ is lower than its stall lift coefficient, then the minimum power required will be obtained at stall, because the aircraft is not able to fly at $C_{L_P}$ in steady level flight.
    In other words, an aircraft so designed would only be able to fly on the right branch of the power performance diagram, because the stall speed would be higher than the speed for minimum power. Therefore, the effective minimum power flyable in steady level flight would be obtained at the stall speed.

    We can now calculate the optimal $\delta_T^*$. As before, define the velocity at which the aircraft is flying for a cleaner solution. Note that $C_L = C_{L_{\mathrm{max}}}$ thus the aircraft is flying at stall speed $V_S$: 

    $$
    V_S= \sqrt{\frac{2W}{\rho S C_{L_{\mathrm{max}}}}}
    $$

    The correrponding $\delta_T^*$, found from the primal feasibility constraint (3): 

    $$
    \delta_T^* = \frac{W}{E_S}\frac{V_S}{P_{a0}\sigma^\beta}
    $$

    This is valid for: 

    $$
    \delta_T^*\lt 1 \Leftrightarrow \frac{W^{3/2}}{\sigma^{\beta+1/2}} \lt  \; P_{a0} \,E_S\sqrt{\frac{1}{2}\rho_0SC_{L_{\mathrm{max}}}}
    $$

    The value of the objective function, power, is calculated as:

    $$
    P^*_{\mathrm{min}} = DV = \frac{W}{E_S}V_S
    $$

    This concludes the analysis for the minimum power of a simplified propeller aircraft in the lift-limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = C_{L_{\mathrm{max}}}}, \quad \boxed{\delta_T^* = \frac{W}{E_S}\frac{V_S}{P_{a0}\sigma^\beta}}, \quad \text{for} \quad C_{L}^* \lt \sqrt{\frac{3C_{D_0}}{K}} \quad \text{and}\quad \frac{W^{3/2}}{\sigma^{\beta+1/2}} \lt  \; P_{a0} \,E_S\sqrt{\frac{1}{2}\rho_0SC_{L_{\mathrm{max}}}}
    $$

    With the optimal value for minimum power: 

    $$
    P^*_{\mathrm{min}}= \frac{W^{3/2}}{\sigma^{1/2}E_S}\sqrt{\frac{2}{\rho_0 S C_{L_\mathrm{max}}}}
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
    class MaxliftCondition(ac.OptimumCondition):
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
            ) & (Model.aircraft.CLmax < Model.aircraft.CL_P)

            self.CLopt = self.CLopt_selected = Model.aircraft.CLmax

            self.compute_optimal(W, h, Model)

    return (MaxliftCondition,)


@app.cell
def _(
    W_selected_analysis,
    analysisModel,
    mass_stack_analysis,
    plot_options_analysis,
    tab_value,
    title_keys,
):
    if tab_value != title_keys[2]:
        mo.stop(True)

    MaxThrust = (MaxThrustCondition(W_selected_analysis, analysisModel),)
    surface_MaxThrust = np.broadcast_to(
        analysisModel.power_required[np.newaxis, :],
        (plot_utils.meshgrid_n, plot_utils.meshgrid_n),
    )

    mo.vstack(
        [
            mo.md(r"""
    ### _Thrust limited solutions_

    In this case: $C_L \lt C_{L_{\mathrm{max}}}$, $\delta_T = 1$, $\mu_1 = 0$, $\mu_2 \gt 0$

    from stationarity condition (2): $\displaystyle \lambda_1 = -\frac{\mu_2}{P_{a0}\sigma^\beta} \quad \Rightarrow \quad \lambda_1 \lt 0$

    Thus, from stationarity condition (1): 

    $$
    \sqrt{\frac{2W^3}{\rho S}}\left(-\frac{3}{2}C_{D_0}C_L^{-5/2} + \frac{1}{2} K C_L^{-1/2}\right)(1-\lambda_1) = 0 \quad \text{with } \quad 1-\lambda_1 \gt 0
    $$

    $$
    \Rightarrow -3C_{D_0}C_L^{-2} + K = 0
    $$


    $$
    C_L^* = \sqrt{\frac{3C_{D_0}}{K}} = \sqrt{3}C_{L_E} = C_{L_P}
    $$

    The condition for which this is true is found using the primal feasibility constraint (3). 

    $$
    \frac{W^{3/2}}{\sigma^{\beta+1/2}} = P_{a0} E_P \sqrt{\frac{\rho_0SC_{L_P}}{2}} \;  = P_{a0} E_P \sqrt{\frac{\rho_0 S}{2}} \sqrt[4]{\frac{3 C_{D_0}}{K}}
    $$

    This can be compared with what we found in the interior of the domain, showing the thrust-limited case represents the limit case of the interior optima.

    The value of the objective function, power, is calculated as:

    $$
    P^*_{\mathrm{min}} = DV = \frac{W}{E_P}\sqrt{\frac{W}{S}\frac{2}{\rho}\frac{1}{C_{L_P}}} = \frac{W^{3/2}}{E_P\sigma^{1/2}}\sqrt[4]{\frac{K}{3C_{D_0}}}\sqrt{\frac{2}{\rho_0S}}
    $$

    This concludes the analysis for the minimum power of a simplified propeller aircraft in the thrust-limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = \sqrt{\frac{3C_{D_0}}{K}}}, \quad \boxed{\delta_T^* =1}, \quad \text{for} \quad C_{L}^* \lt C_{L_\mathrm{max}} \quad \text{and}\quad \frac{W^{3/2}}{\sigma^{\beta+1/2}} = P_{a0} E_P \sqrt{\frac{\rho_0 S}{2}} \sqrt[4]{\frac{3 C_{D_0}}{K}}
    $$

    With the optimal value for minimum power: 

    $$
    P^*_{\mathrm{min}}= \frac{W^{3/2}}{E_P\sigma^{1/2}}\sqrt[4]{\frac{K}{3C_{D_0}}}\sqrt{\frac{2}{\rho_0S}}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.
    """),
            mass_stack_analysis,
            analysisModel.plot_grid(
                MaxThrust,
                plot_options_analysis,
            ).figure,
        ]
    ).callout()
    return


@app.class_definition
class MaxThrustCondition(ac.OptimumCondition):
    def __init__(self, W, Model, modifyModel=True):
        sigma_opt = (
            W**1.5
            / (Model.aircraft.Pa0 * 1e3)
            / Model.aircraft.E_P
            / (np.sqrt(atmos.rho0 * Model.aircraft.S * Model.aircraft.CL_P / 2))
        ) ** (1 / (Model.aircraft.beta + 0.5))
        h_optimum = (
            atmos.altitude(sigma_opt) if sigma_opt > atmos.rhoratio(atmos.hmax) else 0.0
        )

        if modifyModel:
            Model.update_altitude_dependency(h_optimum)
            Model.update_context(W, h_optimum)

        self.dTopt = 1

        self.hopt_array = np.array([h_optimum])
        self.condition = (Model.aircraft.CL_P < Model.aircraft.CLmax) & (
            sigma_opt > atmos.rhoratio(atmos.hmax)
        )

        self.CLopt = self.CLopt_selected = (
            Model.aircraft.CL_P if self.condition else np.nan
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
def _(
    W_selected_analysis,
    analysisModel,
    mass_stack_analysis,
    plot_options_analysis,
    tab_value,
    title_keys,
):
    if tab_value != title_keys[3]:
        mo.stop(True)

    MaxLiftThrust = (MaxLiftThrustCondition(W_selected_analysis, analysisModel),)
    surface_MaxLiftThrust = np.broadcast_to(
        analysisModel.power_required[np.newaxis, :],
        (plot_utils.meshgrid_n, plot_utils.meshgrid_n),
    )

    plot_options_analysis_MaxLiftThrust = {"surface": surface_MaxLiftThrust}
    mo.vstack(
        [
            mo.md(r"""
    ### _Lift- and thrust- limited optimum_

    In this case: $C_L = C_{L_{\mathrm{max}}}$, $\delta_T = 1$, $\mu_1 \gt 0$, $\mu_2 \gt 0$

    from stationarity condition (2): $\displaystyle \lambda_1 = -\frac{\mu_2}{P_{a0}\sigma^\beta} \quad \Rightarrow \quad \lambda_1 \lt 0$

    Thus, from stationarity condition (1), since $1-\lambda_1 \gt 0$: 

    $$
    \mu_1 = \sqrt{\frac{2W^3}{\rho S}}\left(\frac{3}{2}C_{D_0}C_{L_{\mathrm{max}}}^{-5/2} - \frac{1}{2} K C_{L_{\mathrm{max}}}^{-1/2}\right)(1-\lambda_1)\gt 0
    $$

    $$
    \Rightarrow \quad  3 C_{D_0}C_{L_{\mathrm{max}}}^{-2} - K \gt 0
    $$


    $$
    \Rightarrow \quad  C_{L_{\mathrm{max}}} \lt \sqrt{\frac{3C_{D_0}}{K}} = \sqrt{3}C_{L_E} = C_{L_P}
    $$

    which shows once again that the necessary condition to obtain minimum power in stall conditions and maximum throttle. If it were otherwise ($C_{L_{\mathrm{max}}} > C_{L_P}$), it would be impossible to minimise power at stall and maximum thrust as the aircraft would reach the unconstrained minimum power before stalling.

    The condition for which this is true is found using the primal feasibility constraint (3). 

    $$
    \frac{W^{3/2}}{\sigma^{\beta+1/2}} = P_{a0}E_S \sqrt{\frac{1}{2}\rho_0SC_{L_{\mathrm{max}}}}
    $$

    The value of the objective function, power, is calculated as:

    $$
    P^*_{\mathrm{min}} = DV = \frac{W}{E_S}V_S
    $$

    This concludes the analysis for the minimum power of a simplified propeller aircraft in the lift-thrust limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = C_{L_\mathrm{max}}}, \quad \boxed{\delta_T^* =1}, \quad \text{for} \quad C_{L_{\mathrm{max}}} \lt \sqrt{\frac{3C_{D_0}}{K}} \quad \text{and}\quad \frac{W^{3/2}}{\sigma^{\beta+1/2}} = P_{a0} E_S \sqrt{\frac{\rho_0 S C_{L_\mathrm{max}}}{2}}
    $$

    With the optimal value for minimum power: 

    $$
    P^*_{\mathrm{min}}= \frac{W^{3/2}}{\sigma^{1/2}E_S}\sqrt{\frac{2}{\rho_0 S C_{L_\mathrm{max}}}}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.
    """),
            mass_stack_analysis,
            analysisModel.plot_grid(
                MaxLiftThrust,
                plot_options_analysis_MaxLiftThrust,
            ).figure,
        ]
    ).callout()
    return


@app.class_definition
class MaxLiftThrustCondition(ac.OptimumCondition):
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
        self.condition = (Model.aircraft.CLmax < Model.aircraft.CL_P) & (
            sigma_opt > atmos.rhoratio(atmos.hmax)
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Final flight Envelope
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Now after deriving all the optima for each condition we can summarize the flight envelopes in one graph, as shown below. Experiment with the weight of the aircrarft to understand how the theoretical ceiling for minimum speed moves in the graph.
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

    envelopeModel = ac.ModelSimplifiedProp(aircraft)
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
        envelopeModel.power_required[np.newaxis, :],
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


@app.cell
def _():
    mo.md(
        r"""
    | Name | Condition | $C_L^*$ | $\delta_T^*$ | $P^*$ |
    |:-|:-------|:-------:|:------------:|:-------|
    |Interior-optima    | $\displaystyle \quad C_L^* \lt C_{L_\mathrm{max}} \quad \text{and} \quad\frac{W^{3/2}}{\sigma^{\beta+1/2}} \lt P_{a0} E_P \sqrt{\frac{\rho_0 S}{2}} \sqrt[4]{\frac{3 C_{D_0}}{K}}$ | $\displaystyle \sqrt{\frac{3C_{D_0}}{K}}$ | $\displaystyle \frac{W}{E_P}\frac{V_P}{P_{a0}\sigma^\beta}$ | $\displaystyle 4 \sqrt{\frac{2W^3}{S\sigma\rho_0}}\sqrt[4]{\frac{C_{D_0} K^3}{27}}$ |
    |Lift-limited    |  $\displaystyle C_{L_\mathrm{max}} \lt \sqrt{\frac{3C_{D_0}}{K}} \quad \text{and}\quad \frac{W^{3/2}}{\sigma^{\beta+1/2}} \lt  \; P_{a0} \,E_S\sqrt{\frac{\rho_0SC_{L_{\mathrm{max}}}}{2}}$ | $C_{L_\mathrm{max}}$ | $\displaystyle \frac{W}{E_S}\frac{V_S}{P_{a0}\sigma^\beta}$ | $\displaystyle \frac{W^{3/2}}{\sigma^{1/2}E_S}\sqrt{\frac{2}{\rho_0 S C_{L_\mathrm{max}}}}$|
    |Thrust-limited    | $\displaystyle C_L^* \lt C_{L_\mathrm{max}} \quad \text{and}\quad \frac{W^{3/2}}{\sigma^{\beta+1/2}} = P_{a0} E_P \sqrt{\frac{\rho_0 S}{2}} \sqrt[4]{\frac{3 C_{D_0}}{K}}$ | $\displaystyle \sqrt{\frac{3C_{D_0}}{K}}$ | $1$ | $\displaystyle \frac{W^{3/2}}{E_P\sigma^{1/2}}\sqrt[4]{\frac{K}{3C_{D_0}}}\sqrt{\frac{2}{\rho_0S}}$ |
    |Thrust-lift limited    |  $\displaystyle \quad C_{L_{\mathrm{max}}} \lt \sqrt{\frac{3C_{D_0}}{K}} \quad \text{and} \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}} = P_{a0} E_S \sqrt{\frac{\rho_0 S C_{L_\mathrm{max}}}{2}}$ | $C_{L_\mathrm{max}}$ | $1$ | $\displaystyle \frac{W^{3/2}}{\sigma^{1/2}E_S}\sqrt{\frac{2}{\rho_0 S C_{L_\mathrm{max}}}}$|
    """
    ).center()
    return


@app.cell
def _():
    _defaults.nav_footer(
        before_file="Steady_Level_Flight/MinPower_Jet.py",
        before_title="Minimum Power Simplified Jet",
        above_file="Steady_Level_Flight/MinPower.py",
        above_title="Minimum Power Homepage",
        above_before=False,
    )
    return


if __name__ == "__main__":
    app.run()
