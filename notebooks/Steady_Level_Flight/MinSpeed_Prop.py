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
    from core import _defaults
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import numpy as np
    from scipy.optimize import root_scalar
    from core import plot_utils
    from core import atmos
    from core.aircraft import available_aircrafts, OptimumCondition, AircraftBase, ModelSimplifiedProp

    _defaults.FILEURL = _defaults.get_url()

    _defaults.set_plotly_template()
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
    initial_V_slider = mo.ui.slider(
        start=round(
            initialModel.compute_velocity(
                aircraft.OEM * atmos.g0, atmos.a(20e3), aircraft.CLmax
            )
        ),
        stop=atmos.a(0),
        step=1,
        show_value=True,
        label=r"$V$ (m/s)",
    )
    initial_dT_slider = initialControls.dT_slider

    initial_mass_stack, initial_variables_stack = initialControls.init_layout(
        initial_mass_slider, initial_altitude_slider
    )
    return (
        aircraft,
        initialControls,
        initialModel,
        initial_V_slider,
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
def _(W_selected_initial, h_selected_initial, initialModel, initial_V_slider):
    _ = h_selected_initial, W_selected_initial

    initialSurface = np.broadcast_to(
        initialModel.V_CLarray[np.newaxis, :],
        (plot_utils.meshgrid_n, plot_utils.meshgrid_n),
    )

    selected_value = initial_V_slider.value

    plot_options_initial = {
        "surface": initialSurface,
        "title": "Minimum speed",
        "axes": {
            "x": {
                "data_key": "V_CLarray",
                "label": "V (m/s)",
                "bound": 6 * np.min(initialSurface),
            },
            "y": {
                "data_key": "aircraft.dT_array",
                "label": "δ<sub>T</sub> (-)",
                "bound": 1,
            },
            "z": {"label": "V (m/s)"},
        },
        "xy_lowerbound": -0.1,
        "z_lowerbound": np.min(initialSurface),
        "camera": {"x": -1.35, "y": -1.35, "z": 1.35},
        "opacity": 0.9,
        "colorscale": "viridis",
        "factor": 6,
    }
    return plot_options_initial, selected_value


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Minimum airspeed: simplified piston propeller aircraft

    $$
    \begin{aligned}
        \min_{C_L, \delta_T}
        & \quad V \\
        \text{subject to}
        & \quad c_1^\mathrm{eq} = L-W = \frac{1}{2}\rho V^2 S C_L - W = 0 \\
        & \quad c_2^\mathrm{eq} = T-D = \delta_T T_a(V,h) - \frac{1}{2} \rho V^2 S (C_{D_0}+K C_L^2) =0 \\
        \text{for }
        & \quad C_L \in [0, C_{L_\mathrm{max}}] \\
        & \quad \delta_T \in [0, 1] \\
        \text{with }
        & \quad T_a(V,h) =  \frac{P_a(h)}{V} =  \frac{P_{a0}\sigma^\beta}{V} \\
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We could approach the solution of this problem in the same way we have approched the one for simplified jets: obtain the expression of $V$ from $c_1^\mathrm{eq}$, substitute it out of the whole problem, then proceed with deriving with respec to $C_L$ and $\delta_T$.
    In the case of propeller airplanes, this results in the following expression of the horizontal equilibrium contraint, which is unhandy to take derivatives with respect to $C_L$:

    $$
    \delta_T  \frac{P_{a0}\sigma^\beta}{V} - \frac{1}{2} \rho V^2 S \left( C_{D_0} + K C_L^2 \right) = 0
    \quad \Leftrightarrow \quad
    \delta_T  P_{a0}\sigma^\beta - \frac{1}{2} \rho S \left(\frac{2W}{\rho S C_L} \right)^{3/2}\left( C_{D_0} + K C_L^2 \right) = 0
    $$

    Instead, in this case, it is more convenient to reformulate the problem by eliminating $C_L$ instead of $V$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Problem reformulation

    From the vertical equilibrium equation:

    $$
    C_L = \frac{2W}{\rho V^2 S}
    $$

    The horizontal equilibrium equation then becomes:

    $$
    \delta_T  P_{a0}\sigma^\beta - \frac{1}{2} \rho V^3 S \left( C_{D_0} + \frac{4KW^2}{\rho^2 S^2  V^4}\right) = 0
    \quad \Leftrightarrow \quad
    \delta_T  P_{a0}\sigma^\beta - \frac{1}{2} \rho S V^3 C_{D_0} - \frac{2KW^2}{\rho S V} = 0
    $$

    The bounds on $C_L$ can be rewritten as the following inequality constraint:

    $$
    0 \le \frac{2W}{\rho V^2 S} \le C_{L_\mathrm{max}}
    $$

    where the left one is always verified, and the right one is equivalent to:

    $$
    V \ge \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}} = V_s
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## KKT Formulation

    $$
    \begin{aligned}
        \min_{V, \delta_T}
        & \quad V \\
        \text{subject to}
        & \quad g_1 = \delta_T  P_{a0}\sigma^\beta - \frac{1}{2} \rho S V^3 C_{D_0} - \frac{2KW^2}{\rho S V} = 0 \\
        & \quad h_1 = \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}} - V \le 0 \\
        & \quad h_2 = -\delta_T \le 0 \\
        & \quad h_3 = \delta_T - 1 \le 0 \\
    \end{aligned}
    $$
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
    initial_V_slider,
    initial_dT_slider,
    initial_variables_stack,
    plot_options_initial,
    selected_value,
):
    mo.md(f"""
    Here you can modify the control variables to understand how it affects the design: {mo.vstack([mo.hstack([initial_dT_slider, initial_V_slider]), initial_variables_stack, initialModel.plot_initial(plot_options_initial, [selected_value, initial_dT_slider.value, selected_value]).figure])}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Lagrangian function and KKT conditions

    The Lagrangian function combines the objective function with eqaulity constraints using Lagrange multipliers ($\lambda_i$) and inequality constraints using KKT multipliers ($\mu_j$).

    $$
    \begin{aligned}
    \mathcal{L}(V, \delta_T, \lambda_1, \mu_1, \mu_2, \mu_3) =
    \quad \frac{2W}{\rho S C_L}
    & + \\
    & + \lambda_1 \left(\delta_T  P_{a0}\sigma^\beta - \frac{1}{2} \rho S V^3 C_{D_0} - \frac{2KW^2}{\rho S V}\right) + \\
    & + \mu_1 \left( \frac{2W}{\rho S C_{L_\mathrm{max}}} - V \right) + \\
    & + \mu_2 (-\delta_T) + \\
    & + \mu_3 (\delta_T - 1) +\\
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **A. Stationarity conditions($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \frac{\partial \mathcal{L}}{\partial V} = 1 + \lambda_1 \left( \frac{2KW^2}{\rho S V^2} - \frac{3}{2}\rho V^2SC_{D_0} \right) -\mu_1 = 0$
    2. $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1  P_{a0}\sigma^\beta - \mu_2 + \mu_3 = 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $\displaystyle \delta_T  P_{a0}\sigma^\beta - \frac{1}{2} \rho S V^3 C_{D_0} - \frac{2KW^2}{\rho S V} = 0$
    4.  $\displaystyle \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}} - V \le 0$
    5.  $-\delta_T \le 0$
    6.  $\delta_T - 1 \le 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    8.  $\mu_1, \mu_2, \mu_3 \ge 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraint have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

    9.  $\displaystyle \mu_1\left( \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}} - V \right) = 0$
    10. $\mu_2 (\delta_T) = 0$
    11. $\mu_3 (\delta_T - 1) = 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## KKT Analysis

    We can now proceed to systematically examine the conditions where various inequality constraints are active or inactive.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### _Idle thrust boundary active_

    In this case: $\mu_2 > 0, \delta_T=0, \mu_1=\mu_3=0$

    It is easy to see that the primal feasibility constraint 3, in other words the horizontal equilibrium, can never be verified.
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

    analysisSurface = np.broadcast_to(
        analysisModel.V_CLarray[np.newaxis, :],  # V along rows
        (plot_utils.meshgrid_n, plot_utils.meshgrid_n),
    )

    plot_options_analysis = {
        "surface": analysisSurface,
        "axes": {
            "x3": {
                "title": r"$V\:(\text{m/s})$",
                "data_key": "V_CLarray",
                "bound": 6 * np.min(analysisSurface),
                "optimum_key": "V_selected",
            },
        },
    }
    return plot_options_analysis, tab_value


@app.cell
def _(tab_value, title_keys):
    mo.stop(tab_value != title_keys[0])

    mo.md(r"""
    ### _Interior solutions_

    If all inequality constraints as inactive, $\mu_1,\mu_2,\mu_3=0$.
    From stationarity condition 2: $\lambda_1=0$. And from stationarity condition 2: $1=0$.
    Therefore, once again, optimal solutions lie on some boundary.
    """).callout()
    return


@app.cell
def _(
    MaxLiftCondition,
    W_selected_analysis,
    analysisModel,
    h_selected_analysis,
    plot_options_analysis,
    tab_value,
    title_keys,
    variables_stack_analysis,
):
    mo.stop(tab_value != title_keys[1])

    mo.vstack(
        [
            mo.md(r"""
    ### _Stall boundary active_

    In this case: $\mu_1 > 0, V=V_s, \mu_2=\mu_3=0$

    From stationarity conditions: $\lambda_1=0 \Rightarrow \mu_1=0$, which is acceptable.

    The minimum airspeed is of course the stall speed, which seems trivial as a result of how we have reformulated the problem.

    $$
    V^* = V_s = \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}
    $$

    The corresponding optimum lift coefficient is $C_L^* = C_{L_\mathrm{max}}$ and the throttle setting is: 

    $$
    \delta_T^* = \frac{ \displaystyle \frac{1}{2}\rho V^3_s S C_{D_0} + \frac{2KW^2}{\rho S V_s} }{ P_{a0} \sigma^\beta} = 
    \frac{W V_s / E_S}{P_{a0}\sigma^\beta}
    $$

    The condition to achieve this is given by $0 \le \delta_T^* \le 1$, where only the right-hand side is relevant.
    This tells that the required power at stall speed has to be less or equal to the available power at stall speed, and is equivalent to either of the two following conditions:

    $$
    \frac{W^{3/2}}{\sigma^{\beta+1/2}} \le  P_{a0} E_S \sqrt{\frac{1}{2}\rho_0 S C_{L_\mathrm{max}}}
    \quad \Leftrightarrow \quad
    \frac{W}{\sigma^{\beta+1/2}} \le \frac{ P_{a0} E_S}{V_{s0}}
    $$
    """),
            variables_stack_analysis,
            analysisModel.plot_grid(
                (
                    MaxLiftCondition(
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
    class MaxLiftCondition(OptimumCondition):
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
            )

            self.CLopt = self.CLopt_selected = Model.aircraft.CLmax

            self.compute_optimal(W, h, Model)
    return (MaxLiftCondition,)


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
    mo.stop(tab_value != title_keys[2])

    mo.vstack(
        [
            mo.md(r"""
    ### _Max throttle boundary active_

    In this case: $\mu_3 > 0, \delta_T=1, \mu_1=\mu_2=0$

    From the stationarity conditions and the complementary slack conditions:

    $$
    \mu_3 = -\lambda_1  P_{a0}\sigma^\beta\\
    1 + \lambda_1 \left( \frac{2KW^2}{\rho S V^2} - \frac{3}{2}\rho V^2SC_{D_0} \right) = 0
    $$

    Therefore, $\mu_3 > 0$ if $\lambda_1 < 0$, and the latter is true when:

    $$
    \frac{3}{2}\rho V^2SC_{D_0} - \frac{2KW^2}{\rho S V^2} < 0
    \quad \Leftrightarrow \quad 3 C_{D_0} - K C_L^2 < 0
    \quad \Leftrightarrow \quad C_L > \sqrt{\frac{3 C_{D_0}}{K}} = \sqrt{3} C_{L_E} = C_{L_P}
    $$

    This means that minimum speed is achieved at max throttle when flying on the induced branch of the power curve, that is with a lift coefficient that is higher than the one for minimum required power ($C_{L_P}$) and lower than $C_{L_\mathrm{max}}$) of course. This means that minimum speed is achieved at max throttle when flying on the induced branch of the power curve, that is with a lift coefficient that is higher than the one for minimum required power ($C_{L_P}$) and lower than $C_{L_\mathrm{max}}$).
    The corresponding aircraft design condition, which basically ensures the existence of the induced branch of the power curve, is therefore:

    $$
    C_{L_P} < C_L \le C_{L_\mathrm{max}}
    \quad \Rightarrow \quad
    C_{L_\mathrm{max}} > \sqrt{\frac{3 C_{D_0}}{K}}
    $$

    The corresponding minimum speed is obtained by solving the following equation:

    $$
    V^* :  P_{a0}\sigma^\beta - \frac{1}{2} \rho S V^3 C_{D_0} - \frac{2KW^2}{\rho S V} = 0
    $$

    which cannot be solved analytically.
    The solution is valid only if $V^* > V_s$.

    The corresponding throttle is $\delta_T^*=1$ and the optimum lift coefficient is: $C_L^* = \frac{2W}{\rho S V^{*2}}$
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
        C1 = Model.aircraft.Pa0 * 1e3 * sigma**Model.aircraft.beta

        C2 = -0.5 * atmos.rho(h) * Model.aircraft.S * Model.aircraft.CD0

        C3 = -2 * Model.aircraft.K * (W**2) / (atmos.rho(h) * Model.aircraft.S)

        # define H(s) and its derivative
        def H(s):
            # H(s) = C1 * s^(3/2) - W * (CD0 + K * s^2)
            return C1 + C2 * (s**3) - C3 / s

        return H

    class MaxThrustCondition(OptimumCondition):
        def __init__(self, W, h, Model):
            CL_maxthrust_star = []

            for hi in Model.aircraft.h_array:
                H = maxthrust_solver(W, hi, Model)

                sol = root_scalar(
                    H,
                    x0=Model.aircraft.CL_P,
                    method="newton",
                    xtol=1e-12,
                    rtol=1e-12,
                    maxiter=1000,
                )

                CL_maxthrust_star.append(sol.root)

            CL_maxthrust_star = np.array(CL_maxthrust_star)
            self.condition = (CL_maxthrust_star > Model.aircraft.CL_E) & (
                CL_maxthrust_star < Model.aircraft.CLmax
            )

            self.CLopt = CL_maxthrust_star[self.condition]

            self.dTopt = 1

            velocity = Model.compute_velocity(W, h, Model.aircraft.CLmax)

            # Safe single-point lookup
            h_idx = np.where(np.isclose(Model.aircraft.h_array, h))[0]

            if len(h_idx) > 0 and self.condition[h_idx[0]]:
                idx_in_masked = np.where(
                    np.isclose(Model.aircraft.h_array[self.condition], h)
                )[0][0]
                self.CLopt_selected = self.CLopt[idx_in_masked]
            else:
                self.CLopt_selected = np.nan

            self.compute_optimal(W, h, Model)
    return (MaxThrustCondition,)


@app.cell
def _(
    W_selected_analysis,
    analysisModel,
    plot_options_analysis,
    tab_value,
    title_keys,
    variables_stack_analysis,
):
    mo.stop(tab_value != title_keys[3])

    mo.vstack(
        [
            mo.md(r"""
    ### _Max throttle and stall boundaries active_

    In this case: $V=V_s, \delta_T=1, \mu_1 > 0, \mu_2 > 0, \mu_3 > 0$

    From the stationarity conditions and the complementary slack conditions:

    $$
    \mu_3 = -\lambda_1  P_{a0}\sigma^\beta > 0 \\
    \mu_1 = 1 + \lambda_1 \left[ \frac{1}{2}\rho V_s^2 S \left( K C_{L_\mathrm{max}}^2 - 3 C_{D_0}\right)\right] > 0
    $$

    It follows that:

    $$
    \frac{1}{\frac{1}{2}\rho V_s^2 S \left( K C_{L_\mathrm{max}}^2 - 3 C_{D_0}\right)} \le \lambda_1 < 0
    $$

    which corresponds to $C_{L_\mathrm{max}} > \sqrt{\frac{3 C_{D_0}}{K}} = C_{L_P}$.

    The condition in which this optimum is achieved is given by the horizontal equilibrium constraint, which states that the required power has to be equal to the available power in stall conditions and at max throttle. This results in the following equation:

    $$
    \frac{W^{3/2}}{\sigma^{\beta+1/2}} =  P_{a0} E_S \sqrt{\frac{1}{2}\rho_0 S C_{L_\mathrm{max}}}
    \quad \Leftrightarrow \quad
    \frac{W}{\sigma^{\beta+1/2}} = \frac{ P_{a0} E_S}{V_{s0}}
    $$
    """),
            variables_stack_analysis,
            analysisModel.plot_grid(
                (MaxLiftThrustCondition(W_selected_analysis, analysisModel),),
                plot_options_analysis,
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
        self.condition = (Model.aircraft.CLmax > Model.aircraft.CL_P) & (
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
        envelopeModel.V_CLarray[np.newaxis, :],
        (plot_utils.meshgrid_n, plot_utils.meshgrid_n),
    )

    plot_options_envelope = {
        "surface": envelopeSurface,
        "axes": {
            "x3": {
                "title": r"$V\:(\text{m/s})$",
                "data_key": "V_CLarray",
                "bound": 6 * np.min(envelopeSurface),
                "optimum_key": "V_selected",
            },
        },
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
    MaxLiftCondition,
    MaxThrustCondition,
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
                    MaxThrustCondition(
                        W_selected_envelope, h_selected_envelope, envelopeModel
                    ),
                    MaxLiftCondition(
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
    _defaults.nav_footer(
        before_file="Steady_Level_Flight/MinSpeed_Jet.py",
        before_title="Minimum Speed Simplified Jet",
        above_file="Steady_Level_Flight/MinSpeed.py",
        above_title="Minimum Speed Homepage",
        above_before=False,
    )
    return


if __name__ == "__main__":
    app.run()
