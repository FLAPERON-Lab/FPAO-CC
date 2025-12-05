import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    # Initialization code that runs before all other cells
    import marimo as mo

    # Import dependencies
    from core import _defaults
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np
    from core import atmos
    from core.aircraft import available_aircrafts, AircraftBase, ModelSimplifiedJet, OptimumCondition
    from core import plot_utils
    # from core.plot_utils import OptimumGridView

    # Set local/online filepath
    _defaults.FILEURL = _defaults.get_url()

    # Plotly dark mode template
    _defaults.set_plotly_template()

    # Data directory
    data_dir = str(mo.notebook_location() / "public" / "AircraftDB_Standard.csv")
    return (
        AircraftBase,
        ModelSimplifiedJet,
        OptimumCondition,
        available_aircrafts,
        data_dir,
        mo,
        np,
        plot_utils,
    )


@app.cell
def _():
    # Set navbar on the right
    _defaults.set_sidebar()
    return


@app.cell
def _(available_aircrafts, data_dir, plot_utils):
    # Define constants, this cell runs once and is not dependent in any way on any interactive element (not even the ac database)
    data = available_aircrafts(data_dir, ac_type="Jet")
    ac_table = plot_utils.InteractiveElements.init_table(data)

    labels = ["Power (kW)", -15]
    hover_name = "P<sub>min</sub>"
    return ac_table, data


@app.cell
def _(AircraftBase, ModelSimplifiedJet, ac_table, data, plot_utils):
    # Define constants dependent on the ac database. This runs every time another aircraft is selected
    if ac_table.value is not None and ac_table.value.any().any():
        active_selection = ac_table.value.iloc[0]
    else:
        active_selection = data.iloc[0]

    aircraft = AircraftBase(active_selection)

    general_controls = plot_utils.InteractiveElements(aircraft, initial=True)
    initialModel = ModelSimplifiedJet(aircraft)

    initial_mass_slider = general_controls.mass_slider
    initial_altitude_slider = general_controls.altitude_slider
    initial_CL_slider = general_controls.CL_slider
    initial_dT_slider = general_controls.dT_slider

    initial_mass_stack, initial_variables_stack = general_controls.init_layout(
        initial_mass_slider, initial_altitude_slider
    )
    return (
        aircraft,
        initial_CL_slider,
        initial_dT_slider,
        initial_variables_stack,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Minimum airspeed: simplified jet aircraft

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
        & \quad T_a(V,h) = T_a(h) = T_{a0}\sigma^\beta \\
    \end{aligned}
    $$
    """)
    return


@app.cell
def _():
    # # Initial Figure
    # fig_initial = go.Figure()

    # # Minimum velocity surface
    # fig_initial.add_traces(
    #     [
    #         go.Surface(
    #             x=aircraft.CL_array,
    #             y=aircraft.dT_array,
    #             z=objective_surface,
    #             opacity=0.9,
    #             name="Velocity",
    #             colorscale="viridis",
    #             cmax=max_colorbar,
    #             cmin=min_colorbar,
    #             colorbar={"title": "Velocity (m/s)"},
    #         ),
    #         go.Scatter3d(
    #             x=aircraft.CL_array,
    #             y=constraint,
    #             z=objective_surface[0],
    #             opacity=1,
    #             mode="lines",
    #             showlegend=False,
    #             line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
    #             name="g1 constraint",
    #         ),
    #         go.Scatter3d(
    #             x=[aircraft.CL_array[-15]],
    #             y=[constraint[-15]],
    #             z=[objective_surface[0, -15]],
    #             opacity=1,
    #             textposition="middle left",
    #             mode="markers+text",
    #             text=["g<sub>1</sub>"],
    #             marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
    #             showlegend=False,
    #             name="g1 constraint",
    #             textfont=dict(size=14, family="Arial"),
    #         ),
    #         go.Scatter3d(
    #             x=[CL_slider.value],
    #             y=[dT_slider.value],
    #             z=[velocity_selected],
    #             mode="markers",
    #             showlegend=False,
    #             marker=dict(
    #                 size=3,
    #                 color="white",
    #                 symbol="circle",
    #             ),
    #             name="Design Point",
    #             hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub> : %{y}<br>V: %{z}<extra>%{fullData.name}</extra>",
    #         ),
    #     ]
    # )
    # camera = dict(eye=dict(x=1.35, y=1.35, z=1.35))

    # fig_initial.update_layout(
    #     scene=dict(
    #         xaxis=dict(
    #             title="C<sub>L</sub> (-)",
    #             range=[plot_utils.xy_lowerbound, aircraft.CLmax],
    #         ),
    #         yaxis=dict(title="δ<sub>T</sub> (-)", range=[plot_utils.xy_lowerbound, 1]),
    #         zaxis=dict(title="V (m/s)", range=[0, aircraft.a_0]),
    #     ),
    # )

    # fig_initial.update_layout(
    #     scene_camera=camera,
    #     title={
    #         "text": f"Minimum airspeed domain for {aircraft.full_name}",
    #         "font": {"size": 25},
    #         "xanchor": "center",
    #         "yanchor": "top",
    #         "x": 0.5,
    #     },
    # )

    # mo.output.clear()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## KKT formulation
    To be reconducted in the standard KKT analysis format, the objective function is expressed in terms of the controls by direct elimination of $c_1^\mathrm{eq}$.
    Also, minimizing $V$ is equivalent to minimizing $V^2$, because the square power function is monotonically increasing.
    Therefore, to simplify the calculations, the problem is rewritten as follows:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$
    \begin{aligned}
        \min_{C_L, \delta_T}
        & \quad V^2 = \frac{2W}{\rho S C_L} \\
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
def _(mo):
    mo.md(r"""
    In the interactive graph below, select a simplified jet aircraft of your choice and experiment in finding an optimum by changing the control variables, $C_L$ and $\delta_T$. The design point is marked in white in the 3D velocity surface.
    """)
    return


@app.cell(hide_code=True)
def _(ac_table):
    # Database cell (1)
    ac_table
    return


@app.cell(hide_code=True)
def _(initial_CL_slider, initial_dT_slider, initial_variables_stack, mo):
    mo.md(f"""
    Here you can modify the control variables to understand how it affects the design: {mo.vstack([mo.hstack([initial_dT_slider, initial_CL_slider]), initial_variables_stack])}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Lagrangian function and KKT conditions

    The Lagrangian function combines the objective function with equality constraints using Lagrange multipliers ($\lambda_i$) and inequality constraints using KKT multipliers ($\mu_j$).

    $$
    \begin{aligned}
    \mathcal{L}(C_L, \delta_T, \lambda_1, \mu_1, \mu_2, \mu_3, \mu_4) =
    \quad \frac{2W}{\rho S C_L}
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
def _(mo):
    mo.md(r"""
    The multipliers $\lambda_1, \mu_1, \mu_2, \mu_3, \mu_4$ have to meet the following conditions for an optimal solution of the optimization problem $(C_L^*, \delta_T^*)$ to exist.

    **A. Stationarity ($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \frac{\partial \mathcal{L}}{\partial C_L} = -\frac{2W}{\rho S C_L^2} + \lambda_1 \left(\frac{C_{D_0}- KC_L^2}{C_L^2}\right) + \mu_1 - \mu_2 = 0$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1 \frac{T_{a0}\sigma^\beta}{W} + \mu_3 - \mu_4 = 0$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
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
def _(mo):
    mo.md(r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    8.  $\mu_1, \mu_2, \mu_3, \mu_4 \ge 0$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraint have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

    9.  $\mu_1 (C_L - C_{L_\mathrm{max}}) = 0$
    10. $\mu_2 (-C_L) = 0$
    11. $\mu_3 (\delta_T - 1) = 0$
    12. $\mu_4 (-\delta_T) = 0$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## KKT analysis

    We can now proceed to systematically examine the conditions where various inequality constraints are active or inactive.
    ### _Interior solutions_

    Assuming that that $0 < C_L < C_{L_\mathrm{max}}$ and $0 < \delta_T < 1$ is equivalent to consider all inequality constraints as inactive.

    Therefore: $\mu_1,\mu_2,\mu_3,\mu_4=0$.

    From stationarity condition (2): $\lambda_1 = 0$.

    It can now be seen that stationarity condition (1) is never verified.

    It can be concluded that the minimum speed cannot be achieved in the interior of the domain.
    The minimum must lie on at least one of the boundaries defined by $C_L = C_{L_\mathrm{max}}$ or $\delta_T = 1$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### _Lower boundary solutions_
    The case where $C_L=0$ and the case where $\delta_T=0$ can be immediately discarded because of the primal feasibility conditions.
    This means that $\mu_2=\mu_4=0$ in all cases.

    We can then proceed with the analysis of the cases where the boundaries $C_L = C_{L_\mathrm{max}}$ and $\delta_T = 1$ are active in any of the three possible combinations.
    """)
    return


@app.cell
def _(ModelSimplifiedJet, aircraft, plot_utils):
    analysisControls = plot_utils.InteractiveElements(aircraft)

    mass_slider_analysis = analysisControls.mass_slider
    altitude_slider_analysis = analysisControls.altitude_slider

    mass_stack_analysis, variables_stack_analysis = analysisControls.init_layout(
        mass_slider_analysis, altitude_slider_analysis
    )

    analysisModel = ModelSimplifiedJet(aircraft)
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
def _(analysisModel, np, plot_utils, tab):
    tab_value = tab.value
    surface = np.broadcast_to(
        analysisModel.V_CLarray[np.newaxis, :], (plot_utils.meshgrid_n, plot_utils.meshgrid_n)
    )
    return surface, tab_value


@app.cell
def _(mo, tab_value, title_keys):
    if tab_value != title_keys[0]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ### _Interior solutions_ 

    Assuming that that $0 < C_L < C_{L_\mathrm{max}}$ and $0 < \delta_T < 1$ is equivalent to consider all inequality constraints as inactive.

    Therefore: $\mu_1,\mu_2,\mu_3,\mu_4=0$. 

    From stationarity condition (2): $\lambda_1 = 0$.

    It can now be seen that stationarity condition (1) is never verified.

    It can be concluded that the minimum speed cannot be achieved in the interior of the domain. 
    The minimum must lie on at least one of the boundaries defined by $C_L = C_{L_\mathrm{max}}$ or $\delta_T = 1$.
    """)
        ]
    ).callout()
    return


@app.cell
def _(
    MaxThrustCondition,
    W_selected_analysis,
    analysisModel,
    h_selected_analysis,
    mo,
    surface,
    tab_value,
    title_keys,
    variables_stack_analysis,
):
    if tab_value != title_keys[2]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ### _Thrust-limited minimum airspeed_

    $\delta_T=1 \quad \Rightarrow \quad \mu_3 > 0$

    $C_L < C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 = 0$

    From stationarity condition (2): 

    $$
    \lambda_1 = -\mu_3\frac{W}{T_{a0}\sigma^\beta} \quad \Rightarrow \quad \lambda_1 < 0
    $$

    Stationarity condition (1) then becomes:

    $$
    \frac{2T_{a0}\sigma^\beta}{\rho S C_L^2} + \mu_3\left( \frac{C_{D_0}-KC_L^2}{C_L^2}\right) = 0
    \quad \text{and } \quad 
    \mu_3>0 
    \quad \Rightarrow \quad 
    C_L^* > \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$

    and implies that the thrust-limited minimum airspeed is obtained strictly on the left branch of the drag performance diagram, at a lift coefficient strictly higher than the one for maximum aerodynamic efficiency.

    The corresponding optimum value of the $C_L$ is obtained by solving the primal feasibility condition (3) and taking the highest of the two solutions:

    $$
    C_L^* = \frac{T_{a0}\sigma^\beta}{2KW} \left[1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}\right]
    $$

    It has still to be verified that $C_L^* < C_{L_\mathrm{max}}$, which depends on the numerical values of the design parameters, and on the current values of the weight and altitude.

    First, this optimum value of the lift coefficient is achievable for 

    $$
    1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2 \ge 0
    \quad \Rightarrow \quad 
    \frac{W}{\sigma^\beta} \le  T_{a0} E_\mathrm{max}
    $$

    The limit equality can be used to calculate the corresponding limit altitude at which the minimum speed is limited by thrust, for a given weight. This is called the _theoretical ceiling_.

    Second, the optimum value is lower than $C_{L_\mathrm{max}}$ if

    $$
    \frac{W}{\sigma^\beta} > T_{a0} E_\mathrm{S}
    $$

    This concludes the analysis for the minimum airspeed of a simplified jet aircraft in the thrust-limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = \frac{T_{a0}\sigma^\beta}{2KW} \left[1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}\right]}, \quad \boxed{\delta_T^*=1}, \quad \text{for} \quad C_L^* > \sqrt{\frac{C_{D_0}}{K}}\quad \text{and} \quad \frac{W}{\sigma^\beta} > T_{a0} E_\mathrm{S}
    $$

    If the conditions stated above are satisfied, the objective function $V$ takes the value: 

    $$
    V_{\mathrm{min}}^* = \sqrt{\frac{4KW^2/\rho S T_{a0}\sigma^\beta}{1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}}}
    = V_s \sqrt{\frac{2KWC_{L_\mathrm{max}}/T_{a0}\sigma^\beta}{1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}}}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.
    """),
            variables_stack_analysis,
            analysisModel.plot_optimum(
                surface, MaxThrustCondition(W_selected_analysis, h_selected_analysis, analysisModel)
            ).figure,
        ]
    ).callout()
    return


@app.cell
def _(OptimumCondition, np):
    class MaxThrustCondition(OptimumCondition):
        def __init__(self, W, h, Model):
            thrust_envelope = Model.compute_thrust(Model.aircraft.h_array)

            self.dTopt = 1

            self.condition = W > (thrust_envelope) * Model.aircraft.E_S

            A = thrust_envelope[self.condition] / (2 * Model.aircraft.K * W)
            B = 1 + np.sqrt(1 - (W / (Model.aircraft.E_max * thrust_envelope[self.condition])) ** 2)

            self.CLopt = A * B

            thrust_selected = Model.compute_thrust(h)

            if W > thrust_selected * Model.aircraft.E_S:
                A = thrust_selected / (2 * Model.aircraft.K * W)
                B = 1 + np.sqrt(1 - (W / (Model.aircraft.E_max * thrust_selected)) ** 2)
                self.CLopt_selected = A * B
            else:
                self.CLopt_selected = np.nan

            self.compute_optimal(W, h, Model)
    return (MaxThrustCondition,)


@app.cell
def _(
    MaxliftCondition,
    W_selected_analysis,
    analysisModel,
    h_selected_analysis,
    mo,
    surface,
    tab_value,
    title_keys,
    variables_stack_analysis,
):
    if tab_value != title_keys[1]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ###_Lift-limited minimum airspeed_

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 > 0$ 

    $0 < \delta_T < 1 \quad \Rightarrow \quad \mu_3 = 0$.

    From stationarity condition (2): $\lambda_1 = 0$.

    From stationarity condition (1): $\displaystyle \mu_1 = \frac{2W}{\rho S C_{L_\mathrm{max}}^2}>0$, which does not depend on the value of $\delta_T$, and is always verified.

    The corresponding value of the throttle is calculated from the primal feasibility condition (3):

    $$
    \delta_T^*
    = \frac{W}{T_{a0}\sigma^\beta} \frac{C_{D_0} + K C^2_{L_\mathrm{max}}}{C_{L_\mathrm{max}}} 
    = \frac{W}{T_{a0}\sigma^\beta} \frac{1}{E_S} 
    $$

    This is valid only if the calculated $\delta_T^*$ is strictly lower than the maximum allowed, which corresponds to:

    $$
    \frac{W}{\sigma^\beta} < T_{a0} E_S
    $$

    The limit equality can be used to calculate the corresponding limit altitude at which the minimum speed is limited by lift, for a given weight.

    The corresponding minimum airspeed is called the _stall speed_.

    $$
    V^* = \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}
    $$

    This concludes the analysis for the minimum airspeed of a simplified jet aircraft in the lift-limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = C_{L_\mathrm{max}}}, \quad \boxed{\delta_T^*= \frac{W}{T_{a0}\sigma^\beta} \frac{1}{E_S}}, \quad \text{for} \quad \frac{W}{\sigma^\beta} < T_{a0} E_S
    $$

    If the conditions stated above are satisfied, the objective function $V$ takes the value: 

    $$
    V_{\mathrm{min}}^* = \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.
    """),
            variables_stack_analysis,
            analysisModel.plot_optimum(
                surface, MaxliftCondition(W_selected_analysis, h_selected_analysis, analysisModel)
            ).figure,
        ]
    ).callout()
    return


@app.cell
def _(OptimumCondition, aircraft, analysisModel):
    class MaxliftCondition(OptimumCondition):
        def __init__(self, W, h, Model):
            self.CLopt = self.CLopt_selected = Model.aircraft.CLmax
            self.dTopt = (
                W
                / Model.aircraft.E_S
                / (Model.aircraft.Ta0 * 1e3)
                / (Model.rhoratio_selected**Model.aircraft.beta)
            )

            self.condition = W < (analysisModel.compute_thrust(analysisModel.aircraft.h_array)) * aircraft.E_S

            self.compute_optimal(W, h, Model)
    return (MaxliftCondition,)


@app.cell
def _(figure_optimum, mass_stack_analysis, mo, tab_value, title_keys):
    if tab_value != title_keys[3]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ### _Thrust- and lift-limited minimum speed_

    $\delta_T = 1 \quad \Rightarrow \quad \mu_3 > 0$

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 > 0$.

    From the stationary conditions (2):

    $$
    \lambda_1 = -\frac{\mu_3}{T_{a0}\sigma^\beta} \quad \Rightarrow \quad \lambda_1 < 0
    $$

    From stationary condition (1): 

    $$
    \mu_1 = \frac{2W}{\rho S C_{L_\mathrm{max}}^2} + \mu_3\frac{W}{T_{a0}\sigma^\beta}\left(\frac{C_{D_0} - K C_{L_\mathrm{max}}^2}{C_{L_\mathrm{max}}^2}\right) > 0 
    \quad \text{if } \quad
    C_{L_\mathrm{max}} < \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$

    In other words, this condition is reached only if the aircraft is designed in such a way that its maximum lift coefficient is lower than the one for maximum aerodynamic efficiency. 
    It is obvious then that, for the same combination of weight and altitude, its stall speed will be higher than the speed for maximum efficiency (and minimum drag), which would then be unreachable for the aircraft in Steady Level Flight.
    This is, of course, an undesired situation to be in, and should not be resulting out of good aerodynamic design.

    The primal feasibility equation (3) returns the expression of the condition where the minimum speed is limited by both thrust and lift capabilities of the aircraft.

    $$
    \frac{W}{\sigma^\beta} = T_{a0} E_S
    $$

    The corresponding value of the airspeed is once again

    $$
    V^* = \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}
    $$

    This concludes the analysis for the minimum airspeed of a simplified jet aircraft in the lift-thrust limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = C_{L_\mathrm{max}}}, \quad \boxed{\delta_T^*= 1}, \quad \text{for} \quad \frac{W}{\sigma^\beta} = T_{a0} E_S
    $$

    If the conditions stated above are satisfied, the objective function $V$ takes the value: 

    $$
    V_{\mathrm{min}}^* = \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.
    """),
            mass_stack_analysis,
            figure_optimum.figure,
        ]
    ).callout()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Final flight envelope
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Now after deriving all the optima for each condition we can summarize the flight envelopes in one graph, as shown below. Experiment with the weight of the aircrarft to understand how the theoretical ceiling for minimum speed moves in the graph.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    | Name | Condition | $C_L^*$ | $\delta_T^*$ | $V^*$ |
    |:- |:----------|:-------:|:------------:|:------|
    |Interior-optima|\ |\ |\ |\ ||
    |Lift-limited    | $\displaystyle \frac{W}{\sigma^\beta} < T_{a0} E_S$ | $C_{L_\mathrm{max}}$ | $\displaystyle \frac{W}{T_{a0}\sigma^\beta} \frac{1}{E_S}$ | $\displaystyle V_s = \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}$ |
    |Thrust and Lift-limited    | $\displaystyle \frac{W}{\sigma^\beta} =  T_{a0} E_S$, $C_{L_\mathrm{max}} < \sqrt{\frac{C_{D_0}}{K}}$ | $C_{L_\mathrm{max}}$ | $1$ | $\displaystyle V_s =\sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}$ |
    |Thrust-limited    | $\displaystyle T_{a0} E_\mathrm{S} < \frac{W}{\sigma^\beta} \le  T_{a0} E_\mathrm{max}$ | $\displaystyle \frac{T_{a0}\sigma^\beta}{2KW} \left[1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}\right]$ | $1$ | $\displaystyle V_s \sqrt{\frac{2KWC_{L_\mathrm{max}}/T_{a0}\sigma^\beta}{1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}}}$ |
    """
    ).center()
    return


@app.cell
def _():
    _defaults.nav_footer(
        after_file="MinSpeed_Prop_KKT.py",
        after_title="Minimum Speed Simplified Propeller",
        above_file="MinSpeed.py",
        above_title="Minimum Speed Homepage",
        above_before=True,
    )
    return


if __name__ == "__main__":
    app.run()
