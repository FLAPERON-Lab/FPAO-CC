import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")

with app.setup:
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
    data_dir = str(mo.notebook_location() / "public" / "AircraftDB_Standard.csv")


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

    labels = ["Drag (N)", -15]
    hover_name = "D<sub>min</sub>"
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
def _(W_selected_initial, h_selected_initial, initialModel):
    _ = h_selected_initial, W_selected_initial

    initialSurface = np.broadcast_to(
        initialModel.drag_curve[np.newaxis, :],
        (plot_utils.meshgrid_n, plot_utils.meshgrid_n),
    )
    return (initialSurface,)


@app.cell
def _():
    mo.md(r"""
    # Maximum airspeed: simplified jet aircraft

    $$
    \begin{aligned}
        \max_{C_L, \delta_T}
        & \quad V \\
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


@app.cell
def _():
    mo.md(r"""
    ## KKT formulation
    To be reconducted in the standard KKT analysis format, the objective function is expressed in terms of the controls by direct elimination of $c_1^\mathrm{eq}$.
    Also, maximizing $V$ is equivalent to minimizing its inverse, $1/V$.
    Therefore, to simplify the calculations, the problem is rewritten as follows:
    """)
    return


@app.cell
def _():
    mo.md(r"""
    $$
    \begin{aligned}
        \min_{C_L, \delta_T}
        & \quad \frac{1}{V} = \sqrt{\frac{\rho S C_L}{2W}} \\
        \text{subject to}
        & \quad g_1 = \delta_T P_{a0}\sigma^\beta - \frac{W^{3/2}}{\sigma^{1/2}}\sqrt{\frac{2}{\rho_0S}} \left(C_{D_0}C_L^{-3/2} + KC_L^{1/2}\right) = 0 \\
        & \quad h_1 = C_L - C_{L_\mathrm{max}} \le 0 \\
        & \quad h_2 = \delta_T - 1 \le 0 \\
    \end{aligned}
    $$
    """)
    return


@app.cell
def _():
    mo.md(r"""
    In the interactive graph below, select a simplified propeller aircraft of your choice and experiment in finding an optimum by changing the control variables, $C_L$ and $\delta_T$. The design point is marked in white in the 3D velocity surface.
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
    initialSurface,
    initial_CL_slider,
    initial_dT_slider,
    initial_variables_stack,
):
    mo.md(f"""
    Here you can modify the control variables to understand how it affects the design: {mo.vstack([mo.hstack([initial_dT_slider, initial_CL_slider]), initial_variables_stack, initialModel.plot_initial(initialSurface).figure])}
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Lagrangian function and KKT conditions

    The Lagrangian function combines the objective function with equality constraints using Lagrange multipliers ($\lambda_i$) and inequality constraints using KKT multipliers ($\mu_j$).

    $$
    \begin{aligned}
    \mathcal{L}(C_L, \delta_T, \lambda_1, \mu_1, \mu_2) =
    \quad \sqrt{\frac{\rho S C_L}{2W}}
    & + \\
    & + \lambda_1 \left[\delta_T P_{a0}\sigma^\beta - \frac{W^{3/2}}{\sigma^{1/2}}\sqrt{\frac{2}{\rho_0S}} \left(C_{D_0}C_L^{-3/2} + KC_L^{1/2}\right)\right] + \\
    & + \mu_1 (C_L - C_{L_\mathrm{max}}) + \\
    & + \mu_2 (\delta_T - 1)\\
    \end{aligned}
    $$
    """)
    return


@app.cell
def _():
    mo.md(r"""
    A necessary condition for an optimal solution of the optimization problem $(C_L^*, \delta_T^*)$ to exist, the multipliers $\lambda_1, \mu_1, \mu_2$ have to meet the following conditions:

    **A. Stationarity ($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \frac{\partial \mathcal{L}}{\partial C_L} = \frac{1}{2}\sqrt{\rho_0\frac{S}{2}\frac{\sigma}{W}}C_L^{-1/2} - \lambda_1 \frac{W^{3/2}}{\sigma^{1/2}}\sqrt{\frac{2}{\rho_0S}} \left(-\frac{3}{2}C_{D_0}C_L^{-5/2} + \frac{1}{2}KC_L^{-1/2}\right) + \mu_1 = 0$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1 P_{a0}\sigma^\beta + \mu_2 = 0$
    """)
    return


@app.cell
def _():
    mo.md(r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $\displaystyle \delta_T P_{a0}\sigma^\beta - \frac{W^{3/2}}{\sigma^{1/2}}\sqrt{\frac{2}{\rho_0S}} \left(C_{D_0}C_L^{-3/2} + KC_L^{1/2}\right) = 0$
    4.  $C_L - C_{L_\mathrm{max}} \le 0$
    5.  $\delta_T - 1 \le 0$
    """)
    return


@app.cell
def _():
    mo.md(r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    6.  $\mu_1, \mu_2\ge 0$
    """)
    return


@app.cell
def _():
    mo.md(r"""
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraint have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

    7.  $\mu_1 (C_L - C_{L_\mathrm{max}}) = 0$
    8. $\mu_2 (\delta_T - 1) = 0$
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## KKT analysis

    We can now proceed to systematically examine the conditions where various inequality constraints are active or inactive.
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
        1 / analysisModel.V_CLarray[np.newaxis, :],
        (plot_utils.meshgrid_n, plot_utils.meshgrid_n),
    )
    return surface, tab_value


@app.cell
def _(tab_value, title_keys):
    if tab_value != title_keys[0]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ### _Interior solutions_

    Assuming that that $C_L < C_{L_\mathrm{max}}$ and $\delta_T < 1$ is equivalent to consider all inequality constraints as inactive.

    Therefore: $\mu_1,\mu_2=0$.

    From stationarity condition (2): $\lambda_1 = 0$.

    It can now be seen that stationarity condition (1) is never verified.

    It can be concluded that the maximum speed cannot be achieved in the interior of the domain. The maximum speed must lie on at least one of the boundaries defined by $C_L = C_{L_\mathrm{max}}$ or $\delta_T = 1$.
    """)
        ]
    ).callout()
    return


@app.cell(hide_code=True)
def _(
    MaxThrustCondition,
    W_selected_analysis,
    analysisModel,
    h_selected_analysis,
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
    ## _Thrust-limited maximum airspeed_

    $\delta_T=1 \quad \Rightarrow \quad \mu_2 > 0$

    $C_L < C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 = 0$

    From stationarity condition (2):

    $$
    \lambda_1 = -\frac{\mu_2}{P_{a0}\sigma^\beta} \lt 0
    $$

    Stationarity condition (1) then becomes:

    $$
    \begin{align}
    \lambda_1 &= \frac{\frac{\rho_0S}{2}\frac{\sigma^{1/2}}{W^{3/2}}C_L^{2}}{kC_L^2-3C_{D_0}} \lt 0 \quad \mathrm{for} \quad C_L \lt \sqrt{\frac{3C_{D_0}}{K}} = \sqrt{3}C_{L_E} = C_{L_P} \nonumber
    \end{align}
    $$

    This shows that maximum speed is obtained, intuitively, on the positive (right-hand side) branch of the performance diagram.

    The loosest condition is $C_{L_P} \lt C_{L_{}\mathrm{max}}$.

    The corresponding optimum value of the $C_L$ is obtained by solving the primal feasibiliy condition (3), having $\delta_T = 1$:

    $$
    P_{a0}\sigma^\beta - \frac{W^{3/2}}{\sigma^{1/2}}\sqrt{\frac{2}{\rho_0S}} \left(C_{D_0}C_L^{-3/2} + KC_L^{1/2}\right) = 0
    $$

    where it is impractical to obtain analytic solutions. The previous function and the conditions where it intercepts the y = 0 axis can therefore be studied graphically, as a function of $C_L$ (for different values of $W$ and $\sigma$) on the performance diagram.

    The operational condition is also found numerically by setting the numerical soltuion $C_L^*\lt C_{L_P}$
    The conditions
    Thus the optimal values are:

    $$
    \delta_T^* = 1, \quad C_L^* = \mathrm{numerically \: solved}, \quad \text{for} \:\:\frac{W^{1/2}}{\sigma^{\beta+1/2}} \lt  \mathrm{numerically \: solved}, \quad\text{if}\:\: C_{L_\mathrm{max}} \gt \sqrt{\frac{C_{D_0}}{K}}
    $$

    """),
            variables_stack_analysis,
            analysisModel.plot_optimum(
                surface,
                (MaxThrustCondition(W_selected_analysis, h_selected_analysis, analysisModel),),
            ).figure,
        ]
    ).callout()
    return


@app.cell
def _():
    def maxthrust_solver(W, h, Model):
        sigma = atmos.rhoratio(h)

        function = lambda CL: Model.aircraft.Pa0 * 1e3 * sigma**Model.aircraft.beta - W**1.5 / (
            sigma**0.5
        ) * np.sqrt(2 / atmos.rho0 / Model.aircraft.S) * (Model.aircraft.CD0 + Model.aircraft.K * CL**2) / (
            CL ** (3 / 2)
        )

        return function


    class MaxThrustCondition(OptimumCondition):
        def __init__(self, W, h, Model):
            CL_maxthrust_star = []

            for hi in Model.aircraft.h_array:
                func = maxthrust_solver(W, hi, Model)
                CL_sol = root_scalar(func, x0=0.04).root
                CL_maxthrust_star.append(CL_sol)

            CL_maxthrust_star = np.array(CL_maxthrust_star)

            velocity = Model.compute_velocity(W, h, Model.aircraft.CLmax)

            self.dTopt = 1

            self.condition = (Model.aircraft.CL_E < Model.aircraft.CLmax) & (~np.isnan(CL_maxthrust_star))

            self.CLopt = CL_maxthrust_star[self.condition]

            idx = np.where(np.isclose(Model.aircraft.h_array, h))[0][0]
            self.CLopt_selected = self.CLopt[idx]

            self.compute_optimal(W, h, Model, False)
    return (MaxThrustCondition,)


@app.cell
def _(tab_value, title_keys):
    if tab_value != title_keys[1]:
        mo.stop(True)

    mo.vstack(
        [
            mo.md(r"""
    ###_Lift-limited minimum airspeed_

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 > 0$

    $0 < \delta_T < 1 \quad \Rightarrow \quad \mu_2 = 0$.

    From stationarity condition (2): $\lambda_1 = 0$.

    From stationarity condition (1):

    $$
    \mu_1 = -\frac{1}{2}\sqrt{\frac{\rho_0 S \sigma }{2WC_{L_\mathrm{max}}}}>0, \quad \mathrm{for} \quad C_{L_\mathrm{max}} \lt 0, \mathrm{impossible}
    $$

    The solution cannot be obtained at $C_{L_\mathrm{max}}$, which is intuitive. As a matter of fact:

    $$
    \min_{C_L} \frac{1}{V} = \sqrt{\frac{\rho S C_L}{2W}} \quad \Leftrightarrow \quad \min_{C_L} \sqrt{C_L} \quad \Leftrightarrow \quad \min_{C_L} C_L
    $$
    """)
        ]
    ).callout()
    return


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

    mo.vstack(
        [
            mo.md(r"""
    ### _Thrust- and lift-limited minimum speed_

    $\delta_T = 1 \quad \Rightarrow \quad \mu_2 > 0$

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 > 0$.

    From the stationary conditions (2):

    $$
    \lambda_1 = -\frac{\mu_2}{P_{a0}\sigma^\beta} < 0
    $$

    From stationary condition (1):

    $$
    \mu_1 =\lambda_1 \frac{W^{3/2}}{\sigma^{1/2}}\sqrt{\frac{2}{\rho_0S}} \left(-\frac{3}{2}C_{D_0}C_{L_\mathrm{max}}^{-5/2} + \frac{1}{2}KC_{L_\mathrm{max}}^{-1/2}\right) -  \frac{1}{2}\sqrt{\rho_0\frac{S}{2}\frac{\sigma}{W}}C_{L_\mathrm{max}}^{-1/2}\gt 0
    $$

    $$
    \mathrm{for}\quad \quad\frac{\rho_0\frac{S}{2}\frac{\sigma^{1/2}}{W^{3/2}}C_{L_\mathrm{max}}^2}{KC_{L_\mathrm{max}}^2 - 3C_{D_0}} \lt \lambda_1 \lt 0 \quad \Leftrightarrow \quad C_{L_\mathrm{max}} \lt \sqrt{\frac{3C_{D_0}}{K}} = \sqrt{3}C_{L_E} = C_{L_P}
    $$


    In order for this case to occur, the aircraft has to be designed to stall at a higher speed than the one for minimum power, in the same conditions of weight and altitude. $C_{L_\mathrm{max}}$ becomes the limiting $C_L$ when maximizing speed, as it is not possible to lower it even more towards C_{L_P}.

    $$
    C_L^* = C_{L_\mathrm{max}}, \quad \delta_T^*=1, \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}} = P_{a0}E_S\sqrt{\frac{\rho_0 S}{2}C_{L_{\mathrm{max}}}}, \quad \mathrm{if} \quad C_{L_\mathrm{max}} \lt C_{L_P}
    $$
    """),
            mass_stack_analysis,
            analysisModel.plot_optimum(surface_MaxLiftThrust, MaxLiftThrust).figure,
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
        h_optimum = atmos.altitude(sigma_opt) if sigma_opt > atmos.rhoratio(atmos.hmax) else 0.0

        if modifyModel:
            Model.update_altitude_dependency(h_optimum)
            Model.update_context(W, h_optimum)

        self.dTopt = 1

        self.hopt_array = np.array([h_optimum])
        self.condition = (
            (Model.aircraft.CLmax < Model.aircraft.CL_P)
            & (sigma_opt < atmos.rhoratio(atmos.hmax))
        )

        self.CLopt = self.CLopt_selected = Model.aircraft.CLmax if self.condition else np.nan

        self.compute_optimal(W, h_optimum, Model, True)

        self.cond = 1 if ((sigma_opt > atmos.rhoratio(atmos.hmax)) and self.condition) else np.nan

        self.V_selected = Model.compute_velocity(W, h_optimum, self.CLopt_selected) * self.cond

        self.CLopt_selected = self.CLopt_selected * self.cond


@app.cell
def _():
    mo.md(r"""
    Now after deriving all the optima for each condition we can summarize the flight envelopes in one graph, as shown below. Experiment with the weight of the aircrarft to understand how the theoretical ceiling for maximum speed moves in the graph.
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
    return (envelopeSurface,)


@app.cell
def _(W_selected_envelope, envelopeModel):
    MaxLiftThrustEnvelope = MaxLiftThrustCondition(W_selected_envelope, envelopeModel, False)
    return (MaxLiftThrustEnvelope,)


@app.cell
def _(MaxLiftThrustEnvelope):
    equality_trace = plot_utils.add_equality((MaxLiftThrustEnvelope,))
    return (equality_trace,)


@app.cell
def _(
    MaxThrustCondition,
    W_selected_envelope,
    envelopeModel,
    envelopeSurface,
    equality_trace,
    h_selected_envelope,
    variables_stack_envelope,
):
    mo.vstack(
        [
            variables_stack_envelope,
            envelopeModel.plot_optimum(
                envelopeSurface,
                (MaxThrustCondition(W_selected_envelope, h_selected_envelope, envelopeModel),),
            ).figure.add_traces(equality_trace),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## Summary

    | Name | Condition | $C_L^*$ | $\delta_T^*$ | $V^*$ |
    |:-|:----------|:-------:|:------------:|:------|
    |Thrust and Lift-limited    | $\displaystyle \frac{W^{3/2}}{\sigma^{\beta+1/2}} = P_{a0}E_S\sqrt{\frac{\rho_0 S}{2}C_{L_{\mathrm{max}}}}$ | $C_{L_\mathrm{max}}$ | $1$ | $\displaystyle V_s =\sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}$ |
    |Thrust-limited    | $\displaystyle \mathrm{numerical}$ | $\displaystyle \mathrm{numerical}$ | $1$ | $\displaystyle \mathrm{numerical}$ |
    """)
    return


@app.cell
def _():
    _defaults.nav_footer(
        before_file="MaxSpeed_Jet.py",
        before_title="Maximum Speed Simplified Jet",
        above_file="MaxSpeed.py",
        above_title="Maximum Speed Homepage",
        above_before=False,
    )
    return


if __name__ == "__main__":
    app.run()
