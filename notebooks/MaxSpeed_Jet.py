import marimo

__generated_with = "0.18.0"
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
    from core.aircraft import (
        available_aircrafts,
        AircraftBase,
        ModelSimplifiedJet,
        OptimumCondition,
    )
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
        atmos,
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
    return ac_table, data


@app.cell
def _(AircraftBase, ModelSimplifiedJet, ac_table, data, plot_utils):
    # Define constants dependent on the ac database. This runs every time another aircraft is selected
    if ac_table.value is not None and ac_table.value.any().any():
        active_selection = ac_table.value.iloc[0]
    else:
        active_selection = data.iloc[0]

    aircraft = AircraftBase(active_selection)

    initialControls = plot_utils.InteractiveElements(aircraft, initial=True)
    initialModel = ModelSimplifiedJet(aircraft)

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
def _(
    W_selected_initial,
    h_selected_initial,
    initialModel,
    initial_CL_slider,
    np,
    plot_utils,
):
    _ = h_selected_initial, W_selected_initial

    initialSurface = np.broadcast_to(
        1 / initialModel.V_CLarray[np.newaxis, :],
        (plot_utils.meshgrid_n, plot_utils.meshgrid_n),
    )

    selected_value = 1 / initialModel.compute_velocity(
        W_selected_initial, h_selected_initial, initial_CL_slider.value
    )

    plot_options_initial = {
        "surface": initialSurface,
        "title": "Maximum speed",
        "axes": {"z": {"label": "1 / V (s/m)"}},
        "factor": 10,
    }
    return plot_options_initial, selected_value


@app.cell(hide_code=True)
def _(mo):
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
        & \quad T_a(V,h) = T_a(h) = T_{a0}\sigma^\beta \\
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## KKT formulation
    To be reconducted in the standard KKT analysis format, the objective function is expressed in terms of the controls by direct elimination of $c_1^\mathrm{eq}$.
    Also, maximizing $V$ is equivalent to minimizing its inverse, $1/V$.
    Therefore, to simplify the calculations, the problem is rewritten as follows:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$
    \begin{aligned}
        \min_{C_L, \delta_T}
        & \quad \frac{1}{V} = \sqrt{\frac{\rho S C_L}{2W}} \\
        \text{subject to}
        & \quad g_1 = \delta_T T_{a0}\sigma^\beta - W \left(\frac{C_{D_0} + KC_L^2}{C_L}\right) = 0 \\
        & \quad h_1 = C_L - C_{L_\mathrm{max}} \le 0 \\
        & \quad h_2 = \delta_T - 1 \le 0 \\
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
def _(
    initialModel,
    initial_CL_slider,
    initial_dT_slider,
    initial_variables_stack,
    mo,
    plot_options_initial,
    selected_value,
):
    mo.md(f"""
    Here you can modify the control variables to understand how it affects the design: {mo.vstack([mo.hstack([initial_dT_slider, initial_CL_slider]), initial_variables_stack, initialModel.plot_initial(plot_options_initial, [initial_CL_slider.value, initial_dT_slider.value, selected_value]).figure])}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Lagrangian function and KKT conditions

    The Lagrangian function combines the objective function with equality constraints using Lagrange multipliers ($\lambda_i$) and inequality constraints using KKT multipliers ($\mu_j$).

    $$
    \begin{aligned}
    \mathcal{L}(C_L, \delta_T, \lambda_1, \mu_1, \mu_2) =
    \quad \sqrt{\frac{\rho S C_L}{2W}}
    & + \\
    & + \lambda_1 \left[\delta_T T_{a0}\sigma^\beta - W \left(\frac{C_{D_0} + KC_L^2}{C_L}\right)\right] + \\
    & + \mu_1 (C_L - C_{L_\mathrm{max}}) + \\
    & + \mu_2 (\delta_T - 1)\\
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A necessary condition for an optimal solution of the optimization problem $(C_L^*, \delta_T^*)$ to exist, the multipliers $\lambda_1, \mu_1, \mu_2$ have to meet the following conditions:

    **A. Stationarity ($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \frac{\partial \mathcal{L}}{\partial C_L} = \frac{1}{2}\sqrt{\rho_0\frac{S}{2}\frac{\sigma}{W}}C_L^{-1/2} - \lambda_1 W\left(\frac{KC_L^2 - C_{D_0}}{C_L^2}\right) + \mu_1 = 0$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1 T_{a0}\sigma^\beta + \mu_2 = 0$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $\displaystyle \delta_T T_{a0}\sigma^\beta - W \left(\frac{C_{D_0} + KC_L^2}{C_L}\right) = 0$
    4.  $C_L - C_{L_\mathrm{max}} \le 0$
    5.  $\delta_T - 1 \le 0$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    6.  $\mu_1, \mu_2\ge 0$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraint have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

    7.  $\mu_1 (C_L - C_{L_\mathrm{max}}) = 0$
    8. $\mu_2 (\delta_T - 1) = 0$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## KKT analysis

    We can now proceed to systematically examine the conditions where various inequality constraints are active or inactive.
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
def _(
    W_selected_analysis,
    analysisModel,
    h_selected_analysis,
    np,
    plot_utils,
    tab,
):
    tab_value = tab.value

    _ = h_selected_analysis, W_selected_analysis

    surface = np.broadcast_to(
        1 / analysisModel.V_CLarray[np.newaxis, :],
        (plot_utils.meshgrid_n, plot_utils.meshgrid_n),
    )

    plot_options_analysis = {
        "surface": surface,
        "factor": 10,
    }
    return plot_options_analysis, tab_value


@app.cell
def _(mo, tab_value, title_keys):
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


@app.cell
def _(
    MaxThrustCondition,
    W_selected_analysis,
    analysisModel,
    h_selected_analysis,
    mo,
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
    ## _Thrust-limited maximum airspeed_

    $\delta_T=1 \quad \Rightarrow \quad \mu_2 > 0$

    $C_L < C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 = 0$

    From stationarity condition (2):

    $$
    \lambda_1 = -\frac{\mu_2}{T_{a0}\sigma^\beta} \lt 0
    $$

    Stationarity condition (1) then becomes:

    $$
    \begin{align}
    \mu_1 &= -\frac{1}{2}\sqrt{\rho_0\frac{S}{2}\frac{\sigma}{W}}C_L^{-1/2} + \lambda_1 W\left(\frac{KC_L^2 - C_{D_0}}{C_L^2}\right) = 0 \nonumber \\
    \Leftrightarrow \quad \lambda_1 &= \frac{\frac{1}{2}\sqrt{\rho_0\frac{S}{2}\frac{\sigma}{W}}C_L^{-1/2}}{W\left(\frac{KC_L^2 - C_{D_0}}{C_L^2}\right)} \lt 0 \quad \Leftrightarrow \frac{1}{KC_L^2 - C_{D_0}} \lt 0 \quad \Leftrightarrow \quad C_L \lt \sqrt{\frac{C_{D_0}}{K}} = C_{L_E} \nonumber
    \end{align}
    $$

    This shows that maximum speed is obtained, intuitively, on the positive (right-hand side) branch of the performance diagram.

    Note: both $C_L \lt C_{L_E}$ and $C_L\lt C_{L_\mathrm{max}}$ can be true in either case of $C_{L_\mathrm{max}} \geq C_{L_E}$. The loosest, and best-case design is of course when $C_{L_E} \lt C_{L_\mathrm{max}}$, meaning that the aircraft is able to fly on the induced (left-hand side) branch of the performance diagram.

    The corresponding optimum value of the $C_L$ is obtained by solving the primal feasibiliy condition (3), resulting in the well known:

    $$
    C_{L_{1, 2}}^* = \frac{T_{a0}\sigma^\beta}{2KW} \left[1\pm\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}\right]
    $$

    Which exists for:

    $$
    1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2 \ge 0
    \quad \Leftrightarrow \quad \frac{W}{\sigma^\beta} \le  T_{a0} E_\mathrm{max}
    $$

    Where we are interested in the lower value, with the - sign. In the case where $C_{L_E} \lt C_{L_\mathrm{max}}$, this value is feasible when:

    $$
    C_L^* \lt C_{L_E} \quad \Leftrightarrow \quad \frac{W}{\sigma^\beta} \lt  T_{a0} E_\mathrm{max}
    $$

    Meaning that the minimum drag at current altitude and weight is less then the available thrust.

    Thus the optimal values are:

    $$
    \delta_T^* = 1, \quad C_L^* = \frac{T_{a0}\sigma^\beta}{2KW} \left[1-\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}\right], \quad \text{for} \:\:\frac{W}{\sigma^\beta} \lt  T_{a0} E_\mathrm{max}, \quad\text{if}\:\: C_{L_\mathrm{max}} \gt \sqrt{\frac{C_{D_0}}{K}}
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
def _(OptimumCondition, aircraft, analysisModel, np):
    class MaxThrustCondition(OptimumCondition):
        def __init__(self, W, h, Model):
            thrust_envelope = Model.compute_thrust(Model.aircraft.h_array)

            self.dTopt = 1

            self.condition = (
                W
                < analysisModel.compute_thrust(analysisModel.aircraft.h_array)
                * aircraft.E_max
            )

            A = thrust_envelope[self.condition] / (2 * Model.aircraft.K * W)
            B = 1 - np.sqrt(
                1 - (W / (Model.aircraft.E_max * thrust_envelope[self.condition])) ** 2
            )

            self.CLopt = A * B

            thrust_selected = Model.compute_thrust(h)

            if W > thrust_selected * Model.aircraft.E_S:
                self.A = thrust_selected / (2 * Model.aircraft.K * W)
                self.B = 1 - np.sqrt(
                    1 - (W / (Model.aircraft.E_max * thrust_selected)) ** 2
                )
                self.CLopt_selected = self.A * self.B
            else:
                self.A = self.B = np.nan
                self.CLopt_selected = np.nan

            self.compute_optimal(W, h, Model)
    return (MaxThrustCondition,)


@app.cell
def _(mo, tab_value, title_keys):
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
    MaxLiftThrustCondition,
    W_selected_analysis,
    analysisModel,
    mass_stack_analysis,
    mo,
    np,
    plot_utils,
    tab_value,
    title_keys,
):
    if tab_value != title_keys[3]:
        mo.stop(True)

    MaxLiftThrust = MaxLiftThrustCondition(W_selected_analysis, analysisModel)
    surface_MaxLiftThrust = np.broadcast_to(
        1 / analysisModel.V_CLarray[np.newaxis, :],
        (plot_utils.meshgrid_n, plot_utils.meshgrid_n),
    )

    plot_options_analysis_MaxLiftThrust = {
        "surface": surface_MaxLiftThrust,
        "factor": 10,
    }

    mo.vstack(
        [
            mo.md(r"""
    ### _Thrust- and lift-limited minimum speed_

    $\delta_T = 1 \quad \Rightarrow \quad \mu_2 > 0$

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 > 0$.

    From the stationary conditions (2):

    $$
    \lambda_1 = -\frac{\mu_2}{T_{a0}\sigma^\beta} < 0
    $$

    From stationary condition (1):

    $$
    \mu_1 = \lambda_1 W\left(\frac{KC_{L_\mathrm{max}}^2 - C_{D_0}}{C_{L_\mathrm{max}}^2}\right) -\frac{1}{2}\sqrt{\rho_0\frac{S}{2}\frac{\sigma}{W}\frac{1}{C_{L_\mathrm{max}}}} \gt 0
    $$

    $$
    \frac{\frac{1}{2}\sqrt{\rho_0\frac{S}{2}\frac{\sigma}{W}\frac{1}{C_{L_\mathrm{max}}}}}{W\left(\frac{KC_{L_\mathrm{max}}^2 - C_{D_0}}{C_{L_\mathrm{max}}^2}\right)} \lt \lambda_1 \lt 0 \quad \Leftarrow \quad C_{L_\mathrm{max}} \lt \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$


    In other words, this condition is verified only if the aircraft would not be able to fly in the condition of maximum aerodynamic efficiency (or minimum drag in steady level flight) because it woudl stall at a higher speed.

    From (3), the same derivation as the previous case results in

    $$
    C_L^* = C_{L_\mathrm{max}}, \quad \delta_T^*=1, \quad \frac{W}{\sigma^\beta} = T_{a0}E_S, \quad \mathrm{if} \quad C_{L_\mathrm{max}} \lt \sqrt{\frac{C_{D_0}}{K}}
    $$
    """),
            mass_stack_analysis,
            analysisModel.plot_grid(
                (MaxLiftThrust,), plot_options_analysis_MaxLiftThrust
            ).figure,
        ]
    ).callout()
    return


@app.cell
def _(OptimumCondition, atmos, np):
    class MaxLiftThrustCondition(OptimumCondition):
        def __init__(self, W, Model, modifyModel=True):
            h_optimum = atmos.altitude(
                (W / (Model.aircraft.Ta0 * 1e3) / Model.aircraft.E_S)
                ** (1 / Model.aircraft.beta)
            )

            if modifyModel:
                Model.update_altitude_dependency(h_optimum)
                Model.update_context(W, h_optimum)

            self.dTopt = 1

            self.hopt_array = np.array([h_optimum])
            self.condition = Model.aircraft.CLmax < Model.aircraft.CL_E

            self.CLopt = self.CLopt_selected = (
                Model.aircraft.CLmax if self.condition else np.nan
            )

            self.compute_optimal(W, h_optimum, Model, True)

            self.cond = 1 if self.condition else np.nan

            self.V_selected = (
                Model.compute_velocity(W, h_optimum, self.CLopt_selected) * self.cond
            )

            self.CLopt_selected = self.CLopt_selected * self.cond
    return (MaxLiftThrustCondition,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Final flight envelope
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now after deriving all the optima for each condition we can summarize the flight envelopes in one graph, as shown below. Experiment with the weight of the aircrarft to understand how the theoretical ceiling for maximum speed moves in the graph.
    """)
    return


@app.cell
def _(ModelSimplifiedJet, aircraft, plot_utils):
    envelopeControls = plot_utils.InteractiveElements(aircraft)

    mass_slider_envelope = envelopeControls.mass_slider
    altitude_slider_envelope = envelopeControls.altitude_slider

    mass_stack_envelope, variables_stack_envelope = envelopeControls.init_layout(
        mass_slider_envelope, altitude_slider_envelope
    )

    envelopeModel = ModelSimplifiedJet(aircraft)
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
def _(W_selected_envelope, envelopeModel, h_selected_envelope, np, plot_utils):
    _ = h_selected_envelope, W_selected_envelope

    envelopeSurface = np.broadcast_to(
        1 / envelopeModel.V_CLarray[np.newaxis, :],
        (plot_utils.meshgrid_n, plot_utils.meshgrid_n),
    )

    plot_options_envelope = {
        "surface": envelopeSurface,
        "factor" : 10
    }
    return (plot_options_envelope,)


@app.cell
def _(MaxLiftThrustCondition, W_selected_envelope, envelopeModel):
    MaxLiftThrustEnvelope = MaxLiftThrustCondition(
        W_selected_envelope, envelopeModel, False
    )
    return (MaxLiftThrustEnvelope,)


@app.cell
def _(MaxLiftThrustEnvelope, plot_utils):
    equality_trace = plot_utils.add_equality((MaxLiftThrustEnvelope,))
    return (equality_trace,)


@app.cell
def _(
    MaxThrustCondition,
    W_selected_envelope,
    envelopeModel,
    equality_trace,
    h_selected_envelope,
    mo,
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
                ),
                plot_options_envelope,
            ).figure.add_traces(equality_trace),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    | Name | Condition | $C_L^*$ | $\delta_T^*$ | $V^*$ |
    |:-|:----------|:-------:|:------------:|:------|
    |Thrust and Lift-limited    | $\displaystyle \frac{W}{\sigma^\beta} =  T_{a0} E_S$ | $C_{L_\mathrm{max}}$ | $1$ | $\displaystyle V_s =\sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}$ |
    |Thrust-limited    | $\displaystyle \frac{W}{\sigma^\beta} \lt  T_{a0} E_\mathrm{max}$ | $\displaystyle \frac{T_{a0}\sigma^\beta}{2KW} \left[1-\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}\right]$ | $1$ | $\displaystyle V_s \sqrt{\frac{2KWC_{L_\mathrm{max}}/T_{a0}\sigma^\beta}{1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}}}$ |
    """)
    return


@app.cell
def _():
    _defaults.nav_footer(
        after_file="MaxSpeed_Prop.py",
        after_title="Maximum Speed Simplified Propeller",
        above_file="MaxSpeed.py",
        above_title="Maximum Speed Homepage",
        above_before=True,
    )
    return


if __name__ == "__main__":
    app.run()
