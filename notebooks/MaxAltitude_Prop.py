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
    CL_grid, dT_grid = np.meshgrid(aircraft.CL_array, aircraft.dT_array)
    return (
        CL_grid,
        aircraft,
        dT_grid,
        initialControls,
        initialModel,
        initial_CL_slider,
        initial_altitude_slider,
        initial_dT_slider,
        initial_mass_slider,
        initial_mass_stack,
    )


@app.cell
def _(
    CL_grid,
    W_selected_initial,
    aircraft,
    dT_grid,
    h_selected_initial,
    initialModel,
):
    _ = h_selected_initial, W_selected_initial

    initialSurface = np.broadcast_to(
        (
            W_selected_initial**1.5
            / (dT_grid * aircraft.Pa0 * 1e3)
            * (aircraft.CD0 + aircraft.K * CL_grid**2)
            / CL_grid**1.5
            * np.sqrt(2 / atmos.rho0 / initialModel.aircraft.S)
        ),
        (plot_utils.meshgrid_n, plot_utils.meshgrid_n),
    )
    return (initialSurface,)


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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Maximum Altitude: simplified propeller aircraft

    $$
    \begin{aligned}
        \min_{C_L, \delta_T}
        & \quad h \\
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
    Here, h does not appear explicitely but we can transform the problem formulation in a convenient way, by knowing $\rho(h)$ is a monotonically decreasing function of h, as shown in the graph below.

    $$
    \begin{aligned}
        \min_{C_L, \delta_T}
        & \quad h  \qquad \Longleftrightarrow \qquad \max_{C_L, \delta_T} \quad \sigma = \frac{\rho(h)}{\rho_0}\\
    \end{aligned}
    $$
    """)
    return


@app.cell
def _(initialModel):
    figure_height_relation = go.Figure()

    figure_height_relation.add_traces(
        [
            go.Scatter(
                x=initialModel.aircraft.h_array * 1e-3,
                y=initialModel.aircraft.rhoratio_array,
                name=r"$\sigma",
            )
        ]
    )

    figure_height_relation.update_layout(
        yaxis=dict(title=r"$\sigma \quad \mathrm{(-)}$", showgrid=True),
        xaxis=dict(title=r"$h \quad\text{(km)}$", showgrid=True),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Moreover, since density is always positive, and $\beta$ as well, we can say, because $\sigma^\beta$ is a monotically increasing function of $\sigma$, minimizing $\sigma^\beta$ minimizes $\sigma$ which is maximizing $h$.

    $$
    \min_{C_L, \delta_T} \sigma  \quad \Longleftrightarrow \quad \min_{C_L, \delta_T} \quad \sigma^\beta \quad \Longleftrightarrow \quad \min_{C_L, \delta_T} \quad \sigma^{\beta + 1/2}
    $$

    We can thus now susbitute the horizontal equilibrium equation in the objective function directly, and then also substitute the expression of $V$ rom the vertical equilibrium, constraint.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## KKT formulation
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The KKT formulation can now be written:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $$
    \begin{aligned}
        \min_{C_L, \delta_T}
        & \quad \sigma^{\beta + 1/2} = \frac{W^{3/2}}{\delta_T P_{a0}}\sqrt{\frac{2}{\rho_0 S}}\left(\frac{C_{D_0} + K C_L^2}{C_L^{3/2}}\right)\\
        \text{subject to}
        & \quad h_1 = C_L - C_{L_\mathrm{max}} \le 0 \\
        & \quad h_2 = \delta_T - 1 \le 0 \\
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The lower bounds for the lift coefficient ($C_L = 0$), and for $\delta_T$ have already been excluded as they cannot comply with the vertical and horizontal constraints respectively.

    As it can be noted, the problem is now formulated to have only inequality constraints due to the bounds on the decision variables. In other words, it is an unconstrained optimization problem in a partially bounded domain.
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
    initial_mass_stack,
):
    mo.md(f"""
    Here you can modify the control variables to understand how it affects the design: {mo.vstack([mo.hstack([initial_dT_slider, initial_CL_slider]), initial_mass_stack, initialModel.plot_initial(initialSurface, factor=1 / np.min(initialSurface)).figure])}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Lagrangian function and KKT conditions

    The Lagrangian function combines the objective function with the inequality constraints using KKT multipliers ($\mu_j$).

    $$
    \begin{aligned}
    \mathcal{L}(C_L, \delta_T, \mu_1, \mu_2) = & \sigma^{\beta +1/2} + \mu_1 (C_L - C_{L_\mathrm{max}}) +\mu_2 (\delta_T - 1)\\
    =&\left[\frac{W^{3/2}}{\delta_T P_{a0}}\sqrt{\frac{2}{\rho_0 S}}\left(\frac{C_{D_0} + K C_L^2}{C_L^{3/2}}\right)\right] +\\
    & + \mu_1 \left(C_L - C_{L_\mathrm{max}}\right) + \\
    & + \mu_2 (\delta_T - 1) \\
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A necessary condition for an optimal solution of the optimization problem $(C_L^*, \delta_T^*)$ to exist, the multipliers $\lambda_1, \mu_1, \mu_2$ have to meet the following conditions:

    **A. Stationarity ($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \frac{\partial \mathcal{L}}{\partial C_L} = \frac{W^{3/2}}{\delta_T P_{a0}}\sqrt{\frac{2}{\rho_0 S}}\left(-\frac{3}{2}C_{D_0}C_L^{-5/2} + \frac{1}{2}KC_L^{-1/2}\right) + \mu_1= 0$

    3.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = - \frac{W^{3/2}}{\delta_T^2 P_{a0}}\sqrt{\frac{2}{\rho_0 S}}\left(\frac{C_{D_0} + K C_L^2}{C_L^{3/2}}\right) + \mu_2= 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $C_L - C_{L_\mathrm{max}} \le 0$
    4.  $\delta_T - 1 \le 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    5.  $\mu_1, \mu_2\ge 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraint have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

    6.  $\mu_1 (C_L - C_{L_\mathrm{max}}) = 0$
    7. $\mu_3 (\delta_T - 1) = 0$
    """)
    return


@app.cell(hide_code=True)
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
    CL_grid,
    W_selected_analysis,
    aircraft,
    analysisModel,
    dT_grid,
    h_selected_analysis,
    tab,
):
    tab_value = tab.value

    _ = h_selected_analysis, W_selected_analysis

    Surface = np.broadcast_to(
        (
            W_selected_analysis**1.5
            / (dT_grid * aircraft.Pa0 * 1e3)
            * (aircraft.CD0 + aircraft.K * CL_grid**2)
            / CL_grid**1.5
            * np.sqrt(2 / atmos.rho0 / analysisModel.aircraft.S)
        ),
        (plot_utils.meshgrid_n, plot_utils.meshgrid_n),
    )
    return Surface, tab_value


@app.cell
def _(tab_value, title_keys):
    if tab_value != title_keys[0] and tab_value != title_keys[1]:
        mo.stop(True)
    mo.md(r"""
    ### _Interior solutions_

    Assuming that that $C_L < C_{L_\mathrm{max}}$ and $\delta_T < 1$ is equivalent to consider all inequality constraints as inactive.

    Therefore: $\mu_1,\mu_2=0$.

    It is clear from stationarity condition 2, that the equation cannot be solved for any value of $\delta_T$.

    It can be concluded that the maximum speed cannot be achieved in the interior of the domain.
    The minimum must lie on at least one of the boundaries defined by $C_L = C_{L_\mathrm{max}}$ or $\delta_T = 1$.

    Moreover, the stationarity condition 2 can be solved for a value of $\delta_T$ only when $\mu_2 \neq 0$, this means it also pointless to investigate the _max-lift condition_ as we would have $\mu_2 = 0$ again.
    """).callout()
    return


@app.cell(hide_code=True)
def _(
    CL_grid,
    W_selected_analysis,
    aircraft,
    analysisModel,
    dT_grid,
    mass_stack_analysis,
    tab_value,
    title_keys,
):
    if tab_value != title_keys[2]:
        mo.stop(True)

    MaxThrust = (MaxThrustCondition(W_selected_analysis, analysisModel),)

    surface_MaxThrust = np.broadcast_to(
        (
            W_selected_analysis**1.5
            / (dT_grid * aircraft.Pa0 * 1e3)
            * (aircraft.CD0 + aircraft.K * CL_grid**2)
            / CL_grid**1.5
            * np.sqrt(2 / atmos.rho0 / analysisModel.aircraft.S)
        ),
        (plot_utils.meshgrid_n, plot_utils.meshgrid_n),
    )

    mo.vstack(
        [
            mo.md(r"""
    ### _Thrust-limited minimum airspeed_

    $C_L < C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 = 0$

    $\delta_T=1 \quad \Rightarrow \quad \mu_2 > 0$

    From stationarity condition (1):

    $$
    C_L^*= \sqrt{\frac{3C_{D_0}}{K}}=C_{L_P}
    $$

    while stationarity condition (2) is always satisfied given $\delta_T = 1$.

    This condition is achievable only if $C_L^* \lt C_{L_\mathrm{max}}$, meaning that it stalls at lower speed than the airpseed for minimum power in steady level flight, for the same weight and altitude, and is therefore able to fly on the induced brach of the power performance diagram.

    The corresponding altitude is given by the density ratio:

    $$
    \displaystyle \sigma^* = \left(\frac{W^{3/2}}{P_{a0}E_{P}}\sqrt{\frac{2}{\rho_0 SC_{L_P}}}\right)^{\frac{1}{\beta+ 1/2}}
    $$

    which depends on the weight. We call this the "theoretical ceiling", by inspecting the equation for the density ratio, the lower the weight, the lower $\sigma$, and thus the higher the altitude $h$ of the ceiling.

    The operational condition is given by:

    $$
    \frac{W^{3/2}}{\sigma^{*^{\beta+1/2}}} = P_{a0}E_P \sqrt{\frac{1}{2}\rho_0 S C_{L_P}}
    $$
    """),
            mass_stack_analysis,
            analysisModel.plot_optimum(
                surface_MaxThrust,
                MaxThrust,
                factor=1 / np.min(surface_MaxThrust),
                constraint=False,
            ).figure,
        ]
    ).callout()
    return


@app.class_definition
class MaxThrustCondition(OptimumCondition):
    def __init__(self, W, Model, modifyModel=True):
        # sigma_exp = (W**1.5) / Pa0 /?? E_P / np.sqrt(0.5 * atmos.rho0 * S * CL_P)

        sigma_opt = (
            W**1.5
            / (Model.aircraft.Pa0 * 1e3)
            / Model.aircraft.E_P
            / (0.5 * atmos.rho0 * Model.aircraft.S * Model.aircraft.CL_P)
        ) ** (1 / (Model.aircraft.beta + 0.5))

        h_optimum = (
            atmos.altitude(sigma_opt)
            if sigma_opt > atmos.rhoratio(atmos.hmax) and atmos.altitude(sigma_opt) > 0.0
            else 0.0
        )

        if modifyModel:
            Model.update_altitude_dependency(h_optimum)
            Model.update_context(W, h_optimum)

        self.CLopt = self.CLopt_selected = Model.aircraft.CL_P if h_optimum != 0.0 else np.nan
        self.dTopt = 1

        self.hopt_array = np.array([h_optimum])
        self.condition = Model.aircraft.CL_P < Model.aircraft.CLmax

        self.compute_optimal(W, h_optimum, Model, True)

        self.cond = 1 if self.condition else np.nan

        self.V_selected = Model.compute_velocity(W, h_optimum, self.CLopt_selected) * self.cond

        self.CLopt_selected = self.CLopt_selected * self.cond


@app.cell
def _(
    Surface,
    W_selected_analysis,
    analysisModel,
    mass_stack_analysis,
    tab_value,
    title_keys,
):
    if tab_value != title_keys[2]:
        mo.stop(True)

    MaxLiftThrust = (MaxLiftThrustCondition(W_selected_analysis, analysisModel),)

    mo.vstack(
        [
            mo.md(r"""
    ### _Thrust- and lift-limited minimum speed_

    $\delta_T = 1 \quad \Rightarrow \quad \mu_3 > 0$

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 > 0$.

    From the stationary conditions (1):

    $$
    \mu_1 = \frac{W^{3/2}}{\delta_T P_{a0}}\sqrt{\frac{2}{\rho_0 S}}\left(\frac{3}{2}C_{D_0}C_L^{-5/2} - \frac{1}{2}KC_L^{-1/2}\right) \gt 0 \quad \Longleftrightarrow \quad C_{L_\mathrm{max}} \lt \sqrt{\frac{3C_{D_0}}{K}} = C_{L_{E}}
    $$

    In this case the aircraft stalls at a higher speed than the one for minimum required power in steady level flight, for the same altitude and weight.

    The corresponding altitude is given by the density ratio:

    $$
    \displaystyle \sigma^* = \left(\frac{W^{3/2}}{P_{a0}E_{S}}\sqrt{\frac{2}{\rho_0 SC_{L_\mathrm{max}}}}\right)^{\frac{1}{\beta+ 1/2}}
    $$

    While the operational condition is given by:

    $$
    \frac{W^{3/2}}{\sigma^{*^{\beta+1/2}}} = P_{a0}E_S \sqrt{\frac{1}{2}\rho_0 S C_{L_\mathrm{max}}}
    $$
    """),
            mass_stack_analysis,
            analysisModel.plot_optimum(
                Surface,
                MaxLiftThrust,
                factor=1 / np.min(Surface),
                constraint=False,
            ).figure,
        ]
    ).callout()
    return


@app.class_definition
class MaxLiftThrustCondition(OptimumCondition):
    def __init__(self, W, Model, modifyModel=True):
        # sigma_exp = (W**1.5) / Pa0 /?? E_P / np.sqrt(0.5 * atmos.rho0 * S * CL_P)

        sigma_opt = (
            W**1.5
            / (Model.aircraft.Pa0 * 1e3)
            / Model.aircraft.E_S
            / (0.5 * atmos.rho0 * Model.aircraft.S * Model.aircraft.CLmax)
        ) ** (1 / (Model.aircraft.beta + 0.5))

        h_optimum = (
            atmos.altitude(sigma_opt)
            if sigma_opt > atmos.rhoratio(atmos.hmax) and atmos.altitude(sigma_opt) > 0.0
            else 0.0
        )

        if modifyModel:
            Model.update_altitude_dependency(h_optimum)
            Model.update_context(W, h_optimum)

        self.CLopt = self.CLopt_selected = Model.aircraft.CLmax if h_optimum != 0.0 else np.nan
        self.dTopt = 1

        self.hopt_array = np.array([h_optimum])
        self.condition = Model.aircraft.CLmax < Model.aircraft.CL_E

        self.compute_optimal(W, h_optimum, Model, True)

        self.cond = 1 if self.condition else np.nan

        self.V_selected = Model.compute_velocity(W, h_optimum, self.CLopt_selected) * self.cond

        self.CLopt_selected = self.CLopt_selected * self.cond


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

    envelopeModel = ModelSimplifiedProp(aircraft)
    return (
        altitude_slider_envelope,
        envelopeControls,
        envelopeModel,
        mass_slider_envelope,
        mass_stack_envelope,
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
def _(
    CL_grid,
    W_selected_envelope,
    aircraft,
    dT_grid,
    envelopeModel,
    h_selected_envelope,
):
    _ = h_selected_envelope, W_selected_envelope

    envelopeSurface = np.broadcast_to(
        (
            W_selected_envelope**1.5
            / (dT_grid * aircraft.Pa0 * 1e3)
            * (aircraft.CD0 + aircraft.K * CL_grid**2)
            / CL_grid**1.5
            * np.sqrt(2 / atmos.rho0 / envelopeModel.aircraft.S)
        ),
        (plot_utils.meshgrid_n, plot_utils.meshgrid_n),
    )
    return (envelopeSurface,)


@app.cell
def _(W_selected_envelope, envelopeModel):
    MaxThrustEnvelope = MaxThrustCondition(W_selected_envelope, envelopeModel, False)
    MaxLiftThrustEnvelope = MaxLiftThrustCondition(W_selected_envelope, envelopeModel, False)
    return MaxLiftThrustEnvelope, MaxThrustEnvelope


@app.cell
def _(MaxLiftThrustEnvelope, MaxThrustEnvelope):
    equality_trace = plot_utils.add_equality(
        (
            MaxThrustEnvelope,
            MaxLiftThrustEnvelope,
        )
    )
    return (equality_trace,)


@app.cell
def _(
    W_selected_analysis,
    analysisModel,
    envelopeModel,
    envelopeSurface,
    equality_trace,
    mass_stack_envelope,
):
    mo.vstack(
        [
            mass_stack_envelope,
            envelopeModel.plot_optimum(
                envelopeSurface,
                (MaxThrustCondition(W_selected_analysis, analysisModel, False),),
                factor=1 / np.min(envelopeSurface),
                constraint=False,
            ).figure.add_traces(equality_trace),
        ]
    )
    return


@app.cell
def _():
    _defaults.nav_footer(
        before_file="MaxAltitude_Jet.py",
        before_title="Maximum Simplified Jet",
        above_file="MinSpeed.py",
        above_title="Maximum Altitude Homepage",
        above_before=False,
    )
    return


if __name__ == "__main__":
    app.run()
