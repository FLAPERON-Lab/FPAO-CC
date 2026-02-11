import marimo

__generated_with = "0.18.0"
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


@app.cell
def _():
    # Set navbar on the right
    _defaults.set_sidebar()
    return


@app.cell
def _():
    # Define constants, this cell runs once and is not dependent in any way on any interactive element (not even the ac database)
    data = available_aircrafts(data_dir, ac_type="Jet")
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
        active_selection,
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
        initialModel.power_required[np.newaxis, :],
        (plot_utils.meshgrid_n, plot_utils.meshgrid_n),
    )

    selected_value = initialModel.compute_drag(
        W_selected_initial, initial_CL_slider.value
    ) * initialModel.compute_velocity(
        W_selected_initial, h_selected_initial, initial_CL_slider.value
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
    # Minimum Power Required: simplified jet aircraft

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

    Moreover, we know $\delta_T=C_L=0$ does not correspond to a sensible solution, thus we can write:

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
        & \quad g_1 = T - \frac{W}{E}  =\delta_T T_{a0}\sigma^\beta - W\frac{C_{D_0} + K C_L^2}{C_L} = 0 \\
        & \quad h_1 = C_L - C_{L_\mathrm{max}} \le 0 \\
        & \quad h_2 = \delta_T - 1 \le 0 \\
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In the interactive graph below, select a simplified jet aircraft of your choice and experiment in finding an optimum by changing the control variables, $C_L$ and $\delta_T$. The design point is marked in white in the 3D power surface.
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
    \mathcal{L}(C_L, \delta_T, \lambda_1, \mu_1, \mu_2) = & P + \lambda_1 \left[T - D\right]+ \mu_1 (C_L - C_{L_\mathrm{max}}) +\mu_2 (\delta_T - 1)\\
    =&\quad \sqrt{\frac{2W^3}{\rho S}}\left(C_{D_0} C_L^{-3/2}+K C_L^{1/2}\right) +\\
    & + \lambda_1 \left[\delta_T T_{a0}\sigma^\beta - W\frac{C_{D_0} + K C_L^2}{C_L}\right] + \\
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

    1. $\displaystyle \frac{\partial \mathcal{L}}{\partial C_L} = \sqrt{\frac{2W^3}{\rho S}}\left(-\frac{3}{2}C_{D_0}C_L^{-5/2} + \frac{1}{2} K C_L^{-1/2}\right) - \lambda_1 W \left(\frac{KC_L^2 -C_{D_0}}{C_L^2}\right) + \mu_1= 0$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1 T_{a0}\sigma^\beta+ \mu_2= 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $\displaystyle \delta_T T_{a0}\sigma^\beta - W \frac{C_{D_0} + K C_L^2}{C_L} = 0$
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
    8. $\mu_3 (\delta_T - 1) = 0$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## KKT analysis

    We can now proceed to systematically examine the conditions where various inequality constraints are active
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

    render_interior = mo.vstack(
        [
            mo.md(r"""
    ### _Interior optimum_

    In this case: $C_L \lt C_{L_{\mathrm{max}}}$, $\delta_T \lt 1$, $\mu_1=\mu_2= 0$

    from stationarity condition (2): $\lambda_1 = 0$

    from stationarity condition (1):

    $$
    -\frac{3}{2}C_{D_0} C_L^{-5/2}+\frac{1}{2}KC_L^{-1/2}= 0 \quad \Rightarrow \quad KC_L^2 = 3C_{D_0} \quad \Rightarrow \quad C_L^* = \sqrt{\frac{3C_{D_0}}{K}} = \sqrt{3}C_{L_E} = C_{L_P}
    $$

    The corresponding $\delta_T$ value is obtained from primal feasibility constraint (3):

    $$
    \delta_T^* = \frac{W}{T_{a0}\sigma^\beta} \left(\frac{C_{D_0}+K \cdot 3C_{D_0}/K}{\sqrt{3C_{D_0}/K}}\right) = \frac{W}{T_{a0}\sigma^\beta}\sqrt{\frac{16C_{D_0}K}{3}} = \sqrt{\frac{4}{3}}\frac{W}{E_{\mathrm{max}}}\frac{1}{T_{a0}\sigma^\beta} = \frac{W}{E_{\mathrm{P}}}\frac{1}{T_{a0}\sigma^\beta}
    $$

    Where: $\displaystyle E_{\mathrm{P}} = \frac{\sqrt{3}}{2}E_{\mathrm{max}}$

    This is valid for:

    $$
    \delta_T^*\lt 1 \quad \Leftrightarrow \quad \frac{W}{\sigma^\beta} \lt E_{\mathrm{P}}T_{a0} = \frac{\sqrt{3}}{2}E_{\mathrm{max}}T_{a0}
    $$

    Finally, the value of the objective function $P$ can now be calculated:

    $$
    \displaystyle P^*_{\mathrm{min}} = 4 \sqrt{\frac{2W^3}{S\sigma\rho_0}}\sqrt[4]{\frac{C_{D_0} K^3}{27}}
    $$

    This concludes the analysis for the minimum power of a simplified jet aircraft in the domain's interior. Below is a summary of the optima:

    $$
    \boxed{C_L^* = \sqrt{\frac{3C_{D_0}}{K}}}, \quad \boxed{\delta_T^*= \frac{W}{T_{a0}\sigma^\beta}\sqrt{\frac{16C_{D_0}K}{3}}=\frac{W}{E_{\mathrm{P}}}\frac{1}{T_{a0}\sigma^\beta}}, \quad \text{for}\quad  C_{L_\mathrm{max}} \gt C_{L_P}\quad \text{and} \quad \frac{W}{\sigma^\beta} \lt \frac{\sqrt{3}}{2}E_\mathrm{max}T_{a0}
    $$

    With the optimal value for minimum power:

    $$
    P_{\mathrm{min}}^* = 4 \sqrt{\frac{2W^3}{S\sigma\rho_0}}\sqrt[4]{\frac{C_{D_0} K^3}{27}}
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
            mo.md(
                r"""Notice how $C_{L_P}$ (minimum power) $\gt$ $C_{L_E}$ (minimum drag) but $E_\mathrm{P} \lt E_{\mathrm{max}}$ ($E = C_L/C_D$) because the drag coefficient increases more rapidly than $C_L$, as $C_D \propto C_L^2$. Thus, the range of $W/\sigma^\beta$ for which it is possible to fly at minimum power is smaller ($\sqrt{3}/2\lt 1$) than the one for which it is possible to fly at minimum drag. You can check this by increasing the weight of the aircraft here and in [Minimum Drag (simplified Jet)](?file=MinDrag_Jet.py) and finding out at what altitude it is not possible to fly at the optimum anymore, make sure to compare the same aircraft at the same weight."""
            ),
        ]
    )

    render_interior.callout()
    return


@app.cell
def _(aircraft, analysisModel):
    class InteriorCondition(OptimumCondition):
        def __init__(self, W, h, Model):
            self.CLopt = self.CLopt_selected = Model.aircraft.CL_P
            self.dTopt = (
                W
                / Model.aircraft.E_P
                / (Model.aircraft.Ta0 * 1e3)
                / (Model.rhoratio_selected**Model.aircraft.beta)
            )

            self.condition = (
                W
                < analysisModel.compute_thrust(analysisModel.aircraft.h_array)
                * aircraft.E_P
            ) & (analysisModel.aircraft.CLmax > analysisModel.aircraft.CL_P)

            self.compute_optimal(W, h, Model)

    return (InteriorCondition,)


@app.cell
def _(
    MaxliftCondition,
    W_selected_analysis,
    analysisModel,
    fig_lift_limited,
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
    ### _Lift limited solutions (stall)_

    In this case: $C_L = C_{L_{\mathrm{max}}}$, $\delta_T \lt 1$, $\mu_1 \gt 0$, $\mu_2= 0$

    from stationarity condition (2): $\lambda_1 = 0$

    from stationarity condition (1):

    $$
    \mu_1 = - \left.\frac{\partial P}{\partial C_L} \right|_{C_{L_\mathrm{max}}} =- \sqrt{\frac{2W^3}{\rho S}}\left(-\frac{3}{2}C_{D_0}C_{L_\mathrm{max}}^{-5/2} + \frac{1}{2} K C_{L_\mathrm{max}}^{-1/2}\right) \gt 0
    $$

    This inequality is saying that the required power should decrease for an increase in $C_L$ starting from $C_{L_\mathrm{max}}$. In other words, $P$ should decrease for a decrease in speed from the stall speed. Equivalently, $P$ should increase for an increase in speed from the stall speed. This is not an ideal design, as the aircraft would be stalling at a velocity higher than the one for minimum power.
    """),
            fig_lift_limited,
            mo.md(r"""
    As a matter of fact, by substitution from the stationarity constraint (1):

    $$ \frac{3}{2}C_{D_0}C_{L_\mathrm{max}}^{-5/2} + \frac{1}{2} K C_{L_\mathrm{max}}^{-1/2} \lt 0
    $$

    $$
    \Rightarrow -3C_{D_0}+KC_{L_\mathrm{max}}^{2} \lt 0 \quad \Rightarrow \quad C_{L_\mathrm{max}} \lt \sqrt{\frac{3C_{D_0}}{K}} = C_{L_P}
    $$

    We thus find that $C_{L_\mathrm{max}}$ must be smaller than the lift coefficient for minimum power, as predicted in the discussion above.

    From primal feasibility (3), obtain the optimal value for $\delta_T$:

    $$
    \delta_T^* = \frac{W}{T_{a0}\sigma^\beta}\frac{C_{D_0} + KC_{L_\mathrm{max}}^2}{C_{L_\mathrm{max}}} = \frac{W}{T_{a0}\sigma^\beta} \frac{1}{E_S}
    $$

    The operational condition can be found by setting $\delta_T \lt 1$, obtaining:

    $$
    \frac{W}{\sigma^\beta} \lt T_{a0}E_S
    $$

    The value of the objective function, power, is calculated as:

    $$
    P^*_{\mathrm{min}} = DV = \frac{W}{E_S} \sqrt{\frac{W}{S}\frac{2}{\rho}\frac{1}{C_{L_\mathrm{max}}}} = \frac{W^{3/2}}{\sigma^{1/2}}\frac{1}{E_S}\sqrt{\frac{2}{\rho_0SC_{L_\mathrm{max}}}}
    $$

    This concludes the analysis for the minimum power of a simplified jet aircraft in the lift-limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = C_{L_\mathrm{max}}}, \quad \boxed{\delta_T^* = \frac{W}{T_{a0}\sigma^\beta} \frac{1}{E_S}}, \quad \text{for} \quad C_{L_\mathrm{max}} \lt \sqrt{\frac{3C_{D_0}}{K}} \quad \text{and}\quad \frac{W}{\sigma^\beta} \lt T_{a0}E_S
    $$

    With the optimal value for minimum power:

    $$
    P^*_{\mathrm{min}} = DV = \frac{W^{3/2}}{\sigma^{1/2}}\frac{1}{E_S}\sqrt{\frac{2}{\rho_0SC_{L_\mathrm{max}}}}
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
    class MaxliftCondition(OptimumCondition):
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
                * aircraft.E_P
            ) & (analysisModel.aircraft.CLmax < analysisModel.aircraft.CL_P)

            self.compute_optimal(W, h, Model)

    return (MaxliftCondition,)


@app.cell
def _(active_selection, analysisModel, tab_value, title_keys):
    if tab_value != title_keys[1]:
        mo.stop(True)

    fig_lift_limited = go.Figure()

    # Power curve vs CL
    fig_lift_limited.add_traces(
        [
            go.Scatter(
                x=analysisModel.V_CLarray,
                y=analysisModel.power_required / 1e3,
                name="Power",
            ),
        ]
    )

    fig_lift_limited.add_vline(
        x=analysisModel.Vstall_envelope[analysisModel.idx_h],
        line_dash="dot",
        annotation=dict(text=r"$V_{\mathrm{stall}}$", xshift=10, yshift=-10),
        line=dict(color="rgba(255, 0, 0, 0.3)"),
    )

    # Axes configuration
    fig_lift_limited.update_layout(
        legend=dict(
            x=0.01,  # Left edge
            y=1,  # Top edge
            xanchor="auto",
            yanchor="auto",
            bgcolor="rgba(0, 0, 0, 0.0)",  # Semi-transparent background
        ),
        xaxis=dict(title=r"$V \: (\text{m/s})$", range=[0, atmos.a(0)]),
        yaxis=dict(title=r"$P \: (\text{W})$", range=[0, analysisModel.power_ylim]),
    )

    fig_lift_limited.update_layout(
        title={
            "text": f"Power curve for {active_selection.full_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        }
    )

    mo.output.clear()
    return (fig_lift_limited,)


@app.cell
def _(
    MaxThrust,
    analysisModel,
    fig_performance_cl_eq,
    fig_thrust_limited,
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
    ### _Thrust-limited optimum_


    In this case: $C_L \lt C_{L_{\mathrm{max}}}$, $\delta_T = 1$, $\mu_1= 0$, $\mu_2 > 0$

    from stationarity condition (2): $\mu_2= -\lambda_1 T_{a0}\sigma^ \beta \gt 0 \Rightarrow \lambda_1 \lt 0$

    from stationarity condition (1):

    $$
    \frac{\partial P}{\partial C_L} = \lambda_1 \frac{\partial D}{\partial C_L}
    $$
    """),
            mo.md(r"""This tells us that the required power and drag change in opposite directions with respect to the change in $C_L$. If one decreases, then the other one has to increase, given that $\lambda_1 \lt 0$.
    This can only happen in the range of $C_L$ between $C_{L_P}$ and $C_{L_E}$, since they represent the minimum power and maximum aerodynamic efficiency (alternatively minimum drag) respectively.

    This is clearer in the performance diagram:"""),
            variables_stack_analysis,
            fig_thrust_limited,
            mo.md(r"""
    This condition is given by:

    $$
    C_{L_E}\lt C_L^*\lt C_{L_P} \quad \Leftrightarrow \quad \boxed{\sqrt{\frac{C_{D_0}}{K}}\lt C_L^* \lt \sqrt{3} \sqrt{\frac{C_{D_0}}{K}}}
    $$

    The corresponding $C_L^*$ is given by primal feasibility constraint (3):

    $$
    T_{a0} \sigma^\beta - W \left(\frac{C_{D_0}+KC_L^2}{C_L}\right)=0
    $$

    Yielding the following quadratic equation:

    $$
    K C_L^2 - \frac{T_{a0}\sigma^\beta}{W}C_L+C_{D_0} = 0 \quad \Rightarrow \quad C_L = \frac{T_{a0}\sigma^\beta}{2KW}\left[1 \pm\sqrt{1- \left(\frac{W}{T_{a0}\sigma^\beta E_{\mathrm{max}}}\right)^2}\right]
    $$

    where the relevant solution is given by the "${+}$" sign, on the left branch of the drag curve in the performance diagram:

    $$
    \Rightarrow \quad C_L^* = \frac{T_{a0}\sigma^\beta}{2KW}\left[1 +\sqrt{1- \left(\frac{W}{T_{a0}\sigma^\beta E_{\mathrm{max}}}\right)^2}\right]
    $$

    The solution is valid as long as: $\sqrt{\frac{C_{D_0}}{K}}\lt C_L^* \lt \sqrt{3} \sqrt{\frac{C_{D_0}}{K}}$.

    For its existence, the square root must be zero or positive, thus:

    $$
    1 - \left(\frac{W}{T_{a0}\sigma^\beta E_{\mathrm{max}}}\right)^2 \ge 0 \quad \Rightarrow \quad \frac{W}{\sigma^\beta}\le T_{a0}E_{\mathrm{max}}
    $$

    as already seen in multiple occasions.

    Try to find whether there is a combination of altitude and weight for which the solution of the quadratic equation with the "$+$" sign falls within the bounds of $C_{L_P}$ and $C_{L_E}$, denoted by the green area in the graph below. Be careful, this is not always possible and will define the flight envelope where minimum power can be achieved.
    """),
            variables_stack_analysis,
            fig_performance_cl_eq,
            mo.md(r"""
    In order for $C_L^* \gt \sqrt{\frac{C_{D_0}}{K}}$ it must be:

    $$
    1 - \left(\frac{W}{T_{a0}\sigma^\beta E_{\mathrm{max}}}\right)^2  \gt \left(\frac{W}{T_{a0}\sigma^\beta E_{\mathrm{max}}} - 1\right)^2
    $$

    which then simplifies to:

    $$
    \frac{W}{\sigma^\beta}\lt T_{a0}E_{\mathrm{max}}
    $$

    which can be compared to the domain imposed by the square root directly. The strongest condition, or the lower upper bound, for $W/\sigma^\beta$ is given by:


    $$
    \frac{W}{\sigma^\beta}\lt T_{a0}E_{\mathrm{max}}
    $$

    Similarly, for the upper bound,

    $$
    C_L^* \lt \sqrt{3} \sqrt{\frac{C_{D_0}}{K}}\quad \Rightarrow \quad \frac{W}{\sigma^\beta} \gt  \frac{\sqrt{3}}{2} T_{a0} E_{\mathrm{max}}
    $$

    The value of the objective function, power, is calculated as:

    $$
    P^*_{\mathrm{min}} = DV = \frac{1}{2}\rho V^2 S (C_{D_0} + K C_L^{*2})\sqrt{\frac{W}{S}\frac{2}{\rho}\frac{1}{C_{L}^*}} = \frac{W^{3/2}}{\sigma^{1/2}}\left(\frac{C_{D_0}+ K C_L^{*2}}{C_L^{*}}\right)\sqrt{\frac{2}{\rho_0 S C_L^*}}
    $$

    This concludes the analysis for the minimum power of a simplified jet aircraft in the thrust-limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = \frac{T_{a0}\sigma^\beta}{2KW}\left[1 +\sqrt{1- \left(\frac{W}{T_{a0}\sigma^\beta E_{\mathrm{max}}}\right)^2}\right]}, \quad \boxed{\delta_T^* = 1}, \quad \text{for} \quad \frac{\sqrt{3}}{2} T_{a0} E_{\mathrm{max}} \lt \frac{W}{\sigma^\beta} \lt T_{a0} E_{\mathrm{max}}
    $$

    With the optimal value for minimum power:

    $$
    P^*_{\mathrm{min}} = DV = \frac{W^{3/2}}{\sigma^{1/2}}\left(\frac{C_{D_0}+ K C_L^{*2}}{C_L^{*}}\right)\sqrt{\frac{2}{\rho_0 S C_L^*}}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.
    """),
            variables_stack_analysis,
            analysisModel.plot_grid(
                (MaxThrust,),
                plot_options_analysis,
            ).figure,
        ]
    ).callout()
    return


@app.cell
def _(aircraft, analysisModel):
    class MaxThrustCondition(OptimumCondition):
        def __init__(self, W, h, Model):
            thrust_envelope = Model.compute_thrust(Model.aircraft.h_array)

            self.dTopt = 1

            self.condition = (
                W
                > analysisModel.compute_thrust(analysisModel.aircraft.h_array)
                * aircraft.E_P
            ) & (
                W
                < analysisModel.compute_thrust(analysisModel.aircraft.h_array)
                * aircraft.E_max
            )

            A = thrust_envelope[self.condition] / (2 * Model.aircraft.K * W)
            B = 1 + np.sqrt(
                1 - (W / (Model.aircraft.E_max * thrust_envelope[self.condition])) ** 2
            )

            self.CLopt = A * B

            thrust_selected = Model.compute_thrust(h)

            if W > thrust_selected * Model.aircraft.E_S:
                self.A = thrust_selected / (2 * Model.aircraft.K * W)
                self.B = 1 + np.sqrt(
                    1 - (W / (Model.aircraft.E_max * thrust_selected)) ** 2
                )
                self.CLopt_selected = self.A * self.B
            else:
                self.A = self.B = np.nan
                self.CLopt_selected = np.nan

            self.compute_optimal(W, h, Model)

    return (MaxThrustCondition,)


@app.cell
def _(active_selection, analysisModel, tab_value, title_keys):
    if tab_value != title_keys[2]:
        mo.stop(True)

    fig_thrust_limited = go.Figure()

    # Power curve vs CL
    fig_thrust_limited.add_traces(
        [
            go.Scattergl(
                x=analysisModel.V_CLarray,
                y=analysisModel.power_required / 1e3,
                name="Power",
            ),
            go.Scattergl(
                x=analysisModel.V_CLarray,
                y=analysisModel.drag_curve,
                name="Drag",
                yaxis="y2",
            ),
            go.Scattergl(
                x=[
                    analysisModel.Vstall_envelope[analysisModel.idx_h] - 20,
                    atmos.a(0) * 2,
                ],
                y=[0.1 * analysisModel.drag_ylim, 0.1 * analysisModel.drag_ylim],
                mode="lines",
                line=dict(color="grey", width=1),
                yaxis="y2",
                showlegend=False,
            ),
            go.Scattergl(
                x=[analysisModel.Vstall_envelope[analysisModel.idx_h] - 20],
                y=[0.1 * analysisModel.drag_ylim],
                mode="markers",
                yaxis="y2",
                marker=dict(color="grey", size=10, symbol="arrow-left"),
                showlegend=False,
            ),
        ]
    )

    # Change this to label + line in future iterations
    # Add CL_P and CL_E curves
    fig_thrust_limited.add_vline(
        x=analysisModel.V_CLE,
        line_dash="dot",
        annotation=dict(text="$C_{L_E}$", xshift=10, yshift=-10),
        line=dict(color="white"),
    )
    fig_thrust_limited.add_vline(
        x=analysisModel.V_CLP,
        line_dash="dot",
        annotation=dict(text="$C_{L_P}$", xshift=10, yshift=-10),
        line=dict(color="white"),
    )
    fig_thrust_limited.add_vrect(
        x0=analysisModel.V_CLP,
        x1=analysisModel.V_CLE,
        fillcolor="green",
        opacity=0.25,
        line_width=0,
    )

    fig_thrust_limited.add_vline(
        x=analysisModel.Vstall_envelope[analysisModel.idx_h],
        line_dash="dot",
        annotation=dict(text=r"$V_{\mathrm{stall}}$", xshift=10, yshift=-10),
        line=dict(color="rgba(255, 0, 0, 0.3)"),
    )

    # Axes configuration
    fig_thrust_limited.update_layout(
        legend=dict(
            x=0.01,  # Left edge
            y=1,  # Top edge
            xanchor="auto",
            yanchor="auto",
            bgcolor="rgba(0, 0, 0, 0.0)",  # Semi-transparent background
        ),
        xaxis=dict(title=r"$V \: (\text{m/s})$", range=[0, atmos.a(0)]),
        yaxis=dict(title=r"$P \: (\text{W})$"),
        yaxis2=dict(
            title=r"$D \: (\text{N})$",
            overlaying="y",
            side="right",
        ),
    )

    fig_thrust_limited.update_layout(
        title={
            "text": f"Performance diagram for {active_selection.full_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        }
    )

    mo.output.clear()
    return (fig_thrust_limited,)


@app.cell
def _(
    MaxThrustCondition,
    W_selected_analysis,
    active_selection,
    analysisModel,
    h_selected_analysis,
    tab_value,
    title_keys,
):
    if tab_value != title_keys[2]:
        mo.stop(True)

    MaxThrust = MaxThrustCondition(
        W_selected_analysis, h_selected_analysis, analysisModel
    )

    fig_performance_cl_eq = go.Figure()

    fig_performance_cl_eq.add_traces(
        [
            go.Scatter(
                x=analysisModel.V_CLarray, y=analysisModel.power_required, name="Power"
            ),
            go.Scatter(
                x=analysisModel.V_CLarray,
                y=analysisModel.drag_curve,
                name="Drag",
                yaxis="y2",
            ),
        ]
    )

    if ~np.isnan(MaxThrust.A):
        fig_performance_cl_eq.add_vline(
            x=analysisModel.compute_velocity(
                W_selected_analysis, h_selected_analysis, MaxThrust.A * MaxThrust.B
            ),
            line_dash="dot",
            annotation=dict(text="$C_{L}^{*+}$", xshift=10, yshift=-10),
            line=dict(color="white"),
        )
        fig_performance_cl_eq.add_vline(
            x=analysisModel.compute_velocity(
                W_selected_analysis,
                h_selected_analysis,
                MaxThrust.A * (2 - MaxThrust.B),
            ),
            line_dash="dot",
            annotation=dict(text="$C_{L}^{*-}$", xshift=10, yshift=-10),
            line=dict(color="white"),
        )
    fig_performance_cl_eq.add_vline(
        x=analysisModel.V_CLE,
        line_dash="dot",
        annotation=dict(text="$C_{L_E}$", xshift=10, yshift=-10),
        line=dict(color="white"),
    )
    fig_performance_cl_eq.add_vline(
        x=analysisModel.V_CLP,
        line_dash="dot",
        annotation=dict(text="$C_{L_P}$", xshift=10, yshift=-10),
        line=dict(color="white"),
    )
    fig_performance_cl_eq.add_vrect(
        x0=analysisModel.V_CLP,
        x1=analysisModel.V_CLE,
        fillcolor="green",
        opacity=0.25,
        line_width=0,
    )

    # Axes configuration
    fig_performance_cl_eq.update_layout(
        legend=dict(
            x=0.01,  # Left edge
            y=1,  # Top edge
            xanchor="auto",
            yanchor="auto",
            bgcolor="rgba(0, 0, 0, 0.0)",  # Semi-transparent background
        ),
        xaxis=dict(title=r"$V \: (\text{m/s})$", range=[0, atmos.a(0)]),
        yaxis=dict(title=r"$P \: (\text{W})$", range=[0, analysisModel.power_ylim]),
        yaxis2=dict(
            title=r"$D (\text{N})$",
            overlaying="y",
            side="right",
            range=[0, analysisModel.drag_ylim],
        ),
    )

    fig_performance_cl_eq.update_layout(
        title={
            "text": f"Performance diagram for {active_selection.full_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        }
    )

    mo.output.clear()
    return MaxThrust, fig_performance_cl_eq


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

    MaxLiftThrust = MaxLiftThrustCondition(W_selected_analysis, analysisModel)
    surface_MaxLiftThrust = np.broadcast_to(
        analysisModel.power_required[np.newaxis, :],
        (plot_utils.meshgrid_n, plot_utils.meshgrid_n),
    )

    plot_options_analysis_MaxLiftThrust = {
        "surface": surface_MaxLiftThrust,
    }

    liftThrustlimited_solutions = mo.vstack(
        [
            mo.md(r"""
    ### _Lift- and thrust- limited optimum_

    In this case: $C_L = C_{L_{\mathrm{max}}}$, $\delta_T = 1$, $\mu_1 \gt 0$, $\mu_2 \gt 0$

    from stationarity condition (2): $\lambda_1 \lt 0$

    from stationarity condition (1):

    $$
    \mu_1 = - \left.\frac{\partial P}{\partial C_L} \right|_{C_{L_\mathrm{max}}} + \lambda_1 \left.\frac{\partial D}{\partial C_L} \right|_{C_{L_\mathrm{max}}} \gt 0
    $$

    which becomes:

    $$
    \sqrt{\frac{2W^3}{\rho S}}\left(\frac{3}{2}C_{D_0}C_{L_{\mathrm{max}}}^{-5/2} - \frac{1}{2} K C_{L_{\mathrm{max}}}^{-1/2}\right) + \lambda_1 W \left(\frac{KC_{L_{\mathrm{max}}}^2 -C_{D_0}}{C_{L_{\mathrm{max}}}^2}\right) \gt 0
    $$

    $$
    \Rightarrow \sqrt{\frac{2W}{\rho S}}\left(\frac{3}{2}C_{D_0}C_{L_{\mathrm{max}}}^{-5/2} - \frac{1}{2} K C_{L_{\mathrm{max}}}^{-1/2}\right) + \lambda_1 \left(\frac{KC_{L_{\mathrm{max}}}^2 -C_{D_0}}{C_{L_{\mathrm{max}}}^2}\right) \gt 0
    $$

    Multiply on both sides by $2C_{L_\mathrm{max}}^2 (>0)$ and obtain:

    $$
    \Rightarrow 2\lambda_1(KC_{L_\mathrm{max}}^2-C_{D_0}) + \sqrt{\frac{2W}{\rho S}}\left({3}C_{D_0}C_{L_{\mathrm{max}}}^{-1/2} - K C_{L_{\mathrm{max}}}^{3/2}\right)\gt 0
    $$

    $$
    \Rightarrow \lambda_1 (2(KC_{L_{\mathrm{max}}}^2-C_{D_0}))\gt \sqrt{\frac{2W}{\rho S}}\left(K C_{L_{\mathrm{max}}}^{3/2} - {3}C_{D_0}C_{L_{\mathrm{max}}}^{-1/2}\right)
    $$

    Now analyse the two cases from the inequality. First, if $KC_{L_{\mathrm{max}}}^2-C_{D_0}>0$ it follows that:

    $$
    \displaystyle \lambda_1 \gt \frac{\sqrt{\frac{2W}{\rho S}}\left(K C_{L_{\mathrm{max}}}^{3/2} - {3}C_{D_0}C_{L_{\mathrm{max}}}^{-1/2}\right)}{2(KC_{L_{\mathrm{max}}}^2-C_{D_0})}
    $$

    Which, together with the result from the stationarity condition (2): $\lambda_1 < 0$, means that the fraction above must be smaller than 0, thus the numerator must be negative; write:


    $$
    \sqrt{\frac{2W}{\rho S}}\left(K C_{L_{\mathrm{max}}}^{3/2} - {3}C_{D_0}C_{L_{\mathrm{max}}}^{-1/2}\right) < 0
    $$

    $$
    \Rightarrow C_{L_\mathrm{max}} \lt \sqrt{\frac{3C_{D_0}}{K}}
    $$

    This result must be intersected with the condition assumed for the denominator, conclude:

    $$
    \displaystyle \sqrt{\frac{C_{D_0}}{K}} \lt C_{L_\mathrm{max}} \lt \sqrt{\frac{3C_{D_0}}{K}} \quad \Rightarrow \quad C_{L_E} \lt C_{L_\mathrm{max}} \lt C_{L_P}
    $$

    On the other hand, by assuming the denominator is negative,  with $KC_{L_{\mathrm{max}}}^2-C_{D_0}<0$, find:

    $$
    \displaystyle \lambda_1 \lt \frac{\sqrt{\frac{2W}{\rho S}}\left(K C_{L_{\mathrm{max}}}^{3/2} - {3}C_{D_0}C_{L_{\mathrm{max}}}^{-1/2}\right)}{2(KC_{L_{\mathrm{max}}}^2-C_{D_0})} \quad \text{and} \quad \lambda_1 \lt 0
    $$

    Together with stationary condition (2). It is now necessary to investigate the cases when the numerator is positive and negative. Starting with the positive case:

    $$
    K C_{L_{\mathrm{max}}}^{3/2} - {3}C_{D_0}C_{L_{\mathrm{max}}}^{-1/2} \gt 0 \quad \Rightarrow \quad C_{L_\mathrm{max}} \gt \sqrt{3}C_{L_E}
    $$

    In this case, the fraction for $\lambda_1$ will be negative, and the stationary condition (2) is met. However, finding the intersection of the result above and the assumption on the denominator find that no intersection is present, and thus it is impossible to have an optimum.

    $$
    C_{L_\mathrm{max}} \lt C_{L_E}\quad \text{and} \quad C_{L_\mathrm{max}} \gt \sqrt{3}C_{L_E} \quad \mathrm{impossible}
    $$

    Now, assume the numerator is negative, and find:

    $$
    K C_{L_{\mathrm{max}}}^{3/2} - {3}C_{D_0}C_{L_{\mathrm{max}}}^{-1/2} \lt 0 \quad \Rightarrow \quad C_{L_\mathrm{max}} \lt \sqrt{3}C_{L_E}
    $$

    In this case, an intersection can be found between the condition on the denominator and the result from the inequality above. Moreover, since the fraction will be positive, and $\lambda_1$ must be smaller than this fraction and smaller than 0, find the following intersection for the case with a negative denominator:

    $$
    C_{L_\mathrm{max}} \lt C_{L_E}\quad \text{and} \quad C_{L_\mathrm{max}} \lt \sqrt{3}C_{L_E} \quad \Rightarrow \quad C_{L_\mathrm{max}} \lt C_{L_E}
    $$

    Now, by taking the union of the two solutions with a different sign in the denominator find:

    $$
    C_{L_E} \lt C_{L_\mathrm{max}} \lt C_{L_P} \: \cup \:C_{L_\mathrm{max}} \lt C_{L_E} \quad \Rightarrow \quad C_{L_\mathrm{max}} < C_{L_P} \quad \text{with} \quad C_{L_\mathrm{max}} \neq C_{L_E}
    $$

    This concludes the analysis for the condition on $C_{L_\mathrm{max}}$. Opposite to what one might think, the condition $C_{L_{\mathrm{max}}} \lt \sqrt{3}C_{L_E}$ is plausible as this is a design choice. $C_{L_{\mathrm{max}}}$, $C_{D_0}$, and $K$ are in fact all independent with one other.

    Now continuing with the primal feasibility condition (3):

    $$
    T_{a0}\sigma^\beta = W \frac{C_{D_0} + K C_{L_{\mathrm{max}}}^2}{C_{L_{\mathrm{max}}}} = W E_S \quad \Leftrightarrow \quad \frac{W}{\sigma^\beta} = T_{a0} E_S
    $$

    The value of the objective function, power, is calculated as:

    $$
    P^*_{\mathrm{min}} = DV = \frac{W}{E_S} \sqrt{\frac{W}{S}\frac{2}{\rho}\frac{1}{C_{L_\mathrm{max}}}} = \frac{W^{3/2}}{\sigma^{1/2}}\frac{1}{E_S}\sqrt{\frac{2}{\rho_0SC_{L_\mathrm{max}}}}
    $$

    This concludes the analysis for the minimum power of a simplified jet aircraft in the lift-thrust limited case. Below is a summary of the optima:

    $$
    \boxed{C_L^* = C_{L_\mathrm{max}}}, \quad \boxed{\delta_T^* = 1}, \quad \text{for} \quad {C_{L_\mathrm{max}} < C_{L_P} \quad \text{with} \quad C_{L_\mathrm{max}} \neq C_{L_E}}, \quad \text{and} \quad \frac{W}{\sigma^\beta} = T_{a0} E_S
    $$

    With the optimal value for minimum power:

    $$
    P^*_{\mathrm{min}} = DV = \frac{W^{3/2}}{\sigma^{1/2}}\frac{1}{E_S}\sqrt{\frac{2}{\rho_0SC_{L_\mathrm{max}}}}
    $$

    Below is the performance diagram for power and drag, the optimization domain with the objective function as a surface plot, and finally, on the bottom right, the flight envelope where the optima can be achieved.
    """),
            mass_stack_analysis,
            analysisModel.plot_grid(
                (MaxLiftThrust,),
                plot_options_analysis_MaxLiftThrust,
            ).figure,
        ]
    )

    liftThrustlimited_solutions.callout()
    return


@app.class_definition
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
        self.condition = (Model.aircraft.CLmax < Model.aircraft.CL_P) & (
            Model.aircraft.CLmax != Model.aircraft.CL_E
        )

        self.CLopt = self.CLopt_selected = (
            Model.aircraft.CLmax if self.condition else np.nan
        )

        self.compute_optimal(W, h_optimum, Model, True)

        self.cond = 1 if self.condition else np.nan

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


@app.cell(hide_code=True)
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
                    MaxliftCondition(
                        W_selected_envelope, h_selected_envelope, envelopeModel
                    ),
                    MaxThrustCondition(
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
    | Name | Condition | $C_L^*$ | $\delta_T^*$ |
    |:-|:-------|:-------:|:-----:|
    |Interior-optima    | $\displaystyle \quad  C_{L_\mathrm{max}} > C_{L_P} \quad \text{and} \quad \frac{W}{\sigma^\beta} < T_{a0} E_\mathrm{P}$ | $\sqrt{\frac{3C_{D_0}}{K}}$ | $\displaystyle \frac{W}{E_{\mathrm{P}}}\frac{1}{T_{a0}\sigma^\beta}$  |
    |Lift-limited    |  $\displaystyle C_{L_\mathrm{max}} \lt {C_{L_P}} \quad \text{and}\quad \frac{W}{\sigma^\beta} \lt T_{a0}E_S$ | $C_{L_\mathrm{max}}$ | $\displaystyle \frac{W}{E_S} \frac{1}{T_{a0}\sigma^\beta}$ |
    |Thrust-limited    | $\displaystyle\quad T_{a0} E_{\mathrm{P}} \lt \frac{W}{\sigma^\beta} \lt T_{a0} E_{\mathrm{max}}$ | $\displaystyle \frac{T_{a0}\sigma^\beta}{2KW}\left[1 +\sqrt{1- \left(\frac{W}{T_{a0}\sigma^\beta E_{\mathrm{max}}}\right)^2}\right]$ | $1$ |
    |Thrust-lift limited    |  $\displaystyle {C_{L_\mathrm{max}} < C_{L_P},C_{L_\mathrm{max}} \neq C_{L_E}}, \quad \text{and} \quad \frac{W}{\sigma^\beta} = T_{a0} E_S$ | $C_{L_\mathrm{max}}$ | $1$ |
    """
    ).center()
    return


@app.cell(hide_code=True)
def _():
    _defaults.nav_footer(
        after_file="MinPower_Prop.py",
        after_title="Minimum Power Simplified Propeller",
        above_file="MinPower.py",
        above_title="Minimum Power Homepage",
        above_before=True,
    )
    return


if __name__ == "__main__":
    app.run()
