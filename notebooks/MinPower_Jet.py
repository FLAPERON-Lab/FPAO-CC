import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo

    # Import dependencies
    from core import _defaults
    import plotly.graph_objects as go
    import numpy as np
    import copy
    from core import atmos
    from core import aircraft as ac
    from core import plot_utils
    from core.aircraft import velocity
    from core.plot_utils import OptimumGridView

    # Set local/online filepath
    _defaults.FILEURL = _defaults.get_url()

    # Plotly dark mode template
    _defaults.set_plotly_template()

    # Data directory
    data_dir = str(mo.notebook_location() / "public" / "AircraftDB_Standard.csv")


@app.cell
def _():
    # Set navbar on the right
    multiple_tabs = mo.ui.switch(label="Multiple tabs?")
    _defaults.set_sidebar(multiple_tabs)
    return


@app.cell
def _():
    # Define constants, this cell runs once and is not dependent in any way on any interactive element (not even the ac database)
    dT_slider = mo.ui.slider(start=0, stop=1, step=0.1, label=r"$\delta_T$", value=0.5)

    meshgrid_n = 41
    xy_lowerbound = -0.1

    dT_array = np.linspace(0, 1, meshgrid_n)  # -
    h_array = np.linspace(0, 20e3, meshgrid_n)  # meters

    m_slider = mo.ui.slider(start=0, stop=1, step=0.1, label=r"", show_value=True)

    h_slider = mo.ui.slider(
        start=0,
        stop=20,
        step=0.5,
        label=r"Altitude (km)",
        value=0,
        show_value=True,
    )

    data = ac.available_aircrafts(data_dir, ac_type="Jet")[:8]

    labels = ["Power (kW)", -15]

    # Database cell
    ac_table = mo.ui.table(
        data=data,
        pagination=True,
        show_column_summaries=False,
        selection="single",
        initial_selection=[0],
        page_size=4,
        show_data_types=False,
    )
    a_0 = atmos.a(0)

    hover_name = "P<sub>min</sub>"

    mass_stack = mo.hstack(
        [mo.md("**OEW**"), m_slider, mo.md("**MTOW**")],
        align="start",
        justify="start",
    )
    variables_stack = mo.hstack([mass_stack, h_slider])

    rho_array = atmos.rho(h_array)
    sigma_array = atmos.rhoratio(h_array)
    min_sigma = atmos.rhoratio(atmos.hmax)
    a_harray = atmos.a(h_array)

    # Visual computations
    mach_trace = plot_utils.create_mach_trace(h_array, a_harray)
    return (
        a_0,
        a_harray,
        ac_table,
        dT_array,
        dT_slider,
        data,
        h_array,
        h_slider,
        m_slider,
        mach_trace,
        mass_stack,
        meshgrid_n,
        min_sigma,
        rho_array,
        sigma_array,
        variables_stack,
        xy_lowerbound,
    )


@app.cell
def _(a_0, ac_table, dT_array, data, meshgrid_n, xy_lowerbound):
    # Define constants dependent on the ac database. This runs every time another aircraft is selected

    if ac_table.value is not None and ac_table.value.any().any():
        active_selection = ac_table.value.iloc[0]
    else:
        active_selection = data.iloc[0]

    # avoid having zeros for velocity computation
    CL_array = np.linspace(0, active_selection["CLmax_ld"], meshgrid_n + 1)[1:]

    # Extract essential values
    CD0 = active_selection["CD0"]
    S = active_selection["S"]
    K = active_selection["K"]
    CLmax = active_selection["CLmax_ld"]
    Ta0 = active_selection["Ta0"] * 1e3  # Watts
    beta = active_selection["beta"]
    OEM = active_selection["OEM"]
    MTOM = active_selection["MTOM"]

    # Compute design values
    CL_P = np.sqrt(3 * CD0 / K)
    CL_E = np.sqrt(CD0 / K)
    E_max = CL_E / (CD0 + K * CL_E**2)
    E_S = CLmax / (CD0 + K * CLmax**2)
    E_P = (np.sqrt(3) / 2) * E_max
    E_array = CL_array / (CD0 + K * CL_array**2)

    CL_slider = mo.ui.slider(
        start=0,
        stop=CLmax,
        step=0.2,
        label=r"$C_L$",
        value=0.5,
    )

    ranges = [
        xy_lowerbound,
        CLmax + 0.05,
        xy_lowerbound,
        1 + 0.05,
        xy_lowerbound,
        a_0,
        xy_lowerbound,
        20,
    ]

    interior_title = f"Interior minimum power for OEW {active_selection.full_name}"
    maxlift_title = f"Lift-limited minimum power for {active_selection.full_name}"
    maxthrust_title = f"Thrust-limited minimum power for {active_selection.full_name}"
    maxliftThrust_title = f"Lift-thrust limited minimum power for {active_selection.full_name}"
    final_fig_title = f"Flight envelope for minimum power for {active_selection.full_name}"

    axes = (CL_array, dT_array)
    return (
        CD0,
        CL_E,
        CL_P,
        CL_array,
        CL_slider,
        CLmax,
        E_P,
        E_S,
        E_array,
        E_max,
        K,
        MTOM,
        OEM,
        S,
        Ta0,
        active_selection,
        beta,
    )


@app.cell
def _(CL_array, CL_slider):
    # Define variables, this cell runs every time the CL slider is run
    step_CL = CL_array[2] - CL_array[1]
    CL_selected = float(CL_slider.value)
    idx_CL_selected = int((CL_selected - CL_array[0]) / step_CL)
    return (idx_CL_selected,)


@app.cell
def _(CD0, CLmax, E_array, K, MTOM, OEM, S, a_0, h_array, m_slider, rho_array):
    # Define variables, this cell runs every time the mass slider is run
    W_selected = (OEM + (MTOM - OEM) * m_slider.value) * atmos.g0  # Netwons
    drag_curve = W_selected / E_array

    velocity_stall_harray = np.sqrt(2 * W_selected / (rho_array * S * CLmax))

    # Visual computations
    stall_trace = plot_utils.create_stall_trace(h_array, velocity_stall_harray)

    CL_a0 = OEM * atmos.g0 * 2 / (atmos.rho0 * S * a_0**2)

    drag_yrange = 0.8 * OEM * atmos.g0 * (CD0 + K * CL_a0**2) / CL_a0
    power_yrange = drag_yrange * a_0 / 1e3
    return (
        W_selected,
        drag_curve,
        drag_yrange,
        power_yrange,
        stall_trace,
        velocity_stall_harray,
    )


@app.cell
def _(Ta0, beta, h_array, h_slider, meshgrid_n):
    # Define variables, this cell runs every time the altitude slider is run
    h_selected = int(h_slider.value * 1e3)  # meters
    step_h = h_array[1] - h_array[0]
    idx_h_selected = int((h_selected - h_array[0]) / step_h)

    a_selected = atmos.a(h_selected)

    sigma_selected = atmos.rhoratio(h_selected)

    rho_selected = atmos.rho(h_selected)

    thrust_scalar = Ta0 * sigma_selected**beta

    thrust_vector = np.repeat(thrust_scalar, meshgrid_n)
    return (
        h_selected,
        idx_h_selected,
        rho_selected,
        sigma_selected,
        thrust_scalar,
        thrust_vector,
    )


@app.cell
def _(
    CL_E,
    CL_P,
    CL_array,
    CLmax,
    S,
    Ta0,
    W_selected,
    beta,
    dT_array,
    drag_curve,
    drag_yrange,
    idx_h_selected,
    mach_trace,
    power_yrange,
    rho_selected,
    sigma_selected,
    stall_trace,
    thrust_scalar,
    thrust_vector,
    velocity_stall_harray,
):
    # Computation only cell, indexing happens in another cell
    velocity_CLarray = np.sqrt(2 * W_selected / (rho_selected * S * CL_array))
    velocity_CL_E = velocity_CLarray[-1] * np.sqrt(CLmax / CL_E)
    velocity_CL_P = velocity_CLarray[-1] * np.sqrt(CLmax / CL_P)


    power_available = thrust_scalar * velocity_CLarray / 1e3
    power_required = drag_curve * velocity_CLarray / 1e3
    power_surface = np.broadcast_to(
        power_required[np.newaxis, :],  # Shape: (101, 1)
        (len(CL_array), len(dT_array)),  # Target shape: (101, 101)
    )

    constraint = drag_curve / Ta0 / (sigma_selected**beta)

    min_colorbar = np.min(power_required)
    max_colorbar = min_colorbar * 2
    zcolorbar = (min_colorbar, max_colorbar)
    range_performance_diagrams = (drag_yrange, power_yrange, CLmax)


    # Create graphic traces
    configTraces = plot_utils.ConfigTraces(
        CL_array,
        dT_array,
        constraint,
        drag_curve,
        thrust_vector,
        power_required,
        power_available,
        power_surface,
        velocity_CLarray,
        velocity_CL_P,
        velocity_CL_E,
        velocity_stall_harray,
        velocity_stall_harray[idx_h_selected],
        range_performance_diagrams,
        zcolorbar,
        mach_trace,
        stall_trace,
    )
    return (
        configTraces,
        constraint,
        power_required,
        range_performance_diagrams,
        velocity_CL_E,
        velocity_CL_P,
        velocity_CLarray,
    )


@app.cell
def _(idx_CL_selected, power_required):
    power_required_selected = power_required[idx_CL_selected]
    return


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


@app.cell
def _(ac_table):
    mo.lazy(ac_table)
    return


@app.cell(hide_code=True)
def _(CL_slider, dT_slider):
    mo.md(f"""
    Here you can modify the control variables to understand how it affects the design: {mo.hstack([dT_slider, CL_slider])}
    """)
    return


@app.cell(hide_code=True)
def _():
    # pause_initial = mo.ui.checkbox(label="Pause execution")

    # mo.hstack([variables_stack, pause_initial])
    return


@app.cell(hide_code=True)
def _():
    # if pause_initial.value:
    #     mo.stop(mo.md(""))

    # # Create go.Figure() object
    # fig_initial = go.Figure()

    # # Minimum velocity surface
    # fig_initial.add_traces(
    #     [
    #         go.Surface(
    #             x=CL_array,
    #             y=dT_array,
    #             z=power_surface,
    #             opacity=0.9,
    #             name="Power",
    #             colorscale="viridis",
    #             cmin=min_colorbar,
    #             cmax=max_colorbar,
    #             colorbar={"title": "Power (kW)"},
    #         ),
    #         go.Scatter3d(
    #             x=CL_array,
    #             y=constraint,
    #             z=power_surface[0],
    #             opacity=1,
    #             mode="lines",
    #             showlegend=False,
    #             line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
    #             name="g1 constraint",
    #         ),
    #         go.Scatter3d(
    #             x=[CL_array[-15]],
    #             y=[constraint[-15]],
    #             z=[power_surface[0, 0] + 450],
    #             opacity=1,
    #             textposition="middle left",
    #             mode="markers+text",
    #             text=["g<sub>1</sub>"],
    #             marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
    #             textfont=dict(size=14, family="Arial"),
    #             showlegend=False,
    #             name="g1 constraint",
    #         ),
    #         go.Scatter3d(
    #             x=[CL_slider.value],
    #             y=[dT_slider.value],
    #             z=[power_required_selected],
    #             mode="markers",
    #             showlegend=False,
    #             marker=dict(
    #                 size=3,
    #                 color="white",
    #                 symbol="circle",
    #             ),
    #             name="Design Point",
    #             hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub> : %{y}<br>P: %{z}<extra>%{fullData.name}</extra>",
    #         ),
    #     ]
    # )
    # camera = dict(eye=dict(x=1.35, y=1.35, z=1.35))

    # fig_initial.update_layout(
    #     scene=dict(
    #         xaxis=dict(
    #             title="C<sub>L</sub> (-)",
    #             range=[xy_lowerbound, active_selection["CLmax_ld"]],
    #         ),
    #         yaxis=dict(title="δ<sub>T</sub> (-)", range=[xy_lowerbound, 1]),
    #         zaxis=dict(
    #             title="P (kW)",
    #             range=[0, max_colorbar],
    #         ),
    #     ),
    # )
    # fig_initial.update_layout(
    #     scene_camera=camera,
    #     title={
    #         "text": f"Minimum power domain for {active_selection.full_name}",
    #         "font": {"size": 25},
    #         "xanchor": "center",
    #         "yanchor": "top",
    #         "x": 0.5,
    #     },
    # )
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
def _():
    titles_dict = {
        "### Interior solutions": "",
        "### Lift limited solutions": "",
        "### Thrust limited solutions": "",
        "### Lift-thrust limited solutions": "",
    }

    tab = mo.ui.tabs(titles_dict)
    tab.style({"height": "60px", "overflow": "auto"}).callout(kind="info").center()
    return tab, titles_dict


@app.cell
def _(tab, titles_dict):
    title_keys = list(titles_dict.keys())
    tab_value = tab.value
    return tab_value, title_keys


@app.cell
def _(fig_interior_optimum, tab_value, title_keys, variables_stack):
    if tab_value != title_keys[0]:
        mo.stop(True)

    render_interior = mo.vstack(
        [
            mo.md(r"""
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
            variables_stack,
            fig_interior_optimum.figure,
            mo.md(
                r"""Notice how $C_{L_P}$ (minimum power) $\gt$ $C_{L_E}$ (minimum drag) but $E_\mathrm{P} \lt E_{\mathrm{max}}$ ($E = C_L/C_D$) because the drag coefficient increases more rapidly than $C_L$, as $C_D \propto C_L^2$. Thus, the range of $W/\sigma^\beta$ for which it is possible to fly at minimum power is smaller ($\sqrt{3}/2\lt 1$) than the one for which it is possible to fly at minimum drag. You can check this by increasing the weight of the aircraft here and in [Minimum Drag (simplified Jet)](?file=MinDrag_Jet.py) and finding out at what altitude it is not possible to fly at the optimum anymore, make sure to compare the same aircraft at the same weight."""
            ),
        ]
    )

    render_interior.callout()
    return


@app.function
def interior_condition(
    W,
    h_selected,
    Ta0,
    beta,
    CL_P,
    CLmax,
    E_P,
    min_sigma,
    sigma_selected,
    h_array,
):
    sigma_interior = (W / (E_P * Ta0)) ** (1 / beta)
    dT_interior = W / (E_P * Ta0 * sigma_selected**beta)

    if CLmax <= CL_P or sigma_interior <= min_sigma:
        return np.array([np.nan]), np.nan, np.nan, np.nan

    h_interior = atmos.altitude(sigma_interior)
    h_interior_array = h_array[h_array < h_interior]

    h_min = h_interior_array.min()
    h_max = h_interior_array.max()
    cond = 1 if h_min <= h_selected <= h_max else np.nan
    return h_interior_array, dT_interior, CL_P, cond


@app.cell
def _(
    CL_P,
    CLmax,
    E_P,
    Ta0,
    W_selected,
    beta,
    h_array,
    h_selected,
    min_sigma,
    rho_selected,
    sigma_selected,
    velocity_CL_P,
):
    # Interior computation
    h_interior_array, dTopt_interior, CLopt_interior, true_interior = interior_condition(
        W_selected, h_selected, Ta0, beta, CL_P, CLmax, E_P, min_sigma, sigma_selected, h_array
    )

    velocity_interior_selected = velocity_CL_P * true_interior
    velocity_interior_harray = velocity_CL_P * np.sqrt(rho_selected / atmos.rho(h_interior_array))

    power_interior_harray = W_selected / E_P * velocity_interior_harray
    power_interior_selected = W_selected / E_P * velocity_interior_selected
    return (
        CLopt_interior,
        dTopt_interior,
        h_interior_array,
        power_interior_harray,
        power_interior_selected,
        true_interior,
        velocity_interior_harray,
        velocity_interior_selected,
    )


@app.cell
def _(
    CLopt_interior,
    active_selection,
    configTraces,
    dTopt_interior,
    h_interior_array,
    h_selected,
    power_interior_harray,
    power_interior_selected,
    range_performance_diagrams,
    tab_value,
    title_keys,
    true_interior,
    velocity_interior_harray,
    velocity_interior_selected,
):
    if tab_value != title_keys[0]:
        mo.stop(True)
    
    # Interior graphics
    fig_interior_optimum = OptimumGridView(
        configTraces,
        h_selected,
        (velocity_interior_harray, velocity_interior_selected),
        (power_interior_harray, power_interior_selected),
        (h_interior_array, dTopt_interior, CLopt_interior, true_interior),
        f"Interior minimum power for {active_selection.full_name}",
    )

    fig_interior_optimum.update_axes_ranges(range_performance_diagrams)
    return (fig_interior_optimum,)


@app.cell
def _(fig_lift_limited, fig_maxlift_optimum, tab_value, title_keys):
    if tab_value != title_keys[1]:
        mo.stop(True)

    liftlimited_solutions = mo.vstack(
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
            fig_maxlift_optimum.figure,
        ]
    )

    liftlimited_solutions.callout()
    return


@app.cell
def _(
    a_0,
    active_selection,
    idx_h_selected,
    power_required,
    power_yrange,
    tab_value,
    title_keys,
    velocity_CLarray,
    velocity_stall_harray,
):
    if tab_value != title_keys[1]:
        mo.stop(True)
    
    fig_lift_limited = go.Figure()

    # Power curve vs CL
    fig_lift_limited.add_traces(
        [
            go.Scatter(x=velocity_CLarray, y=power_required, name="Power"),
        ]
    )

    fig_lift_limited.add_vline(
        x=velocity_stall_harray[idx_h_selected],
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
        xaxis=dict(title=r"$V \: (\text{m/s})$", range=[0, a_0]),
        yaxis=dict(title=r"$P \: (\text{W})$", range=[0, power_yrange]),
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


@app.function
def maxlift_condition(
    W,
    h_selected,
    CL_P,
    CLmax,
    E_S,
    Ta0,
    beta,
    h_array,
    min_sigma,
    sigma_selected,
):
    sigma_maxlift = (W / E_S / Ta0) ** (1 / beta)
    dT_maxlift = W / E_S / Ta0 / (sigma_selected**beta)

    if CLmax >= CL_P or sigma_maxlift <= min_sigma:
        return np.array([np.nan]), dT_maxlift, np.nan, np.nan

    h_maxlift = atmos.altitude(sigma_maxlift)
    h_maxlift_array = h_array[h_array < h_maxlift]

    h_min = h_maxlift_array.min()
    h_max = h_maxlift_array.max()
    cond = 1 if h_min <= h_selected <= h_max else np.nan

    return h_maxlift_array, dT_maxlift, CLmax, cond


@app.cell
def _(
    CL_P,
    CLmax,
    E_S,
    Ta0,
    W_selected,
    beta,
    h_array,
    h_selected,
    min_sigma,
    rho_selected,
    sigma_selected,
    velocity_CLarray,
):
    # Maxlift condition
    h_maxlift_array, dTopt_maxlift, CLopt_maxlift, true_maxlift = maxlift_condition(
        W_selected,
        h_selected,
        CL_P,
        CLmax,
        E_S,
        Ta0,
        beta,
        h_array,
        min_sigma,
        sigma_selected,
    )

    velocity_maxlift_selected = velocity_CLarray[-1] * true_maxlift
    velocity_maxlift_harray = velocity_CLarray[-1] * np.sqrt(rho_selected / atmos.rho(h_maxlift_array))

    power_maxlift_harray = W_selected / E_S * velocity_maxlift_harray
    power_maxlift_selected = W_selected / E_S * velocity_maxlift_selected
    return (
        CLopt_maxlift,
        dTopt_maxlift,
        h_maxlift_array,
        power_maxlift_harray,
        power_maxlift_selected,
        true_maxlift,
        velocity_maxlift_harray,
        velocity_maxlift_selected,
    )


@app.cell
def _(
    CLopt_maxlift,
    active_selection,
    configTraces,
    dTopt_maxlift,
    h_maxlift_array,
    h_selected,
    power_maxlift_harray,
    power_maxlift_selected,
    range_performance_diagrams,
    tab_value,
    title_keys,
    true_maxlift,
    velocity_maxlift_harray,
    velocity_maxlift_selected,
):
    if tab_value != title_keys[1]:
        mo.stop(True)
    
    # maxlift graphics
    fig_maxlift_optimum = OptimumGridView(
        configTraces,
        h_selected,
        (velocity_maxlift_harray, velocity_maxlift_selected),
        (power_maxlift_harray, power_maxlift_selected),
        (h_maxlift_array, dTopt_maxlift, CLopt_maxlift, true_maxlift),
        f"Lift-limited minimum power for {active_selection.full_name}",
    )

    fig_maxlift_optimum.update_axes_ranges(range_performance_diagrams)
    return (fig_maxlift_optimum,)


@app.cell
def _(
    fig_maxthrust_optimum,
    fig_performance_cl_eq,
    fig_thrust_limited,
    tab_value,
    title_keys,
    variables_stack,
):
    if tab_value != title_keys[2]:
        mo.stop(True)

    thrustlimited_solutions = mo.vstack(
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
            mo.md("""This tells us that the required power and drag change in opposite directions with respect to the change in $C_L$. If one decreases, then the other one has to increase, given that $\lambda_1 \lt 0$.
    This can only happen in the range of $C_L$ between $C_{L_P}$ and $C_{L_E}$, since they represent the minimum power and maximum aerodynamic efficiency (alternatively minimum drag) respectively.

    This is clearer in the performance diagram:"""),
            variables_stack,
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
            variables_stack,
            fig_maxthrust_optimum.figure,
        ]
    )

    thrustlimited_solutions.callout()
    return


@app.cell
def _(
    a_0,
    active_selection,
    drag_curve,
    drag_yrange,
    idx_h_selected,
    power_required,
    tab_value,
    title_keys,
    velocity_CL_E,
    velocity_CL_P,
    velocity_CLarray,
    velocity_stall_harray,
):
    if tab_value != title_keys[2]:
        mo.stop(True)

    fig_thrust_limited = go.Figure()

    # Power curve vs CL
    fig_thrust_limited.add_traces(
        [
            go.Scattergl(x=velocity_CLarray, y=power_required, name="Power"),
            go.Scattergl(
                x=velocity_CLarray,
                y=drag_curve,
                name="Drag",
                yaxis="y2",
            ),
            go.Scattergl(
                x=[velocity_stall_harray[idx_h_selected] - 20, a_0 * 2],
                y=[0.1 * drag_yrange, 0.1 * drag_yrange],
                mode="lines",
                line=dict(color="grey", width=1),
                yaxis="y2",
                showlegend=False,
            ),
            go.Scattergl(
                x=[velocity_stall_harray[idx_h_selected] - 20],
                y=[0.1 * drag_yrange],
                mode="markers",
                yaxis="y2",
                marker=dict(color="grey", size=10, symbol="arrow-left"),
                showlegend=False,
            ),
        ]
    )

    # Add CL_P and CL_E curves
    fig_thrust_limited.add_vline(
        x=velocity_CL_E,
        line_dash="dot",
        annotation=dict(text="$C_{L_E}$", xshift=10, yshift=-10),
        line=dict(color="white"),
    )
    fig_thrust_limited.add_vline(
        x=velocity_CL_P,
        line_dash="dot",
        annotation=dict(text="$C_{L_P}$", xshift=10, yshift=-10),
        line=dict(color="white"),
    )
    fig_thrust_limited.add_vrect(
        x0=velocity_CL_P,
        x1=velocity_CL_E,
        fillcolor="green",
        opacity=0.25,
        line_width=0,
    )

    fig_thrust_limited.add_vline(
        x=velocity_stall_harray[idx_h_selected],
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
        xaxis=dict(title=r"$V \: (\text{m/s})$", range=[0, a_0]),
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
    a_0,
    active_selection,
    drag_curve,
    drag_yrange,
    power_required,
    power_yrange,
    tab_value,
    title_keys,
    velocity_CL_E,
    velocity_CL_P,
    velocity_CLarray,
):
    if tab_value != title_keys[2]:
        mo.stop(True)

    fig_performance_cl_eq = go.Figure()

    fig_performance_cl_eq.add_traces(
        [
            go.Scatter(x=velocity_CLarray, y=power_required, name="Power"),
            go.Scatter(
                x=velocity_CLarray,
                y=drag_curve,
                name="Drag",
                yaxis="y2",
            ),
        ]
    )


    # fig_performance_cl_eq.add_vline(
    #     x=float(velocity_plus_solution),
    #     line_dash="dot",
    #     annotation=dict(text="$C_{L}^{*+}$", xshift=10, yshift=-10),
    #     line=dict(color="white"),
    # )
    # fig_performance_cl_eq.add_vline(
    #     x=float(velocity_minus_solution),
    #     line_dash="dot",
    #     annotation=dict(text="$C_{L}^{*-}$", xshift=10, yshift=-10),
    #     line=dict(color="white"),
    # )
    fig_performance_cl_eq.add_vline(
        x=float(velocity_CL_E),
        line_dash="dot",
        annotation=dict(text="$C_{L_E}$", xshift=10, yshift=-10),
        line=dict(color="white"),
    )
    fig_performance_cl_eq.add_vline(
        x=float(velocity_CL_P),
        line_dash="dot",
        annotation=dict(text="$C_{L_P}$", xshift=10, yshift=-10),
        line=dict(color="white"),
    )
    fig_performance_cl_eq.add_vrect(
        x0=float(velocity_CL_P),
        x1=float(velocity_CL_E),
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
        xaxis=dict(title=r"$V \: (\text{m/s})$", range=[0, a_0]),
        yaxis=dict(title=r"$P \: (\text{W})$", range=[0, power_yrange]),
        yaxis2=dict(title=r"$D (\text{N})$", overlaying="y", side="right", range=[0, drag_yrange]),
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
    return (fig_performance_cl_eq,)


@app.cell
def _(sigma_array):
    def maxthrust_condition(W, h_selected, K, E_max, E_P, h_array, Ta0, beta, min_sigma):
        B = W / E_max / Ta0
        max_sigma_maxthrust = (W / E_P / Ta0) ** (1 / beta)
        min_sigma_maxthrust = (B) ** (1 / beta)

        if min_sigma_maxthrust <= min_sigma:
            return np.array([np.nan]), 1, np.nan, False

        dT_maxthrust = 1
        max_h_maxthrust = atmos.altitude(min_sigma_maxthrust)
        min_h_maxthrust = atmos.altitude(max_sigma_maxthrust)

        h_maxthrust_array = h_array[(h_array > min_h_maxthrust) & (h_array < max_h_maxthrust)]
        sigma_maxthrust = sigma_array[np.isin(h_array, h_maxthrust_array)]

        A = Ta0 * sigma_maxthrust**beta / (2 * K * W)
        B = (B / (sigma_maxthrust**beta)) ** 2

        CL_maxthrust = np.where(B < 1, A * (1 + np.sqrt(1 - B)), np.nan)

        h_min = h_maxthrust_array.min()
        h_max = h_maxthrust_array.max()
        cond = 1 if h_min <= h_selected <= h_max else np.nan
        return (
            h_maxthrust_array,
            dT_maxthrust,
            CL_maxthrust,
            cond,
        )
    return (maxthrust_condition,)


@app.cell
def _(
    CD0,
    E_P,
    E_max,
    K,
    S,
    Ta0,
    W_selected,
    beta,
    h_array,
    h_selected,
    maxthrust_condition,
    min_sigma,
    rho_array,
    rho_selected,
):
    # Maxthrust computations
    h_maxthrust_array, dTopt_maxthrust, CLopt_maxthrust, true_maxthrust = maxthrust_condition(
        W_selected, h_selected, K, E_max, E_P, h_array, Ta0, beta, min_sigma
    )

    CL_maxthrust_selected = (
        CLopt_maxthrust[h_selected == h_maxthrust_array][0]
        if h_maxthrust_array.size > 0 and np.any(h_selected == h_maxthrust_array)
        else np.nan
    )

    rho_maxthrust_array = rho_array[np.isin(h_array, h_maxthrust_array)]
    E_maxthrust = CLopt_maxthrust / (CD0 + K * CLopt_maxthrust**2)
    E_maxthrust_selected = CL_maxthrust_selected / (CD0 + K * CL_maxthrust_selected**2)

    velocity_maxthrust_harray = np.sqrt(2 * W_selected / (rho_maxthrust_array * S * CLopt_maxthrust))
    velocity_maxthrust_selected = np.sqrt(2 * W_selected / (rho_selected * S * CL_maxthrust_selected))

    power_maxthrust_harray = W_selected / E_maxthrust * velocity_maxthrust_harray
    power_maxthrust_selected = W_selected / E_maxthrust_selected * velocity_maxthrust_selected

    return (
        CL_maxthrust_selected,
        dTopt_maxthrust,
        h_maxthrust_array,
        power_maxthrust_harray,
        power_maxthrust_selected,
        true_maxthrust,
        velocity_maxthrust_harray,
        velocity_maxthrust_selected,
    )


@app.cell
def _(
    CL_maxthrust_selected,
    active_selection,
    configTraces,
    dTopt_maxthrust,
    h_maxthrust_array,
    h_selected,
    power_maxthrust_harray,
    power_maxthrust_selected,
    range_performance_diagrams,
    tab_value,
    title_keys,
    true_maxthrust,
    velocity_maxthrust_harray,
    velocity_maxthrust_selected,
):
    if tab_value != title_keys[2]:
        mo.stop(True)
    # Maxthrust graphics
    fig_maxthrust_optimum = OptimumGridView(
        configTraces,
        h_selected,
        (velocity_maxthrust_harray, velocity_maxthrust_selected),
        (power_maxthrust_harray, power_maxthrust_selected),
        (h_maxthrust_array, dTopt_maxthrust, CL_maxthrust_selected, true_maxthrust),
        f"Thrust-limited minimum power for {active_selection.full_name}",
    )

    fig_maxthrust_optimum.update_axes_ranges(range_performance_diagrams)
    return (fig_maxthrust_optimum,)


@app.cell
def _(fig_maxliftThrust_optimum, tab_value, title_keys, variables_stack):
    if tab_value != title_keys[3]:
        mo.stop(True)

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
            variables_stack,
            fig_maxliftThrust_optimum.figure,
        ]
    )

    liftThrustlimited_solutions.callout()
    return


@app.function
def maxliftThrust_condition(W, Ta0, E_S, beta, CL_E, CL_P, CLmax):
    sigma_maxliftThrust = (W / Ta0 / E_S) ** (1 / beta)
    h_maxliftThrust_selected = atmos.altitude(sigma_maxliftThrust)

    if CLmax > CL_P or CLmax == CL_E:
        return h_maxliftThrust_selected, sigma_maxliftThrust, np.nan

    condition = True

    return (
        h_maxliftThrust_selected,
        sigma_maxliftThrust,
        condition,
    )


@app.cell
def _(
    CL_E,
    CL_P,
    CLmax,
    E_S,
    Ta0,
    W_selected,
    beta,
    drag_curve,
    rho_selected,
    sigma_selected,
    thrust_vector,
    velocity_CLarray,
):
    # Max lift Max thrust
    h_maxliftThrust, sigma_maxliftThrust, true_maxliftThrust = maxliftThrust_condition(
        W_selected, Ta0, E_S, beta, CL_E, CL_P, CLmax
    )

    maxliftThrust_multiplier = np.sqrt(rho_selected / (atmos.rho0 * sigma_maxliftThrust))


    thrust_vector_maxliftThrust = thrust_vector * (sigma_maxliftThrust / sigma_selected) ** beta
    velocity_maxliftThrust_CLarray = velocity_CLarray * maxliftThrust_multiplier
    velocity_maxliftThrust_selected = velocity_maxliftThrust_CLarray[-1]

    power_required_maxliftThrust = drag_curve * velocity_maxliftThrust_CLarray / 1e3
    power_maxliftThrust_selected = W_selected / E_S * velocity_maxliftThrust_selected / 1e3
    return (
        h_maxliftThrust,
        maxliftThrust_multiplier,
        power_maxliftThrust_selected,
        power_required_maxliftThrust,
        sigma_maxliftThrust,
        thrust_vector_maxliftThrust,
        true_maxliftThrust,
        velocity_maxliftThrust_CLarray,
        velocity_maxliftThrust_selected,
    )


@app.cell
def _(
    CL_array,
    CLmax,
    active_selection,
    beta,
    constraint,
    dT_array,
    drag_curve,
    drag_yrange,
    h_maxliftThrust,
    h_selected,
    mach_trace,
    maxliftThrust_multiplier,
    power_maxliftThrust_selected,
    power_maxthrust_harray,
    power_required_maxliftThrust,
    power_yrange,
    sigma_maxliftThrust,
    sigma_selected,
    stall_trace,
    tab_value,
    thrust_vector_maxliftThrust,
    title_keys,
    true_maxliftThrust,
    velocity_CL_E,
    velocity_CL_P,
    velocity_maxliftThrust_CLarray,
    velocity_maxliftThrust_selected,
):
    if tab_value != title_keys[3]:
        mo.stop(True)
            
    power_surface_maxliftThrust = np.broadcast_to(
        power_required_maxliftThrust[np.newaxis, :],  # Shape: (101, 1)
        (len(CL_array), len(dT_array)),  # Target shape: (101, 101)
    )

    constraint_maxliftThrust = constraint * (sigma_selected / sigma_maxliftThrust) ** beta

    min_colorbar_maxliftThrust = np.min(power_required_maxliftThrust)
    max_colorbar_maxliftThrust = min_colorbar_maxliftThrust * 2
    zcolorbar_maxliftThrust = (min_colorbar_maxliftThrust, max_colorbar_maxliftThrust)

    configTraces_maxliftThrust = plot_utils.ConfigTraces(
        CL_array,
        dT_array,
        constraint_maxliftThrust,
        drag_curve,
        thrust_vector_maxliftThrust,
        power_required_maxliftThrust,
        thrust_vector_maxliftThrust * velocity_maxliftThrust_CLarray / 1e3,
        power_surface_maxliftThrust,
        velocity_maxliftThrust_CLarray,
        velocity_CL_P * maxliftThrust_multiplier,
        velocity_CL_E * maxliftThrust_multiplier,
        velocity_maxliftThrust_selected,
        velocity_maxliftThrust_selected,
        (drag_yrange, power_yrange, CLmax),
        zcolorbar_maxliftThrust,
        mach_trace,
        stall_trace,
    )


    # Maxliftthrust graphics
    fig_maxliftThrust_optimum = OptimumGridView(
        configTraces_maxliftThrust,
        h_selected,
        (velocity_maxliftThrust_CLarray, velocity_maxliftThrust_selected),
        (power_maxthrust_harray, power_maxliftThrust_selected),
        (h_maxliftThrust, 1, true_maxliftThrust, np.nan),
        f"Thrust-limited minimum power for {active_selection.full_name}",
        equality=True,
    )
    return (fig_maxliftThrust_optimum,)


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
def _(mass_stack):
    mass_stack
    return


@app.cell
def _():
    mo.md(r"""
    ## Summary
    """)
    return


@app.cell
def _(
    a_harray,
    h_array,
    h_interior_array,
    h_maxliftThrust,
    h_maxlift_array,
    h_maxthrust_array,
    velocity_interior_harray,
    velocity_maxliftThrust_selected,
    velocity_maxlift_harray,
    velocity_maxthrust_harray,
    velocity_stall_harray,
):
    plot_utils.create_final_flightenvelope(
        velocity_stall_harray,
        a_harray,
        h_array,
        (
            np.concat((h_interior_array, h_maxthrust_array)),
            np.concat((velocity_interior_harray, velocity_maxthrust_harray)),
            True,
        ),
        (h_maxlift_array, velocity_maxlift_harray, True),
        (h_maxthrust_array, velocity_maxthrust_harray, True),
        (h_maxliftThrust, velocity_maxliftThrust_selected, False),
    )
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
