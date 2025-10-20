import marimo

__generated_with = "0.17.0"
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
    dT_slider = mo.ui.slider(
        start=0, stop=1, step=0.1, label=r"$\delta_T$", value=0.5
    )

    meshgrid_n = 101
    xy_lowerbound = -0.1

    dT_array = np.linspace(0, 1, meshgrid_n)  # -
    h_array = np.linspace(0, 20e3, meshgrid_n)  # meters

    m_slider = mo.ui.slider(start=0, stop=1, step=0.1, label=r"", show_value=True)

    h_slider = mo.ui.slider(
        start=0,
        stop=20,
        label=r"Altitude (km)",
        value=10,
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
    return (
        a_0,
        ac_table,
        dT_array,
        dT_slider,
        data,
        h_array,
        h_slider,
        hover_name,
        labels,
        m_slider,
        meshgrid_n,
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


    interior_title = f"Interior minimum power for {active_selection.full_name}"
    maxlift_title = f"Lift-limited minimum power for {active_selection.full_name}"
    maxthrust_title = (
        f"Thrust-limited minimum power for {active_selection.full_name}"
    )
    maxliftThrust_title = (
        f"Lift-thrust limited minimum power for {active_selection.full_name}"
    )
    final_fig_title = (
        f"Flight envelope for minimum power for {active_selection.full_name}"
    )

    axes = [CL_array, dT_array]
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
        axes,
        beta,
        final_fig_title,
        interior_title,
        maxliftThrust_title,
        maxlift_title,
        maxthrust_title,
    )


@app.cell
def _(CL_array, CL_slider):
    # Define variables, this cell runs every time the CL slider is run
    step_CL = CL_array[2] - CL_array[1]
    CL_selected = float(CL_slider.value)
    idx_CL_selected = int((CL_selected - CL_array[0]) / step_CL)
    return (idx_CL_selected,)


@app.cell
def _(E_array, MTOM, OEM, m_slider):
    # Define variables, this cell runs every time the mass slider is run
    W_selected = (OEM + (MTOM - OEM) * m_slider.value) * atmos.g0  # Netwons
    drag_curve = W_selected / E_array
    return W_selected, drag_curve


@app.cell
def _(h_array, h_slider):
    # Define variables, this cell runs every time the altitude slider is run
    h_selected = int(h_slider.value * 1e3)  # meters
    step_h = h_array[1] - h_array[0]
    idx_h_selected = int((h_selected - h_array[0]) / step_h)

    a_selected = atmos.a(h_selected)
    a_harray = atmos.a(h_array)
    sigma_selected = atmos.rhoratio(h_selected)
    sigma_array = atmos.rhoratio(h_array)
    min_sigma = atmos.rhoratio(atmos.hmax)
    rho_selected = atmos.rho(h_selected)
    return (
        a_harray,
        a_selected,
        h_selected,
        idx_h_selected,
        min_sigma,
        rho_selected,
        sigma_selected,
    )


@app.cell
def _(
    CD0,
    CL_E,
    CL_P,
    CL_array,
    CLmax,
    E_P,
    E_S,
    E_max,
    K,
    S,
    Ta0,
    W_selected,
    a_0,
    a_harray,
    a_selected,
    axes,
    beta,
    drag_curve,
    h_array,
    h_selected,
    h_slider,
    idx_CL_selected,
    idx_h_selected,
    labels,
    m_slider,
    min_sigma,
    rho_selected,
    sigma_selected,
    xy_lowerbound,
):
    # Define constants dependent on both sliders
    mass_stack = mo.hstack(
        [mo.md("**OEW**"), m_slider, mo.md("**MTOW**")],
        align="start",
        justify="start",
    )
    variables_stack = mo.hstack([mass_stack, h_slider])

    # Calculate necessary velocities
    velocity_CL_E = float(velocity(W_selected, h_selected, CL_E, S, False))
    velocity_CL_P = float(velocity(W_selected, h_selected, CL_P, S, False))
    velocity_stall_harray = velocity(W_selected, h_array, CLmax, S, False)
    velocity_CLarray = velocity(W_selected, h_selected, CL_array, S, False)

    velocity_user_selected = velocity_CLarray[idx_CL_selected]
    velocity_stall_selected = velocity_stall_harray[idx_h_selected]
    velocity_CLarray_capped = np.minimum(velocity_CLarray, a_selected)

    idx_a_capped = np.argmin(velocity_CLarray_capped == a_selected)

    thrust = np.repeat(Ta0 * sigma_selected**beta, len(CL_array))
    power_available = thrust * velocity_CLarray
    power_curve = drag_curve * velocity_CLarray
    power_curve_selected = power_curve[idx_CL_selected]

    power_surface = np.tile(power_curve, (len(CL_array), 1))
    min_colorbar = np.min(power_curve) / 1e3
    max_colorbar = min_colorbar * 2
    zcolorbar = (min_colorbar, max_colorbar)
    constraint = drag_curve / Ta0 / (sigma_selected**beta)
    power_available = thrust * velocity_CLarray

    V_ticks = np.linspace(velocity_CLarray[-1] - 10, a_selected, 8)

    CL_ticks = 2 * W_selected / (rho_selected * S * V_ticks**2)
    thrust_capped = thrust[idx_a_capped:]
    velocity_CLarray_capped = velocity_CLarray_capped[idx_a_capped:]
    drag_curve_capped = drag_curve[idx_a_capped:]
    power_curve_capped = power_curve[idx_a_capped:]
    power_available_capped = power_available[idx_a_capped:]

    # Compute all necessary traces
    domain_config_traces = plot_utils.config_domain_traces(
        axes, power_surface / 1e3, constraint, labels, zcolorbar
    )

    # Interior computation
    h_interior_array, dTopt_interior, CLopt_interior = interior_condition(
        W_selected, Ta0, beta, CL_P, CLmax, E_P, min_sigma, sigma_selected, h_array
    )

    velocity_interior_harray = velocity(
        W_selected, h_interior_array, CLopt_interior, S
    )

    power_interior_harray = W_selected / E_P * velocity_interior_harray

    velocity_interior_selected = _defaults.safe_index(
        velocity_interior_harray, idx_h_selected
    )
    power_interior_selected = _defaults.safe_index(
        power_interior_harray, idx_h_selected
    )

    # Maxlift computation
    h_maxlift_array, dTopt_maxlift, CLopt_maxlift = maxlift_condition(
        W_selected,
        CL_P,
        CLmax,
        E_S,
        Ta0,
        beta,
        h_array,
        min_sigma,
        sigma_selected,
    )

    velocity_maxlift_harray = velocity(
        W_selected, h_maxlift_array, CLopt_maxlift, S
    )

    power_maxlift_harray = W_selected / E_S * velocity_maxlift_harray

    velocity_maxlift_selected = _defaults.safe_index(
        velocity_maxlift_harray, idx_h_selected
    )
    power_maxlift_selected = _defaults.safe_index(
        power_maxlift_harray, idx_h_selected
    )

    # Maxthrust computation
    (
        h_maxthrust_array,
        dTopt_maxthrust,
        CLopt_maxthrust,
        idx_maxthrust,
    ) = maxthrust_condition(
        W_selected,
        K,
        E_max,
        E_P,
        h_array,
        idx_h_selected,
        Ta0,
        beta,
        min_sigma,
        h_selected,
    )

    velocity_maxthrust_harray = velocity(
        W_selected, h_maxthrust_array, CLopt_maxthrust, S, cap=False
    )

    power_maxthrust_harray = (
        W_selected
        * (CD0 + K * CLopt_maxthrust**2)
        / CLopt_maxthrust
        * velocity_maxthrust_harray
    )

    velocity_maxthrust_selected = _defaults.safe_index(
        velocity_maxthrust_harray, idx_maxthrust
    )
    power_maxthrust_selected = _defaults.safe_index(
        power_maxthrust_harray, idx_maxthrust
    )

    CLopt_maxthrust_selected = _defaults.safe_index(CLopt_maxthrust, idx_maxthrust)

    # Maxthrust maxlift computation
    (
        h_maxliftThrust_selected,
        sigma_maxliftThrust,
        dTopt_maxliftThrust,
        CLopt_maxliftThrust,
    ) = maxliftThrust_condition(W_selected, Ta0, E_S, beta, CL_E, CL_P, CLmax)

    velocity_maxliftThrust_CLarray = velocity(
        W_selected, h_maxliftThrust_selected, CL_array, S, False
    )
    velocity_maxliftThrust_nonan = velocity_maxliftThrust_CLarray[-1]
    velocity_maxliftThrust_selected = (
        velocity_maxliftThrust_CLarray[-1]
        if ~np.isnan(CLopt_maxliftThrust)
        else np.nan
    )

    power_maxliftThrust_curve = drag_curve * velocity_maxliftThrust_CLarray

    power_maxliftThrust_selected = drag_curve * velocity_maxliftThrust_selected

    power_maxliftThrust_surface = np.tile(
        power_maxliftThrust_curve, (len(CL_array), 1)
    )

    constraint_maxliftThrust = drag_curve / Ta0 / (sigma_maxliftThrust**beta)

    flight_env_config_maxliftThrust = plot_utils.config_flight_env_traces(
        h_array,
        velocity_stall_harray,
        a_harray,
    )

    thrust_maxliftThrust = np.repeat(
        Ta0 * atmos.rhoratio(h_maxliftThrust_selected) ** beta, len(CL_array)
    )
    min_colorbar_maxliftThrust = np.min(power_maxliftThrust_curve) / 1e3
    max_colorbar_maxliftThrust = min_colorbar_maxliftThrust * 2

    domain_config_maxliftThrust = plot_utils.config_domain_traces(
        axes,
        power_maxliftThrust_surface / 1e3,
        constraint_maxliftThrust,
        labels,
        [min_colorbar_maxliftThrust, max_colorbar_maxliftThrust],
    )

    optima_diagram_ranges = (
        velocity_CLarray[-1] - 10,
        a_selected,
        min(min(power_curve), min(power_available_capped)) * 0.9 / 1e3,
        max(power_available) / 1e3,
        min(drag_curve) * 0.95,
        max(thrust),
        xy_lowerbound,
        CLmax + 0.05,
        xy_lowerbound,
        1 + 0.05,
        xy_lowerbound,
        a_0,
        xy_lowerbound,
        20,
    )

    A = Ta0 * sigma_selected**beta / (2 * K * W_selected)
    B = (W_selected / (Ta0 * sigma_selected**beta * E_max)) ** 2
    if B < 1:
        CL_plus = A * (1 + np.sqrt(1 - B))
        CL_minus = A * (1 - np.sqrt(1 - B))
    else:
        CL_plus = np.nan
        CL_minus = np.nan

    velocity_CL_plus = float(velocity(W_selected, h_selected, CL_plus, S, False))

    velocity_CL_minus = float(velocity(W_selected, h_selected, CL_minus, S, False))

    y_performance_ranges = (
        min(drag_curve),
        max(drag_curve),
        min(power_curve) / 1e3,
        max(power_curve) / 1e3,
    )
    return (
        CL_ticks,
        CLopt_interior,
        CLopt_maxlift,
        CLopt_maxliftThrust,
        CLopt_maxthrust_selected,
        constraint,
        dTopt_interior,
        dTopt_maxlift,
        dTopt_maxliftThrust,
        dTopt_maxthrust,
        domain_config_maxliftThrust,
        domain_config_traces,
        drag_curve_capped,
        flight_env_config_maxliftThrust,
        h_interior_array,
        h_maxliftThrust_selected,
        h_maxlift_array,
        h_maxthrust_array,
        idx_a_capped,
        mass_stack,
        max_colorbar,
        min_colorbar,
        optima_diagram_ranges,
        power_available,
        power_curve,
        power_curve_capped,
        power_curve_selected,
        power_interior_selected,
        power_maxliftThrust_curve,
        power_maxliftThrust_selected,
        power_maxlift_selected,
        power_maxthrust_selected,
        power_surface,
        thrust,
        thrust_maxliftThrust,
        variables_stack,
        velocity_CL_E,
        velocity_CL_P,
        velocity_CL_minus,
        velocity_CL_plus,
        velocity_CLarray,
        velocity_CLarray_capped,
        velocity_interior_harray,
        velocity_interior_selected,
        velocity_maxliftThrust_CLarray,
        velocity_maxliftThrust_selected,
        velocity_maxlift_harray,
        velocity_maxlift_selected,
        velocity_maxthrust_harray,
        velocity_maxthrust_selected,
        velocity_stall_harray,
        velocity_stall_selected,
        y_performance_ranges,
    )


@app.cell
def _(
    CL_ticks,
    CLopt_interior,
    CLopt_maxlift,
    CLopt_maxliftThrust,
    CLopt_maxthrust_selected,
    a_harray,
    dTopt_interior,
    dTopt_maxlift,
    dTopt_maxliftThrust,
    dTopt_maxthrust,
    domain_config_maxliftThrust,
    domain_config_traces,
    drag_curve,
    drag_curve_capped,
    final_fig_title,
    flight_env_config_maxliftThrust,
    h_array,
    h_interior_array,
    h_maxliftThrust_selected,
    h_maxlift_array,
    h_maxthrust_array,
    h_selected,
    hover_name,
    idx_a_capped,
    interior_title,
    maxliftThrust_title,
    maxlift_title,
    maxthrust_title,
    optima_diagram_ranges,
    power_available,
    power_curve,
    power_curve_capped,
    power_interior_selected,
    power_maxliftThrust_curve,
    power_maxliftThrust_selected,
    power_maxlift_selected,
    power_maxthrust_selected,
    thrust,
    thrust_maxliftThrust,
    velocity_CL_E,
    velocity_CL_P,
    velocity_CL_minus,
    velocity_CL_plus,
    velocity_CLarray,
    velocity_CLarray_capped,
    velocity_interior_harray,
    velocity_interior_selected,
    velocity_maxliftThrust_CLarray,
    velocity_maxliftThrust_selected,
    velocity_maxlift_harray,
    velocity_maxlift_selected,
    velocity_maxthrust_harray,
    velocity_maxthrust_selected,
    velocity_stall_harray,
    velocity_stall_selected,
    y_performance_ranges,
):
    # Graphic elements, always run when almost any interactive element is modified
    flight_env_config_traces = plot_utils.config_flight_env_traces(
        h_array,
        velocity_stall_harray,
        a_harray,
    )

    fig_optimum_stencil = plot_utils.create_optima_grid_stencil(
        velocity_CLarray,
        domain_config_traces,
        flight_env_config_traces,
        power_curve / 1e3,
        drag_curve,
        optima_diagram_ranges,
        CL_ticks,
    )

    # figure for maxthrust - maxlift
    fig_maxliftThrust_optimum = plot_utils.create_optima_grid_stencil(
        velocity_maxliftThrust_CLarray,
        domain_config_maxliftThrust,
        flight_env_config_maxliftThrust,
        power_maxliftThrust_curve / 1e3,
        drag_curve,
        optima_diagram_ranges,
        CL_ticks,
    )

    combined_performance = plot_utils.create_overlayed_perf_diagram_stencil(
        CL_ticks,
        optima_diagram_ranges,
    )

    fig_interior_optimum = copy.deepcopy(fig_optimum_stencil)
    fig_maxlift_optimum = copy.deepcopy(fig_optimum_stencil)
    fig_maxthrust_optimum = copy.deepcopy(fig_optimum_stencil)
    fig_lift_limited = copy.deepcopy(combined_performance)


    plot_utils.draw_optima(
        fig_maxlift_optimum,
        velocity_CLarray,
        dTopt_maxlift * thrust,
        dTopt_maxlift * power_available / 1e3,
        velocity_maxlift_harray,
        velocity_maxlift_selected,
        h_maxlift_array,
        h_selected,
        CLopt_maxlift,
        dTopt_maxlift,
        power_maxlift_selected / 1e3,
        power_maxlift_selected / 1e3,
        y_performance_ranges,
        hover_name,
        idx_a_capped,
        equality=False,
    )

    plot_utils.draw_optima(
        fig_interior_optimum,
        velocity_CLarray,
        dTopt_interior * thrust,
        dTopt_interior * power_available / 1e3,
        velocity_interior_harray,
        velocity_interior_selected,
        h_array,
        h_selected,
        CLopt_interior,
        dTopt_interior,
        power_interior_selected / 1e3,
        power_interior_selected / 1e3,
        y_performance_ranges,
        hover_name,
        idx_a_capped,
        equality=False,
    )

    plot_utils.draw_optima(
        fig_maxthrust_optimum,
        velocity_CLarray,
        thrust,
        power_available / 1e3,
        velocity_maxthrust_harray,
        velocity_maxthrust_selected,
        h_maxthrust_array,
        h_selected,
        CLopt_maxthrust_selected,
        dTopt_maxthrust,
        power_maxthrust_selected / 1e3,
        power_maxthrust_selected / 1e3,
        y_performance_ranges,
        hover_name,
        idx_a_capped,
        equality=False,
    )

    plot_utils.draw_optima(
        fig_maxliftThrust_optimum,
        velocity_maxliftThrust_CLarray,
        thrust_maxliftThrust,
        thrust_maxliftThrust * velocity_maxliftThrust_CLarray / 1e3,
        None,
        velocity_maxliftThrust_selected,
        None,
        h_selected,
        CLopt_maxliftThrust,
        dTopt_maxliftThrust,
        power_maxliftThrust_selected / 1e3,
        power_maxliftThrust_selected / 1e3,
        y_performance_ranges,
        hover_name,
        idx_a_capped,
        equality=True,
    )

    plot_utils.add_trace(
        fig_lift_limited,
        velocity_CLarray_capped,
        power_curve_capped / 1e3,
        "x1",
        "y1",
        "P",
    )

    plot_utils.add_trace(
        fig_lift_limited,
        velocity_CLarray_capped,
        drag_curve_capped,
        "x1",
        "y2",
        "D",
    )

    fig_thrust_limited = copy.deepcopy(fig_lift_limited)

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
        x=velocity_stall_selected,
        line_dash="dot",
        annotation=dict(text=r"$V_{\mathrm{stall}}$", xshift=10, yshift=-10),
        line=dict(color="rgba(255, 0, 0, 0.3)"),
    )

    fig_performance_CL_equation = copy.deepcopy(fig_thrust_limited)

    fig_performance_CL_equation.add_vline(
        x=velocity_CL_plus,
        line_dash="dot",
        annotation=dict(text="$C_{L}^{*+}$", xshift=10, yshift=-10),
        line=dict(color="white"),
    )
    fig_performance_CL_equation.add_vline(
        x=velocity_CL_minus,
        line_dash="dot",
        annotation=dict(text="$C_{L}^{*-}$", xshift=10, yshift=-10),
        line=dict(color="white"),
    )
    fig_final_flightenvelope = plot_utils.create_final_flightenvelope(
        flight_env_config_traces,
        (
            np.concatenate((h_interior_array, h_maxthrust_array)),
            np.concatenate((velocity_interior_harray, velocity_maxthrust_harray)),
            True,
        ),
        (np.nan, np.nan, False),
        (h_maxlift_array, velocity_maxlift_harray, True),
        (h_maxliftThrust_selected, velocity_maxliftThrust_selected, False),
    )

    plot_utils.add_title(fig_final_flightenvelope, final_fig_title)
    plot_utils.add_title(fig_interior_optimum, interior_title)
    plot_utils.add_title(fig_maxlift_optimum, maxlift_title)
    plot_utils.add_title(fig_maxthrust_optimum, maxthrust_title)
    plot_utils.add_title(fig_maxliftThrust_optimum, maxliftThrust_title)
    return (
        fig_final_flightenvelope,
        fig_interior_optimum,
        fig_lift_limited,
        fig_maxliftThrust_optimum,
        fig_maxlift_optimum,
        fig_maxthrust_optimum,
        fig_performance_CL_equation,
        fig_thrust_limited,
    )


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
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
    """
    )
    return


@app.cell(hide_code=True)
def _(
    CL_array,
    CL_slider,
    active_selection,
    constraint,
    dT_array,
    dT_slider,
    max_colorbar,
    min_colorbar,
    power_curve_selected,
    power_surface,
    xy_lowerbound,
):
    # Create go.Figure() object
    fig_initial = go.Figure()

    # Minimum velocity surface
    fig_initial.add_traces(
        [
            go.Surface(
                x=CL_array,
                y=dT_array,
                z=power_surface / 1e3,
                opacity=0.9,
                name="Power",
                colorscale="viridis",
                cmin=min_colorbar,
                cmax=max_colorbar,
                colorbar={"title": "Power (kW)"},
            ),
            go.Scatter3d(
                x=CL_array,
                y=constraint,
                z=power_surface[0] / 1e3,
                opacity=1,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[CL_array[-15]],
                y=[constraint[-15]],
                z=[power_surface[0, -15] / 1e3 + 450],
                opacity=1,
                textposition="middle left",
                mode="markers+text",
                text=["g<sub>1</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                textfont=dict(size=14, family="Arial"),
                showlegend=False,
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[CL_slider.value],
                y=[dT_slider.value],
                z=[power_curve_selected / 1e3],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=3,
                    color="white",
                    symbol="circle",
                ),
                name="Design Point",
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub> : %{y}<br>P: %{z}<extra>%{fullData.name}</extra>",
            ),
        ]
    )
    camera = dict(eye=dict(x=1.35, y=1.35, z=1.35))

    fig_initial.update_layout(
        scene=dict(
            xaxis=dict(
                title="C<sub>L</sub> (-)",
                range=[xy_lowerbound, active_selection["CLmax_ld"]],
            ),
            yaxis=dict(title="δ<sub>T</sub> (-)", range=[xy_lowerbound, 1]),
            zaxis=dict(
                title="P (kW)",
                range=[0, max_colorbar],
            ),
        ),
    )
    fig_initial.update_layout(
        scene_camera=camera,
        title={
            "text": f"Minimum power domain for {active_selection.full_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        },
    )

    mo.output.clear()
    return (fig_initial,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
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
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
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
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""In the interactive graph below, select a simplified jet aircraft of your choice and experiment in finding an optimum by changing the control variables, $C_L$ and $\delta_T$. The design point is marked in white in the 3D power surface.""")
    return


@app.cell
def _(ac_table):
    ac_table
    return


@app.cell(hide_code=True)
def _(CL_slider, dT_slider):
    mo.md(f"""Here you can modify the control variables to understand how it affects the design: {mo.hstack([dT_slider, CL_slider])}""")
    return


@app.cell(hide_code=True)
def _(variables_stack):
    variables_stack
    return


@app.cell(hide_code=True)
def _(fig_initial):
    fig_initial
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
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
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    The multipliers $\lambda_1, \mu_1, \mu_2$ have to meet the following conditions for an optimal solution of the optimization problem $(C_L^*, \delta_T^*)$ to exist.

    **A. Stationarity ($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \frac{\partial \mathcal{L}}{\partial C_L} = \sqrt{\frac{2W^3}{\rho S}}\left(-\frac{3}{2}C_{D_0}C_L^{-5/2} + \frac{1}{2} K C_L^{-1/2}\right) - \lambda_1 W \left(\frac{KC_L^2 -C_{D_0}}{C_L^2}\right) + \mu_1= 0$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1 T_{a0}\sigma^\beta+ \mu_2= 0$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $\displaystyle \delta_T T_{a0}\sigma^\beta - W \frac{C_{D_0} + K C_L^2}{C_L} = 0$
    4.  $C_L - C_{L_\mathrm{max}} \le 0$
    5.  $\delta_T - 1 \le 0$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    6.  $\mu_1, \mu_2 \ge 0$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraint have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

    7.  $\mu_1 (C_L - C_{L_\mathrm{max}}) = 0$
    8. $\mu_3 (\delta_T - 1) = 0$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## KKT analysis

    We can now proceed to systematically examine the conditions where various inequality constraints are active 
    or inactive.

    ### _Interior solutions_ 

    In this case: $C_L \lt C_{L_{\mathrm{max}}}$, $\delta_T \lt 1$, $\mu_1=\mu_2= 0$

    from stationarity condition (2): $\lambda_1 = 0$

    from stationarity condition (1): 

    $$
    -\frac{3}{2}C_{D_0} C_L^{-5/2}+\frac{1}{2}KC_L^{-1/2}= 0 \quad \Rightarrow \quad KC_L^2 = 3C_{D_0} \quad \Rightarrow \quad C_L^* = \sqrt{\frac{3C_{D_0}}{K}} = \sqrt{3}C_{L_E} = C_{L_P}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
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
    """
    )
    return


@app.function
def interior_condition(
    W, Ta0, beta, CL_P, CLmax, E_P, min_sigma, sigma_selected, h_array
):
    sigma_interior = (W / E_P / Ta0) ** (1 / beta)

    dT_interior = W / E_P / Ta0 / (sigma_selected**beta)

    if CLmax > CL_P and sigma_interior > min_sigma:
        h_interior = atmos.altitude(sigma_interior)

        idx_interior = np.abs(h_array - h_interior).argmin()

        h_interior_array = h_array[:idx_interior]

        CL_interior = CL_P
    else:
        h_interior_array, dT_interior, CL_interior = np.asarray(
            [[np.nan], [np.nan], [np.nan]]
        )
    return h_interior_array, dT_interior, CL_interior


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell
def _(fig_interior_optimum):
    fig_interior_optimum
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""Notice how $C_{L_P}$ (minimum power) $\gt$ $C_{L_E}$ (minimum drag) but $E_\mathrm{P} \lt E_{\mathrm{max}}$ ($E = C_L/C_D$) because the drag coefficient increases more rapidly than $C_L$, as $C_D \propto C_L^2$. Thus, the range of $W/\sigma^\beta$ for which it is possible to fly at minimum power is smaller ($\sqrt{3}/2\lt 1$) than the one for which it is possible to fly at minimum drag. You can check this by increasing the weight of the aircraft here and in [Minimum Drag (simplified Jet)](?file=MinDrag_Jet.py) and finding out at what altitude it is not possible to fly at the optimum anymore, make sure to compare the same aircraft at the same weight.""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### _Lift limited solutions (stall)_

    In this case: $C_L = C_{L_{\mathrm{max}}}$, $\delta_T \lt 1$, $\mu_1 \gt 0$, $\mu_2= 0$

    from stationarity condition (2): $\lambda_1 = 0$

    from stationarity condition (1):

    $$
    \mu_1 = - \left.\frac{\partial P}{\partial C_L} \right|_{C_{L_\mathrm{max}}} =- \sqrt{\frac{2W^3}{\rho S}}\left(-\frac{3}{2}C_{D_0}C_{L_\mathrm{max}}^{-5/2} + \frac{1}{2} K C_{L_\mathrm{max}}^{-1/2}\right) \gt 0
    $$ 

    This inequality is saying that the required power should decrease for an increase in $C_L$ starting from $C_{L_\mathrm{max}}$. In other words, $P$ should decrease for a decrease in speed from the stall speed. Equivalently, $P$ should increase for an increase in speed from the stall speed. This is not an ideal design, as the aircraft would be stalling at a velocity higher than the one for minimum power.
    """
    )
    return


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell
def _(fig_lift_limited):
    fig_lift_limited
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    As a matter of fact, by substitution from the stationarity constraint (1):

    $$ \frac{3}{2}C_{D_0}C_{L_\mathrm{max}}^{-5/2} + \frac{1}{2} K C_{L_\mathrm{max}}^{-1/2} \lt 0 
    $$

    $$
    \Rightarrow -3C_{D_0}+KC_{L_\mathrm{max}}^{2} \lt 0 \quad \Rightarrow \quad C_{L_\mathrm{max}} \lt \sqrt{\frac{3C_{D_0}}{K}} = C_{L_P}
    $$

    We thus find that $C_{L_\mathrm{max}}$ must be smaller than the lift coefficient for minimum power, as predicted in the discussion above.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
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
    """
    )
    return


@app.function
def maxlift_condition(
    W,
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

    if CLmax < CL_P and sigma_maxlift > min_sigma:
        h_maxlift = atmos.altitude(sigma_maxlift)

        idx_maxlift = np.abs(h_array - h_maxlift).argmin()

        h_maxlift_array = h_array[:idx_maxlift]

        CL_maxlift = CLmax
    else:
        h_maxlift_array, CL_maxlift = np.asarray([[np.nan], [np.nan]])
    return h_maxlift_array, dT_maxlift, CL_maxlift


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell
def _(fig_maxlift_optimum):
    fig_maxlift_optimum
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### _Thrust-limited optimum_


    In this case: $C_L \lt C_{L_{\mathrm{max}}}$, $\delta_T = 1$, $\mu_1= 0$, $\mu_2 > 0$

    from stationarity condition (2): $\mu_2= -\lambda_1 T_{a0}\sigma^ \beta \gt 0 \Rightarrow \lambda_1 \lt 0$

    from stationarity condition (1): 

    $$
    \frac{\partial P}{\partial C_L} = \lambda_1 \frac{\partial D}{\partial C_L}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    This tells us that the required power and drag change in opposite directions with respect to the change in $C_L$. If one decreaes, then the other one has to increase, given that $\lambda_1 \lt 0$.
    This can only happen in the range of $C_L$ between $C_{L_P}$ and $C_{L_E}$, since they represent the minimum power and maximum aerodynamic efficiency (alternatively minimum drag) respectively. 

    This is clearer in the performance diagram:
    """
    )
    return


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell
def _(fig_thrust_limited):
    fig_thrust_limited
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    This condition is given by:

    $$
    C_{L_E}\lt C_L^*\lt C_{L_P} \quad \Leftrightarrow \quad \boxed{\sqrt{\frac{C_{D_0}}{K}}\lt C_L^* \lt \sqrt{3} \sqrt{\frac{C_{D_0}}{K}}}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
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
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    For its existence, the square root must be zero or positive, thus: 

    $$
    1 - \left(\frac{W}{T_{a0}\sigma^\beta E_{\mathrm{max}}}\right)^2 \ge 0 \quad \Rightarrow \quad \frac{W}{\sigma^\beta}\le T_{a0}E_{\mathrm{max}}
    $$

    as already seen in multiple occasions.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""Try to find whether there is a combination of altitude and weight for which the solution of the quadratic equation with the "$+$" sign falls within the bounds of $C_{L_P}$ and $C_{L_E}$, denoted by the green area in the graph below. Be careful, this is not always possible and will define the flight envelope where minimum power can be achieved.""")
    return


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell
def _(fig_performance_CL_equation):
    fig_performance_CL_equation
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
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
    """
    )
    return


@app.function
def maxthrust_condition(
    W, K, E_max, E_P, h_array, idx_h_selected, Ta0, beta, min_sigma, h_selected
):
    max_sigma_maxthrust = (W / E_P / Ta0) ** (1 / beta)
    min_sigma_maxthrust = (W / E_max / Ta0) ** (1 / beta)

    if min_sigma_maxthrust > min_sigma:
        dT_maxthrust = 1
        max_h_maxthrust = atmos.altitude(min_sigma_maxthrust)
        min_h_maxthrust = atmos.altitude(max_sigma_maxthrust)

        idx_maxh_maxthrust = np.abs(h_array - max_h_maxthrust).argmin()
        idx_minh_maxthrust = np.abs(h_array - min_h_maxthrust).argmin()

        h_maxthrust_array = h_array[idx_minh_maxthrust:idx_maxh_maxthrust]

        A = Ta0 * atmos.rhoratio(h_maxthrust_array) ** beta / (2 * K * W)
        B = (W / (Ta0 * atmos.rhoratio(h_maxthrust_array) ** beta * E_max)) ** 2
        CL_maxthrust = A * (1 + np.sqrt(1 - B))

        idx_maxthrust = (
            idx_h_selected - idx_minh_maxthrust
            if idx_h_selected > idx_minh_maxthrust
            and idx_h_selected < idx_maxh_maxthrust
            else idx_maxh_maxthrust + 1
        )

    else:
        h_maxthrust_array, dT_maxthrust, CL_maxthrust, idx_maxthrust = (
            np.asarray([[np.nan], [np.nan], [np.nan], [np.nan]])
        )
    return (
        h_maxthrust_array,
        dT_maxthrust,
        CL_maxthrust,
        idx_maxthrust,
    )


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell
def _(fig_maxthrust_optimum):
    fig_maxthrust_optimum
    return


@app.cell
def _():
    mo.md(
        r"""
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
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    Now continuing with the primal feasibility condition (3):

    $$
    T_{a0}\sigma^\beta = W \frac{C_{D_0} + K C_{L_{\mathrm{max}}}^2}{C_{L_{\mathrm{max}}}} = W E_S \quad \Leftrightarrow \quad \frac{W}{\sigma^\beta} = T_{a0} E_S
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
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
    """
    )
    return


@app.function
def maxliftThrust_condition(W, Ta0, E_S, beta, CL_E, CL_P, CLmax):
    sigma_maxliftThrust = (W / Ta0 / E_S) ** (1 / beta)

    h_maxliftThrust_selected = atmos.altitude(sigma_maxliftThrust)
    if CLmax < CL_P and CLmax != CL_E:
        CL_maxliftThrust = CLmax
        dT_maxliftThrust = 1
    else:
        dT_maxliftThrust, CL_maxliftThrust = np.asarray([[np.nan], [np.nan]])
    return (
        h_maxliftThrust_selected,
        sigma_maxliftThrust,
        dT_maxliftThrust,
        CL_maxliftThrust,
    )


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell
def _(fig_maxliftThrust_optimum):
    fig_maxliftThrust_optimum
    return


@app.cell
def _():
    mo.md(r"""## Final flight envelope""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""Now after deriving all the optima for each condition we can summarize the flight envelopes in one graph, as shown below. Experiment with the weight of the aircrarft to understand how the theoretical ceiling for minimum power moves in the graph.""")
    return


@app.cell
def _():
    return


@app.cell
def _(mass_stack):
    mass_stack
    return


@app.cell
def _(fig_final_flightenvelope):
    fig_final_flightenvelope
    return


@app.cell
def _():
    mo.md(r"""## Summary""")
    return


@app.cell
def _():
    mo.md(
        r"""
    | Name | Condition | $C_L^*$ | $\delta_T^*$ |
    |:-|:-------|:-------:|:-----:|
    |Interior-optima    | $\displaystyle \quad  C_{L_\mathrm{max}} > C_{L_P} \quad \text{and} \quad \frac{W}{\sigma^\beta} < T_{a0} E_\mathrm{max}$ | $\sqrt{\frac{3C_{D_0}}{K}}$ | $\displaystyle \frac{W}{E_{\mathrm{P}}}\frac{1}{T_{a0}\sigma^\beta}$  |
    |Lift-limited    |  $\displaystyle C_{L_\mathrm{max}} \lt {C_{L_P}} \quad \text{and}\quad \frac{W}{\sigma^\beta} \lt T_{a0}E_S$ | $C_{L_\mathrm{max}}$ | $\displaystyle \frac{W}{T_{a0}\sigma^\beta} \frac{1}{E_S}$ |
    |Thrust-limited    | $\displaystyle\quad T_{a0} E_{\mathrm{P}} \lt \frac{W}{\sigma^\beta} \lt T_{a0} E_{\mathrm{max}}$ | $\displaystyle \frac{T_{a0}\sigma^\beta}{2KW}\left[1 +\sqrt{1- \left(\frac{W}{T_{a0}\sigma^\beta E_{\mathrm{max}}}\right)^2}\right]$ | $1$ |
    |Thrust-lift limited    |  $\displaystyle {C_{L_\mathrm{max}} < C_{L_P},C_{L_\mathrm{max}} \neq C_{L_E}}, \quad \text{and} \quad \frac{W}{\sigma^\beta} = T_{a0} E_S$ | $C_{L_\mathrm{max}}$ | $1$ |
    """
    ).center()
    return


@app.cell
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
