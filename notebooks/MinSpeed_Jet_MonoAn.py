import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import _defaults
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import numpy as np

    _defaults.set_plotly_template()


@app.cell
def _():
    _defaults.set_sidebar()
    return


@app.cell
def _():
    mo.md(
        r"""
    # Minimum airspeed: simplfied jet aircraft

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



    ## Intuitive analysis
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    ### Direct elimination of $c_1^\mathrm{eq}$

    We cam start by solving $c_1^\mathrm{eq} = 0$ for $V$, in order to express the objective function $V$ in terms of the control variables.
    We will have to do the same for the other equality constraint, and the inequality constraints expressed by the bounds, in order to keep the problem consistent.

    $$ c_1^\mathrm{eq} = 0 \quad \Rightarrow \quad V = \sqrt{\frac{2W}{\rho S C_L}} $$

    With this, we are limiting our attention to only the airspeeds that are intrinsically capable to guarantee vertical equilibrium thanks to the lift-generating capabilities of the aircraft.
    In other words, we are looking for the _stall speed_.
    """
    )
    return


@app.cell
def _():
    mo.callout(
        mo.md(
            r"""Find the minimum speed, among those that intrinsically guarantee vertical equilibrium, which is also able to guarantee horizontal equilibrium, by changing the lift coefficient and throttle within certain limits

    $$
    \begin{aligned}
        \min_{C_L, \delta_T} 
        & \quad V = \sqrt{\frac{2W}{\rho S C_L}} \\
        \text{subject to}
        & \quad c_2^\mathrm{eq} = \frac{T}{W} - \frac{1}{E}  =\frac{\delta_T T_{a0}\sigma^\beta}{W} - \frac{C_{D_0} + K C_L^2}{C_L} =0 \\
        \text{for } 
        & \quad C_L \in [0, C_{L_\mathrm{max}}] \\
        & \quad \delta_T \in [0, 1]
    \end{aligned}
    $$  
    """
        )
    ).style({"text-align": "center"})
    return


@app.cell
def _():
    mo.md(
        r"""
    $V$ has now been eliminated from the expression of the objective and the constraint equations.
    It is now possible to analyse their behaviour as a function of the control variables only.
    Notice how the constraint now expresses the fact that the thrust-to-weight ratio has to be equal to the inverse of the lift-to-drag ratio, at a given altitude and weight.
    This synthesises the two separate equilibrium constraints in an effective and concise way.

    - [ ] Same plots as before, this time showing $c_2^{\mathrm{eq}}$ as a curve in the domain. The 2D plot becomes a contour plot of V, where the constraint curve is also visible.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    from core import aircraft as ac

    data = ac.available_aircrafts().round(decimals=4)

    cols_4dec = [
        "CD0",
        "K",
        "beta",
        "CLmax_cl",
        "CLmax_to",
        "CLmax_ld",
        "cT",
        "cP",
        "MMO",
    ]

    data[cols_4dec] = data[cols_4dec].round(4)

    other_cols = data.columns.difference(cols_4dec)
    data[other_cols] = data[other_cols].round(1)

    ac_table = mo.ui.table(
        data=data,
        pagination=True,
        freeze_columns_left=["full_name"],
        show_column_summaries=False,
        selection="single",
        initial_selection=[0],
        page_size=4,
    )

    ac_table
    return ac, ac_table


@app.cell
def _(ac, ac_table, dT_slider):
    # Computation cell:
    aircraft_list = []

    if ac_table.value is not None and ac_table.value.any().any():
        aircraft_list = ac_table.value["ID"]

    fleet = {ID: ac.Aircraft(ac_ID=ID) for ID in aircraft_list}

    for index, (id, obj) in enumerate(fleet.items()):
        c2_eq = (dT_slider.value) * 1
    return


@app.cell(hide_code=True)
def _(CL_maxld, CL_slider, ac_table, dT_slider):
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "Scatter"}, {"type": "Surface"}]]
    )

    if ac_table.value is not None and ac_table.value.any().any():
        title_text = str(ac_table.value.full_name.values[0])
    else:
        title_text = ""

    fig.data = []

    fig.add_trace(
        go.Scatter(
            x=[float(CL_slider.value)],
            y=[float(dT_slider.value)],
            showlegend=False,
            marker_color="#EF553B",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=[float(CL_slider.value)],
            y=[float(dT_slider.value)],
            z=[0],
            mode="markers",
            marker=dict(size=8, opacity=0.8, color="#EF553B"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        xaxis=dict(title=r"C<sub>L</sub> (-)"),
        yaxis=dict(title=r"δ<sub>T</sub> (-)"),
        scene1=dict(
            xaxis=dict(title=r"C<sub>L</sub> (-)"),
            yaxis=dict(title=r"δ<sub>T</sub> (-)"),
            zaxis=dict(title=r"V (m/s)"),
        ),
    )
    fig.update_xaxes(range=[-0.5, CL_maxld], row=1, col=1)
    fig.update_yaxes(range=[-0.25, 1], row=1, col=1)
    fig.update_layout(
        scene1=dict(
            xaxis=dict(range=[-0.5, CL_maxld]),
            yaxis=dict(range=[-0.25, 1]),
        )
    )
    fig.update_layout(
        title_text=title_text,
        title_x=0.5,
    )
    mo.output.clear()
    return (fig,)


@app.cell(hide_code=True)
def _(ac_table):
    if ac_table.value is not None and ac_table.value.any().any():
        CL_maxld = float(ac_table.value.CLmax_ld.values[0])
    else:
        CL_maxld = 3

    CL_slider = mo.ui.slider(start=0, stop=CL_maxld, step=0.1, label=r"$C_L$")

    dT_slider = mo.ui.slider(start=0, stop=1, step=0.05, label=r"$\delta_T$")
    return CL_maxld, CL_slider, dT_slider


@app.cell
def _(CL_slider, ac_table, dT_slider, fig):
    if ac_table.value is not None and ac_table.value.any().any():
        output = mo.vstack([fig, mo.hstack([CL_slider, dT_slider])])
    else:
        output = fig

    output
    return


@app.cell
def _():
    mo.md(
        r"""
    ### Monotonicity analysis

    $V$ does not depend on $\delta_T$ and is monotonically decreasing with $C_L$. 
    Therefore, it is mininimum at the maximum allowable value of $C_L$.

    $$
    V_s = \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}
    $$
    """
    )
    return


@app.cell
def _():
    mo.callout(
        mo.md(
            r"""The stall speed $V_s= \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}$ is the minimum airspeed at which an aircraft can sustain its weight in Steady Level Flight. As for all airspeeds, it depends on the aircraft weight and altitude"""
        )
    ).style({"text-align": "center"})
    return


@app.cell
def _():
    mo.md(
        r"""
    But this optimum value of $C_L$ has to correspond to a specific value of $\delta_T$ which verifies the constraint $c_2^\mathrm{eq}$ for horizontal equilibrium.
    In other words, we have to make sure that it is possible to fly at the maximum lift coefficient and, if yes, under which conditions.

    Combining $c_2^\mathrm{eq}$ and the bounds for $\delta_T$ we obtain the following condition that has to be satisfied for $V_s$ to be a valid solution to the constrained optimization problem:

    $$
    0 \le 
    \delta_T 
    = \frac{W}{T_{a0}\sigma^\beta} \frac{C_{D_0} + K C^2_{L_\mathrm{max}}}{C_{L_\mathrm{max}}} 
    = \frac{W}{T_{a0}\sigma^\beta} \frac{1}{E_S} 
    \le 1
    $$

    The first inequality is always satisfied, so it is only relevant to analyse the second one.
    It can be rearranged to draw conclusions on the basis of flight and aircraft parameters.

    $$
    \frac{W}{\sigma^\beta} \le T_{a0} E_S
    $$

    This condition should be interpreted as follows:

    - if $W/\sigma^\beta$ is below the threshold value ($<)$ (low weight and/or low altitude), the minimum speed in Steady Level Flight is obtained at $C_L = C_{L_\mathrm{max}}$ and $\delta_T<1$, and is indeed a stall speed $V_s$;
    - if $W/\sigma^\beta$ is exactly equal to the threshold value ($=$), the minimum speed in Steady Level Flight is obtained at $C_L = C_{L_\mathrm{max}}$ and $\delta_T=1$, and is limited both by lift-generating capabilities and propulsive ones (it is still a stall speed); this condition can be used to calculate the highest altitude at which the minimum speed is still limited by aerodynamics, for a certain weight;
    - [ ] Plot in the flight envelope
    - if $W/\sigma^\beta$ is above the threshold value ($>$) (high weight and/or high altitude), this procedure does not allow to find a minimum speed; in other words, the minimum speed is not a stall speed, and we need to look for solutions obtained for $C_L < C_{L_\mathrm{max}}$.
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    ### Direct elimination of $c_2^\mathrm{eq}$

    In this case we express the objective function $V$ and other constraints in terms of the control variables by solving $c_2^\mathrm{eq} = 0$ for $V$.

    $$ 
    c_2^\mathrm{eq} = 0 
    \quad \Rightarrow \quad 
    V = \sqrt{ \frac{2 \delta_T T_{a0}\sigma^\beta}{\rho S (C_{D_0}+K C_L^2)} } 
    =   V_s \sqrt{  \frac{\delta_T T_{a0}\sigma^\beta}{W} \frac{C_{L_\mathrm{max}}}{C_{D_0}+K C_L^2}}
    $$

    With this, we are limiting our attention to only the airspeeds that are intrinsically capable to guarantee horizontal equilibrium thanks to the thrust-generating capabilities of the aircraft.
    In other words, we are looking for the _power-limited speed_.
    """
    )
    return


@app.cell
def _():
    mo.callout(
        mo.md(
            r"""Find the minimum speed, among those that intrinsically guarantee horizontal equilibrium, which is also able to guarantee vertical equilibrium, by changing the lift coefficient and throttle within certain limits.

    $$
    \begin{aligned}
        \min_{C_L, \delta_T} 
        & \quad V = V_s \sqrt{  \frac{\delta_T T_{a0}\sigma^\beta}{W} \frac{C_{L_\mathrm{max}}}{C_{D_0}+K C_L^2}} \\
        \text{subject to}
        & \quad c_1^\mathrm{eq} = \frac{T}{W} - \frac{1}{E} = \frac{\delta_T T_{a0}\sigma^\beta}{W} - \frac{C_{D_0} + K C_L^2}{C_L} =0 \\
        \text{for } 
        & \quad C_L \in [0, C_{L_\mathrm{max}}] \\
        & \quad \delta_T \in [0, 1]
    \end{aligned}
    $$
    """
        )
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    Even if the constraint is exactly the same as in the previous direct elimination, the expression of the objective function is different, allowing us to make new considerations.

    - [ ] Same plots as before, this time showing c2eq as a curve in the domain. The 2D plot becomes a contour plot of V, where the constraint curve is also visible.
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    ### Monotonicity analysis

    The objective function is again a monotonically decreasing function of $C_L$ (leading to similar considerations as before) and now can be appreciated as a monotically increasing function of $\delta_T$.

    Its unconstrained minimum should therefore be sought for $\delta_T = 0$, but for this value the constraint would never be verified.

    Therefore, for every value of the $C_L$, the minimum value of the airspeed is obtained with the minimum _feasible value_ of the

    The question comes then spontaneous: what is the minimum value of $\delta_T$ that satisfies the constraint? 
    This is an unconstrained optimization sub-problem of a continuous bounded function, that can be easily solved.

    $$
    \begin{aligned}
        \min_{C_L} 
        & \quad \delta_T = \frac{W}{T_{a0}\sigma^\beta} \frac{C_{D_0}+K C_L^2}{C_L} \\
        \text{for } 
        & \quad C_L \in [0, C_{L_\mathrm{max}}]
    \end{aligned}
    $$
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    The objective function is not monotonic in this case, so let's start by checking the behaviour of the function at the boundaries.
    $\delta_T$ is undefined for $C_L=0$, and the analysis falls back on the previous one for $C_L=C_{L_\mathrm{max}}$.

    Therefore, the only interesting minimum is the interior one.
    Any interior stationary point is found by equating the gradient of the objecitive function to zero.
    It can be easily verified that this is indeed a minimum by looking at the convexity of the function.

    $$
    \delta_T^\dag =\frac{W \sqrt{4 K C_{D_0}}}{T_{a0}\sigma^\beta}  = \frac{W}{E_\mathrm{max} T_{a0}\sigma^\beta} 
    \quad \text{obtained for} \quad 
    C_L^\dag = \sqrt{\frac{C_{D_0}}{K}}=C_{L_E}
    $$

    $\delta_T^\dag$ is therefore the minimum throttle value that guarantees horizontal equilibrium. 
    It depends on altitude and weight, and can be achieved when flying at the lift coefficient for maximum aerodynamic efficiency.

    The airspeed corresponding to this minimum _feasible_ value of the throttle is obtained by substituting $\delta_T^\dag$ and $C_L^\dag$ in the expression of the original objective function.

    $$
    V^\dag = V_s \sqrt{\frac{C_{L_{\mathrm{max}}}}{C_{L_E}}} = \sqrt{\frac{2W}{\rho S C_{L_E}}}> V_s
    $$

    This tells us that such airspeed is not the globally minimum airspeed, and that minimimizing the throttle is not the most effecitve strategy to minimimize the airspeed.

    This is because a lower throttle requires a higher lift coefficient to remain in Steady Level Flight, and this in turn results in increased induced drag, which needs to be counteracted by thrust.
    In other words, a change in lift coefficient has a stronger impact on the airspeed than a change in throttle.
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    Because $V$ remains monotonically increasing with $\delta_T$ and minimizing the latter has proven to be not a successful strategy, the last analysis remaining to be done is the one for the maximum value of $\delta_T$.

    For $\delta_T=1$, the problem becomes

    $$
    \begin{aligned}
        \min_{C_L} 
        & \quad V = V_s \sqrt{  \frac{T_{a0}\sigma^\beta}{W} \frac{C_{L_\mathrm{max}}}{C_{D_0}+K C_L^2}} \\
        \text{subject to}
        & \quad c_1^\mathrm{eq} = \frac{T_a}{W}-\frac{1}{E} = \frac{ T_{a0}\sigma^\beta}{W} - \frac{C_{D_0} + K C_L^2}{C_L} =0 \\
        \text{for } 
        & \quad C_L \in [0, C_{L_\mathrm{max}}] \\
    \end{aligned}
    $$
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    The objective function is monotonically decreasing with $C_L$, and the case for $C_L=C_{L_\mathrm{max}}$ has already been analysed.

    The question now is: are there other values of $C_L<C_{L_\mathrm{max}}$ that verify the constraint and minimize airspeed at full throttle?

    Contrarily to the previous case, the expression of $c_1^\mathrm{eq}$ is implicit in $C_L$, and therefore it is not possible to rewrite the bounds for $C_L$ as inequality constraints for other parameters.
    Of the two possible values of $C_L$ that verify the constraint, the highest one is obtained using the quadratic formula:

    $$
    0 \le 
    C_L^\ddag 
    = \frac{T_{a0}\sigma^\beta}{2KW} \left[1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}\right]
    = \frac{C_{L_E}}{\delta_T^\dag} \left[1+\sqrt{1-\delta_T^{\dag 2}}\right]
    \le C_{L_\mathrm{max}}
    $$
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    It is hard to compare this expression to $C_{L_\mathrm{max}}$ explicitly, so a graphical representation is provided to discuss its feasibility for different aircraft.

    - [ ] Plot $C_L^\ddag$ as a function of $W/\sigma^\beta$ for an aircraft of choice

    At the same time, this value of $C_L^\ddag$ is even just achievable by the aircraft if the following condition is met:

    $$
    1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2 \ge 0 
    \quad \Leftrightarrow \quad
    \frac{W}{\sigma^\beta} \le T_{a0} E_\mathrm{max} 
    $$
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    This means that:

    - if $W/\sigma^\beta$ is above the threshold value ($>$) (high weight and/or high altitude), the aircraft is not able to fly at $\delta_T=1$ in Steady Level Flight.

    - if $W/\sigma^\beta$ is exactly equal to the threshold value ($=$), the aircraft is able to fly in Steady Level Flight at $\delta_T=1$, and $C_L=C_L^\dag=C_L^\ddag=C_{L_E}$. This eqaution can be used to calculate the corresponding limit altitude at which this condition occurs, for a given weight. This is called the _theoretcal ceiling_. The corresponding airspeed is again
       $$V^\dag=V_s\sqrt{\frac{C_{L_\mathrm{max}}}{C_{L_E}}}$$

    - if $W/\sigma^\beta$ is below the threshold value ($<)$ (low weight and/or low altitude), the aircraft is able to fly at $\delta_T=1$ and $C_L=C_L^\ddag$, but the corresponding airspeed may not be lower than the minimum speed obtained in other cases; its expression is:

    $$
    V^\ddag = V_s \sqrt{\frac{2KWC_{L_\mathrm{max}}}{T_{a0}\sigma^\beta \left[1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}\right]}}
    $$

    - [ ] Plot this expression in the flight envelope ($h$ vs $V$), together with all the other expressions of speeds obtained before. Superimpose a line that covers the minimum speed at each altitude.
    """
    )
    return


@app.cell
def _():
    _defaults.nav_footer(
        before_file="AerodynamicEfficiency.py",
        before_title="Aerodynamic Efficiency",
        above_file="MinSpeed.py",
        above_title="Minimum Speed",
    )
    return


if __name__ == "__main__":
    app.run()
