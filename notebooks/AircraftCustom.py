import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    from core import _defaults
    from core.aircraft import Aircraft, drag_polar
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator

    _defaults.FILEURL = _defaults.get_url()

    # Set local/online filepath
    _defaults.FILEURL = _defaults.get_url()

    # Plotly dark mode template
    _defaults.set_plotly_template()


@app.cell
def _():
    _defaults.set_sidebar()
    return


@app.cell
def _():
    ac_id = "T38_Talon"
    return (ac_id,)


@app.cell
def _(ac_id):
    # Data directory
    meshgrid = 121
    data_dir = str(mo.notebook_location() / "public" / ac_id)
    return data_dir, meshgrid


@app.cell
def _():
    mo.md(
        r"""
    # Custom Aircraft Models
    More complex aero-propulsive models allow FPAO at higher fidelity and greater level of detail, by capturing phenomena in the way that is specific and characteristic of the particular aircraft in analysis.

    In these cases, the models for $C_D$, $T_a$ or $P_a$, $c_T$ or $c_P$, and optionally $C_L$, are typically provided in the form of tabular data, as a function of several flight parameters. 

    Depending on the available data, custom models allow expanding the analysis to flight conditions in which the simplified models are not accurate, such as stall or transonic/supersonic effects.   

    On the other hand, they require that FPAO methodologies have to be tailored to the specific model structure, and therefore are hard to automate.
    """
    )
    return


@app.cell
def _():
    mo.md(r"""## Visualization""")
    return


@app.cell
def _(data_dir):
    ac = Aircraft(data_dir, "", custom=True)

    ac_dict = ac.df_dictionary
    return (ac_dict,)


@app.cell
def _(ac_dict, ac_id):
    fig_CD0vsM = go.Figure()

    fig_CD0vsM.add_traces(
        [go.Scatter(x=ac_dict["CD0vsM"].M, y=ac_dict["CD0vsM"].CD0)]
    )

    fig_CD0vsM.update_xaxes(
        title_text=r"$M\:\text{(-)}$",
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
    )

    fig_CD0vsM.update_yaxes(
        title_text=r"$C_{D_0}\:\text{(-)}$",
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
    )

    fig_CD0vsM.update_layout(
        title={
            "text": f"𝑪<sub>𝑫₀</sub> for {ac_id.replace('_', ' ')}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        },
    )


    mo.output.clear()
    return (fig_CD0vsM,)


@app.cell
def _(fig_CD0vsM):
    fig_CD0vsM
    return


@app.cell
def _(ac_dict, meshgrid):
    # Static, runs when new data is loaded

    M_KvsM = ac_dict["KvsM"].M.unique()
    CL_KvsM = ac_dict["KvsM"].CL.unique()
    CD0_CD0vsM = ac_dict["CD0vsM"].CD0.to_numpy()
    M_CD0vsM = ac_dict["CD0vsM"].M.to_numpy()
    M_range_KvsM = np.linspace(np.nanmin(M_KvsM), np.nanmax(M_KvsM), meshgrid)
    K_grid = ac_dict["KvsM"].pivot(index="CL", columns="M", values="K").values
    CL_range_KvsM = np.linspace(np.nanmin(CL_KvsM), np.nanmax(CL_KvsM), meshgrid)

    # Create interpolator
    compute_K_from_CL_M = RegularGridInterpolator(
        (CL_KvsM, M_KvsM), K_grid, method="linear"
    )

    compute_CD0_from_M = RegularGridInterpolator(
        (M_CD0vsM,), CD0_CD0vsM, method="linear"
    )

    plot_list_KvsM = []
    labels_KvsM = []

    for i in range(1, len(CL_KvsM)):
        curve_KvsM = go.Scatter(
            x=ac_dict["KvsM"][ac_dict["KvsM"].CL == CL_KvsM[i]].M,
            y=ac_dict["KvsM"][ac_dict["KvsM"].CL == CL_KvsM[i]].K,
            line=dict(color="rgba(255, 255, 255, 0.4)"),
            showlegend=False,
        )
        labels_KvsM.append(
            dict(
                x=ac_dict["KvsM"][ac_dict["KvsM"].CL == CL_KvsM[i]].M.iloc[0] + 0.1,
                y=ac_dict["KvsM"][ac_dict["KvsM"].CL == CL_KvsM[i]].K.iloc[0]
                + 0.009,
                xanchor="right",
                yanchor="middle",
                text=rf"$C_L = {CL_KvsM[i]}$",
                font=dict(size=16),
                showarrow=False,
            )
        )

        plot_list_KvsM.append(curve_KvsM)
    return (
        CL_range_KvsM,
        M_range_KvsM,
        compute_CD0_from_M,
        compute_K_from_CL_M,
        labels_KvsM,
        plot_list_KvsM,
    )


@app.cell
def _(
    CL_range_KvsM,
    CL_slider,
    M_range_KvsM,
    M_slider,
    compute_CD0_from_M,
    compute_K_from_CL_M,
):
    # To call when sliders change
    CL_selected = CL_slider.value
    M_selected = float(M_slider.value)
    CD0_selected = compute_CD0_from_M([[M_selected]])
    CL_const_M_domain = np.column_stack(
        (np.full_like(M_range_KvsM, CL_selected), M_range_KvsM)
    )

    M_const_CL_domain = np.column_stack(
        (CL_range_KvsM, np.full_like(CL_range_KvsM, M_selected))
    )

    K_funcM_CL_const = compute_K_from_CL_M(CL_const_M_domain)
    K_funcCL_M_const = compute_K_from_CL_M(M_const_CL_domain)

    CD = drag_polar(
        np.repeat(CD0_selected, len(K_funcCL_M_const)),
        K_funcCL_M_const,
        CL_range_KvsM,
    )
    return CD, CL_selected, K_funcM_CL_const


@app.cell
def _(
    CL_selected,
    K_funcM_CL_const,
    M_range_KvsM,
    ac_id,
    labels_KvsM,
    plot_list_KvsM,
):
    fig_KvsM = go.Figure()

    fig_KvsM.add_trace(
        go.Scatter(
            x=M_range_KvsM,
            y=K_funcM_CL_const,
            name="𝐾 for 𝑪<sub>𝑳</sub> = " + f"{CL_selected}",
        ),
    )

    fig_KvsM.add_traces(plot_list_KvsM)

    fig_KvsM.update_xaxes(
        title_text=r"$M \: 	\text{(-)}$",
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
    )

    fig_KvsM.update_yaxes(
        title_text=r"$K \: 	\text{(-)}$",
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
    )

    fig_KvsM.update_layout(
        title={
            "text": f"𝐾 for different values of 𝑪<sub>𝑳</sub> for {ac_id.replace('_', ' ')}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        },
    )

    fig_KvsM.update_layout(annotations=labels_KvsM)

    mo.output.clear()
    return (fig_KvsM,)


@app.cell
def _(ac_dict):
    CL_slider = mo.ui.slider(
        start=np.nanmin(ac_dict["KvsM"].CL.unique()[1:]),
        stop=np.nanmax(ac_dict["KvsM"].CL),
        step=0.025,
        label=r"$C_L$",
        show_value=True,
    )

    M_slider = mo.ui.slider(
        start=np.nanmin(ac_dict["KvsM"].M.unique()),
        stop=np.nanmax(ac_dict["KvsM"].M),
        step=0.025,
        label=r"$M$",
        show_value=True,
    )
    return CL_slider, M_slider


@app.cell
def _(CL_slider):
    CL_slider
    return


@app.cell
def _(fig_KvsM):
    fig_KvsM
    return


@app.cell
def _(M_slider):
    M_slider
    return


@app.cell
def _(CD, CL_range_KvsM):
    fig_CDvsCL = go.Figure()

    fig_CDvsCL.add_trace(go.Scatter(x=CL_range_KvsM, y=CD))
    return


@app.cell
def _():
    _defaults.nav_footer(
        "AircraftSimplified.py",
        "Simplified Aircraft Models",
        "ProblemFormulation.py",
        "Problem Formulation",
    )
    return


if __name__ == "__main__":
    app.run()
