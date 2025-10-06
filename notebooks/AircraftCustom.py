import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    from core import _defaults
    from core.aircraft import Aircraft
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator

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
    meshgrid = 101
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

    fig_CD0vsM.update_xaxes()

    fig_CD0vsM.update_yaxes()

    fig_CD0vsM.update_layout(
        title={
            "text": f"<b>CD0</b> for {ac_id}",
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
    M_KvsM = ac_dict["KvsM"].M.unique()
    CL_KvsM = ac_dict["KvsM"].CL.unique()

    M_range_KvsM = np.linspace(np.nanmin(M_KvsM), np.nanmax(M_KvsM), meshgrid)
    K_grid = ac_dict["KvsM"].pivot(index="CL", columns="M", values="K").values

    # Create interpolator
    interpolator = RegularGridInterpolator((CL_KvsM, M_KvsM), K_grid)
    return CL_KvsM, M_range_KvsM, interpolator


@app.cell
def _(CL_slider_KvsM, M_range_KvsM):
    points = np.column_stack(
        (np.full_like(M_range_KvsM, CL_slider_KvsM.value), M_range_KvsM)
    )
    return (points,)


@app.cell
def _(CL_KvsM, M_range_KvsM, ac_dict, ac_id, interpolator, points):
    fig_KvsM = go.Figure()

    fig_KvsM.add_traces(
        [
            go.Scatter(
                x=ac_dict["KvsM"][ac_dict["KvsM"].CL == CL_KvsM[0]].M,
                y=ac_dict["KvsM"][ac_dict["KvsM"].CL == CL_KvsM[0]].K,
                line=dict(color="rgba(255, 255, 255, 0.5)"),
                showlegend=False,
            ),
            go.Scatter(
                x=ac_dict["KvsM"][ac_dict["KvsM"].CL == CL_KvsM[1]].M,
                y=ac_dict["KvsM"][ac_dict["KvsM"].CL == CL_KvsM[1]].K,
                line=dict(color="rgba(255, 255, 255, 0.5)"),
                showlegend=False,
            ),
            go.Scatter(
                x=ac_dict["KvsM"][ac_dict["KvsM"].CL == CL_KvsM[2]].M,
                y=ac_dict["KvsM"][ac_dict["KvsM"].CL == CL_KvsM[2]].K,
                line=dict(color="rgba(255, 255, 255, 0.5)"),
                showlegend=False,
            ),
            go.Scatter(
                x=ac_dict["KvsM"][ac_dict["KvsM"].CL == CL_KvsM[3]].M,
                y=ac_dict["KvsM"][ac_dict["KvsM"].CL == CL_KvsM[3]].K,
                line=dict(color="rgba(255, 255, 255, 0.5)"),
                showlegend=False,
            ),
            go.Scatter(
                x=ac_dict["KvsM"][ac_dict["KvsM"].CL == CL_KvsM[4]].M,
                y=ac_dict["KvsM"][ac_dict["KvsM"].CL == CL_KvsM[4]].K,
                line=dict(color="rgba(255, 255, 255, 0.5)"),
                showlegend=False,
            ),
            go.Scatter(
                x=ac_dict["KvsM"][ac_dict["KvsM"].CL == CL_KvsM[5]].M,
                y=ac_dict["KvsM"][ac_dict["KvsM"].CL == CL_KvsM[5]].K,
                line=dict(color="rgba(255, 255, 255, 0.5)"),
                showlegend=False,
            ),
            go.Scatter(x=M_range_KvsM, y=interpolator(points)),
        ]
    )

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
            "text": f"<b>K</b> for {ac_id}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        },
    )

    mo.output.clear()
    return (fig_KvsM,)


@app.cell
def _(ac_dict):
    CL_slider_KvsM = mo.ui.slider(
        start=np.nanmin(ac_dict["KvsM"].CL),
        stop=np.nanmax(ac_dict["KvsM"].CL),
        step=0.05,
    )
    return (CL_slider_KvsM,)


@app.cell
def _(CL_KvsM):
    CL_KvsM
    return


@app.cell
def _(CL_slider_KvsM):
    CL_slider_KvsM
    return


@app.cell
def _(fig_KvsM):
    fig_KvsM
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
