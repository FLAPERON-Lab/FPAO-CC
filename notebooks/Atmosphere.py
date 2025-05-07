import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import _defaults as defs

    defs.set_plotly_template()
    mo.sidebar(
        defs.sidebar,
        width="300px",
        # footer=mo.md(""),
    )
    return defs, mo


@app.cell
def _(mo):
    mo.md(
        r"""
    # Atmospheric Model
    In all cases, the International Standard Atmosphere (ISA) model is used to calculate the air temperature, pressure and density at a given altitude.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Model Equations

    |<div style="width:100px">Parameter</div> | <div style="width:250px">$0 \le h \le 11\,\mathrm{km}$</div> | <div style="width:250px">$h \ge 11\,\mathrm{km}$</div> | 
       |:-|:----------|:----------|
       | Temperature | $\displaystyle \frac{\Tau(h)}{\Tau_0} = \Theta(h) = \left(1 + \frac{\lambda}{\Tau_0} h\right)$ | $\displaystyle \Tau = \Tau_{11}=\mathit{const}$ |
       | Pressure | $\displaystyle \frac{p(h)}{p_0} = \delta(h) = \left(1 + \frac{\lambda}{\Tau_0} h\right)^{-g/(\lambda R)}$ | $\displaystyle \frac{p(h)}{p_0} = \delta(h) = \frac{p_{11}}{p_0} e^{\,-g(h-h_{11})/(RT_{11})}$ |
       | Density | $\displaystyle \frac{\rho(h)}{\rho_0} = \sigma(h) = \left(1 + \frac{\lambda}{\Tau_0} h\right)^{-[g/(\lambda R)+1]}$ | $\displaystyle \frac{\rho(h)}{\rho_0} = \sigma(h) = \frac{\rho_{11}}{\rho_0} e^{\,-g(h-h_{11})/(RT_{11})}$ |
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Tabular data""")
    return


@app.cell
def _(mo):
    import numpy as np

    dh_slider = mo.ui.slider(
        steps=np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]),
        label=r"$\Delta h \ \text{(m)}$",
        value=100,
        show_value=True,
    )

    dh_slider
    return dh_slider, np


@app.cell
def _(dh_slider, np):
    import pandas as pd
    from core import atmos

    h = np.arange(
        0, atmos.hmax + dh_slider.value, dh_slider.value
    )  # Altitude in meters
    atmos_data = pd.DataFrame(
        {
            "h": h,
            "T": atmos.T(h),
            "p": atmos.p(h),
            "rho": atmos.rho(h),
            "a": atmos.a(h),
        }
    )

    atmos_data
    return atmos, atmos_data


@app.cell
def _(mo):
    mo.md(r"""## Visualization""")
    return


@app.cell
def _(atmos, atmos_data):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = (
        make_subplots(
            rows=1,
            cols=4,
            shared_yaxes=True,
            y_title="Altitude (m)",
            horizontal_spacing=0.025,
        )
        .add_trace(
            go.Scatter(
                x=[0, 120000],
                y=[atmos.h11, atmos.h11],
                mode="lines",
                line=dict(color="grey", width=0.5, dash="solid"),
                showlegend=False,
            ),
            row="all",
            col="all",
        )
        .add_trace(
            go.Scatter(
                x=atmos_data["T"],
                y=atmos_data["h"],
                mode="lines",
                name="Temperature (K)",
                line=dict(color="blue", width=3),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        .add_trace(
            go.Scatter(
                x=atmos_data["p"],
                y=atmos_data["h"],
                mode="lines",
                name="Pressure (Pa)",
                line=dict(color="red", width=3),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        .add_trace(
            go.Scatter(
                x=atmos_data["rho"],
                y=atmos_data["h"],
                mode="lines",
                name="Density (kg/m^3)",
                line=dict(color="green", width=3),
                showlegend=False,
            ),
            row=1,
            col=3,
        )
        .add_trace(
            go.Scatter(
                x=atmos_data["a"],
                y=atmos_data["h"],
                mode="lines",
                name="Speed of sound (m/s)",
                line=dict(color="purple", width=3),
                showlegend=False,
            ),
            row=1,
            col=4,
        )
    )

    fig.update_yaxes(
        range=[0, 22000],
        dtick=2000,
    ).update_xaxes(
        title="Temperature (K)",
        row=1,
        col=1,
        range=[210, 290],
        dtick=10,
    ).update_xaxes(
        title="Pressure (Pa)",
        row=1,
        col=2,
        range=[0, 102000],
        dtick=20000,
    ).update_xaxes(
        title="Density (kg/m^3)",
        row=1,
        col=3,
        range=[0, 1.225],
        dtick=0.2,
    ).update_xaxes(
        title="Speed of sound (m/s)",
        row=1,
        col=4,
        range=[290, 350],
        dtick=10,
    )
    return


@app.cell
def _(defs, mo):
    nav_foot = mo.nav_menu(
        {
            f"{defs._fileurl}Nomenclature.py": f"{mo.icon('lucide:arrow-big-left')} Nomenclature",
            f"{defs._fileurl}AircraftSimplified.py": f"Simplified Aircraft {mo.icon('lucide:arrow-big-right')}",
        }
    ).center()
    nav_foot
    return


if __name__ == "__main__":
    app.run()
