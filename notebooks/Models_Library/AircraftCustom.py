# SPDX-FileCopyrightText: 2026 Carmine Varriale <C.varriale@tudelft.nl>
# SPDX-FileCopyrightText: 2026 Federico Angioni <F.angioni@student.tudelft.nl>
# SPDX-FileCopyrightText: 2026 Maarten van Hoven <M.B.vanHoven@tudelft.nl>
#
# SPDX-License-Identifier: Apache-2.0

import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path.cwd()))

    # Initialization code that runs before all other cells
    import marimo as mo

    # Import dependencies
    from core import _defaults
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    import pandas as pd
    from core import atmos
    from core import aircraft as ac
    from core import plot_utils

    # from core.plot_utils import OptimumGridView
    from scipy.interpolate import RegularGridInterpolator
    from scipy.optimize import minimize

    # Set local/online filepath
    _defaults.FILEURL = _defaults.get_url()

    # Plotly dark mode template
    _defaults.set_plotly_template()

    # Data directory
    data_dir = str(mo.notebook_location().parent.parent / "data" / "AircraftDB_Standard.csv")

    # Source tables report altitude as flight level, everything else is SI
    FT_TO_M = 0.3048

    # Resolution of the CL sweep used to draw the constraint curve
    N_CURVE = 400


@app.cell
def _():
    _defaults.set_sidebar()
    return


@app.cell
def _():
    # Custom aircraft database: one row per aircraft, "folder" points at its tables
    data_root = mo.notebook_location().parent.parent / "data"
    ac_db = pd.read_csv(str(data_root / "AircraftDB_Custom.csv"))

    ac_options = {folder.replace("_", " "): folder for folder in ac_db["folder"]}

    ac_dropdown = mo.ui.dropdown(
        options=ac_options,
        value=next(iter(ac_options)),
        label="Aircraft",
    )
    return ac_db, ac_dropdown, data_root


@app.cell
def _(ac_db, ac_dropdown, data_root):
    # Load the selected aircraft's tabular data and scalar parameters
    ac_id = ac_dropdown.value
    aircraft = ac.Aircraft(str(data_root / ac_id), "", custom=True)

    params = ac_db[ac_db["folder"] == ac_id].iloc[0]

    ac_name = params["full_name"]

    # Wing area [m^2]
    S = params["S"].item()

    # Maximum lift coefficient [-]
    CLmax = params["CLmax"].item()

    # Mass sweep between OEM and MTOM of the selected aircraft [kg],
    # rounded inwards so the ends stay within the certified envelope
    m_min = params["OEM"].item()
    m_max = params["MTOM"].item()
    mass_step = 50  # kg


    m_slider = mo.ui.slider(
        start=float(np.ceil(m_min / mass_step) * mass_step),
        stop=float(np.floor(m_max / mass_step) * mass_step),
        step=mass_step,
        value=float(np.ceil(m_min / mass_step) * mass_step),
        label=r"$m$ (kg)",
        show_value=True,
    )

    # Altitude sweep bounded by the aircraft's own thrust table, so a new
    # aircraft gets its ceiling from its data.
    h_max = aircraft.df_dictionary["TvsM"]["FL"].max() * 100 * FT_TO_M
    h_step = 500

    h_slider = mo.ui.slider(
        start=0,
        stop=float(np.floor(h_max / h_step) * h_step),
        step=h_step,
        value=0,
        label=r"$h$ (m)",
        show_value=True,
    )

    # Thrust setting s, read off the TvsM table
    setting_dropdown = mo.ui.dropdown(
        options=list(aircraft.df_dictionary["TvsM"]["Setting"].unique()),
        value="Max",
        label=r"$s$",
    )
    return CLmax, ac_id, aircraft


@app.cell
def _():
    from scipy.spatial import Delaunay


    def inside_hull(hull, points_2d):
        """Check which 2D points are inside a convex hull."""
        delaunay = Delaunay(hull.points[hull.vertices])
        return delaunay.find_simplex(points_2d) >= 0
    return


@app.cell
def _(CLmax, aircraft):
    cd0_table = aircraft.df_dictionary["CD0vsM"]

    K_table = aircraft.df_dictionary["KvsM"].pivot(index="M", columns="CL", values="K")
    K_M = K_table.index.to_numpy(dtype=float)
    K_CL = K_table.columns.to_numpy(dtype=float)
    K_lookup = RegularGridInterpolator((K_M, K_CL), K_table.to_numpy(dtype=float))

    CL_slider = mo.ui.slider(
        start=np.nanmin(K_CL),
        stop=CLmax,
        step=0.025,
        label=r"$C_L$",
        show_value=True,
    )

    M_slider = mo.ui.slider(
        start=round(np.ceil(cd0_table["M"].min() / 0.025) * 0.025, 3),
        stop=round(np.floor(cd0_table["M"].max() / 0.025) * 0.025, 3),
        step=0.025,
        label=r"$M$",
        show_value=True,
    )

    CL_fine = np.linspace(0, CLmax, N_CURVE + 1)[1:]


    def CD0_interp(M):
        """Parasitic drag coefficient at Mach number M."""
        # np.interp holds the end values outside the table, where the digitised curve is flat
        return np.interp(M, cd0_table["M"], cd0_table["CD0"])


    n_mesh = plot_utils.meshgrid_n

    CL_array = np.linspace(0, CLmax, n_mesh + 1)[1:]
    dT_array = np.linspace(0, 1, n_mesh)


    CD_max = np.max(cd0_table["CD0"]) + np.max(K_table.to_numpy(dtype=float)) * CLmax**2
    return (
        CD0_interp,
        CD_max,
        CL_fine,
        CL_slider,
        K_CL,
        K_M,
        K_lookup,
        K_table,
        M_slider,
        cd0_table,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Custom Aircraft Models
    More complex aero-propulsive models allow CAPO at higher fidelity and greater level of detail, by capturing phenomena in the way that is specific and characteristic of the particular aircraft in analysis.

    In these cases, the models for $C_D$, $T_a$ or $P_a$, $c_T$ or $c_P$, and optionally $C_L$, are typically provided in the form of tabular data, as a function of several flight parameters.

    Depending on the available data, custom models allow expanding the analysis to flight conditions in which the simplified models are not accurate, such as stall or transonic/supersonic effects.

    On the other hand, they require that CAPO methodologies have to be tailored to the specific model structure, and therefore are hard to automate.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Visualization
    """)
    return


@app.cell
def _(ac_id, cd0_table):
    fig_CD0vsM = go.Figure()

    fig_CD0vsM.add_traces([go.Scatter(x=cd0_table["M"], y=cd0_table["CD0"])])

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
    return


@app.cell
def _(CL_slider):
    CL_slider.center()
    return


@app.cell
def _(CL_slider, K_CL, K_M, K_lookup, K_table, ac_id):
    fig_KvsM = go.Figure()

    fig_KvsM.add_trace(
        go.Scatter(
            x=K_M,
            y=K_lookup(np.column_stack([K_M, CL_slider.value * np.ones_like(K_M)])),
            name="𝐾 for 𝑪<sub>𝑳</sub> = " + f"{CL_slider.value}",
        ),
    )

    nM, nCL = K_table.to_numpy(dtype=float).shape

    # one row per CL, with a trailing NaN to cut the line
    x = np.full((nCL, nM + 1), np.nan)
    x[:, :nM] = K_M  # broadcasts along rows
    y = np.full((nCL, nM + 1), np.nan)
    y[:, :nM] = K_table.to_numpy(dtype=float).T

    fig_KvsM.add_trace(
        go.Scatter(
            x=x.ravel(),
            y=y.ravel(),
            mode="lines",
            line=dict(color="rgba(255,255,255,0.4)"),
            customdata=np.repeat(K_CL, nM + 1),  # so hover still tells you which CL
            hovertemplate="M=%{x:.3f}<br>K=%{y:.4f}<br>CL=%{customdata:.2f}<extra></extra>",
            showlegend=False,
        )
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
            "text": f"𝐾 for different values of 𝑪<sub>𝑳</sub> for {ac_id.replace('_', ' ')}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        },
    )

    first_M = K_M[0]
    first_K = K_table.to_numpy(dtype=float)[0, :]  # K at the first M, one per CL

    labels_KvsM = [
        dict(
            x=first_M + 0.2,
            y=k + 0.009,
            xanchor="right",
            yanchor="middle",
            text=rf"$C_L = {cl}$",
            font=dict(size=16),
            showarrow=False,
        )
        for cl, k in zip(K_CL[1:], first_K[1:])
    ]

    fig_KvsM.update_layout(annotations=labels_KvsM)
    return


@app.cell
def _(M_slider):
    M_slider
    return


@app.cell(hide_code=True)
def _(CD0_interp, CD_max, CL_fine, CLmax, K_M, K_lookup, M_slider, ac_id):
    fig_CDvsCL = go.Figure()

    fig_CDvsCL.add_trace(
        go.Scatter(
            x=CL_fine,
            # The K grid starts higher in Mach than the CD0 curve the slider is bounded
            # by, so M is clipped to it and K is held flat below the first digitised row
            y=CD0_interp(M_slider.value)
            + K_lookup(
                np.column_stack(
                    [np.full_like(CL_fine, np.clip(M_slider.value, K_M[0], K_M[-1])), CL_fine]
                )
            )
            * CL_fine**2,
            name="𝑪<sub>𝑫</sub> for M = " + f"{M_slider.value}",
            showlegend=True,
        )
    )

    fig_CDvsCL.update_xaxes(
        title_text=r"$C_L \: 	\text{(-)}$", showgrid=True, gridcolor="#515151", gridwidth=1, range=[0, CLmax]
    )

    fig_CDvsCL.update_yaxes(
        title_text=r"$C_{D} \: 	\text{(-)}$",
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        range=[0, CD_max],
    )

    fig_CDvsCL.update_layout(
        title={
            "text": f"𝑪<sub>𝑫</sub> versus 𝑪<sub>𝑳</sub> for {ac_id.replace('_', ' ')}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        },
    )

    fig_CDvsCL.update_legends()
    return


@app.cell
def _():
    mo.md(r"""
    Add notes on how clearly the M_DD is visible around 0.6-0.7.
    """)
    return


@app.cell
def _():
    _defaults.nav_footer(
        "Models_Library/AircraftSimplified.py",
        "Simplified Aircraft Models",
        "Problem_Formulation/PointPerformance.py",
        "Problem Formulation",
    )
    return


if __name__ == "__main__":
    app.run()
