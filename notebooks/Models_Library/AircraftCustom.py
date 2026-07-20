# SPDX-FileCopyrightText: 2026 Carmine Varriale <C.varriale@tudelft.nl>
# SPDX-FileCopyrightText: 2026 Federico Angioni <F.angioni@student.tudelft.nl>
# SPDX-FileCopyrightText: 2026 Maarten van Hoven <M.B.vanHoven@tudelft.nl>
#
# SPDX-License-Identifier: Apache-2.0
import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path.cwd()))

    # Initialization code that runs before all other cells
    import marimo as mo
    from core import _defaults
    from core import atmos
    import pandas as pd
    from core.aircraft import Aircraft, drag_polar
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    from scipy.signal import savgol_filter
    from scipy.spatial import ConvexHull
    from scipy import interpolate
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
    data_dir = str(mo.notebook_location().parent.parent / "data" / ac_id)
    ac = Aircraft(data_dir, "", custom=True)

    ac_dict = ac.df_dictionary
    return ac_dict, meshgrid


@app.cell
def _():
    from scipy.spatial import Delaunay


    def inside_hull(hull, points_2d):
        """Check which 2D points are inside a convex hull."""
        delaunay = Delaunay(hull.points[hull.vertices])
        return delaunay.find_simplex(points_2d) >= 0
    return (inside_hull,)


@app.cell
def _(ac_dict):
    # Prepare drag data

    data_D = ac_dict["DvsM"]

    # Unit conversions
    data_D["h"] = data_D["FL"] * 100 * 0.3048  # feet → meters
    data_D["W"] = data_D["W"] * 0.45359237 * 9.8067  # lbm → N
    data_D["D_mN"] = data_D["D"] * 1000  # convert to mN

    BP_M = np.sort(data_D["M"].dropna().unique())
    BP_h = np.sort(data_D["h"].dropna().unique())
    BP_W = np.sort(data_D["W"].dropna().unique())


    pivot_3d = data_D.pivot_table(index=["M", "h"], columns="W", values="D_mN")

    # Reindex to ensure a full (M, h, W) grid; missing values → NaN
    pivot_3d = pivot_3d.reindex(
        pd.MultiIndex.from_product([BP_M, BP_h], names=["M", "h"]), columns=BP_W
    )

    # Convert to 3D array: shape = (len(M), len(h), len(W))
    BP_D = pivot_3d.to_numpy().reshape(len(BP_M), len(BP_h), len(BP_W))

    # Compute convex hulls for each W (domain boundaries)
    CH_D = []
    for W_val in BP_W:
        valid = data_D[(data_D["W"] == W_val) & (~data_D["D"].isna())]
        if len(valid) > 3:
            points = np.column_stack((valid["M"], valid["h"]))
            CH_D.append(ConvexHull(points))
        else:
            CH_D.append(None)


    # Fill missing values by linear interpolation
    def fill_nan_axis(arr, axis):
        inds = np.arange(arr.shape[axis])

        def interp_func(x):
            mask = ~np.isnan(x)
            if mask.sum() < 2:
                return x
            return np.interp(inds, inds[mask], x[mask])

        return np.apply_along_axis(interp_func, axis, arr)


    BP_D = fill_nan_axis(BP_D, 0)  # interpolate along M
    BP_D = fill_nan_axis(BP_D, 1)  # interpolate along h

    # Smooth data with Savitzky–Golay filter
    win_len = min(9, BP_D.shape[0] - (BP_D.shape[0] + 1) % 2)
    Dsmooth = savgol_filter(
        BP_D, window_length=win_len, polyorder=2, axis=0, mode="interp"
    )

    # Create interpolant function
    F_D = interpolate.RegularGridInterpolator(
        (BP_M, BP_h, BP_W),
        Dsmooth,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,  # prevent extrapolation outside domain
    )
    return BP_M, BP_W, BP_h, CH_D, F_D


@app.cell
def _(BP_W, BP_h, ac_dict, meshgrid):
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

    M_KvsM = ac_dict["KvsM"].M.unique()
    CL_KvsM = ac_dict["KvsM"].CL.unique()

    CD0_CD0vsM = ac_dict["CD0vsM"].CD0.to_numpy()
    M_CD0vsM = ac_dict["CD0vsM"].M.to_numpy()

    M_range_KvsM = np.linspace(np.nanmin(M_KvsM), np.nanmax(M_KvsM), meshgrid)
    K_grid = ac_dict["KvsM"].pivot(index="CL", columns="M", values="K").values

    compute_K_from_CL_M = RegularGridInterpolator(
        (CL_KvsM, M_KvsM), K_grid, method="linear"
    )

    compute_CD0_from_M = RegularGridInterpolator(
        (M_CD0vsM,), CD0_CD0vsM, method="linear"
    )


    CL_range_KvsM = np.linspace(np.nanmin(CL_KvsM), np.nanmax(CL_KvsM), meshgrid)
    CD0_max = np.max(CD0_CD0vsM)
    # Create interpolator
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

    FL_slider = mo.ui.slider(
        start=min(BP_h),
        stop=max(BP_h),
        step=10,
        label="FL",
        show_value=True,
    )

    W_slider = mo.ui.slider(
        start=min(BP_W),
        stop=max(BP_W),
        step=500,
        label=r"$W$",
        show_value=True,
    )
    return (
        CD0_max,
        CL_range_KvsM,
        CL_slider,
        FL_slider,
        M_range_KvsM,
        M_slider,
        W_slider,
        compute_CD0_from_M,
        compute_K_from_CL_M,
        labels_KvsM,
        plot_list_KvsM,
    )


@app.cell
def _(
    BP_M,
    BP_W,
    CD0_max,
    CH_D,
    CL_range_KvsM,
    CL_slider,
    FL_slider,
    F_D,
    M_range_KvsM,
    M_slider,
    W_slider,
    compute_CD0_from_M,
    compute_K_from_CL_M,
    inside_hull,
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
    K_with_M_max = compute_K_from_CL_M(
        (np.max(CL_range_KvsM), np.max(M_range_KvsM))
    )

    CD = drag_polar(
        np.repeat(CD0_selected, len(K_funcCL_M_const)),
        K_funcCL_M_const,
        CL_range_KvsM,
    )

    M_points = BP_M
    h_points = np.full_like(BP_M, FL_slider.value)
    W_points = np.full_like(BP_M, W_slider.value)

    points_D = np.column_stack((M_points, h_points, W_points))

    # Pick W closest index
    W_idx = np.argmin(np.abs(BP_W - W_slider.value))
    hull = CH_D[W_idx]

    # Build 2D points (M, h) for hull check
    points_2d = np.column_stack((M_points, h_points))
    mask = inside_hull(hull, points_2d)

    # Initialize output with NaN
    D_interp = np.full(M_points.shape, np.nan)

    # Evaluate interpolator only inside the hull
    D_interp[mask] = F_D(points_D[mask])

    CD_max = drag_polar(CD0_max, K_with_M_max, np.max(CL_range_KvsM))
    return CD, CL_selected, D_interp, K_funcM_CL_const, M_selected


@app.cell
def _():
    mo.md(
        r"""
    # Custom Aircraft Models
    More complex aero-propulsive models allow CAPO at higher fidelity and greater level of detail, by capturing phenomena in the way that is specific and characteristic of the particular aircraft in analysis.

    In these cases, the models for $C_D$, $T_a$ or $P_a$, $c_T$ or $c_P$, and optionally $C_L$, are typically provided in the form of tabular data, as a function of several flight parameters. 

    Depending on the available data, custom models allow expanding the analysis to flight conditions in which the simplified models are not accurate, such as stall or transonic/supersonic effects.   

    On the other hand, they require that CAPO methodologies have to be tailored to the specific model structure, and therefore are hard to automate.
    """
    )
    return


@app.cell
def _():
    mo.md(r"""## Visualization""")
    return


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
    return


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
def _(CL_slider):
    CL_slider
    return


@app.cell
def _(fig_KvsM):
    fig_KvsM
    return


@app.cell
def _(CD, CL_range_KvsM, M_selected, ac_id):
    fig_CDvsCL = go.Figure()

    fig_CDvsCL.add_trace(
        go.Scatter(
            x=CL_range_KvsM,
            y=CD,
            name="𝑪<sub>𝑫₀</sub> for M = " + f"{M_selected}",
            showlegend=True,
        )
    )

    fig_CDvsCL.update_xaxes(
        title_text=r"$C_L \: 	\text{(-)}$",
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
    )

    fig_CDvsCL.update_yaxes(
        title_text=r"$C_{D_0} \: 	\text{(-)}$",
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        # range=[0, CD_max],
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

    mo.output.clear()
    return (fig_CDvsCL,)


@app.cell
def _(M_slider):
    M_slider
    return


@app.cell
def _(fig_CDvsCL):
    fig_CDvsCL
    return


@app.cell
def _(FL_slider, W_slider):
    mo.hstack([FL_slider, W_slider])
    return


@app.cell
def _(BP_M, D_interp, ac_id):
    fig_performance = go.Figure()

    fig_performance.add_trace(go.Scatter(x=BP_M, y=D_interp))

    fig_performance.update_yaxes(
        title_text=r"$F\: 	\text{(N)}$",
        gridcolor="#515151",
    )

    fig_performance.update_xaxes(
        title_text=r"$M \: 	\text{(-)}$",
        gridcolor="#515151",
        gridwidth=1,
        # range=[0, CD_max],
    )

    fig_performance.update_layout(
        title={
            "text": f"Performance diagram for {ac_id.replace('_', ' ')}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        },
    )


    fig_performance
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
