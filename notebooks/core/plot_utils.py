from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from core import atmos

LIGHTGREY = "rgb(112,128,144)"
SALMON = "rgb(232,158,184)"


def add_trace(figure, x_axis, y_axis, trace_on_x, trace_on_y, name):
    figure.add_trace(
        go.Scatter(
            x=x_axis,
            y=y_axis,
            mode="lines",
            name=name,
            xaxis=trace_on_x,
            yaxis=trace_on_y,
        ),
    )


def add_title(figure, title):
    figure.update_layout(
        title={
            "text": title,
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
            "y": 0.99,
        }
    )


def create_optima_grid_stencil(
    velocities,
    domain_traces,
    flight_env_traces,
    power_required,
    drag,
    ranges,
    CLticks,
):
    """Creates the default traces for the performance diagrams at the end of each optima case"""

    figure = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.15,
    )

    # Traces for subplot (1, 1)
    traces_sub11 = [
        go.Scatter(
            x=velocities, y=drag, name="D", line=dict(color="blue"), mode="lines"
        ),
    ]

    traces_sub21 = [
        go.Scatter(
            x=velocities,
            y=power_required,
            line=dict(color="blue"),
            name="P",
            mode="lines",
        ),
    ]

    # Linearly space the CL ticks, this might have to be changed to have nice ticks in a future revision
    dummy_linspace = np.linspace(CLticks[-1], CLticks[0], 8)

    # Dummy plot for CL axis overlay, must be plotted on .add_trace alone without specifying cols and rows
    figure.add_trace(
        go.Scatter(
            x=CLticks,
            y=CLticks * 10,
            showlegend=False,
            line=dict(color="rgba(0, 0, 0, 0.0)"),
            hoverinfo="none",
            xaxis="x5",
            yaxis="y1",
        ),
    )

    figure.add_trace(
        go.Scatter(
            x=CLticks,
            y=CLticks * 10,
            showlegend=False,
            line=dict(color="rgba(0, 0, 0, 0.0)"),
            hoverinfo="none",
            xaxis="x6",
            yaxis="y2",
        ),
    )

    # Add traces to subplots
    figure.add_traces(traces_sub11, cols=1, rows=1)
    figure.add_traces(traces_sub21, cols=2, rows=1)
    figure.add_traces(domain_traces, cols=1, rows=2)
    figure.add_traces(flight_env_traces, cols=2, rows=2)

    # Layout configuration for subplot (1,1)
    figure.update_layout(
        dragmode=False,
        xaxis2=dict(
            title=r"$V \; (\text{m/s})$",
            side="bottom",
            range=[ranges[0], ranges[1]],
            automargin=False,
        ),
        xaxis6=dict(
            title=r"$C_L \; (\:\text{-}\:)$",
            side="top",
            range=[CLticks[-1], CLticks[0]],
            tickvals=dummy_linspace,
            ticktext=[f"{cl:.2f}" for cl in CLticks],
            overlaying="x2",
        ),
        yaxis2=dict(
            title=r"$P \: (\text{kW})$",
            side="left",
            range=[ranges[2], ranges[3]],
        ),
        xaxis1=dict(
            title=r"$V \; (\text{m/s})$",
            side="bottom",
            range=[ranges[0], ranges[1]],
        ),
        xaxis5=dict(
            title=r"$C_L \; (\:\text{-}\:)$",
            side="top",
            range=[CLticks[-1], CLticks[0]],
            tickvals=dummy_linspace,
            ticktext=[f"{cl:.2f}" for cl in CLticks],
            overlaying="x1",
        ),
        yaxis1=dict(
            title=r"$D \: (\text{N})$",
            side="left",
            range=[ranges[4], ranges[5]],
        ),
        xaxis3=dict(
            title=r"$C_L\:(\text{-})$",
            range=[ranges[6], ranges[7]],
            showgrid=True,
            gridcolor="#515151",
            gridwidth=1,
        ),
        yaxis3=dict(
            title=r"$\delta_T \:(\text{-})$",
            range=[ranges[8], ranges[9]],
            showgrid=True,
            gridcolor="#515151",
            gridwidth=1,
        ),
        xaxis4=dict(
            title=r"$V \: \text{(m/s)}$",
            range=[ranges[10], ranges[11]],
            showgrid=True,
            gridcolor="#515151",
            gridwidth=1,
        ),
        yaxis4=dict(
            title=r"$h \: 	\text{(km)}$",
            range=[ranges[12], ranges[13]],
            showgrid=True,
            gridcolor="#515151",
            gridwidth=1,
        ),
        legend=dict(x=0.02, y=0.98),
        height=800,
    )

    return figure


def create_overlayed_perf_diagram_stencil(CLticks, ranges):
    figure = go.Figure()

    # Linearly space the CL ticks, this might have to be changed to have nice ticks in a future revision
    dummy_linspace = np.linspace(CLticks[-1], CLticks[0], 8)

    # Dummy plot for CL axis overlay, must be plotted on .add_trace alone without specifying cols and rows
    figure.add_trace(
        go.Scatter(
            x=CLticks,
            y=CLticks * 10,
            showlegend=False,
            line=dict(color="rgba(0, 0, 0, 0.0)"),
            hoverinfo="none",
            xaxis="x2",
            yaxis="y1",
        ),
    )

    # Layout configuration for subplot (1,1)
    figure.update_layout(
        dragmode=False,
        xaxis=dict(
            title=r"$V \; (\text{m/s})$",
            side="bottom",
            range=[ranges[0], ranges[1]],
            constrain="domain",
            automargin=False,
        ),
        yaxis=dict(
            title=r"$P \: (\text{kW})$",
            domain=[0, 1],
            anchor="x",
            overlaying=None,
            range=[ranges[2], ranges[3]],
        ),
        xaxis2=dict(
            title=r"$C_L \; (\:\text{-}\:)$",
            side="top",
            range=[CLticks[-1], CLticks[0]],
            tickvals=dummy_linspace,
            ticktext=[f"{cl:.2f}" for cl in CLticks],
            overlaying="x",
            anchor="y",
            constrain="domain",
            automargin=False,
        ),
        yaxis2=dict(
            title=r"$D \: (\text{N})$",
            side="right",
            autorange=True,
            overlaying="y",
            anchor="x",
            range=[ranges[4], ranges[5]],
        ),
        legend=dict(x=0.02, y=0.98),
    )

    return figure


def config_domain_traces(axes, surface, constraint, label, zcolorbar):
    label_pos = label[1]
    label_name = label[0]
    domain_traces = [
        go.Heatmap(
            x=axes[0],
            y=axes[1],
            z=surface,
            opacity=0.9,
            name=label_name.split(maxsplit=1)[0],
            colorscale="viridis",
            zsmooth="best",
            zmin=zcolorbar[0],
            zmax=zcolorbar[1],
            colorbar={"title": label_name},
        ),
        go.Scatter(
            x=axes[0],
            y=constraint,
            mode="lines",
            showlegend=False,
            line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
            name="g1 constraint",
        ),
        go.Scatter(
            x=[axes[0][label_pos]],
            y=[constraint[label_pos]],
            textposition="middle left",
            mode="markers+text",
            text=["g<sub>1</sub>"],
            marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
            showlegend=False,
            name="g1 constraint",
            textfont=dict(size=14),
        ),
    ]

    return domain_traces


def config_flight_env_traces(h_array, v_stall_harray, a_harray):
    flight_envelope_traces = [
        go.Scatter(
            x=v_stall_harray,
            y=h_array / 1e3,
            mode="lines",
            line=dict(width=1, color=LIGHTGREY, dash="dash"),
            name="V<sub>stall</sub>",
            showlegend=False,
        ),
        go.Scatter(
            x=[v_stall_harray[-8]],
            y=[h_array[-8] / 1e3],
            mode="markers+text",
            marker=dict(size=1, color=LIGHTGREY),
            text=["V<sub>stall</sub>"],
            hoverinfo="skip",
            textposition="top left",
            showlegend=False,
        ),
        go.Scatter(
            x=a_harray,
            y=h_array / 1e3,
            mode="lines",
            line=dict(color=LIGHTGREY, width=2, dash="dash"),
            name="M1.0",
            showlegend=False,
        ),
        go.Scatter(
            x=[a_harray[-8] - 5],
            y=[h_array[-8] / 1e3],
            mode="markers+text",
            marker=dict(size=1, color=LIGHTGREY),
            text=["M1.0"],
            hoverinfo="skip",
            textposition="top left",
            showlegend=False,
        ),
    ]

    return flight_envelope_traces


def draw_optima(
    figure,
    velocity_array,
    thrust,
    power_available,
    V_opt_array,
    V_selected,
    h_opt_array,
    h_selected,
    opt_domain_x,
    opt_domain_y,
    domain_value,
    power_value,
    yranges_perf,
    name,
    idx,
    equality=False,
):
    traces = []

    if not equality:
        traces.append(
            go.Scatter(
                x=V_opt_array,
                y=h_opt_array / 1e3,
                mode="lines",
                line=dict(width=3, color=SALMON),
                showlegend=False,
                name=name,
            ),
        )

    traces.append(
        go.Scatter(
            x=[V_selected],
            y=[h_selected / 1e3],
            mode="markers+text",
            marker=dict(size=10, color="#FFFFFF"),
            name=name,
            showlegend=False,
        ),
    )

    trace = go.Scatter(
        x=[opt_domain_x],
        y=[opt_domain_y],
        mode="markers",
        showlegend=False,
        marker=dict(
            size=10,
            color="#FFFFFF",
            symbol="circle",
        ),
        name=name,
        xaxis="x3",
        yaxis="y3",
        customdata=[domain_value],
    )

    power_traces = [
        go.Scatter(
            x=velocity_array,
            y=power_available,
            line=dict(color="red"),
            mode="lines",
            showlegend=True,
            name="P<sub>available</sub>",
            xaxis="x2",
            yaxis="y2",
        ),
        go.Scatter(
            x=[V_selected],
            y=[power_value],
            mode="markers",
            showlegend=False,
            name="T",
            marker=dict(
                size=10,
                color="#FFFFFF",
                symbol="circle",
            ),
        ),
    ]

    thrust_traces = [
        go.Scatter(
            x=velocity_array,
            y=thrust,
            line=dict(color="red"),
            mode="lines",
            showlegend=True,
            name="T",
        ),
        go.Scatter(
            x=[V_selected],
            y=[power_value / V_selected * 1e3],
            mode="markers",
            showlegend=False,
            name="T",
            marker=dict(
                size=10,
                color="#FFFFFF",
                symbol="circle",
            ),
        ),
    ]

    figure.add_traces(thrust_traces, cols=1, rows=1)
    figure.add_traces(power_traces, cols=2, rows=1)
    figure.add_traces(traces, cols=2, rows=2)
    figure.add_trace(trace)

    figure.update_layout(
        yaxis1=dict(
            range=[
                min(yranges_perf[0], min(thrust)) * 0.9,
                min(yranges_perf[1], thrust[idx]) * 1.1,
            ]
        ),
        yaxis2=dict(
            range=[
                min(yranges_perf[2], min(power_available)) * 0.9,
                min(yranges_perf[3], power_available[idx]) * 1.1,
            ]
        ),
    )


def create_final_flightenvelope(
    default_traces, interior, maxthrust, maxlift, maxliftThrust
):
    """Creates the final flight envelope for each notebook
    Format for each entry:

    [0] h_array / h_value
    [1] velocity_array / value
    [2] inequality? bool True/ False

    """
    figure = go.Figure()
    traces = default_traces
    if ~np.isnan(np.asarray(interior[1]).any()):
        if interior[2]:
            traces.append(
                go.Scatter(
                    x=interior[1],
                    y=interior[0] / 1e3,
                    mode="lines",
                    line=dict(width=3, color=SALMON),
                    showlegend=False,
                )
            )
        else:
            traces.append(
                go.Scatter(x=[interior[1]], y=[interior[0] / 1e3], mode="markers")
            )

    if ~np.isnan(np.asarray(maxthrust[1]).any()):
        if maxthrust[2]:
            traces.append(
                go.Scatter(
                    x=maxthrust[1],
                    y=maxthrust[0] / 1e3,
                    mode="lines",
                    line=dict(width=3, color=SALMON),
                    showlegend=False,
                )
            )
        else:
            traces.append(
                go.Scatter(
                    x=[maxthrust[1]],
                    y=[maxthrust[0] / 1e3],
                    mode="markers",
                    showlegend=False,
                )
            )

    if ~np.isnan(np.asarray(maxlift[1]).any()):
        if maxlift[2]:
            traces.append(
                go.Scatter(
                    x=maxlift[1],
                    y=maxlift[0] / 1e3,
                    mode="lines",
                    line=dict(width=3, color=SALMON),
                    showlegend=False,
                )
            )
        else:
            traces.append(
                go.Scatter(
                    x=[maxlift[1]],
                    y=[maxlift[0] / 1e3],
                    mode="markers",
                    showlegend=False,
                )
            )

    if ~np.isnan(np.asarray(maxliftThrust[1]).any()):
        if maxliftThrust[2]:
            traces.append(
                go.Scatter(
                    x=maxliftThrust[1],
                    y=maxliftThrust[0] / 1e3,
                    mode="lines",
                    line=dict(width=3, color=SALMON),
                    showlegend=False,
                )
            )
        else:
            traces.append(
                go.Scatter(
                    x=[maxliftThrust[1]],
                    y=[maxliftThrust[0] / 1e3],
                    mode="markers",
                    showlegend=False,
                )
            )

    figure.add_traces(traces)

    figure.update_layout(
        xaxis=dict(
            title=r"$V \: \text{(m/s)}$",
            range=[-0.15, atmos.a(0) + 15],
            showgrid=True,
            gridcolor="#515151",
            gridwidth=1,
        ),
        yaxis=dict(
            title=r"$h \: 	\text{(km)}$",
            range=[-0.15, 20],
            showgrid=True,
            gridcolor="#515151",
            gridwidth=1,
        ),
    )

    return figure
