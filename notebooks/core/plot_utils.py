from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from core import atmos

LIGHTGREY = "rgb(112,128,144)"
SALMON = "rgb(232,158,184)"
GREEN = "rgb(0, 255, 0)"
DRAG_COLOR = "rgb(111,208,140)"
POWER_COLOR = "rgb(111,208,140)"
CONSTRAINT_CLR = "rgba(255, 0, 0, 0.35)"
AVAILABLE_COLOR = "rgb(230,199,156)"
CLMAX_AXES = "rgb(148,69,69)"

buffer_axes = 0.15

axes_min_speed = 0
axes_max_speed = atmos.a(0)
axes_min_dT = 0 - buffer_axes
axes_max_dT = 1 + buffer_axes
axes_min_h = 0
axes_max_h = 20


class ConfigTraces:
    def __init__(
        self,
        CLarray,
        dTarray,
        constraint,
        drag,
        thrust,
        power_required,
        power_available,
        power_surface,
        velocity_CLarray,
        velocity_CL_P,
        velocity_CL_E,
        velocity_stall_harray,
        velocity_stall,
        ranges,
        zcolorbar,
        mach_trace,
        stall_trace,
    ):
        self.CLaxes_drag = create_CL_axes(
            velocity_stall, 0.1 * ranges[0], "x1", "y1", LIGHTGREY
        )
        self.CLaxes_power = create_CL_axes(
            velocity_stall, 0.1 * ranges[1], "x2", "y2", LIGHTGREY
        )

        self.drag_trace = create_scatter_trace(
            velocity_CLarray, drag, "D", DRAG_COLOR, "x1", "y1", legend=False
        )

        self.power_trace = create_scatter_trace(
            velocity_CLarray, power_required, "P", POWER_COLOR, "x2", "y2", legend=False
        )

        self.power_heatmap = create_heatmap_trace(
            (CLarray, dTarray), power_surface, "Power (kW)", zcolorbar
        )

        self.constraint_trace = create_scatter_trace(
            CLarray, constraint, "constraint", CONSTRAINT_CLR, "x3", "y3", 10, False
        )

        self.CLP_trace_drag = go.Scattergl(
            x=[velocity_CL_P, velocity_CL_P],
            y=[0, 2 * ranges[0]],
            xaxis="x1",
            yaxis="y1",
            mode="lines",
            showlegend=False,
            line=dict(dash="dot", color=LIGHTGREY),
        )

        self.CLP_trace_power = go.Scattergl(
            x=[velocity_CL_P, velocity_CL_P],
            y=[0, 2 * ranges[1]],
            xaxis="x2",
            yaxis="y2",
            mode="lines",
            showlegend=False,
            line=dict(dash="dot", color=LIGHTGREY),
        )

        self.CLE_trace_drag = go.Scattergl(
            x=[velocity_CL_E, velocity_CL_E],
            y=[0, 2 * ranges[0]],
            xaxis="x1",
            yaxis="y1",
            mode="lines",
            showlegend=False,
            line=dict(dash="dot", color=LIGHTGREY),
        )

        self.CLE_trace_power = go.Scattergl(
            x=[velocity_CL_E, velocity_CL_E],
            y=[0, 2 * ranges[1]],
            xaxis="x2",
            yaxis="y2",
            mode="lines",
            showlegend=False,
            line=dict(dash="dot", color=LIGHTGREY),
        )

        self.mach_trace = mach_trace
        self.velocity_CLarray = velocity_CLarray
        self.stall_trace = stall_trace
        self.velocity_stall = velocity_stall
        self.velocity_CL_P = velocity_CL_P
        self.velocity_CL_E = velocity_CL_E
        self.power_available = power_available
        self.thrust = thrust


def create_CL_axes(V_stall, y_pos, plot_on_x, plot_on_y, color=LIGHTGREY):
    output = [
        go.Scattergl(
            x=[V_stall - 20, axes_max_speed * 2],
            y=[y_pos, y_pos],
            mode="lines",
            xaxis=plot_on_x,
            yaxis=plot_on_y,
            line=dict(color=color, width=1),
            showlegend=False,
        ),
        go.Scattergl(
            x=[V_stall - 20],
            y=[y_pos],
            mode="markers",
            xaxis=plot_on_x,
            yaxis=plot_on_y,
            marker=dict(color=color, size=10, symbol="arrow-left"),
            showlegend=False,
        ),
    ]

    return output


def create_scatter_trace(
    x_axes, y_axes, label, color, plot_on_x, plot_on_y, width=2, legend=True
):
    output = go.Scattergl(
        x=x_axes,
        y=y_axes,
        name=label,
        line=dict(color=color, width=width),
        mode="lines",
        xaxis=plot_on_x,
        yaxis=plot_on_y,
        showlegend=legend,
    )

    return output


def create_marker_trace(
    x_axes, y_axes, label, color, plot_on_x, plot_on_y, width=2, legend=False
):
    output = go.Scattergl(
        x=[x_axes],
        y=[y_axes],
        mode="markers",
        showlegend=legend,
        marker=dict(
            size=10,
            color=color,
            symbol="circle",
        ),
        name=label,
        xaxis=plot_on_x,
        yaxis=plot_on_y,
    )
    return output


def create_heatmap_trace(axes, function, label_name, zcolor):
    output = go.Heatmap(
        y=axes[1],
        x=axes[0],
        z=function,
        zsmooth="fast",
        opacity=0.9,
        name=label_name.split(maxsplit=1)[0],
        colorscale="viridis",
        colorbar={"title": label_name},
        xaxis="x3",
        yaxis="y3",
        zmin=zcolor[0],
        zmax=zcolor[1],
    )
    return output


def create_stall_trace(h, V, plot_on_x="x4", plot_on_y="y4"):
    output = [
        go.Scatter(
            x=V,
            y=h / 1e3,
            mode="lines",
            line=dict(width=1, color=LIGHTGREY, dash="dash"),
            name="V<sub>stall</sub>",
            showlegend=False,
            xaxis=plot_on_x,
            yaxis=plot_on_y,
        ),
        go.Scatter(
            x=[V[-8]],
            y=[h[-8] / 1e3],
            mode="markers+text",
            marker=dict(size=1, color=LIGHTGREY),
            text=["V<sub>stall</sub>"],
            hoverinfo="skip",
            textposition="top left",
            showlegend=False,
            xaxis=plot_on_x,
            yaxis=plot_on_y,
        ),
    ]

    return output


def create_mach_trace(h, a, plot_on_x="x4", plot_on_y="y4"):
    output = [
        go.Scatter(
            x=a,
            y=h / 1e3,
            mode="lines",
            line=dict(color=LIGHTGREY, width=2, dash="dash"),
            name="M1.0",
            showlegend=False,
            xaxis=plot_on_x,
            yaxis=plot_on_y,
        ),
        go.Scatter(
            x=[a[-8] - 5],
            y=[h[-8] / 1e3],
            mode="markers+text",
            marker=dict(size=1, color=LIGHTGREY),
            text=["M1.0"],
            hoverinfo="skip",
            textposition="top left",
            showlegend=False,
            xaxis=plot_on_x,
            yaxis=plot_on_y,
        ),
    ]

    return output


def create_title(figure, title):
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


class OptimumGridView:
    def __init__(
        self, configTraces, h_selected, velocity, power, optimum, title, equality=False
    ):
        """
        args:
            - axes_ranges: list
                i = 0: Mach 1.0 at SL
        """

        self.figure = make_subplots(
            rows=2,
            cols=2,
            horizontal_spacing=0.1,
            vertical_spacing=0.15,
        )

        self.figure.add_traces(configTraces.CLaxes_drag)
        self.figure.add_traces(configTraces.CLaxes_power)
        self.figure.add_traces(configTraces.power_trace)
        self.figure.add_traces(configTraces.drag_trace)
        self.figure.add_traces(configTraces.constraint_trace)
        self.figure.add_traces(configTraces.power_heatmap)
        self.figure.add_traces(configTraces.stall_trace)
        self.figure.add_traces(configTraces.mach_trace)
        # self.figure.add_traces(configTraces.CLP_trace_drag)
        # self.figure.add_traces(configTraces.CLP_trace_power)
        # self.figure.add_traces(configTraces.CLE_trace_drag)
        # self.figure.add_traces(configTraces.CLE_trace_power)

        self.add_vertical_trace(
            configTraces.velocity_stall, r"$C_{L_\mathrm{max}}$", color=CLMAX_AXES
        )
        self.add_vertical_trace(configTraces.velocity_CL_P, r"$C_{L_\mathrm{P}}$")
        self.add_vertical_trace(configTraces.velocity_CL_E, r"$C_{L_\mathrm{E}}$")

        if not equality:
            self.figure.add_traces(
                create_scatter_trace(
                    velocity[0], optimum[0] / 1e3, "V", SALMON, "x4", "y4", 3, False
                )
            )
        power_available = create_scatter_trace(
            configTraces.velocity_CLarray,
            optimum[1] * configTraces.power_available,
            "P",
            AVAILABLE_COLOR,
            "x2",
            "y2",
            legend=False,
        )

        thrust_available = create_scatter_trace(
            configTraces.velocity_CLarray,
            optimum[1] * configTraces.thrust,
            "T",
            AVAILABLE_COLOR,
            "x1",
            "y1",
            legend=False,
        )

        self.figure.add_traces(
            (
                thrust_available,
                power_available,
            )
        )

        if ~np.isnan(optimum[3]):  # and not equality:
            velocity_marker = create_marker_trace(
                velocity[1], h_selected / 1e3, "V", "#FFFFFF", "x4", "y4"
            )

            surface_marker = create_marker_trace(
                optimum[2] * optimum[3],
                optimum[1],
                "optimum",
                "#FFFFFF",
                "x3",
                "y3",
            )

            marker_power = create_marker_trace(
                velocity[1],
                power[1] / 1e3,
                "optimum",
                "#FFFFFF",
                "x2",
                "y2",
            )

            marker_drag = create_marker_trace(
                velocity[1],
                power[1] / velocity[1],
                "optimum",
                "#FFFFFF",
                "x1",
                "y1",
            )
            self.figure.add_traces(
                (
                    velocity_marker,
                    surface_marker,
                    marker_power,
                    marker_drag,
                )
            )

        self.add_title(title)

        self.figure.update_layout(
            xaxis1=dict(
                title=r"$V \; (\text{m/s})$",
                side="bottom",
                range=[axes_min_speed, axes_max_speed],
            ),
            yaxis1=dict(
                title=r"$D \: (\text{N})$",
                side="left",
            ),
            xaxis2=dict(
                title=r"$V \; (\text{m/s})$",
                side="bottom",
                range=[axes_min_speed, axes_max_speed],
                automargin=False,
            ),
            yaxis2=dict(
                title=r"$P \: (\text{kW})$",
                side="left",
            ),
            xaxis3=dict(
                title=r"$C_L\:(\text{-})$",
                showgrid=True,
                gridcolor="#515151",
                gridwidth=1,
            ),
            yaxis3=dict(
                title=r"$\delta_T \:(\text{-})$",
                range=[axes_min_dT, axes_max_dT],
                showgrid=True,
                gridcolor="#515151",
                gridwidth=1,
            ),
            xaxis4=dict(
                title=r"$V \: \text{(m/s)}$",
                range=[axes_min_speed, axes_max_speed],
                showgrid=True,
                gridcolor="#515151",
                gridwidth=1,
            ),
            yaxis4=dict(
                title=r"$h \: 	\text{(km)}$",
                range=[axes_min_h, axes_max_h],
                showgrid=True,
                gridcolor="#515151",
                gridwidth=1,
            ),
            legend=dict(x=0.02, y=0.98),
            height=800,
        )

    def add_title(self, title):
        create_title(self.figure, title)

    def add_vertical_trace(self, velocity, label, color=LIGHTGREY, pos="bottom left"):
        self.figure.add_vline(
            x=velocity,
            annotation_text=label,
            row=1,
            line_width=1,
            line_dash="dash",
            line_color=color,
            annotation_position=pos,
        )

    def update_axes_ranges(self, variable_ranges):
        if len(variable_ranges) > 3:
            axes_max_speed = variable_ranges[3]
        else:
            axes_max_speed = atmos.a(0)

        self.figure.update_layout(
            yaxis1=dict(range=[-buffer_axes, variable_ranges[0] + buffer_axes]),
            yaxis2=dict(range=[-buffer_axes, variable_ranges[1] + buffer_axes]),
            xaxis3=dict(range=[-buffer_axes, variable_ranges[2] + buffer_axes]),
            xaxis1=dict(range=[-buffer_axes, axes_max_speed]),
            xaxis2=dict(range=[-buffer_axes, axes_max_speed]),
        )


def create_final_flightenvelope(
    velocity_stall, mach, h_array, interior, maxthrust, maxlift, maxliftThrust
):
    """Creates the final flight envelope for each notebook
    Format for each entry:

    [0] h_array / h_value
    [1] velocity_array / value
    [2] inequality? bool True/ False

    """
    figure = go.Figure()
    traces = create_stall_trace(h_array, velocity_stall, "x1", "y1")

    """write this in a loop!"""
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
                go.Scatter(
                    x=[interior[1]],
                    y=[interior[0] / 1e3],
                    mode="markers",
                    marker=dict(size=10, color=SALMON),
                    showlegend=False,
                )
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
                    marker=dict(size=10, color=SALMON),
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
                    marker=dict(size=10, color=SALMON),
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
                    marker=dict(size=10, color=SALMON),
                    showlegend=False,
                )
            )
        else:
            traces.append(
                go.Scatter(
                    x=[maxliftThrust[1]],
                    y=[maxliftThrust[0] / 1e3],
                    mode="markers",
                    marker=dict(size=10, color=SALMON),
                    showlegend=False,
                )
            )

    figure.add_traces(traces)
    figure.add_traces(create_mach_trace(h_array, mach, "x1", "y1"))

    figure.update_layout(
        xaxis=dict(
            title=r"$V \: \text{(m/s)}$",
            range=[-0.15, axes_max_speed],
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
