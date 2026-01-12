from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from core import atmos

import marimo as mo

LIGHTGREY = "rgb(112,128,144)"
SALMON = "rgb(232,158,184)"
GREEN = "rgb(0, 255, 0)"
DRAG_COLOR = "rgb(111,208,140)"
POWER_COLOR = "rgb(111,208,140)"
CONSTRAINT_CLR = "rgba(255, 0, 0, 0.35)"
AVAILABLE_COLOR = "rgb(235,199,156)"
CLMAX_AXES = "rgb(148,69,69)"
WHITE = "#FFFFFF"

buffer_axes = 0.15
meshgrid_n = 41
xy_lowerbound = -0.1
axes_min_speed = 0
axes_max_speed = atmos.a(0)
axes_min_dT = 0 - buffer_axes
axes_max_dT = 1 + buffer_axes
axes_min_h = 0
axes_max_h = 13


class OptimumGridViewNew:
    def __init__(self, Model, configTraces, Optimums, equality=False):
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

        for name in dir(configTraces):
            if not name.startswith("_"):  # skip private/internal attributes
                trace = getattr(configTraces, name)
                self.figure.add_traces(trace)

        if not equality:
            self.plot_inequality_optimum(Model, Optimums)
        else:
            raise NotImplementedError

        # Very slow, use simple GL lines
        # self.figure.add_vline( ...
        # )
        # self.figure.add_vline( ...
        # )
        # self.figure.add_vline( ...
        # )

        self._update_layout(Model)

    def _update_layout(self, Model):
        self.figure.update_layout(
            xaxis1=dict(
                title=r"$V \; (\text{m/s})$",
                side="bottom",
                range=[axes_min_speed, axes_max_speed],
            ),
            yaxis1=dict(
                title=r"$D \: (\text{N})$",
                side="left",
                range=[0.0, Model.drag_ylim],
            ),
            xaxis2=dict(
                title=r"$V \; (\text{m/s})$",
                side="bottom",
                range=[axes_min_speed, axes_max_speed],
                automargin=False,
            ),
            yaxis2=dict(
                title=r"$P \: (\text{kW})$",
                range=[0.0, Model.power_ylim],
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

    def plot_inequality_optimum(self, Model, Optimums):
        all_traces = []

        hopt_combined = np.hstack([opt.hopt_array for opt in Optimums])

        V_envelope_combined = np.hstack([opt.V_envelope for opt in Optimums])

        # Build dT_optimum list first, then check if empty
        dT_list = [
            opt.dTopt * opt.cond
            for opt in Optimums
            if not np.isnan(opt.dTopt * opt.cond)
        ]

        if dT_list:
            dT_optimum = np.hstack(dT_list)
        else:
            dT_optimum = np.array([1.0])

        for i in range(len(Optimums)):
            Optimum = Optimums[i]

            power_available_trace = go.Scattergl(
                x=Model.V_CLarray,
                y=dT_optimum * Model.power_available / 1e3,
                xaxis="x2",
                yaxis="y2",
                mode="lines",
                showlegend=False,
                line=dict(color=AVAILABLE_COLOR),
            )
            thrust_available = go.Scattergl(
                x=Model.V_CLarray,
                y=dT_optimum * Model.thrust,
                xaxis="x1",
                yaxis="y1",
                mode="lines",
                showlegend=False,
                line=dict(color=AVAILABLE_COLOR),
            )

            flight_envelope_trace = go.Scattergl(
                x=V_envelope_combined,
                y=hopt_combined,
                xaxis="x4",
                yaxis="y4",
                mode="lines",
                showlegend=False,
                line=dict(color=SALMON),
            )

            # Add markers
            drag_marker = go.Scattergl(
                x=[Optimum.V_selected],
                y=[Optimum.power_selected / Optimum.V_selected],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=10,
                    color=WHITE,
                    symbol="circle",
                ),
                name="label",
                xaxis="x1",
                yaxis="y1",
            )

            power_marker = go.Scattergl(
                x=[Optimum.V_selected],
                y=[Optimum.power_selected / 1e3],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=10,
                    color=WHITE,
                    symbol="circle",
                ),
                name="label",
                xaxis="x2",
                yaxis="y2",
            )

            domain_marker = go.Scattergl(
                x=[Optimum.CLopt_selected],
                y=[Optimum.dTopt],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=10,
                    color=WHITE,
                    symbol="circle",
                ),
                name="label",
                xaxis="x3",
                yaxis="y3",
            )

            all_traces.extend(
                [
                    power_available_trace,
                    thrust_available,
                    flight_envelope_trace,
                    drag_marker,
                    power_marker,
                    domain_marker,
                ]
            )

        # Add all traces in a single batch for efficiency
        self.figure.add_traces(tuple(all_traces))

        for i in range(len(Optimums)):
            Optimum = Optimums[i]

            envelope_marker = go.Scattergl(
                x=[Optimum.V_selected],
                y=[Optimum.h_selected],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=10,
                    color=WHITE,
                    symbol="circle",
                ),
                name="label",
                xaxis="x4",
                yaxis="y4",
            )

            self.figure.add_trace(envelope_marker)


def add_equality(Optimums):
    traces = []

    for Optimum in Optimums:
        flight_envelope_trace = go.Scattergl(
            x=Optimum.V_envelope,
            y=Optimum.hopt_array,
            xaxis="x4",
            yaxis="y4",
            mode="markers",
            showlegend=False,
            marker=dict(size=10, color=SALMON),
        )

        traces.append(flight_envelope_trace)

    return traces


class configTraces:
    def __init__(self, Model, surface, constraint=True, factor=2):
        self.heatmap = go.Heatmap(
            x=Model.aircraft.CL_array,
            y=Model.aircraft.dT_array,
            z=surface,
            zsmooth="fast",
            opacity=0.9,
            name="gino",  # label_name.split(maxsplit=1)[0],
            colorscale="viridis",
            colorbar={"title": ""},
            xaxis="x3",
            yaxis="y3",
            zmin=np.min(surface),
            zmax=np.min(surface) * factor,
        )

        self.CLP_trace_drag = go.Scattergl(
            x=[Model.V_CLP, Model.V_CLP],
            y=[0, Model.drag_ylim],
            xaxis="x1",
            yaxis="y1",
            mode="lines",
            showlegend=False,
            line=dict(dash="dot", color=LIGHTGREY),
        )

        self.CLP_trace_power = go.Scattergl(
            x=[Model.V_CLP, Model.V_CLP],
            y=[0, Model.power_ylim],
            xaxis="x2",
            yaxis="y2",
            mode="lines",
            showlegend=False,
            line=dict(dash="dot", color=LIGHTGREY),
        )

        self.CLE_trace_drag = go.Scattergl(
            x=[Model.V_CLE, Model.V_CLE],
            y=[0, Model.drag_ylim],
            xaxis="x1",
            yaxis="y1",
            mode="lines",
            showlegend=False,
            line=dict(dash="dot", color=LIGHTGREY),
        )

        self.CLE_trace_power = go.Scattergl(
            x=[Model.V_CLE, Model.V_CLE],
            y=[0, Model.power_ylim],
            xaxis="x2",
            yaxis="y2",
            mode="lines",
            showlegend=False,
            line=dict(dash="dot", color=LIGHTGREY),
        )

        self.CLmax_trace_drag = go.Scattergl(
            x=[Model.Vstall_envelope[Model.idx_h], Model.Vstall_envelope[Model.idx_h]],
            y=[0, Model.drag_ylim],
            xaxis="x1",
            yaxis="y1",
            mode="lines",
            showlegend=False,
            line=dict(dash="dot", color=CLMAX_AXES),
        )

        self.CLmax_trace_power = go.Scattergl(
            x=[Model.Vstall_envelope[Model.idx_h], Model.Vstall_envelope[Model.idx_h]],
            y=[0, Model.power_ylim],
            xaxis="x2",
            yaxis="y2",
            mode="lines",
            showlegend=False,
            line=dict(dash="dot", color=CLMAX_AXES),
        )

        self.drag_trace = go.Scattergl(
            x=Model.V_CLarray,
            y=Model.drag_curve,
            name="D",
            line=dict(color=DRAG_COLOR, width=2),
            mode="lines",
            xaxis="x1",
            yaxis="y1",
            showlegend=False,
        )

        self.power_required_trace = go.Scattergl(
            x=Model.V_CLarray,
            y=Model.power_required / 1e3,
            name="P",
            line=dict(color=POWER_COLOR, width=2),
            mode="lines",
            xaxis="x2",
            yaxis="y2",
            showlegend=False,
        )

        if constraint:
            self.constraint_trace = go.Scattergl(
                x=Model.aircraft.CL_array,
                y=Model.equilibrium_dT,
                name="constraint",
                line=dict(color=CONSTRAINT_CLR, width=10),
                mode="lines",
                xaxis="x3",
                yaxis="y3",
                showlegend=False,
            )

        self.CLaxes_drag = self._create_CL_axes(
            Model.Vstall_envelope[Model.idx_h],
            0.1 * Model.drag_ylim,
            "x1",
            "y1",
            LIGHTGREY,
        )
        self.CLaxes_power = self._create_CL_axes(
            Model.Vstall_envelope[Model.idx_h],
            0.1 * Model.power_ylim,
            "x2",
            "y2",
            LIGHTGREY,
        )

        self.Vstall_trace = self._create_stall_trace(
            Model.aircraft.h_array, Model.Vstall_envelope
        )

        self.Mach_trace = self._create_mach_trace(
            Model.aircraft.h_array, atmos.a(Model.aircraft.h_array)
        )

    def _create_CL_axes(self, V_stall, y_pos, plot_on_x, plot_on_y, color=LIGHTGREY):
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

    def _create_marker_trace(
        self, x_axes, y_axes, label, color, plot_on_x, plot_on_y, width=2, legend=False
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

    def _create_stall_trace(self, h, V, plot_on_x="x4", plot_on_y="y4"):
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
                y=[axes_max_h * 0.8],
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

    def _create_mach_trace(self, h, a, plot_on_x="x4", plot_on_y="y4"):
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
                y=[axes_max_h * 0.8],
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


class InteractiveElements:
    def __init__(self, aircraft, initial=False):
        self.aircraft = aircraft
        self.init_sliders(initial)

    @staticmethod
    def init_table(data):
        table = mo.ui.table(
            data=data,
            pagination=True,
            show_column_summaries=False,
            selection="single",
            initial_selection=[0],
            page_size=4,
            show_data_types=False,
        )
        return table

    def init_sliders(self, initial):
        self.mass_slider = mo.ui.slider(
            start=0, stop=1, step=0.1, label=r"", show_value=True
        )

        self.altitude_slider = mo.ui.slider(
            start=0,
            stop=20,
            step=0.5,
            label=r"Altitude (km)",
            value=0,
            show_value=True,
        )
        if initial:
            self.CL_slider = mo.ui.slider(
                start=0,
                stop=self.aircraft.CLmax,
                step=0.2,
                label=r"$C_L$",
                value=0.5,
            )
            self.dT_slider = mo.ui.slider(
                start=0, stop=1, step=0.1, label=r"$\delta_T$", value=0.5
            )

    def init_analysis_tabs(self):
        titles_dict = {
            "### Interior solutions": "",
            "### Lift limited solutions": "",
            "### Thrust limited solutions": "",
            "### Lift-thrust limited solutions": "",
        }

        self.tab = mo.ui.tabs(titles_dict)
        view = (
            self.tab.style({"height": "60px", "overflow": "auto"})
            .callout(kind="info")
            .center()
        )

        return view, list(titles_dict.keys())

    def init_layout(self, mass_slider, altitude_slider):
        mass_stack = mo.hstack(
            [mo.md("**OEW**"), mass_slider, mo.md("**MTOW**")],
            align="start",
            justify="start",
        )
        variables_stack = mo.hstack([mass_stack, altitude_slider])

        return mass_stack, variables_stack

    def sense_mass(self, slider):
        self.mass_selected = (
            self.aircraft.OEM + (self.aircraft.MTOM - self.aircraft.OEM) * slider.value
        ) * atmos.g0

        return self.mass_selected

    def sense_altitude(self, slider):
        self.altitude_selected = int(slider.value * 1e3)

        return self.altitude_selected


class InitialFig:
    def __init__(self, Model, surface, Config: configTraces, factor=2):
        figure = go.Figure()

        # Minimum velocity surface
        figure.add_traces(
            [
                go.Surface(
                    x=Model.aircraft.CL_array,
                    y=Model.aircraft.dT_array,
                    z=surface,
                    opacity=0.9,
                    name="Velocity",
                    colorscale="viridis",
                    cmax=factor * np.min(surface),
                    cmin=np.min(surface),
                    colorbar={"title": "Velocity (m/s)"},
                ),
                go.Scatter3d(
                    x=Model.aircraft.CL_array,
                    y=Model.equilibrium_dT,
                    z=surface[0],
                    opacity=1,
                    mode="lines",
                    showlegend=False,
                    line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
                    name="g1 constraint",
                ),
                # go.Scatter3d(
                #     x=[CL_array[50] + 0.35],
                #     y=[constraint[50] + 0.3],
                #     z=[velocity_surface[0, 50] - 0.1],
                #     opacity=1,
                #     textposition="middle left",
                #     mode="markers+text",
                #     text=["g<sub>1</sub>"],
                #     marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                #     showlegend=False,
                #     name="g1 constraint",
                #     textfont=dict(size=14, family="Arial"),
                # ),
                # go.Scatter3d(
                #     x=[CL_slider.value],
                #     y=[dT_slider.value],
                #     z=[velocity_user_selected],
                #     mode="markers",
                #     showlegend=False,
                #     marker=dict(
                #         size=3,
                #         color="white",
                #         symbol="circle",
                #     ),
                #     name="Design Point",
                #     hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub> : %{y}<br>V: %{z}<extra>%{fullData.name}</extra>",
                # ),
            ]
        )
        camera = dict(eye=dict(x=1.35, y=1.35, z=1.35))

        figure.update_layout(
            scene=dict(
                xaxis=dict(
                    title="C<sub>L</sub> (-)",
                    range=[xy_lowerbound, Model.aircraft.CLmax],
                ),
                yaxis=dict(title="δ<sub>T</sub> (-)", range=[xy_lowerbound, 1]),
                zaxis=dict(title="V (m/s)", range=[0, factor * np.min(surface)]),
            ),
        )

        figure.update_layout(
            scene_camera=camera,
            title={
                "text": f"{Model.aircraft.full_name}",
                "font": {"size": 25},
                "xanchor": "center",
                "yanchor": "top",
                "x": 0.5,
            },
        )

        self.figure = figure
