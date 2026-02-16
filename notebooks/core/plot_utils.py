from __future__ import annotations  # Enables postponed evaluation of annotations
from typing import TYPE_CHECKING
import copy


if TYPE_CHECKING:
    # These imports only run during type checking, not at runtime
    from core.aircraft import (
        SimplifiedAircraft,
        ModelSimplifiedJet,
        ModelSimplifiedProp,
    )

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

# from core.aircraft import ModelSimplifiedJet, ModelSimplifiedProp, SimplifiedAircraft
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
    # Class-level defaults
    DEFAULT_OPTIONS = {
        "axes": {
            "x": {
                "data_key": "aircraft.CL_array",
                "label": "C<sub>L</sub> (-)",
                "bound": "aircraft.CLmax",
            },
            "y": {
                "data_key": "aircraft.dT_array",
                "label": "δ<sub>T</sub> (-)",
                "bound": 1,
            },
            "z": {"label": "V (m/s)"},
        },
        "xy_lowerbound": -0.1,
        "z_lowerbound": 0.0,
        "camera": {"x": 1.35, "y": 1.35, "z": 1.35},
        "opacity": 0.9,
        "colorscale": "viridis",
        "factor": 2,
    }

    def __init__(
        self,
        Model: ModelSimplifiedJet | ModelSimplifiedProp | SimplifiedAircraft,
        plot_options: dict,
        selected: list,
    ):
        """
        :param Model: Aircraft model
        :param plot_options: Required keys: "surface", "title"
                            Optional keys: "axes", "xy_lowerbound", "camera", "opacity", "colorscale"
        :param factor: Scaling factor for z-axis range
        """
        # Merge user options with defaults
        opts = self._merge_options(plot_options)

        # Extract commonly used values
        surface = opts["surface"]
        factor = opts["factor"]
        xy_lowerbound = opts["xy_lowerbound"]
        axes = opts["axes"]
        z_lowerbound = opts["z_lowerbound"]

        # Get axis data from model
        x_data = self._get_data(Model, axes["x"]["data_key"])
        y_data = self._get_data(Model, axes["y"]["data_key"])

        # Get bounds (can be a dot-notation string or a number)
        x_bound = self._resolve_bound(axes["x"]["bound"], Model)
        y_bound = self._resolve_bound(axes["y"]["bound"], Model)

        figure = go.Figure()

        figure.add_trace(
            go.Surface(
                x=x_data,
                y=y_data,
                z=surface,
                opacity=opts["opacity"],
                name="Velocity",
                colorscale=opts["colorscale"],
                cmax=factor * np.min(surface),
                cmin=np.min(surface),
                colorbar={"title": axes["z"]["label"]},
            )
        )

        figure.add_trace(
            go.Scatter3d(
                x=x_data,
                y=Model.equilibrium_dT,
                z=surface[0],
                opacity=1,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
                name="g1 constraint",
            )
        )

        figure.add_trace(
            go.Scatter3d(
                x=[selected[0]],
                y=[selected[1]],
                z=[selected[2]],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=3,
                    color="white",
                    symbol="circle",
                ),
            )
        )

        camera = dict(eye=dict(**opts["camera"]))

        figure.update_layout(
            scene_dragmode="turntable",
            scene=dict(
                xaxis=dict(title=axes["x"]["label"], range=[xy_lowerbound, x_bound]),
                yaxis=dict(title=axes["y"]["label"], range=[xy_lowerbound, y_bound]),
                zaxis=dict(
                    title=axes["z"]["label"],
                    range=[z_lowerbound, factor * np.min(surface)],
                ),
            ),
            scene_camera=camera,
            title={
                "text": f"{opts['title']} for {Model.aircraft.full_name}",
                "font": {"size": 25},
                "xanchor": "center",
                "yanchor": "top",
                "x": 0.5,
            },
        )

        self.figure = figure

    def _merge_options(self, user_options: dict) -> dict:
        """Deep merge user options with defaults."""

        opts = copy.deepcopy(self.DEFAULT_OPTIONS)

        for key, value in user_options.items():
            if key in opts and isinstance(opts[key], dict) and isinstance(value, dict):
                opts[key].update(value)
            else:
                opts[key] = value
        return opts

    def _get_data(self, model, data_key: str):
        """Traverse dot-notation path to get data. e.g. 'aircraft.CL_array' or 'CL_array'"""
        obj = model
        for key in data_key.split("."):
            obj = getattr(obj, key)
        return obj

    def _resolve_bound(self, bound, model):
        """Resolve bound: if dot-notation string, traverse path; otherwise return as-is."""
        if isinstance(bound, str):
            return self._get_data(model, bound)
        return bound


class configTraces:
    def __init__(self, Model, surface, constraint=True, factor=2):
        self.heatmap = go.Heatmap(
            x=Model.aircraft.CL_array,
            y=Model.aircraft.dT_array,
            z=surface,
            zsmooth="fast",
            opacity=0.9,
            # name="gino",  # label_name.split(maxsplit=1)[0],
            hovertemplate="x=%{x}<br>y=%{y}<br>z=%{z}<extra></extra>",
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


class OptimumGridViewNew:
    def __init__(
        self,
        Model: ModelSimplifiedJet | ModelSimplifiedProp,
        configTraces: configTraces,
        Optimums: list[ModelSimplifiedProp | ModelSimplifiedJet],
        equality=False,
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
                    domain_marker,
                ]
            )

        # Add all traces in a single batch for efficiency
        self.figure.add_traces(tuple(all_traces))

        for i in range(len(Optimums)):
            Optimum = Optimums[i]

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

            self.figure.add_traces([envelope_marker, power_marker, drag_marker])


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


class OptimumGridView:
    DEFAULT_OPTIONS = {
        "axes": {
            "x1": {
                "title": r"$V \; (\text{m/s})$",
                "range": [axes_min_speed, axes_max_speed],
            },
            "y1": {"title": r"$D \: (\text{N})$", "range_key": "drag_ylim"},
            "x2": {
                "title": r"$V \; (\text{m/s})$",
                "range": [axes_min_speed, axes_max_speed],
            },
            "y2": {"title": r"$P \: (\text{kW})$", "range_key": "power_ylim"},
            "x3": {
                "title": r"$C_L\:(\text{-})$",
                "data_key": "aircraft.CL_array",
                "bound": "aircraft.CLmax",
            },
            "y3": {
                "title": r"$\delta_T \:(\text{-})$",
                "data_key": "aircraft.dT_array",
                "bound": 1.0,
                "range": [axes_min_dT, axes_max_dT],
            },
            "x4": {
                "title": r"$V \: \text{(m/s)}$",
                "range": [axes_min_speed, axes_max_speed],
            },
            "y4": {"title": r"$h \: \text{(km)}$", "range": [axes_min_h, axes_max_h]},
        },
        "constraint": True,
        "factor": 2,
        "height": 800,
        "xy_lowerbound": -0.1,
    }

    def __init__(
        self,
        Model,
        Optimums: list,
        plot_options: dict = None,
        equality=False,
    ):
        opts = self._merge_options(plot_options or {})

        surface = opts["surface"]

        self.figure = make_subplots(
            rows=2,
            cols=2,
            horizontal_spacing=0.1,
            vertical_spacing=0.15,
        )

        # Add all base traces
        self._add_base_traces(Model, surface, opts)

        if not equality:
            self._plot_inequality_optimum(Model, Optimums, opts)
        else:
            raise NotImplementedError

        self._update_layout(Model, opts)

    def _merge_options(self, user_options: dict) -> dict:
        """Deep merge user options with defaults."""
        opts = copy.deepcopy(self.DEFAULT_OPTIONS)

        for key, value in user_options.items():
            if key in opts and isinstance(opts[key], dict) and isinstance(value, dict):
                for k, v in value.items():
                    if (
                        k in opts[key]
                        and isinstance(opts[key][k], dict)
                        and isinstance(v, dict)
                    ):
                        opts[key][k].update(v)
                    else:
                        opts[key][k] = v
            else:
                opts[key] = value
        return opts

    def _get_data(self, model, data_key: str):
        """Traverse dot-notation path to get data. e.g. 'aircraft.CL_array' or 'CL_array'"""
        obj = model
        for key in data_key.split("."):
            obj = getattr(obj, key)
        return obj

    def _resolve_bound(self, bound, model):
        """Resolve bound: if dot-notation string, traverse path; otherwise return as-is."""
        if isinstance(bound, str):
            return self._get_data(model, bound)
        return bound

    def _add_base_traces(self, Model, surface, opts):
        factor = opts["factor"]
        axes = opts["axes"]

        # Get x3 and y3 data from configurable data_key
        x3_data = self._get_data(Model, axes["x3"]["data_key"])
        y3_data = self._get_data(Model, axes["y3"]["data_key"])

        # Heatmap with configurable x/y data
        self.figure.add_trace(
            go.Heatmap(
                x=x3_data,
                y=y3_data,
                z=surface,
                zsmooth="best",
                opacity=0.9,
                hovertemplate="x=%{x}<br>y=%{y}<br>z=%{z}<extra></extra>",
                colorscale="viridis",
                colorbar={"title": ""},
                xaxis="x3",
                yaxis="y3",
                zmin=np.min(surface),
                zmax=np.min(surface) * factor,
            )
        )

        # Vertical reference lines
        self._add_vertical_lines(Model)

        # Drag and power curves
        self._add_performance_curves(Model)

        # Constraint
        if opts["constraint"]:
            self.figure.add_trace(
                go.Scattergl(
                    x=x3_data,
                    y=Model.equilibrium_dT,
                    name="constraint",
                    line=dict(color=CONSTRAINT_CLR, width=10),
                    mode="lines",
                    xaxis="x3",
                    yaxis="y3",
                    showlegend=False,
                )
            )

        # CL axes, stall trace, mach trace
        self._add_cl_axes(Model)
        self._add_stall_trace(Model)
        self._add_mach_trace(Model)

    def _add_vertical_lines(self, Model):
        """Add CLP, CLE, and CLmax vertical lines to drag and power plots."""
        lines = [
            (Model.V_CLP, LIGHTGREY),
            (Model.V_CLE, LIGHTGREY),
            (Model.Vstall_envelope[Model.idx_h], CLMAX_AXES),
        ]

        for V, color in lines:
            # Drag plot
            self.figure.add_trace(
                go.Scattergl(
                    x=[V, V],
                    y=[0, Model.drag_ylim],
                    xaxis="x1",
                    yaxis="y1",
                    mode="lines",
                    showlegend=False,
                    line=dict(dash="dot", color=color),
                )
            )
            # Power plot
            self.figure.add_trace(
                go.Scattergl(
                    x=[V, V],
                    y=[0, Model.power_ylim],
                    xaxis="x2",
                    yaxis="y2",
                    mode="lines",
                    showlegend=False,
                    line=dict(dash="dot", color=color),
                )
            )

    def _add_performance_curves(self, Model):
        self.figure.add_trace(
            go.Scattergl(
                x=Model.V_CLarray,
                y=Model.drag_curve,
                name="D",
                line=dict(color=DRAG_COLOR, width=2),
                mode="lines",
                xaxis="x1",
                yaxis="y1",
                showlegend=False,
            )
        )

        self.figure.add_trace(
            go.Scattergl(
                x=Model.V_CLarray,
                y=Model.power_required / 1e3,
                name="P",
                line=dict(color=POWER_COLOR, width=2),
                mode="lines",
                xaxis="x2",
                yaxis="y2",
                showlegend=False,
            )
        )

    def _add_cl_axes(self, Model):
        V_stall = Model.Vstall_envelope[Model.idx_h]

        for xaxis, yaxis, y_pos in [
            ("x1", "y1", 0.1 * Model.drag_ylim),
            ("x2", "y2", 0.1 * Model.power_ylim),
        ]:
            self.figure.add_traces(
                [
                    go.Scattergl(
                        x=[V_stall - 20, axes_max_speed * 2],
                        y=[y_pos, y_pos],
                        mode="lines",
                        xaxis=xaxis,
                        yaxis=yaxis,
                        line=dict(color=LIGHTGREY, width=1),
                        showlegend=False,
                    ),
                    go.Scattergl(
                        x=[V_stall - 20],
                        y=[y_pos],
                        mode="markers",
                        xaxis=xaxis,
                        yaxis=yaxis,
                        marker=dict(color=LIGHTGREY, size=10, symbol="arrow-left"),
                        showlegend=False,
                    ),
                ]
            )

    def _add_stall_trace(self, Model):
        h = Model.aircraft.h_array
        V = Model.Vstall_envelope

        self.figure.add_traces(
            [
                go.Scatter(
                    x=V,
                    y=h / 1e3,
                    mode="lines",
                    line=dict(width=1, color=LIGHTGREY, dash="dash"),
                    name="V<sub>stall</sub>",
                    showlegend=False,
                    xaxis="x4",
                    yaxis="y4",
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
                    xaxis="x4",
                    yaxis="y4",
                ),
            ]
        )

    def _add_mach_trace(self, Model):
        h = Model.aircraft.h_array
        a = atmos.a(h)

        self.figure.add_traces(
            [
                go.Scatter(
                    x=a,
                    y=h / 1e3,
                    mode="lines",
                    line=dict(color=LIGHTGREY, width=2, dash="dash"),
                    name="M1.0",
                    showlegend=False,
                    xaxis="x4",
                    yaxis="y4",
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
                    xaxis="x4",
                    yaxis="y4",
                ),
            ]
        )

    def _update_layout(self, Model, opts):
        axes = opts["axes"]
        xy_lowerbound = opts.get("xy_lowerbound", -0.1)

        layout_dict = {
            "height": opts["height"],
            "legend": dict(x=0.02, y=0.98),
            "dragmode": "pan",
        }

        # Map short names to Plotly layout names
        axis_name_map = {
            "x1": "xaxis",
            "y1": "yaxis",
            "x2": "xaxis2",
            "y2": "yaxis2",
            "x3": "xaxis3",
            "y3": "yaxis3",
            "x4": "xaxis4",
            "y4": "yaxis4",
        }

        for axis_name, axis_opts in axes.items():
            axis_config = {"title": axis_opts.get("title")}

            # Handle range: explicit range, range_key, or bound-based
            if "range" in axis_opts:
                axis_config["range"] = axis_opts["range"]
            elif "range_key" in axis_opts:
                axis_config["range"] = [0.0, getattr(Model, axis_opts["range_key"])]
            elif "bound" in axis_opts:
                # Use bound (like InitialFig does)
                upper_bound = self._resolve_bound(axis_opts["bound"], Model)
                axis_config["range"] = [xy_lowerbound, upper_bound]

            if axis_name in ("x3", "y3", "x4", "y4"):
                axis_config["showgrid"] = True
                axis_config["gridcolor"] = "#515151"
                axis_config["gridwidth"] = 1

            # Convert x1 -> xaxis, x2 -> xaxis2, etc.
            layout_key = axis_name_map[axis_name]
            layout_dict[layout_key] = axis_config

        self.figure.update_layout(**layout_dict)

    def _plot_inequality_optimum(self, Model, Optimums, opts):
        all_traces = []
        axes = opts["axes"]

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

            # Get the optimum marker position using the configurable data keys
            # We need to map from Optimum's selected values to the appropriate x3/y3 coordinates
            # By default this would be CLopt_selected and dTopt, but could be different
            x3_opt_key = axes["x3"].get("optimum_key", "CLopt_selected")
            y3_opt_key = axes["y3"].get("optimum_key", "dTopt")

            x3_opt_value = getattr(Optimum, x3_opt_key)
            y3_opt_value = getattr(Optimum, y3_opt_key)

            domain_marker = go.Scattergl(
                x=[x3_opt_value],
                y=[y3_opt_value],
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
                    domain_marker,
                ]
            )

        # Add all traces in a single batch for efficiency
        self.figure.add_traces(tuple(all_traces))

        for i in range(len(Optimums)):
            Optimum = Optimums[i]

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

            self.figure.add_traces([envelope_marker, power_marker, drag_marker])
