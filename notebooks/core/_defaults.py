import marimo as mo
from plotly.io import templates as piot
import plotly.io as pio
from pathlib import Path
import numpy as np

FILEURL = None


# Get base url
def get_url():
    base_ = mo.notebook_location()

    if isinstance(base_, Path):
        return "/?file="

    return str(str(base_).rstrip("/") + "/")


# Plotly
def set_plotly_template():
    piot.default = "plotly_dark+xgridoff+ygridoff"
    piot["plotly_dark"].layout.paper_bgcolor = "rgba(0,0,0,0)"
    piot["plotly_dark"].layout.plot_bgcolor = "rgba(0,0,0,0)"
    piot["plotly_dark"].layout.font.family = "Roboto"
    piot["plotly_dark"].layout.font.size = 14


# Navigation sidebar
def set_sidebar(tabs_switch):
    sidebar = [
        mo.md(f"""
            <h1>
            <a href="{FILEURL}Homepage.py" style="color: #FFFFFF; text-decoration: none;">
                FPAO-CC
            </a>
            </h1>
            """),
        mo.nav_menu(
            {
                f"{FILEURL}Scope.py": "Scope",
                "References": {
                    f"{FILEURL}Nomenclature.py": "Nomenclature",
                },
                "Models library": {
                    f"{FILEURL}Atmosphere.py": "Atmosphere",
                    f"{FILEURL}AircraftSimplified.py": "Simplified Aircraft",
                    f"{FILEURL}AircraftCustom.py": "Custom Aircraft",
                },
                "Performance Optimization": {
                    f"{FILEURL}ProblemFormulation.py": "Problem Formulation",
                    f"{FILEURL}FlightConstraints.py": "Flight Constraints",
                    f"{FILEURL}FlightControls.py": "Flight Controls",
                },
                "Steady Level Flight": {
                    f"{FILEURL}AerodynamicEfficiency.py": "Aerodynamic Efficiency",
                    f"{FILEURL}MinDrag.py": "Minimum Drag",
                    f"{FILEURL}MinPower.py": "Minimum Power",
                    f"{FILEURL}MinSpeed.py": "Minimum Speed",
                    f"{FILEURL}MaxSpeed.py": "Maximum Speed",
                    f"{FILEURL}MaxAltitude.py": "Maximum Altitude",
                },
            },
            orientation="vertical",
        ),
    ]

    return mo.sidebar(
        sidebar,
        width="300px",
        footer=tabs_switch,
    )


# Navigation footer
def nav_footer(
    before_file=None,
    before_title=None,
    after_file=None,
    after_title=None,
    above_file=None,
    above_title=None,
    above_before=None,
):
    nav_items = {}
    if above_before:
        if above_file and above_title:
            nav_items[f"{FILEURL}{above_file}"] = (
                f"{mo.icon('lucide:arrow-big-up')} {above_title}"
            )
        if before_file and before_title:
            nav_items[f"{FILEURL}{before_file}"] = (
                f"{mo.icon('lucide:arrow-big-left')} {before_title}"
            )
        if after_file and after_title:
            nav_items[f"{FILEURL}{after_file}"] = (
                f"{after_title} {mo.icon('lucide:arrow-big-right')}"
            )
    else:
        if before_file and before_title:
            nav_items[f"{FILEURL}{before_file}"] = (
                f"{mo.icon('lucide:arrow-big-left')} {before_title}"
            )
        if after_file and after_title:
            nav_items[f"{FILEURL}{after_file}"] = (
                f"{after_title} {mo.icon('lucide:arrow-big-right')}"
            )
        if above_file and above_title:
            nav_items[f"{FILEURL}{above_file}"] = (
                f"{mo.icon('lucide:arrow-big-up')} {above_title}"
            )
    return mo.nav_menu(nav_items).center()


def safe_index(array, idx):
    if 0 <= idx < len(array):
        return array[idx]
    else:
        return np.nan


def clone_figure(fig):
    return pio.from_json(pio.to_json(fig))
