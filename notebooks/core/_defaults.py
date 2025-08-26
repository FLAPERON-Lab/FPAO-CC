import marimo as mo
from plotly.io import templates as piot
from pathlib import Path

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
def set_sidebar():
    sidebar = [
        mo.md("# FPAO-CC"),
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
                    f"{FILEURL}MinSpeed.py": "Minimum Speed",
                    f"{FILEURL}MinPower.py": "Minimum Power",
                },
            },
            orientation="vertical",
        ),
    ]
    return mo.sidebar(
        sidebar,
        width="300px",
        # footer=mo.md(""),
    )


# Navigation footer
def nav_footer(
    before_file=None,
    before_title=None,
    after_file=None,
    after_title=None,
    above_file=None,
    above_title=None,
):
    nav_items = {}
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
