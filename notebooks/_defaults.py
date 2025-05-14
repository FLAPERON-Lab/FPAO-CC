import marimo as mo
from plotly.io import templates as piot

FILEURL = "/?file="


# Plotly
def set_plotly_template():
    piot.default = "plotly_dark+xgridoff+ygridoff"
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
def nav_footer(before_file, before_title, after_file, after_title):
    nav_items = {}
    if before_file and before_title:
        nav_items[f"{FILEURL}{before_file}"] = (
            f"{mo.icon('lucide:arrow-big-left')} {before_title}"
        )
    if after_file and after_title:
        nav_items[f"{FILEURL}{after_file}"] = (
            f"{after_title} {mo.icon('lucide:arrow-big-right')}"
        )
    return mo.nav_menu(nav_items).center()
