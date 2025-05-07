import marimo as mo
from plotly.io import templates as piot


# Plotly
def set_plotly_template():
    piot.default = "plotly_dark+xgridoff+ygridoff"
    piot["plotly_dark"].layout.paper_bgcolor = "rgba(0,0,0,0)"
    piot["plotly_dark"].layout.plot_bgcolor = "rgba(0,0,0,0)"
    piot["plotly_dark"].layout.font.family = "Roboto"
    piot["plotly_dark"].layout.font.size = 14


# Navigation sidebar
_fileurl = "/?file="
sidebar = [
    mo.md("# FPAO-CC"),
    mo.nav_menu(
        {
            f"{_fileurl}Scope.py": "Scope",
            "References": {
                f"{_fileurl}Nomenclature.py": "Nomenclature",
            },
            "Models library": {
                f"{_fileurl}Atmosphere.py": "Atmosphere",
                f"{_fileurl}AircraftSimplified.py": "Simplified Aircraft",
                f"{_fileurl}AircraftCustom.py": "Custom Aircraft",
            },
        },
        orientation="vertical",
    ),
]
