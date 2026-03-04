# SPDX-FileCopyrightText: 2026 Carmine Varriale <C.varriale@tudelft.nl>
# SPDX-FileCopyrightText: 2026 Federico Angioni <F.angioni@student.tudelft.nl>
# SPDX-FileCopyrightText: 2026 Maarten van Hoven <M.B.vanHoven@tudelft.nl>
#
# SPDX-License-Identifier: Apache-2.0
import marimo as mo
from plotly.io import templates as piot
import plotly.io as pio
from pathlib import Path
import numpy as np

FILEURL = None
GITHUB_REPO = "FPAO-CC"


# Get base url
def get_url():
    base_ = mo.notebook_location()

    if isinstance(base_, Path):
        return "/?file="

    # For GitHub Pages deployments, return just the repo root path (e.g., /FPAO-CC/)
    url_str = str(base_).rstrip("/")

    # Remove protocol if present (https://domain.com/path -> /path)
    if "://" in url_str:
        url_str = "/" + url_str.split("://", 1)[1].split("/", 1)[1]

    # Find FPAO-CC in the path and return up to it with trailing slash
    if f"/{GITHUB_REPO}/" in url_str:
        return f"/{GITHUB_REPO}/"
    elif url_str.endswith(f"/{GITHUB_REPO}"):
        return f"/{GITHUB_REPO}/"

    return url_str.rstrip("/") + "/"


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
        mo.md(f"""
            <h1>
            <a href="{FILEURL}Homepage.py" style="color: #FFFFFF; text-decoration: none;">
                CAPO-NBs
            </a>
            </h1>
            """),
        mo.nav_menu(
            {
                f"{FILEURL}Scope.py": "Scope",
                f"{FILEURL}Nomenclature.py": "Nomenclature",
                "Models Library": {
                    f"{FILEURL}Models_Library/Atmosphere.py": "Atmosphere",
                    f"{FILEURL}Models_Library/AircraftSimplified.py": "Simplified Aircraft",
                    f"{FILEURL}Models_Library/AircraftCustom.py": "Custom Aircraft",
                },
                "Problem Formulation": {
                    f"{FILEURL}Problem_Formulation/PointPerformance.py": "Point Performance",
                    f"{FILEURL}Problem_Formulation/FlightConstraints.py": "Flight Constraints",
                    f"{FILEURL}Problem_Formulation/FlightControls.py": "Flight Controls",
                },
                "Optimization Methodology": {
                    f"{FILEURL}Optimization_Methodology/PreambleMethodologies.py": "Preamble Methodologies",
                    f"{FILEURL}Optimization_Methodology/UnivariateOptimization.py": "Univariate Optimization",
                    f"{FILEURL}Optimization_Methodology/BivariateOptimization.py": "Bivariate Optimization",
                    f"{FILEURL}Optimization_Methodology/EqualityConstraints.py": "Equality Constraints",
                    f"{FILEURL}Optimization_Methodology/InequalityConstraints.py": "Inequality Constraints",
                },
                "Steady Level Flight": {
                    f"{FILEURL}Steady_Level_Flight/MinDrag.py": "Minimum Drag",
                    f"{FILEURL}Steady_Level_Flight/MinPower.py": "Minimum Power",
                    f"{FILEURL}Steady_Level_Flight/MinSpeed.py": "Minimum Speed",
                    f"{FILEURL}Steady_Level_Flight/MaxSpeed.py": "Maximum Speed",
                    f"{FILEURL}Steady_Level_Flight/MaxAltitude.py": "Maximum Altitude",
                },
            },
            orientation="vertical",
        ),
    ]

    return mo.sidebar(
        sidebar,
        width="300px",
        # footer=tabs_switch,
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
