import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    from core import _defaults

    _defaults.FILEURL = _defaults.get_url()

    _defaults.set_plotly_template()


@app.cell
def _():
    _defaults.set_sidebar()
    return


@app.cell
def _():
    mo.md(r"""
    # Scope
    These notebooks show fundamental and advanced techniques to analyse, optimize and visualize the flight performance of fixed-wing aircraft of different types and categories.

    Focus is placed on:

    1. formalizing the mathematical formulation of the Flight Performance Analysis and Optimization (FPAO) problem;
    2. highlighting the role of physical and operational constraints on optimal aircraft performance;
    3. comparing the analytical derivation with the numerical solution, creating a bridge from Calculus to Computers (CC)

    For didactic purposes, different types of assumptions are made in the selection of the physical models used.

    Multiple visualizations are provided for a given topic or concept in order to stimulate understanding from different perspectives.

    Interactive elements are provided to incentivize the student to explore the analysis and gain a deeper familiarity with the elements in play.

    The scope is limited to _point_ performance optimization, which means optimization of objective functions that do not depend on time, and therefore are independent on the dynamic evolution of the system.
    """)
    return


@app.cell
def _():
    _defaults.nav_footer("", "", "Nomenclature.py", "Nomenclature")
    return


if __name__ == "__main__":
    app.run()
