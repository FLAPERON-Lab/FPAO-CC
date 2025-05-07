import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import _defaults as defs

    defs.set_plotly_template()
    mo.sidebar(
        defs.sidebar,
        width="300px",
        # footer=mo.md(""),
    )
    return defs, mo


@app.cell
def _(mo):
    mo.md(
        r"""
    # Scope
    These notebooks show fundamental and advanced techniques to analyse, optimize and visualize the flight performance of fixed-wing aircraft of different types and categories.

    Focus is placed on:

    1. formalizing the mathematical formulation of the Flight Performance Analysis and Optimization (FPAO) problem;
    2. highlighting the role of physical and operational constraints on optimal aircraft performance;
    3. comparing the analytical derivation with the numerical solution, creating a bridge from Calculus to Computers (CC)

    For didactic purposes, different types of assumptions are made in the selection of the physical models used.

    Multiple visualizations are provided for a given topic or concept in order to stimulate understanding from different perspectives.

    Interactive elements are provided to incentivize the student to explore the analysis and gain a deeper familiarity with the elements in play.
    """
    )
    return


@app.cell
def _(defs, mo):
    nav_foot = mo.nav_menu(
        {
            f"{defs._fileurl}Nomenclature.py": f"Nomenclature {mo.icon('lucide:arrow-big-right')}",
        }
    ).center()
    nav_foot
    return


if __name__ == "__main__":
    app.run()
