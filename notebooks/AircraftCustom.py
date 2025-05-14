import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import _defaults

    _defaults.set_plotly_template()


@app.cell
def _():
    _defaults.set_sidebar()
    return


@app.cell
def _():
    mo.md(
        r"""
    # Custom Aircraft Models
    More complex aero-propulsive models allow FPAO at higher fidelity and greater level of detail, by capturing phenomena in the way that is specific and characteristic of the particular aircraft in analysis.

    In these cases, the models for $C_D$, $T_a$ or $P_a$, $c_T$ or $c_P$, and optionally $C_L$, are typically provided in the form of tabular data, as a function of several flight parameters. 

    Depending on the available data, custom models allow expanding the analysis to flight conditions in which the simplified models are not accurate, such as stall or transonic/supersonic effects.   

    On the other hand, they require that FPAO methodologies have to be tailored to the specific model structure, and therefore are hard to automate.
    """
    )
    return


@app.cell
def _():
    ## Visualization
    return


@app.cell
def _():
    _defaults.nav_footer(
        "AircraftSimplified.py",
        "Simplified Aircraft Models",
        "ProblemFormulation.py",
        "Problem Formulation",
    )
    return


if __name__ == "__main__":
    app.run()
