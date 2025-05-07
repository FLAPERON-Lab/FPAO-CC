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
def _(defs, mo):
    nav_foot = mo.nav_menu(
        {
            f"{defs._fileurl}AircraftSimplified.py": f"{mo.icon('lucide:arrow-big-left')} Simplified Aircraft Models",
        }
    ).center()
    nav_foot
    return


if __name__ == "__main__":
    app.run()
