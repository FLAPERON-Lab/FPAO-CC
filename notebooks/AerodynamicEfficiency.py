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
def _(mo):
    mo.md(r"""# Aerodynamic Efficiency""")
    return


@app.cell
def _():
    _defaults.nav_footer("FlightMechanics.py", "Flight Mechanics", "", "")
    return


if __name__ == "__main__":
    app.run()
