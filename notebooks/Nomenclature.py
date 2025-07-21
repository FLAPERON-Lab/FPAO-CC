import marimo

__generated_with = "0.13.6"
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
def _(mo):
    mo.md("""# Nomenclature""")
    return


@app.cell
def _(mo):
    mo.md(r"""## List of symbols""")
    return


@app.cell
def _(mo):
    mo.md(r"""## List of acronyms""")
    return


@app.cell
def _():
    _defaults.nav_footer("Scope.py", "Scope", "Atmosphere.py", "Atmosphere")
    return


if __name__ == "__main__":
    app.run()
