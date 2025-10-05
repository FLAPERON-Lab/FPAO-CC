import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    # Initialization code that runs before all other cells
    import marimo as mo

    # Import dependencies
    from core import _defaults

    # Set local/online filepath
    _defaults.FILEURL = _defaults.get_url()

    # Plotly dark mode template
    _defaults.set_plotly_template()

    # Data directory
    data_dir = str(mo.notebook_location() / "public" / "AircraftDB_Standard.csv")
    return (mo,)


@app.cell
def _():
    # Set navbar on the right
    _defaults.set_sidebar()
    return


@app.cell
def _(mo):
    mo.md('<h1 style="font-size: 100px">FPAO-CC</h1>').center()
    return


@app.cell
def _(mo):
    mo.md(
        """## Flight Performance Analysis and Optimization: from Calculus to Computers"""
    ).center()
    return


@app.cell
def _(mo):
    mo.md(r"""# The open education stimulation fund (OESF)""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The Open Education Stimulation Fund is at the base of this marimo notebooks collection. The fund promotes open education to TU Delft's bachelor or master programs. 

    The goal of the Open Education Stimulation Fund is the one of empowering educators and students, with new technologies, such as interactive textbooks, welcoming the growing emphasis of students' autonomy in educational processes. 

    You can find more informations on the [OESF](https://www.tudelft.nl/en/open-science/articles-tu-delft/call-for-proposals-2025) page by TUDelft.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""# About the project""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    This is a collection of marimo notebooks in Python, to strengthen the connection between applied Flight Performance Analysis and Optimization (FPAO) and the calculus fundamentals of constrained optimization for multi-variate functions.

    The notebooks provide:

    - A concise presentation of the mathematical concepts of constrained optimization for multi-variate functions, with a focus on the Lagrange multipliers method.
    - Presentation of classic FPAO problems using standard engineering derivations
    - Structured mathematical analysis of the problem formulation: independent variables, dependent variables, objectives, constraints, domain, boundaries.
    - Re-formulation of the FPAO problem using formal mathematical symbology
    - Recall of relevant theorems to apply a solution method and/or predict the existence and properties of solutions
    - Interactive visualizations to observe the influence of common flight parameters (altitude, speed, weight) on flight performance metrics (climb rate, cruise speed, …), and highlight the mathematical role of physical and operational constraints (stall, limit load factor, procedures).

    A set of functions and classes provides a backend for the notebooks, to allow for a more structured and modular approach to the problem formulation and separate the didactic software implementation from the administrative one.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""# About the authors""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""

    <table>
    <tr>
        <td align="center" style="word-wrap: break-word; width: 150.0; height: 150.0">
            <a href=https://github.com/CarmVarriale>
                <img src=https://github.com/CarmVarriale.png width="100;"  style="border-radius:50%;align-items:center;justify-content:center;overflow:hidden;padding-top:10px" alt=Carmine Varriale/>
                <br />
                <sub style="font-size:14px"><b>Carmine Varriale</b></sub>
            </a>
        </td>
        <td align="center" style="word-wrap: break-word; width: 150.0; height: 150.0">
            <a href=https://github.com/federicoangioni>
                <img src=https://github.com/federicoangioni.png width="100;"  style="border-radius:50%;align-items:center;justify-content:center;overflow:hidden;padding-top:10px" alt=Carmine Varriale/>
                <br />
                <sub style="font-size:14px"><b>Federico Angioni</b></sub>
            </a>
        </td>
    </tr>
    </table>

    """
    ).center()
    return


@app.cell
def _(mo):
    mo.md(r"""### [Dr. Carmine Varriale](https://www.tudelft.nl/staff/c.varriale/)""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Carmine Varriale is Assistant Professor at the Flight Performance and Propulsion section of TU Delft. His research interests cover the fields of flight mechanics, direct lift control, and multi-body dynamics modelling and simulation. He is co-instructor of the BSc course "Flight and Orbital Mechanics", responsible instructor for the online course "Aircraft Performance: Physics and Simulation" targeted at professional education, and for the MOOC "Sustainable Aviation" targeted at the general public.

    He obtained his PhD in 2022, with a thesis on the Flight Mechanics and Performance of Direct Lift Control. He performed his research work in the framework of the European Union Horizon2020 program, for which he has been leader of the flight dynamics activities within the PARSIFAL project. His research work has brought him to investigate the dynamic behavior of innovative aircraft with redundant and/or interacting control effectors.

    He graduated at the University of Naples Federico II in 2017, cum laude,  honorable mentions and scholarships for "excellence and promptness" throughout the studies. He carried out part of his MSc thesis project during an internship at the Fraunhofer Institute for Wind Energy Systems in Oldenburg, Germany. During this project, he investigated flight encounters of light aircraft with wind turbine wakes, with the purpose of providing safety guidelines for the construction of wind farms in proximity of small airports.

    He was born on August 11th, 1992 in Naples, Italy. In his free time, he enjoys hiking, being by the sea, taking photos of natural landscapes, playing chess and other board games.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Federico Angioni""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""Federico Angioni is an undergraduate honours student in Aerospace Engineering at TU Delft, pursuing a minor in Computational Science and Engineering at the Faculty of Electrical Engineering, Mathematics and Computer Science. His interests lie at the intersection of optimal control, dynamic modeling, and, more broadly, scientific computing"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""He is currently involved in providing the visualisations and the layout of the FPAO-CC notebook collection, enhancing the narrative written by Dr. Varriale to increase the students' understanding of the connection between calculus and flight performance optimization."""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""He is developing a dynamic model of a Flapping-Wing Micro Air Vehicle (FWMAV) to enable time-optimal flight, where the ultra-light, bioinspired drone rapidly navigates through gateways. Instead of a conventional PID controller, a neural network trained via Reinforcement Learning provides direct motor commands, eliminating intermediate filters that would otherwise slow actuation."""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""He was born in March, 2004 in Casarsa della Delizia, Italy. In his free time, he enjoys playing the guitar, going for runs and learning about fields outside aerospace."""
    )
    return


@app.cell
def _(mo):
    mo.md(r"""# Getting Started""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    In each notebook there is a navigation bar on the left, from there, you can explore the entire collection of marimo notebooks, and learn about FPAO! Moreover, at the bottom of each notebook, you will find a small navigation bar, allowing you to switch with the previous and next notebooks. 

    If you ever want to go back to this page, click "FPAO-CC", on the top of the left sidebar, just as if it was the homepage of a website!

    You can start exploring by selected the notebook "Scope" on the left, or in the footer navbar below.
    """
    )
    return


@app.cell
def _():
    _defaults.nav_footer(after_file="Scope.py", after_title="Scope")
    return


if __name__ == "__main__":
    app.run()
