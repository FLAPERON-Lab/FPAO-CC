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
    mo.md(
        r"""
    This collection of marimo notebooks in Python is aimed at strengthening the connection between the calculus fundamentals of constrained optimization for multi-variate functions and classic applied problems of Flight Performance Analysis and Optimization (FPAO).

    The notebooks provide:

    - A concise presentation of the mathematical concepts of constrained optimization for multi-variate functions, including methods based on Lagrange multipliers and Karush-Kuhn-Tucker conditions.
    - Presentation of classic FPAO problems using standard engineering derivations
    - Structured mathematical analysis of the problem formulation: independent variables, dependent variables, objectives, constraints, domain, boundaries.
    - Re-formulation of the FPAO problem using formal mathematical symbology
    - Recall of relevant theorems to apply a solution method and/or predict the existence and properties of solutions
    - Interactive visualizations to observe the influence of common flight parameters (altitude, speed, weight) on flight performance metrics (climb rate, cruise speed, …), and highlight the mathematical role of physical and operational constraints (stall, limit load factor, procedures).

    A set of functions and classes provides a backend for the notebooks, to modularize calculations and visualization and separate the software implementation from conceptual explanations.

    The backend is also available open-access on GitHub.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Getting Started

    Use the navigation bar on the left to explore the entire collection of marimo notebooks and learn how FPAO problems can be formulated in a formal and complete way using the power of calculus. 

    You will find a small navigation bar at bottom of each notebook, which allows you to move to the next or previous notebook just as if you were flipping through the pages of a book. 

    You can come back to this homepage by clicking on "FPAO-CC" on the top of the left of the sidebar, just as if it was the homepage of a website.

    You can start exploring by selected the notebook "Scope" on the left, or in the footer navbar below.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # About the authors
    """)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    <style>
    # body {
    #   background-color: #f0f0f3; /* soft gray background */
    #   font-family: Arial, sans-serif;
    # }

    .profile-container {
      display: flex;
      justify-content: center;
      gap: 40px;
      margin-top: 50px;
      flex-wrap: wrap;
    }

    .profile-card {
      text-align: center;
      # background: #ffffff;
      border-radius: 15px;
      padding: 20px;
      width: 150px;
      box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    .profile-card:hover {
      transform: translateY(-5px);
      box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
    }

    .profile-card a {
      text-decoration: none;
      color: inherit;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    .profile-card img {
      width: 100px;
      border-radius: 50%;
      margin-bottom: 10px;
      display: block;
    }

    .profile-card b {
      color: #FFFFFF;
      font-size: 14px;
      text-align: center;
    }
    </style>

    <div class="profile-container">
      <div class="profile-card">
        <a href="https://github.com/CarmVarriale" target="_blank">
          <img src="https://github.com/CarmVarriale.png" alt="Carmine Varriale">
          <b>Carmine Varriale</b>
        </a>
      </div>

      <div class="profile-card">
        <a href="https://github.com/federicoangioni" target="_blank">
          <img src="https://github.com/federicoangioni.png" alt="Federico Angioni">
          <b>Federico Angioni</b>
        </a>
      </div>
    </div>


    """
    ).center()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Carmine Varriale
    Carmine Varriale is [Assistant Professor](https://www.tudelft.nl/en/staff/c.varriale/) at the Flight Performance and Propulsion section of TU Delft. 

    He is a passionate educator with research interests in the flight mechanics and performance of unconventional aircraft.
    His latest interests cover Knowledge-Based Flight Mechanics modelling and simulation techniques, Path performance optimization, and
    Performance optimization of aircraft with redundant and interacting control effectors.

    He is responsible instructor of the BSc course "Flight and Orbital Mechanics", the online course "[Aircraft Performance: Physics and Simulation](https://learningforlife.tudelft.nl/aircraft-performance-physics-and-simulation/)" targeted at professional education, and the MOOC "[Sustainable Aviation: The route to climate neutral aviation](https://learningforlife.tudelft.nl/sustainable-aviation-the-route-to-climate-neutral-aviation/)" targeted at the general public.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Federico Angioni

    Federico Angioni is an undergraduate honours student in Aerospace Engineering at TU Delft, pursuing a minor in Computational Science and Engineering at the Faculty of Electrical Engineering, Mathematics and Computer Science. His interests lie at the intersection of optimal control, dynamic modeling, and, more broadly, scientific computing

    He is currently involved in providing the visualisations and the layout of the FPAO-CC notebook collection, enhancing the narrative written by Dr. Varriale to increase the students' understanding of the connection between calculus and flight performance optimization.

    He is developing a dynamic model of a Flapping-Wing Micro Air Vehicle (FWMAV) to enable time-optimal flight, where the ultra-light, bioinspired drone rapidly navigates through gateways. Instead of a conventional PID controller, a neural network trained via Reinforcement Learning provides direct motor commands, eliminating intermediate filters that would otherwise slow actuation.

    He was born in March, 2004 in Casarsa della Delizia, Italy. In his free time, he enjoys playing the guitar, going for runs and learning about fields outside aerospace.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Funding

    This project has received funding from the TU Delft Open Education Stimulation Fund (OESF) 2025, under the project title “Flight Performance Analysis and Optimization: from Calculus to Computers”.

    The OESF promotes open education, starting from TU Delft's BSc and MSc programs. 

    The goal of the OESF is to empower educators and students with new technologies such as interactive textbooks, in order to grow emphasis on students' autonomy in educational processes. 

    You can find more informations on the [Open Education](https://www.tudelft.nl/en/open-science/about/projects/open-education) webpage by TUDelft.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Acknowledgments

    The authors would like to thank Dr. Maarten van Hoven for his precious feedback on aspects related to the didactics of calculus for aerospace engineering students.
    """
    )
    return


@app.cell
def _():
    _defaults.nav_footer(after_file="Scope.py", after_title="Scope")
    return


if __name__ == "__main__":
    app.run()
