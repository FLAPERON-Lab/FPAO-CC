# Flight Performance Analysis and Optimization: from Calculus to Computers

A collection of [_marimo_ notebooks](https://marimo.io/) in Python, to strengthen the connection between applied Flight Performance Analysis and Optimization (FPAO) and the calculus fundamentals of constrained optimization for multi-variate functions.

The notebooks provide:
- A concise presentation of the mathematical concepts of constrained optimization for multi-variate functions, with a focus on the Lagrange multipliers method.
- Presentation of classic FPAO problems using standard engineering derivations
- Structured mathematical analysis of the problem formulation: independent variables, dependent variables, objectives, constraints, domain, boundaries.
- Re-formulation of the FPAO problem using formal mathematical symbology
- Recall of relevant theorems to apply a solution method and/or predict the existence and properties of solutions
- Interactive visualizations to observe the influence of common flight parameters (altitude, speed, weight) on flight performance metrics (climb rate, cruise speed, …), and highlight the mathematical role of physical and operational constraints (stall, limit load factor, procedures).

A set of Object-Oriented classes provides a backend for the notebooks, to allow for a more structured and modular approach to the problem formulation and separate the didactic software implementation from the administrative one.

## Scope 
The notebooks are available for end-users in various forms: 
- editable notebooks themselves, in this GitHub repository
- static documents, downloadable via link
- interactive web apps, accessible via URL

While the source code that is visible in the notebooks should be intended to have didactic values as well, end-users should also be able to consult them only for the sake of their content.

## Aircraft data and models
The notebooks allow calculating and visualizing the flight performance of aircraft of different types and categories.

In all cases, the International Standard Atmosphere (ISA) model is used to calculate the air temperature, pressure and density at a given altitude.

Aircraft data is stored in two types of databases, built around different sets of assumptions and models.

### Standard models
The data for these models is stored in the `AircraftDB_Standard_Jets.ssv` and `AircraftDB_Standard_Props.ssv` semi-column separated files.

These models rely on to the following assumptions:
- simplified jet-aircraft with 
    - available trust independent of airspeed: $\mathrm{d}T_{\mathrm{a}}/\mathrm{d}V=0$
    - constant Thrust-Specific Fuel Consumption (TSFC): $c_{T}=\mathit{const}$

- simplified propeller aircraft with 
    - available power independent of airspeed: $\mathrm{d}P_{\mathrm{a}}/\mathrm{d}V=0$ 
    - constant Power-Specific Fuel Consumption (PSFC) $c_{P}=\mathit{const}$

- thrust/power lapse with altitude based on a power law such as 
$$\frac{T_a(h)}{T_a(h=0)} = \left[\frac{\rho(h)}{\rho(h=0)}\right]^\beta 
\quad \text{or} \quad 
\frac{P_a(h)}{P_a(h=0)} = \left[\frac{\rho(h)}{\rho(h=0)}\right]^\beta$$


- parabolic drag polar: $$C_D = C_{D0} + K C_L^2$$

- linear lift polar until maximum lift coefficient: $$C_L = C_{L_0} + C_{L_\alpha} \alpha \quad \text{if } C_L <= C_{L_\mathrm{max}}$$

### Custom models
The data for these models is stored in the `AircraftDB_Custom_Jets.ssv` and `AircraftDB_Custom_Props.ssv` semi-column separated files.

In these cases, the models for $C_D$, $T_a$ or $P_a$, $cT$ or $cP$, and optionally $C_L$, are provided in the form of tabular data. 
The relevant data is stored in aircraft-specific folders, which are indicated in the `AircraftDB_Custom.ssv` file itself.

### Managing the aircraft databases
You can view and manipulate the `AircraftDB_*.ssv` files by opening them in the [VSCode](https://code.visualstudio.com/) editor using the [Rainbow CSV](https://marketplace.visualstudio.com/items?itemName=mechatroner.rainbow-csv) and the [Edit CSV](https://marketplace.visualstudio.com/items?itemName=janisdd.vscode-edit-csv) extensions. 

Please DO NOT to use Microsoft Excel, as it is not able to preserve the correct format of the documents.

## Licensing

### Authors

The content of this repository has been developed by:
1. Carmine Varriale  ![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png) [0000-0001-7419-992X](https://orcid.org/0000-0001-7419-992X), Technische Universiteit Delft

2. Federico Angioni, Technische Universiteit Delft

3. Maarten van Hoven, Technische Universiteit Delft

### License

The contents of the folder `\notebooks` are licensed under an <a href="https://opensource.org/licenses/Apache-2.0" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Apache License 2.0</a>. 
See the [Apache-2.0.txt](LICENSES/Apache-2.0.txt) file for details.

The contents of the folders `\data` and `\output` are licensed under a <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0</a> license. See the [CC-BY-NC-SA-4.0.txt](LICENSES/CC-BY-NC-SA-4.0.txt) file for details.

### Copyright notice:
Technische Universiteit Delft hereby disclaims all copyright interest in the work "Flight Performance Analysis and Optimization: from Calculus to Computers". 
It is a collection of Marimo notebooks to explain and visualize the theory of aircraft performance optimization using interactive visualizations in Python, written by the Author(s).
Henri Werij, Dean of Faculty of Aerospace Engineering, Technische Universiteit Delft.

© 2025, Carmine Varriale, Federico Angioni, Maarten van Hoven


## Contribution
Contributions are welcome and encouraged in any form through Issues and Pull Requests on GitHub. For major changes, please open an issue first to discuss what you would like to change.

The main repository is located at [https://github.com/CarmVarriale/FlightPerfCalculus](https://github.com/CarmVarriale/FlightPerfCalculus)

The following instructions and guidelines are meant for those brave and motivated persons that want to actively contribute to the development of the material.

### Guidelines for contributing

1. Follow the [instructions](#installation-instructions) to set up this folder on your local machine.

1. Familiarize yourself with the guide on how to [quick start](https://docs.marimo.io/getting_started/quickstart/) with _marimo_ notebooks and their [key concepts](https://docs.marimo.io/getting_started/key_concepts/).

1. Create and modify _marimo_ notebooks primarily from the Command Line Interface (CLI) in a terminal. Use the VSCode _marimo_ extension _exclusively_ for minor, cosmetic, changes

### Installation instructions

1. Install [Python 3.12 or higher](https://www.python.org/downloads/) and optionally add it to your PATH.

2. Clone the repository from Github: 

    ```git clone https://github.com/CarmVarriale/FlightPerfCalculus.git```

3. Navigate to the cloned directory:

    ```cd your/path/to/FlightPerfCalculus```

4. Create a virtual environment (optional but recommended):

    ```.\.venv\Scripts\python -m venv .venv```

5. Activate the virtual environment:
   - On Windows: ```.\.venv\Scripts\activate.bat```
   - On macOS/Linux: ```source .venv/bin/activate```

6. Install the required dependencies:
    
    ```.venv\Scripts\python -m pip install -r requirements.txt```

7. Test the correct installation of the _marimo_ dependencies by running the following command:

    ```.venv\Scripts\python -m marimo tutorial intro```

## Contributors

<table>
<tr>
    <td align="center" style="word-wrap: break-word; width: 150.0; height: 150.0">
        <a href=https://github.com/CarmVarriale>
            <img src=https://github.com/CarmVarriale.png width="100;"  style="border-radius:50%;align-items:center;justify-content:center;overflow:hidden;padding-top:10px" alt=Carmine Varriale/>
            <br />
            <sub style="font-size:14px"><b>Carmine Varriale</b></sub>
            <br />
            <sub><a href=https://orcid.org/0000-0001-7419-992X style="font-size:12px">0000-0001-7419-992X</a></sub>
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

