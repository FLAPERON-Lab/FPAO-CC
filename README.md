# Flight Performance Analysis and Optimization: from Calculus to Computers

A collection of [_marimo_ notebooks](https://marimo.io/) in Python, to strengthen the connection between applied Flight Performance Analysis and Optimization (FPAO) and the calculus fundamentals of constrained optimization for multi-variate functions.

The notebooks provide:
- A concise presentation of the mathematical concepts of constrained optimization for multi-variate functions, with a focus on the Lagrange multipliers method.
- Presentation of classic FPAO problems using standard engineering derivations
- Structured mathematical analysis of the problem formulation: independent variables, dependent variables, objectives, constraints, domain, boundaries.
- Re-formulation of the FPAO problem using formal mathematical symbology
- Recall of relevant theorems to apply a solution method and/or predict the existence and properties of solutions
- Interactive visualizations to observe the influence of common flight parameters (altitude, speed, weight) on flight performance metrics (climb rate, cruise speed, …), and highlight the mathematical role of physical and operational constraints (stall, limit load factor, procedures).

A set of functions and classes provides a backend for the notebooks, to allow for a more structured and modular approach to the problem formulation and separate the didactic software implementation from the administrative one.

## Getting started 
The notebooks are available for end-users in various forms: 
- editable notebooks to be deployed locally
- static documents, downloadable via link
- interactive web apps, accessible via URL

While the source code that is visible in the notebooks should be intended to have didactic values as well, end-users should also be able to consult them only for the sake of their content.

### Local deployment
The notebooks can be run locally using the _marimo_ server. This allows for interactive visualizations and the ability to modify the code directly.

1. Follow the [installation instructions](CONTRIBUTING.md#installation-instructions) to set up this repository on your local machine.
2. Open a terminal and navigate to the notebooks directory:
    ```bash
    cd your/path/to/FlightPerfCalculus/
    cd notebooks
    ```
3. Run the following command to start the _marimo_ server:
    ```bash
    marimo run Scope.py
    ```

## Licensing

### Authors

The content of this repository has been developed by:
1. Carmine Varriale, Technische Universiteit Delft (![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png) [0000-0001-7419-992X](https://orcid.org/0000-0001-7419-992X))

2. Federico Angioni, Technische Universiteit Delft

3. Maarten van Hoven, Technische Universiteit Delft

### License

The contents of the folder `\notebooks\*.py` are licensed under an <a href="https://opensource.org/licenses/Apache-2.0" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Apache License 2.0</a>. 
See the [Apache-2.0.txt](LICENSES/Apache-2.0.txt) file for details.

The contents of the folder `\notebooks\public\*.csv` are licensed under a <a href="https://creativecommons.org/licenses/by/4.0/ " target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY 4.0</a> license.
See the [CC-BY-4.0.txt](LICENSES/CC-BY-4.0.txt) file for details.

The contents of the folders `\output` are licensed under a <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0</a> license. See the [CC-BY-NC-SA-4.0.txt](LICENSES/CC-BY-NC-SA-4.0.txt) file for details.

### Copyright notice
Technische Universiteit Delft hereby disclaims all copyright interest in the work "Flight Performance Analysis and Optimization: from Calculus to Computers (FPAO-CC)". 
It is a collection of Marimo notebooks to explain and visualize the theory of aircraft performance optimization using interactive visualizations in Python, written by the Author(s).
Henri Werij, Dean of Faculty of Aerospace Engineering, Technische Universiteit Delft.

© 2025, Carmine Varriale, Federico Angioni, Maarten van Hoven

## Contributors

Contributions are welcome and encouraged in any form through Issues and Pull Requests on GitHub. 

The main repository for this project is located at [https://github.com/CarmVarriale/FlightPerfCalculus](https://github.com/CarmVarriale/FlightPerfCalculus)

Please consult our [Contributing Guidelines](CONTRIBUTING.md) for more information.


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

