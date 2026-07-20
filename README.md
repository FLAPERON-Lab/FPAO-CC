[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18850911.svg)](https://doi.org/10.5281/zenodo.18850911)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![marimo](https://img.shields.io/badge/marimo-darkgreen?)](https://marimo.io/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](LICENSE.Apache-2.0)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

# Constrained Aircraft Performance Optimization Notebooks (CAPO-NBs)

A collection of [_marimo_](https://marimo.io/) notebooks in Python, to strengthen the connection between applied Constrained Aircraft Performance Optimization and the calculus fundamentals of constrained optimization for multi-variate functions.

The notebooks provide:
- A concise presentation of the mathematical concepts of constrained optimization for multi-variate functions, with a focus on the Lagrange multipliers method.
- Presentation of classic CAPO problems using standard engineering derivations
- Structured mathematical analysis of the problem formulation: independent variables, dependent variables, objectives, constraints, domain, boundaries.
- Re-formulation of the CAPO problem using formal mathematical symbology
- Recall of relevant theorems to apply a solution method and/or predict the existence and properties of solutions
- Interactive visualizations to observe the influence of common flight parameters (altitude, speed, weight) on flight performance metrics (climb rate, cruise speed, …), and highlight the mathematical role of physical and operational constraints (stall, limit load factor, procedures).

A set of functions and classes provides a backend for the notebooks, to allow for a more structured and modular approach to the problem formulation and separate the didactic software implementation from the administrative one.

While the source code that is visible in the notebooks should be intended to have didactic values as well, end-users should also be able to consult them only for the sake of their content.

## Getting started 
The notebooks are available for end-users in various forms: 
- interactive web apps, accessible at this link: https://flaperon-lab.github.io/CAPO-NBs/Homepage.html
- static documents, downloadable via the top-right menu in each webpage
- editable notebooks to be deployed locally (see instructions below)

### Local deployment
The notebooks can be run locally using the _marimo_ server. This allows for interactive visualizations and the ability to modify the code directly.

1. Follow the [installation instructions](CONTRIBUTING.md#installation-instructions) to set up this repository on your local machine.
2. Open a terminal and navigate to the notebooks directory:
    ```bash
    cd your/path/to/CAPO-NBs/
    cd notebooks
    ```
3. Run the following command to start the _marimo_ server:
    ```bash
    uv run marimo run <notebook_name>.py
    ```

## Authors

The content of this repository has been developed by:
1. Carmine Varriale, Technische Universiteit Delft (![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png) [0000-0001-7419-992X](https://orcid.org/0000-0001-7419-992X))

2. Federico Angioni, Technische Universiteit Delft (![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png) [0009-0005-7028-4840](https://orcid.org/0009-0005-7028-4840))

3. Maarten van Hoven, Technische Universiteit Delft

## License

The contents of the `notebooks/` folder are licensed under an <a href="https://opensource.org/licenses/Apache-2.0" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Apache License 2.0</a>. 
See the [Apache-2.0.txt](LICENSE.Apache-2.0.txt) file for details.

The contents of the `data/` folder are licensed under a <a href="https://creativecommons.org/licenses/by/4.0/ " target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY 4.0</a> license.
See the [CC-BY-4.0.txt](LICENSES.CC-BY-4.0.txt) file for details.

### Copyright notice

Technische Universiteit Delft hereby disclaims all copyright interest in the work "Constrained Aircraft Performance Optimization Notebooks (CAPO-NBs)". 
It is a collection of Marimo notebooks to explain and visualize the theory of aircraft performance optimization using interactive visualizations in Python, written by the Author(s).
Henri Werij, Dean of Faculty of Aerospace Engineering, Technische Universiteit Delft.

See also copyright notices in the individual files, provided following REUSE standard (https://reuse.software)

&copy; 2026, Carmine Varriale, Federico Angioni, Maarten van Hoven

## Cite this repository
If you use this software, please cite it as below or check out the [CITATION.cff](CITATION.cff) file.

**BIBTEX**

```
@software{Varriale2026,
    author = {Varriale, Carmine and Angioni, Federico and {van Hoven}, Maarten},
    license = {Apache-2.0},
    month = apr,
    title = {{Constrained Aircraft Performance Optimization Notebooks (CAPO-NBs)}},
    doi = {https://doi.org/10.5281/zenodo.18850911},
    url = {https://github.com/FLAPERON-Lab/CAPO-NBs},
    version = {1.2},
    year = {2026}
}
```

**APA**

_Varriale, C., Angioni, F., & van Hoven, M.B. (2026). Constrained Aircraft Performance Optimization Notebooks (CAPO-NBs) (Version 1.1) [Computer software]. DOI: 10.5281/zenodo.18850911, URL: https://github.com/FLAPERON-Lab/CAPO-NBs_

## Contributors

You are welcome to contribute to this project by submitting issues or pull requests on GitHub: https://github.com/FLAPERON-Lab/CAPO-NBs

Suggestions, corrections and additions will contribute to the improvement of this open educational resource.

Please consult our [Contributing Guidelines](CONTRIBUTING.md) for more information on how to contribute.


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

