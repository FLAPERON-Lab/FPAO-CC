# Contributing

The following instructions and guidelines are meant for those brave and motivated persons who want to actively contribute to the development of the material.

For major changes, please open an issue first to discuss what you would like to change.

## Installation instructions

This repository relies on `uv`, an extremely fast Python project manager.

1. **Install `uv`** in your machine. 
    
    On Windows, open a PowerShell window and run

    ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

    On a MacOS or Linux, run in your terminal window:

    ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

   Alternatively, you can install `uv` directly from PyPI using `pip`:

   ```bash
   pip install uv
   ``` 

2. **Clone the repository** from GitHub:
    ```bash
    # Using HTTPS:
    git clone https://github.com/FLAPERON-Lab/FPAO-CC.git
    # Using SSH (recommended by GitHub):
    git clone git@github.com:FLAPERON-Lab/FPAO-CC.git
    ```

3. **Navigate** to the cloned directory:
    ```bash
    cd your/path/to/FPAO-CC
    ```

4. **Create a virtual environment** and sync it with `uv`:
    ```bash
    uv sync
    ```
    This will install the required Python version and all necessary dependencies into the virtual environment.

5. **Verify the installation** by running the following command:
    ```bash
    uv run marimo tutorial intro
    ```

## Guidelines for contributing

1. Follow the [instructions](#installation-instructions) to set up this repository on your local machine.

2. Familiarize yourself with the guide on how to [quick start](https://docs.marimo.io/getting_started/quickstart/) with _marimo_ notebooks and their [key concepts](https://docs.marimo.io/getting_started/key_concepts/).

3. Create and modify _marimo_ notebooks primarily from their native environment. You can do so by running the command:

    ```bash
    cd ./notebooks
    # Either one of the following:
    uv run marimo edit <notebook_name>.py # example: marimo edit Scope.py -> opens the specified notebook 
    # or
    uv run marimo edit # -> opens a marimo server giving an overview of all notebooks
    ```
    
    > [!TIP]
    > Use the VSCode _marimo_ extension _exclusively_ for minor, cosmetic changes, or to automate search-and-replace (or similar) tasks.

### Managing the aircraft databases
You can view and manipulate the `AircraftDB_*.ssv` files by opening them in the [VSCode](https://code.visualstudio.com/) editor using the [Rainbow CSV](https://marketplace.visualstudio.com/items?itemName=mechatroner.rainbow-csv) and the [Edit CSV](https://marketplace.visualstudio.com/items?itemName=janisdd.vscode-edit-csv) extensions. 

> [!CAUTION]
> Please DO NOT use Microsoft Excel, as it is not able to preserve the correct format of the documents.

