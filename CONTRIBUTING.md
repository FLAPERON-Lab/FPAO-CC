# Contributing

The following instructions and guidelines are meant for those brave and motivated persons who want to actively contribute to the development of the material.

We welcome any kind of contribution to our software, from simple comment or question to a full fledged [pull request](https://help.github.com/articles/about-pull-requests/). 

A contribution can be one of the following cases:

1. you have a question;
1. you think you may have found a mistake;
1. you want to make some kind of change to the code base (e.g. to fix a mistake, to add a new section or derivation);

The sections below outline the steps in each case, and provide installation instructions.

## You have a question

1. use the search functionality [here](https://github.com/FLAPERON-Lab/CAPO-NBs/issues) to see if someone already filed the same issue;
1. if your issue search did not yield any relevant results, make a new issue;
1. apply the "Question" label; apply other labels when relevant.

## You think you may have found a mistake

1. use the search functionality [here](https://github.com/FLAPERON-Lab/CAPO-NBs/issues) to see if someone already filed the same issue;
1. if your issue search did not yield any relevant results, make a new issue, making sure to provide enough information to the rest of the community to understand the cause and context of the problem.
1. apply relevant labels to the newly created issue.

## You want to make some kind of change to the code base

1. **IMPORTANT**: announce your plan to the rest of the community *before you start working*. This announcement should be in the form of a (new) issue;
1. **IMPORTANT**: wait until some kind of consensus is reached about your idea being a good idea;
1. if needed, fork the repository to your own Github profile and create your own feature branch off of the latest commit in `main`. Your feature branch should be named: `<issue-nr>-<name-of-feature>`. While working on your feature branch, make sure to stay up to date with the `main` branch by pulling in changes, possibly from the 'upstream' repository (follow the instructions [here](https://help.github.com/articles/configuring-a-remote-for-a-fork/) and [here](https://help.github.com/articles/syncing-a-fork/));
1. [push](https://docs.github.com/en/get-started/using-git/pushing-commits-to-a-remote-repository) your feature branch to (your fork of) the CAPO-NBs repository on GitHub;
1. create a pull request to the `main` branch, e.g. following the instructions [here](https://help.github.com/articles/creating-a-pull-request/).

Before you request to merge, you should verify that:
1. the (new) notebooks run without errors, that all visualizations render correctly
2. the (new) data files are only in the `data\` folder
3. there is a footer in each notebook with the proper links, following how the other notebooks work
4. you are using the file naming convention you used: `NameDirectory/NameNotebook.py`)
   
Please add your copyright and selected license to your contributions, following the [REUSE standard](https://reuse.software).
You can do so by adding before any contribution the following text:
```
# SPDX-FileCopyrightText: year name(s)
#
# SPDX-License-Identifier: spdx-license
```
You can install `reuse` (for example with `uv pip install reuse`) and run it to make sure the copyright notices are compliant with the standard (`reuse lint`)


## Guidelines for contributing

1. Follow the [installation instructions](#installation-instructions) below to set up this repository on your local machine.

2. Familiarize yourself with the guide on how to [quick start](https://docs.marimo.io/getting_started/quickstart/) with _marimo_ notebooks and their [key concepts](https://docs.marimo.io/getting_started/key_concepts/).

3. Create and modify _marimo_ notebooks primarily from their native environment. You can do so by running the command:

    ```bash
    cd ./notebooks
    # Either one of the following:
    uv run marimo edit <notebook_name>.py # example: uv run marimo edit Scope.py -> opens the specified notebook 
    # or
    uv run marimo edit # -> opens a marimo server giving an overview of all notebooks
    ```
    
> [!TIP]
> Use the VSCode _marimo_ extension _exclusively_ for minor, cosmetic changes, or to automate search-and-replace (or similar) tasks.

### Managing the aircraft databases
You can view and manipulate the `data\*.csv` files by opening them in the [VSCode](https://code.visualstudio.com/) editor using the [Rainbow CSV](https://marketplace.visualstudio.com/items?itemName=mechatroner.rainbow-csv) and the [Edit CSV](https://marketplace.visualstudio.com/items?itemName=janisdd.vscode-edit-csv) extensions. 

> [!CAUTION]
> Please DO NOT use Microsoft Excel, as it is not able to preserve the correct format of the documents.

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
    # Using SSH (recommended by GitHub):
    git clone git@github.com:FLAPERON-Lab/CAPO-NBs.git
    # Using HTTPS:
    git clone https://github.com/FLAPERON-Lab/CAPO-NBs.git
    ```

3. **Navigate** to the cloned directory:
    ```bash
    cd your/path/to/CAPO-NBs
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

    If the notebooks run and no errors are shown in the bottom-left of the marimo server, everything was installed correctly.
