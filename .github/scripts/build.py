"""
Build script for marimo notebooks.

This script exports marimo notebooks to HTML/WebAssembly format and generates
an index.html file that lists all the notebooks. It handles both regular notebooks
(from the notebooks/ directory) and apps (from the apps/ directory).

The script can be run from the command line with optional arguments:
    uv run .github/scripts/build.py [--output-dir OUTPUT_DIR]

The exported files will be placed in the specified output directory (default: _site).
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jinja2==3.1.3",
#     "fire==0.7.0",
#     "loguru==0.7.0"
# ]
# ///

import subprocess
from typing import List, Union
from pathlib import Path

import jinja2
import fire

from loguru import logger

GITHUB_REPO = "FPAO-CC"


def _adapt_to_wasm(notebook_path: Path, output_dir: Path):
    block_to_insert = """    # For online support with WASM and Pyodide ===================
    import micropip

    async def install_requirements():
        # Read requirements from remote
        requirements = [
            "plotly",
            "pandas",
            "polars",
            "pyarrow",
            "scipy",
            "pymdown-extensions>=10.15,<11",
        ]
        # Add local or remote .whl
        wheel_path = str(
            mo.notebook_location() / "public" / "fpao_cc-0.0.1-py3-none-any.whl"
        )
        
        requirements.append(wheel_path)

        await micropip.install(requirements)

    await install_requirements()

    # ===========================================================
"""

    for nb in list(notebook_path.glob("*.py")):
        with open(nb, "r") as f:
            lines = f.readlines()

        new_lines = []

        for line in lines:
            modified_line = line.replace(".py", ".html")

            modified_line = modified_line.replace("/?file=", f"/{GITHUB_REPO}/")
            new_lines.append(modified_line)
            if "import marimo as mo" in line:
                new_lines.append(block_to_insert)

        with open(nb, "w") as f:
            f.writelines(new_lines)

    with open(notebook_path / "core" / "_defaults.py", "r") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        modified_line = line.replace(".py", ".html")
        new_lines.append(modified_line)

    with open(notebook_path / "core" / "_defaults.py", "w") as f:
        f.writelines(new_lines)


def _export_html_wasm(
    notebook_path: Path, output_dir: Path, as_app: bool = False
) -> bool:
    """Export a single marimo notebook to HTML/WebAssembly format.

    This function takes a marimo notebook (.py file) and exports it to HTML/WebAssembly format.
    If as_app is True, the notebook is exported in "run" mode with code hidden, suitable for
    applications. Otherwise, it's exported in "edit" mode, suitable for interactive notebooks.

    Args:
        notebook_path (Path): Path to the marimo notebook (.py file) to export
        output_dir (Path): Directory where the exported HTML file will be saved
        as_app (bool, optional): Whether to export as an app (run mode) or notebook (edit mode).
                                Defaults to False.

    Returns:
        bool: True if export succeeded, False otherwise
    """

    # Convert .py extension to .html for the output file
    output_path: Path = notebook_path.with_suffix(".html")

    output_path = output_path.relative_to("notebooks")

    # Base command for marimo export
    cmd: List[str] = ["uvx", "marimo", "export", "html-wasm", "--sandbox"]

    # Configure export mode based on whether it's an app or a notebook
    if as_app:
        logger.info(f"Exporting {notebook_path} to {output_path} as app")
        cmd.extend(
            ["--mode", "run", "--no-show-code"]
        )  # Apps run in "run" mode with hidden code
    else:
        logger.info(f"Exporting {notebook_path} to {output_path} as notebook")
        cmd.extend(["--mode", "edit"])  # Notebooks run in "edit" mode

    try:
        # Create full output path and ensure directory exists
        output_file: Path = output_dir / notebook_path.with_suffix(".html")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Add notebook path and output file to command
        cmd.extend([str(notebook_path), "-o", str(output_file)])

        # Run marimo export command
        logger.debug(f"Running command: {cmd}")
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Successfully exported {notebook_path}")
        return True
    except subprocess.CalledProcessError as e:
        # Handle marimo export errors
        logger.error(f"Error exporting {notebook_path}:")
        logger.error(f"Command output: {e.stderr}")
        return False
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error exporting {notebook_path}: {e}")
        return False


def _export(folder: Path, output_dir: Path, as_app: bool = False) -> List[dict]:
    """Export all marimo notebooks in a folder to HTML/WebAssembly format.

    This function finds all Python files in the specified folder and exports them
    to HTML/WebAssembly format using the export_html_wasm function. It returns a
    list of dictionaries containing the data needed for the template.

    Args:
        folder (Path): Path to the folder containing marimo notebooks
        output_dir (Path): Directory where the exported HTML files will be saved
        as_app (bool, optional): Whether to export as apps (run mode) or notebooks (edit mode).

    Returns:
        List[dict]: List of dictionaries with "display_name" and "html_path" for each notebook
    """

    # Check if the folder exists
    if not folder.exists():
        logger.warning(f"Directory not found: {folder}")
        return []

    _adapt_to_wasm(folder, output_dir)
    # Find all Python files recursively in the folder
    notebooks = list(folder.glob("*.py"))
    logger.debug(f"Found {len(notebooks)} Python files in {folder}")

    # Exit if no notebooks were found
    if not notebooks:
        logger.warning(f"No notebooks found in {folder}!")
        return []

    # For each successfully exported notebook, add its data to the notebook_data list
    notebook_data = [
        {
            "display_name": (nb.stem.replace("_", " ").title()),
            "html_path": str(nb.with_suffix(".html")),
        }
        for nb in notebooks
        if _export_html_wasm(nb, output_dir, as_app=as_app)
    ]

    logger.info(
        f"Successfully exported {len(notebook_data)} out of {len(notebooks)} files from {folder}"
    )
    return notebook_data


def main(
    output_dir: Union[str, Path] = "_site",
) -> None:
    """Main function to export marimo notebooks.

    This function:
    1. Parses command line arguments
    2. Exports all marimo notebooks in the 'notebooks'

    Command line arguments:
        --output-dir: Directory where the exported files will be saved (default: _site)

    Returns:
        None
    """
    logger.info("Starting marimo build process")

    # Convert output_dir explicitly to Path (not done by fire)
    output_dir: Path = Path(output_dir)
    logger.info(f"Output directory: {output_dir}")

    # Make sure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export apps from the apps/ directory
    apps_data = _export(Path("notebooks"), output_dir, as_app=True)

    # Exit if no notebooks or apps were found
    if not apps_data:
        logger.warning("No notebooks or apps found!")
        return

    logger.info(f"Build completed successfully. Output directory: {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
