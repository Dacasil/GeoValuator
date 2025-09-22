# GeoValuator

ML hands-on Final Project

## Installation
Step-by-step instructions on how to get the development environment running.

1. **Clone the repository**
```
git clone https://github.com/Dacasil/GeoValuator.git
```
2. **Install Miniconda if missing** (look [here](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer) for Miniconda installation details)

4. **Use the make commands**

## Using the Makefile

This project provides a Makefile to automate common development tasks. Below is a list of all available commands.


- **First-time setup:**
    ```bash
    make create_environment
    ```

-   **To update the `environment.yml`:**
    ```bash
    make requirements
    conda activate GeoValuator
    ```
-   **Runs the main data processing script to generate or update the project's datasets:**
    ```bash
    make data
    ```

-   **Automatically formats the source code and fixes linting errors using `ruff`:**
    ```bash
    make format
    ```
-   **Checks the source code for style and formatting errors using `ruff`:**
    ```bash
    make lint
    ```

-   **Deletes all compiled Python files and `__pycache__`:**
    ```bash
    make clean
    ```

-   **Displays this list of available commands and their descriptions:**
    ```
    make help
    ```

## Project Organization
```
├── LICENSE            <- MIT Open-source licence
├── Makefile           <- Makefile with convenience commands
├── README.md          
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks for development
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         GeoValuator and configuration for tools like flake
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials
│
├── reports            <- Generated analysis
│   └── figures        <- Generated graphics and figures
│
├── environment.yml   <- The environment.yml file for reproducing the analysis environment
│
└── GeoValuator   <- Source code for use in this project
    │
    ├── __init__.py 
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## License
This project is licensed under the MIT License.

## References

**2025 Machine Learning hands-on course Göttingen** \
*Lecturer: PD Matthias Schröter*


--------
