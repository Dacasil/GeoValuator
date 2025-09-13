# GeoValuator

ML Handson Final Project

## Using the Makefile

This project provides a Makefile to automate common development tasks. Below is a list of all available commands.


- **First-time setup:**
    ```bash
    make create_environment
    ```

-   **To update the `environment.yml`:**
    ```bash
    make requirements
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
    ```bash
    make help
    ```

## Project Organization
