#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = GeoValuator
PYTHON_VERSION = 3.10.13
PYTHON_INTERPRETER = python


#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune
	
## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Available rules:"
	@echo ""
	@echo "requirements             Install Python dependencies"
	@echo "clean                    Delete all compiled Python files"
	@echo "lint                     Lint using ruff"
	@echo "format                   Format source code with ruff"
	@echo "create_environment       Set up Python interpreter environment"
	@echo "data                     Make dataset"
	@echo ""
	@echo "View the Makefile for detailed descriptions."
