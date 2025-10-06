#!/bin/bash -i
module load miniforge3

# Check if conda is installed
check_conda_installed() {
    if command -v conda &> /dev/null; then
        echo "Conda is already installed."
    else
        echo "Conda is not installed. Installing Miniconda..."
        install_miniconda
    fi
}

# Install Miniconda
install_miniconda() {
    echo "Downloading Miniconda installer..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest.sh
    echo "Running Miniconda installer..."
    bash Miniconda3-latest.sh -b -p $HOME/miniconda
    echo "Initializing Miniconda..."
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda init
    source ~/.bashrc
}

# Check if conda environment exists
check_conda_env() {
    if conda env list | grep -q "GeoValuator"; then
        echo "Conda environment 'GeoValuator' already exists."
    else
        echo "Conda environment 'GeoValuator' does not exist. Creating environment..."
        conda create -n GeoValuator python=3.10.13 -y
    fi
}

# Main script execution
check_conda_installed
check_conda_env

set -e

# Initialize Conda for the current shell
eval "$(conda shell.bash hook)"

echo "Activating conda environment 'GeoValuator'..."
conda activate GeoValuator

echo $CONDA_DEFAULT_ENV

# Install Packages
conda install -y -c conda-forge -c pytorch -c defaults \
    pip \
    typer \
    loguru \
    tqdm \
    ipython \
    jupyterlab \
    matplotlib \
    notebook \
    numpy \
    pandas \
    scikit-learn \
    ruff \
    yaml \
    pytorch \
    torchvision \
    torchaudio

pip install \
    python-dotenv \
    osmnx \
    utm \
    folium \
    requests \
    shapely \
    rtree \
    contextily \
    timm

pip install -e .

echo "GeoValuator environment setup completed successfully!"