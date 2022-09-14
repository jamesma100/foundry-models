# Foundry Models

Experimental machine learning models to predict properties of organic molecules.

Created by myself (James Ma) and Sahas Gelli over the summer of 2021 under the supervision of Professor Dane Morgan.

## Installation

These instructions are only for running the KRR/SOAP model. 

Creating a [conda](https://docs.conda.io/en/latest/) environment and installing all the dependencies there is highly recommended. Some packages may need to be installed via [pip](https://pip.pypa.io/en/stable/#).
```bash
conda create -c conda-forge -n foundry rdkit
```

To activate the environment, run
```bash
conda activate foundry
```

If this doesn't work, try
```bash
cd [anaconda folder]/bin
source activate foundry
```

For Windows users:
```bash
activate foundry
```

Installing packages:
```bash
pip install numpy
conda install -c conda-forge dscribe
conda install -c conda-forge ase
pip install -U scikit-learn
conda install -c conda-forge matplotlib
```

## Usage

Run Jupyter notebook:
```bash
cd foundry-models/models/notebooks
jupyter notebook
```

## Credits

Feature Optimization for Atomistic Machine Learning Yields A Data-Driven
Michael J. Willatt, F ́elix Musil, Michele Ceriotti
Laboratory of Computational Science and Modeling
