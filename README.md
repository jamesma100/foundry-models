# Foundry Models

This repository contains experimental machine learning models that predict properties of organic molecules. The objective is to build and deploy different models on an open-source Foundry cloud platform where researchers can run them with ease, as well as upload their own models. Our models are adapted from learning methods by many top researchers and we often use slightly different featurizers/models for the sake of simplicity and ease of implementation. Currently, we are only working on and maintaining the KRR/SOAP model, which can be found in models/notebooks/qm9_krr_soap.ipynb.

This project is done by myself (James Ma) and Sahas Gelli over the summer of 2021 under the supervision of Professor Dane Morgan.

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
