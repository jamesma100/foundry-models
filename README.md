# Foundry Models

This repository contains experimental machine learning models that predict properties of organic molecules. The objective is to build and deploy different models on an open-source Foundry cloud platform where researchers can run them with ease, as well as upload their own models. Our models are adapted from learning methods by many top researchers and we often use slightly different featurizers/models for the sake of simplicity and ease of implementation. This project is done by myself (James Ma) and Sahas Gelli over the summer of 2021 under the supervision of Professor Dane Morgan.

## Installation

Creating a [conda] environment and installing all the dependencies there is highly recommended. Some packages may need to be installed via [pip].
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
