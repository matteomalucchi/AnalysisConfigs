# AnalysisConfigs

Repository containing analysis configurations for PocketCoffea

## Setup

### Fast installation with micromamba

```bash
# Clone PocketCoffea
git clone git@github.com:PocketCoffea/PocketCoffea.git

# Clone AnalysisConfigs
git git@github.com:matteomalucchi/AnalysisConfigs.git

# Create a local environment and install the packages
cd AnalysisConfigs
micromamba env create -f pocket-coffea-environment.yml
micromamba activate pocket-coffea-env
```

### General installation

The first step is installing the main `PocketCoffea` package in your python environment.

Please have a look at the [Installation guide](https://pocketcoffea.readthedocs.io/en/latest/installation.html).

The `configs` package has been created to separate the core of the framework from all the necessary configuration files
and customization code needed the different analyses. The configuration is structured as a python package to make easier
the import of customization code into the framework configuration and also to make the sharing of analysis code easier.

Once you have a `PocketCoffea` local installation, you can install the `configs` and `utils` package with:

```bash
cd AnalysisConfigs
pip install -e .
```

This will install the `configs` package in editable mode.

## HH4b analysis

The instruction to run the HH4b analysis are in the [HH4b README](./configs/HH4b_common/README.md).

## MC Truth corrections for pT regression derivation

The instruction to run the derivation of the MC Truth corrections for pT regression are in the [jme README](./configs/jme/README.md).

## MET Type-1 correction analysis

The instruction to run the analysis for the MET Type-1 correction are in the [MET_studies README](./configs/MET_studies/README.md).