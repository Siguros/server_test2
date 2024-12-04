______________________________________________________________________

<div align="center">

# NMDL ML Codebase

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

[![Interrogate](./interrogate_badge.svg)](https://github.com/econchick/interrogate)
[![codecov](https://codecov.io/gh/nmdlkg/ml/graph/badge.svg?token=PGDGVA7A3J)](https://codecov.io/gh/nmdlkg/ml)
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)

</div>

## Description

Encapsulates all the boilerplate code for training and evaluating a model.

## Installation

#### A. Pip

```bash
# clone project
git clone https://github.com/nmdlkg/ml
cd ml

# [OPTIONAL] create conda environment
conda create -n ml python=3.10
conda activate ml

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### B. Conda

```bash
# clone project
git clone https://github.com/nmdlkg/ml
cd ml

# create common conda environment and install dependencies
CONDA_OVERRIDE_CUDA="11.8" conda env create -f environment.yaml
# or for cpu only
# conda env create -f environment-cpu.yaml

# activate conda environment
conda activate ml
```

#### Setup as package

```bash
pip install -e .
```

#### [Optional] eqprop

```bash
#you can create environment for specific project with additional packages
conda env update -n ml -f src/_eqprop/environment.yaml
# or
pip install -r src/_eqprop/requirements.txt

```

```bash
conda activate ml
src/core/eqprop/build_proxsuite_w_openmp.sh
```

#### [Optional] SPICE

```bash
conda install -c conda-forge pyspice

# or

pip install PySpice

# and eventually

pyspice-post-installation --check-install
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Override config group (optimizer&scheduler)

```bash
python src/train.py model/optimizer=sgd model/scheduler=steplr

```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

You can see default/configurable parameters from command line.
See [link](https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/debugging/) for more details

```bash
python src/train.py --cfg job
```

## Project Structure

The directory structure of new project looks like this:

```
├── .github                   <- Github Actions workflows
│
├── configs                   <- Hydra configs
│   ├── experiment                 <- Experiment configs
├── data                   <- Project data
│
├── logs                   <- Logs generated by hydra and lightning loggers
│
├── notebooks              <- Jupyter notebooks. Naming convention is a number (for ordering),
│                             the creator's initials, and a short `-` delimited description,
│                             e.g. `1.0-lhr-initial-data-exploration.ipynb`.
│
├── scripts                <- Shell scripts
│
├── src                    <- Source code
│   ├── core                     <- Core code
|   │   ├── eqprop                   <- EqProp implementation
|   │   ├── aihwkit                  <- AIHWKit implementation
|   │   ├── ...
|   ├── project-A               <- Project A code
│   ├── ...
│   │
│   ├── data                     <- Data scripts
│   ├── models                   <- Model scripts
│   ├── utils                    <- Utility scripts
│   │
│   ├── eval.py                  <- Run evaluation
│   └── train.py                 <- Run training
│
├── tests                  <- Tests of any kind
│
├── .env.example              <- Example of file for storing private environment variables
├── .gitignore                <- List of files ignored by git
├── .pre-commit-config.yaml   <- Configuration of pre-commit hooks for code formatting
├── .project-root             <- File for inferring the position of project root directory
├── environment.yaml          <- File for installing conda environment
├── Makefile                  <- Makefile with commands like `make train` or `make test`
├── pants.toml                <- Pants build system configuration
├── pyproject.toml            <- Configuration options for testing and linting
├── requirements.txt          <- File for installing python dependencies
├── setup.py                  <- File for installing project as a package
└── README.md
```

## Development

### Pre-commit hooks

This project uses [pre-commit](https://pre-commit.com/) hooks for code formatting. To install them run:

```bash
pre-commit install
```

### Pants

This project uses [pants](https://www.pantsbuild.org/) for building and testing. To install it run:

```bash
curl --proto '=https' --tlsv1.2 -fsSL https://static.pantsbuild.org/setup/get-pants.sh | bash
```
