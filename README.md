# Learning cell fate landscapes from spatial transcriptomics using Fused Gromov-Wasserstein

[![codecov](https://codecov.io/gh/cantinilab/stories/graph/badge.svg?token=5DWDYPAUYI)](https://codecov.io/gh/cantinilab/stories)
[![Tests](https://github.com/cantinilab/stories/actions/workflows/main.yml/badge.svg)](https://github.com/cantinilab/stories/actions/workflows/main.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/stories/badge/?version=latest)](https://stories.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/stories-jax?color=blue)](https://img.shields.io/pypi/v/stories-jax?color=blue)

STORIES is a trajectory inference method capable of learning a causal model of cellular differentiation from spatial transcriptomics through time using Fused Gromov-Wasserstein. STORIES learns a potential function that defines each cell's stage of differentiation and allows one to predict the evolution of cells at future time points. In addition, STORIES uncovers possible driver genes and transcriptional regulators of cellular differentiation.

[Read the preprint here](https://www.biorxiv.org/content/10.1101/2024.07.26.605241v1) and [the documentation here](https://stories.rtfd.io)!

STORIES is based on the Scverse ecosystem, making it easy to interface with existing tools for single-cell analysis such as Scanpy and CellRank. In addition, STORIES benefits from the JAX ecosystem for deep learning and OT computation, enabling the fast handling of large datasets.

![introductory figure](docs/_static/fig1.png)

## Install the package

- STORIES is implemented as a Python package seamlessly integrated within the scverse ecosystem. It relies on JAX for fast GPU computations and JIT compilation, and OTT for Optimal Transport computations.
- **System requirements**: Python >= 3.10. Continuously tested on Ubuntu 22.04 LTS. Installation time <1mn. Benefits from CUDA or TPU for large datasets (on a GPU A40 and for 800k cells, training time ~ 30mn).
- The verified modern NVIDIA GPU installation path documented below was tested on Ubuntu 24.04 with Python 3.11 on an NVIDIA GeForce RTX 5070 Ti Laptop GPU (Blackwell, `sm_100`) using NVIDIA driver `580.126.09`, which supports up to CUDA 13.0.

### via PyPI (recommended)

For a reproducible install, start from a clean Python environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install "numpy<2"
```

#### CPU-only install

```bash
pip install stories-jax
```

#### NVIDIA GPU install

On the tested machine, the reproducible path used a conda environment plus the `jax[cuda12]` wheel family, even though the NVIDIA driver supports CUDA 13.0.

```bash
conda create --name stories_test python=3.11 -y
conda activate stories_test
conda install -c conda-forge cuda-version=12.6 -y
pip install "numpy==1.26.4" "pandas==2.0.3"
pip install stories-jax
pip install "jax==0.6.2" "jaxlib==0.6.2" \
    "jax-cuda12-plugin[with-cuda]==0.6.2" \
    "jax-cuda12-pjrt==0.6.2"
```

The explicit `cuda-version=12.6`, `numpy==1.26.4`, and `pandas==2.0.3` pins are currently recommended for reproducibility. In a fresh environment, `pip` may otherwise resolve combinations that either break import-time compatibility in the broader scverse stack or fail to initialize the CUDA backend correctly.

On newer NVIDIA GPUs such as Ada Lovelace RTX 4xxx and Blackwell RTX 5xxx, installing a recent CUDA-enabled JAX build is important because older JAX wheels may bundle a `ptxas` that cannot target those architectures. For the currently validated STORIES setup, use the `cuda12` JAX wheel family together with the conda `cuda-version=12.6` package. The current dependency set should stay below JAX 0.7 until the `ott/equinox` stack used by STORIES is upgraded to a compatible release. For use in Google Colab, restart the kernel after installing the CUDA-enabled JAX wheel.

#### Verify the install

```bash
python -c "import stories; print('stories import ok')"
python -c "import jax; print(jax.devices())"
```

On a CUDA machine, the second command should list a `CudaDevice`.

### via GitHub (development version)

For a reproducible development install on a machine with a recent NVIDIA driver, use the same validated package set:

```bash
git clone git@github.com:cantinilab/stories.git
cd stories
conda create --name stories_dev python=3.11 -y
conda activate stories_dev
conda install -c conda-forge cuda-version=12.6 -y
pip install "numpy==1.26.4" "pandas==2.0.3"
pip install -e .
pip install "jax==0.6.2" "jaxlib==0.6.2" \
    "jax-cuda12-plugin[with-cuda]==0.6.2" \
    "jax-cuda12-pjrt==0.6.2"
python -c "import stories; print('stories import ok')"
python -c "import jax; print(jax.devices())"
```

If you are installing the development version without a GPU, replace the CUDA JAX command above with:

```bash
pip install "numpy==1.26.4" "pandas==2.0.3"
pip install -e .
```

## Getting started

STORIES takes as an input an AnnData object, where omics information and spatial coordinates are stored in `obsm`, and `obs` contains time information, and optionally a proliferation weight. Visit the **Getting started** and **API** sections for tutorials and documentation.
