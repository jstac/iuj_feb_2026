# Schelling Model Workshop

[![Build & Publish](https://github.com/QuantEcon/iuj_feb_2026/actions/workflows/publish.yml/badge.svg)](https://github.com/QuantEcon/iuj_feb_2026/actions/workflows/publish.yml)

A workshop studying the Schelling segregation model with accelerated implementations using Numba and JAX.

## Overview

This repository contains lecture materials for a workshop on the Schelling segregation model, demonstrating various computational approaches:

1. **Segregation Background** - Introduction to the segregation problem
2. **Schelling Model** - The classic Schelling model implementation
3. **NumPy Implementation** - Vectorized implementation using NumPy
4. **JAX Implementation** - GPU-accelerated implementation using JAX
5. **JAX Parallel** - Parallel computing with JAX

## Building the Book

To build the Jupyter Book locally:

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate quantecon

# Build the book
cd lectures
jupyter-book build .
```

The HTML output will be in `lectures/_build/html/`.

## Requirements

- Python 3.13
- Anaconda
- GPU support (optional, for JAX acceleration)

See [environment.yml](environment.yml) for the full list of dependencies.

## Source

These lectures are developed by [QuantEcon](https://quantecon.org/).

## License

This work is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).
