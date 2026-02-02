# Computational Methods for Simulation

## An Analysis of the Schelling Model

* Prepared for the [International University of Japan](https://www.iuj.ac.jp/)

* Author: [John Stachurski](https://johnstachurski.net/)


---

This workshop explores simulation for economic analysis.

It demonstrates how to accelerate simulations using modern Python tools.

## Simulation Methods

Simulation is a very important methodology for policy analysis

Examples:

* Using DSGE models to compare alternative monetary regimes by simulating inflation, output gaps, and welfare.

* Simulation of a pension reform that changes retirement ages, tracking poverty risk and fiscal costs over several decades.

* Climate simulations of carbon tax versus emissions trading to identify least‑cost mitigation paths.

* Agent‑based urban traffic models that test infrastructure changes, measuring vehicle speeds, delays, emissions, etc.

* Simulation of urban low‑emission zones that first predict changes in pollution and then translate these into health outcomes.


## Focus on the Schelling Model

This workshop focuses on Thomas Schelling's segregation model as our running example.

The Schelling segregation model attempts to understand the underlying causes of
high degrees of racial segregation in some areas.

It shows how mild individual preferences can lead to extreme aggregate outcomes. 

The Schelling model is a classic example of the value of simulation.

Simulations for the model were first performed using a checker board and a dice.

Now of course we do them on computers.

In these lectures we will run the basic simulations and then consider how to do
them more efficiently.

Computational efficiency is very important because it allows us to run more
realistic scenarios.

Our search for computational efficiency will lead us to working with modern
hardware and software tools developed for AI.

The key ideas can be applied in other settings.

## Overview

We study:

1. **The basic model** — Understanding the dynamics of racial segregation using Python classes
2. **NumPy implementation** — Rewriting the model with arrays and functions for clarity and speed
3. **JAX implementation** — Translating the model to JAX syntax and concepts
3. **Further parallelization** — How can we exploit modern parallel hardware (e.g., GPUs)
3. **Extensions** — Making the model more realistic and studying outcomes


```{tableofcontents}
```
