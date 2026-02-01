---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Schelling Model with JAX: Parallel Algorithm

## Overview

In the {doc}`previous lecture <schelling_jax>`, we translated the Schelling model
to JAX, keeping the same sequential algorithm. While that demonstrated JAX syntax
and concepts, it didn't leverage JAX's main strength: **parallel computation**.

In this lecture, we redesign the algorithm to be **fully parallelizable**. The
key insight is to have all agents evaluate multiple candidate locations
simultaneously, then update all locations at once.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, vmap, lax
import time
```

## The Parallel Algorithm

Our modified algorithm works as follows:

```{prf:algorithm} Parallel Schelling Update
:label: parallel_schelling

**Input:** Agent locations, types, random key, number of candidates K

**Output:** Updated locations

1. Generate K random candidate locations for each agent
2. For each agent, check happiness at current location and all K candidates
3. Each agent moves to the first candidate that makes them happy (if any)
4. Repeat until no one moves or max iterations reached

```

The key to efficiency: **no while loops inside vectorized operations**. Instead,
we generate a fixed number of candidates and check them all in parallel.

## Parameters

```{code-cell} ipython3
num_of_type_0 = 1000    # number of agents of type 0 (orange)
num_of_type_1 = 1000    # number of agents of type 1 (green)
n = num_of_type_0 + num_of_type_1  # total number of agents
k = 10                  # number of agents regarded as neighbors
require_same_type = 5   # want >= require_same_type neighbors of the same type
num_candidates = 10     # candidate locations per agent per iteration
```

## Initialization

```{code-cell} ipython3
def initialize_state(key):
    """Initialize agent locations and types."""
    locations = random.uniform(key, shape=(n, 2))
    types = jnp.array([0] * num_of_type_0 + [1] * num_of_type_1)
    return locations, types
```

## Checking Happiness at a Location

First, a function to check if an agent would be happy at a given location:

```{code-cell} ipython3
@jit
def is_happy_at_location(agent_idx, loc, locations, types):
    """
    Check if agent would be happy at the given location.

    Uses current locations of all OTHER agents.
    """
    agent_type = types[agent_idx]

    # Compute distances from loc to all agents
    diff = loc - locations
    distances = jnp.sqrt(jnp.sum(diff ** 2, axis=1))

    # Exclude self
    distances = distances.at[agent_idx].set(jnp.inf)

    # Find k nearest neighbors
    neighbor_indices = jnp.argsort(distances)[:k]
    neighbor_types = types[neighbor_indices]
    num_same = jnp.sum(neighbor_types == agent_type)

    return num_same >= require_same_type
```

## Vectorized Happiness Checking

We need to check happiness for:
- Each agent at their current location
- Each agent at each of K candidate locations

We use nested `vmap` to vectorize these operations:

```{code-cell} ipython3
# Check one agent at multiple locations
# vmap over locations (axis 0), agent_idx fixed
is_happy_at_locations = vmap(is_happy_at_location, in_axes=(None, 0, None, None))

# Check all agents, each at multiple locations
# vmap over agent_idx and their candidate locations
is_happy_all_agents = vmap(is_happy_at_locations, in_axes=(0, 0, None, None))
```

## The Parallel Update Step

```{code-cell} ipython3
@jit
def parallel_update(locations, types, key):
    """
    Perform one parallel update step.

    Each agent generates num_candidates random locations and moves to the
    first one that makes them happy (if any).

    Returns
    -------
    new_locations : array (n, 2)
        Updated locations
    num_moved : int
        Number of agents who moved
    key : PRNGKey
        Updated random key
    """
    num_agents = locations.shape[0]

    # Check current happiness for all agents
    all_indices = jnp.arange(num_agents)
    current_happy = vmap(is_happy_at_location, in_axes=(0, 0, None, None))(
        all_indices, locations, locations, types
    )

    # Generate candidates: shape (n, num_candidates, 2)
    key, subkey = random.split(key)
    candidates = random.uniform(subkey, shape=(num_agents, num_candidates, 2))

    # Check happiness at all candidates for all agents: shape (n, num_candidates)
    happy_at_candidates = is_happy_all_agents(all_indices, candidates, locations, types)

    # For each agent, find the first happy candidate (if any)
    # Use argmax on boolean array - returns index of first True, or 0 if all False
    # We need to handle the case where no candidate is happy

    # Create mask: agent needs to move AND at least one candidate is happy
    needs_move = ~current_happy
    has_happy_candidate = jnp.any(happy_at_candidates, axis=1)
    will_move = needs_move & has_happy_candidate

    # Find first happy candidate for each agent
    # argmax returns first True index; if all False, returns 0
    first_happy_idx = jnp.argmax(happy_at_candidates, axis=1)

    # Get the selected candidate location for each agent
    selected_candidates = candidates[all_indices, first_happy_idx, :]

    # Update locations: move if will_move, otherwise stay
    new_locations = jnp.where(
        will_move[:, None],
        selected_candidates,
        locations
    )

    num_moved = jnp.sum(will_move)

    return new_locations, num_moved, key
```

## Fully Compiled Simulation Loop

```{code-cell} ipython3
@jit
def run_until_convergence(locations, types, key, max_iter):
    """
    Run the simulation until convergence, fully compiled.
    """
    def cond_fn(state):
        """Continue while someone moved and under max iterations."""
        locations, key, iteration, num_moved = state
        return (num_moved > 0) & (iteration < max_iter)

    def body_fn(state):
        """Perform one iteration."""
        locations, key, iteration, num_moved = state
        locations, num_moved, key = parallel_update(locations, types, key)
        return locations, key, iteration + 1, num_moved

    # Run first iteration to initialize
    locations, num_moved, key = parallel_update(locations, types, key)
    init_state = (locations, key, 1, num_moved)

    # Run until convergence
    final_locations, final_key, iterations, _ = lax.while_loop(
        cond_fn, body_fn, init_state
    )

    return final_locations, iterations, final_key
```

## Visualization

```{code-cell} ipython3
def plot_distribution(locations, types, title):
    """Plot the distribution of agents."""
    locations_np = np.asarray(locations)
    types_np = np.asarray(types)

    fig, ax = plt.subplots()
    plot_args = {'markersize': 6, 'alpha': 0.8}
    ax.set_facecolor('azure')
    colors = 'orange', 'green'
    for agent_type, color in zip((0, 1), colors):
        idx = (types_np == agent_type)
        ax.plot(locations_np[idx, 0],
                locations_np[idx, 1],
                'o',
                markerfacecolor=color,
                **plot_args)
    ax.set_title(title)
    plt.show()
```

## The Simulation

```{code-cell} ipython3
def run_simulation(max_iter=1000, seed=1234):
    """
    Run the parallel Schelling simulation.
    """
    key = random.PRNGKey(seed)
    key, init_key = random.split(key)
    locations, types = initialize_state(init_key)

    plot_distribution(locations, types, 'Initial distribution')

    # Run fully compiled simulation
    start_time = time.time()
    final_locations, iterations, key = run_until_convergence(
        locations, types, key, max_iter
    )
    # Block until computation complete for accurate timing
    final_locations.block_until_ready()
    elapsed = time.time() - start_time

    plot_distribution(final_locations, types, f'After {iterations} iterations')

    print(f'Converged in {elapsed:.2f} seconds after {iterations} iterations.')

    return final_locations, types
```

## Warming Up JAX

JAX compiles functions on first call and **recompiles when array shapes change**.
We warm up with the actual problem size:

```{code-cell} ipython3
# Warm up with ACTUAL problem size
key = random.PRNGKey(42)
key, init_key = random.split(key)
warmup_locations, warmup_types = initialize_state(init_key)

# Compile by running once
_, _, _ = run_until_convergence(warmup_locations, warmup_types, key, max_iter=3)

print("JAX functions compiled and ready!")
```

## Results

```{code-cell} ipython3
locations, types = run_simulation()
```

## Performance Comparison

Let's time the simulation:

```{code-cell} ipython3
# Fresh run with timing
key = random.PRNGKey(1234)
key, init_key = random.split(key)
locations, types = initialize_state(init_key)

start_time = time.time()
final_locations, iterations, _ = run_until_convergence(
    locations, types, key, max_iter=1000
)
final_locations.block_until_ready()
elapsed = time.time() - start_time

print(f"Converged in {iterations} iterations")
print(f"Total time: {elapsed:.3f} seconds")
print(f"Time per iteration: {elapsed/int(iterations)*1000:.2f} ms")
```

## Why This Approach Works

The key insight is avoiding `while` loops inside `vmap`. When you `vmap` over a
`while_loop`, JAX must run all parallel loops for the maximum number of
iterations needed by any single element. This is wasteful if iterations vary.

Instead, we:
1. Generate a **fixed number** of candidates (e.g., 50)
2. Check **all candidates** for **all agents** in one batched operation
3. Use `argmax` to find the first happy candidate

This gives predictable cost: O(n × num_candidates) happiness checks per iteration,
all fully parallelizable.

## Trade-offs

**Advantages:**
- No while loops inside vmap — predictable parallel cost
- Fully vectorized happiness checking
- Entire simulation loop is JIT-compiled

**Considerations:**
- More candidates = better chance of finding happy spot, but more computation
- If num_candidates is too small, convergence may be slow
- Memory scales as O(n × num_candidates)

## Summary

The key techniques in this parallel implementation:

1. **Fixed candidate count**: Generate K candidates per agent upfront, avoiding
   variable-length while loops inside vectorized operations.

2. **Nested vmap**: Use `vmap(vmap(...))` to check all agents × all candidates
   in a single batched operation.

3. **Vectorized selection**: Use `argmax` to find the first happy candidate for
   each agent, avoiding explicit loops.

4. **Full JIT compilation**: The entire simulation loop uses `lax.while_loop`.

5. **Proper benchmarking**: Use `block_until_ready()` for accurate timing, and
   warm up at actual problem size.
