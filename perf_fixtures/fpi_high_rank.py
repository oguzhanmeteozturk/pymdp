"""Factorized state inference with a high-rank observation modality.

A generative model with five hidden-state factors (sizes 6, 5, 4, 5, 4) and
six observation modalities. One "global observation" modality depends on all
five factors at once, while each remaining modality depends on a single factor,
so per-factor evidence enters alongside the joint signal at every iteration.
Running `run_factorized_fpi` for 16 iterations from a uniform prior infers the
posterior marginal over each factor. The all-factor modality makes this a
high-rank inference problem where the joint observation couples every factor.
"""

import jax.numpy as jnp
import jax.random as jr
from pymdp import utils
from pymdp.algos import run_factorized_fpi


def fixture():
    key = jr.PRNGKey(0)
    a_key, _ = jr.split(key, 2)

    # 5 hidden-state factors with mixed sizes.
    num_states = [6, 5, 4, 5, 4]
    # 6 modalities: one "global observation" over all factors + one per factor.
    num_obs = [8] + [4] * len(num_states)
    A_dependencies = [list(range(len(num_states)))] + [[f] for f in range(len(num_states))]

    A = utils.random_A_array(a_key, num_obs, num_states, A_dependencies=A_dependencies)
    obs = [jnp.eye(num_obs[m])[0] for m in range(len(num_obs))]
    prior = [jnp.ones(num_states[f]) / num_states[f] for f in range(len(num_states))]

    def fn(A, obs, prior):
        return run_factorized_fpi(A, obs, prior, A_dependencies, num_iter=16)

    return fn, (A, obs, prior), {}


tolerance = {"atol": 1e-5, "rtol": 1e-5}
