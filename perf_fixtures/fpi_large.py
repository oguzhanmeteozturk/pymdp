"""Factorized fixed-point state inference over a larger three-factor model.

Runs `run_factorized_fpi` on a generative model with three hidden-state factors
(sizes 32, 24, 16) and three observation modalities (sizes 16, 32, 8). The
ragged observation likelihood A ties the modalities to factor sets [0,1],
[1,2] and [0,1,2]. Given one one-hot observation per modality and a uniform
prior over each factor, the variational message-passing loop runs 16 iterations
to a fixed point and returns the posterior marginal over every factor. The
larger state and observation dimensions reflect a richer multi-factor model.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from pymdp import utils
from pymdp.algos import run_factorized_fpi


def fixture():
    key = jr.PRNGKey(0)
    a_key, b_key = jr.split(key, 2)

    num_obs = [16, 32, 8]
    num_states = [32, 24, 16]
    num_controls = [4, 6, 3]

    A_dependencies = [[0, 1], [1, 2], [0, 1, 2]]
    B_dependencies = [[0], [0, 1], [1, 2]]

    A = utils.random_A_array(a_key, num_obs, num_states, A_dependencies=A_dependencies)
    obs = [jnp.eye(num_obs[m])[0] for m in range(len(num_obs))]
    prior = [jnp.ones(num_states[f]) / num_states[f] for f in range(len(num_states))]

    def fn(A, obs, prior):
        return run_factorized_fpi(A, obs, prior, A_dependencies, num_iter=16)

    return fn, (A, obs, prior), {}


tolerance = {"atol": 1e-5, "rtol": 1e-5}
