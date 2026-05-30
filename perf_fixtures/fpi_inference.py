"""Factorized fixed-point state inference over a three-factor generative model.

Runs `run_factorized_fpi` to infer beliefs about three hidden-state factors
(sizes 8, 6, 4) from a single one-hot observation in each of three modalities
(sizes 5, 8, 4). The observation likelihood A is ragged: the three modalities
depend on factor sets [0,1], [1,2] and [0,1,2] respectively. Starting from a
uniform prior over each factor, the variational message-passing loop iterates
16 times to a fixed point, returning the posterior marginal over every factor.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from pymdp import utils
from pymdp.algos import run_factorized_fpi


def fixture():
    key = jr.PRNGKey(0)
    a_key, b_key = jr.split(key, 2)

    num_obs = [5, 8, 4]
    num_states = [8, 6, 4]
    num_controls = [3, 4, 2]

    A_dependencies = [[0, 1], [1, 2], [0, 1, 2]]
    B_dependencies = [[0], [0, 1], [1, 2]]

    A = utils.random_A_array(a_key, num_obs, num_states, A_dependencies=A_dependencies)
    # Observations as one-hot distributions
    obs = [jnp.eye(num_obs[m])[0] for m in range(len(num_obs))]
    prior = [jnp.ones(num_states[f]) / num_states[f] for f in range(len(num_states))]

    def fn(A, obs, prior):
        return run_factorized_fpi(A, obs, prior, A_dependencies, num_iter=16)

    return fn, (A, obs, prior), {}


tolerance = {"atol": 1e-5, "rtol": 1e-5}
