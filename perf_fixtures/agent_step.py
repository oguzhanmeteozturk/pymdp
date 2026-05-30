"""One full perception-and-decision cycle for a batch of active-inference agents.

Builds an Agent over a three-factor (sizes 8, 6, 4), three-modality
(sizes 5, 8, 4) generative model with ragged observation dependencies
([0,1], [1,2], [0,1,2]) and factor-local-to-pairwise transition dependencies,
batched across four agents. Given one observation per modality, the agent first
runs FPI state inference from its prior D to obtain posterior beliefs, then
evaluates its policies, returning the policy posterior and the negative expected
free energy of each policy.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from pymdp import utils
from pymdp.agent import Agent


def fixture():
    key = jr.PRNGKey(42)
    a_key, b_key = jr.split(key, 2)

    num_obs = [5, 8, 4]
    num_states = [8, 6, 4]
    num_controls = [3, 4, 2]
    batch_size = 4

    A_dependencies = [[0, 1], [1, 2], [0, 1, 2]]
    B_dependencies = [[0], [0, 1], [1, 2]]

    A = utils.random_A_array(a_key, num_obs, num_states, A_dependencies=A_dependencies)
    B = utils.random_B_array(b_key, num_states, num_controls, B_dependencies=B_dependencies)

    def _broadcast(arr_list):
        return [jnp.broadcast_to(jnp.array(arr), (batch_size,) + arr.shape) for arr in arr_list]

    A_b = _broadcast(A)
    B_b = _broadcast(B)

    agent = Agent(
        A_b, B_b,
        A_dependencies=A_dependencies,
        B_dependencies=B_dependencies,
        num_controls=num_controls,
        batch_size=batch_size,
    )

    obs = [jnp.eye(num_obs[m])[jnp.zeros(batch_size, dtype=jnp.int32)] for m in range(len(num_obs))]

    def fn(agent, obs):
        qs = agent.infer_states(obs, empirical_prior=agent.D)
        qpi, neg_efe = agent.infer_policies(qs)
        return qpi, neg_efe

    return fn, (agent, obs), {}


tolerance = {"atol": 1e-4, "rtol": 1e-4}
