"""Closed-loop active-inference rollout over ten timesteps.

Runs a batch of four agents through a complete perception-action loop in a
matched generative-process environment: at each step the agent infers its
beliefs over three hidden-state factors (sizes 32, 24, 16) from three
observation modalities (sizes 16, 32, 8), evaluates its one-step policies by
expected free energy, samples an action, and the environment returns the next
observation. The observation likelihood is ragged ([0,1], [1,2], [0,1,2]) and
transitions range from factor-local to pairwise. The loop runs for ten
timesteps and returns the trajectory of states, observations and beliefs.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from pymdp import utils
from pymdp.agent import Agent
from pymdp.envs.env import PymdpEnv
from pymdp.envs.rollout import rollout


def fixture():
    key = jr.PRNGKey(0)
    a_key, b_key, d_key, run_key = jr.split(key, 4)

    num_obs = [16, 32, 8]
    num_states = [32, 24, 16]
    num_controls = [2, 3, 2]
    batch_size = 4
    num_timesteps = 10

    A_dependencies = [[0, 1], [1, 2], [0, 1, 2]]
    B_dependencies = [[0], [0, 1], [1, 2]]

    A = utils.random_A_array(a_key, num_obs, num_states, A_dependencies=A_dependencies)
    B = utils.random_B_array(b_key, num_states, num_controls, B_dependencies=B_dependencies)
    D = [jnp.ones(n) / n for n in num_states]

    def _broadcast(arr_list):
        return [jnp.broadcast_to(jnp.array(arr), (batch_size,) + arr.shape) for arr in arr_list]

    A_b = _broadcast(A)
    B_b = _broadcast(B)
    D_b = _broadcast(D)

    agent = Agent(
        A_b, B_b,
        D=D_b,
        A_dependencies=A_dependencies,
        B_dependencies=B_dependencies,
        num_controls=num_controls,
        batch_size=batch_size,
        policy_len=1,
    )

    env = PymdpEnv(A_dependencies=A_dependencies, B_dependencies=B_dependencies)
    env_params = {"A": A_b, "B": B_b, "D": D_b}

    def fn(agent, env_params, run_key):
        return rollout(agent, env, num_timesteps, run_key, env_params=env_params)

    return fn, (agent, env_params, run_key), {}


tolerance = {"atol": 1e-3, "rtol": 1e-3}
