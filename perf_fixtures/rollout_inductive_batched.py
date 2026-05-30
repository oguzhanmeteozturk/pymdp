"""Closed-loop active-inference rollout with goal-directed inductive planning.

Runs a batch of four agents through a ten-step perception-action loop in a
matched environment. The generative model has three hidden-state factors
(sizes 32, 24, 16) and three observation modalities (sizes 16, 32, 8) with
ragged observation dependencies ([0,1], [1,2], [0,1,2]) and factor-local
transitions. The agent plans three steps ahead over 12^3 = 1728 policies under
inductive planning: an I matrix built from per-factor goal-state indicators
guides it toward target states. Peaked log-preferences C give each modality an
explicit goal, so expected free energy combines utility with state information
gain rather than reducing to pure exploration. Each step runs FPI state
inference, scores policies, acts, and steps the environment.
"""

import jax.numpy as jnp
import jax.random as jr
from pymdp import utils
from pymdp.agent import Agent
from pymdp.control import generate_I_matrix
from pymdp.envs.env import PymdpEnv
from pymdp.envs.rollout import rollout


def fixture():
    key = jr.PRNGKey(0)
    a_key, b_key, run_key = jr.split(key, 3)

    num_obs = [16, 32, 8]
    num_states = [32, 24, 16]
    num_controls = [2, 3, 2]
    batch_size = 4
    num_timesteps = 10
    policy_len = 3

    A_dependencies = [[0, 1], [1, 2], [0, 1, 2]]
    B_dependencies = [[0], [1], [2]]  # factor-local — required by generate_I_matrix (per-factor backward reachability)

    A = utils.random_A_array(a_key, num_obs, num_states, A_dependencies=A_dependencies)
    B = utils.random_B_array(b_key, num_states, num_controls, B_dependencies=B_dependencies)
    D = [jnp.ones(n) / n for n in num_states]

    # Peaked log-preferences over the last observation in each modality.
    C = [jnp.eye(n_o)[-1] * 4.0 - 2.0 for n_o in num_obs]

    # H/I for inductive inference — goal is the last hidden state per factor.
    H = [jnp.eye(n)[-1] for n in num_states]
    I = generate_I_matrix(H, B, threshold=1.0 / 16, depth=policy_len)

    def _broadcast(arr_list):
        return [jnp.broadcast_to(jnp.array(arr), (batch_size,) + arr.shape) for arr in arr_list]

    A_b = _broadcast(A)
    B_b = _broadcast(B)
    C_b = _broadcast(C)
    D_b = _broadcast(D)
    I_b = _broadcast(I)

    agent = Agent(
        A_b, B_b,
        C=C_b,
        D=D_b,
        I=I_b,
        A_dependencies=A_dependencies,
        B_dependencies=B_dependencies,
        num_controls=num_controls,
        batch_size=batch_size,
        policy_len=policy_len,
        use_inductive=True,
        inductive_epsilon=1e-3,
    )

    env = PymdpEnv(A=A, B=B, D=D, A_dependencies=A_dependencies, B_dependencies=B_dependencies)
    env_params = {"A": A_b, "B": B_b, "D": D_b}

    def fn(agent, env_params, run_key):
        return rollout(agent, env, num_timesteps, run_key, env_params=env_params)

    return fn, (agent, env_params, run_key), {}


tolerance = {"atol": 1e-3, "rtol": 1e-3}
