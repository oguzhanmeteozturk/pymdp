"""Closed-loop active-inference rollout with online model learning.

Runs a batch of four agents through a ten-step perception-action loop in a
matched environment while they learn their generative model from experience.
The model has three hidden-state factors (sizes 32, 24, 16) and three
observation modalities (sizes 16, 32, 8) with ragged observation dependencies
([0,1], [1,2], [0,1,2]) and factor-local transitions, and the agent plans three
steps ahead under inductive planning toward per-factor goal states. With
`learn_A` and `learn_B` enabled in online mode, each step infers beliefs (FPI),
scores policies by utility and state information gain, acts, and then folds the
observed outcome back into the Dirichlet parameters of the observation and
transition models, so the agent's model is refined as the rollout proceeds.
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
    B_dependencies = [[0], [1], [2]]  # factor-local — required by generate_I_matrix

    A = utils.random_A_array(a_key, num_obs, num_states, A_dependencies=A_dependencies)
    B = utils.random_B_array(b_key, num_states, num_controls, B_dependencies=B_dependencies)
    D = [jnp.ones(n) / n for n in num_states]

    pA = utils.list_array_scaled([a.shape for a in A], scale=1.0)
    pB = utils.list_array_scaled([b.shape for b in B], scale=1.0)

    C = [jnp.eye(n_o)[-1] * 4.0 - 2.0 for n_o in num_obs]

    H = [jnp.eye(n)[-1] for n in num_states]
    I = generate_I_matrix(H, B, threshold=1.0 / 16, depth=policy_len)

    def _broadcast(arr_list):
        return [jnp.broadcast_to(jnp.array(arr), (batch_size,) + arr.shape) for arr in arr_list]

    A_b = _broadcast(A)
    B_b = _broadcast(B)
    C_b = _broadcast(C)
    D_b = _broadcast(D)
    I_b = _broadcast(I)
    pA_b = _broadcast(pA)
    pB_b = _broadcast(pB)

    agent = Agent(
        A_b, B_b,
        C=C_b,
        D=D_b,
        I=I_b,
        pA=pA_b,
        pB=pB_b,
        A_dependencies=A_dependencies,
        B_dependencies=B_dependencies,
        num_controls=num_controls,
        batch_size=batch_size,
        policy_len=policy_len,
        use_inductive=True,
        inductive_epsilon=1e-3,
        learn_A=True,
        learn_B=True,
        learning_mode="online",
    )

    env = PymdpEnv(A=A, B=B, D=D, A_dependencies=A_dependencies, B_dependencies=B_dependencies)
    env_params = {"A": A_b, "B": B_b, "D": D_b}

    def fn(agent, env_params, run_key):
        return rollout(agent, env, num_timesteps, run_key, env_params=env_params)

    return fn, (agent, env_params, run_key), {}


tolerance = {"atol": 1e-3, "rtol": 1e-3}
