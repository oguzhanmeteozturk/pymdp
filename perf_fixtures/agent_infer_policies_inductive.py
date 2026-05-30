"""Batched inductive policy evaluation through the Agent interface.

Builds an Agent over a three-factor (sizes 32, 24, 16), three-modality
(sizes 16, 32, 8) generative model planning three steps ahead, batched across
four agents. With 2, 3 and 2 controls per factor the agent enumerates
12^3 = 1728 policies. The observation likelihood is ragged ([0,1], [1,2],
[0,1,2]) and transitions are factor-local so inductive planning applies: an I
matrix built from per-factor goal-state indicators steers the agent toward its
target states. Given uniform posterior beliefs, `infer_policies` scores every
policy by expected free energy and returns the resulting policy posterior.
"""

import jax.numpy as jnp
import jax.random as jr
from pymdp import utils
from pymdp.agent import Agent
from pymdp.control import construct_policies, generate_I_matrix


def fixture():
    key = jr.PRNGKey(0)
    a_key, b_key = jr.split(key, 2)

    num_obs = [16, 32, 8]
    num_states = [32, 24, 16]
    num_controls = [2, 3, 2]  # 12^3 = 1728 policies
    policy_len = 3
    batch_size = 4

    A_dependencies = [[0, 1], [1, 2], [0, 1, 2]]
    B_dependencies = [[0], [1], [2]]  # factor-local — required by generate_I_matrix (per-factor backward reachability)

    A = utils.random_A_array(a_key, num_obs, num_states, A_dependencies=A_dependencies)
    B = utils.random_B_array(b_key, num_states, num_controls, B_dependencies=B_dependencies)

    def _broadcast(arr_list):
        return [jnp.broadcast_to(jnp.array(a), (batch_size,) + a.shape) for a in arr_list]

    A_b = _broadcast(A)
    B_b = _broadcast(B)

    H = [jnp.eye(n)[-1] for n in num_states]
    I = generate_I_matrix(H, B, threshold=1.0 / 16, depth=policy_len)
    I_b = _broadcast(I)

    agent = Agent(
        A_b, B_b,
        A_dependencies=A_dependencies,
        B_dependencies=B_dependencies,
        num_controls=num_controls,
        batch_size=batch_size,
        policy_len=policy_len,
        I=I_b,
        use_inductive=True,
        inductive_epsilon=1e-3,
    )

    qs = [
        jnp.broadcast_to(jnp.ones(n) / n, (batch_size, 1, n))
        for n in num_states
    ]

    def fn(agent, qs):
        return agent.infer_policies(qs)

    return fn, (agent, qs), {}


tolerance = {"atol": 1e-4, "rtol": 1e-4}
