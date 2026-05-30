"""Batched inductive policy evaluation with parameter novelty (curiosity).

Builds an Agent over a three-factor (sizes 32, 24, 16), three-modality
(sizes 16, 32, 8) generative model planning three steps ahead, batched across
four agents (12^3 = 1728 policies). The model is learnable (learn_A, learn_B)
with Dirichlet priors pA/pB, and `use_param_info_gain=True` adds the parameter
novelty term to the expected free energy: each policy is scored by utility,
state information gain and the information it is expected to yield about the
observation and transition parameters. Inductive planning via an I matrix from
per-factor goal indicators biases the agent toward its target states. This is
the intrinsically-motivated regime where an agent acts to improve the model it
is still learning. `infer_policies` returns the policy posterior.
"""

import jax.numpy as jnp
import jax.random as jr
from pymdp import utils
from pymdp.agent import Agent
from pymdp.control import generate_I_matrix


def fixture():
    key = jr.PRNGKey(0)
    a_key, b_key = jr.split(key, 2)

    num_obs = [16, 32, 8]
    num_states = [32, 24, 16]
    num_controls = [2, 3, 2]   # 12^3 = 1728 policies
    policy_len = 3
    batch_size = 4

    A_dependencies = [[0, 1], [1, 2], [0, 1, 2]]
    B_dependencies = [[0], [1], [2]]   # factor-local — required by generate_I_matrix

    A = utils.random_A_array(a_key, num_obs, num_states, A_dependencies=A_dependencies)
    B = utils.random_B_array(b_key, num_states, num_controls, B_dependencies=B_dependencies)

    # Dirichlet concentration — typical "loose prior" choice that gives
    # non-trivial novelty signal (large pA → low novelty; pA near zero → high novelty).
    pA = [3.0 * a + 0.1 for a in A]
    pB = [3.0 * b + 0.1 for b in B]

    def _broadcast(arr_list):
        return [jnp.broadcast_to(jnp.array(a), (batch_size,) + a.shape) for a in arr_list]

    A_b = _broadcast(A); B_b = _broadcast(B)
    pA_b = _broadcast(pA); pB_b = _broadcast(pB)

    H = [jnp.eye(n)[-1] for n in num_states]
    I = generate_I_matrix(H, B, threshold=1.0 / 16, depth=policy_len)
    I_b = _broadcast(I)

    agent = Agent(
        A_b, B_b,
        pA=pA_b,
        pB=pB_b,
        A_dependencies=A_dependencies,
        B_dependencies=B_dependencies,
        num_controls=num_controls,
        batch_size=batch_size,
        policy_len=policy_len,
        I=I_b,
        use_inductive=True,
        use_param_info_gain=True,
        learn_A=True,
        learn_B=True,
        inductive_epsilon=1e-3,
    )

    qs = [jnp.broadcast_to(jnp.ones(n) / n, (batch_size, 1, n)) for n in num_states]

    def fn(agent, qs):
        return agent.infer_policies(qs)

    return fn, (agent, qs), {}


tolerance = {"atol": 1e-3, "rtol": 1e-3}
