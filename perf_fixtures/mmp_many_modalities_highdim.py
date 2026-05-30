"""Marginal message passing over a high-dimensional, many-modality model.

One full perception-and-decision cycle for a batch of four agents whose
generative model has two high-dimensional hidden-state factors (64 states each)
feeding ten observation modalities (8 outcomes each), every modality depending
on both factors, with factor-local transitions.

`infer_states` runs marginal message passing (MMP) over a four-step backward
inference window: given a sequence of observations per modality and the
agent's past actions, it infers the trajectory of posterior beliefs. The
inferred beliefs then feed `infer_policies`, which scores each one-step policy
by utility and state information gain and returns the policy posterior.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from pymdp import utils
from pymdp.agent import Agent


def fixture():
    key = jr.PRNGKey(7)
    a_key, b_key, o_key, act_key = jr.split(key, 4)

    num_factors = 2
    num_states = [64, 64]            # few factors, high-dimensional
    num_controls = [4, 4]
    num_modalities = 10              # many modalities
    num_obs = [8] * num_modalities
    batch_size = 4
    window = 4                       # MMP backward inference horizon (T)

    A_dependencies = [[0, 1]] * num_modalities
    B_dependencies = [[0], [1]]

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
        inference_algo="mmp",
        inference_horizon=window,
        policy_len=1,
        batch_size=batch_size,
    )

    # Sequence obs: integer indices per modality with a time axis -> (batch, T).
    obs = [
        jr.randint(jr.fold_in(o_key, m), (batch_size, window), 0, num_obs[m]).astype(jnp.int32)
        for m in range(num_modalities)
    ]
    # Action history aligned to the window -> (batch, T-1, num_factors).
    past_actions = jnp.stack(
        [
            jr.randint(jr.fold_in(act_key, f), (batch_size, window - 1), 0, num_controls[f])
            for f in range(num_factors)
        ],
        axis=-1,
    ).astype(jnp.int32)

    def fn(agent, obs, past_actions):
        qs = agent.infer_states(obs, empirical_prior=agent.D, past_actions=past_actions)
        qpi, neg_efe = agent.infer_policies(qs)
        return qpi, neg_efe

    return fn, (agent, obs, past_actions), {}


tolerance = {"atol": 1e-4, "rtol": 1e-4}
