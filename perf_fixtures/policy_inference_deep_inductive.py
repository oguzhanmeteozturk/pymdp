"""Inductive policy evaluation over a deep multi-step planning tree.

Evaluates expected free energy for every policy of a three-factor (sizes
32, 24, 16), three-modality (sizes 16, 32, 8) generative model planning three
steps ahead. With 2, 3 and 2 controls per factor this enumerates 12^3 = 1728
policies. The observation likelihood A is ragged ([0,1], [1,2], [0,1,2]) and
transitions are factor-local, which inductive planning requires: an I matrix is
built from per-factor goal-state indicators H by backward reachability, biasing
the agent toward policies that can reach each factor's last state. Each policy
is scored with utility and state information gain plus the inductive prior.
"""

import inspect

import jax.numpy as jnp
import jax.random as jr
from pymdp import utils
from pymdp.control import (
    update_posterior_policies_inductive,
    construct_policies,
    generate_I_matrix,
)

# `ffi_cache_params_static` is a perf hint added after some baselines. Only pass
# it when the resolved `update_posterior_policies_inductive` accepts it, so this
# fixture runs unchanged against older revisions (e.g. perf-baseline-compare,
# which overlays current perf_fixtures/ onto an older pymdp/control.py).
_SUPPORTS_CACHE_STATIC = (
    "ffi_cache_params_static"
    in inspect.signature(update_posterior_policies_inductive).parameters
)


def fixture():
    key = jr.PRNGKey(0)
    a_key, b_key = jr.split(key, 2)

    num_obs = [16, 32, 8]
    num_states = [32, 24, 16]
    num_controls = [2, 3, 2]  # 12^3 = 1728 policies
    policy_len = 3

    A_dependencies = [[0, 1], [1, 2], [0, 1, 2]]
    # B must be factor-local for generate_I_matrix (backward reachability is
    # only defined for per-factor transitions). This matches Agent setups that
    # use inductive planning.
    B_dependencies = [[0], [1], [2]]

    A = utils.random_A_array(a_key, num_obs, num_states, A_dependencies=A_dependencies)
    B = utils.random_B_array(b_key, num_states, num_controls, B_dependencies=B_dependencies)
    C = [jnp.zeros((policy_len, n)) for n in num_obs]
    E = jnp.ones(construct_policies(num_states, num_controls, policy_len).shape[0])

    policy_matrix = construct_policies(num_states, num_controls, policy_len)
    qs_init = [jnp.ones(n) / n for n in num_states]

    # Inductive planning matrix: goal-set indicator vectors H[f], then generate I[f]
    # via backward reachability. Pick goal = last state of each factor to match a
    # generic "reach target" setup.
    H = [jnp.eye(n)[-1] for n in num_states]
    I = generate_I_matrix(H, B, threshold=1.0 / 16, depth=policy_len)

    # Model parameters are fixed (no online learning in this scenario), so the
    # static-cache fast path is safe — but only on revisions that expose it.
    extra_kwargs = (
        {"ffi_cache_params_static": True} if _SUPPORTS_CACHE_STATIC else {}
    )

    def fn(policy_matrix, qs_init, A, B, C, E, I):
        return update_posterior_policies_inductive(
            policy_matrix, qs_init, A, B, C, E,
            pA=None, pB=None,
            A_dependencies=A_dependencies,
            B_dependencies=B_dependencies,
            I=I,
            gamma=16.0,
            inductive_epsilon=1e-3,
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=False,
            use_inductive=True,
            **extra_kwargs,
        )

    return fn, (policy_matrix, qs_init, A, B, C, E, I), {}


tolerance = {"atol": 1e-4, "rtol": 1e-4}
