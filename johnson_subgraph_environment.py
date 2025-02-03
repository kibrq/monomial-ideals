from typing import Optional, Any, Dict

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from functools import partial
from itertools import combinations
import math

from johnson_graph import (
    init_johnson_graph,
    add_subset_to_johnson_graph,
    remove_subset_from_johnson_graph,
    visualize_johson_graph,
)

from linearity_test import linearity_test


# This file implements `gymnasium` environment for building subgraph of Johnson graph
# (https://en.wikipedia.org/wiki/Johnson_graph) with maximum possible diameter and if subsets 
# inducing this subgraph creates "linear ideal" in Polynomial Ring.
#
# Environment creates a subgraph of J(n, k) and during reset it makes it empty and adds 
# a random number (from zero to `init_max_length`) random subsets to initialize the environment.
#
# There is a reward model that can be set. For __init__ method you can provide any keywords 
# starting from `reward_` and they will be passed to `reward_model` function. There are two 
# regimes supported: either environment rewards the agent as the difference of the previously 
# acquired diameter and the current one or just rewards with the current diameter.
#
# It may check or not check for connectivity, linearity, and give or not give negative rewards 
# for non-linearity or non-connectivity.
#
# Also, there are two regimes of the action model. One can either populate the environment 
# by adding one element to the subset or adding the whole subset at once. This is regulated by 
# the parameter `action_is_subset` or `action_is_element_of_subset`.
#
# For linearity check, one can choose from `M2` or `sympy` backend by providing `linearity_backend`.

def binom(n, k):
    if k > n or k < 0:
        return 0
    return math.comb(n, k)

def unrank(n, k, r):
    """Return the r-th k-combination of {1,2,...,n} in lexicographic order.
       Here r is 1-indexed: 1 <= r <= binom(n, k)."""
    S, x = [], 1
    for i in range(1, k+1):
        while binom(n - x, k - i) < r:
            r -= binom(n - x, k - i)
            x += 1
        S.append(x)
        x += 1
    return frozenset(S)

def rank(n, k, S):
    """Return the lexicographic rank (1-indexed) of k-combination S
       where S is a sorted list (e.g., [s1, s2, ..., sk])."""
    r = 1
    previous = 0
    for i in range(k):
        for j in range(previous + 1, S[i]):
            r += binom(n - j, k - i - 1)
        previous = S[i]
    return r


def extract_matching_kwargs(prefix, kwargs, do_pop=True, delimiter="_"):
    """Extracts keyword arguments that match a given prefix."""
    matching_keys = [k for k in kwargs.keys() if k.startswith(prefix + delimiter)]
    func = getattr(kwargs, "pop") if do_pop else getattr(kwargs, "get")
    return {k[len(prefix) + len(delimiter):]: func(k) for k in matching_keys}

def reward_model(
    internal_state: Any,
    state: np.ndarray,
    action: int,
    terminated: bool,
    truncated: bool,
    
    use_diameter: bool = False,
    use_diameter_difference: bool = True,
    check_connectivity: bool = True,
    disconnected: float = 0,
    check_linearity: bool = True,
    non_linear: float = 0,
    only_at_the_end: bool = False,

    statistics: Optional[Dict] = None,

    **kwargs
):
    """Defines the reward model for the environment based on graph properties."""
    if not statistics:
        statistics = defaultdict(list)

    statistics['connected'].append(False)
    statistics['linear'].append(False)
    statistics['diameter'].append(-1e8)
    statistics.setdefault('previous_diameter', 0)
        
    graph = internal_state['graph']

    zero_if_only_at_the_end = lambda r: 0 if not terminated and only_at_the_end else r

    if check_connectivity and not nx.is_connected(graph):
        return zero_if_only_at_the_end(disconnected), statistics
    statistics['connected'][-1] = True

    if check_linearity and not linearity_test(graph.nodes, **kwargs):
        return zero_if_only_at_the_end(non_linear), statistics
    statistics['linear'][-1] = True

    statistics['diameter'][-1] = nx.diameter(graph)

    if use_diameter:
        return zero_if_only_at_the_end(statistics['diameter'][-1]), statistics

    if use_diameter_difference:
        reward = statistics['diameter'][-1] - statistics['previous_diameter']
        statistics['previous_diameter'] = statistics['diameter'][-1]
        return zero_if_only_at_the_end(reward), statistics

    assert False


def init_element_incremental_action_model(n: int, k: int):
    space = spaces.Discrete(n + 1)

    def step(internal_state, action, current_subset):
        if not action:
            return False, current_subset
        current_subset.add(action)
        return True, current_subset

    return space, step


def init_subset_incremental_action_model(n: int, k: int):
    space = spaces.Discrete(math.comb(n, k) + 1)
    
    # comb_list = list(combinations(range(1, n + 1), k))
    # # Create a dictionary mapping each combination to its rank (0-indexed)
    # rank_dict = {comb: idx + 1 for idx, comb in enumerate(comb_list)}

    def step(internal_state, action, current_subset):
        assert not current_subset
        assert action <= math.comb(n, k)
        
        graph = internal_state['graph']

        if not action:
            return False, {}
        newset = unrank(n = n, k = k, r = action)
        # newset = frozenset(comb_list[action])
        if newset in graph.nodes:
            return False, {}
        return True, newset
        
    return space, step


def init_action_model(n: int, k: int, is_subset: bool = False, is_element_of_subset: bool = True):
    """Initializes the action model for adding elements or subsets."""
    if is_subset:
        return init_subset_incremental_action_model(n, k)
    if is_element_of_subset:
        return init_element_incremental_action_model(n, k)


class JohnsonSubgraphEnv(gym.Env):
    def __init__(
        self, n: int, k: int,
        max_length: Optional[int] = None, 
        seed: Optional[int] = None,

        # init_*, action_*, reward_*, linearity_*,
        **kwargs,
    ):
        super().__init__()
        self.n = n
        self.k = k
        self.max_length = max_length if max_length is not None else math.comb(n, k)

        init_kwargs = extract_matching_kwargs("init", kwargs)
        self.init_max_length = init_kwargs.get("max_length", 5)
        assert self.init_max_length <= self.max_length

        action_kwargs = extract_matching_kwargs("action", kwargs)
        self.action_space, self.action_model = init_action_model(n, k, **action_kwargs)
    
        self.observation_space = spaces.MultiDiscrete([[self.n + 1] * self.k] * self.max_length)

        reward_kwargs = extract_matching_kwargs("reward", kwargs)
        linearity_kwargs = extract_matching_kwargs("linearity", kwargs)
        self.reward_model = partial(reward_model, **reward_kwargs, **linearity_kwargs)
        
        self.reset(seed = seed)

    def _get_info(self, **kwargs):
        return {**kwargs, **self.info, "graph": self.state["graph"]}

    def _get_observation(self, *args, **kwargs):
        return self.np_subsets

    def _random_subsets(self):
        while True:
            yield frozenset(self.np_random.choice(self.n, size=(self.k, ), replace=False).tolist())

    def reset(self, seed: Optional[int] = None, options: Optional[Any] = None):
        """Resets the environment to its initial state."""
        super().reset(seed = seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

        self.np_subsets = np.zeros((self.max_length, self.k), dtype = int)
        self.num_subsets = 0

        self.state = init_johnson_graph(self.n, self.k)
        graph = self.state['graph']

        initial_length = self.np_random.integers(self.init_max_length + 1)

        for subset in self._random_subsets():
            if self.num_subsets >= initial_length:
                break
                
            if not subset in graph.nodes:
                self.state = add_subset_to_johnson_graph(subset, self.state)
                self.np_subsets[self.num_subsets] = np.array(list(subset), dtype=int)
                self.num_subsets += 1
            
        
        self.current_subset = set()
        self.statistics = {}

        self.info = {
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "max_diameter": 0,
        }
        
        return self._get_observation(), self._get_info()


    def step(self, action):
        """Performs one step in the environment."""
        reward = 0
        terminated = False
        truncated = False

        # return self._get_observation(), reward, terminated, truncated, self._get_info()

        changed, self.current_subset = self.action_model(self.state, action, self.current_subset)
        if not changed:
            return self._get_observation(), reward, terminated, truncated, self._get_info()

        # Check if the subset is complete
        graph = self.state["graph"]
        if len(self.current_subset) == self.k and not frozenset(self.current_subset) in graph.nodes:
            self.state = add_subset_to_johnson_graph(self.current_subset, self.state)
            self.np_subsets[self.num_subsets] = np.array(list(self.current_subset))
            self.num_subsets += 1

            if self.num_subsets >= self.max_length:
                terminated = True

            reward, self.statistics = self.reward_model(self.state, self.np_subsets, action,
                                                        terminated, truncated, statistics=self.statistics)
            self.info["max_diameter"] = max(self.info["max_diameter"], self.statistics["diameter"][-1])    

            # Reset the current subset
            self.current_subset = set()

        self.info["nodes"] = len(graph.nodes)
        self.info["edges"] = len(graph.edges)

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def render(self, mode="human"):
        """Renders the current graph without the unfinished subset."""
        graph = self.state["graph"]
        print("Current Graph Nodes:", list(graph.nodes))
        print("Current Graph Edges:", list(graph.edges))
        print("Current unfinished subset:", self.current_subset)


# Register the environment
gym.envs.registration.register(
    id='JohnsonSubgraph-v0',
    entry_point='johnson_subgraph_environment:JohnsonSubgraphEnv',
)

