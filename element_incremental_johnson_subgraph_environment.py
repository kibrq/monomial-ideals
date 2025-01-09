from typing import Optional, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import networkx as nx

from johnson_graph import (
    init_johnson_graph,
    add_subset_to_johnson_graph,
    remove_subset_from_johnson_graph,
    visualize_johson_graph,
)

from linearity_test import linearity_test

class ElementIncrementalJohnsonSubgraphEnv(gym.Env):
    def __init__(
        self, n: int, k: int,
        max_length: Optional[int] = None,
        disconnected_reward: float = 0,
        non_linearity_reward: float = 0,
        use_diameter_as_reward: bool = False,
        use_diameter_difference_as_reward: bool = True,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.n = n
        self.k = k
        self.max_length = max_length if max_length is not None else int(comb(n, k))
        
        # Large negative reward for disconnected graph
        self.disconnected_reward = disconnected_reward
        self.non_linearity_reward = non_linearity_reward
        self.use_diameter_as_reward = use_diameter_as_reward
        self.use_diameter_difference_as_reward = use_diameter_difference_as_reward
        
        # Action space: selecting elements from 0 to n, 0 is reserved for padding
        self.action_space = spaces.Discrete(self.n + 1)

        self.observation_space = spaces.MultiDiscrete([[self.n + 1] * self.k] * self.max_length)
        
        self.reset(seed = seed)

        self.kwargs = kwargs

    def _get_info(self, **kwargs):
        return {**kwargs}

    def _get_observation(self, *args, **kwargs):
        return self.np_subsets

    def reset(self, seed: Optional[int] = None, options: Optional[Any] = None):
        """Resets the environment to its initial state."""
        super().reset(seed = seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        
        self.state = init_johnson_graph(self.n, self.k)
        
        self.current_subset = set()
        self.state_sequence = []  # Sequence of subsets

        self.np_subsets = np.zeros((self.max_length, self.k), dtype = int)
        
        self.num_subsets = 0
        self.previous_diameter = 0
        return self._get_observation(), self._get_info()

    def step(self, action):
        """Performs one step in the environment."""
        reward = 0
        terminated = False
        truncated = False

        # return self._get_observation(), reward, terminated, truncated, self._get_info()

        if action == 0 or action in self.current_subset:
            return self._get_observation(), reward, terminated, truncated, self._get_info()

        # Add the action to the current subset
        self.np_subsets[self.num_subsets, len(self.current_subset)] = action
        self.current_subset.add(action)
        
        # Check if the subset is complete
        if len(self.current_subset) == self.k:
            self.state = add_subset_to_johnson_graph(self.current_subset, self.state)
            self.num_subsets += 1

            # Calculate reward
            graph = self.state["graph"]
            # if not nx.is_connected(graph):
                # self.state = remove_subset_from_johnson_graph(self.current_subset, self.state)

            if not nx.is_connected(graph):
                reward = self.disconnected_reward

            elif not linearity_test(graph.nodes, **self.kwargs):
                reward = self.non_linearity_reward

            else:
                current_diameter = nx.diameter(graph)
                if self.use_diameter_as_reward:
                    reward = current_diameter
                elif self.use_diameter_difference_as_reward:
                    reward = current_diameter - self.previous_diameter
                else:
                    assert False
                    
                self.previous_diameter = current_diameter
                

            # Reset the current subset
            self.current_subset = set()

        # Check if the environment is done
        if self.num_subsets >= self.max_length:
            terminated = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def render(self, mode="human"):
        """Renders the current graph without the unfinished subset."""
        graph = self.state["graph"]
        print("Current Graph Nodes:", list(graph.nodes))
        print("Current Graph Edges:", list(graph.edges))
        print("Current unfinished subset:", self.current_subset)


# Register the environment
gym.envs.registration.register(
    id='ElementIncrementalJohnsonSubgraph-v0',
    entry_point='element_incremental_johnson_subgraph_environment:ElementIncrementalJohnsonSubgraphEnv',
)
