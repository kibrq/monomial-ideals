import networkx as nx
import matplotlib.pyplot as plt


def init_johnson_graph(n, k, subsets = None):
    """Initializes a Johnson graph with parameters n and k."""
    state = {"graph": nx.Graph(), "n": n, "k": k}
  
    for subset in (subsets or []):
        state = add_subset_to_johnson_graph(subset, state)
    return state    
    

def add_subset_to_johnson_graph(subset, state):
    """Adds a subset to the Johnson graph and updates its state."""
    graph, n, k = state["graph"], state["n"], state["k"]
    subset = frozenset(subset)

    assert len(subset) == k
    
    if subset in graph:
        return state  # If the subset is already in the graph, return immediately

    graph.add_node(subset)    

    # Connect to other nodes with |intersection| == k-1
    for existing_node in graph.nodes:
        if len(existing_node & subset) == k - 1:
            graph.add_edge(existing_node, subset)

    return state


def remove_subset_from_johnson_graph(subset, state):
    graph, subset = state["graph"], frozenset(subset)
    graph.remove_node(subset)
    return state


def visualize_johson_graph(state, title=None):
    graph, n, k = state["graph"], state["n"], state["k"]
    pos = nx.spring_layout(graph)  # Layout for consistent visualization
    plt.figure(figsize=(10, 8))
    nx.draw(
        graph, pos, with_labels=True, node_color='lightblue', 
        node_size=700, font_weight='bold'
    )
    plt.title(title or f"subgraph of J({n}, {k})")
    plt.show()