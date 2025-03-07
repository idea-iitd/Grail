import numpy as np
import itertools
import networkx as nx

def priority_v1(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes the Graph Edit Distance (GED), a measure of the dissimilarity between two graphs. 
    GED is defined as the minimum number of operations required to transform one graph into another.
    The primary operations considered in GED calculations include:

    - **Node Insertion/Deletion:** Adding or removing a node incurs a cost of +1.
    - **Edge Insertion/Deletion:** Adding or removing an edge between two nodes incurs a cost of +1.
    - **Node Relabeling:** Modifying the label of a node (if labels are present) adds a cost of +1 for each mismatch.

    Args:
        graph1: The adjacency matrix of the first graph.
        graph2: The adjacency matrix of the second graph.
        weights: A weight matrix representing the initial probabilities of mapping nodes between `graph1` and `graph2`.
                 Each entry is a probability value, where a higher value indicates a higher likelihood and similarity 
                 of mapping nodes. The size of the weight matrix is determined by the maximum number of nodes in both graphs squared.

    Returns:
        A refined weight matrix (float) using the initial input matrix and the adjacency matrices of graphs where each entry represents the probability of a node in 
        `graph1` being mapped to a node in `graph2` in a way that minimizes the overall graph edit distance.
  """
  max_node = len(graph1)
  weights = [[0.0] * max_node for _ in range(max_node)]
  return weights