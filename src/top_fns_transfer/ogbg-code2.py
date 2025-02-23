#Num Unique Functions in logs: 260
import itertools
import numpy as np
import networkx as nx
import copy
import math
import heapq
import random
import math

def priority_v0(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes a refined weight matrix based on graph structure and initial weights.

    Args:
        graph1: The adjacency matrix of the first graph.
        graph2: The adjacency matrix of the second graph.
        weights: A weight matrix representing the initial probabilities of mapping nodes between `graph1` and `graph2`.

    Returns:
        A refined weight matrix (float).
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] == 1 and graph2[j][l] == 1:  # Check if edge exists in both graphs
            neighbor_similarity += weights[k][l] if k < len(weights) and l < len(weights[k]) else 0


      refined_weights[i][j] = neighbor_similarity  # Update weights based on structural similarity

  return refined_weights

def priority_v1(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0` with normalization."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] == 1 and graph2[j][l] == 1:
            neighbor_similarity += weights[k][l] if k < len(weights) and l < len(weights[k]) else 0

      degree_i = sum(graph1[i])
      degree_j = sum(graph2[j])

      if degree_i > 0 and degree_j > 0:  # Normalize by degrees
          refined_weights[i][j] = neighbor_similarity / (degree_i * degree_j)
      elif neighbor_similarity > 0 : # if no neighbors exist for a potential match then default to a small value.
          refined_weights[i][j] = 0.0001
      

  return refined_weights

def priority_v2(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights matrix with zeros
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Iterate through potential node mappings
    for i in range(n1):
        for j in range(n2):
            # Calculate a similarity score based on node degrees and neighborhood similarity
            degree_similarity = 1.0 / (1.0 + abs(sum(graph1[i]) - sum(graph2[j])))

            neighbor_similarity = 0.0
            common_neighbors = 0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] == 1 and graph2[j][l] == 1:  # Check if nodes are neighbors
                        neighbor_similarity += weights[k][l] # Use initial weights to estimate neighbor similarity
                        common_neighbors +=1
            if common_neighbors > 0:
                neighbor_similarity /= common_neighbors


            # Combine the similarity scores and initial weight (if available)
            if i < len(weights) and j < len(weights[i]):
                refined_weights[i][j] = (degree_similarity + neighbor_similarity + weights[i][j]) / 3
            else:
                refined_weights[i][j] = (degree_similarity + neighbor_similarity) / 2


    return refined_weights

def priority_v3(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights matrix with zeros
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Iterate through all possible node pairs (i, j) from graph1 and graph2
    for i in range(n1):
        for j in range(n2):
            # Calculate a similarity score based on the neighborhood structure
            neighbor_similarity = 0
            for k in range(n1):
                for l in range(n2):
                    neighbor_similarity += graph1[i][k] * graph2[j][l] * weights[k][l]

            # Incorporate the initial weight and the neighbor similarity into the refined weight
            refined_weights[i][j] = weights[i][j] + neighbor_similarity  # Or another combining function

    # Normalize the refined weights (optional, but often beneficial)
    for i in range(n1):
      row_sum = sum(refined_weights[i][:n2]) # Only normalize up to n2
      if row_sum > 0:
        for j in range(n2):
          refined_weights[i][j] /= row_sum

    return refined_weights

def priority_v4(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes the Graph Edit Distance (GED), a measure of the dissimilarity between two graphs. 

    Args:
        graph1: The adjacency matrix of the first graph.
        graph2: The adjacency matrix of the second graph.
        weights: A weight matrix representing the initial probabilities of mapping nodes between `graph1` and `graph2`.

    Returns:
        A refined weight matrix (float) using the initial input matrix and the adjacency matrices of graphs where each entry represents the probability of a node in `graph1` 
        being mapped to a node in `graph2` in a way that minimizes the overall graph edit distance.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))  # Difference in node degrees
      neighbor_similarity = 0
      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] == 1 and graph2[j][l] == 1:
            neighbor_similarity += weights[k][l] #Initial weights contribute to neighbor similarity

      refined_weights[i][j] = (1.0 / (1.0 + degree_diff)) * (1.0 + neighbor_similarity)  # Combine degree difference and neighbor similarity

  return refined_weights

def priority_v5(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0` with normalization."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    row_sum = 0
    for j in range(n2):
      degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
      neighbor_similarity = 0
      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] == 1 and graph2[j][l] == 1:
            neighbor_similarity += weights[k][l]

      refined_weights[i][j] = (1.0 / (1.0 + degree_diff)) * (1.0 + neighbor_similarity)
      row_sum += refined_weights[i][j]

    # Normalize row-wise
    if row_sum > 0:  # Avoid division by zero
      for j in range(n2):
        refined_weights[i][j] /= row_sum

  return refined_weights

def priority_v6(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities for node mappings based on node degrees.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      degree_diff = abs(degrees1[i] - degrees2[j])
      weights[i][j] = 1.0 / (1 + degree_diff)  # Higher similarity for similar degrees

  return weights

def priority_v7(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Considers neighbor degrees in addition to node degrees.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for neighbor1 in range(n1):
        if graph1[i][neighbor1]:
          for neighbor2 in range(n2):
            if graph2[j][neighbor2]:
              neighbor_similarity += 1.0 / (1 + abs(sum(graph1[neighbor1]) - sum(graph2[neighbor2])))
      weights[i][j] = (1.0 / (1 + abs(sum(graph1[i]) - sum(graph2[j])))) + neighbor_similarity

  return weights

def priority_v8(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """
    Computes initial probabilities based on node degrees.
    """
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    weights = [[0.0] * max_node for _ in range(max_node)]

    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            weights[i][j] = 1.0 / (abs(degrees1[i] - degrees2[j]) + 1)

    return weights

def priority_v9(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """
    Considers neighbor degrees in addition to node degrees.
    """
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    weights = [[0.0] * max_node for _ in range(max_node)]

    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]

    for i in range(n1):
        neighbor_degrees1 = [degrees1[k] for k in range(n1) if graph1[i][k]]
        for j in range(n2):
            neighbor_degrees2 = [degrees2[k] for k in range(n2) if graph2[j][k]]

            degree_diff = abs(degrees1[i] - degrees2[j])
            neighbor_diff = 0
            for d1 in neighbor_degrees1:
                min_diff = float('inf')
                for d2 in neighbor_degrees2:
                    min_diff = min(min_diff, abs(d1 - d2))
                if min_diff != float('inf'):
                    neighbor_diff += min_diff

            weights[i][j] = 1.0 / (degree_diff + neighbor_diff + 1)
    return weights

def priority_v10(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities based solely on node degrees.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  deg1 = [sum(row) for row in graph1]
  deg2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      weights[i][j] = 1.0 / (1.0 + abs(deg1[i] - deg2[j]))

  return weights

def priority_v11(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Considers neighbor degrees in addition to node degrees."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  deg1 = [sum(row) for row in graph1]
  deg2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
        if graph1[i][k]:
          for l in range(n2):
            if graph2[j][l]:
              neighbor_similarity += 1.0 / (1.0 + abs(deg1[k] - deg2[l]))

      weights[i][j] = (1.0 / (1.0 + abs(deg1[i] - deg2[j]))) * (1 + neighbor_similarity)

  return weights

def priority_v12(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities for node mappings based on node degrees.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      weights[i][j] = 1.0 / (1.0 + abs(degrees1[i] - degrees2[j]))

  return weights

def priority_v13(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Considers neighbor degrees in addition to node degrees."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
        if graph1[i][k]:
          for l in range(n2):
            if graph2[j][l]:
              neighbor_similarity += 1.0 / (1.0 + abs(degrees1[k] - degrees2[l]))
      weights[i][j] = 1.0 / (1.0 + abs(degrees1[i] - degrees2[j])) + neighbor_similarity

  return weights

def priority_v14(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`.

    This version calculates node similarity based on neighborhood structure and uses it to refine the initial weights.
    """
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            # Calculate neighborhood similarity
            neighbors1 = [k for k in range(n1) if graph1[i][k] == 1]
            neighbors2 = [k for k in range(n2) if graph2[j][k] == 1]

            common_neighbors = 0
            for neighbor1 in neighbors1:
                for neighbor2 in neighbors2:
                   if weights[neighbor1][neighbor2] > 0:  # Consider existing weight as an indication of potential mapping
                       common_neighbors +=1


            similarity = 0.0
            if len(neighbors1) + len(neighbors2) > 0: # avoid division by zero
              similarity = (2 * common_neighbors) / (len(neighbors1) + len(neighbors2))


            # Combine initial weight and neighborhood similarity. You can adjust the weights (0.5 and 0.5 here)
            # to give more importance to either initial weights or structural similarity.
            if i < len(weights) and j < len(weights[i]):
                refined_weights[i][j] = 0.5 * (weights[i][j] if i < len(weights) and j < len(weights[i]) else 0 )+ 0.5 * similarity

    return refined_weights

def priority_v15(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights matrix with zeros
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Handle cases where input weights are not provided or have incorrect dimensions
    if weights is None or len(weights) != max_node or len(weights[0]) != max_node:
        weights = [[1.0 / max_node] * max_node for _ in range(max_node)]  # Uniform initial weights


    for i in range(n1):
        for j in range(n2):
            # Calculate the structural similarity based on neighborhood information
            neighbors1 = [k for k in range(n1) if graph1[i][k] == 1]
            neighbors2 = [k for k in range(n2) if graph2[j][k] == 1]

            common_neighbors = 0
            for neighbor1 in neighbors1:
                for neighbor2 in neighbors2:
                    common_neighbors += weights[neighbor1][neighbor2]  # Utilize initial weights

            # Update refined weights considering both initial weights and structural similarity
            refined_weights[i][j] = weights[i][j] * (1 + common_neighbors)


    # Normalize the refined weights to represent probabilities
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])  # Normalize only for valid node indices
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum

    return refined_weights

def priority_v16(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`.
    Calculates node mapping priorities based on structural similarity.
    """
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            degree1 = sum(graph1[i])
            degree2 = sum(graph2[j])
            neighbor_similarity = 0

            for neighbor1 in range(n1):
                if graph1[i][neighbor1]:
                    for neighbor2 in range(n2):
                        if graph2[j][neighbor2]:
                            neighbor_similarity += weights[neighbor1][neighbor2]  # Use initial weights for neighbors

            # Combine degree difference and neighbor similarity
            degree_diff = abs(degree1 - degree2)
            refined_weights[i][j] = (1 + neighbor_similarity) / (1 + degree_diff) if (1+degree_diff) > 0 else 0


    # Normalize weights (optional but recommended)
    total_weight = sum(sum(row) for row in refined_weights)
    if total_weight > 0:
        for i in range(max_node):
            for j in range(max_node):
                refined_weights[i][j] /= total_weight


    return refined_weights

def priority_v17(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights matrix with zeros
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Handle cases where initial weights are not provided or are of incorrect dimensions
    if weights is None or len(weights) != max_node or len(weights[0]) != max_node:
        weights = [[1.0 / max_node] * max_node for _ in range(max_node)]  # Uniform initial probabilities


    for i in range(n1):
        for j in range(n2):
            # Calculate node similarity based on degree difference
            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
            node_similarity = 1.0 / (1.0 + degree_diff)  # Higher similarity for smaller degree difference

            # Calculate edge similarity
            edge_similarity = 0.0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] == 1 and graph2[j][l] == 1:  # If edge exists in both graphs
                        edge_similarity += weights[k][l] # Use initial weights here to consider potential mappings of neighbors


            # Combine node and edge similarity with initial weights
            refined_weights[i][j] = (weights[i][j] + node_similarity + edge_similarity) / 3


    # Normalize the refined weights (optional, but recommended) so they sum to 1 for each row
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2]) # Only sum up to n2 to avoid including padded zeros
        if row_sum > 0:  # Avoid division by zero
            for j in range(n2):
                refined_weights[i][j] /= row_sum


    return refined_weights

def priority_v18(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes the Graph Edit Distance (GED), a measure of the dissimilarity between two graphs. 
    GED is defined as the minimum number of operations required to transform one graph into another.

    Args:
        graph1: The adjacency matrix of the first graph.
        graph2: The adjacency matrix of the second graph.
        weights: A weight matrix representing the initial probabilities of mapping nodes between `graph1` and `graph2`.

    Returns:
        A refined weight matrix (float) using the initial input matrix and the adjacency matrices of graphs where each entry represents the probability of a node in `graph1` 
        being mapped to a node in `graph2` in a way that minimizes the overall graph edit distance.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      cost = 0
      for k in range(n1):
        for l in range(n2):
          cost += abs(graph1[i][k] - graph2[j][l])  # Edge difference cost
      refined_weights[i][j] = 1 / (1 + cost) # Convert cost to similarity/probability


  return refined_weights

def priority_v19(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0` incorporating initial weights."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      cost = 0
      for k in range(n1):
        for l in range(n2):
          cost += abs(graph1[i][k] - graph2[j][l])
      if i < len(weights) and j < len(weights[0]): # Check bounds for weights
         refined_weights[i][j] = weights[i][j] / (1 + cost)  # Combine initial weight with structural similarity
      else:
          refined_weights[i][j] = 1 / (1 + cost) # If initial weight not available, just use structural similarity

  return refined_weights

def priority_v20(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights matrix with zeros
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Iterate through all possible node pairings
    for i in range(n1):
        for j in range(n2):
            # Calculate a similarity score based on node degree and neighborhood structure
            degree_similarity = 1.0 / (abs(sum(graph1[i]) - sum(graph2[j])) + 1)  # Inversely proportional to degree difference

            neighborhood_similarity = 0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] == 1 and graph2[j][l] == 1:  # If both nodes have an edge to another pair of nodes
                        neighborhood_similarity += weights[k][l] if k < len(weights) and l < len(weights[0]) else 0 # Use provided weights if available


            # Combine the similarity scores and the initial weight (if available)
            refined_weights[i][j] = (degree_similarity + neighborhood_similarity) * (weights[i][j] if i < len(weights) and j < len(weights[0]) else 1.0/max_node)

    # Normalize the weights so they sum to 1 for each row (optional but often beneficial)
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])  # Only sum up to n2, as those are the valid mappings for graph2 nodes
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum



    return refined_weights

def priority_v21(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights matrix
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Handle cases where input weights are not provided or have incorrect dimensions
    if weights is None or len(weights) != max_node or len(weights[0]) != max_node:
        weights = [[1.0 / max_node] * max_node for _ in range(max_node)] # Uniform initial probabilities if not provided


    for i in range(n1):
        for j in range(n2):
            # Calculate a similarity score based on node degree and neighborhood similarity
            degree_similarity = 1.0 - abs(sum(graph1[i]) - sum(graph2[j])) / max(sum(graph1[i]), sum(graph2[j])) if max(sum(graph1[i]), sum(graph2[j])) > 0 else 0.0

            neighbor_similarity = 0.0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] and graph2[j][l]:  # If both nodes have an edge to these neighbors
                        neighbor_similarity += weights[k][l]

            # Combine initial weight, degree similarity, and neighborhood similarity
            refined_weights[i][j] = weights[i][j] * (0.5 * degree_similarity + 0.5 * neighbor_similarity/max(1,sum(graph1[i])*sum(graph2[j])))


    # Normalize the refined weights (optional, but recommended for probabilities)
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum



    return refined_weights

def priority_v22(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes a refined weight matrix based on graph structure similarity.

    Args:
        graph1: The adjacency matrix of the first graph.
        graph2: The adjacency matrix of the second graph.
        weights: A weight matrix representing the initial probabilities of mapping nodes between `graph1` and `graph2`.

    Returns:
        A refined weight matrix (float).
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] == 1 and graph2[j][l] == 1:
            neighbor_similarity += weights[k][l]  # Use initial weights for neighbor comparison

      refined_weights[i][j] = neighbor_similarity

  return refined_weights

def priority_v23(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0` with normalization and optional initial weights."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] == 1 and graph2[j][l] == 1:
            neighbor_similarity += weights[k][l] if weights else 1 # Use 1 if no initial weights

      refined_weights[i][j] = neighbor_similarity


  # Normalize the refined weights
  for i in range(n1):
      row_sum = sum(refined_weights[i][:n2])  # Sum only up to n2
      if row_sum > 0:
          for j in range(n2):
              refined_weights[i][j] /= row_sum

  return refined_weights

def priority_v25(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Improves initial probabilities by considering neighbor degrees.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
        if graph1[i][k]:
          for l in range(n2):
            if graph2[j][l]:
              neighbor_similarity += 1.0 / (1.0 + abs(degrees1[k] - degrees2[l]))
      weights[i][j] = (1.0 / (1.0 + abs(degrees1[i] - degrees2[j]))) * (1 + neighbor_similarity)

  return weights

def priority_v27(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Considers neighbor degrees in addition to node degrees.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
        if graph1[i][k]:
          for l in range(n2):
            if graph2[j][l]:
              neighbor_similarity += 1.0 / (1.0 + abs(degrees1[k] - degrees2[l]))
      weights[i][j] = 1.0 / (1.0 + abs(degrees1[i] - degrees2[j])) + neighbor_similarity

  return weights

def priority_v29(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Considers neighbor degrees in addition to node degrees."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
        if graph1[i][k]:
          for l in range(n2):
            if graph2[j][l]:
              neighbor_similarity += 1.0 / (1.0 + abs(degrees1[k] - degrees2[l]))
      weights[i][j] = (1.0 / (1.0 + abs(degrees1[i] - degrees2[j]))) * (1 + neighbor_similarity)


  return weights

def priority_v30(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities based on node degrees.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      weights[i][j] = 1.0 / (1.0 + abs(degrees1[i] - degrees2[j]))

  return weights

def priority_v31(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Considers neighbor degrees in addition to node degrees.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      neighbor_diff = 0
      for k in range(n1):
        if graph1[i][k]:
          for l in range(n2):
            if graph2[j][l]:
              neighbor_diff += abs(degrees1[k] - degrees2[l])
      weights[i][j] = 1.0 / (1.0 + abs(degrees1[i] - degrees2[j]) + neighbor_diff)

  return weights

def priority_v32(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities for node mappings between two graphs.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  new_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
      new_weights[i][j] = 1.0 / (1 + degree_diff)  # Higher similarity for similar degrees

  return new_weights

def priority_v33(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version using neighborhood similarity."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  new_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
          for l in range(n2):
              if graph1[i][k] == 1 and graph2[j][l] == 1:
                  neighbor_similarity += 1  # Counts common neighbors

      degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
      new_weights[i][j] = (neighbor_similarity + 1) / (1 + degree_diff) # Combines neighbor similarity and degree difference


  return new_weights

def priority_v34(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version using normalized neighborhood similarity."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  new_weights = [[0.0] * max_node for _ in range(max_node)]


  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
          for l in range(n2):
              if graph1[i][k] == 1 and graph2[j][l] == 1:
                  neighbor_similarity += 1

      degree1 = sum(graph1[i])
      degree2 = sum(graph2[j])

      # Normalize neighbor similarity by the degrees
      if degree1 > 0 and degree2 > 0:
          normalized_similarity = neighbor_similarity / (math.sqrt(degree1 * degree2))
      else:
          normalized_similarity = 0


      degree_diff = abs(degree1 - degree2)
      new_weights[i][j] = (normalized_similarity + 1) / (1 + degree_diff)

  return new_weights

def priority_v35(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities based on node degrees.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  deg1 = [sum(row) for row in graph1]
  deg2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      weights[i][j] = 1.0 / (abs(deg1[i] - deg2[j]) + 1)  # Higher probability for similar degrees

  return weights

def priority_v36(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Considers neighbor degrees in addition to node degrees.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  deg1 = [sum(row) for row in graph1]
  deg2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
        if graph1[i][k]:
          for l in range(n2):
            if graph2[j][l]:
              neighbor_similarity += 1.0 / (abs(deg1[k] - deg2[l]) + 1)
      weights[i][j] = (1.0 / (abs(deg1[i] - deg2[j]) + 1)) * (neighbor_similarity + 1) # Combine node and neighbor similarity


  return weights

def priority_v37(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Baseline: Returns a matrix of zeros."""
    max_node = len(graph1)
    weights = [[0.0] * max_node for _ in range(max_node)]
    return weights

def priority_v38(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Considers node degree similarity."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    new_weights = [[0.0] * max_node for _ in range(max_node)]

    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            degree_diff = abs(degrees1[i] - degrees2[j])
            new_weights[i][j] = 1.0 / (1 + degree_diff)  # Higher similarity for closer degrees

    return new_weights

def priority_v39(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Combines degree similarity with neighbor similarity."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    new_weights = [[0.0] * max_node for _ in range(max_node)]

    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            degree_similarity = 1.0 / (1 + abs(degrees1[i] - degrees2[j]))

            neighbor_similarity = 0.0
            neighbors1 = [k for k in range(n1) if graph1[i][k]]
            neighbors2 = [k for k in range(n2) if graph2[j][k]]

            for n1_neigh in neighbors1:
                for n2_neigh in neighbors2:
                      neighbor_similarity += weights[n1_neigh][n2_neigh] # Use initial weights for neighbor comparison

            new_weights[i][j] = degree_similarity + (neighbor_similarity / (len(neighbors1) * len(neighbors2) + 1e-6)) #Avoid division by zero


    return new_weights

def priority_v40(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities based on node degrees.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      weights[i][j] = 1.0 / (abs(degrees1[i] - degrees2[j]) + 1)

  return weights

def priority_v41(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Considers neighbor degrees for improved probability estimation."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for neighbor1 in range(n1):
        if graph1[i][neighbor1]:
          for neighbor2 in range(n2):
            if graph2[j][neighbor2]:
              neighbor_similarity += 1.0 / (abs(sum(graph1[neighbor1]) - sum(graph2[neighbor2])) + 1)
      weights[i][j] = neighbor_similarity

  return weights

def priority_v42(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Combines degree and neighbor degree similarity."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      degree_similarity = 1.0 / (abs(degrees1[i] - degrees2[j]) + 1)
      neighbor_similarity = 0
      for neighbor1 in range(n1):
        if graph1[i][neighbor1]:
          for neighbor2 in range(n2):
            if graph2[j][neighbor2]:
              neighbor_similarity += 1.0 / (abs(sum(graph1[neighbor1]) - sum(graph2[neighbor2])) + 1)

      weights[i][j] = (degree_similarity + neighbor_similarity) / 2  # Averaging the two similarities

  return weights

def priority_v43(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities based on node degrees.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  deg1 = [sum(row) for row in graph1]
  deg2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      weights[i][j] = 1.0 / (abs(deg1[i] - deg2[j]) + 1)  # Higher similarity for similar degrees

  return weights

def priority_v44(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Considers neighbor degrees in addition to node degrees.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  deg1 = [sum(row) for row in graph1]
  deg2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
          if graph1[i][k]:
              for l in range(n2):
                  if graph2[j][l]:
                      neighbor_similarity += 1.0 / (abs(deg1[k]-deg2[l]) + 1)

      weights[i][j] = (1.0 / (abs(deg1[i] - deg2[j]) + 1)) + neighbor_similarity


  return weights

def priority_v46(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Considers neighbor degrees in addition to node degrees.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      neighbor_diff = 0
      for k in range(n1):
        if graph1[i][k]:
          for l in range(n2):
            if graph2[j][l]:
              neighbor_diff += abs(degrees1[k] - degrees2[l])

      weights[i][j] = 1.0 / (1.0 + abs(degrees1[i] - degrees2[j]) + neighbor_diff)

  return weights

def priority_v47(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities for node mappings based on node degrees.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      weights[i][j] = 1.0 / (abs(degrees1[i] - degrees2[j]) + 1)  # Prioritize similar degrees

  return weights

def priority_v48(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities based on node degrees.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      weights[i][j] = 1.0 / (abs(degrees1[i] - degrees2[j]) + 1)  # Prioritize similar degrees

  return weights

def priority_v49(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Considers neighbor degrees in addition to node degrees.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
        if graph1[i][k]:
          for l in range(n2):
            if graph2[j][l]:
              neighbor_similarity += 1.0 / (abs(degrees1[k] - degrees2[l]) + 1)

      weights[i][j] = (1.0 / (abs(degrees1[i] - degrees2[j]) + 1)) * (neighbor_similarity + 1)  # Combine node and neighbor degree similarity

  return weights

def priority_v50(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    weights = [([0.0] * max_node) for _ in range(max_node)]
    
    # Precompute neighbor sets for efficiency
    neighbors1 = [set(idx for idx, val in enumerate(row) if val) for row in graph1]
    neighbors2 = [set(idx for idx, val in enumerate(row) if val) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            common_neighbors = 0
            for neighbor_i in neighbors1[i]:
                for neighbor_j in neighbors2[j]:
                    if neighbor_i < n1 and neighbor_j < n2 and graph1[neighbor_i][i] and graph2[neighbor_j][j]: # Check if edge exists
                        common_neighbors += 1
            
            degree_diff = abs(len(neighbors1[i]) - len(neighbors2[j]))
            
            # Combine degree difference and common neighbors
            weights[i][j] = (1.0 + common_neighbors) / (1.0 + degree_diff)  # Prioritize common neighbors

    return weights

def priority_v51(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    weights = [[0.0] * max_node for _ in range(max_node)]
    
    # Precompute neighbor sets for efficiency
    neighbors1 = [set(idx for idx, val in enumerate(row) if val) for row in graph1]
    neighbors2 = [set(idx for idx, val in enumerate(row) if val) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            common_neighbors = 0
            for neighbor_i in neighbors1[i]:
                for neighbor_j in neighbors2[j]:
                    if neighbor_i < n1 and neighbor_j < n2 and graph1[neighbor_i][i] and graph2[neighbor_j][j]:  # Check if edge exists
                        common_neighbors += 1
            
            # Consider both degree difference and common neighbors
            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
            similarity = (common_neighbors + 1) / (degree_diff + 1 + (n1*n2)**0.5/10) # Added a small constant to the denominator to prevent division by zero and to balance the impact of common neighbors vs degree difference. Also, using n1*n2 avoids favoring very small or very large graphs compared to medium sized graphs.
            weights[i][j] = similarity 

    return weights

def priority_v52(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    weights = [[0.0] * max_node for _ in range(max_node)]
    
    # Calculate node degrees and neighbor sets
    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]
    neighbors1 = [set(i for i, val in enumerate(row) if val) for row in graph1]
    neighbors2 = [set(i for i, val in enumerate(row) if val) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            # Degree difference
            degree_diff = abs(degrees1[i] - degrees2[j])

            # Common neighbor similarity
            common_neighbors = len(neighbors1[i].intersection(neighbors2[j]))
            
            # Incorporate both degree difference and common neighbors
            #  Higher weight for similar degrees and more common neighbors
            if degrees1[i] + degrees2[j] > 0: # Avoid division by zero
              weights[i][j] = (common_neighbors + 1) / (degree_diff + 1) # Added 1 for smoothing
            elif degrees1[i] == 0 and degrees2[j] == 0: # if both degrees are 0 give it a high similarity score
              weights[i][j] = 1.0
            else:
              weights[i][j] = 0.0


    return weights

def priority_v53(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]
  
  # Precompute neighbor sets for efficiency
  neighbors1 = [set(idx for idx, val in enumerate(row) if val) for row in graph1]
  neighbors2 = [set(idx for idx, val in enumerate(row) if val) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      common_neighbors = len(neighbors1[i].intersection(neighbors2[j]))
      union_neighbors = len(neighbors1[i].union(neighbors2[j]))
      
      if union_neighbors == 0:  # Handle cases where both nodes have no neighbors
          similarity = 1.0 if len(neighbors1[i]) == len(neighbors2[j]) else 0.0
      else:
          similarity = common_neighbors / union_neighbors  # Jaccard similarity

      weights[i][j] = similarity
  return weights

def priority_v54(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  if not weights or len(weights) != max_node or len(weights[0]) != max_node:
    weights = [[1.0 / max_node] * max_node for _ in range(max_node)]

  deg1 = [sum(row) for row in graph1]
  deg2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      degree_similarity = 1.0 - (abs(deg1[i] - deg2[j]) / max(1, deg1[i], deg2[j])) # Avoid division by zero

      neighbor_similarity = 0.0
      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] and graph2[j][l]:
            neighbor_similarity += weights[k][l]

      # Normalize neighbor similarity by the product of the degrees (if not zero) to prevent over-emphasis on highly connected nodes
      if deg1[i] * deg2[j] > 0:
          neighbor_similarity /= (deg1[i] * deg2[j])

      refined_weights[i][j] = weights[i][j] * (0.5 * degree_similarity + 0.5 * neighbor_similarity)

  for i in range(n1):
    row_sum = sum(refined_weights[i][:n2])
    if row_sum > 0:
      for j in range(n2):
        refined_weights[i][j] /= row_sum

  return refined_weights

def priority_v55(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  if weights is None or len(weights) != max_node or len(weights[0]) != max_node:
    weights = [[1.0 / max_node] * max_node for _ in range(max_node)]

  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      degree_similarity = 1.0 - (abs(degrees1[i] - degrees2[j]) / max(degrees1[i], degrees2[j], 1)) # Avoid division by zero
      neighbor_similarity = 0.0
      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] and graph2[j][l]:
            neighbor_similarity += weights[k][l]
      refined_weights[i][j] = weights[i][j] * (0.5 * degree_similarity + 0.5 * neighbor_similarity / max(1, degrees1[i] * degrees2[j])) # Avoid division by zero


  for i in range(n1):
    row_sum = sum(refined_weights[i][:n2])
    if row_sum > 0:
      for j in range(n2):
        refined_weights[i][j] /= row_sum

  return refined_weights

def priority_v56(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    if weights is None or len(weights) != max_node or len(weights[0]) != max_node:
        weights = [[1.0 / max_node] * max_node for _ in range(max_node)]

    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            degree_similarity = 1.0 - (abs(degrees1[i] - degrees2[j]) / max(1, degrees1[i], degrees2[j]))
            neighbor_similarity = 0.0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] and graph2[j][l]:
                        neighbor_similarity += weights[k][l]
            refined_weights[i][j] = weights[i][j] * (0.5 * degree_similarity + 0.5 * neighbor_similarity / max(1, degrees1[i] * degrees2[j]))

    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum
    return refined_weights

def priority_v57(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`. Uses numpy for efficiency."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    graph1_np = np.array(graph1)
    graph2_np = np.array(graph2)
    
    if weights is None:
        weights = np.full((max_node, max_node), 1.0 / max_node)
    elif not (len(weights) == max_node and len(weights[0]) == max_node):
        weights = np.full((max_node, max_node), 1.0 / max_node)
    else:
        weights = np.array(weights)

    refined_weights = np.zeros((max_node, max_node))

    deg1 = graph1_np.sum(axis=1)
    deg2 = graph2_np.sum(axis=1)

    for i in range(n1):
        for j in range(n2):
            degree_similarity = (1.0 - (np.abs(deg1[i] - deg2[j]) / max(deg1[i], deg2[j]))) if max(deg1[i], deg2[j]) > 0 else 0.0
            neighbor_similarity = np.sum(graph1_np[i, :][:, np.newaxis] * graph2_np[j, :] * weights[:n1, :n2]) #More efficient calculation
            refined_weights[i, j] = weights[i, j] * (0.5 * degree_similarity + 0.5 * neighbor_similarity / max(1, deg1[i] * deg2[j]))

    row_sums = refined_weights[:n1, :n2].sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    refined_weights[:n1, :n2] /= row_sums

    return refined_weights.tolist() # Convert back to list

def priority_v58(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`.  Uses numpy for efficiency and handles edge cases more robustly."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    if weights is None or len(weights) != max_node or len(weights[0]) != max_node:
        weights = np.full((max_node, max_node), 1.0 / max_node)
    else:
        weights = np.array(weights, dtype=float)

    refined_weights = np.zeros((max_node, max_node), dtype=float)
    degrees1 = np.sum(graph1, axis=1)
    degrees2 = np.sum(graph2, axis=1)


    for i in range(n1):
        for j in range(n2):
            degree_diff = abs(degrees1[i] - degrees2[j])
            max_degree = max(degrees1[i], degrees2[j])
            degree_similarity = (1.0 - (degree_diff / max_degree)) if max_degree > 0 else 0.0

            neighbor_similarity = 0.0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] and graph2[j][l]:
                        neighbor_similarity += weights[k, l]

            denominator = max(1, degrees1[i] * degrees2[j]) # Avoid division by zero
            refined_weights[i, j] = weights[i, j] * (0.5 * degree_similarity + 0.5 * neighbor_similarity / denominator)

    row_sums = np.sum(refined_weights[:n1, :n2], axis=1, keepdims=True) # More efficient row sum calculation
    row_sums[row_sums == 0] = 1  # Prevent division by zero where row sums are zero

    refined_weights[:n1, :n2] /= row_sums # Normalize in-place


    return refined_weights.tolist() # Convert back to list of lists for consistency

def priority_v59(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1` using numpy for efficiency."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    graph1_np = np.array(graph1)
    graph2_np = np.array(graph2)

    if weights is None:
        weights = np.full((max_node, max_node), 1.0 / max_node)
    elif not (len(weights) == max_node and len(weights[0]) == max_node):
        weights = np.full((max_node, max_node), 1.0 / max_node)
    else:
        weights = np.array(weights)


    refined_weights = np.zeros((max_node, max_node))

    degrees1 = graph1_np.sum(axis=1)
    degrees2 = graph2_np.sum(axis=1)

    for i in range(n1):
        for j in range(n2):
            degree_similarity = (1.0 - (abs(degrees1[i] - degrees2[j]) / max(degrees1[i], degrees2[j]))) if max(degrees1[i], degrees2[j]) > 0 else 0.0
            
            neighbor_similarity = (graph1_np[i, :][:, np.newaxis] * graph2_np[j, :]).astype(float) #Efficient neighbor similarity calculation
            neighbor_similarity = np.sum(neighbor_similarity * weights)

            refined_weights[i, j] = weights[i, j] * (0.5 * degree_similarity + 0.5 * neighbor_similarity / max(1, degrees1[i] * degrees2[j]))


    row_sums = refined_weights[:, :n2].sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    refined_weights[:n1, :n2] = (refined_weights[:n1, :n2].T / row_sums).T  # Normalize row-wise


    return refined_weights.tolist()

def priority_v60(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1` using numpy for efficiency."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    graph1_np = np.array(graph1)
    graph2_np = np.array(graph2)

    if weights is None:
        weights = np.full((max_node, max_node), 1.0 / max_node)
    elif not (len(weights) == max_node and len(weights[0]) == max_node):
        weights = np.full((max_node, max_node), 1.0 / max_node)
    else:
        weights = np.array(weights)

    refined_weights = np.zeros((max_node, max_node))

    deg1 = graph1_np.sum(axis=1)
    deg2 = graph2_np.sum(axis=1)

    for i in range(n1):
        for j in range(n2):
            degree_similarity = (1.0 - (abs(deg1[i] - deg2[j]) / max(deg1[i], deg2[j]))) if max(deg1[i], deg2[j]) > 0 else 0.0
            neighbor_similarity = (graph1_np[i, :][:, np.newaxis] * graph2_np[j, :]).sum() # Efficient neighbor similarity calculation
            refined_weights[i, j] = weights[i, j] * (0.5 * degree_similarity + 0.5 * neighbor_similarity / max(1, deg1[i] * deg2[j]))

    row_sums = refined_weights[:, :n2].sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    refined_weights[:, :n2] /= row_sums

    return refined_weights.tolist()

def priority_v61(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [([0.0] * max_node) for _ in range(max_node)]
    if ((weights is None) or (len(weights) != max_node) or (len(weights[0]) != max_node)):
        weights = [([(1.0 / max_node)] * max_node) for _ in range(max_node)]

    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            # Degree Similarity
            degree_similarity = (1.0 - (abs(degrees1[i] - degrees2[j]) / max(degrees1[i], degrees2[j], 1)))


            # Neighbor Similarity
            neighbor_similarity = 0.0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] and graph2[j][l]:
                        neighbor_similarity += weights[k][l]
            neighbor_similarity /= max(1, (degrees1[i] * degrees2[j]))


            refined_weights[i][j] = weights[i][j] * (0.5 * degree_similarity + 0.5 * neighbor_similarity)


    # Normalize rows
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum
        else: #Handle cases where row_sum is zero. Distribute probability equally among nodes in graph2
            for j in range(n2):
                refined_weights[i][j] = 1.0/n2


    return refined_weights

def priority_v62(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [([0.0] * max_node) for _ in range(max_node)]
    if not weights or len(weights) != max_node or len(weights[0]) != max_node:
        weights = [([(1.0 / max_node)] * max_node) for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            degree1 = sum(graph1[i])
            degree2 = sum(graph2[j])
            degree_similarity = (1.0 - abs(degree1 - degree2) / max(1, degree1, degree2)) if max(degree1, degree2) > 0 else 0.0  # Avoid division by zero

            neighbor_similarity = 0.0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] and graph2[j][l]:
                        neighbor_similarity += weights[k][l]

            # Normalize neighbor similarity by the product of degrees (only if product > 0)
            normalized_neighbor_similarity = neighbor_similarity / (degree1 * degree2) if (degree1 * degree2) > 0 else 0.0  

            refined_weights[i][j] = weights[i][j] * (0.5 * degree_similarity + 0.5 * normalized_neighbor_similarity)


    # Row-wise normalization
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum
        else: # If row sum is zero, distribute uniformly
            for j in range(n2):
                refined_weights[i][j] = 1.0 / n2


    return refined_weights

def priority_v63(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [([0.0] * max_node) for _ in range(max_node)]
    if ((weights is None) or (len(weights) != max_node) or (len(weights[0]) != max_node)):
        weights = [([(1.0 / max_node)] * max_node) for _ in range(max_node)]

    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            degree_similarity = (1.0 - (abs(degrees1[i] - degrees2[j]) / max(1, degrees1[i], degrees2[j])))

            neighbor_similarity = 0.0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] and graph2[j][l]:
                        neighbor_similarity += weights[k][l]
            
            refined_weights[i][j] = weights[i][j] * (0.5 * degree_similarity + 0.5 * neighbor_similarity / max(1, degrees1[i] * degrees2[j]))

    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum
    return refined_weights

def priority_v64(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [([0.0] * max_node) for _ in range(max_node)]
    if not weights or len(weights) != max_node or len(weights[0]) != max_node:
        weights = [([(1.0 / max_node)] * max_node) for _ in range(max_node)]

    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            degree_similarity = 1.0 - (abs(degrees1[i] - degrees2[j]) / max(1, degrees1[i], degrees2[j]))  # Avoid division by zero
            neighbor_similarity = 0.0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] and graph2[j][l]:
                        neighbor_similarity += weights[k][l]

            refined_weights[i][j] = weights[i][j] * (0.5 * degree_similarity + 0.5 * neighbor_similarity / max(1, degrees1[i] * degrees2[j])) # Avoid division by zero


    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum
        else: #Handle cases where row_sum is zero. Distribute probability evenly.
            for j in range(n2):
                refined_weights[i][j] = 1.0 / n2


    return refined_weights

def priority_v65(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [([0.0] * max_node) for _ in range(max_node)]

  if weights is None or len(weights) != max_node or len(weights[0]) != max_node:
    weights = [([(1.0 / max_node)] * max_node) for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      degree1 = sum(graph1[i])
      degree2 = sum(graph2[j])

      # Improved degree similarity calculation using a smoother function
      degree_similarity = 1.0 / (1.0 + abs(degree1 - degree2))


      neighbor_similarity = 0.0
      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] and graph2[j][l]:
            neighbor_similarity += weights[k][l]

      # Normalize neighbor similarity by the number of potential neighbor connections
      neighbor_similarity /= (degree1 * degree2) if (degree1 * degree2 > 0) else 1

      # Combine degree and neighbor similarity with more balanced weights
      refined_weights[i][j] = weights[i][j] * (0.5 * degree_similarity + 0.5 * neighbor_similarity)


  # Normalize rows to ensure they sum to 1
  for i in range(n1):
    row_sum = sum(refined_weights[i][:n2])
    if row_sum > 0:
      for j in range(n2):
        refined_weights[i][j] /= row_sum
    else: # Handle cases where row_sum is zero, distributing probability evenly
        for j in range(n2):
            refined_weights[i][j] = 1.0 / n2


  return refined_weights

def priority_v66(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    if not weights or len(weights) != max_node or len(weights[0]) != max_node:
        weights = [[1.0 / max_node] * max_node for _ in range(max_node)]

    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            degree_similarity = 1.0 - (abs(degrees1[i] - degrees2[j]) / max(degrees1[i], degrees2[j], 1))  # Avoid division by zero
            neighbor_similarity = 0.0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] and graph2[j][l]:
                        neighbor_similarity += weights[k][l]
            
            # Normalize neighbor similarity by the product of degrees (only if both degrees are non-zero)
            if degrees1[i] > 0 and degrees2[j] > 0:
                neighbor_similarity /= (degrees1[i] * degrees2[j])

            refined_weights[i][j] = weights[i][j] * (0.5 * degree_similarity + 0.5 * neighbor_similarity)

    # Row-wise normalization within the relevant submatrix (n1 x n2)
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum

    return refined_weights

def priority_v67(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [([0.0] * max_node) for _ in range(max_node)]

  if weights is None or len(weights) != max_node or len(weights[0]) != max_node:
    weights = [([1.0 / max_node] * max_node) for _ in range(max_node)]

  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
        # Improved degree similarity calculation using a more robust similarity measure
        degree_similarity = 1.0 - (abs(degrees1[i] - degrees2[j]) / max(1, degrees1[i], degrees2[j]))  # Avoid division by zero

        neighbor_similarity = 0.0
        for k in range(n1):
            for l in range(n2):
                if graph1[i][k] and graph2[j][l]:
                    neighbor_similarity += weights[k][l]

        # Combine degree and neighbor similarity with weights, potentially adding more sophisticated logic
        refined_weights[i][j] = weights[i][j] * (0.5 * degree_similarity + 0.5 * neighbor_similarity / max(1, degrees1[i] * degrees2[j]))

  # Normalize rows to ensure they sum to 1 (probability distribution)
  for i in range(n1):
    row_sum = sum(refined_weights[i][:n2])
    if row_sum > 0:
        for j in range(n2):
            refined_weights[i][j] /= row_sum
    else: # Handle the case when row_sum is zero, distribute probability evenly
      for j in range(n2):
        refined_weights[i][j] = 1.0/n2


  return refined_weights

def priority_v68(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [([0.0] * max_node) for _ in range(max_node)]
  if not weights or len(weights) != max_node or len(weights[0]) != max_node:
    weights = [([1.0 / max_node] * max_node) for _ in range(max_node)]

  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
        # Avoid division by zero and handle cases where degrees are zero
        degree_similarity = 0.0
        if degrees1[i] > 0 and degrees2[j] > 0:
            degree_similarity = 1.0 - (abs(degrees1[i] - degrees2[j]) / max(degrees1[i], degrees2[j]))
        elif degrees1[i] == 0 and degrees2[j] == 0:
            degree_similarity = 1.0  # Perfect match if both degrees are zero

        neighbor_similarity = 0.0
        for k in range(n1):
            for l in range(n2):
                if graph1[i][k] and graph2[j][l]:
                    neighbor_similarity += weights[k][l]
        
        # Avoid potential division by zero
        normalization_factor = max(1, degrees1[i] * degrees2[j])
        refined_weights[i][j] = weights[i][j] * (0.5 * degree_similarity + 0.5 * neighbor_similarity / normalization_factor)

  # Normalize rows
  for i in range(n1):
    row_sum = sum(refined_weights[i][:n2])
    if row_sum > 0:
        for j in range(n2):
            refined_weights[i][j] /= row_sum

  return refined_weights

