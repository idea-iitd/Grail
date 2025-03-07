#Num Unique Functions in logs: 964
import itertools
import numpy as np
import networkx as nx
import copy
import math
import heapq
import random
import math

def priority_v0(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights matrix with zeros
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Iterate through all possible node pairings
    for i in range(n1):
        for j in range(n2):
            # Calculate a similarity score based on node degrees and neighbor similarity
            degree_similarity = 1.0 / (1.0 + abs(sum(graph1[i]) - sum(graph2[j])))

            neighbor_similarity = 0.0
            common_neighbors = 0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] and graph2[j][l]:  # If both nodes have an edge to their respective neighbors
                        neighbor_similarity += weights[k][l]  # Add the weight of the neighbor mapping
                        common_neighbors += 1
            
            if common_neighbors > 0:
                neighbor_similarity /= common_neighbors

            # Combine the degree similarity and neighbor similarity (and potentially initial weights)
            refined_weights[i][j] = degree_similarity * (1 + neighbor_similarity)


    # Normalize the refined weights to probabilities (optional, but often beneficial)
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum

    return refined_weights

def priority_v1(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes a refined weight matrix based on graph structure similarity.

    Args:
        graph1: The adjacency matrix of the first graph.
        graph2: The adjacency matrix of the second graph.
        weights: A weight matrix representing the initial probabilities of mapping nodes between `graph1` and `graph2`.

    Returns:
        A refined weight matrix (float) reflecting structural similarity.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      similarity = 0
      for k in range(n1):
        for l in range(n2):
          similarity += graph1[i][k] * graph2[j][l]  # Neighbor structural similarity

      refined_weights[i][j] = similarity 

  return refined_weights

def priority_v2(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Further improved version of `priority_v0` incorporating initial weights."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      structural_similarity = 0
      for k in range(n1):
        for l in range(n2):
          structural_similarity += graph1[i][k] * graph2[j][l]

      if i < len(weights) and j < len(weights[i]):  # Handle potential size mismatch in weights
          refined_weights[i][j] = weights[i][j] * (1 + structural_similarity) # Combine initial weight and structural info
      else:
          refined_weights[i][j] = structural_similarity # Fallback if initial weight not available


  return refined_weights

def priority_v3(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights matrix with zeros
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Iterate through potential node mappings
    for i in range(n1):
        for j in range(n2):
            # Calculate a similarity score based on neighborhood structure
            score = 0.0

            # Consider node degrees as a basic similarity measure
            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
            score -= degree_diff  # Penalize large degree differences

            # Consider shared neighbors (common connections)
            neighbors1 = [k for k in range(n1) if graph1[i][k] == 1]
            neighbors2 = [k for k in range(n2) if graph2[j][k] == 1]

            common_neighbors = 0
            for n1 in neighbors1:
                for n2 in neighbors2:
                    if weights[n1][n2] > 0: # If there's a potential mapping between neighbors
                        common_neighbors +=1
            score += common_neighbors

            # Incorporate initial weights (if provided)
            if i < len(weights) and j < len(weights[i]):
                score += weights[i][j]


            # Normalize the score (optional, but can improve results)
            #  e.g., score = math.exp(score)  or a sigmoid function

            refined_weights[i][j] = score

    return refined_weights

def priority_v4(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`.

    Calculates refined node mapping probabilities based on structural similarity.
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
                    if graph1[i][k] == 1 and graph2[j][l] == 1:  # Connected nodes in both graphs
                        neighbor_similarity += weights[k][l]  # Add the weight of the neighbor mapping

            refined_weights[i][j] = neighbor_similarity


    # Normalize the weights (optional, but can improve performance)
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum


    return refined_weights

def priority_v5(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            node_similarity = weights[i][j]  # Use provided weights as a base

            # Consider edge similarity
            neighbors_i = [k for k in range(n1) if graph1[i][k] == 1]
            neighbors_j = [k for k in range(n2) if graph2[j][k] == 1]

            common_neighbors = 0
            for ni in neighbors_i:
                for nj in neighbors_j:
                     if weights[ni][nj] > 0: # Check if neighbors are likely to be mapped
                        common_neighbors += 1

            edge_similarity = (common_neighbors) / (len(neighbors_i) + len(neighbors_j) - common_neighbors + 1e-6) if (len(neighbors_i) + len(neighbors_j)) > 0 else 0 # Smoothing to handle cases with no neighbors


            refined_weights[i][j] = node_similarity + edge_similarity  # Combine node and edge similarity


    # Normalize the weights (optional, but can be helpful)
    for i in range(n1):
      row_sum = sum(refined_weights[i])
      if row_sum > 0:
        for j in range(max_node):
          refined_weights[i][j] /= row_sum


    return refined_weights

def priority_v6(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            node_similarity = weights[i][j]  # Use initial weights as a base

            # Penalize structural differences
            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
            neighbor_similarity = 0
            for k in range(n1):
                if graph1[i][k]:
                    for l in range(n2):
                        if graph2[j][l]:
                            neighbor_similarity += weights[k][l]


            refined_weights[i][j] = node_similarity - 0.1 * degree_diff + 0.05 * neighbor_similarity


    # Normalize weights (optional but recommended) to get probabilities
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum


    return refined_weights

def priority_v7(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`.
    Calculates node mapping priorities based on structural similarity.
    """
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            # Calculate node degree similarity
            degree_sim = 1.0 / (abs(sum(graph1[i]) - sum(graph2[j])) + 1)  # Inverse of degree difference

            # Calculate neighborhood similarity (consider common neighbors)
            neighbors1 = set(k for k in range(n1) if graph1[i][k])
            neighbors2 = set(k for k in range(n2) if graph2[j][k])

            common_neighbors = len(neighbors1.intersection(neighbors2))
            neighbor_sim = (common_neighbors + 1) / (max(len(neighbors1), len(neighbors2)) + 1) # Smoothing to avoid division by zero


            # Combine similarities (you can adjust the weights)
            refined_weights[i][j] = degree_sim * 0.5 + neighbor_sim * 0.5

            # Incorporate initial weights if provided.  If no initial weights are given
            # (all zeros as in priority_v0 example), this term has no effect.
            if weights and i < len(weights) and j < len(weights[0]):
                refined_weights[i][j] *= (weights[i][j] + 1e-6)  # Avoid multiplying by zero


    return refined_weights

def priority_v8(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights matrix with zeros
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Handle cases where input weights are not provided or have incorrect dimensions
    if weights is None or len(weights) != max_node or len(weights[0]) != max_node:
        weights = [[1.0 / max_node] * max_node for _ in range(max_node)]  # Uniform initial probabilities


    for i in range(n1):
        for j in range(n2):
            # Calculate a similarity score based on node degree and neighborhood similarity
            degree_similarity = 1.0 / (abs(sum(graph1[i]) - sum(graph2[j])) + 1)  # Inversely proportional to degree difference
            
            neighborhood_similarity = 0.0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] == 1 and graph2[j][l] == 1:
                        neighborhood_similarity += weights[k][l] # Encourage mappings that preserve neighborhood structure


            refined_weights[i][j] = weights[i][j] * degree_similarity + neighborhood_similarity # Combine initial weight, degree similarity and neighborhood similarity.
            
    # Normalize the refined weights (optional, but can improve performance)
    for i in range(n1):
      row_sum = sum(refined_weights[i])
      if row_sum > 0:
          for j in range(max_node):
              refined_weights[i][j] /= row_sum


    return refined_weights

def priority_v9(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

def priority_v10(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities for node mappings based on degree similarity.
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
      weights[i][j] = 1.0 / (1 + degree_diff)  # Higher similarity for closer degrees

  return weights

def priority_v11(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      weights[i][j] = 1.0 / (abs(degrees1[i] - degrees2[j]) + 1)  # Inversely proportional to degree difference

  return weights

def priority_v12(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
              neighbor_similarity += 1.0 / (abs(sum(graph1[neighbor1]) - sum(graph2[neighbor2])) + 1)

      weights[i][j] = (1.0 / (abs(sum(graph1[i]) - sum(graph2[j])) + 1)) * (neighbor_similarity + 1)  # Combine node and neighbor degree similarity

  return weights

def priority_v14(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
              neighbor_similarity += 1.0 / (abs(sum(graph1[neighbor1]) - sum(graph2[neighbor2])) + 1)

      weights[i][j] = (1.0 / (abs(sum(graph1[i]) - sum(graph2[j])) + 1)) + neighbor_similarity

  return weights

def priority_v15(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
            weights[i][j] = 1.0 / (1.0 + abs(deg1[i] - deg2[j]))

    return weights

def priority_v16(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
                            neighbor_similarity += 1.0 / (1.0 + abs(deg1[k] - deg2[l]))
            weights[i][j] = 1.0 / (1.0 + abs(deg1[i] - deg2[j])) + neighbor_similarity

    return weights

def priority_v17(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Computes initial probabilities based solely on node degrees."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    weights = [[0.0] * max_node for _ in range(max_node)]

    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            degree_diff = abs(degrees1[i] - degrees2[j])
            weights[i][j] = 1.0 / (degree_diff + 1)  # Higher similarity for closer degrees

    return weights

def priority_v18(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Considers neighbor degrees in addition to node degrees."""
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

            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
            weights[i][j] = (1.0 / (degree_diff + 1)) + neighbor_similarity

    return weights

def priority_v19(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities for node mappings based on node degrees.
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

def priority_v20(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Improves `priority_v0` by considering common neighbors.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      common_neighbors = 0
      for k in range(n1):
        if graph1[i][k] and any(graph2[j][l] for l in range(n2)): # Check if neighbors exist
          common_neighbors += 1
      weights[i][j] = 1.0 / (n1 + n2 - 2 * common_neighbors + 1e-6) # Avoid division by zero

  return weights

def priority_v21(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """
    Computes initial probabilities based solely on node degrees.
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

def priority_v22(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
            weights[i][j] = (1.0 / (1.0 + abs(degrees1[i] - degrees2[j]))) * (1 + neighbor_similarity)


    return weights

def priority_v23(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`.  Uses Jaccard similarity for neighbor comparison."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      degree_sim = 1.0 / (abs(sum(graph1[i]) - sum(graph2[j])) + 1)
      neighbors1 = set(k for k in range(n1) if graph1[i][k])
      neighbors2 = set(k for k in range(n2) if graph2[j][k])

      # Jaccard Similarity
      intersection = len(neighbors1.intersection(neighbors2))
      union = len(neighbors1.union(neighbors2))
      neighbor_sim = (intersection) / (union + 1e-6) if union else 0  # Handle empty sets and prevent division by zero

      refined_weights[i][j] = (degree_sim * 0.5) + (neighbor_sim * 0.5)

      if weights and i < len(weights) and j < len(weights[0]):
        refined_weights[i][j] *= (weights[i][j] + 1e-06)  # Incorporate prior weights

  return refined_weights

def priority_v24(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`.  Uses Jaccard similarity and incorporates weights more effectively."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbors1 = set(k for k in range(n1) if graph1[i][k])
      neighbors2 = set(k for k in range(n2) if graph2[j][k])

      if neighbors1 or neighbors2:  # Avoid division by zero if both nodes have no neighbors
        jaccard_sim = len(neighbors1.intersection(neighbors2)) / len(neighbors1.union(neighbors2)) if neighbors1 or neighbors2 else 1.0 #if both nodes have no neighbors, jaccard similarity should be 1
      else:
        jaccard_sim = 1.0 # Handles the case where both nodes are isolated.

      degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
      degree_sim = 1 / (degree_diff + 1)  # Higher similarity for smaller degree difference


      combined_sim = (jaccard_sim + degree_sim) / 2

      if weights and i < len(weights) and j < len(weights[0]):
        refined_weights[i][j] = combined_sim * (weights[i][j] + 1e-6) #Multiply directly, rather than adding a small constant
      else:
        refined_weights[i][j] = combined_sim

  return refined_weights

def priority_v25(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`. Uses Jaccard similarity for neighbor comparison."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            degree_sim = 1.0 / (abs(sum(graph1[i]) - sum(graph2[j])) + 1)
            neighbors1 = set(k for k in range(n1) if graph1[i][k])
            neighbors2 = set(k for k in range(n2) if graph2[j][k])

            # Jaccard Similarity
            intersection_size = len(neighbors1.intersection(neighbors2))
            union_size = len(neighbors1.union(neighbors2))
            if union_size == 0:  # Handle empty neighbor sets
                neighbor_sim = 1.0 
            else:
                neighbor_sim = intersection_size / union_size


            refined_weights[i][j] = (degree_sim * 0.5) + (neighbor_sim * 0.5)

            if weights and i < len(weights) and j < len(weights[0]):
                refined_weights[i][j] *= (weights[i][j] + 1e-6)

    return refined_weights

def priority_v26(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`.  Uses Jaccard similarity for neighbor comparison."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            degree_sim = 1.0 / (abs(sum(graph1[i]) - sum(graph2[j])) + 1)
            neighbors1 = set(k for k in range(n1) if graph1[i][k])
            neighbors2 = set(k for k in range(n2) if graph2[j][k])

            intersection_size = len(neighbors1.intersection(neighbors2))
            union_size = len(neighbors1.union(neighbors2))
            
            if union_size == 0: # Handle cases where both nodes have no neighbors
                neighbor_sim = 1.0  
            else:
                neighbor_sim = intersection_size / union_size # Jaccard Similarity

            refined_weights[i][j] = (degree_sim * 0.5) + (neighbor_sim * 0.5)

            if weights and i < len(weights) and j < len(weights[0]):
                refined_weights[i][j] *= (weights[i][j] + 1e-6)

    return refined_weights

def priority_v27(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [([0.0] * max_node) for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      
      # Optimization: Pre-calculate neighbor sets
      neighbors_i = {k for k in range(n1) if graph1[i][k]}
      neighbors_j = {l for l in range(n2) if graph2[j][l]}
      
      for k in neighbors_i:
        for l in neighbors_j:
          neighbor_similarity += weights[k][l]  # Use initial weights for influence
      
      # Incorporate node similarity (if available in 'weights')
      refined_weights[i][j] = weights[i][j] + neighbor_similarity  # Combine node and neighbor similarity

  # Normalize rows (optional, but often beneficial)
  for i in range(n1):
      row_sum = sum(refined_weights[i][:n2])
      if row_sum > 0:
          for j in range(n2):
              refined_weights[i][j] /= row_sum


  return refined_weights

def priority_v28(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
        for l in range(n2):
          neighbor_similarity += weights[k][l] * (graph1[i][k] * graph2[j][l])  # Consider weights
      refined_weights[i][j] = neighbor_similarity

  # Normalize rows (optional but often beneficial)
  for i in range(n1):
    row_sum = sum(refined_weights[i][:n2])
    if row_sum > 0:
      for j in range(n2):
        refined_weights[i][j] /= row_sum
  return refined_weights

def priority_v29(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes a refined weight matrix based on neighborhood similarity.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [([0.0] * max_node) for _ in range(max_node)]
  for i in range(n1):
    for j in range(n2):
        neighbor_similarity = 0
        for k in range(n1):
            for l in range(n2):
                if ((graph1[i][k] == 1) and (graph2[j][l] == 1)):
                    neighbor_similarity += weights[k][l]
        refined_weights[i][j] = neighbor_similarity
  for i in range(n1):
    row_sum = sum(refined_weights[i][:n2])
    if (row_sum > 0):
        for j in range(n2):
            refined_weights[i][j] /= row_sum
  return refined_weights

def priority_v30(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes a refined weight matrix based on common neighbors.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [([0.0] * max_node) for _ in range(max_node)]
  for i in range(n1):
    for j in range(n2):
        similarity = 0
        for k in range(n1):
            for l in range(n2):
                similarity += (graph1[i][k] * graph2[j][l])
        refined_weights[i][j] = similarity
  return refined_weights

def priority_v31(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes a refined weight matrix based on common neighbors, normalized by node degrees.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [([0.0] * max_node) for _ in range(max_node)]

  deg1 = [sum(row) for row in graph1]
  deg2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
        similarity = 0
        for k in range(n1):
            for l in range(n2):
                similarity += (graph1[i][k] * graph2[j][l])
        if deg1[i] > 0 and deg2[j] > 0:  # Avoid division by zero
            refined_weights[i][j] = similarity / (deg1[i] * deg2[j])
  return refined_weights

def priority_v32(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [([0.0] * max_node) for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
        for l in range(n2):
          neighbor_similarity += weights[k][l] * (graph1[i][k] * graph2[j][l])  # Consider initial weights
      refined_weights[i][j] = neighbor_similarity

  # Normalize rows for probabilities (optional but recommended)
  for i in range(n1):
      row_sum = sum(refined_weights[i][:n2])
      if row_sum > 0:
          for j in range(n2):
              refined_weights[i][j] /= row_sum
  return refined_weights

def priority_v33(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            # Degree Similarity
            degree1 = sum(graph1[i])
            degree2 = sum(graph2[j])
            degree_similarity = 1.0 / (1.0 + abs(degree1 - degree2))

            # Neighbor Similarity
            neighbor_similarity = 0.0
            common_neighbors = 0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] and graph2[j][l]:
                        neighbor_similarity += weights[k][l]
                        common_neighbors += 1
            if common_neighbors > 0:
                neighbor_similarity /= common_neighbors

            # Combine similarities (potentially with different weights)
            refined_weights[i][j] = degree_similarity * (1 + neighbor_similarity)  #Example weighting

    # Normalize rows
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum
    return refined_weights

def priority_v34(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1` focusing on efficiency and handling edge cases."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    degrees1 = [sum(row) for row in graph1]  # Pre-calculate degrees for efficiency
    degrees2 = [sum(row) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            degree_similarity = 1.0 / (1.0 + abs(degrees1[i] - degrees2[j]))
            neighbor_similarity = 0.0
            common_neighbors = 0

            # Optimized neighbor comparison
            neighbors1 = [k for k in range(n1) if graph1[i][k]]
            neighbors2 = [l for l in range(n2) if graph2[j][l]]

            for k in neighbors1:
                for l in neighbors2:
                    neighbor_similarity += weights[k][l]
                    common_neighbors += 1
            
            if common_neighbors > 0:
                neighbor_similarity /= common_neighbors
            refined_weights[i][j] = degree_similarity * (1 + neighbor_similarity)


        row_sum = sum(refined_weights[i][:n2])  # Only sum up to n2
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum
        # Handle case where row_sum is 0 (no similar neighbors) - distribute probability evenly
        elif n2 > 0: #Avoid division by zero if n2 is 0
            for j in range(n2):
                refined_weights[i][j] = 1.0 / n2

    return refined_weights

def priority_v35(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`.  Uses numpy for efficiency and handles cases where common_neighbors is 0 more robustly."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = np.zeros((max_node, max_node))
    graph1_np = np.array(graph1)
    graph2_np = np.array(graph2)
    weights_np = np.array(weights)

    for i in range(n1):
        for j in range(n2):
            degree_similarity = 1.0 / (1.0 + abs(graph1_np[i].sum() - graph2_np[j].sum()))

            neighbor_similarity = 0.0
            common_neighbors = (graph1_np[i, :, None] * graph2_np[None, j, :]).sum() # Efficiently count common neighbors

            if common_neighbors > 0:
                neighbor_similarity = (weights_np * (graph1_np[i, :, None] * graph2_np[None, j, :])).sum() / common_neighbors
            
            refined_weights[i, j] = degree_similarity * (1 + neighbor_similarity)

    for i in range(n1):
        row_sum = refined_weights[i, :n2].sum()
        if row_sum > 0:
            refined_weights[i, :n2] /= row_sum
    
    return refined_weights.tolist()  # Convert back to list for consistency

def priority_v36(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`.  Uses numpy for efficiency and handles edge cases where common_neighbors is 0."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = np.zeros((max_node, max_node), dtype=float)
  graph1_np = np.array(graph1)
  graph2_np = np.array(graph2)
  weights_np = np.array(weights)

  for i in range(n1):
    for j in range(n2):
      degree_similarity = 1.0 / (1.0 + abs(graph1_np[i].sum() - graph2_np[j].sum()))

      common_neighbors = np.sum(graph1_np[i, :][:, np.newaxis] * graph2_np[j, :]) # More efficient way to calculate common neighbors
      if common_neighbors > 0:
        neighbor_similarity = np.sum(weights_np * graph1_np[i, :][:, np.newaxis] * graph2_np[j, :]) / common_neighbors
      else:
        neighbor_similarity = 0.0 # Handle the case where there are no common neighbors

      refined_weights[i, j] = degree_similarity * (1 + neighbor_similarity)

  for i in range(n1):
    row_sum = refined_weights[i, :n2].sum()
    if row_sum > 0:
      refined_weights[i, :n2] /= row_sum

  return refined_weights.tolist() # Convert back to list of lists if needed

def priority_v37(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities of mapping nodes between two graphs based on degree differences.
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
        weights[i][j] = 1.0 / (1 + degree_diff)
  return weights

def priority_v38(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Improves `priority_v0` by considering neighbor degrees.
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
        neighbor_similarity = 0
        for k in range(n1):
            if graph1[i][k]:
                for l in range(n2):
                    if graph2[j][l]:
                        neighbor_similarity += 1.0 / (1 + abs(degrees1[k]-degrees2[l]))
        weights[i][j] = (1.0 / (1 + degree_diff)) * (1 + neighbor_similarity) 
  return weights

def priority_v39(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities for node mappings based on degree differences.
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
        weights[i][j] = 1.0 / (1 + degree_diff)
  return weights

def priority_v40(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      degree_diff = abs(degrees1[i] - degrees2[j])
      neighbor_diff = 0
      for neighbor1 in range(n1):
        if graph1[i][neighbor1]:
          for neighbor2 in range(n2):
            if graph2[j][neighbor2]:
              neighbor_diff += abs(degrees1[neighbor1] - degrees2[neighbor2])
      weights[i][j] = 1.0 / (1 + degree_diff + 0.1 * neighbor_diff)  # Reduced weight for neighbor difference
  return weights

def priority_v42(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Improves `priority_v0` by considering neighbor degrees.
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
              neighbor_similarity += 1.0 / (1 + abs(degrees1[k] - degrees2[l]))
      weights[i][j] = (1.0 / (1 + abs(degrees1[i] - degrees2[j]))) * (1 + neighbor_similarity)

  return weights

def priority_v43(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities of mapping nodes between two graphs based on degree difference.
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
        weights[i][j] = 1.0 / (1 + degree_diff)
  return weights

def priority_v44(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Improves `priority_v0` by considering neighbor degrees.
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
              neighbor_similarity += 1.0 / (1 + abs(degrees1[k] - degrees2[l]))
      degree_diff = abs(degrees1[i] - degrees2[j])
      weights[i][j] = (1.0 / (1 + degree_diff)) * (1 + neighbor_similarity)  # Combine degree difference and neighbor similarity
  return weights

def priority_v45(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`.  Uses Jaccard similarity and handles empty graphs."""
  n1 = len(graph1)
  n2 = len(graph2)

  if n1 == 0 or n2 == 0:  # Handle empty graphs
      max_node = max(n1, n2)
      return [[0.0] * max_node for _ in range(max_node)]


  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]
  for i in range(n1):
      for j in range(n2):
          degree_sim = 1.0 / (abs(sum(graph1[i]) - sum(graph2[j])) + 1)

          neighbors1 = set(k for k in range(n1) if graph1[i][k])
          neighbors2 = set(k for k in range(n2) if graph2[j][k])

          intersection_size = len(neighbors1.intersection(neighbors2))
          union_size = len(neighbors1.union(neighbors2))

          if union_size == 0:  # Handle cases where both neighborhoods are empty
              neighbor_sim = 1.0  # Consider them perfectly similar
          else:
              neighbor_sim = intersection_size / union_size # Jaccard similarity
          
          refined_weights[i][j] = (degree_sim * 0.5) + (neighbor_sim * 0.5)
          if weights and i < len(weights) and j < len(weights[0]):
              refined_weights[i][j] *= (weights[i][j] + 1e-6)

  return refined_weights

def priority_v46(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`.  Uses Jaccard similarity for neighbor comparison."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      degree_sim = 1.0 / (abs(sum(graph1[i]) - sum(graph2[j])) + 1)
      neighbors1 = set(k for k in range(n1) if graph1[i][k])
      neighbors2 = set(k for k in range(n2) if graph2[j][k])

      if neighbors1 or neighbors2: # Avoid division by zero if both sets are empty
          common_neighbors = len(neighbors1.intersection(neighbors2))
          all_neighbors = len(neighbors1.union(neighbors2))
          neighbor_sim = common_neighbors / all_neighbors if all_neighbors else 1.0 # Jaccard similarity
      else:
          neighbor_sim = 1.0 # If both nodes have no neighbors, they are perfectly similar

      refined_weights[i][j] = (degree_sim * 0.5) + (neighbor_sim * 0.5)

      if weights and i < len(weights) and j < len(weights[0]):
        refined_weights[i][j] *= (weights[i][j] + 1e-6)

  return refined_weights

def priority_v47(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            # Degree Similarity
            degree1 = sum(graph1[i])
            degree2 = sum(graph2[j])
            degree_diff = abs(degree1 - degree2)
            degree_sim = 1.0 / (degree_diff + 1)

            # Neighborhood Similarity
            neighbors1 = set(k for k in range(n1) if graph1[i][k])
            neighbors2 = set(k for k in range(n2) if graph2[j][k])
            common_neighbors = len(neighbors1.intersection(neighbors2))
            
            # Consider total number of possible connections for normalization
            max_possible_connections = min(len(neighbors1), len(neighbors2))
            if max_possible_connections > 0 :
              neighbor_sim = (common_neighbors) / (max_possible_connections)
            else:
              neighbor_sim = 1 if len(neighbors1) == len(neighbors2) == 0 else 0


            # Combine similarities (potentially with different weights)
            refined_weights[i][j] = (degree_sim * 0.5) + (neighbor_sim * 0.5)

            # Incorporate prior weights
            if weights and i < len(weights) and j < len(weights[0]):
                refined_weights[i][j] *= (weights[i][j] + 1e-6)  # Avoid multiplying by zero

    return refined_weights

def priority_v48(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`.  Uses Jaccard similarity for neighbor comparison."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      degree_sim = 1.0 / (abs(sum(graph1[i]) - sum(graph2[j])) + 1)
      neighbors1 = set(k for k in range(n1) if graph1[i][k])
      neighbors2 = set(k for k in range(n2) if graph2[j][k])

      if neighbors1 or neighbors2:  # Handle cases where either graph has isolated nodes.
          common_neighbors = len(neighbors1.intersection(neighbors2))
          all_neighbors = len(neighbors1.union(neighbors2))
          neighbor_sim = (common_neighbors) / (all_neighbors) if all_neighbors > 0 else 0  # Jaccard Similarity
      else:
          neighbor_sim = 1.0 if (not neighbors1 and not neighbors2) else 0  # Both isolated, so max similarity.
        
      refined_weights[i][j] = (degree_sim * 0.5) + (neighbor_sim * 0.5)

      if weights and i < len(weights) and j < len(weights[0]):
        refined_weights[i][j] *= (weights[i][j] + 1e-6)  # Incorporate prior weights.

  return refined_weights

def priority_v49(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`.  Normalizes across all cells, handles zero sums gracefully, and uses common neighbor counts."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]
    total_similarity = 0.0

    for i in range(n1):
        for j in range(n2):
            node_similarity = weights[i][j]
            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
            common_neighbors = 0
            for k in range(n1):
                if graph1[i][k]:
                    for l in range(n2):
                        if graph2[j][l] and weights[k][l] > 0: # Consider existing mappings
                            common_neighbors += 1


            refined_weights[i][j] = node_similarity - (0.1 * degree_diff) + (0.05 * common_neighbors)
            total_similarity += refined_weights[i][j]  # Accumulate total similarity

    # Normalize across all cells to ensure they sum up to 1 (or close to it if total_similarity is near zero).
    if total_similarity > 0:
        for i in range(n1):
            for j in range(n2):
                refined_weights[i][j] /= total_similarity
    elif n1 > 0 and n2 > 0: # Handle the case where total_similarity is zero - distribute probability evenly
        for i in range(n1):
            for j in range(n2):
                refined_weights[i][j] = 1.0 / (n1 * n2)

    return refined_weights

def priority_v50(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`.  Normalizes across all cells and incorporates neighbor similarity more effectively."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            node_similarity = weights[i][j]
            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
            neighbor_similarity = 0
            
            neighbors1 = [k for k in range(n1) if graph1[i][k]]
            neighbors2 = [l for l in range(n2) if graph2[j][l]]

            for k in neighbors1:
                for l in neighbors2:
                    neighbor_similarity += weights[k][l]
            
            # Normalize neighbor similarity by the number of potential neighbor connections
            if len(neighbors1) * len(neighbors2) > 0:
                neighbor_similarity /= (len(neighbors1) * len(neighbors2))

            refined_weights[i][j] = node_similarity + 0.05 * neighbor_similarity - 0.1 * degree_diff


    # Normalize across all cells
    total_weight = sum(sum(row[:n2]) for row in refined_weights[:n1])
    if total_weight > 0:
        for i in range(n1):
            for j in range(n2):
                refined_weights[i][j] /= total_weight

    return refined_weights

def priority_v51(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`.  Normalizes across all potential mappings,
    considers edge presence/absence, and uses a more balanced weighting scheme."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    total_similarity = 0  # For normalization across all pairs

    for i in range(n1):
        for j in range(n2):
            node_similarity = weights[i][j]
            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))

            neighbor_similarity = 0
            for k in range(n1):
                for l in range(n2):
                    edge_similarity = 0
                    if graph1[i][k] and graph2[j][l]:
                        edge_similarity = weights[k][l]
                    elif not graph1[i][k] and not graph2[j][l]:
                        edge_similarity = 1 - weights[k][l] # account for absence of edges

                    neighbor_similarity += edge_similarity


            refined_weights[i][j] = node_similarity - 0.1 * degree_diff + 0.2 * neighbor_similarity
            total_similarity += refined_weights[i][j]

    # Normalize across all pairs
    if total_similarity > 0:
        for i in range(n1):
            for j in range(n2):
                refined_weights[i][j] /= total_similarity
    
    # if total_similarity is 0 (unlikely, but possible with negative weights), 
    # assign uniform probabilities
    if total_similarity == 0:
        for i in range(n1):
            for j in range(n2):
                refined_weights[i][j] = 1.0 / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0
                
    return refined_weights

def priority_v52(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`.  Uses numpy for efficiency and normalizes across all possible mappings."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = np.zeros((max_node, max_node))
    graph1_np = np.array(graph1)
    graph2_np = np.array(graph2)
    weights_np = np.array(weights)

    for i in range(n1):
        for j in range(n2):
            node_similarity = weights_np[i, j]
            degree_diff = abs(graph1_np[i].sum() - graph2_np[j].sum())
            neighbor_similarity = (graph1_np[i] @ weights_np[:, j].T).sum()  # More efficient neighbor similarity calculation
            refined_weights[i, j] = node_similarity - (0.1 * degree_diff) + (0.05 * neighbor_similarity)

    # Normalize across all possible mappings (i.e., the entire matrix)
    total_sum = refined_weights.sum()
    if total_sum > 0:
        refined_weights /= total_sum
    
    return refined_weights.tolist()  # Convert back to list of lists if needed

def priority_v53(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            node_similarity = weights[i][j]
            neighbors_i = [k for k in range(n1) if graph1[i][k] == 1]
            neighbors_j = [k for k in range(n2) if graph2[j][k] == 1]

            edge_similarity = 0
            for ni in neighbors_i:
                for nj in neighbors_j:
                    edge_similarity += weights[ni][nj]  # Use weights instead of just counting

            refined_weights[i][j] = node_similarity + edge_similarity

    # Normalize rows (optional, but recommended for probabilities)
    for i in range(n1):
        row_sum = sum(refined_weights[i])
        if row_sum > 0:
            for j in range(max_node):
                refined_weights[i][j] /= row_sum

    return refined_weights

def priority_v54(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1` using numpy for efficiency."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  
  graph1_np = np.array(graph1)
  graph2_np = np.array(graph2)

  refined_weights = np.zeros((max_node, max_node), dtype=float)

  for i in range(n1):
      for j in range(n2):
          similarity = np.sum(graph1_np[i, :][:, np.newaxis] * graph2_np[j, :])  # Efficiently compute similarity
          refined_weights[i, j] = similarity

  return refined_weights.tolist()  # Convert back to list of lists

def priority_v55(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Further improved version considering initial weights and normalizing."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    graph1_np = np.array(graph1)
    graph2_np = np.array(graph2)
    weights_np = np.array(weights)

    refined_weights = np.zeros((max_node, max_node), dtype=float)

    for i in range(n1):
        for j in range(n2):
            structural_similarity = np.sum(graph1_np[i, :][:, np.newaxis] * graph2_np[j, :])
            refined_weights[i, j] = weights_np[i, j] * (1 + structural_similarity) # Incorporate initial weights

    # Normalize rows
    row_sums = refined_weights.sum(axis=1, keepdims=True)
    refined_weights = np.where(row_sums > 0, refined_weights / row_sums, 0) 

    return refined_weights.tolist()

def priority_v56(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            node_similarity = weights[i][j]
            neighbors_i = [k for k in range(n1) if graph1[i][k] == 1]
            neighbors_j = [k for k in range(n2) if graph2[j][k] == 1]

            common_neighbor_weights = 0
            for ni in neighbors_i:
                for nj in neighbors_j:
                    common_neighbor_weights += weights[ni][nj]  # Use weights instead of just counting

            edge_similarity = common_neighbor_weights / (len(neighbors_i) * len(neighbors_j) + 1e-6) if (len(neighbors_i) * len(neighbors_j) > 0) else 0
            refined_weights[i][j] = node_similarity + edge_similarity  # Combine node and edge similarity

    # Normalize rows (optional, but often helpful)
    for i in range(n1):
        row_sum = sum(refined_weights[i])
        if row_sum > 0:
            for j in range(max_node):
                refined_weights[i][j] /= row_sum

    return refined_weights

def priority_v57(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      node_similarity = weights[i][j]  # Use the provided node similarity

      neighbors_i = [k for k in range(n1) if graph1[i][k] == 1]
      neighbors_j = [k for k in range(n2) if graph2[j][k] == 1]

      edge_similarity = 0
      for ni in neighbors_i:
        for nj in neighbors_j:
          edge_similarity += weights[ni][nj] # Use weights here for neighbor similarity

      refined_weights[i][j] = node_similarity + edge_similarity


  # Normalize rows (optional but recommended for probabilities)
  for i in range(n1):
    row_sum = sum(refined_weights[i])
    if row_sum > 0:
      for j in range(max_node):
        refined_weights[i][j] /= row_sum

  return refined_weights

def priority_v58(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`.  Normalizes across all potential mappings."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [([0.0] * max_node) for _ in range(max_node)]
    for i in range(n1):
        for j in range(n2):
            node_similarity = weights[i][j]
            degree_diff = abs((sum(graph1[i]) - sum(graph2[j])))
            neighbor_similarity = 0
            for k in range(n1):
                if graph1[i][k]:
                    for l in range(n2):
                        if graph2[j][l]:
                            neighbor_similarity += weights[k][l]
            refined_weights[i][j] = ((node_similarity - (0.1 * degree_diff)) + (0.05 * neighbor_similarity))

    # Normalize across all weights, not just row-wise
    total_sum = sum(sum(row[:n2]) for row in refined_weights[:n1])
    if total_sum > 0:
        for i in range(n1):
            for j in range(n2):
                refined_weights[i][j] /= total_sum

    return refined_weights

def priority_v59(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`.  Normalizes across both rows and columns."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [([0.0] * max_node) for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            node_similarity = weights[i][j]
            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
            neighbor_similarity = 0
            for k in range(n1):
                if graph1[i][k]:
                    for l in range(n2):
                        if graph2[j][l]:
                            neighbor_similarity += weights[k][l]
            refined_weights[i][j] = (node_similarity - (0.1 * degree_diff) + (0.05 * neighbor_similarity))


    # Row normalization
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum

    # Column normalization
    for j in range(n2):
        col_sum = sum(refined_weights[i][j] for i in range(n1))
        if col_sum > 0:
            for i in range(n1):
                refined_weights[i][j] /= col_sum

    return refined_weights

def priority_v60(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`.  Normalizes across all potential mappings."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [([0.0] * max_node) for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            node_similarity = weights[i][j]
            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
            neighbor_similarity = 0
            for k in range(n1):
                if graph1[i][k]:
                    for l in range(n2):
                        if graph2[j][l]:
                            neighbor_similarity += weights[k][l]
            refined_weights[i][j] = (node_similarity - (0.1 * degree_diff) + (0.05 * neighbor_similarity))

    # Normalize across all mappings, not just row-wise
    total_sum = sum(sum(row[:n2]) for row in refined_weights[:n1])
    if total_sum > 0:
        for i in range(n1):
            for j in range(n2):
                refined_weights[i][j] /= total_sum

    return refined_weights

def priority_v61(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`.  Normalizes across both rows and columns
    and incorporates neighbor similarity more effectively."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            node_similarity = weights[i][j]
            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
            neighbor_similarity = 0

            neighbors1 = [k for k in range(n1) if graph1[i][k]]
            neighbors2 = [l for l in range(n2) if graph2[j][l]]

            for k in neighbors1:
                for l in neighbors2:
                    neighbor_similarity += weights[k][l]
            
            neighbor_similarity /= (len(neighbors1) * len(neighbors2) + 1e-6) # avoid division by zero

            refined_weights[i][j] = node_similarity + 0.1 * neighbor_similarity - 0.1 * degree_diff



    # Row normalization
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum

    # Column normalization (added for improved performance)
    for j in range(n2):
        col_sum = sum(refined_weights[i][j] for i in range(n1))
        if col_sum > 0:
            for i in range(n1):
                refined_weights[i][j] /= col_sum

    return refined_weights

def priority_v62(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]
    deg1 = [sum(row) for row in graph1]
    deg2 = [sum(row) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            neighbor_similarity = 0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] and graph2[j][l]:
                        neighbor_similarity += weights[k][l]  # Use input weights for neighbors

            degree_similarity = 1.0 / (1.0 + abs(deg1[i] - deg2[j]))
            refined_weights[i][j] = degree_similarity * (1 + neighbor_similarity) # Combine degree and neighbor similarity

    # Normalize the weights
    total_similarity = sum(sum(row) for row in refined_weights)
    if total_similarity > 0:
        for i in range(n1):
            for j in range(n2):
                refined_weights[i][j] /= total_similarity
    elif n1 > 0 and n2 > 0:  # Handle empty graphs or zero similarity
        for i in range(n1):
            for j in range(n2):
                refined_weights[i][j] = 1.0 / (n1 * n2)

    return refined_weights

def priority_v63(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]
    deg1 = [sum(row) for row in graph1]
    deg2 = [sum(row) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            neighbor_similarity = 0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] and graph2[j][l]:  # Check if edge exists
                        neighbor_similarity += weights[k][l]  # Use initial weights

            degree_similarity = 1.0 / (1.0 + abs(deg1[i] - deg2[j]))
            refined_weights[i][j] = (degree_similarity + neighbor_similarity)


    total_similarity = sum(sum(row) for row in refined_weights)

    if total_similarity > 0:
        for i in range(n1):
            for j in range(n2):
                refined_weights[i][j] /= total_similarity
    elif n1 > 0 and n2 > 0:
        for i in range(n1):
            for j in range(n2):
                refined_weights[i][j] = 1.0 / (n1 * n2)
    
    return refined_weights

def priority_v64(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]
    deg1 = [sum(row) for row in graph1]
    deg2 = [sum(row) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            degree_similarity = 1.0 / (1.0 + abs(deg1[i] - deg2[j]))

            neighbor_similarity = 0.0
            common_neighbors = 0
            for k in range(n1):
                if graph1[i][k]:
                    for l in range(n2):
                        if graph2[j][l]:
                            common_neighbors += weights[k][l]  # Use the input weights here

            if (deg1[i] > 0 and deg2[j] > 0):
                neighbor_similarity = common_neighbors / (deg1[i] * deg2[j])
            elif (deg1[i] == 0 and deg2[j] == 0):  # Handle cases where both degrees are 0
                neighbor_similarity = 1.0 # maximum similarity


            refined_weights[i][j] = (degree_similarity + neighbor_similarity) / 2 # Combine both similarities


    # Normalize the weights
    total_similarity = sum(sum(row) for row in refined_weights)
    if total_similarity > 0:
        for i in range(n1):
            for j in range(n2):
                refined_weights[i][j] /= total_similarity
    elif n1 > 0 and n2 > 0:  # Handle empty graphs or cases where all similarities are 0
        for i in range(n1):
            for j in range(n2):
                refined_weights[i][j] = 1.0 / (n1 * n2)


    return refined_weights

def priority_v65(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]
    deg1 = [sum(row) for row in graph1]
    deg2 = [sum(row) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            neighbor_similarity = 0
            for k in range(n1):
                if graph1[i][k]:
                    for l in range(n2):
                        if graph2[j][l]:
                            neighbor_similarity += weights[k][l]  # Use input weights here

            degree_similarity = 1.0 / (1.0 + abs(deg1[i] - deg2[j]))
            refined_weights[i][j] = (degree_similarity + neighbor_similarity)

    # Normalize the weights
    total_similarity = sum(sum(row) for row in refined_weights)
    if total_similarity > 0:
        for i in range(n1):
            for j in range(n2):
                refined_weights[i][j] /= total_similarity
    elif n1 > 0 and n2 > 0:  # Handle empty graphs or no similarity
        for i in range(n1):
            for j in range(n2):
                refined_weights[i][j] = 1.0 / (n1 * n2)

    return refined_weights

def priority_v66(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [([0.0] * max_node) for _ in range(max_node)]
  deg1 = [sum(row) for row in graph1]
  deg2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
        for l in range(n2):
          neighbor_similarity += graph1[i][k] * graph2[j][l]  # Check for common neighbors

      degree_difference = abs(deg1[i] - deg2[j])
      weights[i][j] = (1.0 + neighbor_similarity) / (1.0 + degree_difference) # Combine neighbor similarity and degree difference

  return weights

