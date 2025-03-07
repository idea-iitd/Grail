#Num Unique Functions in logs: 527
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

    # Initialize the refined weights matrix
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Handle cases where initial weights are not provided or are of incorrect dimensions
    if weights is None or len(weights) != max_node or len(weights[0]) != max_node:
        weights = [[1.0 / max_node] * max_node for _ in range(max_node)]  # Uniform initial probabilities

    # Calculate node degrees for both graphs
    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            # Consider node degrees and initial weights in the refined probability calculation
            degree_similarity = 1.0 - abs(degrees1[i] - degrees2[j]) / max(1, degrees1[i] + degrees2[j])  # Similarity based on degrees
            refined_weights[i][j] = weights[i][j] * degree_similarity  # Combine initial weight and degree similarity

            # Consider neighborhood similarity (edges)
            neighbors1 = [k for k in range(n1) if graph1[i][k]]
            neighbors2 = [k for k in range(n2) if graph2[j][k]]
            
            common_neighbors = 0
            for n1_neighbor in neighbors1:
                for n2_neighbor in neighbors2:
                    common_neighbors += weights[n1_neighbor][n2_neighbor] # Weighted common neighbors

            refined_weights[i][j] *= (1 + common_neighbors) # Boost probability based on neighbor similarity


    # Normalize the refined weights (ensure they sum to 1 for each row)
    for i in range(n1):
      row_sum = sum(refined_weights[i][:n2]) # Sum only up to valid nodes in graph2
      if row_sum > 0:  # Avoid division by zero
          for j in range(n2):
              refined_weights[i][j] /= row_sum

    return refined_weights

def priority_v1(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`.
    Calculates node mapping priorities based on structural similarity.
    """
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            degree_similarity = 1.0 / (abs(sum(graph1[i]) - sum(graph2[j])) + 1)  # Inversely proportional to degree difference

            neighbor_similarity = 0
            neighbors1 = [k for k in range(n1) if graph1[i][k] == 1]
            neighbors2 = [k for k in range(n2) if graph2[j][k] == 1]

            for n1_neighbor in neighbors1:
                for n2_neighbor in neighbors2:
                    neighbor_similarity += weights[n1_neighbor][n2_neighbor] # Encourage mappings that align neighbors

            refined_weights[i][j] = degree_similarity + neighbor_similarity


    # Normalize weights (optional but recommended)
    total_weight = sum(sum(row) for row in refined_weights)
    if total_weight > 0:  # Avoid division by zero
        for i in range(max_node):
            for j in range(max_node):
                refined_weights[i][j] /= total_weight


    return refined_weights

def priority_v2(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
                    if weights[neighbor1][neighbor2] > 0:  # Check if neighbors are likely mapped
                        common_neighbors += 1

            similarity = 0.0
            if len(neighbors1) > 0 and len(neighbors2) > 0:
                similarity = common_neighbors / (math.sqrt(len(neighbors1) * len(neighbors2)))

            # Combine initial weight and neighborhood similarity
            refined_weights[i][j] = weights[i][j] * (1 + similarity)  # Emphasize nodes with similar neighborhoods


    # Normalize weights (optional, but can be helpful)
    total_weight = sum(sum(row) for row in refined_weights)
    if total_weight > 0:
        for i in range(max_node):
            for j in range(max_node):
                refined_weights[i][j] /= total_weight


    return refined_weights

def priority_v3(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities for node mappings between two graphs based on structural similarity.

    Args:
        graph1: The adjacency matrix of the first graph.
        graph2: The adjacency matrix of the second graph.
        weights:  A placeholder weight matrix.  Its contents are ignored and overwritten.

    Returns:
        A weight matrix (float) where each entry represents the initial probability of a node in `graph1` 
        being mapped to a node in `graph2`.  Higher values indicate higher similarity.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      degree1 = sum(graph1[i])
      degree2 = sum(graph2[j])
      # Similarity based on degree difference (can be expanded)
      weights[i][j] = 1.0 / (1.0 + abs(degree1 - degree2)) 

  return weights

def priority_v4(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0` incorporating neighbor degree comparison."""

  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      degree1 = sum(graph1[i])
      degree2 = sum(graph2[j])
      neighbor_similarity = 0
      for neighbor1 in range(n1):
          if graph1[i][neighbor1]:
              for neighbor2 in range(n2):
                  if graph2[j][neighbor2]:
                      neighbor_degree1 = sum(graph1[neighbor1])
                      neighbor_degree2 = sum(graph2[neighbor2])
                      neighbor_similarity += 1.0 / (1.0 + abs(neighbor_degree1 - neighbor_degree2))
      
      weights[i][j] = (1.0 / (1.0 + abs(degree1 - degree2)))  + neighbor_similarity

  # Normalize weights (optional but often beneficial)
  max_weight = max(max(row) for row in weights)
  if max_weight > 0:  # Avoid division by zero
      for i in range(n1):
          for j in range(n2):
              weights[i][j] /= max_weight
              
  return weights

def priority_v5(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

def priority_v6(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

def priority_v7(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes a priority matrix based on structural similarity.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  priority_matrix = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
      priority_matrix[i][j] = 1.0 / (1 + degree_diff)  # Higher priority for similar degrees

  return priority_matrix

def priority_v8(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version using neighbor degree comparison."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  priority_matrix = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for neighbor1 in range(n1):
          if graph1[i][neighbor1]:
              for neighbor2 in range(n2):
                  if graph2[j][neighbor2]:
                      neighbor_similarity += 1.0 / (1 + abs(sum(graph1[neighbor1]) - sum(graph2[neighbor2])))
      priority_matrix[i][j] = neighbor_similarity

  return priority_matrix

def priority_v9(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Combines degree difference and neighbor similarity."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  priority_matrix = [[0.0] * max_node for _ in range(max_node)]


  degree_priority = priority_v0(graph1, graph2, weights)
  neighbor_priority = priority_v1(graph1, graph2, weights)

  for i in range(n1):
    for j in range(n2):
      priority_matrix[i][j] = degree_priority[i][j] * neighbor_priority[i][j]  # Combine both metrics


  return priority_matrix

def priority_v10(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

def priority_v11(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

      weights[i][j] = (1.0 / (abs(sum(graph1[i]) - sum(graph2[j])) + 1)) + neighbor_similarity

  return weights

def priority_v12(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      weights[i][j] = 1.0 / (1.0 + degree_diff)  # Higher similarity for closer degrees

  return weights

def priority_v13(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights matrix with zeros
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Iterate through the potential mappings between nodes in graph1 and graph2
    for i in range(n1):
        for j in range(n2):
            # Calculate a similarity score based on node degrees and neighbor connectivity
            degree_similarity = 1.0 - abs(sum(graph1[i]) - sum(graph2[j])) / max(sum(graph1[i]), sum(graph2[j]), 1) if (sum(graph1[i]) > 0 or sum(graph2[j])> 0) else 0.0


            neighbor_similarity = 0.0
            common_neighbors = 0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] == 1 and graph2[j][l] == 1:  # if k is a neighbor of i and l is a neighbor of j
                         neighbor_similarity += weights[k][l]  # Add the weight indicating k maps to l
                         common_neighbors += 1



            if common_neighbors > 0:
                neighbor_similarity /= common_neighbors

            # Combine the degree similarity, neighbor similarity, and initial weight
            refined_weights[i][j] = (degree_similarity + neighbor_similarity + weights[i][j]) / 3



    return refined_weights

def priority_v14(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`.
    Calculates node mapping priorities based on structural similarity.
    """
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            # Calculate node similarity based on neighborhood structure
            neighbors1 = [k for k in range(n1) if graph1[i][k]]
            neighbors2 = [k for k in range(n2) if graph2[j][k]]

            common_neighbors = 0
            for n1_neighbor in neighbors1:
              for n2_neighbor in neighbors2:
                if weights[n1_neighbor][n2_neighbor] > 0: # Check if neighbors are also likely mapped
                  common_neighbors += 1


            degree1 = len(neighbors1)
            degree2 = len(neighbors2)

            # Similarity based on common neighbors and degree difference
            if degree1 + degree2 > 0:  # Avoid division by zero
                similarity = (2 * common_neighbors) / (degree1 + degree2)
            else:
                similarity = 0  # If both nodes have degree 0, consider them similar


            refined_weights[i][j] = similarity


    return refined_weights

def priority_v15(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights with zeros
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Handle cases where initial weights are not provided or are incorrect dimensions.
    if weights is None or len(weights) != max_node or len(weights[0]) != max_node:
        weights = [[1.0 / max_node] * max_node for _ in range(max_node)] # Uniform initial probabilities


    for i in range(n1):
        for j in range(n2):
            # Calculate node similarity based on neighborhood structure.
            neighbors_i = [k for k in range(n1) if graph1[i][k] == 1]
            neighbors_j = [k for k in range(n2) if graph2[j][k] == 1]

            common_neighbors = 0
            for neighbor_i in neighbors_i:
                for neighbor_j in neighbors_j:
                     if weights[neighbor_i][neighbor_j] > 0: # Check if neighbors are likely to be mapped.
                        common_neighbors += weights[neighbor_i][neighbor_j]


            degree_i = len(neighbors_i)
            degree_j = len(neighbors_j)

            # Combine initial weight, neighborhood similarity, and degree difference
            if degree_i > 0 and degree_j > 0:
                neighborhood_similarity = common_neighbors / (degree_i * degree_j)**0.5 if common_neighbors > 0 else 0  #Normalize
                degree_difference = abs(degree_i - degree_j)
                refined_weights[i][j] = weights[i][j] * (1 + neighborhood_similarity) * (1 / (1 + degree_difference))  # Reward similar degrees and penalize difference
            elif degree_i == 0 and degree_j == 0: # Both nodes are isolated
                refined_weights[i][j] = weights[i][j]
            else: #  One node isolated and the other is not - low probability
                refined_weights[i][j] = weights[i][j] * 0.1




    # Normalize refined weights (optional but recommended).
    row_sums = [sum(row) for row in refined_weights]
    for i in range(n1):
        for j in range(n2):
            if row_sums[i] > 0:
                refined_weights[i][j] /= row_sums[i]
            else:
                refined_weights[i][j] = 0.0 # handle division by zero

    return refined_weights

def priority_v16(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
          if graph1[i][k] == 1 and graph2[j][l] == 1:  # Check for edge existence in both graphs
            neighbor_similarity += weights[k][l] if k < len(weights) and l < len(weights[k]) else 0

      refined_weights[i][j] = neighbor_similarity  # Update weight based on neighborhood similarity

  return refined_weights

def priority_v17(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0` with normalization and initial weight consideration."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      deg_i = sum(graph1[i])
      deg_j = sum(graph2[j])

      if deg_i > 0 and deg_j > 0: # Avoid division by zero
        for k in range(n1):
          for l in range(n2):
            if graph1[i][k] == 1 and graph2[j][l] == 1:
              neighbor_similarity += weights[k][l] if k < len(weights) and l < len(weights[k]) else 0
        refined_weights[i][j] = (neighbor_similarity / (deg_i * deg_j)) * (weights[i][j] if i < len(weights) and j < len(weights[i]) else 0) # Normalize and incorporate initial weight
      elif deg_i == 0 and deg_j == 0: # Handle nodes with no edges. Give some weight if initial weights exist.
          refined_weights[i][j] = weights[i][j] if i < len(weights) and j < len(weights[i]) else 0


  return refined_weights

def priority_v18(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities for node mappings between two graphs.

    Args:
        graph1: The adjacency matrix of the first graph.
        graph2: The adjacency matrix of the second graph.
        weights:  A placeholder for initial weights (not used in this version).

    Returns:
        A matrix of initial probabilities, where each entry (i, j) represents the 
        probability of mapping node i from graph1 to node j from graph2.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  result = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      result[i][j] = 1.0 / (max_node) # Uniform initial probabilities

  return result

def priority_v19(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0` considering node degrees."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  result = [[0.0] * max_node for _ in range(max_node)]

  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      degree_diff = abs(degrees1[i] - degrees2[j])
      result[i][j] = 1.0 / (degree_diff + 1)  # Higher probability for similar degrees

  # Normalize probabilities
  for i in range(n1):
    row_sum = sum(result[i][:n2])
    if row_sum > 0:
      for j in range(n2):
        result[i][j] /= row_sum


  return result

def priority_v20(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1` considering neighbor degrees."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  result = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
          for l in range(n2):
              if graph1[i][k] and graph2[j][l]: #If edge exists in both graphs
                  neighbor_similarity += 1 / (abs(sum(graph1[k])-sum(graph2[l]))+1) #Add similarity based on neighbor degrees
      result[i][j] = neighbor_similarity


  # Normalize probabilities
  for i in range(n1):
    row_sum = sum(result[i][:n2])
    if row_sum > 0:
      for j in range(n2):
        result[i][j] /= row_sum

  return result

def priority_v21(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

def priority_v22(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Considers neighbors' degrees in addition to node degrees.
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
      weights[i][j] = 1.0 / (abs(sum(graph1[i]) - sum(graph2[j])) + 1) + neighbor_similarity

  return weights

def priority_v24(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Considers neighbor degrees in addition to node degrees."""
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
                      neighbor_diff += abs(degrees1[k]-degrees2[l])
      weights[i][j] = 1.0 / (1.0 + abs(degrees1[i] - degrees2[j]) + neighbor_diff)


  return weights

def priority_v25(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      weights[i][j] = 1.0 / (abs(degrees1[i] - degrees2[j]) + 1)  # Inversely proportional to degree difference

  return weights

def priority_v26(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Refines initial probabilities using neighbor similarity.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  new_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for neighbor1 in range(n1):
        if graph1[i][neighbor1]:
          for neighbor2 in range(n2):
            if graph2[j][neighbor2]:
              neighbor_similarity += weights[neighbor1][neighbor2]
      new_weights[i][j] = weights[i][j] * (1 + neighbor_similarity) # Combine degree-based and neighbor similarity

  return new_weights

def priority_v27(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Normalizes the weights to represent probabilities.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  new_weights = [[0.0] * max_node for _ in range(max_node)]


  for i in range(n1):
    row_sum = sum(weights[i][:n2])  # Normalize only within possible mappings
    if row_sum > 0:
      for j in range(n2):
        new_weights[i][j] = weights[i][j] / row_sum
  return new_weights

def priority_v28(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights matrix with zeros
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Iterate through all possible node pairings
    for i in range(n1):
        for j in range(n2):
            # Calculate a similarity score based on node degrees and neighborhood structure
            degree_similarity = 1.0 - abs(sum(graph1[i]) - sum(graph2[j])) / (max(sum(graph1[i]), sum(graph2[j])) + 1e-6)  # Avoid division by zero

            neighbor_similarity = 0
            common_neighbors = 0
            for k in range(n1):
                if graph1[i][k] == 1:
                    for l in range(n2):
                        if graph2[j][l] == 1 and weights[k][l] > 0:  # Consider existing weights as a prior
                            common_neighbors += 1
                            break
            neighbor_similarity = common_neighbors / (max(sum(graph1[i]), sum(graph2[j])) + 1e-6)

            # Combine the similarity scores and the initial weight (if available)
            if i < len(weights) and j < len(weights[i]):
                combined_score = (degree_similarity + neighbor_similarity) / 2 * (weights[i][j] + 1) # Incorporate initial weights
            else:
                combined_score = (degree_similarity + neighbor_similarity) / 2

            refined_weights[i][j] = combined_score

    return refined_weights

def priority_v29(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
            
            edge_similarity = (common_neighbors) / (max(1, len(neighbors_i) * len(neighbors_j)))  # Avoid division by zero
            
            # Combine node and edge similarity (adjust weights as needed)
            refined_weights[i][j] = node_similarity + edge_similarity  # Simple additive combination.
            # Other combinations like multiplying or using a weighted average are possible.

    # Normalize weights (optional, but can be helpful)
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum

    return refined_weights

def priority_v30(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)

  # Initialize the refined weights with zeros
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  # Iterate through potential node mappings
  for i in range(n1):
    for j in range(n2):
      # Calculate a similarity score based on node degree and neighborhood similarity
      degree_similarity = 1.0 / (abs(sum(graph1[i]) - sum(graph2[j])) + 1)  # Add 1 to avoid division by zero

      neighborhood_similarity = 0
      for k in range(n1):
          for l in range(n2):
              if graph1[i][k] == 1 and graph2[j][l] == 1: # Check if both nodes have an edge to their respective neighbors
                  neighborhood_similarity += weights[k][l]  # Use initial weights to contribute to similarity


      # Combine the similarity measures and initial weights (if available)
      refined_weights[i][j] = degree_similarity * (neighborhood_similarity + 1e-6) # add small constant to prevent zero values

        
      if len(weights) > 0 and len(weights[0]) > 0:
            refined_weights[i][j] *= weights[min(i, len(weights) -1)][min(j,len(weights[0]) - 1)]



  # Normalize the weights (optional but recommended for probabilities)
  total_weight = sum(sum(row) for row in refined_weights)
  if total_weight > 0:
      for i in range(max_node):
          for j in range(max_node):
              refined_weights[i][j] /= total_weight


  return refined_weights

def priority_v31(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights matrix
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Handle cases where input weights are not properly sized
    if len(weights) != max_node or not all(len(row) == max_node for row in weights):
        weights = [[0.0] * max_node for _ in range(max_node)]

    # Calculate node degrees for both graphs
    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]


    for i in range(n1):
        for j in range(n2):
            # Consider existing weights (prior information)
            refined_weights[i][j] = weights[i][j]

            # Incorporate node degree similarity
            degree_diff = abs(degrees1[i] - degrees2[j])
            degree_similarity = 1.0 / (degree_diff + 1)  # Higher similarity for smaller degree difference
            refined_weights[i][j] += degree_similarity

            # Incorporate neighborhood similarity (using common neighbors)
            neighbors1 = [k for k in range(n1) if graph1[i][k]]
            neighbors2 = [k for k in range(n2) if graph2[j][k]]
            common_neighbors = sum(1 for k in neighbors1 if any(graph2[l][j] for l in neighbors2 if l<n2 and k<n1))
            refined_weights[i][j] += common_neighbors


    # Normalize refined weights (optional, but can improve performance)
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])  # Normalize only within the valid range
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum



    return refined_weights

def priority_v32(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights matrix with zeros
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Iterate through all possible node mappings
    for i in range(n1):
        for j in range(n2):
            # Calculate a score based on structural similarity
            score = 0.0

            # Consider node degrees
            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
            score -= degree_diff  # Penalize large degree differences

            # Consider neighborhood similarity (common neighbors)
            neighbors_i = set(k for k in range(n1) if graph1[i][k])
            neighbors_j = set(k for k in range(n2) if graph2[j][k])
            common_neighbors = len(neighbors_i.intersection(neighbors_j))
            score += common_neighbors  # Reward common neighbors


            # Incorporate initial weights (if available)
            if i < len(weights) and j < len(weights[0]):
                score += weights[i][j]

            refined_weights[i][j] = score


    # Normalize the refined weights to probabilities (optional)
    # This step can be helpful if you want to interpret the weights as probabilities.
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])  # Sum only over valid indices for graph2
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum


    return refined_weights

def priority_v33(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)

  # Initialize the refined weights matrix with zeros
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  # Iterate through potential node mappings
  for i in range(n1):
    for j in range(n2):
      # Calculate a similarity score based on node degree and neighborhood similarity
      degree_similarity = 1.0 - abs(sum(graph1[i]) - sum(graph2[j])) / max(sum(graph1[i]), sum(graph2[j]), 1)  # Avoid division by zero
      neighborhood_similarity = 0.0
      for neighbor1 in range(n1):
        for neighbor2 in range(n2):
          if graph1[i][neighbor1] and graph2[j][neighbor2]:  # If both nodes have an edge to their respective neighbors
            neighborhood_similarity += weights[neighbor1][neighbor2]  # Use initial weights to estimate neighborhood similarity
      
      # Combine the initial weight, degree similarity, and neighborhood similarity
      refined_weights[i][j] = weights[i][j] * degree_similarity * (1 + neighborhood_similarity)  # Prioritize mappings with similar degree and connected neighborhoods

  # Normalize the refined weights (optional, but can be helpful)
  for i in range(n1):
      row_sum = sum(refined_weights[i][:n2]) # Only normalize up to n2
      if row_sum > 0:
          for j in range(n2):
              refined_weights[i][j] /= row_sum

  return refined_weights

def priority_v34(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights matrix with zeros
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Iterate through all possible node mappings
    for i in range(n1):
        for j in range(n2):
            # Calculate a score based on node degree and edge similarity
            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
            edge_similarity = 0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] == 1 and graph2[j][l] == 1:
                       edge_similarity += weights[k][l]  # Use initial weights here

            # Combine the scores and update the refined weights
            score =  edge_similarity #- degree_diff  # Prioritize edge similarity, penalize degree difference
            refined_weights[i][j] = score


    # Normalize the refined weights to represent probabilities
    total_score = sum(sum(row) for row in refined_weights)
    if total_score > 0:
        for i in range(n1):
            for j in range(n2):
                refined_weights[i][j] /= total_score


    return refined_weights

def priority_v35(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`.
    Calculates node mapping priorities based on structural similarity.
    """
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            degree_similarity = 1.0 / (abs(sum(graph1[i]) - sum(graph2[j])) + 1)  # Consider degree difference
            neighbor_similarity = 0
            for k in range(n1):
                if graph1[i][k]:
                    for l in range(n2):
                        if graph2[j][l]:
                             neighbor_similarity += weights[k][l] # Consider neighbor mappings

            refined_weights[i][j] = (degree_similarity + neighbor_similarity) # Combine similarities

    # Normalize weights (optional, but can improve performance)
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum


    return refined_weights

def priority_v36(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`.  Normalizes neighbor similarity based on the number of neighbors."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      degree_similarity = 1.0 / (abs(sum(graph1[i]) - sum(graph2[j])) + 1)
      neighbor_similarity = 0
      neighbors1 = [k for k in range(n1) if graph1[i][k] == 1]
      neighbors2 = [k for k in range(n2) if graph2[j][k] == 1]
      
      n_neighbors1 = len(neighbors1)
      n_neighbors2 = len(neighbors2)

      for n1_neighbor in neighbors1:
        for n2_neighbor in neighbors2:
          neighbor_similarity += weights[n1_neighbor][n2_neighbor]

      if n_neighbors1 > 0 and n_neighbors2 > 0: # Normalize if there are neighbors
          neighbor_similarity /= (n_neighbors1 * n_neighbors2)
      
      refined_weights[i][j] = degree_similarity + neighbor_similarity
      
  total_weight = sum(sum(row) for row in refined_weights)
  if total_weight > 0:
    for i in range(max_node):
      for j in range(max_node):
        refined_weights[i][j] /= total_weight
  return refined_weights

def priority_v37(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
      degree_similarity = 1.0 / (degree_diff + 1)

      neighbor_similarity = 0
      neighbors1 = [k for k in range(n1) if graph1[i][k] == 1]
      neighbors2 = [k for k in range(n2) if graph2[j][k] == 1]

      if neighbors1 and neighbors2:  # Only compute if both nodes have neighbors
          for n1_neighbor in neighbors1:
              for n2_neighbor in neighbors2:
                  neighbor_similarity += weights[n1_neighbor][n2_neighbor]
          # Normalize neighbor similarity by the number of neighbor pairs considered
          neighbor_similarity /= (len(neighbors1) * len(neighbors2))


      refined_weights[i][j] = degree_similarity + neighbor_similarity

  total_weight = sum(sum(row) for row in refined_weights)
  if total_weight > 0:
    for i in range(max_node):
      for j in range(max_node):
        refined_weights[i][j] /= total_weight
  return refined_weights

def priority_v38(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`.  Normalizes neighbor similarity based on 
  the number of neighbors to avoid overemphasis on nodes with many neighbors.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [([0.0] * max_node) for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      degree_similarity = (1.0 / (abs((sum(graph1[i]) - sum(graph2[j]))) + 1))
      neighbor_similarity = 0
      neighbors1 = [k for k in range(n1) if graph1[i][k] == 1]
      neighbors2 = [k for k in range(n2) if graph2[j][k] == 1]

      n_neighbors1 = len(neighbors1)
      n_neighbors2 = len(neighbors2)

      for n1_neighbor in neighbors1:
        for n2_neighbor in neighbors2:
          neighbor_similarity += weights[n1_neighbor][n2_neighbor]

      if n_neighbors1 > 0 and n_neighbors2 > 0:  # Normalize if there are neighbors
        neighbor_similarity /= (n_neighbors1 * n_neighbors2)

      refined_weights[i][j] = (degree_similarity + neighbor_similarity)

  total_weight = sum(sum(row) for row in refined_weights)
  if total_weight > 0:
    for i in range(max_node):
      for j in range(max_node):
        refined_weights[i][j] /= total_weight
  return refined_weights

def priority_v39(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`.  Normalizes neighbor similarity by the number of neighbors."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      degree_similarity = 1.0 / (abs(sum(graph1[i]) - sum(graph2[j])) + 1)
      neighbor_similarity = 0
      neighbors1 = [k for k in range(n1) if graph1[i][k] == 1]
      neighbors2 = [k for k in range(n2) if graph2[j][k] == 1]
      n1_neighbors_count = len(neighbors1)
      n2_neighbors_count = len(neighbors2)

      for n1_neighbor in neighbors1:
        for n2_neighbor in neighbors2:
          neighbor_similarity += weights[n1_neighbor][n2_neighbor]
      
      if n1_neighbors_count > 0 and n2_neighbors_count > 0:
          neighbor_similarity /= (n1_neighbors_count * n2_neighbors_count)  # Normalize
      
      refined_weights[i][j] = degree_similarity + neighbor_similarity

  total_weight = sum(sum(row) for row in refined_weights)
  if total_weight > 0:
    for i in range(max_node):
      for j in range(max_node):
        refined_weights[i][j] /= total_weight
  return refined_weights

def priority_v40(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      weights[i][j] = 1.0 / (1.0 + degree_diff)
  return weights

def priority_v41(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Considers neighbor similarity in addition to degree difference.
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
              neighbor_similarity += 1

      weights[i][j] = (1.0 / (1.0 + degree_diff)) + (neighbor_similarity / (n1 * n2))  # Combine degree diff and neighbor similarity
  return weights

def priority_v42(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
        weights[i][j] = 1.0 / (1.0 + degree_diff)
  return weights

def priority_v43(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Considers neighbor degrees for improved probability estimation.
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
              neighbor_similarity += 1.0 / (1.0 + abs(degrees1[k]-degrees2[l]))
      degree_diff = abs(degrees1[i] - degrees2[j])
      weights[i][j] = (1.0 / (1.0 + degree_diff)) + neighbor_similarity

  return weights

def priority_v44(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
        weights[i][j] = 1.0 / (1.0 + degree_diff)
  return weights

def priority_v45(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
                      neighbor_similarity += 1.0 / (1.0 + abs(degrees1[k]-degrees2[l]))
      weights[i][j] = (1.0 / (1.0 + abs(degrees1[i] - degrees2[j]))) * (1 + neighbor_similarity)

  return weights

def priority_v47(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      for neighbor1 in range(n1):
          if graph1[i][neighbor1]:
              for neighbor2 in range(n2):
                  if graph2[j][neighbor2]:
                      neighbor_similarity += 1.0 / (1.0 + abs(degrees1[neighbor1] - degrees2[neighbor2]))

      degree_diff = abs(degrees1[i] - degrees2[j])
      weights[i][j] = (1.0 / (1.0 + degree_diff)) + neighbor_similarity
  return weights

def priority_v48(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
        weights[i][j] = 1.0 / (1.0 + degree_diff)
  return weights

def priority_v49(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
                      neighbor_similarity += 1.0 / (1.0 + abs(degrees1[k] - degrees2[l]))
      weights[i][j] = (1.0 / (1.0 + abs(degrees1[i] - degrees2[j]))) * (1 + neighbor_similarity)

  return weights

def priority_v51(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      for neighbor1 in range(n1):
        if graph1[i][neighbor1]:
          for neighbor2 in range(n2):
            if graph2[j][neighbor2]:
              neighbor_similarity += 1.0 / (1.0 + abs(degrees1[neighbor1] - degrees2[neighbor2]))
      weights[i][j] = (1.0 / (1.0 + abs(degrees1[i] - degrees2[j]))) * (1 + neighbor_similarity)
  return weights

def priority_v53(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Considers neighbor similarity in addition to degree difference.
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
              neighbor_similarity += 1

      weights[i][j] = 1.0 / (1.0 + degree_diff) + neighbor_similarity / (n1 * n2) # Combine degree diff and neighbor similarity

  return weights

def priority_v55(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      for k in range(n1):
        if graph1[i][k]:
          for l in range(n2):
            if graph2[j][l]:
              neighbor_diff += abs(degrees1[k] - degrees2[l])
      weights[i][j] = 1.0 / (1.0 + degree_diff + neighbor_diff)
  return weights

def priority_v56(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1` considering neighbor degrees."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    weights = [([0.0] * max_node) for _ in range(max_node)]
    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]

    neighbor_degrees1 = [[degrees1[k] for k in range(n1) if graph1[i][k]] for i in range(n1)]
    neighbor_degrees2 = [[degrees2[k] for k in range(n2) if graph2[j][k]] for j in range(n2)]

    for i in range(n1):
        for j in range(n2):
            degree_diff = abs(degrees1[i] - degrees2[j])
            neighbor_diff = 0
            
            for n1_deg in neighbor_degrees1[i]:
                min_diff = float('inf')
                for n2_deg in neighbor_degrees2[j]:
                    min_diff = min(min_diff, abs(n1_deg - n2_deg))
                if min_diff != float('inf'):
                    neighbor_diff += min_diff
                    
            weights[i][j] = 1.0 / (1.0 + degree_diff + neighbor_diff)

    return weights

def priority_v57(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      weights[i][j] = (1.0 / (1.0 + abs(degrees1[i] - degrees2[j])))
  return weights

def priority_v58(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities based on neighbor degree similarity.
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
              neighbor_similarity += 1.0 / (1.0 + abs(sum(graph1[neighbor1]) - sum(graph2[neighbor2])))
      weights[i][j] = neighbor_similarity
  return weights

def priority_v59(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Combines degree difference and neighbor similarity for initial probabilities.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]
  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      degree_similarity = 1.0 / (1.0 + abs(degrees1[i] - degrees2[j]))
      neighbor_similarity = 0
      for neighbor1 in range(n1):
        if graph1[i][neighbor1]:
          for neighbor2 in range(n2):
            if graph2[j][neighbor2]:
              neighbor_similarity += 1.0 / (1.0 + abs(sum(graph1[neighbor1]) - sum(graph2[neighbor2])))
      weights[i][j] = degree_similarity * (1 + neighbor_similarity) # Combine both metrics

  return weights

def priority_v60(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      weights[i][j] = (1.0 / (1.0 + abs(degrees1[i] - degrees2[j])))
  return weights

def priority_v61(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Improves `priority_v0` by considering neighbor degrees.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]
  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]
  neighbor_degrees1 = [[degrees1[k] for k in range(n1) if graph1[i][k]] for i in range(n1)]
  neighbor_degrees2 = [[degrees2[k] for k in range(n2) if graph2[j][k]] for j in range(n2)]

  for i in range(n1):
    for j in range(n2):
      degree_diff = abs(degrees1[i] - degrees2[j])
      neighbor_diff = 0
      for n1_deg in neighbor_degrees1[i]:
          min_diff = float('inf')
          for n2_deg in neighbor_degrees2[j]:
              min_diff = min(min_diff, abs(n1_deg - n2_deg))
          if min_diff != float('inf'):
              neighbor_diff += min_diff
      weights[i][j] = 1.0 / (1.0 + degree_diff + neighbor_diff)
  return weights

def priority_v63(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Improves `priority_v0` by considering neighbor degrees.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]
  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]
  
  neighbor_degrees1 = [[degrees1[k] for k in range(n1) if graph1[i][k]] for i in range(n1)]
  neighbor_degrees2 = [[degrees2[k] for k in range(n2) if graph2[j][k]] for j in range(n2)]

  for i in range(n1):
    for j in range(n2):
      degree_diff = abs(degrees1[i] - degrees2[j])
      neighbor_diff = 0
      for n1_deg in neighbor_degrees1[i]:
          min_diff = float('inf')
          for n2_deg in neighbor_degrees2[j]:
              min_diff = min(min_diff, abs(n1_deg - n2_deg))
          if min_diff != float('inf'):
              neighbor_diff += min_diff
      weights[i][j] = 1.0 / (1.0 + degree_diff + neighbor_diff) 
  return weights

def priority_v64(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    weights = [([0.0] * max_node) for _ in range(max_node)]
    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]
    neighbor_degrees1 = [[degrees1[k] for k in range(n1) if graph1[i][k]] for i in range(n1)]
    neighbor_degrees2 = [[degrees2[k] for k in range(n2) if graph2[j][k]] for j in range(n2)]

    for i in range(n1):
        for j in range(n2):
            degree_diff = abs(degrees1[i] - degrees2[j])
            neighbor_diff = 0
            
            # Optimized neighbor difference calculation using sorting and early stopping
            n1_neighbors_sorted = sorted(neighbor_degrees1[i])
            n2_neighbors_sorted = sorted(neighbor_degrees2[j])
            
            k = 0
            l = 0
            while k < len(n1_neighbors_sorted) and l < len(n2_neighbors_sorted):
                diff = abs(n1_neighbors_sorted[k] - n2_neighbors_sorted[l])
                neighbor_diff += diff
                if n1_neighbors_sorted[k] < n2_neighbors_sorted[l]:
                    k += 1
                elif n1_neighbors_sorted[k] > n2_neighbors_sorted[l]:
                    l += 1
                else:
                    k += 1
                    l += 1

            # Handle remaining neighbors if lists are of different lengths
            while k < len(n1_neighbors_sorted):
                neighbor_diff += n1_neighbors_sorted[k]  # Assuming comparison to 0 for remaining elements
                k += 1
            while l < len(n2_neighbors_sorted):
                neighbor_diff += n2_neighbors_sorted[l]  # Assuming comparison to 0 for remaining elements
                l += 1

            weights[i][j] = 1.0 / (1.0 + degree_diff + neighbor_diff)

    return weights

def priority_v65(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [([0.0] * max_node) for _ in range(max_node)]
  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]
  neighbor_degrees1 = [[degrees1[k] for k in range(n1) if graph1[i][k]] for i in range(n1)]
  neighbor_degrees2 = [[degrees2[k] for k in range(n2) if graph2[j][k]] for j in range(n2)]

  for i in range(n1):
    for j in range(n2):
      degree_diff = abs(degrees1[i] - degrees2[j])
      neighbor_diff = 0
      
      # Efficiently calculate neighbor difference using sorting
      n1_deg_sorted = sorted(neighbor_degrees1[i])
      n2_deg_sorted = sorted(neighbor_degrees2[j])
      
      len_n1 = len(n1_deg_sorted)
      len_n2 = len(n2_deg_sorted)
      
      p1 = 0
      p2 = 0
      while p1 < len_n1 and p2 < len_n2:
          neighbor_diff += abs(n1_deg_sorted[p1] - n2_deg_sorted[p2])
          p1 += 1
          p2 += 1
          
      # Account for remaining neighbors in the larger list
      while p1 < len_n1:
          neighbor_diff += abs(n1_deg_sorted[p1] - degrees2[j])  # Compare with degree of j
          p1 += 1
      while p2 < len_n2:
          neighbor_diff += abs(degrees1[i] - n2_deg_sorted[p2])  # Compare with degree of i
          p2 += 1

      denominator = 1.0 + degree_diff + neighbor_diff
      if denominator == 0: #handle the case where denominator is 0, avoiding division by zero.
        weights[i][j] = 1.0
      else:
        weights[i][j] = 1.0 / denominator 
  return weights

def priority_v66(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    weights = [([0.0] * max_node) for _ in range(max_node)]
    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]

    neighbor_degrees1 = [[degrees1[k] for k in range(n1) if graph1[i][k]] for i in range(n1)]
    neighbor_degrees2 = [[degrees2[k] for k in range(n2) if graph2[j][k]] for j in range(n2)]

    for i in range(n1):
        for j in range(n2):
            degree_diff = abs(degrees1[i] - degrees2[j])
            neighbor_diff = 0
            
            # Use a more efficient comparison of neighbor degrees
            n1_nbr_deg = sorted(neighbor_degrees1[i])
            n2_nbr_deg = sorted(neighbor_degrees2[j])
            
            len_n1_nbr = len(n1_nbr_deg)
            len_n2_nbr = len(n2_nbr_deg)
            
            p1 = 0
            p2 = 0
            while p1 < len_n1_nbr and p2 < len_n2_nbr:
                neighbor_diff += abs(n1_nbr_deg[p1] - n2_nbr_deg[p2])
                p1 += 1
                p2 += 1

            # Account for remaining unmatched neighbors
            while p1 < len_n1_nbr:
                neighbor_diff += n1_nbr_deg[p1]  # or some penalty for missing neighbor
                p1 += 1
            while p2 < len_n2_nbr:
                neighbor_diff += n2_nbr_deg[p2]  # or some penalty for missing neighbor
                p2 += 1


            weights[i][j] = 1.0 / (1.0 + degree_diff + neighbor_diff)

    return weights

def priority_v67(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
            # Degree similarity
            degree_diff = abs(degrees1[i] - degrees2[j])
            degree_sum = degrees1[i] + degrees2[j]
            degree_similarity = 1.0 if degree_sum == 0 else (1.0 - (degree_diff / degree_sum))

            # Neighbor similarity
            neighbors1 = [k for k in range(n1) if graph1[i][k]]
            neighbors2 = [k for k in range(n2) if graph2[j][k]]
            common_neighbors = 0
            for n1_neighbor in neighbors1:
                for n2_neighbor in neighbors2:
                    common_neighbors += weights[n1_neighbor][n2_neighbor]

            refined_weights[i][j] = weights[i][j] * degree_similarity * (1 + common_neighbors)

    # Normalize rows
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum

    return refined_weights

def priority_v68(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
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
            # Degree similarity
            degree_diff = abs(degrees1[i] - degrees2[j])
            degree_sum = max(1, degrees1[i] + degrees2[j])  # Avoid division by zero
            degree_similarity = 1.0 - (degree_diff / degree_sum)

            # Neighbor similarity
            neighbors1 = [k for k in range(n1) if graph1[i][k]]
            neighbors2 = [k for k in range(n2) if graph2[j][k]]
            common_neighbors = 0
            for n1_neighbor in neighbors1:
                for n2_neighbor in neighbors2:
                    common_neighbors += weights[n1_neighbor][n2_neighbor]


            refined_weights[i][j] = weights[i][j] * (1 + common_neighbors) * degree_similarity  # Combine similarities



    # Normalize rows (important for probabilities)
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum
        else: #Handle cases where row_sum is zero - distribute probability evenly
            for j in range(n2):
                refined_weights[i][j] = 1.0 / n2


    return refined_weights

def priority_v69(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
            # Degree Similarity
            degree_diff = abs(degrees1[i] - degrees2[j])
            degree_sum = degrees1[i] + degrees2[j]
            degree_similarity = 1.0 - (degree_diff / max(1, degree_sum)) if degree_sum else 1.0  # Handle cases where both degrees are 0

            # Neighbor Similarity
            neighbors1 = [k for k in range(n1) if graph1[i][k]]
            neighbors2 = [k for k in range(n2) if graph2[j][k]]
            common_neighbors = sum(weights[n1_neighbor][n2_neighbor] for n1_neighbor in neighbors1 for n2_neighbor in neighbors2)


            refined_weights[i][j] = weights[i][j] * (1 + common_neighbors) * degree_similarity # Combine similarities


    # Normalize rows
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum
        else: # Handle cases where row_sum is zero - distribute probability evenly
            for j in range(n2):
                refined_weights[i][j] = 1.0 / n2


    return refined_weights

def priority_v70(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
            # Degree similarity
            degree_diff = abs(degrees1[i] - degrees2[j])
            degree_sum = degrees1[i] + degrees2[j]
            degree_similarity = 1.0 - (degree_diff / max(1, degree_sum)) if degree_sum > 0 else 0.0  # Handle cases where degree_sum is 0

            refined_weights[i][j] = weights[i][j] * degree_similarity

            # Neighbor similarity
            neighbors1 = [k for k in range(n1) if graph1[i][k]]
            neighbors2 = [k for k in range(n2) if graph2[j][k]]

            common_neighbors = 0
            for n1_neighbor in neighbors1:
                for n2_neighbor in neighbors2:
                    common_neighbors += weights[n1_neighbor][n2_neighbor]

            refined_weights[i][j] *= (1 + common_neighbors)


    # Normalize rows to ensure they sum to 1 (or 0 if no similarity)
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum

    return refined_weights

