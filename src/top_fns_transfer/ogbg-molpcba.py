#Num Unique Functions in logs: 44
import itertools
import numpy as np
import networkx as nx
import copy
import math
import heapq
import random
import math

def priority_v0(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`.
    Calculates node similarity based on neighborhood structure and initial weights.
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
                    if graph1[i][k] == 1 and graph2[j][l] == 1:  # Check if both nodes have an edge
                        neighbor_similarity += weights[k][l]  # Add the weight of the corresponding edge

            refined_weights[i][j] = weights[i][j] + neighbor_similarity  # Combine initial weight and neighborhood similarity

    # Normalize the weights (optional, but can be helpful)
    max_weight = 0
    for i in range(n1):
        for j in range(n2):
            max_weight = max(max_weight, refined_weights[i][j])

    if max_weight > 0:  # Avoid division by zero
        for i in range(n1):
            for j in range(n2):
                refined_weights[i][j] /= max_weight

    return refined_weights

def priority_v1(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
          if graph1[i][k] == 1 and graph2[j][l] == 1:  # Check for edge existence
            neighbor_similarity += weights[k][l] # Add weight if neighbors are also likely mapped

      refined_weights[i][j] = neighbor_similarity  # Update refined weight based on neighborhood similarity

  return refined_weights

def priority_v2(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0` with normalization."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      neighbor_count = 0
      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] == 1 and graph2[j][l] == 1:  # Check for edge existence
            neighbor_similarity += weights[k][l]  # Add weight if neighbors are also likely mapped
            neighbor_count += 1


      if neighbor_count > 0:
        refined_weights[i][j] = neighbor_similarity / neighbor_count  # Normalize by the number of neighbor pairs
      else:
        refined_weights[i][j] = 0 #If no neighbors are found the refined weight should be 0



  return refined_weights

def priority_v3(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes a refined weight matrix based on graph structure similarity.

    Args:
        graph1: The adjacency matrix of the first graph.
        graph2: The adjacency matrix of the second graph.
        weights: An initial weight matrix.

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
      for neighbor1 in range(n1):
        if graph1[i][neighbor1]:
          for neighbor2 in range(n2):
            if graph2[j][neighbor2]:
              neighbor_similarity += weights[neighbor1][neighbor2]  # Use initial weights here

      refined_weights[i][j] = neighbor_similarity

  return refined_weights

def priority_v4(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Further improved version of `priority_v0` with normalization and initial weight consideration."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      norm_factor = 0

      for neighbor1 in range(n1):
        if graph1[i][neighbor1]:
          for neighbor2 in range(n2):
            if graph2[j][neighbor2]:
              neighbor_similarity += weights[neighbor1][neighbor2]
              norm_factor += 1  # Increment for each potential neighbor pair


      if norm_factor > 0:
        refined_weights[i][j] = (neighbor_similarity / norm_factor) * weights[i][j] # Incorporate initial weights and normalize
      else:  # Handle cases where a node has no neighbors. Could use initial weights as fallback
        refined_weights[i][j] = weights[i][j]

  return refined_weights

def priority_v5(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights matrix with zeros
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Iterate through the possible mappings between nodes in graph1 and graph2
    for i in range(n1):
        for j in range(n2):
            # Calculate a score based on node degree similarity and initial weights
            degree_similarity = 1.0 - abs(sum(graph1[i]) - sum(graph2[j])) / (max(sum(graph1[i]), sum(graph2[j])) + 1e-6)  # Avoid division by zero
            
            # Incorporate the initial weights if provided. If weights are not provided, default to a uniform probability
            initial_weight = weights[i][j] if i < len(weights) and j < len(weights[i]) else 1.0 / (n1 * n2) 

            refined_weights[i][j] = degree_similarity * initial_weight


    # Normalize the refined weights so that they sum to 1 for each node in graph1
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2]) # Only sum up to the valid nodes in graph2
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum  # Normalize

    return refined_weights

def priority_v6(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights matrix with zeros
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Handle cases where weights matrix is not properly initialized or sized
    if not weights or len(weights) != max_node or any(len(row) != max_node for row in weights):
        # Initialize with uniform probabilities if weights are missing or incorrect
        initial_prob = 1.0 / max_node
        weights = [[initial_prob] * max_node for _ in range(max_node)]

    # Iterate through possible node mappings and compute a similarity score
    for i in range(n1):
        for j in range(n2):
            # Calculate a similarity score based on node degree and neighborhood similarity
            degree_similarity = 1.0 / (1.0 + abs(sum(graph1[i]) - sum(graph2[j])))  # Compare node degrees

            neighborhood_similarity = 0.0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] == 1 and graph2[j][l] == 1: # Check for common neighbors
                        neighborhood_similarity += weights[k][l] # Use initial weights to gauge neighbor similarity

            # Combine the similarities and the initial weight
            refined_weights[i][j] = weights[i][j] * degree_similarity * (1 + neighborhood_similarity)

    # Normalize the refined weights to represent probabilities
    total_weight = sum(sum(row) for row in refined_weights)
    if total_weight > 0:  # Avoid division by zero
        for i in range(n1):
            for j in range(n2):
                refined_weights[i][j] /= total_weight

    return refined_weights

def priority_v7(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights with zeros
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Iterate through all possible node mappings
    for i in range(n1):
        for j in range(n2):
            # Calculate a score based on structural similarity
            score = 0.0

            # Consider node degree similarity
            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
            score -= degree_diff  # Penalize large degree differences

            # Consider neighborhood similarity (common neighbors)
            neighbors_i = set(k for k in range(n1) if graph1[i][k])
            neighbors_j = set(k for k in range(n2) if graph2[j][k])
            common_neighbors = len(neighbors_i.intersection(neighbors_j))
            score += common_neighbors

            # Incorporate initial weights (if provided)
            if i < len(weights) and j < len(weights[i]):
                score += weights[i][j] # Add initial weight as a prior

            refined_weights[i][j] = score


    # Normalize the refined weights (optional, but often beneficial)
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])  # Sum only relevant parts
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum

    return refined_weights

def priority_v8(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights matrix with zeros
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Iterate through the nodes of graph1
    for i in range(n1):
        # Iterate through the nodes of graph2
        for j in range(n2):
            # Calculate the structural similarity between nodes i and j
            similarity = 0.0

            # Consider node degrees as a basic structural feature
            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
            similarity -= degree_diff  # Penalize degree differences

            # Consider neighborhood similarity (common neighbors)
            neighbors_i = set(k for k in range(n1) if graph1[i][k])
            neighbors_j = set(k for k in range(n2) if graph2[j][k])
            common_neighbors = len(neighbors_i.intersection(neighbors_j))
            similarity += common_neighbors


            # Combine the initial weight with the structural similarity
            if i < len(weights) and j < len(weights[0]):  # Handle potential size mismatch
                refined_weights[i][j] = weights[i][j] + similarity


    # Normalize the refined weights to probabilities (optional)
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2]) # Only sum up to n2
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum
        

    return refined_weights

def priority_v9(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`.
    Calculates node mapping probabilities based on structural similarity.
    """
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            # Calculate node similarity based on degree difference
            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
            similarity = 1 / (1 + degree_diff)  # Higher similarity for smaller degree difference

            # Calculate neighborhood similarity
            neighbors_i = set(k for k in range(n1) if graph1[i][k])
            neighbors_j = set(k for k in range(n2) if graph2[j][k])

            common_neighbors = 0
            for ni in neighbors_i:
                for nj in neighbors_j:
                    if weights[ni][nj] > 0: # If the neighbors are likely mapped
                        common_neighbors +=1
            neighbor_similarity = common_neighbors / (len(neighbors_i) * len(neighbors_j) + 1e-6) # avoid division by zero


            # Combine similarities and initial weights (if provided)
            if len(weights) >= n1 and len(weights[0]) >= n2:
                refined_weights[i][j] = (similarity + neighbor_similarity + weights[i][j]) / 3
            else:
                refined_weights[i][j] = (similarity + neighbor_similarity) / 2


    # Normalize the weights (optional, but often beneficial)
    for i in range(n1):
       row_sum = sum(refined_weights[i])
       if row_sum > 0:
           for j in range(n2):
               refined_weights[i][j] /= row_sum

    return refined_weights

def priority_v10(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
            degree_sim = 1.0 / (1.0 + abs(sum(graph1[i]) - sum(graph2[j])))

            # Calculate neighborhood similarity (common neighbors)
            neighbors1 = set(k for k in range(n1) if graph1[i][k])
            neighbors2 = set(k for k in range(n2) if graph2[j][k])

            common_neighbors = len(neighbors1.intersection(neighbors2))
            neighbor_sim = 1.0 / (1.0 + abs(len(neighbors1) + len(neighbors2) - 2 * common_neighbors) ) # consider union, not sum


            # Combine similarities and initial weights (if available)
            if weights: # check if weights matrix is not empty or None
                refined_weights[i][j] = (degree_sim + neighbor_sim) * (weights[min(i,len(weights)-1)][min(j,len(weights[0])-1)] + 1e-6) / 2.0  #avoid 0 
            else:
                refined_weights[i][j] = (degree_sim + neighbor_sim) / 2.0

    return refined_weights

def priority_v11(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes a refined weight matrix based on graph structure similarity.

    Args:
        graph1: The adjacency matrix of the first graph.
        graph2: The adjacency matrix of the second graph.
        weights: An initial weight matrix.

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
            neighbor_similarity += weights[k][l]  # Use initial weights for neighbors

      refined_weights[i][j] = neighbor_similarity

  return refined_weights

def priority_v12(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Further improved version of `priority_v0` with normalization and node degree consideration."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]
  
  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] == 1 and graph2[j][l] == 1:
            neighbor_similarity += weights[k][l]

      degree_similarity = 1.0 / (1.0 + abs(degrees1[i] - degrees2[j]))  # Consider node degrees

      if degrees1[i] > 0 and degrees2[j] > 0: # Normalize neighbor similarity
          neighbor_similarity /= (degrees1[i] * degrees2[j])

      refined_weights[i][j] = neighbor_similarity * degree_similarity  # Combine similarities

  return refined_weights

def priority_v13(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            node_similarity = weights[i][j]  # Use the provided initial weights

            # Calculate structural similarity based on neighborhood
            neighbors_i = [k for k in range(n1) if graph1[i][k] == 1]
            neighbors_j = [k for k in range(n2) if graph2[j][k] == 1]

            common_neighbors = 0
            for ni in neighbors_i:
                for nj in neighbors_j:
                    if weights[ni][nj] > 0: # Check if neighbors are likely to be mapped
                        common_neighbors += weights[ni][nj]


            structural_similarity = (common_neighbors) / (len(neighbors_i) + len(neighbors_j) + 1e-6) # Avoid division by zero

            # Combine node and structural similarity (you can adjust the weights)
            refined_weights[i][j] = 0.5 * node_similarity + 0.5 * structural_similarity


    return refined_weights

def priority_v14(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes a refined weight matrix based on graph structure similarity.

    Args:
        graph1: The adjacency matrix of the first graph.
        graph2: The adjacency matrix of the second graph.
        weights: An initial weight matrix.

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
          if graph1[i][k] == 1 and graph2[j][l] == 1:  # Check for edge existence
            neighbor_similarity += weights[k][l] if k < len(weights) and l < len(weights[k]) else 0

      refined_weights[i][j] = neighbor_similarity  # Update weight based on neighborhood similarity

  return refined_weights

def priority_v15(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Further improved version of `priority_v0` with normalization."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      neighbor_count = 0  # Keep track of the number of neighbor pairs

      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] == 1 and graph2[j][l] == 1:
            neighbor_similarity += weights[k][l] if k < len(weights) and l < len(weights[k]) else 0
            neighbor_count += 1

      if neighbor_count > 0:
        refined_weights[i][j] = neighbor_similarity / neighbor_count # Normalize by the number of neighbor pairs
      else:  # Handle cases where a node has no neighbors
        refined_weights[i][j] = 0


  return refined_weights

def priority_v16(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

def priority_v17(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Improves `priority_v0` by considering neighbor degrees.
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
      weights[i][j] = neighbor_similarity

  return weights

def priority_v18(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Combines degree similarity and neighbor similarity.
  """
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

      weights[i][j] = degree_similarity * neighbor_similarity # Combine both similarities

  return weights

def priority_v19(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

def priority_v20(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      weights[i][j] = 1.0 / (abs(deg1[i] - deg2[j]) + 1)

  return weights

def priority_v21(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Considers neighbor degrees for improved probability estimation.
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
      weights[i][j] = neighbor_similarity

  return weights

def priority_v22(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Combines degree difference and neighbor similarity.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  deg1 = [sum(row) for row in graph1]
  deg2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      degree_diff = 1.0 / (abs(deg1[i] - deg2[j]) + 1)

      neighbor_similarity = 0
      for neighbor1 in range(n1):
        if graph1[i][neighbor1]:
          for neighbor2 in range(n2):
            if graph2[j][neighbor2]:
              neighbor_similarity += 1.0 / (abs(sum(graph1[neighbor1]) - sum(graph2[neighbor2])) + 1)
      
      weights[i][j] = degree_diff * neighbor_similarity  # Combine both metrics

  return weights

def priority_v23(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      neighbor_similarity = 0
      for k in range(n1):
        if graph1[i][k]:
          for l in range(n2):
            if graph2[j][l]:
              neighbor_similarity += 1.0 / (1.0 + abs(degrees1[k] - degrees2[l]))
      weights[i][j] = 1.0 / (1.0 + abs(degrees1[i] - degrees2[j])) + neighbor_similarity

  return weights

def priority_v25(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities for node mappings.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
      refined_weights[i][j] = 1.0 / (1.0 + degree_diff)  # Higher similarity for similar degrees

  return refined_weights

def priority_v26(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version using neighbor similarity."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
          for l in range(n2):
              if graph1[i][k] and graph2[j][l]:
                  neighbor_similarity += weights[k][l] # Use initial weights for neighbor comparison

      degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
      refined_weights[i][j] = (1.0 / (1.0 + degree_diff)) + neighbor_similarity

  return refined_weights

def priority_v27(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version with normalization and neighbor weighting."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      neighbors1 = [k for k in range(n1) if graph1[i][k]]
      neighbors2 = [l for l in range(n2) if graph2[j][l]]

      for k in neighbors1:
          for l in neighbors2:
              neighbor_similarity += weights[k][l]

      degree_similarity = 1.0 / (1.0 + abs(sum(graph1[i]) - sum(graph2[j])))
      
      refined_weights[i][j] = degree_similarity + neighbor_similarity


  # Normalize weights
  for i in range(n1):
      row_sum = sum(refined_weights[i][:n2])  # Only sum within valid range
      if row_sum > 0:
          for j in range(n2):
              refined_weights[i][j] /= row_sum



  return refined_weights

def priority_v28(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

def priority_v29(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
          
      weights[i][j] = 1.0 / (1.0 + degree_diff + neighbor_diff)

  return weights

def priority_v32(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      weights[i][j] = 1.0 / abs(degrees1[i] - degrees2[j] + 1)  # Inversely proportional to degree difference

  return weights

def priority_v33(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
        if graph1[i][k] and any(graph2[j][l] for l in range(n2)): # Check if there's a connection to any node in graph2
           common_neighbors +=1
      weights[i][j] = 1.0 / (abs(sum(graph1[i]) - sum(graph2[j])) + 1) + common_neighbors # Combine degree difference and common neighbors


  return weights

def priority_v34(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights with zeros
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Handle cases where input weights are not provided or are incorrectly sized
    if weights is None or len(weights) != max_node or len(weights[0]) != max_node:
        weights = [[1.0 / max_node] * max_node for _ in range(max_node)]  # Uniform initial weights

    # Calculate node degrees for both graphs
    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            # Calculate a similarity score based on node degrees and initial weights
            degree_similarity = 1.0 / (1.0 + abs(degrees1[i] - degrees2[j]))
            refined_weights[i][j] = weights[i][j] * degree_similarity

            # Consider neighborhood similarity (edges)
            neighbors1 = [k for k in range(n1) if graph1[i][k] == 1]
            neighbors2 = [k for k in range(n2) if graph2[j][k] == 1]

            neighbor_similarity = 0.0
            for n1_neighbor in neighbors1:
                for n2_neighbor in neighbors2:
                    neighbor_similarity += weights[n1_neighbor][n2_neighbor]

            refined_weights[i][j] *= (1.0 + neighbor_similarity) # boost based on neighbor similarity


    # Normalize the refined weights (optional, but can improve performance)
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2]) # Only sum relevant part
        if row_sum > 0:
          for j in range(n2):
              refined_weights[i][j] /= row_sum


    return refined_weights

def priority_v35(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`.
    Calculates node mapping probabilities based on structural similarity."""

    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            # Calculate structural similarity based on neighborhood
            neighbors1 = [n for n in range(n1) if graph1[i][n] == 1]
            neighbors2 = [n for n in range(n2) if graph2[j][n] == 1]

            common_neighbors = 0
            for neighbor1 in neighbors1:
                for neighbor2 in neighbors2:
                    if weights[neighbor1][neighbor2] > 0:  # Check if neighbors are likely mapped
                        common_neighbors += 1

            # Normalize the similarity score (add 1 to avoid division by zero)
            similarity = (common_neighbors + 1) / (len(neighbors1) + len(neighbors2) + 2)
            
            refined_weights[i][j] = similarity

    return refined_weights

def priority_v36(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes the Graph Edit Distance (GED), a measure of the dissimilarity between two graphs. 
    GED is defined as the minimum number of operations required to transform one graph into another.

    Args:
        graph1: The adjacency matrix of the first graph.
        graph2: The adjacency matrix of the second graph.
        weights: A weight matrix representing the initial probabilities of mapping nodes between `graph1` and `graph2`.

    Returns:
        A refined weight matrix (float) using the initial input matrix and the adjacency matrices of graphs.
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
            neighbor_similarity += weights[k][l] if k < n1 and l < n2 and k < len(weights) and l < len(weights[k]) else 0

      refined_weights[i][j] = neighbor_similarity

  return refined_weights

def priority_v37(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0` with normalization and handling of different sized graphs."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      deg_i = sum(graph1[i])
      deg_j = sum(graph2[j])

      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] == 1 and graph2[j][l] == 1:
              neighbor_similarity += weights[k][l] if k < len(weights) and l < len(weights[k]) else 0


      if deg_i > 0 and deg_j > 0:  # Normalize by degrees if both nodes have neighbors
          refined_weights[i][j] = neighbor_similarity / (deg_i * deg_j)
      elif deg_i == 0 and deg_j == 0: # if both nodes don't have neighbors, set high similarity
          refined_weights[i][j] = 1.0 if i < len(weights) and j < len(weights[0]) and weights[i][j] > 0 else 0.0  # Or handle as needed

  return refined_weights

def priority_v38(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`.

    This version calculates node similarity based on neighborhood structure and incorporates initial weights.
    """
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Pad adjacency matrices and weights to max_node x max_node
    padded_graph1 = np.pad(graph1, ((0, max_node - n1), (0, max_node - n1)), 'constant')
    padded_graph2 = np.pad(graph2, ((0, max_node - n2), (0, max_node - n2)), 'constant')
    padded_weights = np.pad(weights, ((0, max_node - n1), (0, max_node - n2)), 'constant')

    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            # Calculate neighborhood similarity
            neighborhood_similarity = 0
            for k in range(max_node):
                neighborhood_similarity += abs(padded_graph1[i, k] - padded_graph2[j, k])

            # Combine initial weight and neighborhood similarity.  Invert neighborhood similarity
            # as higher similarity should result in a higher weight.
            refined_weights[i][j] = padded_weights[i][j] + (1 / (1 + neighborhood_similarity)) # avoid division by zero

    # Normalize refined weights (optional, but can be helpful)
    row_sums = [sum(row) for row in refined_weights]
    for i in range(n1):
        for j in range(max_node): # Normalize over the full padded size
            if row_sums[i] > 0: # avoid div by zero
                refined_weights[i][j] /= row_sums[i]

    return refined_weights

def priority_v39(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

      refined_weights[i][j] = 1 / (1 + cost) if cost > 0 else 1  # Inversely proportional to cost

  return refined_weights

def priority_v40(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0` using initial weights."""
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

      if i < len(weights) and j < len(weights[0]):
          refined_weights[i][j] = weights[i][j] / (1 + cost) if cost > 0 else weights[i][j] # Incorporate initial weights
      else:
          refined_weights[i][j] = 1 / (1 + cost) if cost > 0 else 1

  return refined_weights

def priority_v41(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      weights[i][j] = 1.0 / (abs(degrees1[i] - degrees2[j]) + 1)

  return weights

def priority_v42(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Considers neighbor degrees for improved probability estimation.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
        if graph1[i][k]:
          for l in range(n2):
            if graph2[j][l]:
              neighbor_similarity += 1.0 / (abs(sum(graph1[k]) - sum(graph2[l])) + 1)
      weights[i][j] = (1.0 / (abs(sum(graph1[i]) - sum(graph2[j])) + 1)) + neighbor_similarity

  return weights

def priority_v44(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      weights[i][j] = (1.0 / (1.0 + abs(degrees1[i] - degrees2[j]))) + neighbor_similarity

  return weights

def priority_v45(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      weights[i][j] = 1.0 / (1.0 + abs(deg1[i] - deg2[j]))

  return weights

def priority_v46(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Improves initial probabilities by considering neighbor degrees.
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

