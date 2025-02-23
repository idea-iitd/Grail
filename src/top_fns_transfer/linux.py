#Num Unique Functions in logs: 329
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
      neighbor_similarity = 0
      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] == 1 and graph2[j][l] == 1:
            neighbor_similarity += weights[k][l]  # Use provided weights

      refined_weights[i][j] = neighbor_similarity

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
      neighbor_count1 = sum(graph1[i])
      neighbor_count2 = sum(graph2[j])

      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] == 1 and graph2[j][l] == 1:
            neighbor_similarity += weights[k][l]

      if neighbor_count1 > 0 and neighbor_count2 > 0:  # Avoid division by zero
        refined_weights[i][j] = neighbor_similarity / (neighbor_count1 * neighbor_count2)  # Normalize
      # Or a different normalization scheme like:
      # refined_weights[i][j] = neighbor_similarity / math.sqrt(neighbor_count1 * neighbor_count2)


  return refined_weights

def priority_v2(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`.

    This version calculates node similarity based on neighborhood structure and uses the provided weights as a prior.
    """
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    
    # Pad graphs and weights to max_node x max_node
    padded_graph1 = np.pad(graph1, ((0, max_node - n1), (0, max_node - n1)), 'constant')
    padded_graph2 = np.pad(graph2, ((0, max_node - n2), (0, max_node - n2)), 'constant')
    padded_weights = np.pad(weights, ((0, max_node - n1), (0, max_node - n2)), 'constant')

    refined_weights = np.zeros((max_node, max_node), dtype=float)

    for i in range(n1):
        for j in range(n2):
            neighbor_similarity = 0
            for k in range(n1):
                for l in range(n2):
                    neighbor_similarity += padded_graph1[i][k] * padded_graph2[j][l] * padded_weights[k][l]  #Consider existing mappings

            refined_weights[i][j] = padded_weights[i][j] + neighbor_similarity  #Combine prior weights with neighborhood similarity

    # Normalize refined weights (optional, but often beneficial)
    row_sums = refined_weights.sum(axis=1, keepdims=True)
    refined_weights = refined_weights / row_sums.clip(min=1e-6)  #Avoid division by zero
    
    #Truncate back to original sizes if needed:
    refined_weights = refined_weights[:n1,:n2]

    return refined_weights.tolist()

def priority_v3(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights matrix with zeros
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Iterate through all possible node mappings between graph1 and graph2
    for i in range(n1):
        for j in range(n2):
            # Calculate a similarity score based on node degree and neighborhood similarity
            degree_similarity = 1.0 / (abs(sum(graph1[i]) - sum(graph2[j])) + 1)  # Inverse of degree difference
            neighborhood_similarity = 0.0

            # Compare neighbors of i in graph1 with neighbors of j in graph2
            for k in range(n1):
                if graph1[i][k] == 1: # i's neighbor in graph1
                    for l in range(n2):
                        if graph2[j][l] == 1: # j's neighbor in graph2
                            neighborhood_similarity += weights[k][l] if k < len(weights) and l < len(weights[k]) else 0


            # Combine the initial weight, degree similarity, and neighborhood similarity
            refined_weights[i][j] = (weights[i][j] if i < len(weights) and j < len(weights[i]) else 0) * degree_similarity * (1 + neighborhood_similarity)


    return refined_weights

def priority_v4(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            node_similarity = weights[i][j]  # Use initial weights as a base

            # Consider edge similarities
            neighbor_similarity = 0
            for k in range(n1):
                if graph1[i][k]:
                    for l in range(n2):
                        if graph2[j][l]:
                            neighbor_similarity += weights[k][l]

            # Combine node and edge similarities (you can adjust the weighting)
            refined_weights[i][j] = node_similarity + neighbor_similarity


    # Normalize the weights to probabilities (optional but often beneficial)
    for i in range(n1):
      row_sum = sum(refined_weights[i][:n2]) # Only sum over valid indices for graph2
      if row_sum > 0:
        for j in range(n2):
          refined_weights[i][j] /= row_sum


    return refined_weights

def priority_v5(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
            neighbor_similarity += weights[k][l]  # Use initial weights for neighbor comparison

      refined_weights[i][j] = neighbor_similarity

  return refined_weights

def priority_v6(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Further improved version of `priority_v0` with normalization."""
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
            neighbor_similarity += weights[k][l]

      # Normalize by the number of neighbors
      degree1 = sum(graph1[i])
      degree2 = sum(graph2[j])
      if degree1 > 0 and degree2 > 0:  # Avoid division by zero
        refined_weights[i][j] = neighbor_similarity / (degree1 * degree2)


  return refined_weights

def priority_v7(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities for node mappings between two graphs.

    Args:
        graph1: The adjacency matrix of the first graph.
        graph2: The adjacency matrix of the second graph.
        weights:  Not used in this version.  Placeholder for future versions.

    Returns:
        A matrix of initial probabilities. Each element (i, j) represents the 
        probability of mapping node i in graph1 to node j in graph2.  This 
        version initializes all probabilities to 0.0.
  """
  max_node = max(len(graph1), len(graph2))
  weights = [[0.0] * max_node for _ in range(max_node)]
  return weights

def priority_v8(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0` that calculates initial probabilities 
  based on node degrees.

  Args:
      graph1: The adjacency matrix of the first graph.
      graph2: The adjacency matrix of the second graph.
      weights: Not used. Placeholder for future versions.

  Returns:
      A matrix of initial probabilities based on node degree similarity.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  prob_matrix = [[0.0] * max_node for _ in range(max_node)]

  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      degree_diff = abs(degrees1[i] - degrees2[j])
      prob_matrix[i][j] = 1.0 / (degree_diff + 1.0)  # Higher probability for similar degrees

  # Normalize probabilities (optional, but recommended)
  for i in range(n1):
    row_sum = sum(prob_matrix[i][:n2])  # Only sum over valid nodes in graph2
    if row_sum > 0:
      for j in range(n2):
        prob_matrix[i][j] /= row_sum

  return prob_matrix

def priority_v9(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights matrix
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Iterate through potential node mappings
    for i in range(n1):
        for j in range(n2):
            # Calculate a similarity score based on node degree and neighborhood structure
            degree_similarity = 1.0 / (1.0 + abs(sum(graph1[i]) - sum(graph2[j])))

            neighborhood_similarity = 0.0
            common_neighbors = 0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] == 1 and graph2[j][l] == 1:  # Check if nodes are neighbors
                        if weights[k][l] > 0: # or use a threshold
                           common_neighbors += weights[k][l]  # Consider existing weight as indicator of neighbor mapping
            if sum(graph1[i]) > 0 and sum(graph2[j]) > 0: # avoid division by zero
                neighborhood_similarity =  common_neighbors / (sum(graph1[i]) * sum(graph2[j])) if common_neighbors > 0 else 0.0


            # Combine similarities and initial weight (if available)
            refined_weights[i][j] = (degree_similarity + neighborhood_similarity) * (weights[i][j] if i < len(weights) and j < len(weights[i]) else 1/max_node) # use small value if no prior weight

    # Normalize the refined weights (optional, but recommended)
    for i in range(n1):
        row_sum = sum(refined_weights[i])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum



    return refined_weights

def priority_v10(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights matrix with zeros
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Iterate through possible mappings and calculate a similarity score
    for i in range(n1):
        for j in range(n2):
            # Consider initial weights (if available)
            initial_weight = weights[i][j] if i < len(weights) and j < len(weights[i]) else 0

            # Calculate a similarity score based on neighborhood structure
            neighbors1 = [k for k in range(n1) if graph1[i][k]]
            neighbors2 = [k for k in range(n2) if graph2[j][k]]

            common_neighbors = 0
            for neighbor1 in neighbors1:
              for neighbor2 in neighbors2:
                if weights[neighbor1][neighbor2] > 0: # or some threshold
                  common_neighbors +=1


            similarity = initial_weight + common_neighbors  # Combine initial weight and structural similarity

            refined_weights[i][j] = similarity

    # Normalize the refined weights (optional, but often helpful)
    for i in range(n1):
        row_sum = sum(refined_weights[i])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum

    return refined_weights

def priority_v11(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

def priority_v12(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Improves initial probabilities by considering neighbor degrees.
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

def priority_v13(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      
      weights[i][j] = (degree_similarity + neighbor_similarity)/2


  return weights

def priority_v14(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities based purely on node degrees.
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
      weights[i][j] = 1.0 / (degree_diff + 1)  # Higher similarity for closer degrees

  return weights

def priority_v15(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Considers neighbor degrees in addition to node degrees."""
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
      for neighbor1 in range(n1):
          if graph1[i][neighbor1]:
              for neighbor2 in range(n2):
                  if graph2[j][neighbor2]:
                      neighbor_similarity += 1.0 / (abs(degrees1[neighbor1] - degrees2[neighbor2]) + 1)

      weights[i][j] = (1.0 / (degree_diff + 1)) + (neighbor_similarity / (n1 * n2)) # Combine node and neighbor degree similarity


  return weights

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
      weights[i][j] = 1.0 / (abs(degrees1[i] - degrees2[j]) + 1)  # Inversely proportional to degree difference

  return weights

def priority_v17(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
              neighbor_similarity += 1.0 / (abs(degrees1[k] - degrees2[l]) + 1)
      weights[i][j] = (1.0 / (abs(degrees1[i] - degrees2[j]) + 1)) * (neighbor_similarity + 1) # Combine node and neighbor similarity

  return weights

def priority_v18(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

def priority_v19(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities for node mappings between two graphs based on their structure.

    Args:
        graph1: The adjacency matrix of the first graph.
        graph2: The adjacency matrix of the second graph.
        weights: A weight matrix (currently unused, placeholder for future versions).

    Returns:
        A matrix of initial probabilities where each entry (i, j) represents the probability 
        of mapping node i from graph1 to node j from graph2.  This version prioritizes
        mappings based on degree similarity.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  prob_matrix = [[0.0] * max_node for _ in range(max_node)]

  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      degree_diff = abs(degrees1[i] - degrees2[j])
      prob_matrix[i][j] = 1.0 / (1 + degree_diff)  # Higher probability for similar degrees

  # Normalize probabilities for each row (node in graph1)
  for i in range(n1):
    row_sum = sum(prob_matrix[i][:n2])  # Only consider valid mappings
    if row_sum > 0:
      for j in range(n2):
        prob_matrix[i][j] /= row_sum

  return prob_matrix

def priority_v20(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0` considering common neighbors."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  prob_matrix = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      common_neighbors = 0
      for k in range(n1):
        if graph1[i][k] == 1 and (k < n2 and graph2[j][k] == 1) : # Check if k is a common neighbor
          common_neighbors += 1


      prob_matrix[i][j] = 1.0 + common_neighbors  # Higher probability for more common neighbors

  # Normalize probabilities (same as in v0)
  for i in range(n1):
    row_sum = sum(prob_matrix[i][:n2])
    if row_sum > 0:
      for j in range(n2):
        prob_matrix[i][j] /= row_sum

  return prob_matrix

def priority_v21(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            node_similarity = weights[i][j]  # Use initial weights as a base

            # Consider edge similarity
            neighbors1 = [k for k in range(n1) if graph1[i][k] == 1]
            neighbors2 = [k for k in range(n2) if graph2[j][k] == 1]

            common_neighbors = 0
            for n1_neighbor in neighbors1:
                for n2_neighbor in neighbors2:
                    if weights[n1_neighbor][n2_neighbor] > 0: # Check for potential mapping in initial weights
                        common_neighbors += 1

            edge_similarity = (2 * common_neighbors) / (len(neighbors1) + len(neighbors2)) if (len(neighbors1) + len(neighbors2)) > 0 else 0


            # Combine node and edge similarity (you can adjust the weights)
            refined_weights[i][j] = 0.5 * node_similarity + 0.5 * edge_similarity


    return refined_weights

def priority_v22(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights matrix with zeros
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Iterate through all possible node mappings between the two graphs
    for i in range(n1):
        for j in range(n2):
            # Calculate a similarity score based on node degrees and neighborhood structure
            degree_similarity = 1 - abs(sum(graph1[i]) - sum(graph2[j])) / max(sum(graph1[i]), sum(graph2[j])) if max(sum(graph1[i]), sum(graph2[j])) > 0 else 0  # Avoid division by zero

            neighborhood_similarity = 0
            common_neighbors = 0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] == 1 and graph2[j][l] == 1:
                      common_neighbors +=1


            if (sum(graph1[i]) > 0 and  sum(graph2[j]) >0):
              neighborhood_similarity = common_neighbors / (sum(graph1[i]) * sum(graph2[j]))
            elif sum(graph1[i]) ==0 and sum(graph2[j]) ==0:
                neighborhood_similarity =1


            # Combine the similarity scores and initial weights
            refined_weights[i][j] = (degree_similarity + neighborhood_similarity )/2 * (weights[i][j] if i < len(weights) and j < len(weights[0]) else 0) # use initial weights if available

    return refined_weights

def priority_v23(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
             neighbor_similarity += weights[k][l] if 0 <= k < len(weights) and 0 <= l < len(weights[0]) else 0 # Handle potential index errors

      refined_weights[i][j] = neighbor_similarity  # Update weights based on neighborhood similarity

  return refined_weights

def priority_v24(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

def priority_v25(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      neighbors1 = [k for k in range(n1) if graph1[i][k]]
      neighbors2 = [k for k in range(n2) if graph2[j][k]]

      for n1_neighbor in neighbors1:
        for n2_neighbor in neighbors2:
             neighbor_similarity += 1.0 / (1 + abs(sum(graph1[n1_neighbor]) - sum(graph2[n2_neighbor])))

      weights[i][j] = (1.0 / (1 + abs(sum(graph1[i]) - sum(graph2[j])))) + neighbor_similarity


  return weights

def priority_v26(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
              neighbor_similarity += 1.0 / (abs(degrees1[k] - degrees2[l]) + 1)
      weights[i][j] = (1.0 / (abs(degrees1[i] - degrees2[j]) + 1)) + neighbor_similarity


  return weights

def priority_v28(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

def priority_v29(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Refines probabilities based on neighbor similarity.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  new_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0.0
      for k in range(n1):
        if graph1[i][k]:
          for l in range(n2):
            if graph2[j][l]:
              neighbor_similarity += weights[k][l]
      new_weights[i][j] = weights[i][j] + neighbor_similarity # Combine degree and neighbor similarity
      
  return new_weights

def priority_v30(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Normalizes probabilities.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  new_weights = [[0.0] * max_node for _ in range(max_node)]


  for i in range(n1):
      row_sum = sum(weights[i][:n2]) # Only consider valid mappings
      if row_sum > 0:
          for j in range(n2):
              new_weights[i][j] = weights[i][j] / row_sum
      else: #Handle cases where a node has no similar neighbors. Distribute probability evenly.
          for j in range(n2):
              new_weights[i][j] = 1.0 / n2

  return new_weights

def priority_v31(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

def priority_v32(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

def priority_v33(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      weights[i][j] = 1.0 / (abs(degrees1[i] - degrees2[j]) + 1)
  return weights

def priority_v34(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities considering neighbor degrees.
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
      weights[i][j] = (1.0 / (abs(degrees1[i] - degrees2[j]) + 1)) + neighbor_similarity
  return weights

def priority_v35(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities using common neighbors.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      common_neighbors = 0
      for k in range(n1):
        if graph1[i][k]:
          for l in range(n2):
            if graph2[j][l]:
               # Hypothetical common neighbor based on degree similarity
               if abs(sum(graph1[k]) - sum(graph2[l])) <= 1: # Threshold for degree similarity
                 common_neighbors += 1
      weights[i][j] = common_neighbors

  return weights

def priority_v36(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      weights[i][j] = (1.0 / (abs(degrees1[i] - degrees2[j]) + 1))
  return weights

def priority_v37(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities considering neighbor degrees.
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
      weights[i][j] = (1.0 / (abs(degrees1[i] - degrees2[j]) + 1)) * (neighbor_similarity + 1) # Adding 1 to avoid multiplication by zero

  return weights

def priority_v38(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities considering common neighbors.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]
  
  for i in range(n1):
    for j in range(n2):
      common_neighbors = 0
      for k in range(n1):
        if graph1[i][k]:
          for l in range(n2):
            if graph2[j][l]: #and graph1[i][k] == graph2[j][l] :  Removed potential edge label comparison, adjust if needed.
              common_neighbors +=1
      weights[i][j] = common_neighbors

  return weights

def priority_v40(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities considering neighbor degrees.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]
  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      neighbor_degrees1 = [degrees1[k] for k in range(n1) if graph1[i][k]]
      neighbor_degrees2 = [degrees2[k] for k in range(n2) if graph2[j][k]]

      neighbor_similarity = 0
      for deg1 in neighbor_degrees1:
        for deg2 in neighbor_degrees2:
          neighbor_similarity += 1.0 / (abs(deg1 - deg2) + 1)

      weights[i][j] = (1.0 / (abs(degrees1[i] - degrees2[j]) + 1)) * (neighbor_similarity + 1) # Added +1 to handle cases where no neighbors exist

  return weights

def priority_v41(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities using common neighbors.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      common_neighbors = 0
      for k in range(n1):
          for l in range(n2):
              if graph1[i][k] and graph2[j][l]: #Check if k is neighbor of i and l is neighbor of j
                  #Estimate probability of k mapping to l (simple degree comparison for now)
                  prob_k_l = 1.0 / (abs(sum(graph1[k]) - sum(graph2[l])) + 1) 
                  common_neighbors += prob_k_l
      weights[i][j] = common_neighbors

  return weights

def priority_v42(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      weights[i][j] = (1.0 / (abs(degrees1[i] - degrees2[j]) + 1))
  return weights

def priority_v43(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities considering neighbor degrees.
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
      weights[i][j] = (1.0 / (abs(degrees1[i] - degrees2[j]) + 1)) * (neighbor_similarity + 1)  # Added +1 to avoid multiplying by zero

  return weights

def priority_v44(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities using common neighbors.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      common_neighbors = 0
      for k in range(n1):
        if graph1[i][k]:
          for l in range(n2):
            if graph2[j][l]:
               # Hypothetical common neighbor check based on index similarity (needs improvement for real-world scenarios)
              if abs(k-l) <= 1: # Adjust this threshold based on the problem
                common_neighbors += 1
      weights[i][j] = common_neighbors + 1 # Add 1 to avoid zero values


  return weights

def priority_v46(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Improves `priority_v1` by considering common neighbors.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      common_neighbors = 0
      for k in range(max_node):  # Iterate through potential neighbors
        if k < n1 and k < n2:  # Ensure valid indices
          if graph1[i][k] == 1 and graph2[j][k] == 1:
            common_neighbors += 1
      weights[i][j] = common_neighbors


  # Normalize (important to avoid overly large values dominating)
  total_weight = sum(sum(row) for row in weights)
  if total_weight > 0:
      for i in range(n1):
          for j in range(n2):
              weights[i][j] /= total_weight
  return weights

def priority_v48(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
        weights[i][j] = (1.0 / (abs(degrees1[i] - degrees2[j]) + 1))
  return weights

def priority_v49(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities based on common neighbors.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      common_neighbors = 0
      for k in range(max(n1, n2)):
        try:
          if graph1[i][k] and graph2[j][k]:
            common_neighbors += 1
        except IndexError:
          pass  # Handle cases where k is out of bounds for one of the graphs
      weights[i][j] = common_neighbors


  # Normalize the weights (optional, but often beneficial)
  max_weight = max(max(row) for row in weights) if any(any(row) for row in weights) else 1  # Avoid division by zero
  for i in range(n1):
    for j in range(n2):
      weights[i][j] /= max_weight  

  return weights

def priority_v50(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Combines degree difference and common neighbors."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]
  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      degree_similarity = 1.0 / (abs(degrees1[i] - degrees2[j]) + 1)
      common_neighbors = 0
      for k in range(max(n1, n2)):
        try:
          if graph1[i][k] and graph2[j][k]:
            common_neighbors += 1
        except IndexError:
          pass
      weights[i][j] = degree_similarity * (common_neighbors + 1)  # Combine the two measures
      
  return weights

def priority_v51(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      weights[i][j] = 1.0 / (abs(degrees1[i] - degrees2[j]) + 1)
  return weights

def priority_v52(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Considers neighbor degrees for improved mapping probabilities.
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
          neighbor_diff += min_diff if min_diff != float('inf') else 0 # handle empty neighbor list
      weights[i][j] = 1.0 / (degree_diff + neighbor_diff + 1)
  return weights

def priority_v53(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  prob_matrix = [([0.0] * max_node) for _ in range(max_node)]
  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
        neighbor_similarity = 0
        for k in range(n1):
            for l in range(n2):
                neighbor_similarity += graph1[i][k] * graph2[j][l] * weights[k][l]
        degree_diff = abs(degrees1[i] - degrees2[j])
        prob_matrix[i][j] = (1.0 / (1 + degree_diff)) + neighbor_similarity


  for i in range(n1):
    row_sum = sum(prob_matrix[i][:n2])
    if row_sum > 0:
        for j in range(n2):
            prob_matrix[i][j] /= row_sum
  return prob_matrix

def priority_v54(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  prob_matrix = [([0.0] * max_node) for _ in range(max_node)]
  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
        neighbor_similarity = 0
        for k in range(n1):
            for l in range(n2):
                neighbor_similarity += graph1[i][k] * graph2[j][l] * weights[k][l]  # Consider neighbor relationships
        degree_diff = abs(degrees1[i] - degrees2[j])
        prob_matrix[i][j] = (1.0 / (1 + degree_diff)) * (1 + neighbor_similarity) # Combine degree difference and neighbor similarity

  for i in range(n1):
    row_sum = sum(prob_matrix[i][:n2])
    if row_sum > 0:
        for j in range(n2):
            prob_matrix[i][j] /= row_sum  # Normalize probabilities
  return prob_matrix

def priority_v55(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  prob_matrix = [([0.0] * max_node) for _ in range(max_node)]
  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
        for l in range(n2):
          neighbor_similarity += graph1[i][k] * graph2[j][l] * weights[k][l]  # Consider neighbor relationships

      degree_diff = abs(degrees1[i] - degrees2[j])
      prob_matrix[i][j] = (1.0 / (1 + degree_diff)) * (1 + neighbor_similarity)  # Combine degree difference and neighbor similarity


  for i in range(n1):
    row_sum = sum(prob_matrix[i][:n2])
    if row_sum > 0:
        for j in range(n2):
            prob_matrix[i][j] /= row_sum  # Normalize probabilities
  return prob_matrix

def priority_v56(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  prob_matrix = [[0.0] * max_node for _ in range(max_node)]
  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
        for l in range(n2):
          neighbor_similarity += graph1[i][k] * graph2[j][l] * weights[k][l]
      degree_diff = abs(degrees1[i] - degrees2[j])
      prob_matrix[i][j] = (1.0 / (1 + degree_diff)) + neighbor_similarity


  for i in range(n1):
    row_sum = sum(prob_matrix[i][:n2])
    if row_sum > 0:
      for j in range(n2):
        prob_matrix[i][j] /= row_sum
  return prob_matrix

def priority_v58(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities considering neighbor degrees.
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
      weights[i][j] = (1.0 / (abs(degrees1[i] - degrees2[j]) + 1)) * (neighbor_similarity + 1) # Added 1 to avoid multiplication by zero

  return weights

def priority_v59(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities considering common neighbors.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]
  
  for i in range(n1):
    for j in range(n2):
      common_neighbors = 0
      for k in range(n1):
        if graph1[i][k]:
          for l in range(n2):
            if graph2[j][l]:  # Potential optimization: pre-calculate neighbors
              common_neighbors += graph1[i][k] * graph2[j][l] # Should be 1 if both are connected

      weights[i][j] = common_neighbors + 1 # Prioritize nodes with common neighbors

  return weights

def priority_v61(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version using neighborhood similarity."""
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
              neighbor_similarity += 1
      weights[i][j] = neighbor_similarity

  return weights

def priority_v62(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Combines degree difference and neighborhood similarity."""
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
      for k in range(n1):
        if graph1[i][k]:
          for l in range(n2):
            if graph2[j][l]:
              neighbor_similarity += 1

      # Combine both similarities (e.g., using a weighted average)
      weights[i][j] = 0.5 * degree_similarity + 0.5 * neighbor_similarity 
      # Adjust weights as needed based on the importance of each feature

  return weights

def priority_v64(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Improves `priority_v1` by considering common neighbors.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      common_neighbors = 0
      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] == 1 and graph2[j][l] == 1:  # Check for edge existence
            common_neighbors += 1
      weights[i][j] = common_neighbors

  # Normalize the weights (important for comparison and later use)
  total_weight = sum(sum(row) for row in weights)
  if total_weight > 0:
      for i in range(n1):
          for j in range(n2):
              weights[i][j] /= total_weight
  return weights

def priority_v66(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Considers neighbor degrees for improved probability estimation.
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
        neighbor_diff += min_diff
      weights[i][j] = 1.0 / (degree_diff + neighbor_diff + 1)
  return weights

def priority_v67(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0`.
  This version considers neighbor similarity in addition to node degree difference.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [([0.0] * max_node) for _ in range(max_node)]
  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      degree_diff = abs(degrees1[i] - degrees2[j])
      neighbor_similarity = 0

      neighbors1 = [k for k in range(n1) if graph1[i][k] == 1]
      neighbors2 = [k for k in range(n2) if graph2[j][k] == 1]

      for n1_neighbor in neighbors1:
        for n2_neighbor in neighbors2:
           # Check if the neighbors are also likely to be mapped to each other
           neighbor_similarity += weights[n1_neighbor][n2_neighbor] if n1_neighbor < len(weights) and n2_neighbor < len(weights[0]) else 0


      # Combine degree difference and neighbor similarity.  Weights are higher for smaller degree difference and higher neighbor similarity.
      weights[i][j] = 1.0 / (1 + degree_diff) +  neighbor_similarity / (len(neighbors1) * len(neighbors2) + 1) # Avoid division by zero
      
  return weights

def priority_v68(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    updated_weights = [[0.0] * max_node for _ in range(max_node)]  # Initialize with 0.0

    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]

    neighbors1 = [[] for _ in range(n1)]
    neighbors2 = [[] for _ in range(n2)]

    for i in range(n1):
        for j in range(n1):
            if graph1[i][j] == 1:
                neighbors1[i].append(j)

    for i in range(n2):
        for j in range(n2):
            if graph2[i][j] == 1:
                neighbors2[i].append(j)


    for i in range(n1):
        for j in range(n2):
            degree_diff = abs(degrees1[i] - degrees2[j])
            common_neighbors = 0

            # Efficiently count common neighbors using set intersection
            set1 = set(neighbors1[i])
            set2 = set(neighbors2[j])
            common_neighbors = len(set1.intersection(set2))


            if degrees1[i] > 0 and degrees2[j] > 0: # Avoid division by zero
                neighbor_similarity = (2 * common_neighbors) / (degrees1[i] + degrees2[j])
            elif degrees1[i] == 0 and degrees2[j] == 0: # Handle case where both nodes have degree 0
                neighbor_similarity = 1.0 # Consider them perfectly similar
            else:
                neighbor_similarity = 0.0


            updated_weights[i][j] = (1.0 / (1 + degree_diff)) * (1 + neighbor_similarity) #Combine degree difference and neighbor similarity

    return updated_weights

def priority_v70(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities based on common neighbors.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]
  for i in range(n1):
    for j in range(n2):
      common_neighbors = 0
      for k in range(min(n1, n2)):
        if graph1[i][k] == 1 and graph2[j][k] == 1:  # Simplified common neighbor check
          common_neighbors += 1
      weights[i][j] = common_neighbors  # Directly use the number of common neighbors
  return weights

def priority_v71(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Combines degree difference and common neighbors for initial probabilities.
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
      common_neighbors = 0
      for k in range(min(n1, n2)):
        if graph1[i][k] == 1 and graph2[j][k] == 1:
          common_neighbors += 1

      weights[i][j] = degree_similarity * (common_neighbors + 1) # Combine both factors

  return weights

def priority_v72(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
        weights[i][j] = (1.0 / (abs(degrees1[i] - degrees2[j]) + 1))
  return weights

