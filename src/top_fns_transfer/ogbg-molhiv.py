#Num Unique Functions in logs: 725
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
    Calculates node mapping probabilities based on structural similarity.
    """
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            degree_similarity = 1.0 / (abs(sum(graph1[i]) - sum(graph2[j])) + 1)  # Inversely proportional to degree difference

            neighbor_similarity = 0.0
            neighbors1 = [k for k in range(n1) if graph1[i][k]]
            neighbors2 = [k for k in range(n2) if graph2[j][k]]

            for n1_neighbor in neighbors1:
                for n2_neighbor in neighbors2:
                     neighbor_similarity += weights[n1_neighbor][n2_neighbor] # Encourage mappings that preserve neighbor relationships

            refined_weights[i][j] = degree_similarity  + neighbor_similarity # Combine degree and neighbor similarity

    # Normalize probabilities
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
      neighbor_similarity = 0
      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] == 1 and graph2[j][l] == 1:  # Check for edge existence in both graphs
            neighbor_similarity += weights[k][l] if k < len(weights) and l < len(weights[k]) else 0 # prevent indexerror
      refined_weights[i][j] = neighbor_similarity

  return refined_weights

def priority_v2(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights with zeros
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Iterate through all possible node mappings
    for i in range(n1):
        for j in range(n2):
            # Calculate a similarity score based on node degree and neighborhood similarity
            degree_similarity = 1.0 / (1.0 + abs(sum(graph1[i]) - sum(graph2[j])))
            neighborhood_similarity = 0.0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] == 1 and graph2[j][l] == 1:  # Check if nodes are neighbors in both graphs
                        neighborhood_similarity += weights[k][l] if k < len(weights) and l < len(weights[k]) else 0
            
            # Combine the similarity scores and initial weights
            refined_weights[i][j] = (degree_similarity + neighborhood_similarity) * (weights[i][j] if i < len(weights) and j < len(weights[i]) else 0)
        

    # Normalize the refined weights (optional, but can be beneficial)
    for i in range(n1):
        row_sum = sum(refined_weights[i])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum


    return refined_weights

def priority_v3(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`.

    This version calculates node similarity based on neighborhood structure and uses it to refine the initial weights.
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
                    if graph1[i][k] == 1 and graph2[j][l] == 1:  # Check if both edges exist
                        neighbor_similarity += weights[k][l]  # Add the weight of corresponding node mappings

            refined_weights[i][j] = (weights[i][j] + neighbor_similarity) / (1 + sum(graph1[i]) + sum(graph2[j])) # Normalize by degree

    return refined_weights

def priority_v4(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            node_similarity = weights[i][j]  # Use provided weights as a starting point

            # Consider structural similarity based on neighborhood
            neighbors_i = [k for k in range(n1) if graph1[i][k] == 1]
            neighbors_j = [k for k in range(n2) if graph2[j][k] == 1]

            common_neighbors = 0
            for ni in neighbors_i:
                for nj in neighbors_j:
                    if weights[ni][nj] > 0: # Check for potential mappings in the weight matrix
                        common_neighbors += 1

            # Combine node similarity and structural similarity (adjust weights as needed)
            structural_similarity = (common_neighbors) / (len(neighbors_i) + len(neighbors_j) + 1e-6) # Avoid division by zero

            refined_weights[i][j] =  node_similarity + structural_similarity  #Example combined weight. Experiment with different combinations/formulas.


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
          if graph1[i][k] == 1 and graph2[j][l] == 1:  # Check for edge existence
            neighbor_similarity += weights[k][l] if k < len(weights) and l < len(weights[k]) else 0

      refined_weights[i][j] = neighbor_similarity  # Update with neighbor similarity


  return refined_weights

def priority_v6(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Further improved version of `priority_v0` using normalization."""
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

      # Normalize by degree if either degree is non-zero
      if deg_i > 0 and deg_j > 0:
        refined_weights[i][j] = neighbor_similarity / (deg_i * deg_j)
      elif deg_i == 0 and deg_j == 0: #handle the cases when both nodes have 0 degree. 
        refined_weights[i][j] =  1 if i==j else 0 # if i == j then it's a valid mapping


  return refined_weights

def priority_v7(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities for node mappings based on node degree similarity.
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

def priority_v8(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0` considering neighbor degree similarity."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      neighbors1 = [k for k in range(n1) if graph1[i][k] == 1]
      neighbors2 = [k for k in range(n2) if graph2[j][k] == 1]

      for n1_neighbor in neighbors1:
        for n2_neighbor in neighbors2:
          neighbor_similarity += 1.0 / (1 + abs(sum(graph1[n1_neighbor]) - sum(graph2[n2_neighbor])))

      weights[i][j] = (1.0 / (1 + abs(sum(graph1[i]) - sum(graph2[j])))  # Node degree similarity
                      + neighbor_similarity) # Neighbor degree similarity

      # Normalize the weights
      if sum(weights[i]) > 0:
        weights[i] = [w / sum(weights[i]) for w in weights[i]]


  return weights

def priority_v9(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
            neighbor_similarity += weights[k][l] # Add the weight of the corresponding neighbor mapping

      refined_weights[i][j] = neighbor_similarity  # Update the weight based on neighbor similarity

  return refined_weights

def priority_v10(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0` with normalization."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      neighbor_count = 0  # Count the number of considered neighbor pairs
      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] == 1 and graph2[j][l] == 1:
            neighbor_similarity += weights[k][l]
            neighbor_count += 1


      if neighbor_count > 0:  # Normalize by the number of neighbor pairs
        refined_weights[i][j] = neighbor_similarity / neighbor_count
      #If no neighbors, refined weight remains 0 which makes sense.

  return refined_weights

def priority_v11(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Initialize the refined weights with zeros
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    # Iterate through potential node mappings
    for i in range(n1):
        for j in range(n2):
            # Calculate a score based on node degree and neighborhood similarity
            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
            neighborhood_similarity = 0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] == 1 and graph2[j][l] == 1:  # Check for common neighbors
                        neighborhood_similarity += weights[k][l] # Use initial weights here

            # Combine the initial weight, degree difference, and neighborhood similarity
            score = weights[i][j] * math.exp(-degree_diff) * (1 + neighborhood_similarity) # Exponential decay for degree diff
            refined_weights[i][j] = score


    # Normalize the refined weights (Optional, but can improve performance in some cases)
    for i in range(n1):
      row_sum = sum(refined_weights[i][:n2])
      if row_sum > 0:
        for j in range(n2):
          refined_weights[i][j] /= row_sum

    return refined_weights

def priority_v12(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
            for k in range(n1):
                if graph1[i][k]:
                    for l in range(n2):
                        if graph2[j][l]:
                           neighbor_similarity += weights[k][l] if k < len(weights) and l < len(weights[k]) else 0

            # Normalize by degrees to avoid bias towards high-degree nodes
            if degree1 > 0 and degree2 > 0:
                neighbor_similarity /= (degree1 * degree2)  

            refined_weights[i][j] = (weights[i][j] if i < len(weights) and j < len(weights[i]) else 0 ) + neighbor_similarity


    return refined_weights

def priority_v13(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`.
    Calculates refined node mapping probabilities based on structural similarity.
    """
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            # Calculate node similarity based on degree difference
            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
            degree_similarity = 1 / (1 + degree_diff)  # Higher similarity for smaller degree difference

            # Calculate neighborhood similarity
            neighbors1 = [k for k in range(n1) if graph1[i][k]]
            neighbors2 = [k for k in range(n2) if graph2[j][k]]

            common_neighbors = 0
            for n1_neighbor in neighbors1:
                for n2_neighbor in neighbors2:
                    if weights[n1_neighbor][n2_neighbor] > 0:  # Check if neighbors are potentially mapped
                        common_neighbors += 1

            neighborhood_similarity = (2 * common_neighbors) / (len(neighbors1) + len(neighbors2) + 1e-6) # avoid division by zero


            # Combine similarities and initial weights (if provided)
            if len(weights) >= n1 and len(weights[0]) >= n2:  #use weights if provided
                 refined_weights[i][j] = (degree_similarity + neighborhood_similarity + weights[i][j]) / 3
            else:
                 refined_weights[i][j] = (degree_similarity + neighborhood_similarity) / 2



    return refined_weights

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
            neighbor_similarity = 0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] == 1 and graph2[j][l] == 1:  # Check for edges in both graphs
                        neighbor_similarity += weights[k][l]  # Add similarity based on neighbors

            refined_weights[i][j] = weights[i][j] + neighbor_similarity  # Combine initial weight and neighborhood similarity


    # Normalize the refined weights to represent probabilities
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum
        else:
            # Handle cases where a node has no neighbors, distribute probability evenly
            for j in range(n2):
                refined_weights[i][j] = 1/n2



    return refined_weights

def priority_v15(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes a refined probability matrix for node mappings between two graphs.

    Args:
        graph1: The adjacency matrix of the first graph.
        graph2: The adjacency matrix of the second graph.
        weights: A weight matrix representing the initial probabilities of mapping nodes between `graph1` and `graph2`.

    Returns:
        A refined weight matrix (float) representing the probabilities of node mappings.
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


      refined_weights[i][j] = neighbor_similarity 

  # Normalize
  for i in range(n1):
    row_sum = sum(refined_weights[i])
    if row_sum > 0:
      for j in range(max_node):
        refined_weights[i][j] /= row_sum


  return refined_weights

def priority_v16(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0` with node degree consideration and normalization by maximum similarity."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]
  max_similarity = 0


  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      deg1 = sum(graph1[i])
      deg2 = sum(graph2[j])

      if deg1 > 0 and deg2 > 0 : # avoid division by zero
           for k in range(n1):
             for l in range(n2):
               if graph1[i][k] == 1 and graph2[j][l] == 1:
                 neighbor_similarity += weights[k][l] if k < len(weights) and l < len(weights[k]) else 0

           refined_weights[i][j] = neighbor_similarity / (deg1 * deg2) # consider node degrees
           max_similarity = max(max_similarity, refined_weights[i][j]) #Update maximum similarity

  # Normalize by maximum similarity
  if max_similarity > 0:
    for i in range(n1):
      for j in range(max_node):
        refined_weights[i][j] /= max_similarity

  return refined_weights

def priority_v17(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes a refined weight matrix based on graph structure similarity.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
      neighbor_similarity = 0
      for k in range(n1):
        if graph1[i][k]:
          for l in range(n2):
            if graph2[j][l]:
              neighbor_similarity += weights[k][l] if k < len(weights) and l < len(weights[k]) else 0

      refined_weights[i][j] = (1 / (1 + degree_diff)) * neighbor_similarity

  return refined_weights

def priority_v18(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Further improved version of `priority_v0` with normalization."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      degree_i = sum(graph1[i])
      degree_j = sum(graph2[j])
      if degree_i == 0 and degree_j == 0:
        degree_similarity = 1  # Handle isolated nodes
      elif degree_i == 0 or degree_j == 0:
        degree_similarity = 0 # penalize matching a node with edges to a node without edges
      else:
        degree_similarity = 1 / (1 + abs(degree_i - degree_j))

      neighbor_similarity = 0
      for k in range(n1):
        if graph1[i][k]:
          for l in range(n2):
            if graph2[j][l]:
              neighbor_similarity += weights[k][l] if k < len(weights) and l < len(weights[k]) else 0

      refined_weights[i][j] = degree_similarity * neighbor_similarity


  # Normalize rows
  for i in range(n1):
    row_sum = sum(refined_weights[i])
    if row_sum > 0:
      for j in range(n2):
        refined_weights[i][j] /= row_sum


  return refined_weights

def priority_v19(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
              neighbor_similarity += weights[k][l]  # Use the initial weights

      refined_weights[i][j] = neighbor_similarity

  return refined_weights

def priority_v20(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0` with normalization."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      norm_factor = 0 # Normalization factor

      for k in range(n1):
          if graph1[i][k] == 1:
              norm_factor +=1 # Number of neighbours in graph1


      for l in range(n2):
          if graph2[j][l] == 1:
               for k in range(n1): 
                   if graph1[i][k] == 1: # Check if there are any neighbours in graph 1 first
                       neighbor_similarity += weights[k][l] 
      if norm_factor > 0: # Check if graph1 node has neighbours
            refined_weights[i][j] = neighbor_similarity/norm_factor
      else:
            refined_weights[i][j] = 0


  return refined_weights

def priority_v21(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
            degree_similarity = 1.0 / (1.0 + abs(sum(graph1[i]) - sum(graph2[j])))

            neighborhood_similarity = 0.0
            common_neighbors = 0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] == 1 and graph2[j][l] == 1:  # Check for common neighbors
                        common_neighbors += 1
            if common_neighbors > 0:  # Avoid division by zero
                neighborhood_similarity = common_neighbors / (math.sqrt(sum(graph1[i]) * sum(graph2[j])))


            # Combine the initial weight, degree similarity, and neighborhood similarity
            refined_weights[i][j] = weights[i][j] * degree_similarity * neighborhood_similarity


    # Normalize the refined weights (optional but recommended) so they sum to 1 for each node in graph1
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2]) # Only sum up to n2 for valid mappings.
        if row_sum > 0:  # Avoid division by zero
            for j in range(n2):
                refined_weights[i][j] /= row_sum

    return refined_weights

def priority_v22(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
            # Calculate a similarity score based on the number of common neighbors
            common_neighbors = 0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] == 1 and graph2[j][l] == 1:
                        common_neighbors += 1
            
            # Incorporate the initial weights and the common neighbors into the refined weights
            if weights: # Check if weights are provided
                refined_weights[i][j] = weights[i][j] * (1 + common_neighbors) # Prioritize mappings with higher initial weights and more common neighbors
            else:
                refined_weights[i][j] = 1 + common_neighbors # If no initial weights are provided, solely rely on common neighbors


    # Normalize the refined weights to represent probabilities
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])  # Sum only up to n2
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum

    return refined_weights

def priority_v23(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`.
    Calculates node mapping probabilities based on structural similarity.
    """
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))  # Difference in degrees
            neighbor_similarity = 0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] == 1 and graph2[j][l] == 1:  # Check for common neighbors
                        neighbor_similarity += weights[k][l] if k < len(weights) and l < len(weights[k]) else 0


            # Calculate similarity based on degree difference and neighbor similarity
            similarity = (1 / (1 + degree_diff) + neighbor_similarity / max(sum(graph1[i]), sum(graph2[j])) if max(sum(graph1[i]), sum(graph2[j])) != 0 else 1/(1+degree_diff)) if max(n1,n2) > 1 else 1


            refined_weights[i][j] = similarity

    # Normalize weights (optional, but can improve performance)
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum

    return refined_weights

def priority_v24(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes a refined weight matrix based on graph structure and initial weights.
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
            neighbor_similarity += weights[k][l] # Add weight if edges exist in both graphs

      refined_weights[i][j] = neighbor_similarity # Update weight based on neighbor similarity


  return refined_weights

def priority_v25(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0` with normalization and node degree consideration."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  degree1 = [sum(row) for row in graph1]
  degree2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] == 1 and graph2[j][l] == 1:
            neighbor_similarity += weights[k][l]

      # Normalize by degree and add initial weight
      if degree1[i] > 0 and degree2[j] > 0: # Handle cases where degree is zero
          normalized_similarity = neighbor_similarity / (degree1[i] * degree2[j])
      elif degree1[i] == 0 and degree2[j] ==0:
          normalized_similarity = 1.0 # if both nodes have no neighbors, they are considered similar
      else:
          normalized_similarity = 0.0 # if only one node has no neighbors, they are considered dissimilar

      refined_weights[i][j] = normalized_similarity + weights[i][j]

  return refined_weights

def priority_v26(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
            degree_similarity = 1.0 / (abs(sum(graph1[i]) - sum(graph2[j])) + 1)  # Inverse of degree difference

            neighborhood_similarity = 0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] == 1 and graph2[j][l] == 1:  # If both nodes have an edge to another node
                        neighborhood_similarity += weights[k][l] if k < len(weights) and l < len(weights[0]) else 0 #Use initial weights


            # Combine the degree similarity and neighborhood similarity with the initial weights
            refined_weights[i][j] = (degree_similarity  + neighborhood_similarity) * (weights[i][j] if i < len(weights) and j < len(weights[0]) else 0)


    # Normalize the weights to represent probabilities
    row_sums = [sum(row) for row in refined_weights]
    for i in range(n1):
        for j in range(n2):
            if row_sums[i] > 0:
                refined_weights[i][j] /= row_sums[i]
            else: #Handle cases where the row sum is zero (no similar nodes found) – distribute probability evenly. Could also just leave as 0
                refined_weights[i][j] = 1/n2 if n2 > 0 else 0 

    return refined_weights

def priority_v27(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

def priority_v28(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

def priority_v30(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Refines probabilities based on neighbor similarity.
  """
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
            neighbor_similarity += weights[k][l]
      new_weights[i][j] = weights[i][j] * (1 + neighbor_similarity)  # Incorporate neighbor similarity

  return new_weights

def priority_v31(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Normalizes and refines probabilities using a Gaussian kernel.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  new_weights = [[0.0] * max_node for _ in range(max_node)]
  sigma = 1.0  # Adjust sigma as needed

  for i in range(n1):
    for j in range(n2):
      diff = 0
      for k in range(n1):
          for l in range(n2):
              diff += (graph1[i][k] - graph2[j][l])**2 * weights[k][l]
      new_weights[i][j] = math.exp(-diff / (2 * sigma**2)) #Gaussian Kernel


  # Normalize
  total_weight = sum(sum(row) for row in new_weights)
  if total_weight > 0:
    for i in range(n1):
      for j in range(n2):
        new_weights[i][j] /= total_weight

  return new_weights

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
      weights[i][j] = 1.0 / (abs(degrees1[i] - degrees2[j]) + 1)  # Prioritize similar degrees

  return weights

def priority_v33(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Refines probabilities based on neighbor degrees.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  new_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
        if graph1[i][k]:
          for l in range(n2):
            if graph2[j][l]:
              neighbor_similarity += weights[k][l] # Consider existing weights
      new_weights[i][j] = weights[i][j] * (1 + neighbor_similarity) # Combine with initial weights

  return new_weights

def priority_v34(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Normalizes probabilities.
  """

  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  new_weights = copy.deepcopy(weights)


  for i in range(n1):
      row_sum = sum(new_weights[i][:n2]) # Normalize per row against the other graph's nodes
      if row_sum > 0:
          for j in range(n2):
              new_weights[i][j] /= row_sum
      else: # Handle cases where a node has no similar neighbors
          for j in range(n2):
             new_weights[i][j] = 1/n2 if n2 >0 else 0


  return new_weights

def priority_v36(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Refines probabilities based on common neighbors.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  new_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      common_neighbors = 0
      for k in range(min(n1, n2)):
        if graph1[i][k] and graph2[j][k]:
          common_neighbors += 1
      new_weights[i][j] = weights[i][j] + common_neighbors  # Add bonus for common neighbors

  return new_weights

def priority_v37(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """
    Normalizes and refines probabilities based on neighbor similarity.
    """
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    new_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            neighbor_similarity = 0
            neighbors_i = [k for k in range(n1) if graph1[i][k]]
            neighbors_j = [k for k in range(n2) if graph2[j][k]]

            for ni in neighbors_i:
                for nj in neighbors_j:
                    neighbor_similarity += weights[ni][nj]  # Accumulate neighbor similarity

            new_weights[i][j] = weights[i][j] + neighbor_similarity

    # Normalize weights
    for i in range(n1):
        row_sum = sum(new_weights[i])
        if row_sum > 0:
            for j in range(n2):
                new_weights[i][j] /= row_sum

    return new_weights

def priority_v38(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`.  Uses numpy for efficiency and handles edge cases."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = np.zeros((max_node, max_node))
    graph1_np = np.array(graph1)
    graph2_np = np.array(graph2)
    weights_np = np.array(weights)

    for i in range(n1):
        for j in range(n2):
            degree_diff = abs(graph1_np[i].sum() - graph2_np[j].sum())
            neighborhood_similarity = (graph1_np[i, :n1, None] * graph2_np[None, j, :n2] * weights_np[:n1, :n2]).sum()
            score = weights_np[i, j] * math.exp(-degree_diff) * (1 + neighborhood_similarity)
            refined_weights[i, j] = score

    row_sums = refined_weights[:n1, :n2].sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    refined_weights[:n1, :n2] /= row_sums

    return refined_weights.tolist()

def priority_v39(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`.  Uses numpy for efficiency and handles cases where row_sum is zero."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = np.zeros((max_node, max_node))
  
  # Convert to numpy for faster calculations
  graph1_np = np.array(graph1)
  graph2_np = np.array(graph2)
  weights_np = np.array(weights)

  for i in range(n1):
    for j in range(n2):
        degree_diff = abs(graph1_np[i].sum() - graph2_np[j].sum())
        neighborhood_similarity = (graph1_np[i, :n1, np.newaxis] * graph2_np[np.newaxis, j, :n2] * weights_np[:n1, :n2]).sum()
        score = weights_np[i, j] * math.exp(-degree_diff) * (1 + neighborhood_similarity)
        refined_weights[i, j] = score

  row_sums = refined_weights[:, :n2].sum(axis=1)
  for i in range(n1):
      if row_sums[i] > 0:  # Avoid division by zero
          refined_weights[i, :n2] /= row_sums[i]

  return refined_weights.tolist()  # Convert back to list of lists

def priority_v40(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`.  Uses numpy for efficiency and handles edge cases."""
  n1 = len(graph1)
  n2 = len(graph2)

  if n1 == 0 or n2 == 0:  # Handle empty graphs
      return [[0.0] * max(n1, n2) for _ in range(max(n1, n2))]

  graph1_np = np.array(graph1)
  graph2_np = np.array(graph2)
  weights_np = np.array(weights)

  refined_weights = np.zeros((n1, n2))

  for i in range(n1):
      for j in range(n2):
          degree_diff = abs(graph1_np[i].sum() - graph2_np[j].sum())
          neighborhood_similarity = (graph1_np[i, :][:, np.newaxis] * graph2_np[j, :] * weights_np[:n1, :n2]).sum() # More efficient calculation
          score = (weights_np[i, j] * math.exp(-degree_diff) * (1 + neighborhood_similarity))
          refined_weights[i, j] = score

  row_sums = refined_weights.sum(axis=1, keepdims=True)
  row_sums[row_sums == 0] = 1  # Avoid division by zero
  refined_weights = refined_weights / row_sums


  max_node = max(n1, n2)
  final_weights = np.zeros((max_node, max_node))
  final_weights[:n1, :n2] = refined_weights  # Pad with zeros if necessary


  return final_weights.tolist()

def priority_v41(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            degree1 = sum(graph1[i])
            degree2 = sum(graph2[j])
            degree_diff = abs(degree1 - degree2)
            degree_similarity = 1 / (1 + degree_diff)

            neighbors1 = [k for k in range(n1) if graph1[i][k]]
            neighbors2 = [k for k in range(n2) if graph2[j][k]]

            common_neighbors = 0
            total_neighbors = len(neighbors1) + len(neighbors2)
            
            # Optimized neighbor comparison
            for n1_neighbor in neighbors1:
                for n2_neighbor in neighbors2:
                    if weights[n1_neighbor][n2_neighbor] > 0:
                        common_neighbors += 1

            if total_neighbors > 0: # Avoid division by zero
                neighborhood_similarity = (2 * common_neighbors) / total_neighbors
            else:
                neighborhood_similarity = 0  # If both nodes have no neighbors, similarity is 0 or 1 (depending on the desired behavior)
                if degree1 == 0 and degree2 ==0: # If both have degree 0, consider them similar
                    neighborhood_similarity = 1



            if len(weights) >= n1 and len(weights[0]) >= n2:
                refined_weights[i][j] = (degree_similarity + neighborhood_similarity + weights[i][j]) / 3
            else:
                refined_weights[i][j] = (degree_similarity + neighborhood_similarity) / 2

    return refined_weights

def priority_v42(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            degree1 = sum(graph1[i])
            degree2 = sum(graph2[j])
            degree_diff = abs(degree1 - degree2)
            degree_similarity = 1 / (1 + degree_diff)

            neighbors1 = [k for k in range(n1) if graph1[i][k]]
            neighbors2 = [k for k in range(n2) if graph2[j][k]]

            common_neighbors = 0
            for n1_neighbor in neighbors1:
                for n2_neighbor in neighbors2:
                    if len(weights) > n1_neighbor and len(weights[n1_neighbor]) > n2_neighbor and weights[n1_neighbor][n2_neighbor] > 0:  # Check bounds
                        common_neighbors += weights[n1_neighbor][n2_neighbor] # Weighted common neighbors

            neighborhood_similarity = (2 * common_neighbors) / (degree1 + degree2 + 1e-06)  # Use degrees instead of neighbor count


            if len(weights) >= n1 and len(weights[0]) >= n2:
                refined_weights[i][j] = (degree_similarity + neighborhood_similarity + weights[i][j]) / 3
            else:
                refined_weights[i][j] = (degree_similarity + neighborhood_similarity) / 2

    return refined_weights

def priority_v43(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`. Uses numpy for efficiency."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = np.zeros((max_node, max_node))
    
    # Convert to numpy arrays for faster calculations
    graph1_np = np.array(graph1)
    graph2_np = np.array(graph2)
    weights_np = np.array(weights)

    for i in range(n1):
        for j in range(n2):
            degree_diff = abs(graph1_np[i].sum() - graph2_np[j].sum())
            degree_similarity = 1 / (1 + degree_diff)

            neighbors1 = np.where(graph1_np[i] == 1)[0]
            neighbors2 = np.where(graph2_np[j] == 1)[0]

            # Efficiently calculate common neighbors using numpy
            common_neighbors = np.sum(weights_np[np.ix_(neighbors1, neighbors2)] > 0)

            neighborhood_similarity = (2 * common_neighbors) / (len(neighbors1) + len(neighbors2) + 1e-06)
            
            if weights_np.shape[0] >= n1 and weights_np.shape[1] >= n2:
                refined_weights[i, j] = (degree_similarity + neighborhood_similarity + weights_np[i, j]) / 3
            else:
                refined_weights[i, j] = (degree_similarity + neighborhood_similarity) / 2

    return refined_weights.tolist() # Convert back to list for consistency

def priority_v44(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes the Graph Edit Distance (GED), a measure of the dissimilarity between two graphs. 
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
                    neighbor_similarity += weights[k][l]
        refined_weights[i][j] = (weights[i][j] + neighbor_similarity) / (1 + sum(graph1[i]) + sum(graph2[j]))
  return refined_weights

def priority_v45(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0` using numpy for efficiency."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  graph1_np = np.array(graph1)
  graph2_np = np.array(graph2)
  weights_np = np.array(weights)

  refined_weights = np.zeros((max_node, max_node))
  for i in range(n1):
    for j in range(n2):
        neighbor_similarity = np.sum(graph1_np[i, : , np.newaxis] * graph2_np[np.newaxis, j, :] * weights_np)
        refined_weights[i, j] = (weights_np[i, j] + neighbor_similarity) / (1 + np.sum(graph1_np[i]) + np.sum(graph2_np[j]))

  return refined_weights.tolist()

def priority_v46(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes a refined weight matrix based on node neighborhood similarity.
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
                    neighbor_similarity += weights[k][l]
        refined_weights[i][j] = (weights[i][j] + neighbor_similarity) / (1 + sum(graph1[i]) + sum(graph2[j]))
  return refined_weights

def priority_v47(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0` using numpy for efficiency."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  graph1_np = np.array(graph1)
  graph2_np = np.array(graph2)
  weights_np = np.array(weights)
  refined_weights = np.zeros((max_node, max_node))

  for i in range(n1):
      for j in range(n2):
          neighbor_similarity = np.sum(weights_np * np.outer(graph1_np[i], graph2_np[j]))
          refined_weights[i, j] = (weights_np[i, j] + neighbor_similarity) / (1 + np.sum(graph1_np[i]) + np.sum(graph2_np[j]))
  return refined_weights.tolist()

def priority_v48(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`. Uses numpy for efficiency."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    graph1_np = np.array(graph1)
    graph2_np = np.array(graph2)
    weights_np = np.array(weights)

    refined_weights = np.zeros((max_node, max_node), dtype=float)

    for i in range(n1):
        for j in range(n2):
            neighbor_similarity = np.sum(graph1_np[i, :][:, np.newaxis] * graph2_np[j, :] * weights_np[:n1, :n2])
            refined_weights[i, j] = (weights_np[i, j] + neighbor_similarity) / (1 + np.sum(graph1_np[i]) + np.sum(graph2_np[j]))

    return refined_weights.tolist()

def priority_v49(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes the Graph Edit Distance (GED), a measure of the dissimilarity between two graphs. 
    GED is defined as the minimum number of operations required to transform one graph into another.

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
                    neighbor_similarity += weights[k][l]
        refined_weights[i][j] = (weights[i][j] + neighbor_similarity) / (1 + sum(graph1[i]) + sum(graph2[j]))
  return refined_weights

def priority_v50(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0` using numpy for efficiency."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  graph1_np = np.array(graph1)
  graph2_np = np.array(graph2)
  weights_np = np.array(weights)
  refined_weights = np.zeros((max_node, max_node))

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = np.sum(weights_np * np.outer(graph1_np[i], graph2_np[j]))
      refined_weights[i, j] = (weights_np[i, j] + neighbor_similarity) / (1 + np.sum(graph1_np[i]) + np.sum(graph2_np[j]))

  return refined_weights.tolist()

def priority_v51(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Further improved version of `priority_v1` with pre-computed sums."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  graph1_np = np.array(graph1)
  graph2_np = np.array(graph2)
  weights_np = np.array(weights)
  refined_weights = np.zeros((max_node, max_node))

  graph1_sums = np.sum(graph1_np, axis=1)
  graph2_sums = np.sum(graph2_np, axis=1)


  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = np.sum(weights_np * np.outer(graph1_np[i], graph2_np[j]))
      refined_weights[i, j] = (weights_np[i, j] + neighbor_similarity) / (1 + graph1_sums[i] + graph2_sums[j])

  return refined_weights.tolist()

def priority_v52(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`.
    This version incorporates normalization within the neighborhood similarity calculation to prevent
    overemphasis on nodes with larger neighborhoods.
    """
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
            neighborhood_similarity = 0
            neighbor_count1 = sum(graph1[i])
            neighbor_count2 = sum(graph2[j])

            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] == 1 and graph2[j][l] == 1:
                        if neighbor_count1 > 0 and neighbor_count2 > 0:  # Normalize if neighbors exist
                            neighborhood_similarity += weights[k][l] / (neighbor_count1 * neighbor_count2) #changed to product of neighbor counts
                        elif neighbor_count1 == 0 and neighbor_count2 == 0: #If they both have degree of 0, give maximum match
                            neighborhood_similarity = 1

            score = (weights[i][j] * math.exp(-degree_diff)) * (1 + neighborhood_similarity)
            refined_weights[i][j] = score

    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum
    return refined_weights

def priority_v53(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`.  Uses numpy for efficiency and handles zero-degree nodes."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = np.zeros((max_node, max_node), dtype=float)
  degrees1 = np.sum(graph1, axis=1)
  degrees2 = np.sum(graph2, axis=1)

  for i in range(n1):
    for j in range(n2):
      degree_diff = abs(degrees1[i] - degrees2[j])
      neighborhood_similarity = 0
      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] == 1 and graph2[j][l] == 1:
            neighborhood_similarity += weights[k][l]

      # Handle cases where degree difference is large, avoiding potential overflow with exp
      if degree_diff > 700:  # exp(-700) is close to zero
          score = 0.0
      else:
          score = (weights[i][j] * math.exp(-degree_diff)) * (1 + neighborhood_similarity)
      refined_weights[i, j] = score

  row_sums = np.sum(refined_weights[:, :n2], axis=1)
  for i in range(n1):
    if row_sums[i] > 0:
      refined_weights[i, :n2] /= row_sums[i]
    # Handle cases where the row sum is zero (isolated nodes)
    else:
       refined_weights[i, :n2] = 1/n2 # Distribute probability evenly if no connections


  return refined_weights.tolist()

def priority_v54(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`.  Uses numpy for efficiency and handles edge cases."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = np.zeros((max_node, max_node))
  weights = np.array(weights)

  for i in range(n1):
    for j in range(n2):
      degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
      neighborhood_similarity = 0
      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] == 1 and graph2[j][l] == 1:
            neighborhood_similarity += weights[k, l]

      score = weights[i, j] * math.exp(-degree_diff) * (1 + neighborhood_similarity)
      refined_weights[i, j] = score

  for i in range(n1):
    row_sum = np.sum(refined_weights[i, :n2])
    if row_sum > 0:
      refined_weights[i, :n2] /= row_sum
    # Handle case where row_sum is 0 (no similar neighbors): distribute probability evenly
    elif n2 > 0:
        refined_weights[i, :n2] = 1.0 / n2


  return refined_weights.tolist()  # Convert back to list of lists

def priority_v55(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`.  Uses numpy for efficiency and handles
    zero row sums more robustly by distributing probability equally."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    graph1_np = np.array(graph1)
    graph2_np = np.array(graph2)
    weights_np = np.array(weights)

    refined_weights = np.zeros((max_node, max_node))

    for i in range(n1):
        for j in range(n2):
            degree_diff = abs(graph1_np[i].sum() - graph2_np[j].sum())
            neighborhood_similarity = (graph1_np[i, :n1, None] * graph2_np[None, j, :n2] * weights_np[:n1, :n2]).sum()
            score = weights_np[i, j] * math.exp(-degree_diff) * (1 + neighborhood_similarity)
            refined_weights[i, j] = score

    for i in range(n1):
        row_sum = refined_weights[i, :n2].sum()
        if row_sum > 0:
            refined_weights[i, :n2] /= row_sum
        else: # Distribute probability equally if row sum is zero.
            refined_weights[i, :n2] = 1.0 / n2

    return refined_weights.tolist()

def priority_v56(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`. Uses numpy for efficiency and handles edge weights."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = np.zeros((max_node, max_node))
    
    # Convert to numpy for faster calculations
    graph1_np = np.array(graph1)
    graph2_np = np.array(graph2)
    weights_np = np.array(weights)

    for i in range(n1):
        for j in range(n2):
            degree1 = graph1_np[i].sum()
            degree2 = graph2_np[j].sum()
            degree_diff = abs(degree1 - degree2)
            degree_similarity = 1 / (1 + degree_diff)

            neighbors1 = np.where(graph1_np[i] > 0)[0]
            neighbors2 = np.where(graph2_np[j] > 0)[0]

            common_neighbors_weighted = 0
            for n1_neighbor in neighbors1:
                for n2_neighbor in neighbors2:
                    if weights_np.shape[0] > n1_neighbor and weights_np.shape[1] > n2_neighbor:  # Check bounds
                        common_neighbors_weighted += weights_np[n1_neighbor, n2_neighbor] * graph1_np[i,n1_neighbor]*graph2_np[j,n2_neighbor] # Consider edge weights


            neighborhood_similarity = (2 * common_neighbors_weighted) / (degree1+degree2 + 1e-06)  # Use weighted common neighbors and degrees

            if weights_np.shape[0] > i and weights_np.shape[1] > j:
                refined_weights[i, j] = (degree_similarity + neighborhood_similarity + weights_np[i, j]) / 3
            else:
                refined_weights[i, j] = (degree_similarity + neighborhood_similarity) / 2

    return refined_weights.tolist()  # Convert back to list for consistency

def priority_v57(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`. Uses numpy for efficiency and handles edge weights."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = np.zeros((max_node, max_node))
    
    # Convert to numpy arrays for faster calculations
    graph1 = np.array(graph1)
    graph2 = np.array(graph2)
    weights = np.array(weights)

    for i in range(n1):
        for j in range(n2):
            degree1 = np.sum(graph1[i])  # Use np.sum for efficiency
            degree2 = np.sum(graph2[j])
            degree_diff = abs(degree1 - degree2)
            degree_similarity = 1 / (1 + degree_diff)
            
            neighbors1 = np.where(graph1[i] > 0)[0]
            neighbors2 = np.where(graph2[j] > 0)[0]

            common_neighbors = 0
            for n1_neighbor in neighbors1:
                for n2_neighbor in neighbors2:
                    if weights.shape[0] > n1_neighbor and weights.shape[1] > n2_neighbor and weights[n1_neighbor, n2_neighbor] > 0:
                         common_neighbors +=1


            neighborhood_similarity = (2 * common_neighbors) / (len(neighbors1) + len(neighbors2) + 1e-06)

            if weights.shape[0] >= n1 and weights.shape[1] >= n2:
                refined_weights[i, j] = (degree_similarity + neighborhood_similarity + weights[i, j]) / 3
            else:
                refined_weights[i, j] = (degree_similarity + neighborhood_similarity) / 2

    return refined_weights.tolist()  # Convert back to list for consistency

def priority_v58(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`. Uses numpy for efficiency."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = np.zeros((max_node, max_node))
    
    # Convert to numpy arrays for faster processing
    graph1_np = np.array(graph1)
    graph2_np = np.array(graph2)
    weights_np = np.array(weights)

    degrees1 = graph1_np.sum(axis=1)
    degrees2 = graph2_np.sum(axis=1)

    for i in range(n1):
        for j in range(n2):
            degree_diff = abs(degrees1[i] - degrees2[j])
            degree_similarity = 1 / (1 + degree_diff)

            neighbors1 = np.where(graph1_np[i])[0]
            neighbors2 = np.where(graph2_np[j])[0]

            # Efficiently calculate common neighbors using numpy
            common_neighbors = np.sum(weights_np[neighbors1][:, neighbors2])  

            neighborhood_similarity = (2 * common_neighbors) / (len(neighbors1) + len(neighbors2) + 1e-06)

            if weights_np.shape[0] >= n1 and weights_np.shape[1] >= n2:
                refined_weights[i, j] = (degree_similarity + neighborhood_similarity + weights_np[i, j]) / 3
            else:
                refined_weights[i, j] = (degree_similarity + neighborhood_similarity) / 2
    return refined_weights.tolist()  # Convert back to list for consistency

def priority_v59(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`. Uses numpy for efficiency."""
  n1 = len(graph1)
  n2 = len(graph2)

  graph1_np = np.array(graph1)
  graph2_np = np.array(graph2)
  weights_np = np.array(weights)

  max_node = max(n1, n2)
  refined_weights = np.zeros((max_node, max_node))

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = np.sum(graph1_np[i, :n1].reshape(-1, 1) * graph2_np[j, :n2] * weights_np[:n1, :n2])
      refined_weights[i, j] = (weights_np[i, j] + neighbor_similarity) / (1 + np.sum(graph1_np[i]) + np.sum(graph2_np[j]))

  return refined_weights.tolist()

def priority_v60(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`. Uses numpy for efficiency."""
    n1 = len(graph1)
    n2 = len(graph2)

    graph1_np = np.array(graph1)
    graph2_np = np.array(graph2)
    weights_np = np.array(weights)

    max_node = max(n1, n2)
    refined_weights = np.zeros((max_node, max_node))

    for i in range(n1):
        for j in range(n2):
            neighbor_similarity = np.sum(graph1_np[i, :].reshape(-1, 1) * graph2_np[j, :] * weights_np[:n1, :n2])
            refined_weights[i, j] = (weights_np[i, j] + neighbor_similarity) / (1 + np.sum(graph1_np[i]) + np.sum(graph2_np[j]))

    return refined_weights.tolist()

def priority_v61(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes a refined weight matrix based on graph structure and initial weights.
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
                    neighbor_similarity += weights[k][l]
        refined_weights[i][j] = (weights[i][j] + neighbor_similarity) / (1 + sum(graph1[i]) + sum(graph2[j]))
  return refined_weights

def priority_v62(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0` using numpy for efficiency."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  graph1_np = np.array(graph1)
  graph2_np = np.array(graph2)
  weights_np = np.array(weights)
  refined_weights = np.zeros((max_node, max_node))

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = np.sum(weights_np[graph1_np[i,:] == 1,:][:,graph2_np[j,:] == 1])
      refined_weights[i, j] = (weights_np[i, j] + neighbor_similarity) / (1 + np.sum(graph1_np[i,:]) + np.sum(graph2_np[j,:]))
  
  return refined_weights.tolist()

def priority_v63(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Further improved version using matrix multiplication for neighbor similarity calculation."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    graph1_np = np.array(graph1, dtype=np.float64)
    graph2_np = np.array(graph2, dtype=np.float64)
    weights_np = np.array(weights, dtype=np.float64)
    refined_weights = np.zeros((max_node, max_node))

    neighbor_similarity = np.matmul(graph1_np, np.matmul(weights_np, graph2_np.T))


    for i in range(n1):
        for j in range(n2):
            refined_weights[i, j] = (weights_np[i, j] + neighbor_similarity[i, j]) / (1 + np.sum(graph1_np[i,:]) + np.sum(graph2_np[j,:]))

    return refined_weights.tolist()

def priority_v66(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            degree_similarity = 1.0 / (1.0 + abs(degrees1[i] - degrees2[j]))

            neighbor_similarity = 0
            common_neighbors = 0
            for k in range(n1):
                if graph1[i][k]:
                    for l in range(n2):
                        if graph2[j][l]:
                            neighbor_weight = weights[k][l] if k < len(weights) and l < len(weights[k]) else 0
                            neighbor_similarity += neighbor_weight
                            common_neighbors += 1


            if degrees1[i] > 0 and degrees2[j] > 0 and common_neighbors > 0:
                neighbor_similarity /= (degrees1[i] * degrees2[j]) # Normalize by the product of degrees

            refined_weights[i][j] = degree_similarity * (1 + neighbor_similarity)


    return refined_weights

def priority_v67(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [([0.0] * max_node) for _ in range(max_node)]

  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      degree_similarity = (1.0 / (1.0 + abs(degrees1[i] - degrees2[j]))) if (degrees1[i] > 0 and degrees2[j] > 0) else 0.0

      neighbor_similarity = 0
      for k in range(n1):
        if graph1[i][k]:
          for l in range(n2):
            if graph2[j][l]:
              neighbor_similarity += (weights[k][l] if ((k < len(weights)) and (l < len(weights[k]))) else 0)

      if degrees1[i] > 0 and degrees2[j] > 0:
        neighbor_similarity /= (degrees1[i] * degrees2[j])

      refined_weights[i][j] = (degree_similarity + neighbor_similarity)


  return refined_weights

