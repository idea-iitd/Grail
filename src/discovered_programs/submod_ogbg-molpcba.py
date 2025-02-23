import copy
import itertools
import math
import numpy as np


def priority_v1(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1` using numpy for efficiency."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = np.zeros((max_node, max_node), dtype=float)
    degrees1 = np.array([sum(row) for row in graph1])
    degrees2 = np.array([sum(row) for row in graph2])
    weights = np.array(weights)

    for i in range(n1):
        for j in range(n2):
            degree_similarity = 1.0 / (1.0 + abs(degrees1[i] - degrees2[j]))

            neighbors1 = np.where(graph1[i])[0]
            neighbors2 = np.where(graph2[j])[0]

            neighborhood_similarity = 0.0
            
            if len(neighbors1) > 0 and len(neighbors2) > 0:
                neighbor_weights = weights[neighbors1[:, None], neighbors2]
                neighborhood_similarity = np.sum(neighbor_weights)
                union_neighbors = len(neighbors1) + len(neighbors2) - np.sum(neighbor_weights > 0)
                if union_neighbors > 0:
                    neighborhood_similarity /= union_neighbors
            
            initial_weight = weights[i, j] if (0 <= i < len(weights) and 0 <= j < len(weights[0])) else (1.0 / max_node)
            refined_weights[i, j] = ((degree_similarity + neighborhood_similarity) / 2) * initial_weight
            
    return refined_weights.tolist()

def priority_v2(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [([0.0] * max_node) for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
            neighbor_similarity = 0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] == 1 and graph2[j][l] == 1:
                        neighbor_similarity += weights[k][l]

            normalization_factor = sum(graph1[i]) * sum(graph2[j])
            if normalization_factor > 0:
                neighbor_similarity /= normalization_factor

            # Enhanced weight calculation considering both degree difference and neighbor similarity
            refined_weights[i][j] = (1.0 / (1.0 + degree_diff)) * (1.0 + neighbor_similarity)

            # Additional check for structural similarity beyond immediate neighbors
            common_neighbors = 0
            for k in range(n1):
                for l in range(n2):
                  if graph1[i][k] and graph2[j][l]:
                    for m in range(n1):
                      for n in range(n2):
                        if graph1[k][m] and graph2[l][n]:
                          common_neighbors += weights[m][n]
            if common_neighbors > 0 :
              refined_weights[i][j] =  refined_weights[i][j] +  (common_neighbors / (normalization_factor * normalization_factor)) if normalization_factor > 0 else  refined_weights[i][j] + common_neighbors



    return refined_weights

def priority_v3(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]
    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            initial_weight = weights[i][j] if i < len(weights) and j < len(weights[0]) else 0.0

            neighbors1 = [k for k in range(n1) if graph1[i][k] == 1]
            neighbors2 = [k for k in range(n2) if graph2[j][k] == 1]

            common_neighbors = 0
            for neighbor1 in neighbors1:
                for neighbor2 in neighbors2:
                    if neighbor1 < len(weights) and neighbor2 < len(weights[0]) and weights[neighbor1][neighbor2] > 0:
                        common_neighbors += 1

            neighbor_similarity = 0.0
            if neighbors1 and neighbors2:
                neighbor_similarity = common_neighbors / (math.sqrt(len(neighbors1) * len(neighbors2)))
            
            degree_similarity = 1.0 / (abs(degrees1[i] - degrees2[j]) + 1)

            # Give more weight to initial weights if they are significant, otherwise rely more on structural similarity
            if initial_weight > 0.5:  # Example threshold, adjust as needed
                refined_weights[i][j] = (0.7 * initial_weight + 0.15 * neighbor_similarity + 0.15 * degree_similarity)
            else:
                refined_weights[i][j] = (0.3 * initial_weight + 0.35 * neighbor_similarity + 0.35 * degree_similarity)

    return refined_weights


def priority_v41(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      weights[i][j] = (1.0 / (abs(degrees1[i] - degrees2[j]) + 1)) * (neighbor_similarity + 1)  # Adding 1 to avoid multiplication by zero

  return weights



def priority_v4(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Improves `priority_v1` by normalizing the weights.
  """
  weights = priority_v41(graph1, graph2, weights)
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)

  # Normalize weights
  max_weight = 0
  for i in range(n1):
    for j in range(n2):
        max_weight = max(max_weight, weights[i][j])
  
  if max_weight > 0: # Avoid division by zero if all weights are zero
    for i in range(n1):
      for j in range(n2):
        weights[i][j] /= max_weight

  return weights

def priority_v50(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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


def priority_v51(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Refines initial probabilities using neighbor degrees.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = priority_v50(graph1, graph2, weights)  # Initialize with degree-based probabilities

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for neighbor1 in range(n1):
        if graph1[i][neighbor1]:
          best_match_score = 0
          for neighbor2 in range(n2):
            if graph2[j][neighbor2]:
              best_match_score = max(best_match_score, weights[neighbor1][neighbor2])
          neighbor_similarity += best_match_score
      weights[i][j] *= (1 + neighbor_similarity)

  return weights


def priority_v5(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Normalizes probabilities and incorporates initial weights (if provided).
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)

  if weights is None or len(weights) != max_node or len(weights[0]) != max_node:
      weights = [[0.0] * max_node for _ in range(max_node)]
      initial_weights = False
  else:
      initial_weights = True

  refined_weights = priority_v51(graph1, graph2, weights)

  for i in range(n1):
      for j in range(n2):
          if initial_weights:
              weights[i][j] = refined_weights[i][j] * weights[i][j]  # Combine with initial weights
          else:
              weights[i][j] = refined_weights[i][j]

  # Normalize
  for i in range(n1):
    row_sum = sum(weights[i][:n2])
    if row_sum > 0:
      for j in range(n2):
        weights[i][j] /= row_sum

  return weights

def priority_v6(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0` with normalization and initial weight consideration."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
      neighbor_similarity = 0
      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] == 1 and graph2[j][l] == 1:
              neighbor_similarity += weights[k][l] if k < len(weights) and l < len(weights[0]) else 0 # Use initial weights

      refined_weights[i][j] = (1.0 / (1.0 + degree_diff)) * (1.0 + neighbor_similarity) + (weights[i][j] if i < len(weights) and j < len(weights[0]) else 0)

  # Normalize refined_weights (important for probabilities)
  for i in range(n1):
    row_sum = sum(refined_weights[i][:n2])
    if row_sum > 0:
        for j in range(n2):
            refined_weights[i][j] /= row_sum


  return refined_weights

def priority_v7(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0` using initial weights."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      cost = 0

      # Edge cost, weighted by initial probabilities
      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] != graph2[j][l]:
            cost += 1 * weights[k][l] if k < len(weights) and l < len(weights[0]) else 1 # Use provided weights if available

      if i < len(weights) and j < len(weights[0]):
          refined_weights[i][j] = weights[i][j] / (1.0 + cost) if cost > 0 and weights[i][j] != 0 else weights[i][j]  # Incorporate and normalize initial weights
      else: #handle cases where initial weights are smaller than the actual graph size
          refined_weights[i][j] = 1.0 / (1.0 + cost) if cost > 0 else 1.0




  return refined_weights

def priority_v81(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      neighbor_degrees1 = [degrees1[k] for k in range(n1) if graph1[i][k]]
      neighbor_degrees2 = [degrees2[k] for k in range(n2) if graph2[j][k]]
      
      degree_diff = abs(degrees1[i] - degrees2[j])
      
      neighbor_diff = 0
      for deg1 in neighbor_degrees1:
          min_diff = float('inf')
          for deg2 in neighbor_degrees2:
              min_diff = min(min_diff, abs(deg1 - deg2))
          if min_diff != float('inf'):
              neighbor_diff += min_diff
              
      weights[i][j] = 1.0 / (degree_diff + neighbor_diff + 1)
      
  return weights




def priority_v8(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Normalizes probabilities for better comparison.
  """
  weights = priority_v81(graph1, graph2, weights)  # Use v1 as a base
  n1 = len(graph1)
  n2 = len(graph2)
  
  for i in range(n1):
    row_sum = sum(weights[i][:n2])  # Only sum up to n2
    if row_sum > 0:
      for j in range(n2):
        weights[i][j] /= row_sum
  return weights


def priority_v9(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1` using numpy for efficiency."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = np.zeros((max_node, max_node))
    graph1_np = np.array(graph1)
    graph2_np = np.array(graph2)
    weights_np = np.array(weights)

    degrees1 = graph1_np.sum(axis=1)
    degrees2 = graph2_np.sum(axis=1)

    for i in range(n1):
        for j in range(n2):
            degree_diff = abs(degrees1[i] - degrees2[j])
            neighbor_similarity = np.sum(graph1_np[i, :][:, np.newaxis] * graph2_np[j, :] * weights_np[:n1, :n2])
            normalization_factor = degrees1[i] * degrees2[j]

            if normalization_factor > 0:
                neighbor_similarity /= normalization_factor

            refined_weights[i, j] = (1.0 / (1.0 + degree_diff)) * (1.0 + neighbor_similarity)

            common_neighbors = np.sum(graph1_np[i, :][:, np.newaxis] * graph2_np[j, :] * np.dot(graph1_np[:n1,:n1], weights_np[:n1, :n2]) * graph2_np[:n2, :n2])


            if common_neighbors > 0:
                if normalization_factor > 0:
                    refined_weights[i, j] += common_neighbors / (normalization_factor * normalization_factor)
                else:
                    refined_weights[i, j] += common_neighbors

    return refined_weights.tolist()


def priority_v101(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Refines initial probabilities using neighbor degree similarity.
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
      new_weights[i][j] = weights[i][j] + neighbor_similarity

  # Normalize
  max_weight = max(max(row) for row in new_weights) if any(any(row) for row in new_weights) else 1.0 # avoid division by zero
  for i in range(n1):
      for j in range(n2):
          new_weights[i][j] /= max_weight

  return new_weights


def priority_v10(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Iteratively refines probabilities using neighbor similarity (like v1, but iterated).
  """
  n1 = len(graph1)
  n2 = len(graph2)
  current_weights = copy.deepcopy(weights)

  for _ in range(3):  # Perform a few iterations
      current_weights = priority_v101(graph1, graph2, current_weights)

  return current_weights


def priority_v11(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            node_similarity = weights[i][j]  # Use the provided weights as a base

            # Consider edge similarity
            neighbors1 = [k for k in range(n1) if graph1[i][k]]
            neighbors2 = [k for k in range(n2) if graph2[j][k]]

            common_neighbors = 0
            for n1_neighbor in neighbors1:
                for n2_neighbor in neighbors2:
                    common_neighbors += weights[n1_neighbor][n2_neighbor]  # Leverage existing weights

            edge_similarity = common_neighbors / (len(neighbors1) * len(neighbors2) + 1e-6) # Avoid division by zero


            # Combine node and edge similarity (you can adjust the weights)
            refined_weights[i][j] = 0.5 * node_similarity + 0.5 * edge_similarity

    # Normalize the weights (optional, but often helpful)
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2]) # normalize only within the valid part of the matrix.
        if row_sum > 0:
          for j in range(n2):
            refined_weights[i][j] /= row_sum


    return refined_weights


def priority_v121(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Considers neighbor degrees for initial mapping probabilities.
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
      neighbor_diff = sum(abs(d1 - d2) for d1, d2 in itertools.zip_longest(neighbor_degrees1, neighbor_degrees2, fillvalue=0))
      weights[i][j] = 1.0 / (degree_diff + neighbor_diff + 1)
  return weights



def priority_v12(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Normalizes probabilities within each row to sum to 1.
  """
  weights = priority_v121(graph1, graph2, weights) # Or use v0 as a base
  n1 = len(graph1)
  n2 = len(graph2)

  for i in range(n1):
    row_sum = sum(weights[i][:n2])  # Only consider relevant part of the row
    if row_sum > 0:
        for j in range(n2):
            weights[i][j] /= row_sum

  return weights

def priority_v130(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities for node mappings based on degree difference.
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


def priority_v131(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Improves `priority_v0` by normalizing weights.
  """
  weights = priority_v130(graph1, graph2, weights)
  n1 = len(graph1)
  n2 = len(graph2)
  for i in range(n1):
    row_sum = sum(weights[i][:n2])  # Sum only over valid nodes in graph2
    if row_sum > 0:
        for j in range(n2):
            weights[i][j] /= row_sum
  return weights


def priority_v13(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Improves `priority_v1` by considering neighbor similarity.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  weights = priority_v131(graph1, graph2, weights)  # Initialize with v1

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
          if graph1[i][k]:
              for l in range(n2):
                  if graph2[j][l]:
                      neighbor_similarity += weights[k][l]
      weights[i][j] *= (1 + neighbor_similarity)  # Boost by neighbor similarity

  for i in range(n1):
      row_sum = sum(weights[i][:n2])
      if row_sum > 0:
          for j in range(n2):
              weights[i][j] /= row_sum # Renormalize

  return weights



def priority_v141(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Considers neighbor degrees for improved probability estimation."""
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



def priority_v14(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Normalizes probabilities for better comparison."""
  weights = priority_v141(graph1, graph2, weights)  # Use v1 as a base
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)

  # Normalize across rows (for each node in graph1)
  for i in range(n1):
    row_sum = sum(weights[i][:n2])  # Only sum within the valid range of graph2
    if row_sum > 0:
      for j in range(n2):
        weights[i][j] /= row_sum

  return weights


def priority_v15(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    weights = [[0.0] * max_node for _ in range(max_node)]
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
            common_neighbors = 0
            for neighbor1 in neighbors1[i]:
                for neighbor2 in neighbors2[j]:
                    # Key improvement: consider existing weight in common neighbor calculation
                    if weights[neighbor1][neighbor2] > 0:  
                        common_neighbors += weights[neighbor1][neighbor2] # weighted sum
            
            degree_diff = abs(degrees1[i] - degrees2[j])
            degree_similarity = 1.0 / (1.0 + degree_diff) # avoid division by zero

            neighbor_similarity = 0.0
            if neighbors1[i] or neighbors2[j]: # avoid division by zero
                neighbor_similarity = (1.0 + common_neighbors) / (1.0 + max(len(neighbors1[i]), len(neighbors2[j])))
            

            weights[i][j] = (degree_similarity * neighbor_similarity)
            
    return weights
# Input Scores: [39.0, 24.0, 21.0, 30.0, 33.0, 42.0, 52.0, 41.0, 47.0, 21.0, 38.0, 23.0, 29.0, 37.0, 25.0, 26.0, 24.0, 50.0, 44.0, 33.0, 42.0, 33.0, 25.0, 19.0, 48.0, 43.0, 41.0, 35.0, 51.0, 38.0, 42.0, 38.0, 56.0, 12.0, 27.0, 28.0, 40.0, 32.0, 23.0, 51.0, 20.0, 35.0, 22.0, 45.0, 30.0, 40.0, 21.0, 32.0, 33.0, 31.0, 29.0, 35.0, 24.0, 29.0, 37.0, 20.0, 21.0, 24.0, 26.0, 27.0, 26.0, 35.0, 41.0, 42.0, 45.0, 30.0, 43.0, 33.0, 35.0, 47.0, 30.0, 29.0, 35.0, 33.0, 39.0, 26.0, 31.0, 50.0, 16.0, 30.0, 25.0, 37.0, 25.0, 27.0, 23.0, 36.0, 41.0, 16.0, 40.0, 49.0, 28.0, 40.0, 27.0, 19.0, 24.0, 27.0, 35.0, 30.0, 31.0, 35.0, 20.0, 38.0, 53.0, 33.0, 40.0, 38.0, 31.0, 41.0, 23.0, 37.0, 20.0, 41.0, 29.0, 17.0, 31.0, 23.0, 44.0, 28.0, 19.0, 26.0, 41.0, 23.0, 25.0, 26.0, 18.0, 32.0, 42.0, 28.0, 46.0, 50.0, 32.0, 28.0, 28.0, 40.0, 19.0, 28.0, 28.0, 42.0, 54.0, 19.0, 26.0, 41.0, 17.0, 33.0, 24.0, 53.0, 31.0, 19.0, 52.0, 36.0, 73.0, 30.0, 30.0, 36.0, 28.0, 32.0, 24.0, 24.0, 24.0, 37.0, 25.0, 40.0, 25.0, 27.0, 30.0, 25.0, 32.0, 25.0, 25.0, 31.0, 28.0, 36.0, 23.0, 53.0, 25.0, 29.0, 43.0, 32.0, 50.0, 34.0, 34.0, 31.0, 32.0, 33.0, 41.0, 31.0, 39.0, 28.0, 46.0, 37.0, 21.0, 42.0, 33.0, 29.0, 29.0, 21.0, 40.0, 26.0, 48.0, 51.0, 34.0, 40.0, 21.0, 42.0, 58.0, 26.0, 45.0, 26.0, 33.0, 44.0, 26.0, 33.0, 27.0, 52.0, 49.0, 30.0, 35.0, 31.0, 30.0, 49.0, 61.0, 28.0, 54.0, 46.0, 25.0, 29.0, 51.0, 39.0, 37.0, 35.0, 28.0, 35.0, 20.0, 24.0, 35.0, 43.0, 26.0, 46.0, 29.0, 30.0, 22.0, 16.0, 40.0, 27.0, 16.0, 21.0, 27.0, 22.0, 29.0, 30.0, 22.0, 25.0, 34.0, 25.0, 22.0, 33.0, 29.0, 33.0, 31.0, 31.0, 24.0, 27.0, 32.0, 32.0, 26.0, 21.0, 38.0, 45.0, 29.0, 28.0, 40.0, 20.0, 37.0, 28.0, 55.0, 31.0, 26.0, 16.0, 42.0, 28.0, 29.0, 32.0, 16.0, 32.0, 33.0, 34.0, 27.0, 34.0, 18.0, 40.0, 20.0, 14.0, 44.0, 33.0, 25.0, 45.0, 26.0, 62.0, 24.0, 24.0, 19.0, 34.0, 42.0, 32.0, 26.0, 22.0, 31.0, 28.0, 49.0, 38.0, 33.0, 31.0, 25.0, 23.0, 27.0, 66.0, 37.0, 33.0, 43.0, 36.0, 49.0, 32.0, 43.0, 44.0, 48.0, 48.0, 30.0, 43.0, 34.0, 19.0, 32.0, 25.0, 44.0, 43.0, 18.0, 51.0, 21.0, 21.0, 28.0, 56.0, 35.0, 32.0, 17.0, 45.0, 22.0, 32.0, 46.0, 36.0, 18.0, 37.0, 36.0, 50.0, 28.0, 22.0, 28.0, 30.0, 27.0, 26.0, 29.0, 37.0, 36.0, 28.0, 21.0, 29.0, 18.0, 27.0, 27.0, 33.0, 27.0, 42.0, 26.0, 16.0, 26.0, 28.0, 38.0, 41.0, 26.0, 32.0, 29.0, 34.0, 48.0, 37.0, 28.0, 32.0, 17.0, 38.0, 28.0, 35.0, 24.0, 28.0, 43.0, 31.0, 13.0, 48.0, 35.0, 45.0, 36.0, 60.0, 26.0, 56.0, 33.0, 29.0, 18.0, 17.0, 35.0, 22.0, 41.0, 23.0, 27.0, 37.0, 20.0, 23.0, 33.0, 23.0, 22.0, 15.0, 36.0, 41.0, 38.0, 39.0, 34.0, 73.0, 25.0, 28.0, 37.0, 31.0, 32.0, 18.0, 37.0, 30.0, 38.0, 47.0, 41.0, 27.0, 26.0, 39.0, 38.0, 29.0, 38.0, 41.0, 24.0, 35.0, 23.0, 23.0, 42.0, 37.0, 28.0, 67.0, 43.0, 28.0, 26.0, 31.0, 32.0, 34.0, 32.0, 33.0, 35.0, 24.0, 30.0, 36.0, 26.0, 26.0, 34.0, 30.0, 29.0, 29.0, 43.0, 34.0, 24.0, 33.0, 33.0, 40.0, 25.0, 28.0, 15.0, 40.0, 48.0, 39.0, 44.0, 30.0, 29.0, 34.0, 39.0, 22.0, 39.0, 36.0, 20.0, 44.0, 43.0, 25.0, 33.0, 31.0, 40.0, 30.0, 26.0, 31.0, 28.0, 39.0, 31.0, 22.0, 20.0, 22.0, 33.0, 24.0, 39.0, 50.0, 24.0, 39.0, 33.0, 35.0, 52.0, 41.0, 25.0, 55.0, 30.0, 27.0, 34.0, 26.0, 45.0, 37.0, 51.0, 32.0, 40.0, 28.0, 31.0, 22.0, 41.0, 33.0, 47.0, 35.0, 31.0, 25.0, 25.0, 31.0, 33.0, 49.0, 37.0, 43.0, 40.0, 24.0, 34.0, 60.0, 21.0, 36.0, 34.0, 20.0, 35.0, 23.0, 37.0, 38.0, 18.0, 23.0, 34.0, 49.0, 24.0, 26.0, 55.0, 34.0, 43.0, 32.0, 18.0, 22.0, 34.0, 34.0, 28.0, 39.0, 38.0, 26.0, 43.0, 63.0, 22.0, 26.0, 26.0, 29.0, 38.0, 23.0, 32.0, 36.0, 40.0, 37.0, 24.0, 26.0, 26.0, 41.0, 42.0, 20.0, 32.0, 44.0, 37.0, 25.0, 18.0, 25.0, 31.0, 30.0, 50.0, 27.0, 18.0, 25.0, 24.0, 64.0, 17.0, 44.0, 33.0, 19.0, 56.0, 22.0, 25.0, 47.0, 24.0, 24.0, 64.0, 27.0, 33.0, 43.0, 26.0, 30.0, 27.0, 27.0, 53.0, 24.0, 48.0, 17.0, 57.0, 27.0, 25.0, 59.0, 28.0, 25.0, 30.0, 31.0, 28.0, 50.0, 24.0, 28.0, 39.0, 26.0, 21.0, 23.0, 35.0, 35.0, 33.0, 20.0, 34.0, 30.0, 28.0, 19.0, 27.0, 30.0, 25.0, 28.0, 40.0, 27.0, 32.0, 24.0, 18.0, 19.0, 16.0, 37.0, 36.0, 21.0, 56.0, 35.0, 36.0, 32.0, 31.0, 40.0, 34.0, 40.0, 29.0, 40.0, 31.0, 29.0, 29.0, 25.0, 30.0, 34.0, 17.0, 29.0, 42.0, 31.0, 36.0, 34.0, 21.0, 20.0, 30.0, 34.0, 39.0, 57.0, 30.0, 30.0, 26.0, 26.0, 32.0, 28.0, 43.0, 28.0, 35.0, 20.0, 32.0, 23.0, 43.0, 40.0, 29.0, 27.0, 28.0, 22.0, 27.0, 28.0, 39.0, 70.0, 31.0, 56.0, 39.0, 44.0, 50.0, 43.0, 24.0, 41.0, 21.0, 28.0, 41.0, 39.0, 25.0, 27.0, 27.0, 34.0, 29.0, 30.0, 32.0, 53.0, 13.0, 27.0, 33.0, 24.0, 25.0, 30.0, 35.0, 53.0, 33.0, 45.0, 32.0, 36.0, 27.0, 31.0, 21.0, 42.0, 32.0, 33.0, 40.0, 23.0, 39.0, 36.0, 26.0, 36.0, 19.0, 50.0, 34.0, 24.0, 31.0, 25.0, 38.0, 22.0, 39.0, 32.0, 33.0, 30.0, 29.0, 29.0, 26.0, 13.0, 32.0, 46.0, 29.0, 21.0, 26.0, 29.0, 17.0, 35.0, 17.0, 32.0, 33.0, 21.0, 30.0, 27.0, 28.0, 28.0, 31.0, 11.0, 36.0, 24.0, 25.0, 19.0, 21.0, 55.0, 36.0, 26.0, 43.0, 29.0, 37.0, 29.0, 26.0, 28.0, 29.0, 32.0, 45.0, 38.0, 39.0, 34.0, 26.0, 34.0, 48.0, 35.0, 22.0, 25.0, 23.0, 34.0, 17.0, 36.0, 27.0, 30.0, 21.0, 29.0, 49.0, 48.0, 28.0, 27.0, 22.0, 27.0, 21.0, 55.0, 40.0, 49.0, 33.0, 37.0, 41.0, 22.0, 33.0, 13.0, 25.0, 24.0, 40.0, 22.0, 41.0, 35.0, 39.0, 19.0, 25.0, 43.0, 34.0, 29.0, 32.0, 41.0, 47.0, 31.0, 49.0, 34.0, 38.0, 44.0, 22.0]

##### Best Scores: [38.0, 23.0, 12.0, 26.0, 18.0, 37.0, 47.0, 32.0, 40.0, 21.0, 34.0, 19.0, 21.0, 23.0, 17.0, 19.0, 19.0, 41.0, 37.0, 27.0, 38.0, 29.0, 19.0, 16.0, 45.0, 36.0, 36.0, 28.0, 48.0, 34.0, 34.0, 30.0, 51.0, 12.0, 26.0, 26.0, 31.0, 28.0, 16.0, 48.0, 18.0, 24.0, 22.0, 43.0, 27.0, 37.0, 19.0, 26.0, 26.0, 27.0, 29.0, 27.0, 23.0, 20.0, 36.0, 20.0, 12.0, 20.0, 20.0, 21.0, 26.0, 27.0, 29.0, 38.0, 39.0, 26.0, 40.0, 23.0, 31.0, 39.0, 24.0, 24.0, 19.0, 26.0, 35.0, 23.0, 24.0, 41.0, 12.0, 21.0, 25.0, 29.0, 20.0, 27.0, 18.0, 26.0, 40.0, 11.0, 37.0, 41.0, 22.0, 28.0, 26.0, 17.0, 22.0, 21.0, 29.0, 26.0, 28.0, 27.0, 16.0, 26.0, 45.0, 28.0, 30.0, 31.0, 25.0, 37.0, 20.0, 33.0, 12.0, 33.0, 23.0, 15.0, 24.0, 17.0, 35.0, 28.0, 19.0, 20.0, 37.0, 16.0, 20.0, 17.0, 15.0, 19.0, 36.0, 22.0, 42.0, 40.0, 24.0, 14.0, 28.0, 28.0, 16.0, 21.0, 21.0, 35.0, 42.0, 19.0, 22.0, 36.0, 15.0, 27.0, 23.0, 41.0, 28.0, 13.0, 47.0, 33.0, 64.0, 22.0, 27.0, 34.0, 17.0, 27.0, 17.0, 17.0, 23.0, 34.0, 22.0, 26.0, 18.0, 18.0, 23.0, 20.0, 21.0, 23.0, 21.0, 26.0, 26.0, 29.0, 17.0, 43.0, 18.0, 27.0, 39.0, 26.0, 41.0, 33.0, 26.0, 22.0, 24.0, 21.0, 37.0, 15.0, 27.0, 20.0, 38.0, 27.0, 16.0, 42.0, 26.0, 25.0, 21.0, 19.0, 35.0, 24.0, 41.0, 48.0, 28.0, 39.0, 18.0, 38.0, 55.0, 23.0, 38.0, 26.0, 30.0, 35.0, 20.0, 27.0, 26.0, 47.0, 42.0, 29.0, 35.0, 24.0, 25.0, 42.0, 52.0, 23.0, 50.0, 41.0, 13.0, 24.0, 45.0, 30.0, 32.0, 24.0, 18.0, 27.0, 19.0, 24.0, 29.0, 40.0, 22.0, 44.0, 28.0, 28.0, 22.0, 16.0, 37.0, 21.0, 15.0, 21.0, 26.0, 22.0, 18.0, 25.0, 18.0, 14.0, 31.0, 17.0, 20.0, 22.0, 21.0, 26.0, 30.0, 25.0, 14.0, 25.0, 25.0, 25.0, 19.0, 19.0, 32.0, 34.0, 20.0, 20.0, 35.0, 19.0, 29.0, 27.0, 52.0, 19.0, 25.0, 10.0, 34.0, 21.0, 25.0, 28.0, 16.0, 31.0, 22.0, 24.0, 23.0, 16.0, 17.0, 37.0, 15.0, 13.0, 38.0, 29.0, 22.0, 41.0, 26.0, 60.0, 23.0, 24.0, 17.0, 27.0, 38.0, 30.0, 23.0, 17.0, 28.0, 25.0, 44.0, 30.0, 23.0, 24.0, 24.0, 18.0, 24.0, 61.0, 29.0, 23.0, 27.0, 30.0, 46.0, 24.0, 34.0, 43.0, 48.0, 45.0, 26.0, 41.0, 30.0, 13.0, 30.0, 20.0, 40.0, 37.0, 17.0, 48.0, 21.0, 14.0, 20.0, 50.0, 30.0, 28.0, 17.0, 35.0, 22.0, 24.0, 40.0, 27.0, 17.0, 37.0, 27.0, 46.0, 26.0, 18.0, 18.0, 26.0, 25.0, 17.0, 21.0, 32.0, 23.0, 23.0, 20.0, 22.0, 17.0, 21.0, 25.0, 23.0, 15.0, 42.0, 22.0, 14.0, 14.0, 21.0, 31.0, 35.0, 26.0, 28.0, 23.0, 28.0, 44.0, 29.0, 20.0, 24.0, 9.0, 21.0, 25.0, 26.0, 19.0, 24.0, 41.0, 23.0, 10.0, 44.0, 32.0, 43.0, 33.0, 56.0, 17.0, 51.0, 19.0, 21.0, 18.0, 16.0, 30.0, 21.0, 41.0, 17.0, 18.0, 31.0, 16.0, 17.0, 19.0, 21.0, 19.0, 14.0, 20.0, 33.0, 29.0, 33.0, 26.0, 63.0, 22.0, 20.0, 33.0, 28.0, 28.0, 17.0, 27.0, 21.0, 23.0, 33.0, 35.0, 25.0, 24.0, 34.0, 24.0, 21.0, 24.0, 38.0, 23.0, 28.0, 18.0, 23.0, 42.0, 36.0, 25.0, 66.0, 39.0, 23.0, 22.0, 19.0, 30.0, 29.0, 24.0, 26.0, 30.0, 21.0, 28.0, 32.0, 18.0, 18.0, 28.0, 18.0, 22.0, 25.0, 34.0, 24.0, 15.0, 32.0, 25.0, 39.0, 21.0, 24.0, 14.0, 34.0, 39.0, 24.0, 44.0, 18.0, 25.0, 30.0, 35.0, 22.0, 37.0, 30.0, 14.0, 43.0, 37.0, 18.0, 25.0, 30.0, 39.0, 22.0, 25.0, 20.0, 23.0, 36.0, 16.0, 16.0, 17.0, 13.0, 31.0, 24.0, 30.0, 41.0, 21.0, 35.0, 24.0, 26.0, 46.0, 35.0, 21.0, 51.0, 27.0, 25.0, 32.0, 22.0, 32.0, 25.0, 43.0, 28.0, 40.0, 24.0, 22.0, 15.0, 34.0, 30.0, 43.0, 24.0, 21.0, 18.0, 19.0, 31.0, 27.0, 34.0, 33.0, 37.0, 29.0, 20.0, 23.0, 56.0, 21.0, 26.0, 32.0, 16.0, 19.0, 17.0, 29.0, 28.0, 14.0, 22.0, 27.0, 44.0, 18.0, 18.0, 46.0, 28.0, 37.0, 27.0, 15.0, 21.0, 27.0, 34.0, 26.0, 26.0, 34.0, 17.0, 31.0, 57.0, 19.0, 22.0, 23.0, 24.0, 30.0, 18.0, 25.0, 26.0, 40.0, 28.0, 17.0, 23.0, 20.0, 39.0, 39.0, 15.0, 28.0, 34.0, 33.0, 18.0, 13.0, 24.0, 25.0, 30.0, 44.0, 18.0, 13.0, 19.0, 21.0, 55.0, 13.0, 43.0, 31.0, 19.0, 54.0, 17.0, 21.0, 38.0, 13.0, 22.0, 60.0, 23.0, 29.0, 36.0, 16.0, 25.0, 24.0, 21.0, 49.0, 19.0, 38.0, 13.0, 47.0, 25.0, 20.0, 54.0, 22.0, 19.0, 22.0, 25.0, 21.0, 44.0, 21.0, 22.0, 38.0, 18.0, 19.0, 19.0, 31.0, 27.0, 27.0, 15.0, 31.0, 21.0, 18.0, 18.0, 22.0, 26.0, 21.0, 28.0, 33.0, 24.0, 25.0, 23.0, 17.0, 18.0, 12.0, 25.0, 32.0, 18.0, 52.0, 21.0, 33.0, 21.0, 29.0, 33.0, 22.0, 37.0, 28.0, 31.0, 25.0, 26.0, 18.0, 19.0, 25.0, 31.0, 17.0, 26.0, 35.0, 27.0, 24.0, 25.0, 20.0, 13.0, 25.0, 23.0, 35.0, 55.0, 26.0, 28.0, 19.0, 22.0, 20.0, 26.0, 31.0, 24.0, 23.0, 16.0, 29.0, 9.0, 31.0, 30.0, 14.0, 19.0, 28.0, 18.0, 21.0, 23.0, 29.0, 65.0, 21.0, 50.0, 30.0, 34.0, 46.0, 32.0, 22.0, 40.0, 21.0, 24.0, 35.0, 37.0, 18.0, 13.0, 20.0, 26.0, 23.0, 30.0, 30.0, 53.0, 13.0, 27.0, 19.0, 24.0, 19.0, 25.0, 33.0, 37.0, 20.0, 41.0, 28.0, 28.0, 25.0, 29.0, 19.0, 41.0, 28.0, 30.0, 28.0, 20.0, 32.0, 25.0, 22.0, 33.0, 18.0, 43.0, 31.0, 23.0, 29.0, 24.0, 30.0, 16.0, 29.0, 19.0, 30.0, 23.0, 19.0, 19.0, 16.0, 12.0, 21.0, 44.0, 27.0, 16.0, 21.0, 24.0, 16.0, 28.0, 15.0, 23.0, 26.0, 20.0, 17.0, 21.0, 21.0, 23.0, 29.0, 10.0, 27.0, 18.0, 23.0, 18.0, 18.0, 49.0, 32.0, 19.0, 43.0, 21.0, 28.0, 29.0, 22.0, 19.0, 19.0, 31.0, 36.0, 34.0, 35.0, 22.0, 24.0, 27.0, 42.0, 23.0, 21.0, 19.0, 19.0, 28.0, 14.0, 24.0, 24.0, 24.0, 17.0, 23.0, 39.0, 39.0, 27.0, 20.0, 22.0, 19.0, 18.0, 47.0, 25.0, 43.0, 33.0, 34.0, 32.0, 19.0, 23.0, 11.0, 18.0, 20.0, 33.0, 18.0, 35.0, 30.0, 38.0, 14.0, 22.0, 32.0, 24.0, 23.0, 24.0, 35.0, 47.0, 23.0, 44.0, 27.0, 35.0, 32.0, 19.0]

##### Ground Truths: [33, 20, 12, 24, 15, 35, 46, 23, 38, 17, 33, 16, 19, 23, 16, 15, 16, 39, 37, 25, 35, 26, 17, 13, 41, 33, 34, 27, 47, 33, 32, 29, 51, 10, 23, 19, 30, 24, 13, 41, 17, 19, 19, 39, 24, 35, 17, 22, 22, 23, 28, 26, 22, 16, 32, 17, 12, 16, 16, 16, 20, 23, 28, 38, 36, 23, 38, 16, 23, 34, 20, 22, 18, 21, 30, 21, 21, 34, 12, 16, 19, 25, 18, 24, 15, 21, 40, 11, 36, 41, 21, 28, 24, 15, 19, 20, 23, 22, 22, 24, 12, 26, 45, 24, 22, 31, 24, 35, 15, 29, 11, 26, 20, 14, 22, 13, 32, 25, 15, 18, 35, 11, 18, 14, 15, 15, 32, 14, 40, 35, 19, 14, 26, 26, 13, 15, 17, 34, 40, 14, 21, 32, 15, 25, 18, 41, 23, 11, 46, 33, 64, 20, 24, 33, 17, 22, 14, 15, 21, 31, 18, 24, 14, 15, 22, 17, 17, 18, 19, 24, 22, 25, 14, 39, 14, 26, 39, 22, 41, 33, 25, 22, 23, 18, 34, 15, 26, 16, 35, 20, 12, 39, 25, 21, 16, 15, 31, 18, 40, 47, 26, 36, 16, 38, 54, 18, 37, 21, 25, 33, 19, 26, 24, 43, 41, 25, 34, 23, 23, 40, 52, 20, 47, 41, 13, 20, 42, 29, 31, 23, 16, 26, 17, 20, 28, 38, 20, 42, 24, 25, 19, 15, 35, 17, 13, 18, 26, 19, 16, 25, 17, 12, 27, 17, 15, 16, 17, 25, 24, 21, 13, 24, 25, 23, 18, 15, 22, 33, 15, 18, 31, 16, 27, 22, 50, 16, 20, 10, 29, 18, 23, 27, 13, 29, 19, 20, 21, 16, 16, 33, 12, 11, 36, 27, 19, 38, 25, 60, 21, 21, 15, 22, 37, 27, 18, 15, 22, 20, 39, 26, 21, 22, 20, 17, 21, 60, 23, 19, 21, 28, 46, 23, 33, 41, 47, 43, 24, 38, 27, 10, 28, 19, 38, 35, 13, 45, 16, 11, 18, 48, 27, 22, 16, 32, 19, 21, 38, 26, 17, 37, 23, 46, 23, 14, 14, 25, 23, 16, 17, 29, 20, 21, 16, 20, 13, 17, 24, 21, 12, 37, 20, 14, 12, 19, 25, 34, 21, 24, 19, 28, 41, 26, 20, 22, 8, 21, 21, 26, 19, 17, 41, 21, 9, 36, 29, 42, 30, 55, 16, 50, 16, 18, 16, 13, 28, 17, 37, 17, 16, 25, 14, 15, 19, 18, 16, 12, 18, 30, 24, 29, 26, 59, 15, 20, 27, 24, 25, 16, 24, 18, 23, 28, 35, 23, 22, 34, 23, 16, 20, 36, 20, 26, 17, 22, 38, 31, 24, 64, 37, 21, 18, 19, 29, 29, 21, 25, 25, 18, 26, 32, 14, 17, 25, 16, 18, 22, 30, 19, 15, 27, 21, 39, 15, 19, 13, 33, 37, 19, 43, 14, 23, 28, 32, 20, 31, 26, 13, 41, 31, 14, 22, 26, 34, 20, 21, 18, 18, 33, 15, 13, 13, 12, 26, 23, 25, 38, 18, 32, 22, 23, 45, 31, 18, 49, 24, 21, 25, 20, 31, 21, 40, 24, 38, 20, 19, 13, 31, 25, 39, 21, 17, 15, 17, 27, 25, 34, 27, 34, 21, 20, 20, 54, 17, 25, 24, 13, 17, 17, 28, 26, 13, 21, 22, 39, 14, 16, 45, 23, 35, 22, 13, 18, 26, 32, 20, 22, 33, 14, 30, 57, 17, 21, 22, 21, 30, 15, 23, 24, 40, 25, 15, 19, 19, 35, 36, 12, 23, 34, 30, 17, 11, 20, 23, 26, 44, 14, 12, 14, 18, 52, 11, 37, 29, 13, 51, 15, 19, 34, 12, 22, 58, 20, 27, 33, 16, 22, 23, 18, 45, 17, 37, 12, 44, 21, 18, 50, 18, 17, 16, 25, 14, 41, 18, 18, 37, 14, 16, 14, 28, 23, 22, 13, 27, 18, 14, 16, 18, 24, 19, 24, 30, 22, 24, 17, 13, 16, 12, 24, 30, 16, 51, 17, 31, 16, 22, 27, 18, 35, 28, 31, 21, 24, 14, 15, 22, 25, 14, 24, 32, 27, 22, 25, 16, 10, 25, 22, 32, 55, 25, 22, 16, 18, 19, 26, 27, 18, 21, 14, 24, 8, 30, 26, 12, 17, 24, 15, 19, 18, 28, 59, 21, 50, 24, 31, 41, 28, 20, 38, 18, 22, 34, 32, 18, 12, 16, 24, 19, 27, 25, 53, 13, 27, 19, 22, 16, 25, 30, 36, 19, 38, 25, 20, 22, 25, 15, 36, 26, 25, 27, 15, 28, 24, 20, 30, 18, 39, 30, 20, 26, 20, 28, 14, 23, 16, 27, 19, 18, 18, 15, 11, 18, 41, 23, 14, 19, 21, 10, 24, 14, 19, 22, 14, 17, 20, 15, 18, 22, 9, 27, 16, 21, 14, 15, 49, 30, 14, 42, 17, 28, 27, 17, 18, 19, 27, 31, 28, 32, 21, 21, 25, 40, 21, 19, 17, 15, 20, 14, 20, 22, 20, 13, 19, 38, 38, 23, 20, 20, 15, 14, 42, 23, 41, 30, 28, 28, 13, 20, 11, 17, 17, 23, 16, 33, 27, 37, 13, 17, 28, 23, 15, 23, 34, 42, 20, 40, 24, 29, 30, 15]
##### Test Results: RMSE - 3.2845282351641036, MAE: 2.727590221187427, Num Gt: 96/859

