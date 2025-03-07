import itertools
import numpy as np
import networkx as nx
import copy
import math
import heapq
import random

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


def priority_v3(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    if not weights or len(weights) != max_node or not all(len(row) == max_node for row in weights):
        weights = [[0.0] * max_node for _ in range(max_node)]

    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            refined_weights[i][j] = weights[i][j]  # Start with initial weights

            # Degree similarity
            degree_diff = abs(degrees1[i] - degrees2[j])
            degree_similarity = 1.0 / (degree_diff + 1)
            refined_weights[i][j] += degree_similarity

            # Neighbor similarity (using sets for efficiency)
            neighbors1 = {k for k in range(n1) if graph1[i][k]}
            neighbors2 = {l for l in range(n2) if graph2[j][l]}

            common_neighbors = len(neighbors1.intersection(neighbors2))  # More efficient
            refined_weights[i][j] += common_neighbors


            # Consider 2-hop neighbors (this is the main improvement)
            two_hop_neighbors1 = set()
            for neighbor in neighbors1:
                two_hop_neighbors1.update(k for k in range(n1) if graph1[neighbor][k] and k != i)

            two_hop_neighbors2 = set()
            for neighbor in neighbors2:
                two_hop_neighbors2.update(l for l in range(n2) if graph2[neighbor][l] and l != j)


            common_two_hop_neighbors = len(two_hop_neighbors1.intersection(two_hop_neighbors2))
            refined_weights[i][j] += (0.5* common_two_hop_neighbors) # Reduced weight for 2-hop



    # Normalize rows (important for probabilities)
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum
        else: # Handle cases where row sum is zero (avoid division by zero)
            for j in range(n2):
                refined_weights[i][j] = 1.0 / n2 if n2 > 0 else 0.0  # Uniform distribution if n2 > 0

    return refined_weights

def priority_v41(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Refines probabilities based on neighbor similarity."""
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
                    neighbor_similarity += (weights[k][l] if ((k < len(weights)) and (l < len(weights[k]))) else 0)
        refined_weights[i][j] = neighbor_similarity
  return refined_weights


def priority_v4(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Normalizes the refined weights from priority_v1."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = priority_v41(graph1, graph2, weights)
  
  # Normalize the weights
  for i in range(n1):
    row_sum = sum(refined_weights[i][:n2])  # Sum only up to n2
    if row_sum > 0:
      for j in range(n2):
        refined_weights[i][j] /= row_sum

  return refined_weights


def priority_v5(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Normalizes the weights to probabilities.
  """
  weights = priority_v1(graph1, graph2, weights) # Or use v0 if preferred

  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)

  total_weight = sum(sum(row) for row in weights[:n1])  # Sum only up to n1

  if total_weight == 0:  # Handle the case where all weights are zero
      return [[1.0/(n1*n2) if i < n1 and j < n2 else 0.0 for j in range(max_node)] for i in range(max_node)]

  for i in range(n1):
      for j in range(n2):
          weights[i][j] /= total_weight
  return weights

def priority_v6(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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


def priority_v7(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1` using numpy for efficiency."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    graph1_np = np.array(graph1)
    graph2_np = np.array(graph2)

    degrees1 = graph1_np.sum(axis=1)
    degrees2 = graph2_np.sum(axis=1)

    weights = np.zeros((max_node, max_node))

    for i in range(n1):
        for j in range(n2):
            degree_diff = abs(degrees1[i] - degrees2[j])

            neighbor_indices1 = np.where(graph1_np[i])[0]
            neighbor_indices2 = np.where(graph2_np[j])[0]

            neighbor_degrees1 = degrees1[neighbor_indices1]
            neighbor_degrees2 = degrees2[neighbor_indices2]

            neighbor_diff = 0
            for deg1 in neighbor_degrees1:
                for deg2 in neighbor_degrees2:
                    neighbor_diff += abs(deg1 - deg2)

            denominator = 1.0 + degree_diff + neighbor_diff
            weights[i, j] = 1.0 / denominator if denominator > 0 else 1.0

    total_weight = weights.sum()
    if total_weight > 0:
        weights /= total_weight

    return weights.tolist()

def priority_v8(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            score = 0.0

            # Degree Difference (penalize larger differences)
            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
            score -= degree_diff * 2  # Increased penalty for degree differences

            # Common Neighbors (reward higher similarity)
            neighbors_i = set(k for k in range(n1) if graph1[i][k])
            neighbors_j = set(k for k in range(n2) if graph2[j][k])
            common_neighbors = len(neighbors_i.intersection(neighbors_j))
            score += common_neighbors * 2 # Increased reward for common neighbors


            # Incorporate initial weights (if available)
            if i < len(weights) and j < len(weights[0]):
                score += weights[i][j] * 5 # Give more weight to initial provided weights

            refined_weights[i][j] = max(0,score) # Ensure scores are non-negative

    # Normalize rows to probabilities (only if row sum is positive)
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum
        else: #handle cases where row sum is 0, distribute uniformly
            for j in range(n2):
                refined_weights[i][j] = 1/n2 if n2>0 else 0


    return refined_weights


def priority_v9(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
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
            neighbors1 = [k for k in range(n1) if graph1[i][k]]
            neighbors2 = [k for k in range(n2) if graph2[j][k]]

            # Use set operations for efficient neighbor comparison
            common_neighbors = len(set(neighbors1) & set(neighbors2))
            neighbor_diff = (len(neighbors1) + len(neighbors2)) - (2 * common_neighbors)  # This is now a set-based difference

            denominator = (1.0 + degree_diff + neighbor_diff)
            if denominator == 0:
                weights[i][j] = 1.0
            else:
                weights[i][j] = (1.0 / denominator)

    return weights


def priority_v10(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0`.  Normalizes based on the maximum score
  and uses a similarity measure that accounts for both common and different neighbors."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [([0.0] * max_node) for _ in range(max_node)]
  max_score = 0  # Keep track of the maximum score

  for i in range(n1):
    for j in range(n2):
      score = 0.0
      neighbors_i = set((k for k in range(n1) if graph1[i][k]))
      neighbors_j = set((k for k in range(n2) if graph2[j][k]))

      common_neighbors = len(neighbors_i.intersection(neighbors_j))
      all_neighbors = len(neighbors_i.union(neighbors_j))
      
      if all_neighbors > 0:  # Avoid division by zero
          score = common_neighbors / all_neighbors  # Similarity based on Jaccard index
      else:
          score = 0  # If both nodes have no neighbors, their similarity is 0

      if ((i < len(weights)) and (j < len(weights[0]))):
        score += weights[i][j]

      refined_weights[i][j] = score
      max_score = max(max_score, score)  # Update max_score


  # Normalize by the maximum score (if it's not zero)
  if max_score > 0:
      for i in range(n1):
          for j in range(n2):
              refined_weights[i][j] /= max_score
              
  return refined_weights

def priority_v11(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

def priority_v12(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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


def priority_v13(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

def priority_v14(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [([0.0] * max_node) for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            score = 0.0
            degree_diff = abs((sum(graph1[i]) - sum(graph2[j])))
            score -= degree_diff

            neighbors_i = [k for k in range(n1) if graph1[i][k]]
            neighbors_j = [k for k in range(n2) if graph2[j][k]]

            common_neighbors = 0
            for ni in neighbors_i:
                for nj in neighbors_j:
                    if weights and ni < len(weights) and nj < len(weights[0]):  # Check bounds for weights
                        common_neighbors += weights[ni][nj]  # Use weights to measure neighbor similarity

            score += common_neighbors


            if weights and i < len(weights) and j < len(weights[0]):
                score += weights[i][j]

            refined_weights[i][j] = max(0, score) # Ensure scores are non-negative

        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum
        else:
            for j in range(n2): # If row_sum is 0, distribute probability evenly.
                refined_weights[i][j] = 1.0 / n2 if n2 > 0 else 0  # Avoid division by zero

    return refined_weights
# Input Scores: [82.0, 0.0, 17.0, 0.0, 10.0, 32.0, 94.0, 34.0, 74.0, 17.0, 18.0, 25.0, 25.0, 12.0, 8.0, 88.0, 28.0, 47.0, 37.0, 22.0, 117.0, 10.0, 18.0, 114.0, 36.0, 14.0, 117.0, 27.0, 17.0, 424.0, 70.0, 27.0, 102.0, 47.0, 33.0, 8.0, 174.0, 111.0, 17.0, 12.0, 14.0, 38.0, 1.0, 27.0, 73.0, 55.0, 121.0, 41.0, 53.0, 9.0, 30.0, 27.0, 198.0, 31.0, 19.0, 121.0, 31.0, 8.0, 8.0, 50.0, 23.0, 8.0, 27.0, 102.0, 17.0, 31.0, 44.0, 9.0, 118.0, 28.0, 121.0, 59.0, 38.0, 229.0, 73.0, 64.0, 33.0, 36.0, 9.0, 33.0, 58.0, 447.0, 20.0, 77.0, 69.0, 56.0, 28.0, 19.0, 10.0, 104.0, 25.0, 63.0, 111.0, 37.0, 0.0, 8.0, 31.0, 10.0, 13.0, 19.0, 17.0, 54.0, 44.0, 108.0, 61.0, 172.0, 0.0, 38.0, 36.0, 0.0, 8.0, 8.0, 55.0, 692.0, 14.0, 27.0, 72.0, 23.0, 50.0, 52.0, 0.0, 13.0, 52.0, 0.0, 123.0, 12.0, 0.0, 8.0, 19.0, 37.0, 17.0, 17.0, 66.0, 175.0, 53.0, 30.0, 705.0, 16.0, 30.0, 399.0, 27.0, 162.0, 17.0, 50.0, 87.0, 77.0, 8.0, 0.0, 61.0, 27.0, 33.0, 17.0, 0.0, 123.0, 8.0, 35.0, 19.0, 41.0, 329.0, 0.0, 27.0, 19.0, 0.0, 19.0, 9.0, 50.0, 23.0, 78.0, 0.0, 6.0, 33.0, 73.0, 45.0, 56.0, 34.0, 106.0, 33.0, 424.0, 67.0, 17.0, 25.0, 14.0, 94.0, 19.0, 23.0, 108.0, 30.0, 113.0, 135.0, 12.0, 8.0, 9.0, 88.0, 36.0, 61.0, 24.0, 68.0, 37.0, 124.0, 112.0, 17.0, 27.0, 44.0, 58.0, 8.0, 109.0, 43.0, 42.0, 19.0, 12.0, 24.0, 82.0, 8.0, 5.0, 86.0, 12.0, 18.0, 23.0, 19.0, 57.0, 39.0, 26.0, 27.0, 12.0, 69.0, 126.0, 125.0, 57.0, 65.0, 56.0, 92.0, 50.0, 226.0, 20.0, 21.0, 84.0, 8.0, 56.0, 45.0, 55.0, 310.0, 9.0, 17.0, 37.0, 22.0, 42.0, 31.0, 112.0, 19.0, 81.0, 21.0, 49.0, 0.0, 174.0, 31.0, 15.0, 24.0, 492.0, 60.0, 135.0, 22.0, 120.0, 196.0, 0.0, 50.0, 25.0, 66.0, 42.0, 0.0, 21.0, 38.0, 9.0, 22.0, 27.0, 13.0, 500.0, 24.0, 121.0, 567.0, 38.0, 9.0, 55.0, 10.0, 182.0, 68.0, 32.0, 124.0, 23.0, 8.0, 70.0, 173.0, 27.0, 70.0, 9.0, 71.0, 61.0, 55.0, 145.0, 49.0, 37.0, 9.0, 8.0, 173.0, 27.0, 63.0, 0.0, 21.0, 47.0, 600.0, 19.0, 16.0, 60.0, 88.0, 82.0, 0.0, 0.0, 116.0, 85.0, 143.0, 102.0, 23.0, 21.0, 8.0, 23.0, 75.0, 36.0, 32.0, 35.0, 30.0, 42.0, 8.0, 19.0, 8.0, 27.0, 0.0, 41.0, 10.0, 606.0, 26.0, 17.0, 58.0, 8.0, 20.0, 33.0, 20.0, 555.0, 32.0, 33.0, 19.0, 12.0, 23.0, 35.0, 21.0, 61.0, 94.0, 28.0, 118.0, 50.0, 19.0, 17.0, 0.0, 14.0, 17.0, 0.0, 17.0, 19.0, 9.0, 17.0, 26.0, 159.0, 44.0, 83.0, 103.0, 8.0, 50.0, 77.0, 17.0, 84.0, 53.0, 0.0, 14.0, 19.0, 15.0, 46.0, 12.0, 117.0, 0.0, 8.0, 36.0, 11.0, 183.0, 20.0, 8.0, 283.0, 164.0, 0.0, 61.0, 49.0, 70.0, 29.0, 17.0, 23.0, 11.0, 92.0, 160.0, 33.0, 23.0, 111.0, 9.0, 23.0, 21.0, 113.0, 60.0, 98.0, 0.0, 31.0, 429.0, 194.0, 117.0, 46.0, 5.0, 104.0, 20.0, 108.0, 12.0, 182.0, 8.0, 17.0, 27.0, 17.0, 69.0, 20.0, 27.0, 33.0, 34.0, 17.0, 8.0, 0.0, 42.0, 27.0, 139.0, 19.0, 79.0, 0.0, 0.0, 74.0, 50.0, 416.0, 289.0, 12.0, 8.0, 32.0, 38.0, 25.0, 39.0, 8.0, 101.0, 17.0, 37.0, 51.0, 25.0, 45.0, 162.0, 8.0, 93.0, 22.0, 154.0, 32.0, 113.0, 410.0, 9.0, 22.0, 0.0, 27.0, 30.0, 121.0, 27.0, 17.0, 32.0, 20.0, 41.0, 135.0, 7.0, 366.0, 12.0, 0.0, 8.0, 97.0, 17.0, 14.0, 116.0, 492.0, 182.0, 43.0, 204.0, 17.0, 19.0, 78.0, 33.0, 79.0, 9.0, 64.0, 20.0, 183.0, 14.0, 17.0, 38.0, 571.0, 8.0, 135.0, 27.0, 21.0, 144.0, 41.0, 79.0, 19.0, 31.0, 554.0, 26.0, 72.0, 113.0, 8.0, 8.0, 200.0, 0.0, 23.0, 168.0, 0.0, 104.0, 46.0, 92.0, 19.0, 70.0, 49.0, 163.0, 8.0, 0.0, 8.0, 34.0, 70.0, 224.0, 24.0, 34.0, 15.0, 629.0, 105.0, 52.0, 70.0, 17.0, 19.0, 100.0, 0.0, 77.0, 0.0, 446.0, 17.0, 9.0, 107.0, 25.0, 108.0, 47.0, 27.0, 19.0, 20.0, 114.0, 38.0, 28.0, 0.0, 19.0, 107.0, 36.0, 17.0, 50.0, 26.0, 33.0, 40.0, 8.0, 29.0, 8.0, 36.0, 18.0, 40.0, 29.0, 47.0, 84.0, 19.0, 24.0, 4.0, 75.0, 39.0, 30.0, 37.0, 52.0, 34.0, 118.0, 182.0, 32.0, 17.0, 9.0, 10.0, 143.0, 55.0, 22.0, 19.0, 69.0, 112.0, 103.0, 30.0, 96.0, 8.0, 8.0, 25.0, 8.0, 43.0, 8.0, 23.0, 135.0, 182.0, 56.0, 50.0, 105.0, 78.0, 27.0, 19.0, 42.0, 64.0, 0.0, 36.0, 41.0, 60.0, 540.0, 40.0, 31.0, 58.0, 70.0, 186.0, 36.0, 43.0, 0.0, 23.0, 113.0, 21.0, 48.0, 68.0, 9.0, 34.0, 27.0, 38.0, 21.0, 88.0, 25.0, 39.0, 23.0, 12.0, 86.0, 30.0, 110.0, 39.0, 0.0, 9.0, 216.0, 23.0, 24.0, 59.0, 156.0, 19.0, 43.0, 289.0, 17.0, 156.0, 14.0, 8.0, 45.0, 65.0, 46.0, 27.0, 77.0, 67.0, 27.0, 51.0, 0.0, 11.0, 75.0, 24.0, 9.0, 8.0, 182.0, 41.0, 18.0, 122.0, 33.0, 124.0, 8.0, 521.0, 1488.0, 17.0, 129.0, 26.0, 22.0, 20.0, 19.0, 18.0, 19.0, 71.0, 182.0, 4.0, 329.0, 20.0, 16.0, 45.0, 28.0, 41.0, 20.0, 19.0, 14.0, 0.0, 0.0, 346.0, 101.0, 26.0, 66.0, 113.0, 13.0, 84.0, 14.0, 23.0, 27.0, 8.0, 45.0, 181.0, 26.0, 1006.0, 121.0, 17.0, 27.0, 77.0, 22.0, 143.0, 12.0, 31.0, 51.0, 14.0, 31.0, 58.0, 17.0, 57.0, 24.0, 0.0, 18.0, 27.0, 18.0, 32.0, 27.0, 8.0, 35.0, 78.0, 30.0, 0.0, 20.0, 50.0, 31.0, 9.0, 329.0, 100.0, 108.0, 881.0, 9.0, 54.0, 151.0, 45.0, 72.0, 39.0, 11.0, 57.0, 112.0, 105.0, 26.0, 13.0, 38.0, 55.0, 17.0, 77.0, 25.0, 18.0, 17.0, 77.0, 69.0, 54.0, 26.0, 61.0, 82.0, 125.0, 21.0, 124.0, 8.0, 17.0, 13.0, 20.0, 47.0, 36.0, 9.0, 43.0, 10.0, 18.0, 55.0, 0.0, 100.0, 239.0, 290.0, 41.0, 15.0, 27.0, 116.0, 17.0, 19.0, 8.0, 8.0, 14.0, 675.0, 0.0, 0.0, 0.0, 215.0, 71.0, 88.0, 151.0, 600.0, 94.0, 8.0, 9.0, 120.0, 52.0, 36.0, 37.0, 36.0, 61.0, 27.0, 155.0, 18.0, 39.0, 49.0, 17.0, 9.0, 77.0, 0.0, 41.0, 27.0, 406.0, 48.0, 127.0, 30.0, 9.0, 45.0, 225.0, 134.0, 35.0, 21.0, 56.0, 0.0, 8.0, 8.0, 9.0, 12.0, 45.0, 23.0, 10.0, 94.0, 77.0, 69.0, 31.0, 0.0, 33.0, 99.0, 689.0, 9.0, 338.0, 9.0, 23.0, 58.0, 19.0, 33.0, 110.0, 0.0, 35.0, 0.0, 17.0, 50.0, 22.0, 14.0, 72.0, 27.0, 0.0, 9.0, 18.0, 45.0, 17.0, 81.0, 8.0, 45.0, 21.0, 18.0, 8.0, 26.0, 130.0, 17.0, 102.0, 0.0, 8.0, 17.0, 216.0, 329.0, 0.0, 68.0, 93.0, 78.0, 0.0, 151.0, 125.0, 226.0, 48.0, 58.0, 49.0, 86.0, 303.0, 52.0, 46.0, 11.0, 42.0, 0.0, 906.0, 52.0, 843.0, 0.0, 0.0, 34.0, 69.0, 67.0, 17.0, 7.0, 91.0, 19.0, 17.0, 17.0, 8.0, 9.0, 19.0, 91.0, 456.0, 54.0, 36.0, 42.0, 116.0, 9.0, 319.0, 93.0, 101.0, 15.0, 36.0, 500.0, 429.0, 68.0, 0.0, 19.0, 9.0, 27.0, 33.0, 9.0]

##### Best Scores: [82.0, 0.0, 17.0, 0.0, 10.0, 32.0, 94.0, 34.0, 62.0, 17.0, 18.0, 25.0, 25.0, 12.0, 8.0, 80.0, 24.0, 39.0, 37.0, 22.0, 117.0, 10.0, 18.0, 106.0, 36.0, 14.0, 117.0, 27.0, 17.0, 424.0, 70.0, 27.0, 94.0, 47.0, 33.0, 8.0, 174.0, 111.0, 17.0, 12.0, 14.0, 38.0, 1.0, 27.0, 73.0, 51.0, 103.0, 41.0, 45.0, 9.0, 22.0, 27.0, 186.0, 31.0, 19.0, 121.0, 31.0, 8.0, 8.0, 50.0, 23.0, 8.0, 27.0, 88.0, 17.0, 31.0, 44.0, 9.0, 118.0, 28.0, 121.0, 51.0, 38.0, 229.0, 73.0, 64.0, 33.0, 30.0, 9.0, 33.0, 52.0, 429.0, 20.0, 77.0, 55.0, 56.0, 28.0, 19.0, 10.0, 104.0, 25.0, 61.0, 91.0, 37.0, 0.0, 8.0, 31.0, 10.0, 13.0, 19.0, 17.0, 54.0, 42.0, 108.0, 61.0, 166.0, 0.0, 38.0, 36.0, 0.0, 8.0, 8.0, 55.0, 692.0, 14.0, 27.0, 72.0, 23.0, 50.0, 52.0, 0.0, 13.0, 52.0, 0.0, 123.0, 12.0, 0.0, 8.0, 19.0, 37.0, 17.0, 17.0, 54.0, 171.0, 53.0, 30.0, 705.0, 16.0, 30.0, 399.0, 27.0, 162.0, 17.0, 50.0, 87.0, 77.0, 8.0, 0.0, 41.0, 27.0, 27.0, 17.0, 0.0, 97.0, 8.0, 35.0, 19.0, 35.0, 329.0, 0.0, 27.0, 19.0, 0.0, 19.0, 9.0, 38.0, 17.0, 78.0, 0.0, 6.0, 33.0, 71.0, 45.0, 56.0, 34.0, 90.0, 27.0, 424.0, 57.0, 17.0, 25.0, 14.0, 94.0, 19.0, 23.0, 108.0, 30.0, 113.0, 135.0, 12.0, 8.0, 9.0, 88.0, 28.0, 53.0, 24.0, 52.0, 33.0, 96.0, 112.0, 17.0, 27.0, 44.0, 48.0, 8.0, 109.0, 43.0, 42.0, 19.0, 12.0, 24.0, 64.0, 8.0, 5.0, 86.0, 12.0, 18.0, 23.0, 19.0, 33.0, 39.0, 26.0, 27.0, 12.0, 49.0, 126.0, 101.0, 57.0, 65.0, 32.0, 92.0, 50.0, 212.0, 12.0, 19.0, 84.0, 8.0, 56.0, 45.0, 55.0, 310.0, 9.0, 17.0, 31.0, 22.0, 42.0, 31.0, 112.0, 19.0, 73.0, 21.0, 49.0, 0.0, 174.0, 31.0, 15.0, 24.0, 492.0, 60.0, 135.0, 22.0, 114.0, 178.0, 0.0, 50.0, 25.0, 66.0, 36.0, 0.0, 21.0, 38.0, 9.0, 22.0, 27.0, 13.0, 500.0, 20.0, 121.0, 507.0, 38.0, 9.0, 55.0, 10.0, 182.0, 60.0, 24.0, 124.0, 23.0, 8.0, 70.0, 173.0, 27.0, 70.0, 9.0, 71.0, 61.0, 51.0, 139.0, 49.0, 37.0, 9.0, 8.0, 157.0, 27.0, 63.0, 0.0, 21.0, 37.0, 600.0, 19.0, 16.0, 52.0, 88.0, 82.0, 0.0, 0.0, 116.0, 85.0, 143.0, 94.0, 23.0, 21.0, 8.0, 17.0, 75.0, 36.0, 24.0, 35.0, 30.0, 26.0, 8.0, 19.0, 8.0, 27.0, 0.0, 41.0, 10.0, 606.0, 26.0, 11.0, 58.0, 8.0, 20.0, 33.0, 20.0, 555.0, 32.0, 27.0, 19.0, 12.0, 23.0, 35.0, 21.0, 61.0, 94.0, 28.0, 114.0, 50.0, 19.0, 17.0, 0.0, 14.0, 17.0, 0.0, 17.0, 19.0, 9.0, 17.0, 26.0, 149.0, 44.0, 83.0, 103.0, 8.0, 50.0, 77.0, 17.0, 84.0, 53.0, 0.0, 14.0, 19.0, 13.0, 46.0, 10.0, 77.0, 0.0, 8.0, 36.0, 11.0, 183.0, 20.0, 8.0, 283.0, 164.0, 0.0, 61.0, 45.0, 70.0, 29.0, 17.0, 23.0, 11.0, 84.0, 120.0, 33.0, 23.0, 91.0, 9.0, 23.0, 13.0, 113.0, 56.0, 98.0, 0.0, 31.0, 429.0, 194.0, 117.0, 46.0, 5.0, 104.0, 20.0, 108.0, 12.0, 182.0, 8.0, 17.0, 27.0, 17.0, 53.0, 20.0, 27.0, 25.0, 34.0, 17.0, 8.0, 0.0, 42.0, 21.0, 135.0, 19.0, 79.0, 0.0, 0.0, 74.0, 50.0, 416.0, 289.0, 12.0, 8.0, 28.0, 38.0, 25.0, 39.0, 8.0, 101.0, 17.0, 37.0, 45.0, 25.0, 45.0, 162.0, 8.0, 93.0, 22.0, 148.0, 26.0, 113.0, 410.0, 9.0, 16.0, 0.0, 27.0, 30.0, 121.0, 27.0, 17.0, 26.0, 20.0, 41.0, 135.0, 7.0, 366.0, 12.0, 0.0, 8.0, 97.0, 17.0, 14.0, 116.0, 492.0, 182.0, 43.0, 204.0, 17.0, 19.0, 66.0, 33.0, 79.0, 9.0, 64.0, 20.0, 183.0, 14.0, 17.0, 38.0, 571.0, 8.0, 135.0, 21.0, 21.0, 144.0, 41.0, 79.0, 17.0, 19.0, 554.0, 26.0, 72.0, 91.0, 8.0, 8.0, 200.0, 0.0, 17.0, 168.0, 0.0, 104.0, 46.0, 92.0, 19.0, 70.0, 49.0, 143.0, 8.0, 0.0, 8.0, 34.0, 70.0, 224.0, 24.0, 34.0, 15.0, 629.0, 99.0, 52.0, 70.0, 17.0, 19.0, 100.0, 0.0, 77.0, 0.0, 446.0, 17.0, 9.0, 99.0, 25.0, 108.0, 39.0, 27.0, 19.0, 20.0, 114.0, 38.0, 28.0, 0.0, 19.0, 95.0, 34.0, 17.0, 40.0, 26.0, 29.0, 40.0, 8.0, 29.0, 8.0, 36.0, 18.0, 40.0, 29.0, 47.0, 84.0, 19.0, 22.0, 4.0, 75.0, 39.0, 24.0, 33.0, 40.0, 34.0, 92.0, 182.0, 32.0, 17.0, 9.0, 10.0, 143.0, 49.0, 22.0, 19.0, 53.0, 112.0, 103.0, 30.0, 96.0, 8.0, 8.0, 25.0, 8.0, 39.0, 8.0, 23.0, 135.0, 182.0, 56.0, 50.0, 91.0, 78.0, 27.0, 19.0, 42.0, 44.0, 0.0, 36.0, 33.0, 52.0, 520.0, 40.0, 31.0, 58.0, 46.0, 186.0, 36.0, 43.0, 0.0, 23.0, 95.0, 21.0, 48.0, 68.0, 9.0, 34.0, 23.0, 22.0, 21.0, 88.0, 25.0, 39.0, 23.0, 12.0, 84.0, 30.0, 94.0, 39.0, 0.0, 9.0, 204.0, 23.0, 24.0, 59.0, 156.0, 19.0, 39.0, 289.0, 17.0, 156.0, 14.0, 8.0, 25.0, 63.0, 28.0, 27.0, 77.0, 67.0, 27.0, 41.0, 0.0, 11.0, 75.0, 24.0, 9.0, 8.0, 182.0, 41.0, 18.0, 122.0, 33.0, 124.0, 8.0, 521.0, 1488.0, 17.0, 129.0, 26.0, 22.0, 20.0, 19.0, 18.0, 19.0, 71.0, 182.0, 4.0, 329.0, 20.0, 16.0, 45.0, 28.0, 41.0, 20.0, 19.0, 14.0, 0.0, 0.0, 346.0, 95.0, 26.0, 66.0, 95.0, 13.0, 84.0, 12.0, 23.0, 27.0, 8.0, 45.0, 181.0, 26.0, 1006.0, 105.0, 17.0, 27.0, 77.0, 22.0, 143.0, 12.0, 31.0, 51.0, 14.0, 31.0, 58.0, 17.0, 47.0, 24.0, 0.0, 18.0, 27.0, 18.0, 32.0, 27.0, 8.0, 35.0, 78.0, 30.0, 0.0, 20.0, 50.0, 31.0, 9.0, 329.0, 100.0, 108.0, 881.0, 9.0, 54.0, 151.0, 37.0, 72.0, 39.0, 11.0, 57.0, 112.0, 105.0, 26.0, 13.0, 38.0, 35.0, 17.0, 77.0, 25.0, 18.0, 17.0, 77.0, 69.0, 48.0, 22.0, 61.0, 78.0, 125.0, 21.0, 124.0, 8.0, 17.0, 13.0, 20.0, 39.0, 28.0, 9.0, 41.0, 10.0, 12.0, 37.0, 0.0, 100.0, 217.0, 290.0, 41.0, 15.0, 27.0, 116.0, 17.0, 19.0, 8.0, 8.0, 14.0, 675.0, 0.0, 0.0, 0.0, 209.0, 71.0, 88.0, 151.0, 600.0, 94.0, 8.0, 9.0, 120.0, 52.0, 36.0, 37.0, 30.0, 57.0, 27.0, 155.0, 18.0, 35.0, 45.0, 17.0, 9.0, 77.0, 0.0, 33.0, 27.0, 406.0, 48.0, 91.0, 18.0, 9.0, 43.0, 225.0, 134.0, 35.0, 21.0, 52.0, 0.0, 8.0, 8.0, 9.0, 12.0, 39.0, 17.0, 10.0, 86.0, 59.0, 69.0, 31.0, 0.0, 33.0, 99.0, 689.0, 9.0, 338.0, 9.0, 23.0, 58.0, 19.0, 33.0, 94.0, 0.0, 35.0, 0.0, 17.0, 50.0, 22.0, 14.0, 72.0, 27.0, 0.0, 9.0, 18.0, 45.0, 17.0, 81.0, 8.0, 45.0, 21.0, 18.0, 8.0, 26.0, 130.0, 17.0, 102.0, 0.0, 8.0, 17.0, 216.0, 329.0, 0.0, 68.0, 93.0, 78.0, 0.0, 145.0, 125.0, 226.0, 42.0, 58.0, 43.0, 86.0, 303.0, 52.0, 40.0, 11.0, 42.0, 0.0, 906.0, 50.0, 843.0, 0.0, 0.0, 34.0, 69.0, 67.0, 17.0, 7.0, 75.0, 19.0, 17.0, 17.0, 8.0, 9.0, 19.0, 91.0, 456.0, 54.0, 30.0, 42.0, 96.0, 9.0, 319.0, 89.0, 101.0, 15.0, 36.0, 500.0, 429.0, 60.0, 0.0, 19.0, 9.0, 27.0, 33.0, 9.0]

##### Ground Truths: [82, 0, 17, 0, 10, 32, 94, 34, 62, 17, 18, 25, 25, 12, 8, 80, 24, 39, 37, 22, 117, 10, 18, 106, 36, 14, 117, 27, 17, 424, 70, 27, 94, 47, 33, 8, 174, 111, 17, 12, 14, 38, 1, 27, 73, 51, 103, 41, 45, 9, 22, 27, 186, 31, 19, 121, 31, 8, 8, 50, 23, 8, 27, 88, 17, 31, 44, 9, 118, 28, 121, 49, 38, 229, 73, 64, 33, 30, 9, 33, 52, 429, 20, 77, 55, 56, 28, 19, 10, 104, 25, 61, 91, 37, 0, 8, 31, 10, 13, 19, 17, 54, 42, 108, 61, 166, 0, 38, 36, 0, 8, 8, 55, 692, 14, 27, 72, 23, 50, 52, 0, 13, 52, 0, 123, 12, 0, 8, 19, 37, 17, 17, 54, 171, 53, 30, 705, 16, 30, 399, 27, 162, 17, 50, 87, 77, 8, 0, 41, 27, 27, 17, 0, 97, 8, 35, 19, 35, 329, 0, 27, 19, 0, 19, 9, 38, 17, 78, 0, 6, 33, 71, 45, 52, 34, 90, 25, 424, 57, 17, 25, 14, 94, 19, 23, 108, 30, 113, 135, 12, 8, 9, 88, 28, 53, 24, 52, 33, 96, 112, 17, 27, 44, 48, 8, 109, 43, 42, 19, 12, 24, 64, 8, 5, 86, 12, 18, 23, 19, 33, 39, 26, 27, 12, 49, 126, 101, 57, 65, 32, 92, 50, 212, 12, 19, 84, 8, 56, 45, 55, 310, 9, 17, 31, 22, 42, 31, 112, 19, 73, 21, 49, 0, 174, 31, 15, 24, 492, 60, 135, 22, 114, 178, 0, 50, 25, 66, 28, 0, 21, 38, 9, 22, 27, 13, 500, 20, 121, 507, 38, 9, 55, 10, 182, 60, 24, 124, 23, 8, 70, 173, 27, 70, 9, 71, 61, 51, 139, 49, 37, 9, 8, 157, 27, 63, 0, 21, 37, 600, 19, 16, 52, 88, 82, 0, 0, 116, 85, 143, 94, 23, 21, 8, 17, 75, 36, 24, 35, 30, 26, 8, 19, 8, 27, 0, 41, 10, 606, 26, 11, 58, 8, 20, 33, 20, 555, 32, 27, 19, 12, 23, 35, 21, 61, 94, 28, 114, 50, 19, 17, 0, 14, 17, 0, 17, 19, 9, 17, 26, 149, 40, 83, 103, 8, 50, 77, 17, 84, 53, 0, 14, 19, 11, 46, 10, 77, 0, 8, 36, 11, 183, 20, 8, 283, 164, 0, 61, 45, 70, 29, 17, 23, 11, 84, 120, 33, 23, 91, 9, 23, 13, 113, 56, 98, 0, 31, 429, 194, 117, 46, 5, 104, 20, 108, 12, 182, 8, 17, 27, 17, 53, 20, 27, 25, 34, 17, 8, 0, 42, 21, 135, 19, 79, 0, 0, 74, 50, 416, 289, 12, 8, 28, 38, 25, 39, 8, 101, 17, 37, 45, 25, 45, 162, 8, 93, 22, 148, 26, 113, 410, 9, 16, 0, 27, 30, 121, 27, 17, 26, 20, 41, 135, 7, 366, 12, 0, 8, 97, 17, 14, 116, 492, 182, 43, 204, 17, 19, 66, 33, 79, 9, 64, 20, 183, 14, 17, 38, 571, 8, 135, 21, 21, 144, 41, 79, 17, 19, 554, 26, 72, 91, 8, 8, 200, 0, 17, 168, 0, 104, 46, 92, 19, 70, 49, 143, 8, 0, 8, 34, 70, 224, 24, 34, 15, 629, 99, 52, 70, 17, 19, 100, 0, 77, 0, 446, 17, 9, 99, 25, 108, 39, 27, 19, 20, 114, 38, 28, 0, 19, 95, 34, 17, 40, 26, 29, 40, 8, 29, 8, 36, 18, 40, 29, 47, 84, 17, 22, 4, 75, 39, 24, 33, 40, 34, 92, 182, 32, 17, 9, 10, 143, 49, 22, 19, 53, 112, 103, 30, 96, 8, 8, 25, 8, 39, 8, 23, 135, 182, 56, 50, 91, 78, 27, 19, 42, 44, 0, 36, 33, 52, 520, 40, 31, 58, 46, 186, 36, 43, 0, 23, 95, 21, 48, 68, 9, 34, 23, 22, 21, 88, 25, 39, 23, 12, 84, 30, 94, 39, 0, 9, 204, 23, 24, 59, 156, 19, 39, 289, 17, 156, 14, 8, 25, 63, 28, 27, 77, 67, 27, 41, 0, 11, 75, 24, 9, 8, 182, 41, 18, 122, 33, 124, 8, 521, 1488, 17, 129, 26, 22, 20, 19, 18, 19, 71, 182, 4, 329, 20, 16, 45, 28, 41, 20, 19, 14, 0, 0, 346, 95, 26, 66, 95, 13, 84, 12, 23, 27, 8, 45, 181, 26, 1006, 105, 17, 27, 77, 22, 143, 12, 31, 51, 14, 31, 58, 17, 47, 24, 0, 18, 27, 18, 32, 27, 8, 35, 78, 30, 0, 20, 50, 31, 9, 329, 100, 108, 881, 9, 54, 151, 37, 72, 39, 11, 57, 112, 105, 26, 13, 38, 35, 17, 77, 25, 18, 17, 77, 69, 48, 22, 61, 72, 125, 21, 124, 8, 17, 13, 20, 35, 28, 9, 37, 10, 12, 37, 0, 100, 213, 290, 41, 15, 27, 116, 17, 19, 8, 8, 14, 675, 0, 0, 0, 209, 71, 80, 151, 600, 94, 8, 9, 120, 52, 36, 37, 30, 57, 27, 155, 18, 35, 45, 17, 9, 77, 0, 33, 27, 406, 48, 83, 18, 9, 43, 225, 134, 35, 21, 52, 0, 8, 8, 9, 12, 39, 17, 10, 86, 59, 69, 31, 0, 33, 99, 689, 9, 338, 9, 23, 58, 19, 33, 94, 0, 35, 0, 17, 50, 22, 14, 72, 27, 0, 9, 18, 45, 17, 81, 8, 45, 21, 18, 8, 26, 130, 17, 102, 0, 8, 17, 216, 329, 0, 68, 93, 78, 0, 145, 125, 226, 42, 58, 43, 86, 303, 52, 40, 11, 42, 0, 906, 50, 843, 0, 0, 34, 69, 67, 17, 7, 75, 19, 17, 17, 8, 9, 19, 91, 456, 54, 30, 42, 96, 9, 319, 89, 101, 15, 36, 500, 429, 60, 0, 19, 9, 27, 33, 9]
##### Test Results: RMSE - 0.5788409772458983, MAE: 0.05997931747673216, Num Gt: 954/967

