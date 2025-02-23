"""Generating node mappings between 2 labelled graphs such that edit distance is minimal"""
import itertools
import numpy as np
import networkx as nx
import copy
import math
import heapq
import random

def priority_v1(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1` using numpy for efficiency."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    graph1_np = np.array(graph1)
    graph2_np = np.array(graph2)
    weights_np = np.array(weights)

    degrees1 = graph1_np.sum(axis=1)
    degrees2 = graph2_np.sum(axis=1)

    refined_weights = np.zeros((max_node, max_node), dtype=float)

    for i in range(n1):
        neighbors_i = np.where(graph1_np[i] == 1)[0]
        for j in range(n2):
            score = 0.0
            if i < weights_np.shape[0] and j < weights_np.shape[1]:
                score += weights_np[i, j]

            degree_diff = abs(degrees1[i] - degrees2[j])
            score += 1 / (1 + degree_diff)

            neighbors_j = np.where(graph2_np[j] == 1)[0]

            common_neighbors = 0
            if neighbors_i.size > 0 and neighbors_j.size > 0:
                neighbor_weights = weights_np[neighbors_i.reshape(-1, 1), neighbors_j]
                common_neighbors = neighbor_weights.sum()
            score += common_neighbors

            neighbor_similarity = 0
            if neighbors_i.size > 0 and neighbors_j.size > 0:
                degree_diffs = np.abs(degrees1[neighbors_i.reshape(-1, 1)] - degrees2[neighbors_j])
                neighbor_similarity = (neighbor_weights * (1 / (1 + degree_diffs))).sum()

            score += neighbor_similarity
            refined_weights[i, j] = score


    total_score = refined_weights[:n1, :n2].sum()

    if total_score > 0:
        refined_weights[:n1, :n2] /= total_score
    else:
        refined_weights[:n1, :n2] = 1 / (n1 * n2) if (n1 * n2) > 0 else 0

    return refined_weights.tolist()



def priority_v2(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0`.
  This version uses a normalized degree similarity and handles cases where one or both nodes have no neighbors more gracefully.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [([0.0] * max_node) for _ in range(max_node)]
  for i in range(n1):
    for j in range(n2):
      node_similarity = weights[i][j]
      degree_i = sum(graph1[i])
      degree_j = sum(graph2[j])

      # Normalized degree similarity
      max_degree = max(1, degree_i, degree_j) # Avoid division by zero
      degree_similarity = 1 - (abs(degree_i - degree_j) / max_degree)


      neighbors_i = [k for k in range(n1) if graph1[i][k]]
      neighbors_j = [l for l in range(n2) if graph2[j][l]]
      neighbor_similarity = 0
      common_neighbors = 0
      for ni in neighbors_i:
        for nj in neighbors_j:
          neighbor_similarity += weights[ni][nj]
          common_neighbors += 1

      if common_neighbors > 0:
        neighbor_similarity /= common_neighbors
      elif (degree_i > 0) and (degree_j > 0):  # Both have neighbors but no common ones
        neighbor_similarity = 0.0           # Explicitly set to 0 for dissimilarity
      else:
          neighbor_similarity = 1.0 if (degree_i == 0 and degree_j == 0) else 0.0 # 1 if both have no neighbors, else 0


      # Improved neighbor similarity factor, avoids potential division by zero
      neighbor_similarity_factor = neighbor_similarity * (min(degree_i, degree_j) / max(1, max(degree_i, degree_j)))


      refined_weights[i][j] = ((2 * node_similarity * (1 + degree_similarity) + neighbor_similarity_factor) / (3 + degree_similarity))
  return refined_weights


def priority_v3(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            score = 0.0
            if i < len(weights) and j < len(weights[0]):
                score += weights[i][j]

            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
            score -= degree_diff * 0.5

            neighbors_i = [k for k in range(n1) if graph1[i][k] == 1]
            neighbors_j = [k for k in range(n2) if graph2[j][k] == 1]

            common_neighbors = 0
            for ni in neighbors_i:
                for nj in neighbors_j:
                    if ni < len(weights) and nj < len(weights[0]):
                        common_neighbors += weights[ni][nj]
            score += common_neighbors


            neighbor_degree_similarity = 0
            for ni in neighbors_i:
                for nj in neighbors_j:
                    if ni < n1 and nj < n2:
                        neighbor_degree_similarity += (1 - abs(sum(graph1[ni]) - sum(graph2[nj])) * 0.1)  # Removed unnecessary parenthesis
            score += neighbor_degree_similarity


            # Consider 2-hop neighbors
            neighbors2_i = set()
            for ni in neighbors_i:
                neighbors2_i.update(k for k in range(n1) if graph1[ni][k] == 1 and k != i)  # Exclude the original node

            neighbors2_j = set()
            for nj in neighbors_j:
                neighbors2_j.update(k for k in range(n2) if graph2[nj][k] == 1 and k != j)

            common_neighbors2 = 0
            for ni2 in neighbors2_i:
                for nj2 in neighbors2_j:
                    if ni2 < len(weights) and nj2 < len(weights[0]):
                        common_neighbors2 += weights[ni2][nj2]
            score += common_neighbors2 * 0.5  # Reduced weight for 2-hop neighbors


            refined_weights[i][j] = max(0, score)

    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum
        else:
            for j in range(n2):
                refined_weights[i][j] = 1.0 / n2 if n2 > 0 else 0.0

    return refined_weights


def priority_v4(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]
    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]
    neighbors1 = [set(idx for idx, val in enumerate(row) if val) for row in graph1]
    neighbors2 = [set(idx for idx, val in enumerate(row) if val) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            common_neighbors = len(neighbors1[i].intersection(neighbors2[j]))
            
            degree_diff = abs(degrees1[i] - degrees2[j])
            max_degree = max(degrees1[i], degrees2[j])
            degree_similarity = (1 - (degree_diff / (max_degree + 1e-9))) if max_degree > 0 else (1 if degrees1[i] == degrees2[j] else 0) # Handle cases where both degrees are 0

            neighbor_diff = abs(len(neighbors1[i]) - len(neighbors2[j]))
            max_neighbors = max(len(neighbors1[i]), len(neighbors2[j]))
            neighbor_similarity = (1 - (neighbor_diff / (max_neighbors + 1e-9))) if max_neighbors > 0 else (1 if len(neighbors1[i]) == len(neighbors2[j]) else 0) # Handle cases where both neighbor counts are 0

            score = (common_neighbors * 0.5 + degree_similarity * 0.25 + neighbor_similarity * 0.25) # Added weights for better balancing


            if i < len(weights) and j < len(weights[0]):
                score += weights[i][j]  # Incorporate initial weights

            refined_weights[i][j] = max(0, score)

    return refined_weights


def priority_v5(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            score = 0.0
            if i < len(weights) and j < len(weights[0]):
                score += weights[i][j]

            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
            score -= degree_diff

            neighbors_i = [k for k in range(n1) if graph1[i][k] == 1]
            neighbors_j = [k for k in range(n2) if graph2[j][k] == 1]

            common_neighbors = 0
            for ni in neighbors_i:
                for nj in neighbors_j:
                    if ni < len(weights) and nj < len(weights[0]):
                        common_neighbors += weights[ni][nj]  # Use the weight instead of just checking > 0

            score += common_neighbors
            refined_weights[i][j] = score if score > 0 else 0 # avoid negative scores


    # Normalize row-wise and column-wise using numpy for efficiency
    refined_weights_np = np.array(refined_weights)
    row_sums = refined_weights_np.sum(axis=1)
    col_sums = refined_weights_np.sum(axis=0)


    for i in range(n1):
        if row_sums[i] > 0:
           refined_weights_np[i, :n2] /= row_sums[i]


    for j in range(n2):
        if col_sums[j] > 0:
            refined_weights_np[:n1, j] /= col_sums[j]
            
    return refined_weights_np.tolist()


def priority_v61(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      weights[i][j] = 1.0 / (1 + degree_diff + neighbor_diff)
  return weights


def priority_v6(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Normalizes the weights to sum to 1 for each row.
  """
  weights = priority_v61(graph1, graph2, weights) # Or use v0 as a base

  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)


  for i in range(n1):
    row_sum = sum(weights[i][:n2])  # Sum only up to n2
    if row_sum > 0:
      for j in range(n2):
        weights[i][j] /= row_sum
    else: # handle the case where row_sum is 0.  Distribute probability evenly
        for j in range(n2):
            weights[i][j] = 1.0 / n2
            
  return weights



def priority_v7(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`.  Normalizes by both row and column."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [([0.0] * max_node) for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      score = 0.0
      if ((i < len(weights)) and (j < len(weights[0]))):
        score += weights[i][j]
      degree_diff = abs((sum(graph1[i]) - sum(graph2[j])))
      score -= degree_diff  # Penalize degree difference
      neighbors_i = [k for k in range(n1) if (graph1[i][k] == 1)]
      neighbors_j = [k for k in range(n2) if (graph2[j][k] == 1)]
      common_neighbors = 0
      for ni in neighbors_i:
        for nj in neighbors_j:
          if ((ni < len(weights)) and (nj < len(weights[0])) and (weights[ni][nj] > 0)):
            common_neighbors += 1
      score += common_neighbors
      refined_weights[i][j] = score


  # Row normalization
  for i in range(n1):
    row_sum = sum(refined_weights[i][:n2])
    if row_sum > 0:
      for j in range(n2):
        refined_weights[i][j] /= row_sum

  # Column normalization  (This is the key improvement)
  for j in range(n2):
    col_sum = sum(refined_weights[i][j] for i in range(n1))
    if col_sum > 0:
      for i in range(n1):
        refined_weights[i][j] /= col_sum
  return refined_weights


def priority_v8(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1` using matrix operations for efficiency."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)

    # Convert to numpy arrays for efficient matrix operations
    graph1_np = np.array(graph1)
    graph2_np = np.array(graph2)
    weights_np = np.array(weights)

    refined_weights = np.zeros((max_node, max_node), dtype=float)

    for i in range(n1):
        for j in range(n2):
            # Efficiently calculate neighbor similarity using matrix multiplication
            neighbor_similarity = np.sum(graph1_np[i, :].reshape(1, -1) @ weights_np @ graph2_np[:, j].reshape(-1, 1))
            neighbor_similarity += weights_np[i, j]  # Add the direct weight
            refined_weights[i, j] = neighbor_similarity

    row_sums = refined_weights.sum(axis=1)
    for i in range(n1):
        if row_sums[i] > 0:
            refined_weights[i] /= row_sums[i]

    return refined_weights.tolist()


def priority_v9(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]
    neighbors1 = [set(idx for idx, val in enumerate(row) if val) for row in graph1]
    neighbors2 = [set(idx for idx, val in enumerate(row) if val) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            node_similarity = weights[i][j]
            degree_i = len(neighbors1[i])
            degree_j = len(neighbors2[j])
            degree_diff = abs(degree_i - degree_j)

            # Improved degree similarity calculation
            degree_similarity_factor = 1.0 / (1 + degree_diff) if degree_diff > 0 else 1.0

            common_neighbors = neighbors1[i].intersection(neighbors2[j])
            neighbor_similarity = 0.0
            if common_neighbors:
                # Normalize neighbor similarity by the maximum possible common neighbors
                max_common_neighbors = min(degree_i, degree_j)
                neighbor_similarity = sum(weights[k][l] for k in neighbors1[i] for l in neighbors2[j] if k < n1 and l < n2) / (max_common_neighbors * max_common_neighbors) if max_common_neighbors > 0 else 0


            # Combine similarities with adjusted weights for better balance
            refined_weights[i][j] = (2 * node_similarity * degree_similarity_factor + neighbor_similarity) / (2 + degree_similarity_factor)


    return refined_weights


def priority_v10(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Combines degree difference and neighbor similarity for a more robust estimation.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]
  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      degree_diff_weight = 1.0 / (1 + abs(degrees1[i] - degrees2[j]))

      neighbor_similarity = 0
      for neighbor1 in range(n1):
        if graph1[i][neighbor1]:
          for neighbor2 in range(n2):
            if graph2[j][neighbor2]:
              neighbor_similarity += 1.0 / (1 + abs(sum(graph1[neighbor1]) - sum(graph2[neighbor2])))

      weights[i][j] = (degree_diff_weight + neighbor_similarity) / 2  # Average the two weights

  return weights


def priority_v11(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`.
    This version incorporates neighbor structural similarity into the score calculation.
    """
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            score = 0.0
            if i < len(weights) and j < len(weights[0]):
                score += weights[i][j]

            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
            score -= degree_diff * 0.5

            neighbors_i = [k for k in range(n1) if graph1[i][k] == 1]
            neighbors_j = [k for k in range(n2) if graph2[j][k] == 1]

            common_neighbors = 0
            for ni in neighbors_i:
                for nj in neighbors_j:
                    if ni < len(weights) and nj < len(weights[0]):
                        common_neighbors += weights[ni][nj]
            score += common_neighbors

            neighbor_degree_similarity = 0
            for ni in neighbors_i:
                for nj in neighbors_j:
                    if ni < n1 and nj < n2:
                        neighbor_degree_similarity += (1 - abs(sum(graph1[ni]) - sum(graph2[nj])) * 0.1)
            score += neighbor_degree_similarity


            # Neighbor structural similarity
            neighbor_structure_similarity = 0
            for ni in neighbors_i:
                for nj in neighbors_j:
                    if ni < n1 and nj < n2:
                        ni_neighbors = [k for k in range(n1) if graph1[ni][k] == 1]
                        nj_neighbors = [k for k in range(n2) if graph2[nj][k] == 1]
                        common_ni_nj_neighbors = len(set(ni_neighbors) & set(nj_neighbors))
                        neighbor_structure_similarity += common_ni_nj_neighbors
            score += neighbor_structure_similarity * 0.1 # Weight the structural similarity

            refined_weights[i][j] = max(0, score)

    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum
        else:
            for j in range(n2):
                refined_weights[i][j] = 1.0 / n2 if n2 > 0 else 0.0

    return refined_weights


def priority_v121(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Improves `priority_v0` by considering neighbor degrees.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]
  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]
  neighbor_degrees1 = [[degrees1[k] for k,val in enumerate(row) if val == 1] for row in graph1]
  neighbor_degrees2 = [[degrees2[k] for k,val in enumerate(row) if val == 1] for row in graph2]

  for i in range(n1):
    for j in range(n2):
        degree_diff = abs(degrees1[i] - degrees2[j])
        neighbor_diff = 0
        for n1_deg in neighbor_degrees1[i]:
            min_diff = float('inf')
            for n2_deg in neighbor_degrees2[j]:
                min_diff = min(min_diff, abs(n1_deg - n2_deg))
            neighbor_diff += min_diff if min_diff != float('inf') else 0 # Add only if a corresponding neighbor exists

        weights[i][j] = 1.0 / (1 + degree_diff + neighbor_diff)  # Combine both differences
  return weights



def priority_v12(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Improves `priority_v1` by normalizing weights.
  """
  weights = priority_v121(graph1, graph2, weights)  # Use v1 as a base
  n1 = len(graph1)
  n2 = len(graph2)

  for i in range(n1):
      row_sum = sum(weights[i][:n2])  # Normalize based on the actual graph size
      if row_sum > 0:
          for j in range(n2):
              weights[i][j] /= row_sum
  return weights


def priority_v13(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            score = 0.0
            if i < len(weights) and j < len(weights[0]):
                score += weights[i][j]

            # Penalize degree difference more smoothly
            degree_i = sum(graph1[i])
            degree_j = sum(graph2[j])
            degree_diff = abs(degree_i - degree_j)
            max_degree = max(degree_i, degree_j)
            if max_degree > 0:
                score -= (degree_diff / max_degree)  # Normalize by maximum degree

            neighbors_i = [k for k in range(n1) if graph1[i][k] == 1]
            neighbors_j = [k for k in range(n2) if graph2[j][k] == 1]

            common_neighbors = 0
            for ni in neighbors_i:
                for nj in neighbors_j:
                    if ni < len(weights) and nj < len(weights[0]):
                        common_neighbors += weights[ni][nj]
            score += common_neighbors


            neighbor_similarity = 0
            for ni in neighbors_i:
                for nj in neighbors_j:
                    if ni < len(graph1) and nj < len(graph2):
                        deg_ni = sum(graph1[ni])
                        deg_nj = sum(graph2[nj])
                        max_deg_neighbor = max(deg_ni, deg_nj)
                        if max_deg_neighbor > 0:
                            neighbor_similarity += (1 - abs(deg_ni - deg_nj) / max_deg_neighbor)
            
            if len(neighbors_i) > 0 and len(neighbors_j) > 0:  # Avoid division by zero
                score += neighbor_similarity / (len(neighbors_i) * len(neighbors_j))


            refined_weights[i][j] = max(0, score)

    # Normalize rows to sum to 1
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum
        else:  # Handle cases where row sum is 0 (e.g., isolated nodes)
            for j in range(n2):
                refined_weights[i][j] = 1.0 / n2 if n2 > 0 else 0.0

    return refined_weights



def priority_v14(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`.  Uses sets for neighbor checking and handles edge cases better."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    neighbors1 = [set(n for n, adj in enumerate(row) if adj) for row in graph1]
    neighbors2 = [set(n for n, adj in enumerate(row) if adj) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            node_similarity = weights[i][j] if i < len(weights) and j < len(weights[i]) else 0  # Handle potential index errors

            degree_i = len(neighbors1[i])
            degree_j = len(neighbors2[j])
            degree_diff = abs(degree_i - degree_j)
            edge_similarity = 1.0 / (1.0 + degree_diff)

            common_neighbors = neighbors1[i].intersection(neighbors2[j])
            neighbor_similarity = 0
            if common_neighbors:  # Only calculate if there are common neighbors
              neighbor_similarity = sum(weights[k][l] for k in neighbors1[i] for l in neighbors2[j] if k < len(weights) and l < len(weights[k])) / len(common_neighbors)


            refined_weights[i][j] = (node_similarity + edge_similarity + neighbor_similarity) / 3

    return refined_weights



def priority_v15(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1` using neighbor comparison."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]
    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]
    neighbors1 = [set(idx for idx, val in enumerate(row) if val) for row in graph1]
    neighbors2 = [set(idx for idx, val in enumerate(row) if val) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            degree_diff = abs(degrees1[i] - degrees2[j])
            common_neighbors = len(neighbors1[i].intersection(neighbors2[j]))
            neighbor_diff = abs(len(neighbors1[i]) - len(neighbors2[j]))

            # Combine degree difference, common neighbors, and neighbor difference
            score = common_neighbors - degree_diff - neighbor_diff  # Prioritize common neighbors

            # Incorporate initial weights if available
            if i < len(weights) and j < len(weights[0]):
                score += weights[i][j]

            refined_weights[i][j] = max(0, score)  # Ensure non-negative scores


    # Normalize the scores to probabilities
    max_score = max(max(row) for row in refined_weights) if any(any(row) for row in refined_weights) else 1.0
    if max_score > 0:
        for i in range(n1):
            for j in range(n2):
                refined_weights[i][j] /= max_score
                
    return refined_weights

