"""Generating node mappings between 2 labelled graphs such that edit distance is minimal"""
import itertools
import numpy as np
import networkx as nx
import copy
import math
import heapq
import random

def priority_v1(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    if weights is None or len(weights) != max_node or len(weights[0]) != max_node:
        weights = [[(1.0 / max_node) if i < n1 and j < n2 else 0.0 for j in range(max_node)] for i in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            degree1 = sum(graph1[i])
            degree2 = sum(graph2[j])

            # Node Similarity Calculation (Improved)
            node_similarity = 1.0 if degree1 == degree2 == 0 else (1.0 - abs(degree1 - degree2) / max(degree1, degree2, 1)) # Avoid division by zero

            edge_similarity = 0.0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] == 1 and graph2[j][l] == 1:
                        edge_similarity += weights[k][l]
            
            # Combining Similarities (Weighted Average - potentially adjustable)
            refined_weights[i][j] = (weights[i][j] * 0.5 + node_similarity * 0.25 + edge_similarity * 0.25)


    # Normalization (Improved - handles cases with zero row sums more gracefully)
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum
        else:  # Distribute uniformly if no similar nodes found
            for j in range(n2):
                refined_weights[i][j] = 1.0 / n2 if n2 > 0 else 0.0
    
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


def priority_v3(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`. Uses numpy for efficiency."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)

  if not weights or len(weights) != max_node or len(weights[0]) != max_node:
      weights = np.full((max_node, max_node), 1.0 / max_node)
  else:
      weights = np.array(weights, dtype=float)

  refined_weights = np.zeros((max_node, max_node))

  graph1_np = np.array(graph1)
  graph2_np = np.array(graph2)

  for i in range(n1):
      for j in range(n2):
          degree_diff = abs(graph1_np[i].sum() - graph2_np[j].sum())
          node_similarity = 1.0 / (1.0 + degree_diff)

          edge_similarity = 0.0
          num_potential_edges = graph1_np[i].sum() * graph2_np[j].sum()

          if num_potential_edges > 0:
              edge_similarity = (graph1_np[i, :, None] * graph2_np[j]) * weights.T[:n1, :n2]
              edge_similarity = edge_similarity.sum() / num_potential_edges

          refined_weights[i, j] = (weights[i, j] + node_similarity + edge_similarity) / 3

  for i in range(n1):
      row_sum = refined_weights[i, :n2].sum()
      if row_sum > 0:
          refined_weights[i, :n2] /= row_sum

  return refined_weights.tolist()


def priority_v4(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1` using common neighbors."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [([0.0] * max_node) for _ in range(max_node)]
  for i in range(n1):
      for j in range(n2):
          common_neighbors = 0
          for k in range(n1):
              for l in range(n2):
                  if graph1[i][k] == 1 and graph2[j][l] == 1 and weights[k][l] > 0: # Consider existing mappings
                      common_neighbors += 1
          refined_weights[i][j] = common_neighbors
  return refined_weights


def priority_v5(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0`.
  Considers neighbor similarity in addition to node degree difference."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [([0.0] * max_node) for _ in range(max_node)]
  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      degree_similarity = 1.0 / (1.0 + abs(degrees1[i] - degrees2[j]))

      neighbor_similarity = 0.0
      neighbors1 = [k for k in range(n1) if graph1[i][k] == 1]
      neighbors2 = [k for k in range(n2) if graph2[j][k] == 1]

      if neighbors1 and neighbors2:  # Avoid division by zero if either node has no neighbors
        common_neighbors = 0
        for n1_neighbor in neighbors1:
          for n2_neighbor in neighbors2:
              common_neighbors += weights[n1_neighbor][n2_neighbor] # Leverage existing weights as a proxy for neighbor similarity
        neighbor_similarity = common_neighbors / (len(neighbors1) * len(neighbors2))

      weights[i][j] = (degree_similarity + neighbor_similarity) / 2  # Combine degree and neighbor similarity

  return weights


def priority_v6(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0`. Handles edge cases, normalizes correctly, and uses a more robust similarity measure."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  if weights is None or len(weights) != max_node or len(weights[0]) != max_node:
    weights = [[1.0 / max_node] * max_node for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      # Degree Similarity (using Jaccard index for better handling of different sized graphs)
      neighbors1 = sum(graph1[i])
      neighbors2 = sum(graph2[j])
      if neighbors1 + neighbors2 > 0:  # Avoid division by zero
          node_similarity = sum(graph1[i][k] & graph2[j][l] for k in range(n1) for l in range(n2) if graph1[i][k] and graph2[j][l]) / (neighbors1 + neighbors2)
      else:
          node_similarity = 1.0 if neighbors1 == neighbors2 == 0 else 0.0 # if both nodes have degree 0, they are similar


      edge_similarity = 0.0
      for k in range(n1):
          for l in range(n2):
              if graph1[i][k] and graph2[j][l]:
                  edge_similarity += weights[k][l]

      refined_weights[i][j] = (weights[i][j] + node_similarity + edge_similarity) / 3


  # Row-wise normalization within the relevant submatrix
  for i in range(n1):
    row_sum = sum(refined_weights[i][:n2])
    if row_sum > 0:
      for j in range(n2):
        refined_weights[i][j] /= row_sum
    else: # if row sum is 0, distribute probability evenly
        for j in range(n2):
            refined_weights[i][j] = 1.0 / n2


  return refined_weights


def priority_v7(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

def priority_v8(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    if not weights or len(weights) != max_node or len(weights[0]) != max_node:
        weights = [[(1.0 / max_node) if i < n1 and j < n2 else 0.0 for j in range(max_node)] for i in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            degree1 = sum(graph1[i])
            degree2 = sum(graph2[j])

            # Node Similarity Calculation (improved)
            node_similarity = 1.0 - (abs(degree1 - degree2) / (degree1 + degree2 + 1e-6)) # Adding a small value to avoid division by zero


            edge_similarity = 0.0
            neighbor1 = [k for k in range(n1) if graph1[i][k]]
            neighbor2 = [l for l in range(n2) if graph2[j][l]]

            for k in neighbor1:
                for l in neighbor2:
                    edge_similarity += weights[k][l]

            # Consider the number of neighbors when calculating edge similarity
            edge_similarity /= (len(neighbor1) * len(neighbor2) + 1e-6) # Adding a small value to avoid division by zero

            refined_weights[i][j] = (weights[i][j] + node_similarity + edge_similarity) / 3


    # Normalization (improved) - normalize across columns as well as rows
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum
        else:  # Handle cases where row_sum is zero
            for j in range(n2):
                refined_weights[i][j] = 1.0 / n2 if n2 > 0 else 0.0

    for j in range(n2):
        col_sum = sum(refined_weights[i][j] for i in range(n1))
        if col_sum > 0:
            for i in range(n1):
                refined_weights[i][j] /= col_sum
        else:
            for i in range(n1):
                refined_weights[i][j] = 1.0 / n1 if n1 > 0 else 0.0


    return refined_weights


def priority_v9(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            neighbors1 = graph1[i]  # Directly use the adjacency list
            neighbors2 = graph2[j]  # Directly use the adjacency list

            common_neighbors = sum(1 for n1 in neighbors1 for n2 in neighbors2 if weights[n1][n2] > 0)

            similarity = 0.0
            total_neighbors = len(neighbors1) + len(neighbors2)
            if total_neighbors > 0:
                similarity = (2 * common_neighbors) / total_neighbors

            if i < len(weights) and j < len(weights[i]):
                refined_weights[i][j] = 0.5 * weights[i][j] + 0.5 * similarity

    return refined_weights




def priority_v10(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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


def priority_v111(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

def priority_v11(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Normalizes the weights from v1 to probabilities.
  """
  weights = priority_v111(graph1, graph2, weights)  # Use v1 as a base

  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)


  # Normalize weights to probabilities
  for i in range(n1):
    row_sum = sum(weights[i][:n2]) # Only sum up to n2 to handle different sized graphs
    if row_sum > 0:  # Avoid division by zero
        for j in range(n2):
            weights[i][j] /= row_sum
    else:
        # If row_sum is 0, distribute probability evenly
        for j in range(n2):
            weights[i][j] = 1.0 / n2 if n2 > 0 else 0.0


  return weights


def priority_v12(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

def priority_v13(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`. Uses numpy for efficiency and handles empty graphs."""
    n1 = len(graph1)
    n2 = len(graph2)

    if n1 == 0 or n2 == 0:  # Handle empty graphs
        max_node = max(n1, n2)
        return [[0.0] * max_node for _ in range(max_node)]

    max_node = max(n1, n2)
    refined_weights = np.zeros((max_node, max_node))
    
    if weights is None or len(weights) != max_node or len(weights[0]) != max_node:
        weights = np.full((max_node, max_node), 1.0 / max_node)
    else:
        weights = np.array(weights) # Convert to numpy array for efficiency

    graph1_np = np.array(graph1)
    graph2_np = np.array(graph2)

    for i in range(n1):
        for j in range(n2):
            degree_diff = abs(graph1_np[i].sum() - graph2_np[j].sum())
            node_similarity = 1.0 / (1.0 + degree_diff)
            
            edge_similarity = (graph1_np[i,:][None,:] * graph2_np[j, :][:,None] * weights[:n1,:n2]).sum() #Efficient edge similarity calculation

            refined_weights[i, j] = (weights[i, j] + node_similarity + edge_similarity) / 3

    row_sums = refined_weights[:n1,:n2].sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    refined_weights[:n1,:n2] /= row_sums

    return refined_weights.tolist() # Convert back to list for consistency



def priority_v141(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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



def priority_v14(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Normalizes the weights to sum to 1 for each node in graph1.
  """
  weights = priority_v141(graph1, graph2, weights)  # Use v1 as a base

  n1 = len(graph1)
  n2 = len(graph2)

  for i in range(n1):
      row_sum = sum(weights[i][:n2]) # Only sum over valid nodes in graph2
      if row_sum > 0:  # Avoid division by zero
          for j in range(n2):
              weights[i][j] /= row_sum

  return weights

def priority_v15(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [([0.0] * max_node) for _ in range(max_node)]
    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            degree_similarity = 1.0 / (abs(degrees1[i] - degrees2[j]) + 1)
            neighbor_similarity = 0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] == 1 and graph2[j][l] == 1:
                        neighbor_similarity += (weights[k][l] if k < len(weights) and l < len(weights[k]) else 0)
            refined_weights[i][j] = degree_similarity * neighbor_similarity  # Combine both similarities

    return refined_weights
# Input Scores: [61.0, 66.0, 50.0, 43.0, 43.0, 34.0, 50.0, 58.0, 49.0, 35.0, 43.0, 51.0, 31.0, 26.0, 35.0, 64.0, 59.0, 48.0, 19.0, 52.0, 58.0, 65.0, 39.0, 57.0, 41.0, 39.0, 57.0, 48.0, 37.0, 41.0, 46.0, 36.0, 31.0, 60.0, 38.0, 34.0, 50.0, 35.0, 43.0, 53.0, 48.0, 34.0, 64.0, 43.0, 40.0, 32.0, 55.0, 54.0, 56.0, 57.0, 30.0, 66.0, 30.0, 57.0, 48.0, 48.0, 52.0, 60.0, 43.0, 33.0, 29.0, 41.0, 35.0, 39.0, 40.0, 43.0, 47.0, 37.0, 34.0, 35.0, 42.0, 77.0, 56.0, 44.0, 50.0, 42.0, 35.0, 55.0, 44.0, 32.0, 62.0, 38.0, 42.0, 38.0, 25.0, 41.0, 45.0, 54.0, 67.0, 52.0, 61.0, 60.0, 49.0, 20.0, 41.0, 52.0, 50.0, 22.0, 45.0, 35.0, 60.0, 34.0, 49.0, 46.0, 60.0, 43.0, 32.0, 38.0, 45.0, 44.0, 60.0, 46.0, 44.0, 50.0, 43.0, 39.0, 36.0, 52.0, 38.0, 52.0, 31.0, 26.0, 55.0, 33.0, 40.0, 35.0, 22.0, 33.0, 58.0, 42.0, 38.0, 43.0, 68.0, 28.0, 28.0, 31.0, 33.0, 27.0, 54.0, 24.0, 38.0, 45.0, 41.0, 46.0, 54.0, 20.0, 41.0, 52.0, 59.0, 40.0, 31.0, 52.0, 31.0, 43.0, 33.0, 55.0, 25.0, 23.0, 31.0, 49.0, 59.0, 59.0, 31.0, 55.0, 50.0, 46.0, 36.0, 18.0, 37.0, 54.0, 25.0, 34.0, 24.0, 46.0, 41.0, 28.0, 53.0, 20.0, 41.0, 35.0, 31.0, 31.0, 25.0, 60.0, 38.0, 38.0, 42.0, 38.0, 30.0, 57.0, 52.0, 49.0, 33.0, 21.0, 39.0, 32.0, 48.0, 60.0, 47.0, 56.0, 73.0, 40.0, 46.0, 38.0, 32.0, 34.0, 32.0, 48.0, 47.0, 33.0, 56.0, 33.0, 32.0, 44.0, 28.0, 33.0, 45.0, 50.0, 51.0, 51.0, 33.0, 46.0, 45.0, 60.0, 38.0, 39.0, 14.0, 33.0, 40.0, 60.0, 42.0, 39.0, 30.0, 53.0, 36.0, 36.0, 33.0, 66.0, 53.0, 20.0, 43.0, 38.0, 64.0, 48.0, 38.0, 20.0, 62.0, 38.0, 39.0, 41.0, 49.0, 33.0, 31.0, 28.0, 63.0, 66.0, 35.0, 62.0, 46.0, 45.0, 42.0, 43.0, 49.0, 44.0, 43.0, 63.0, 44.0, 41.0, 40.0, 52.0, 49.0, 42.0, 49.0, 31.0, 39.0, 41.0, 38.0, 43.0, 28.0, 44.0, 55.0, 43.0, 72.0, 46.0, 38.0, 42.0, 32.0, 63.0, 37.0, 51.0, 39.0, 25.0, 35.0, 63.0, 38.0, 53.0, 38.0, 40.0, 56.0, 56.0, 43.0, 44.0, 41.0, 56.0, 31.0, 53.0, 31.0, 45.0, 32.0, 14.0, 34.0, 31.0, 62.0, 48.0, 46.0, 79.0, 19.0, 49.0, 46.0, 49.0, 44.0, 23.0, 39.0, 38.0, 29.0, 13.0, 38.0, 47.0, 44.0, 35.0, 37.0, 63.0, 17.0, 59.0, 56.0, 35.0, 22.0, 47.0, 44.0, 49.0, 59.0, 52.0, 58.0, 48.0, 40.0, 53.0, 41.0, 54.0, 53.0, 37.0, 65.0, 38.0, 41.0, 36.0, 23.0, 43.0, 55.0, 35.0, 24.0, 49.0, 40.0, 35.0, 34.0, 30.0, 51.0, 75.0, 50.0, 33.0, 26.0, 21.0, 40.0, 9.0, 37.0, 43.0, 35.0, 40.0, 52.0, 34.0, 38.0, 61.0, 28.0, 34.0, 47.0, 72.0, 39.0, 20.0, 21.0, 55.0, 42.0, 52.0, 39.0, 53.0, 26.0, 37.0, 34.0, 41.0, 41.0, 37.0, 60.0, 42.0, 36.0, 40.0, 50.0, 54.0, 46.0, 47.0, 41.0, 35.0, 56.0, 52.0, 36.0, 40.0, 26.0, 43.0, 44.0, 57.0, 15.0, 47.0, 22.0, 58.0, 28.0, 64.0, 18.0, 46.0, 57.0, 35.0, 36.0, 16.0, 55.0, 39.0, 41.0, 29.0, 57.0, 47.0, 39.0, 30.0, 49.0, 46.0, 54.0, 56.0, 36.0, 20.0, 64.0, 38.0, 22.0, 56.0, 50.0, 50.0, 54.0, 59.0, 47.0, 20.0, 37.0, 38.0, 50.0, 46.0, 44.0, 59.0, 39.0, 35.0, 42.0, 41.0, 42.0, 55.0, 48.0, 43.0, 35.0, 63.0, 41.0, 61.0, 49.0, 37.0, 47.0, 26.0, 34.0, 23.0, 28.0, 39.0, 47.0, 86.0, 36.0, 39.0, 46.0, 67.0, 45.0, 37.0, 56.0, 60.0, 35.0, 34.0, 42.0, 49.0, 32.0, 29.0, 39.0, 45.0, 31.0, 62.0, 40.0, 33.0, 39.0, 20.0, 48.0, 45.0, 62.0, 32.0, 59.0, 46.0, 60.0, 58.0, 32.0, 35.0, 57.0, 61.0, 44.0, 52.0, 64.0, 47.0, 46.0, 36.0, 42.0, 51.0, 48.0, 49.0, 47.0, 53.0, 62.0, 37.0, 47.0, 46.0, 43.0, 54.0, 30.0, 44.0, 31.0, 51.0, 68.0, 37.0, 29.0, 48.0, 54.0, 45.0, 48.0, 55.0, 54.0, 41.0, 54.0, 49.0, 49.0, 23.0, 27.0, 59.0, 38.0, 57.0, 28.0, 29.0, 52.0, 69.0, 40.0, 47.0, 40.0, 56.0, 30.0, 37.0, 51.0, 31.0, 66.0, 41.0, 43.0, 42.0, 33.0, 35.0, 30.0, 52.0, 40.0, 66.0, 40.0, 54.0, 48.0, 25.0, 37.0, 37.0, 41.0, 41.0, 38.0, 36.0, 36.0, 40.0, 37.0, 31.0, 58.0, 58.0, 52.0, 57.0, 32.0, 55.0, 33.0, 16.0, 39.0, 43.0, 34.0, 51.0, 29.0, 45.0, 35.0, 51.0, 38.0, 54.0, 29.0, 57.0, 46.0, 51.0, 69.0, 45.0, 50.0, 84.0, 54.0, 38.0, 35.0, 49.0, 28.0, 38.0, 37.0, 42.0, 49.0, 36.0, 52.0, 34.0, 51.0, 60.0, 45.0, 35.0, 28.0, 61.0, 42.0, 37.0, 51.0, 47.0, 22.0, 52.0, 68.0, 57.0, 72.0, 39.0, 45.0, 48.0, 70.0, 73.0, 30.0, 37.0, 60.0, 55.0, 51.0, 47.0, 16.0, 63.0, 19.0, 43.0, 48.0, 50.0, 58.0, 40.0, 40.0, 49.0, 35.0, 36.0, 33.0, 52.0, 53.0, 44.0, 43.0, 55.0, 38.0, 63.0, 33.0, 42.0, 41.0, 53.0, 29.0, 68.0, 37.0, 44.0, 39.0, 31.0, 27.0, 21.0, 53.0, 58.0, 52.0, 33.0, 48.0, 53.0, 44.0, 46.0, 45.0, 60.0, 59.0, 50.0, 44.0, 38.0, 36.0, 43.0, 43.0, 33.0, 43.0, 49.0, 46.0, 66.0, 67.0, 51.0, 37.0, 36.0, 28.0, 52.0, 47.0, 54.0, 53.0, 28.0, 41.0, 56.0, 61.0, 45.0, 40.0, 58.0, 41.0, 32.0, 49.0, 34.0, 51.0, 43.0, 27.0, 54.0, 33.0, 52.0, 52.0, 40.0, 40.0, 48.0, 51.0, 44.0, 56.0, 46.0, 25.0, 40.0, 47.0, 27.0, 42.0, 51.0, 50.0, 63.0, 45.0, 48.0, 41.0, 29.0, 56.0, 42.0, 55.0, 49.0, 64.0, 31.0, 50.0, 62.0, 40.0, 50.0, 47.0, 47.0, 30.0, 42.0, 33.0, 44.0, 32.0, 53.0, 65.0, 43.0, 50.0, 46.0, 52.0, 26.0, 28.0, 56.0, 64.0, 34.0, 24.0, 31.0, 46.0, 47.0, 55.0, 55.0, 48.0, 50.0, 59.0, 40.0, 56.0, 44.0, 31.0, 46.0, 52.0, 28.0, 36.0, 46.0, 47.0, 37.0, 55.0, 31.0, 60.0, 19.0, 49.0, 49.0, 37.0, 31.0, 37.0, 54.0, 28.0, 49.0, 45.0, 28.0, 45.0, 26.0, 54.0, 48.0, 38.0, 47.0, 41.0, 14.0, 35.0, 40.0, 38.0, 38.0, 25.0, 59.0, 33.0, 45.0, 64.0, 50.0, 44.0, 43.0, 38.0, 47.0, 54.0, 55.0, 31.0, 45.0, 36.0, 37.0, 55.0, 23.0, 59.0, 43.0, 41.0, 59.0, 38.0, 17.0, 68.0, 56.0, 51.0, 59.0, 42.0, 48.0, 46.0, 53.0, 65.0, 36.0, 68.0, 58.0, 61.0, 29.0, 49.0, 48.0, 39.0, 34.0, 48.0, 29.0, 17.0, 48.0, 50.0, 50.0, 61.0, 37.0, 48.0, 37.0, 68.0, 62.0, 45.0, 27.0, 47.0, 28.0, 25.0, 63.0, 39.0, 42.0, 50.0, 54.0, 42.0, 41.0, 28.0, 41.0, 26.0, 54.0, 54.0, 55.0, 64.0, 51.0, 33.0, 32.0, 58.0, 22.0, 58.0, 63.0, 76.0, 33.0, 33.0, 59.0, 38.0, 31.0, 28.0, 42.0, 35.0, 49.0, 59.0, 27.0, 25.0, 32.0, 32.0, 17.0, 38.0, 44.0, 47.0, 49.0, 56.0, 40.0, 32.0, 50.0, 35.0, 46.0, 40.0, 47.0, 37.0, 22.0, 38.0, 50.0, 36.0, 43.0, 77.0, 56.0, 45.0, 29.0, 22.0, 39.0, 35.0, 60.0, 37.0, 28.0, 49.0, 37.0, 36.0, 64.0, 40.0, 65.0, 47.0, 64.0, 51.0, 27.0, 56.0, 44.0, 54.0, 49.0, 22.0, 53.0]

##### Best Scores: [59.0, 53.0, 47.0, 42.0, 28.0, 33.0, 50.0, 58.0, 44.0, 33.0, 35.0, 51.0, 29.0, 26.0, 33.0, 57.0, 49.0, 43.0, 18.0, 52.0, 55.0, 52.0, 38.0, 54.0, 41.0, 39.0, 52.0, 41.0, 37.0, 40.0, 35.0, 36.0, 29.0, 50.0, 36.0, 31.0, 41.0, 35.0, 36.0, 53.0, 45.0, 27.0, 59.0, 38.0, 40.0, 30.0, 52.0, 47.0, 50.0, 48.0, 29.0, 55.0, 14.0, 40.0, 48.0, 38.0, 44.0, 54.0, 34.0, 32.0, 29.0, 34.0, 31.0, 34.0, 39.0, 38.0, 45.0, 37.0, 30.0, 31.0, 37.0, 73.0, 35.0, 37.0, 44.0, 38.0, 34.0, 46.0, 41.0, 32.0, 52.0, 31.0, 35.0, 32.0, 25.0, 40.0, 41.0, 50.0, 56.0, 42.0, 61.0, 55.0, 48.0, 19.0, 39.0, 47.0, 42.0, 17.0, 40.0, 29.0, 56.0, 29.0, 41.0, 41.0, 56.0, 39.0, 32.0, 32.0, 35.0, 39.0, 55.0, 40.0, 37.0, 50.0, 38.0, 35.0, 30.0, 49.0, 35.0, 43.0, 27.0, 23.0, 49.0, 32.0, 37.0, 32.0, 22.0, 33.0, 51.0, 38.0, 36.0, 39.0, 57.0, 28.0, 28.0, 30.0, 31.0, 27.0, 49.0, 23.0, 34.0, 41.0, 40.0, 37.0, 46.0, 15.0, 30.0, 43.0, 54.0, 35.0, 29.0, 38.0, 28.0, 33.0, 29.0, 55.0, 23.0, 21.0, 29.0, 45.0, 58.0, 52.0, 31.0, 48.0, 46.0, 38.0, 25.0, 15.0, 32.0, 50.0, 23.0, 31.0, 24.0, 43.0, 39.0, 20.0, 50.0, 20.0, 37.0, 33.0, 27.0, 26.0, 25.0, 52.0, 25.0, 35.0, 34.0, 33.0, 25.0, 49.0, 50.0, 49.0, 28.0, 21.0, 32.0, 29.0, 41.0, 56.0, 39.0, 45.0, 60.0, 37.0, 38.0, 32.0, 27.0, 30.0, 28.0, 33.0, 46.0, 24.0, 54.0, 33.0, 28.0, 42.0, 22.0, 27.0, 33.0, 45.0, 45.0, 50.0, 27.0, 46.0, 43.0, 59.0, 35.0, 37.0, 14.0, 33.0, 36.0, 53.0, 39.0, 38.0, 30.0, 49.0, 36.0, 26.0, 32.0, 50.0, 43.0, 20.0, 42.0, 38.0, 50.0, 41.0, 35.0, 20.0, 62.0, 37.0, 31.0, 41.0, 40.0, 30.0, 30.0, 26.0, 58.0, 66.0, 32.0, 58.0, 35.0, 42.0, 35.0, 42.0, 45.0, 41.0, 39.0, 57.0, 40.0, 37.0, 40.0, 51.0, 43.0, 31.0, 45.0, 26.0, 39.0, 36.0, 31.0, 39.0, 24.0, 44.0, 46.0, 41.0, 63.0, 46.0, 38.0, 40.0, 31.0, 54.0, 35.0, 42.0, 35.0, 25.0, 35.0, 58.0, 35.0, 43.0, 35.0, 37.0, 47.0, 43.0, 37.0, 44.0, 39.0, 51.0, 31.0, 36.0, 28.0, 43.0, 28.0, 14.0, 34.0, 29.0, 54.0, 44.0, 39.0, 72.0, 14.0, 42.0, 40.0, 46.0, 43.0, 23.0, 35.0, 27.0, 28.0, 12.0, 38.0, 42.0, 40.0, 34.0, 34.0, 59.0, 15.0, 53.0, 55.0, 29.0, 22.0, 42.0, 42.0, 42.0, 48.0, 49.0, 54.0, 47.0, 36.0, 45.0, 39.0, 46.0, 46.0, 30.0, 55.0, 37.0, 35.0, 36.0, 22.0, 38.0, 47.0, 34.0, 23.0, 38.0, 34.0, 29.0, 26.0, 24.0, 45.0, 66.0, 48.0, 26.0, 22.0, 21.0, 30.0, 9.0, 33.0, 39.0, 29.0, 34.0, 48.0, 30.0, 34.0, 48.0, 28.0, 34.0, 39.0, 49.0, 37.0, 16.0, 19.0, 48.0, 37.0, 43.0, 34.0, 52.0, 24.0, 34.0, 30.0, 31.0, 41.0, 32.0, 54.0, 39.0, 35.0, 36.0, 48.0, 49.0, 40.0, 43.0, 37.0, 24.0, 52.0, 39.0, 32.0, 36.0, 21.0, 34.0, 41.0, 53.0, 14.0, 42.0, 20.0, 50.0, 22.0, 58.0, 15.0, 44.0, 44.0, 32.0, 36.0, 15.0, 53.0, 36.0, 41.0, 25.0, 50.0, 43.0, 37.0, 26.0, 46.0, 45.0, 51.0, 52.0, 36.0, 20.0, 61.0, 34.0, 22.0, 54.0, 44.0, 47.0, 54.0, 54.0, 44.0, 18.0, 34.0, 31.0, 50.0, 37.0, 30.0, 55.0, 38.0, 30.0, 35.0, 36.0, 35.0, 51.0, 44.0, 41.0, 35.0, 58.0, 38.0, 56.0, 44.0, 30.0, 44.0, 24.0, 28.0, 22.0, 25.0, 32.0, 47.0, 86.0, 34.0, 33.0, 43.0, 51.0, 40.0, 33.0, 52.0, 55.0, 35.0, 27.0, 37.0, 46.0, 32.0, 27.0, 34.0, 35.0, 30.0, 60.0, 37.0, 30.0, 37.0, 18.0, 46.0, 34.0, 57.0, 27.0, 56.0, 40.0, 46.0, 57.0, 28.0, 35.0, 47.0, 57.0, 41.0, 47.0, 59.0, 42.0, 42.0, 35.0, 38.0, 48.0, 43.0, 42.0, 45.0, 47.0, 58.0, 36.0, 45.0, 38.0, 35.0, 49.0, 29.0, 39.0, 26.0, 51.0, 60.0, 31.0, 25.0, 45.0, 48.0, 42.0, 48.0, 50.0, 51.0, 39.0, 51.0, 49.0, 35.0, 21.0, 22.0, 49.0, 38.0, 50.0, 24.0, 26.0, 47.0, 59.0, 35.0, 44.0, 38.0, 52.0, 29.0, 30.0, 47.0, 27.0, 50.0, 37.0, 37.0, 39.0, 27.0, 33.0, 28.0, 49.0, 35.0, 54.0, 34.0, 51.0, 46.0, 22.0, 33.0, 30.0, 32.0, 39.0, 28.0, 27.0, 36.0, 37.0, 36.0, 25.0, 58.0, 47.0, 41.0, 57.0, 27.0, 55.0, 30.0, 14.0, 35.0, 37.0, 30.0, 45.0, 25.0, 40.0, 35.0, 41.0, 33.0, 53.0, 25.0, 49.0, 35.0, 47.0, 62.0, 42.0, 46.0, 84.0, 41.0, 36.0, 29.0, 34.0, 27.0, 34.0, 28.0, 36.0, 46.0, 31.0, 47.0, 34.0, 48.0, 50.0, 43.0, 23.0, 27.0, 53.0, 40.0, 30.0, 48.0, 35.0, 20.0, 50.0, 60.0, 47.0, 69.0, 37.0, 35.0, 45.0, 66.0, 59.0, 29.0, 25.0, 54.0, 55.0, 44.0, 37.0, 15.0, 63.0, 17.0, 38.0, 38.0, 48.0, 53.0, 39.0, 39.0, 47.0, 32.0, 32.0, 32.0, 49.0, 50.0, 44.0, 42.0, 50.0, 28.0, 55.0, 29.0, 37.0, 41.0, 53.0, 24.0, 63.0, 37.0, 39.0, 38.0, 30.0, 25.0, 19.0, 49.0, 50.0, 48.0, 33.0, 45.0, 49.0, 41.0, 45.0, 44.0, 47.0, 59.0, 45.0, 37.0, 34.0, 31.0, 39.0, 32.0, 31.0, 40.0, 41.0, 41.0, 60.0, 55.0, 41.0, 30.0, 28.0, 26.0, 40.0, 44.0, 44.0, 49.0, 28.0, 39.0, 53.0, 51.0, 42.0, 36.0, 56.0, 38.0, 32.0, 46.0, 33.0, 49.0, 42.0, 23.0, 40.0, 28.0, 52.0, 46.0, 38.0, 37.0, 41.0, 36.0, 39.0, 51.0, 41.0, 24.0, 38.0, 41.0, 23.0, 37.0, 51.0, 45.0, 60.0, 37.0, 42.0, 35.0, 29.0, 46.0, 37.0, 47.0, 35.0, 63.0, 30.0, 46.0, 61.0, 40.0, 43.0, 37.0, 39.0, 29.0, 38.0, 31.0, 41.0, 31.0, 49.0, 59.0, 40.0, 47.0, 41.0, 47.0, 22.0, 27.0, 47.0, 63.0, 34.0, 19.0, 30.0, 39.0, 45.0, 55.0, 51.0, 45.0, 43.0, 56.0, 37.0, 53.0, 33.0, 26.0, 42.0, 52.0, 24.0, 33.0, 42.0, 37.0, 34.0, 53.0, 30.0, 57.0, 18.0, 47.0, 49.0, 28.0, 29.0, 31.0, 51.0, 24.0, 45.0, 41.0, 25.0, 40.0, 23.0, 50.0, 41.0, 38.0, 39.0, 35.0, 14.0, 34.0, 32.0, 29.0, 34.0, 21.0, 59.0, 30.0, 45.0, 49.0, 43.0, 44.0, 43.0, 35.0, 33.0, 54.0, 49.0, 25.0, 38.0, 32.0, 35.0, 45.0, 22.0, 47.0, 37.0, 33.0, 57.0, 38.0, 16.0, 63.0, 42.0, 42.0, 52.0, 37.0, 44.0, 43.0, 46.0, 54.0, 36.0, 62.0, 41.0, 51.0, 26.0, 43.0, 47.0, 37.0, 34.0, 45.0, 24.0, 13.0, 47.0, 46.0, 44.0, 54.0, 37.0, 40.0, 37.0, 66.0, 43.0, 41.0, 22.0, 38.0, 27.0, 20.0, 61.0, 31.0, 39.0, 40.0, 53.0, 37.0, 41.0, 24.0, 40.0, 26.0, 49.0, 50.0, 44.0, 56.0, 38.0, 27.0, 28.0, 56.0, 20.0, 55.0, 60.0, 63.0, 29.0, 31.0, 54.0, 38.0, 26.0, 27.0, 33.0, 35.0, 41.0, 51.0, 27.0, 25.0, 21.0, 30.0, 17.0, 30.0, 37.0, 36.0, 37.0, 47.0, 39.0, 29.0, 46.0, 29.0, 42.0, 38.0, 38.0, 36.0, 22.0, 34.0, 36.0, 34.0, 40.0, 70.0, 52.0, 41.0, 27.0, 19.0, 35.0, 28.0, 59.0, 36.0, 26.0, 38.0, 33.0, 23.0, 49.0, 38.0, 57.0, 42.0, 55.0, 47.0, 22.0, 55.0, 40.0, 50.0, 40.0, 21.0, 42.0]

##### Ground Truths: [57, 51, 45, 35, 27, 32, 49, 57, 42, 31, 35, 47, 24, 25, 31, 53, 43, 38, 13, 51, 50, 48, 35, 47, 38, 36, 52, 35, 29, 39, 31, 35, 26, 46, 33, 31, 38, 33, 31, 53, 41, 24, 59, 36, 37, 28, 45, 44, 47, 39, 27, 54, 14, 35, 48, 33, 39, 47, 29, 31, 29, 27, 26, 26, 36, 33, 42, 36, 27, 29, 33, 70, 30, 34, 40, 33, 29, 40, 38, 30, 48, 25, 31, 26, 22, 33, 36, 47, 54, 37, 57, 53, 42, 10, 35, 41, 37, 17, 38, 24, 50, 22, 32, 40, 54, 34, 31, 28, 33, 37, 52, 35, 30, 42, 34, 34, 30, 49, 30, 40, 24, 22, 45, 30, 29, 31, 20, 33, 50, 36, 34, 37, 54, 26, 24, 25, 27, 26, 45, 23, 30, 36, 40, 33, 41, 13, 27, 41, 51, 32, 22, 32, 22, 30, 25, 48, 23, 21, 27, 42, 58, 49, 29, 42, 43, 37, 24, 12, 29, 47, 21, 25, 24, 39, 33, 18, 42, 18, 29, 27, 24, 25, 22, 45, 25, 31, 26, 29, 20, 46, 44, 48, 24, 21, 28, 25, 38, 52, 34, 39, 55, 33, 35, 30, 24, 27, 27, 31, 43, 21, 52, 29, 26, 42, 20, 24, 31, 43, 38, 44, 24, 43, 43, 55, 32, 34, 14, 29, 36, 53, 39, 32, 25, 46, 34, 25, 27, 44, 42, 17, 41, 28, 46, 39, 33, 18, 55, 33, 25, 37, 36, 26, 27, 23, 55, 63, 26, 51, 30, 36, 28, 41, 39, 36, 37, 52, 35, 30, 35, 47, 40, 25, 42, 26, 33, 34, 28, 34, 24, 39, 38, 32, 59, 46, 34, 40, 28, 52, 34, 41, 32, 23, 34, 57, 31, 33, 29, 37, 42, 36, 36, 35, 37, 46, 29, 27, 28, 40, 27, 14, 32, 25, 51, 42, 32, 63, 14, 39, 33, 43, 37, 22, 34, 26, 27, 10, 38, 34, 35, 31, 28, 58, 15, 47, 55, 23, 22, 40, 40, 39, 38, 46, 45, 43, 35, 42, 29, 41, 45, 27, 52, 30, 32, 34, 21, 32, 40, 33, 22, 30, 31, 24, 22, 22, 39, 63, 44, 25, 20, 21, 26, 9, 31, 38, 24, 31, 41, 26, 28, 44, 27, 32, 38, 47, 33, 13, 16, 45, 32, 37, 29, 51, 20, 29, 29, 30, 39, 27, 51, 36, 33, 34, 47, 47, 39, 39, 29, 19, 51, 37, 32, 32, 19, 33, 36, 51, 14, 34, 20, 48, 20, 54, 15, 44, 38, 28, 35, 15, 49, 32, 39, 24, 45, 42, 35, 24, 41, 39, 48, 42, 34, 18, 55, 30, 21, 52, 43, 44, 51, 51, 42, 16, 32, 26, 48, 34, 25, 54, 37, 29, 29, 34, 33, 44, 36, 32, 32, 56, 37, 53, 37, 29, 42, 22, 21, 20, 24, 29, 46, 86, 34, 27, 43, 51, 38, 28, 48, 51, 32, 25, 36, 45, 32, 27, 31, 29, 27, 57, 33, 20, 33, 16, 44, 30, 53, 25, 51, 35, 43, 54, 24, 33, 46, 53, 36, 42, 54, 38, 37, 34, 37, 44, 37, 36, 41, 45, 55, 29, 42, 35, 30, 47, 29, 35, 23, 50, 58, 31, 21, 45, 45, 37, 44, 45, 49, 35, 50, 49, 28, 17, 21, 47, 33, 47, 19, 22, 45, 58, 33, 42, 36, 43, 26, 27, 41, 23, 50, 31, 32, 36, 26, 27, 25, 49, 33, 50, 30, 50, 45, 21, 30, 25, 30, 31, 23, 23, 31, 31, 30, 23, 57, 43, 38, 56, 26, 54, 30, 14, 32, 37, 29, 40, 20, 34, 28, 37, 30, 51, 25, 39, 29, 45, 55, 34, 46, 81, 40, 36, 27, 30, 25, 30, 24, 35, 44, 27, 37, 34, 46, 49, 38, 22, 25, 46, 37, 28, 44, 30, 20, 45, 57, 40, 69, 30, 30, 43, 65, 50, 24, 24, 52, 52, 40, 34, 14, 62, 14, 35, 32, 46, 52, 33, 34, 43, 31, 31, 28, 42, 46, 41, 38, 48, 25, 52, 28, 33, 40, 51, 21, 57, 37, 37, 33, 25, 21, 18, 48, 48, 46, 30, 41, 46, 36, 40, 42, 43, 51, 41, 32, 29, 27, 35, 28, 26, 37, 37, 29, 58, 50, 33, 23, 24, 25, 39, 43, 39, 41, 25, 39, 49, 50, 38, 31, 53, 30, 32, 45, 31, 47, 41, 21, 40, 21, 49, 45, 33, 35, 38, 34, 34, 51, 41, 24, 33, 39, 23, 31, 49, 39, 58, 32, 32, 33, 29, 44, 32, 44, 35, 63, 30, 41, 56, 40, 39, 30, 38, 28, 34, 24, 37, 29, 43, 54, 37, 45, 36, 38, 22, 19, 37, 61, 30, 15, 27, 37, 41, 53, 48, 39, 38, 54, 31, 49, 30, 24, 38, 52, 20, 33, 36, 31, 33, 45, 25, 48, 17, 45, 44, 24, 23, 29, 45, 21, 39, 35, 24, 31, 19, 45, 41, 36, 34, 35, 14, 30, 22, 28, 26, 19, 56, 29, 44, 47, 39, 42, 37, 32, 33, 51, 47, 23, 34, 29, 30, 40, 18, 43, 31, 27, 54, 35, 16, 59, 36, 36, 48, 36, 43, 38, 41, 47, 35, 58, 38, 45, 22, 41, 44, 34, 34, 42, 23, 13, 41, 42, 41, 50, 35, 34, 37, 65, 36, 35, 21, 35, 26, 20, 59, 30, 34, 36, 50, 31, 33, 24, 38, 26, 45, 41, 38, 50, 33, 24, 25, 56, 20, 51, 59, 59, 26, 29, 49, 35, 24, 24, 26, 34, 34, 49, 27, 25, 18, 25, 13, 28, 35, 34, 30, 43, 39, 25, 40, 23, 42, 36, 31, 32, 21, 27, 26, 28, 33, 61, 48, 41, 26, 16, 34, 25, 54, 27, 24, 36, 31, 22, 47, 34, 51, 39, 52, 43, 20, 53, 37, 49, 36, 18, 41]
##### Test Results: RMSE - 4.103264171303638, MAE: 3.3677685950413223, Num Gt: 101/968

