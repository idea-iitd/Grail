import itertools
import math


def priority_v1(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Combines degree similarity and neighbor similarity."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]


  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      degree_sim = 1.0 / (abs(degrees1[i] - degrees2[j]) + 1)  # Degree similarity

      neighbor_sim = 0
      for neighbor1 in range(n1):
        if graph1[i][neighbor1]:
          for neighbor2 in range(n2):
            if graph2[j][neighbor2]:
              neighbor_sim += 1.0 / (abs(sum(graph1[neighbor1]) - sum(graph2[neighbor2])) + 1)

      weights[i][j] = degree_sim + neighbor_sim # Combine both similarities
  
  return weights


def priority_v2(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    weights = [([0.0] * max_node) for _ in range(max_node)]
    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]
    neighbors1 = [set((j for j in range(n1) if graph1[i][j])) for i in range(n1)]
    neighbors2 = [set((j for j in range(n2) if graph2[i][j])) for i in range(n2)]

    for i in range(n1):
        for j in range(n2):
            common_neighbors = len(neighbors1[i] & neighbors2[j])
            # Smoothing added to avoid division by zero and enhance stability
            degree_similarity = (1.0 / (abs(degrees1[i] - degrees2[j]) + 1))  
            neighbor_similarity = (common_neighbors + 1e-6) / (math.sqrt(len(neighbors1[i]) * len(neighbors2[j])) + 1e-6)

            # Consider both structural similarity and initial weights
            weights[i][j] = (degree_similarity * neighbor_similarity) #* (weights[i][j] + 1e-6)


            # Incorporate local neighborhood structure more effectively
            union_neighbors1 = set()
            for neighbor in neighbors1[i]:
                union_neighbors1.update(neighbors1[neighbor])
            union_neighbors2 = set()
            for neighbor in neighbors2[j]:
                union_neighbors2.update(neighbors2[neighbor])

            common_neighbors_2hop = len(union_neighbors1 & union_neighbors2)
            if common_neighbors_2hop > 0:  # Avoid overweighting if no 2-hop neighbors are common
                weights[i][j] *= (common_neighbors_2hop + 1e-6) / (math.sqrt(len(union_neighbors1) * len(union_neighbors2)) + 1e-6)

    return weights



def priority_v3(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """
    Combines node degree similarity and neighbor degree similarity.
    """
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    weights = [[0.0] * max_node for _ in range(max_node)]

    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            node_similarity = 1.0 / abs(degrees1[i] - degrees2[j] + 1e-6)
            neighbor_similarity = 0
            for k in range(n1):
                for l in range(n2):
                    neighbor_similarity += graph1[i][k] * graph2[j][l] * (1.0 / abs(sum(graph1[k]) - sum(graph2[l]) + 1e-6))

            weights[i][j] = node_similarity * neighbor_similarity # Combine both similarities

    return weights



def priority_v41(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      neighbor_degrees1 = [degrees1[k] for k in range(n1) if graph1[i][k]]
      neighbor_degrees2 = [degrees2[k] for k in range(n2) if graph2[j][k]]

      degree_diff = abs(degrees1[i] - degrees2[j])
      neighbor_diff = sum(abs(d1 - d2) for d1, d2 in itertools.zip_longest(neighbor_degrees1, neighbor_degrees2, fillvalue=0))
      weights[i][j] = 1.0 / (degree_diff + neighbor_diff + 1e-6)
  return weights



def priority_v4(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Normalizes weights to represent probabilities.
  """
  weights = priority_v41(graph1, graph2, weights) #  Building on v1
  n1 = len(graph1)
  n2 = len(graph2)

  for i in range(n1):
      row_sum = sum(weights[i][:n2]) # Only sum over valid indices
      if row_sum > 0:  # Avoid division by zero
          for j in range(n2):
              weights[i][j] /= row_sum
  return weights


def priority_v5(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Computes initial probabilities based on neighborhood similarity using Jaccard index."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]
  neighbors1 = [set(i for i, x in enumerate(row) if x) for row in graph1] # Use sets for efficient intersection
  neighbors2 = [set(i for i, x in enumerate(row) if x) for row in graph2]

  for i in range(n1):
      for j in range(n2):
          intersection = len(neighbors1[i].intersection(neighbors2[j]))
          union = len(neighbors1[i].union(neighbors2[j]))
          if union == 0:  # Handle empty neighborhoods
              weights[i][j] = 1.0 if len(neighbors1[i]) == 0 and len(neighbors2[j]) == 0 else 0.0
          else:
              weights[i][j] = intersection / union

  return weights


def priority_v60(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities of mapping nodes between two graphs based on degree similarity and neighbor matching.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)] # Initialize weights
  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]
  for i in range(n1):
    for j in range(n2):
        neighbor_similarity = 0
        for k in range(n1):
            for l in range(n2):
                if graph1[i][k] == 1 and graph2[j][l] == 1:
                    neighbor_similarity += weights[k][l] # Use current weights for neighbor similarity

        degree_difference = abs(degrees1[i] - degrees2[j])
        if degree_difference == 0:
            weights[i][j] = neighbor_similarity + 1
        else:
            weights[i][j] = (neighbor_similarity + 1) / (degree_difference + 1)
  return weights


def priority_v61(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0` using normalized weights."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = priority_v60(graph1, graph2, weights) # Initialize with v0

  # Normalize weights
  for i in range(n1):
    row_sum = sum(weights[i][:n2])  # Sum only over valid nodes in graph2
    if row_sum > 0:
        for j in range(n2):
            weights[i][j] /= row_sum
  return weights


def priority_v6(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1` with iterative refinement."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = priority_v61(graph1, graph2, weights) # Initialize with v1

  # Iterative refinement (example - can be adjusted)
  for _ in range(5): #  A few iterations for refinement
      new_weights = [[0.0] * max_node for _ in range(max_node)]
      for i in range(n1):
          for j in range(n2):
              neighbor_similarity = 0
              for k in range(n1):
                  for l in range(n2):
                      if graph1[i][k] == 1 and graph2[j][l] == 1:
                          neighbor_similarity += weights[k][l]
              new_weights[i][j] = neighbor_similarity * weights[i][j]  # Combine with previous weights
      weights = new_weights

      # Normalize after each iteration
      for i in range(n1):
          row_sum = sum(weights[i][:n2])
          if row_sum > 0:
              for j in range(n2):
                  weights[i][j] /= row_sum

  return weights


def priority_v7(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`.  Uses sets for neighbor comparison."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]
  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]
  neighbors1 = [set(j for j in range(n1) if graph1[i][j]) for i in range(n1)]
  neighbors2 = [set(j for j in range(n2) if graph2[i][j]) for i in range(n2)]

  for i in range(n1):
    for j in range(n2):
      common_neighbors = len(neighbors1[i] & neighbors2[j])  # Efficient intersection using sets
      degree_similarity = 1.0 / (abs(degrees1[i] - degrees2[j]) + 1)
      neighbor_similarity = (common_neighbors + 1e-6) / (math.sqrt(len(neighbors1[i]) * len(neighbors2[j])) + 1e-6)
      weights[i][j] = degree_similarity * neighbor_similarity
  return weights
# Input Scores: [4.0, 0.0, 2.0, 6.0, 6.0, 4.0, 6.0, 8.0, 8.0, 3.0, 4.0, 3.0, 10.0, 6.0, 8.0, 1.0, 8.0, 8.0, 5.0, 5.0, 3.0, 4.0, 3.0, 3.0, 4.0, 6.0, 5.0, 0.0, 4.0, 5.0, 4.0, 5.0, 2.0, 5.0, 7.0, 2.0, 7.0, 9.0, 0.0, 4.0, 4.0, 4.0, 1.0, 7.0, 7.0, 2.0, 6.0, 9.0, 7.0, 5.0, 6.0, 6.0, 3.0, 2.0, 6.0, 0.0, 6.0, 2.0, 5.0, 4.0, 4.0, 3.0, 4.0, 4.0, 6.0, 2.0, 8.0, 5.0, 7.0, 8.0, 0.0, 6.0, 6.0, 4.0, 6.0, 4.0, 2.0, 2.0, 4.0, 4.0, 2.0, 0.0, 2.0, 6.0, 2.0, 8.0, 9.0, 7.0, 4.0, 6.0, 6.0, 10.0, 2.0, 4.0, 0.0, 11.0, 6.0, 0.0, 0.0, 4.0, 5.0, 6.0, 4.0, 4.0, 4.0, 2.0, 0.0, 0.0, 5.0, 6.0, 4.0, 3.0, 8.0, 11.0, 10.0, 6.0, 0.0, 2.0, 8.0, 10.0, 5.0, 4.0, 5.0, 9.0, 7.0, 4.0, 2.0, 2.0, 6.0, 6.0, 9.0, 4.0, 5.0, 5.0, 10.0, 1.0, 1.0, 9.0, 1.0, 4.0, 2.0, 2.0, 9.0, 2.0, 6.0, 9.0, 6.0, 4.0, 4.0, 6.0, 6.0, 6.0, 10.0, 3.0, 4.0, 2.0, 3.0, 0.0, 9.0, 7.0, 1.0, 5.0, 4.0, 6.0, 4.0, 2.0, 5.0, 4.0, 0.0, 7.0, 3.0, 9.0, 9.0, 8.0, 5.0, 6.0, 6.0, 6.0, 8.0, 3.0, 6.0, 0.0, 3.0, 8.0, 8.0, 9.0, 2.0, 6.0, 3.0, 6.0, 0.0, 4.0, 7.0, 2.0, 3.0, 5.0, 4.0, 7.0, 7.0, 4.0, 4.0, 8.0, 7.0, 2.0, 4.0, 2.0, 3.0, 8.0, 4.0, 4.0, 0.0, 8.0, 7.0, 5.0, 6.0, 7.0, 6.0, 4.0, 11.0, 6.0, 4.0, 6.0, 5.0, 6.0, 2.0, 7.0, 6.0, 9.0, 6.0, 1.0, 3.0, 2.0, 4.0, 4.0, 2.0, 8.0, 0.0, 0.0, 6.0, 8.0, 4.0, 4.0, 4.0, 4.0, 9.0, 4.0, 4.0, 2.0, 6.0, 4.0, 3.0, 6.0, 4.0, 6.0, 3.0, 3.0, 2.0, 4.0, 10.0, 4.0, 4.0, 6.0, 6.0, 4.0, 3.0, 8.0, 8.0, 3.0, 2.0, 4.0, 4.0, 6.0, 5.0, 5.0, 4.0, 7.0, 2.0, 6.0, 8.0, 9.0, 3.0, 6.0, 8.0, 4.0, 6.0, 3.0, 4.0, 5.0, 2.0, 4.0, 6.0, 4.0, 8.0, 3.0, 4.0, 0.0, 4.0, 6.0, 3.0, 6.0, 3.0, 6.0, 8.0, 6.0, 10.0, 6.0, 6.0, 2.0, 7.0, 6.0, 9.0, 4.0, 4.0, 6.0, 6.0, 4.0, 9.0, 8.0, 8.0, 9.0, 3.0, 9.0, 2.0, 6.0, 3.0, 9.0, 4.0, 0.0, 0.0, 5.0, 0.0, 6.0, 7.0, 0.0, 3.0, 4.0, 5.0, 6.0, 3.0, 0.0, 5.0, 0.0, 6.0, 7.0, 2.0, 6.0, 2.0, 4.0, 6.0, 9.0, 6.0, 3.0, 5.0, 4.0, 4.0, 10.0, 5.0, 4.0, 4.0, 4.0, 6.0, 9.0, 5.0, 5.0, 6.0, 4.0, 7.0, 8.0, 0.0, 4.0, 4.0, 4.0, 2.0, 6.0, 9.0, 0.0, 3.0, 8.0, 11.0, 6.0, 5.0, 4.0, 0.0, 4.0, 4.0, 2.0, 11.0, 7.0, 6.0, 8.0, 6.0, 6.0, 6.0, 7.0, 4.0, 0.0, 9.0, 4.0, 2.0, 2.0, 8.0, 1.0, 8.0, 7.0, 6.0, 7.0, 7.0, 3.0, 3.0, 4.0, 0.0, 6.0, 5.0, 8.0, 4.0, 4.0, 3.0, 11.0, 4.0, 4.0, 9.0, 4.0, 7.0, 5.0, 7.0, 3.0, 6.0, 9.0, 6.0, 6.0, 3.0, 4.0, 8.0, 6.0, 6.0, 8.0, 3.0, 10.0, 7.0, 12.0, 6.0, 6.0, 6.0, 5.0, 6.0, 7.0, 4.0, 10.0, 7.0, 5.0, 6.0, 7.0, 5.0, 4.0, 8.0, 6.0, 4.0, 6.0, 8.0, 4.0, 4.0, 8.0, 2.0, 7.0, 7.0, 6.0, 3.0, 3.0, 4.0, 13.0, 8.0, 6.0, 10.0, 2.0, 9.0, 6.0, 9.0, 6.0, 5.0, 5.0, 4.0, 10.0, 4.0, 2.0, 3.0, 8.0, 8.0, 6.0, 2.0, 5.0, 7.0, 10.0, 4.0, 4.0, 2.0, 9.0, 4.0, 7.0, 9.0, 8.0, 2.0, 4.0, 8.0, 3.0, 6.0, 4.0, 0.0, 2.0, 5.0, 6.0, 7.0, 8.0, 2.0, 9.0, 6.0, 2.0, 2.0, 10.0, 0.0, 4.0, 5.0, 11.0, 3.0, 4.0, 4.0, 5.0, 6.0, 10.0, 0.0, 5.0, 7.0, 0.0, 4.0, 3.0, 2.0, 11.0, 2.0, 2.0, 2.0, 5.0, 0.0, 2.0, 4.0, 4.0, 10.0, 6.0, 5.0, 1.0, 2.0, 7.0, 3.0, 4.0, 2.0, 6.0, 2.0, 4.0, 4.0, 2.0, 7.0, 4.0, 9.0, 3.0, 7.0, 6.0, 6.0, 4.0, 2.0, 4.0, 6.0, 4.0, 4.0, 10.0, 5.0, 6.0, 8.0, 5.0, 4.0, 4.0, 8.0, 3.0, 2.0, 11.0, 2.0, 11.0, 6.0, 10.0, 2.0, 3.0, 0.0, 6.0, 5.0, 0.0, 4.0, 2.0, 4.0, 4.0, 7.0, 3.0, 6.0, 4.0, 9.0, 6.0, 12.0, 4.0, 6.0, 6.0, 6.0, 5.0, 4.0, 2.0, 3.0, 2.0, 4.0, 2.0, 0.0, 2.0, 6.0, 7.0, 4.0, 6.0, 5.0, 4.0, 8.0, 10.0, 4.0, 4.0, 6.0, 2.0, 9.0, 3.0, 6.0, 5.0, 9.0, 4.0, 5.0, 6.0, 2.0, 6.0, 0.0, 4.0, 2.0, 6.0, 6.0, 2.0, 2.0, 4.0, 2.0, 8.0, 7.0, 9.0, 7.0, 9.0, 2.0, 4.0, 2.0, 2.0, 4.0, 4.0, 4.0, 5.0, 4.0, 6.0, 3.0, 4.0, 5.0, 4.0, 4.0, 5.0, 0.0, 0.0, 4.0, 9.0, 10.0, 4.0, 3.0, 6.0, 0.0, 7.0, 4.0, 11.0, 4.0, 4.0, 6.0, 7.0, 2.0, 4.0, 9.0, 2.0, 2.0, 9.0, 2.0, 6.0, 9.0, 2.0, 2.0, 4.0, 5.0, 4.0, 7.0, 8.0, 5.0, 6.0, 10.0, 7.0, 4.0, 6.0, 9.0, 3.0, 4.0, 4.0, 4.0, 3.0, 4.0, 4.0, 11.0, 2.0, 1.0, 2.0, 4.0, 4.0, 3.0, 4.0, 1.0, 3.0, 3.0, 0.0, 2.0, 4.0, 8.0, 2.0, 4.0, 2.0, 8.0, 9.0, 2.0, 4.0, 0.0, 4.0, 10.0, 9.0, 8.0, 9.0, 5.0, 5.0, 2.0, 2.0, 1.0, 6.0, 9.0, 6.0, 7.0, 7.0, 4.0, 7.0, 4.0, 3.0, 4.0, 4.0, 0.0, 4.0, 4.0, 7.0, 2.0, 9.0, 3.0, 7.0, 4.0, 6.0, 4.0, 6.0, 5.0, 3.0, 4.0, 6.0, 5.0, 1.0, 4.0, 2.0, 5.0, 3.0, 3.0, 6.0, 6.0, 5.0, 4.0, 6.0, 2.0, 11.0, 2.0, 2.0, 4.0, 4.0, 4.0, 7.0, 5.0, 4.0, 3.0, 9.0, 5.0, 4.0, 8.0, 3.0, 9.0, 14.0, 2.0, 0.0, 4.0, 2.0, 4.0, 4.0, 6.0, 4.0, 4.0, 4.0, 9.0, 8.0, 1.0, 2.0, 4.0, 6.0, 7.0, 6.0, 9.0, 6.0, 0.0, 7.0, 2.0, 11.0, 6.0, 4.0, 2.0, 12.0, 5.0, 9.0, 7.0, 7.0, 6.0, 4.0, 2.0, 4.0, 8.0, 5.0, 0.0, 4.0, 5.0, 4.0, 4.0, 7.0, 9.0, 9.0, 5.0, 4.0, 6.0, 10.0, 3.0, 4.0, 6.0, 0.0, 4.0, 6.0, 3.0, 6.0, 4.0, 2.0, 3.0, 3.0, 0.0, 4.0, 6.0, 4.0, 5.0, 6.0, 7.0, 5.0, 6.0, 4.0, 3.0, 6.0, 10.0, 4.0, 5.0, 2.0, 3.0, 7.0, 6.0, 6.0, 6.0, 7.0, 2.0, 2.0, 4.0, 4.0, 0.0, 4.0, 4.0, 5.0, 4.0, 5.0, 1.0, 4.0, 3.0, 6.0, 5.0, 9.0, 4.0, 2.0, 0.0, 4.0, 5.0, 10.0, 8.0, 4.0, 6.0, 8.0, 3.0, 8.0, 6.0, 6.0, 8.0, 4.0, 8.0, 9.0, 2.0, 4.0, 5.0, 2.0, 4.0, 3.0, 2.0, 4.0, 10.0, 4.0, 5.0, 0.0, 6.0, 9.0, 2.0, 8.0, 8.0, 6.0, 6.0, 5.0, 4.0, 8.0, 0.0, 4.0, 10.0, 1.0, 6.0, 6.0, 2.0, 5.0, 9.0, 3.0, 10.0, 9.0, 0.0, 2.0, 0.0, 4.0, 8.0, 2.0, 4.0, 4.0, 6.0, 7.0, 2.0, 9.0, 5.0, 0.0, 4.0, 4.0, 8.0, 2.0, 9.0, 4.0, 6.0, 3.0, 6.0, 6.0, 8.0, 7.0, 5.0, 3.0, 3.0, 3.0, 6.0, 4.0, 7.0, 8.0, 7.0, 5.0, 7.0, 3.0, 4.0, 6.0, 5.0, 4.0, 3.0, 4.0]

##### Best Scores: [4.0, 0.0, 2.0, 6.0, 6.0, 4.0, 6.0, 8.0, 8.0, 3.0, 4.0, 3.0, 8.0, 6.0, 8.0, 1.0, 8.0, 6.0, 5.0, 5.0, 3.0, 4.0, 1.0, 3.0, 4.0, 6.0, 5.0, 0.0, 4.0, 5.0, 4.0, 5.0, 2.0, 5.0, 7.0, 2.0, 7.0, 9.0, 0.0, 4.0, 4.0, 4.0, 1.0, 7.0, 7.0, 2.0, 4.0, 9.0, 7.0, 5.0, 6.0, 6.0, 3.0, 2.0, 6.0, 0.0, 6.0, 2.0, 5.0, 4.0, 4.0, 3.0, 4.0, 4.0, 6.0, 2.0, 8.0, 5.0, 7.0, 6.0, 0.0, 6.0, 6.0, 4.0, 6.0, 4.0, 2.0, 2.0, 4.0, 4.0, 2.0, 0.0, 2.0, 6.0, 2.0, 8.0, 9.0, 7.0, 4.0, 6.0, 6.0, 10.0, 2.0, 4.0, 0.0, 9.0, 6.0, 0.0, 0.0, 4.0, 3.0, 6.0, 4.0, 4.0, 4.0, 2.0, 0.0, 0.0, 5.0, 6.0, 4.0, 3.0, 4.0, 11.0, 10.0, 6.0, 0.0, 2.0, 8.0, 10.0, 5.0, 4.0, 5.0, 9.0, 5.0, 4.0, 2.0, 2.0, 6.0, 6.0, 9.0, 4.0, 5.0, 3.0, 10.0, 1.0, 1.0, 9.0, 1.0, 4.0, 2.0, 2.0, 9.0, 2.0, 6.0, 9.0, 4.0, 4.0, 2.0, 6.0, 4.0, 6.0, 8.0, 3.0, 4.0, 2.0, 3.0, 0.0, 9.0, 7.0, 1.0, 3.0, 4.0, 6.0, 4.0, 2.0, 3.0, 4.0, 0.0, 7.0, 3.0, 9.0, 9.0, 8.0, 5.0, 6.0, 4.0, 6.0, 8.0, 1.0, 6.0, 0.0, 3.0, 8.0, 8.0, 9.0, 2.0, 6.0, 3.0, 6.0, 0.0, 4.0, 7.0, 2.0, 3.0, 5.0, 4.0, 5.0, 5.0, 4.0, 4.0, 8.0, 7.0, 2.0, 4.0, 2.0, 3.0, 8.0, 4.0, 4.0, 0.0, 8.0, 5.0, 5.0, 6.0, 7.0, 6.0, 4.0, 11.0, 6.0, 4.0, 6.0, 5.0, 6.0, 2.0, 7.0, 4.0, 9.0, 6.0, 1.0, 3.0, 2.0, 4.0, 4.0, 2.0, 8.0, 0.0, 0.0, 6.0, 8.0, 4.0, 4.0, 4.0, 4.0, 9.0, 4.0, 4.0, 2.0, 6.0, 4.0, 3.0, 6.0, 4.0, 6.0, 3.0, 3.0, 2.0, 4.0, 10.0, 4.0, 4.0, 6.0, 6.0, 4.0, 3.0, 8.0, 8.0, 3.0, 2.0, 4.0, 2.0, 6.0, 5.0, 5.0, 4.0, 7.0, 2.0, 6.0, 8.0, 9.0, 3.0, 6.0, 8.0, 4.0, 4.0, 1.0, 4.0, 5.0, 2.0, 4.0, 6.0, 4.0, 8.0, 3.0, 4.0, 0.0, 4.0, 4.0, 1.0, 6.0, 3.0, 6.0, 8.0, 6.0, 8.0, 4.0, 4.0, 2.0, 7.0, 6.0, 9.0, 4.0, 4.0, 6.0, 6.0, 4.0, 5.0, 8.0, 8.0, 9.0, 3.0, 9.0, 2.0, 6.0, 3.0, 9.0, 4.0, 0.0, 0.0, 5.0, 0.0, 6.0, 7.0, 0.0, 3.0, 4.0, 3.0, 6.0, 3.0, 0.0, 5.0, 0.0, 6.0, 7.0, 2.0, 4.0, 2.0, 4.0, 6.0, 9.0, 6.0, 3.0, 5.0, 4.0, 4.0, 10.0, 5.0, 4.0, 4.0, 4.0, 6.0, 7.0, 5.0, 5.0, 6.0, 2.0, 7.0, 8.0, 0.0, 4.0, 4.0, 2.0, 2.0, 6.0, 9.0, 0.0, 3.0, 8.0, 11.0, 6.0, 5.0, 4.0, 0.0, 4.0, 4.0, 2.0, 11.0, 7.0, 4.0, 8.0, 6.0, 6.0, 6.0, 3.0, 4.0, 0.0, 7.0, 4.0, 2.0, 2.0, 8.0, 1.0, 8.0, 7.0, 6.0, 5.0, 7.0, 3.0, 3.0, 4.0, 0.0, 6.0, 5.0, 4.0, 4.0, 4.0, 3.0, 11.0, 4.0, 4.0, 9.0, 4.0, 5.0, 5.0, 5.0, 3.0, 6.0, 7.0, 6.0, 4.0, 3.0, 4.0, 8.0, 6.0, 6.0, 6.0, 1.0, 10.0, 7.0, 12.0, 4.0, 6.0, 6.0, 5.0, 6.0, 7.0, 4.0, 6.0, 7.0, 5.0, 6.0, 7.0, 5.0, 4.0, 8.0, 6.0, 4.0, 6.0, 8.0, 4.0, 2.0, 6.0, 2.0, 7.0, 7.0, 6.0, 3.0, 1.0, 4.0, 13.0, 8.0, 6.0, 10.0, 2.0, 5.0, 6.0, 9.0, 2.0, 5.0, 5.0, 2.0, 10.0, 4.0, 2.0, 3.0, 8.0, 6.0, 6.0, 2.0, 5.0, 7.0, 10.0, 4.0, 4.0, 2.0, 9.0, 4.0, 7.0, 9.0, 8.0, 2.0, 4.0, 8.0, 3.0, 4.0, 4.0, 0.0, 2.0, 5.0, 6.0, 7.0, 8.0, 2.0, 9.0, 6.0, 2.0, 2.0, 10.0, 0.0, 4.0, 3.0, 11.0, 3.0, 4.0, 4.0, 5.0, 2.0, 8.0, 0.0, 5.0, 7.0, 0.0, 4.0, 3.0, 2.0, 11.0, 2.0, 2.0, 2.0, 5.0, 0.0, 2.0, 4.0, 4.0, 8.0, 6.0, 5.0, 1.0, 2.0, 7.0, 3.0, 4.0, 2.0, 6.0, 2.0, 4.0, 2.0, 2.0, 7.0, 4.0, 9.0, 3.0, 7.0, 6.0, 6.0, 4.0, 2.0, 4.0, 6.0, 4.0, 4.0, 10.0, 3.0, 6.0, 8.0, 5.0, 4.0, 4.0, 8.0, 3.0, 2.0, 11.0, 2.0, 11.0, 6.0, 10.0, 2.0, 3.0, 0.0, 6.0, 5.0, 0.0, 4.0, 2.0, 4.0, 4.0, 7.0, 1.0, 4.0, 4.0, 9.0, 6.0, 12.0, 4.0, 6.0, 4.0, 6.0, 5.0, 4.0, 2.0, 1.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 7.0, 4.0, 6.0, 3.0, 2.0, 8.0, 10.0, 2.0, 2.0, 4.0, 2.0, 9.0, 3.0, 6.0, 5.0, 9.0, 4.0, 5.0, 4.0, 2.0, 6.0, 0.0, 4.0, 2.0, 6.0, 6.0, 2.0, 2.0, 4.0, 2.0, 8.0, 7.0, 9.0, 7.0, 9.0, 2.0, 4.0, 2.0, 2.0, 4.0, 4.0, 4.0, 5.0, 4.0, 6.0, 3.0, 4.0, 5.0, 4.0, 4.0, 5.0, 0.0, 0.0, 4.0, 9.0, 8.0, 4.0, 3.0, 6.0, 0.0, 7.0, 4.0, 11.0, 4.0, 4.0, 6.0, 5.0, 2.0, 4.0, 7.0, 2.0, 2.0, 9.0, 2.0, 6.0, 9.0, 2.0, 2.0, 4.0, 5.0, 4.0, 7.0, 8.0, 5.0, 6.0, 8.0, 7.0, 4.0, 6.0, 9.0, 3.0, 4.0, 4.0, 4.0, 3.0, 4.0, 4.0, 11.0, 2.0, 1.0, 2.0, 4.0, 4.0, 3.0, 4.0, 1.0, 3.0, 3.0, 0.0, 2.0, 4.0, 8.0, 2.0, 4.0, 2.0, 8.0, 9.0, 2.0, 4.0, 0.0, 4.0, 8.0, 9.0, 8.0, 7.0, 5.0, 5.0, 2.0, 2.0, 1.0, 6.0, 9.0, 6.0, 7.0, 7.0, 4.0, 7.0, 4.0, 3.0, 4.0, 4.0, 0.0, 2.0, 4.0, 7.0, 2.0, 9.0, 3.0, 7.0, 4.0, 4.0, 4.0, 6.0, 5.0, 3.0, 4.0, 6.0, 5.0, 1.0, 4.0, 2.0, 5.0, 3.0, 3.0, 4.0, 6.0, 5.0, 4.0, 6.0, 2.0, 11.0, 2.0, 2.0, 4.0, 4.0, 4.0, 7.0, 5.0, 4.0, 3.0, 7.0, 5.0, 4.0, 6.0, 3.0, 9.0, 14.0, 2.0, 0.0, 2.0, 2.0, 4.0, 4.0, 6.0, 4.0, 4.0, 4.0, 9.0, 8.0, 1.0, 2.0, 2.0, 6.0, 7.0, 6.0, 9.0, 4.0, 0.0, 7.0, 2.0, 11.0, 6.0, 4.0, 2.0, 10.0, 5.0, 7.0, 7.0, 7.0, 6.0, 4.0, 2.0, 4.0, 8.0, 5.0, 0.0, 4.0, 5.0, 4.0, 4.0, 7.0, 9.0, 7.0, 5.0, 4.0, 6.0, 10.0, 3.0, 4.0, 6.0, 0.0, 4.0, 6.0, 3.0, 6.0, 4.0, 2.0, 3.0, 3.0, 0.0, 4.0, 6.0, 4.0, 5.0, 6.0, 7.0, 5.0, 6.0, 4.0, 3.0, 6.0, 10.0, 4.0, 5.0, 2.0, 3.0, 5.0, 6.0, 6.0, 4.0, 7.0, 2.0, 2.0, 4.0, 4.0, 0.0, 4.0, 2.0, 5.0, 4.0, 5.0, 1.0, 4.0, 3.0, 6.0, 5.0, 9.0, 4.0, 2.0, 0.0, 4.0, 5.0, 10.0, 8.0, 4.0, 6.0, 8.0, 3.0, 8.0, 4.0, 6.0, 8.0, 2.0, 8.0, 9.0, 2.0, 4.0, 5.0, 2.0, 4.0, 3.0, 2.0, 4.0, 10.0, 4.0, 5.0, 0.0, 6.0, 9.0, 2.0, 6.0, 6.0, 6.0, 6.0, 3.0, 4.0, 8.0, 0.0, 4.0, 8.0, 1.0, 6.0, 6.0, 2.0, 3.0, 9.0, 3.0, 8.0, 9.0, 0.0, 2.0, 0.0, 4.0, 6.0, 2.0, 4.0, 4.0, 6.0, 7.0, 2.0, 9.0, 5.0, 0.0, 4.0, 0.0, 8.0, 2.0, 9.0, 4.0, 6.0, 3.0, 6.0, 6.0, 8.0, 7.0, 5.0, 3.0, 3.0, 3.0, 6.0, 4.0, 7.0, 6.0, 5.0, 5.0, 7.0, 3.0, 4.0, 4.0, 5.0, 4.0, 3.0, 4.0]

##### Ground Truths: [4, 0, 2, 6, 6, 4, 6, 8, 8, 3, 4, 3, 8, 6, 8, 1, 8, 6, 5, 5, 3, 4, 1, 3, 4, 4, 5, 0, 4, 5, 4, 5, 2, 5, 7, 2, 7, 9, 0, 4, 4, 4, 1, 7, 7, 2, 4, 9, 7, 5, 6, 6, 3, 2, 6, 0, 6, 2, 5, 4, 4, 3, 4, 4, 6, 2, 8, 5, 7, 6, 0, 6, 6, 4, 6, 4, 2, 2, 4, 4, 2, 0, 2, 6, 2, 8, 9, 7, 4, 6, 6, 10, 2, 4, 0, 9, 6, 0, 0, 4, 3, 6, 4, 4, 4, 2, 0, 0, 5, 6, 4, 3, 4, 11, 10, 6, 0, 2, 8, 10, 5, 4, 5, 9, 5, 4, 2, 2, 6, 6, 9, 4, 5, 3, 10, 1, 1, 9, 1, 4, 2, 2, 9, 2, 6, 9, 4, 4, 2, 6, 4, 6, 8, 3, 4, 2, 3, 0, 9, 7, 1, 3, 4, 6, 4, 2, 3, 4, 0, 7, 3, 9, 9, 8, 5, 6, 4, 6, 8, 1, 6, 0, 3, 8, 8, 9, 2, 6, 3, 6, 0, 4, 7, 2, 3, 5, 4, 5, 5, 4, 4, 8, 7, 2, 4, 2, 3, 8, 4, 4, 0, 8, 5, 5, 6, 7, 6, 4, 11, 6, 4, 6, 5, 6, 2, 7, 4, 9, 6, 1, 3, 2, 4, 4, 2, 8, 0, 0, 6, 8, 4, 4, 4, 4, 9, 4, 4, 2, 6, 4, 3, 6, 4, 6, 3, 3, 2, 4, 10, 4, 4, 6, 6, 4, 3, 8, 8, 3, 2, 4, 2, 6, 5, 5, 4, 7, 2, 6, 8, 9, 3, 6, 8, 4, 4, 1, 4, 5, 2, 4, 6, 4, 8, 3, 4, 0, 4, 4, 1, 6, 3, 6, 8, 6, 8, 4, 4, 2, 7, 6, 9, 4, 4, 6, 6, 4, 5, 8, 8, 9, 3, 9, 2, 6, 3, 9, 4, 0, 0, 5, 0, 6, 7, 0, 3, 4, 3, 6, 3, 0, 5, 0, 6, 7, 2, 4, 2, 4, 6, 9, 6, 3, 5, 4, 4, 10, 5, 4, 4, 4, 6, 7, 5, 5, 6, 2, 7, 8, 0, 4, 4, 2, 2, 6, 9, 0, 3, 8, 11, 6, 5, 4, 0, 4, 4, 2, 11, 7, 4, 8, 6, 6, 6, 3, 4, 0, 5, 4, 2, 2, 8, 1, 8, 7, 6, 5, 7, 3, 3, 4, 0, 6, 5, 4, 4, 4, 3, 11, 4, 4, 9, 4, 5, 5, 5, 3, 6, 7, 6, 4, 3, 4, 8, 6, 6, 6, 1, 10, 7, 12, 4, 6, 6, 5, 6, 7, 4, 6, 7, 5, 6, 7, 5, 4, 8, 6, 4, 6, 8, 4, 2, 6, 2, 7, 7, 6, 3, 1, 4, 13, 8, 6, 10, 2, 5, 6, 9, 2, 5, 5, 2, 10, 4, 2, 3, 8, 6, 6, 2, 5, 7, 10, 4, 4, 2, 9, 4, 7, 9, 8, 2, 4, 8, 3, 4, 4, 0, 2, 5, 6, 7, 8, 2, 9, 6, 2, 2, 10, 0, 4, 3, 11, 3, 4, 4, 5, 2, 8, 0, 5, 7, 0, 4, 3, 2, 11, 2, 2, 2, 5, 0, 2, 4, 4, 8, 6, 5, 1, 2, 7, 3, 4, 2, 6, 2, 4, 2, 2, 7, 4, 9, 3, 7, 6, 6, 4, 2, 4, 6, 4, 4, 10, 3, 6, 8, 5, 4, 4, 8, 3, 2, 11, 2, 11, 6, 10, 2, 3, 0, 6, 5, 0, 4, 2, 4, 4, 7, 1, 4, 4, 9, 6, 12, 4, 6, 4, 6, 5, 4, 2, 1, 2, 4, 2, 0, 2, 4, 7, 4, 6, 3, 2, 8, 10, 2, 2, 4, 2, 9, 3, 6, 5, 9, 4, 5, 4, 2, 6, 0, 4, 2, 6, 6, 2, 2, 4, 2, 8, 7, 9, 7, 9, 2, 4, 2, 2, 4, 4, 4, 5, 4, 6, 3, 4, 5, 4, 4, 5, 0, 0, 4, 9, 8, 4, 3, 6, 0, 7, 4, 11, 4, 4, 6, 5, 2, 4, 7, 2, 2, 9, 2, 6, 9, 2, 2, 4, 5, 4, 7, 8, 5, 6, 8, 7, 4, 6, 9, 3, 4, 4, 4, 3, 4, 4, 11, 2, 1, 2, 4, 4, 3, 4, 1, 3, 3, 0, 2, 4, 8, 2, 4, 2, 8, 9, 2, 4, 0, 4, 8, 9, 8, 7, 5, 5, 2, 2, 1, 6, 9, 6, 7, 7, 4, 7, 4, 3, 4, 4, 0, 2, 4, 7, 2, 9, 3, 7, 4, 4, 4, 6, 5, 3, 4, 6, 5, 1, 4, 2, 5, 3, 3, 4, 6, 5, 4, 6, 2, 11, 2, 2, 4, 4, 4, 7, 5, 4, 3, 7, 5, 4, 6, 3, 9, 14, 2, 0, 2, 2, 4, 4, 6, 4, 4, 4, 9, 8, 1, 2, 2, 6, 7, 6, 9, 4, 0, 7, 2, 11, 6, 4, 2, 10, 5, 7, 7, 7, 6, 4, 2, 4, 8, 5, 0, 4, 5, 4, 4, 7, 9, 7, 5, 4, 6, 10, 3, 4, 6, 0, 4, 6, 3, 6, 4, 2, 3, 3, 0, 4, 6, 4, 5, 4, 7, 5, 6, 4, 3, 6, 10, 4, 5, 2, 3, 5, 6, 6, 2, 7, 2, 2, 4, 4, 0, 4, 2, 5, 4, 5, 1, 4, 3, 6, 5, 9, 4, 2, 0, 4, 5, 10, 8, 4, 6, 8, 3, 8, 4, 6, 8, 2, 8, 9, 2, 4, 5, 2, 4, 3, 2, 4, 10, 4, 5, 0, 6, 9, 2, 6, 6, 6, 6, 3, 4, 8, 0, 4, 8, 1, 6, 6, 2, 3, 9, 3, 8, 9, 0, 2, 0, 4, 6, 2, 4, 4, 6, 7, 2, 9, 5, 0, 4, 0, 8, 2, 9, 4, 6, 3, 6, 6, 8, 7, 5, 3, 3, 3, 6, 4, 7, 6, 5, 5, 7, 3, 4, 4, 5, 4, 3, 4]
##### Test Results: RMSE - 0.12649110640673517, MAE: 0.008, Num Gt: 996/1000

