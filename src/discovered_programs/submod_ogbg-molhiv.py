
import math

import numpy as np


def priority_v1(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1` using numpy for efficiency."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = np.zeros((max_node, max_node))
    degrees1 = np.array([sum(row) for row in graph1])
    degrees2 = np.array([sum(row) for row in graph2])
    graph1_np = np.array(graph1)
    graph2_np = np.array(graph2)
    weights_np = np.array(weights)

    for i in range(n1):
        for j in range(n2):
            neighbor_similarity = np.sum(graph1_np[i, :][:, np.newaxis] * graph2_np[j, :] * weights_np[:n1, :n2])
            common_neighbors = np.sum(graph1_np[i, :][:, np.newaxis] * graph2_np[j, :])
            if common_neighbors > 0:
                neighbor_similarity /= common_neighbors
            degree_diff = abs(degrees1[i] - degrees2[j])
            similarity = weights[i][j] + neighbor_similarity - 0.1 * degree_diff
            refined_weights[i, j] = 1 / (1 + np.exp(-similarity))

    row_sums = np.sum(refined_weights[:n1, :n2], axis=1, keepdims=True)
    row_sums[row_sums == 0] = n2  # Avoid division by zero
    refined_weights[:n1, :n2] /= row_sums

    col_sums = np.sum(refined_weights[:n1, :n2], axis=0, keepdims=True)
    col_sums[col_sums == 0] = n1  # Avoid division by zero
    refined_weights[:n1, :n2] /= col_sums
    
    return refined_weights.tolist()

def priority_v2(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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


def priority_v3(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`.  Uses sets for neighbor lookup."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]
    
    # Pre-calculate neighbors for efficiency
    neighbors1 = [set(idx for idx, val in enumerate(row) if val == 1) for row in graph1]
    neighbors2 = [set(idx for idx, val in enumerate(row) if val == 1) for row in graph2]
    
    degrees1 = [len(neighbors) for neighbors in neighbors1]
    degrees2 = [len(neighbors) for neighbors in neighbors2]

    for i in range(n1):
        for j in range(n2):
            neighbor_similarity = 0
            common_neighbors = neighbors1[i].intersection(neighbors2[j])
            num_common_neighbors = len(common_neighbors)
            
            for k in neighbors1[i]:
                for l in neighbors2[j]:
                    neighbor_similarity += weights[k][l]
            
            if num_common_neighbors > 0:
                neighbor_similarity /= num_common_neighbors

            degree_diff = abs(degrees1[i] - degrees2[j])
            similarity = weights[i][j] + neighbor_similarity - 0.1 * degree_diff  # Combined similarity
            refined_weights[i][j] = 1 / (1 + math.exp(-similarity)) # Sigmoid normalization

    # Row-wise normalization
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum
        else: # Handle cases where row_sum is zero
            for j in range(n2):
                refined_weights[i][j] = 1 / n2

    return refined_weights


def priority_v4(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [([0.0] * max_node) for _ in range(max_node)]
  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]
  neighbors1 = [set((j for j in range(n1) if graph1[i][j])) for i in range(n1)]
  neighbors2 = [set((j for j in range(n2) if graph2[i][j])) for i in range(n2)]

  for i in range(n1):
    for j in range(n2):
      common_neighbors = neighbors1[i].intersection(neighbors2[j])
      neighbor_similarity = 0
      if common_neighbors:
        for k in neighbors1[i]:
          for l in neighbors2[j]:
            neighbor_similarity += weights[k][l]
        # Normalize by the total number of neighbor pairs considered, not just common neighbors
        neighbor_similarity /= (len(neighbors1[i]) * len(neighbors2[j]))  # Improved normalization

      degree_diff = abs(degrees1[i] - degrees2[j])
      # Scale degree difference by the average degree to make it relative
      avg_degree = (sum(degrees1) + sum(degrees2)) / (n1 + n2) if (n1 + n2) > 0 else 1 # avoid division by zero
      scaled_degree_diff = degree_diff / avg_degree if avg_degree > 0 else degree_diff # avoid division by zero

      # Combine similarity measures with tunable weights
      similarity = weights[i][j] + 0.5 * neighbor_similarity - 0.1 * scaled_degree_diff

      # Use sigmoid function to convert similarity to probability
      refined_weights[i][j] = 1 / (1 + math.exp(-similarity))

  # Normalize rows to ensure they sum to 1
  for i in range(n1):
    row_sum = sum(refined_weights[i][:n2])
    if row_sum > 0:
      for j in range(n2):
        refined_weights[i][j] /= row_sum
    else: # Handle cases where a node has no similar nodes in the other graph.
      for j in range(n2):
        refined_weights[i][j] = 1 / n2

  return refined_weights


def priority_v5(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes initial probabilities using a more robust neighbor comparison.
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

      neighbors1 = [k for k in range(n1) if graph1[i][k]]
      neighbors2 = [l for l in range(n2) if graph2[j][l]]

      neighbor_similarity = 0.0
      if neighbors1 and neighbors2:  # Check if both nodes have neighbors
        for k in neighbors1:
          best_match_similarity = 0
          for l in neighbors2:
            if weights[k][l] > best_match_similarity: # Find the best matching neighbor
                best_match_similarity = weights[k][l]
          neighbor_similarity += best_match_similarity
        neighbor_similarity /= len(neighbors1) # Average over neighbors of node i

      weights[i][j] = (degree_similarity + neighbor_similarity) / 2
  return weights


def priority_v60(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      weights[i][j] = (1.0 / (1.0 + abs((degrees1[i] - degrees2[j]))))
  return weights


def priority_v61(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Improves `priority_v0` by normalizing the weights.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = priority_v60(graph1, graph2, weights)  # Initialize with v0

  # Normalize the weights
  row_sums = [sum(row) for row in weights]
  for i in range(n1):
    if row_sums[i] > 0:
      for j in range(n2):
        weights[i][j] /= row_sums[i]
  return weights


def priority_v6(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Further improves `priority_v1` by considering neighbor similarity.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = priority_v61(graph1, graph2, weights) # Initialize with v1

  for i in range(n1):
    for j in range(n2):
      common_neighbors = 0
      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] and graph2[j][l]:  # If they are neighbors in respective graphs
            common_neighbors += weights[k][l]  # Add the probability of those neighbors being mapped

      weights[i][j] *= (1 + common_neighbors) # Increase weight based on common neighbors
      
  # Renormalize after neighbor similarity update
  row_sums = [sum(row) for row in weights]
  for i in range(n1):
      if row_sums[i] > 0:
          for j in range(n2):
              weights[i][j] /= row_sums[i]


  return weights


def priority_v7(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]
    neighbors1 = [set(idx for idx, val in enumerate(row) if val == 1) for row in graph1]
    neighbors2 = [set(idx for idx, val in enumerate(row) if val == 1) for row in graph2]
    degrees1 = [len(neighbors) for neighbors in neighbors1]
    degrees2 = [len(neighbors) for neighbors in neighbors2]

    for i in range(n1):
        for j in range(n2):
            neighbor_similarity = 0
            common_neighbors = neighbors1[i].intersection(neighbors2[j])
            num_common_neighbors = len(common_neighbors)
            if num_common_neighbors > 0:
                for k in neighbors1[i]:
                    for l in neighbors2[j]:
                        neighbor_similarity += weights[k][l]
                neighbor_similarity /= num_common_neighbors  # Corrected division

            degree_diff = abs(degrees1[i] - degrees2[j])
            avg_degree = (sum(degrees1) + sum(degrees2)) / (n1 + n2) if (n1 + n2) > 0 else 1
            scaled_degree_diff = degree_diff / avg_degree if avg_degree > 0 else degree_diff # Re-introduce scaled degree diff

            similarity = weights[i][j] + neighbor_similarity - 0.1 * scaled_degree_diff # Use scaled degree diff

            refined_weights[i][j] = 1 / (1 + math.exp(-similarity))


    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum
        else:  # Handle cases where row_sum is zero
            for j in range(n2):
                refined_weights[i][j] = 1 / n2

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
      weights[i][j] = 1.0 / (1.0 + degree_diff + neighbor_diff) 
  return weights



def priority_v8(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Normalizes weights for better comparison across nodes.
  """
  weights = priority_v81(graph1, graph2, weights)  # Use v1 as a base
  n1 = len(graph1)
  n2 = len(graph2)

  row_sums = [sum(row) for row in weights[:n1]]
  col_sums = [sum(weights[i][:n2]) for i in range(len(weights))] # consider up to n2 only

  for i in range(n1):
    for j in range(n2):
      if row_sums[i] > 0 and col_sums[j] > 0:  # Avoid division by zero
          weights[i][j] /= (row_sums[i] * col_sums[j])**(0.5) #geometric mean

  return weights


def priority_v9(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`.  Uses numpy for efficiency and avoids redundant calculations."""
    n1 = len(graph1)
    n2 = len(graph2)

    graph1_np = np.array(graph1)
    graph2_np = np.array(graph2)
    weights_np = np.array(weights)

    refined_weights = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            neighbor_similarity = np.sum(graph1_np[i, :].reshape(-1, 1) * graph1_np[i, :].reshape(-1, 1).T * graph2_np[j, :].reshape(1, -1) * graph2_np[j, :].reshape(1, -1).T * weights_np[:n1, :n2])
            degree_diff = abs(np.sum(graph1_np[i]) - np.sum(graph2_np[j]))
            refined_weights[i, j] = (weights[i][j] * (1 + neighbor_similarity)) * math.exp(-degree_diff)

    row_sums = np.sum(refined_weights, axis=1, keepdims=True)
    row_sums[row_sums == 0] = n2  # Avoid division by zero
    refined_weights = refined_weights / row_sums

    refined_weights[row_sums[:,0] == n2] = 1/n2 # Correctly handle cases where row_sum was zero originally

    return refined_weights.tolist()


def priority_v10(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

            # Use a smoother degree similarity function
            degree_similarity = 1 / (1 + degree_diff)  # Or even math.exp(-degree_diff)

            neighbors1 = [k for k in range(n1) if graph1[i][k]]
            neighbors2 = [k for k in range(n2) if graph2[j][k]]
            common_neighbors = 0
            total_neighbors = len(neighbors1) + len(neighbors2)

            for n1_neighbor in neighbors1:
                for n2_neighbor in neighbors2:
                    if 0 <= n1_neighbor < len(weights) and 0 <= n2_neighbor < len(weights[0]) and weights[n1_neighbor][n2_neighbor] > 0:
                        common_neighbors += weights[n1_neighbor][n2_neighbor]

            if total_neighbors > 0:
                neighborhood_similarity = (2 * common_neighbors) / total_neighbors
            else:
                neighborhood_similarity = 1 if (degree1 == 0 and degree2 == 0) else 0

            if 0 <= i < len(weights) and 0 <= j < len(weights[0]):
                # Give more weight to neighborhood similarity if degrees are similar
                if degree_similarity > 0.5: # Threshold for similar degrees
                    refined_weights[i][j] = (0.4 * degree_similarity + 0.5 * neighborhood_similarity + 0.1 * weights[i][j])
                else:
                    refined_weights[i][j] = (0.3 * degree_similarity + 0.4 * neighborhood_similarity + 0.3 * weights[i][j])

            else:  # Handle cases where initial weights are not available for all nodes
                refined_weights[i][j] = (degree_similarity + neighborhood_similarity) / 2


    return refined_weights


def priority_v11(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`.
  This version pre-calculates the neighbors for efficiency and uses numpy for faster calculations.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = np.zeros((max_node, max_node))
  neighbors1 = [set(j for j in range(n1) if graph1[i][j]) for i in range(n1)]
  neighbors2 = [set(j for j in range(n2) if graph2[i][j]) for i in range(n2)]

  for i in range(n1):
    for j in range(n2):
      common_neighbors = 0
      for neighbor_i in neighbors1[i]:
        for neighbor_j in neighbors2[j]:
          if any(graph2[j][l] and graph1[neighbor_i][l] for l in range(n2)):
            common_neighbors += 1
            break # Optimization: Once a common neighbor is found, move to the next neighbor_i

      if weights:
        refined_weights[i, j] = weights[i][j] * (1 + common_neighbors)
      else:
        refined_weights[i, j] = 1 + common_neighbors

  row_sums = refined_weights[:, :n2].sum(axis=1, keepdims=True)
  row_sums[row_sums == 0] = 1  # Avoid division by zero
  refined_weights[:, :n2] /= row_sums

  return refined_weights.tolist() # Convert back to list of lists



def priority_v12(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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

      weights[i][j] = (degree_similarity + neighbor_similarity) / 2 # Averaging both similarities
  return weights

def priority_v131(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
      neighbor_diff = sum(abs(a - b) for a in neighbor_degrees1[i] for b in neighbor_degrees2[j]) / (len(neighbor_degrees1[i]) * len(neighbor_degrees2[j]) + 1e-6)  # Avoid division by zero
      weights[i][j] = 1.0 / (1.0 + degree_diff + neighbor_diff)
  return weights


def priority_v13(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Normalizes probabilities to sum to 1 for each node in graph1.
  """
  weights = priority_v131(graph1, graph2, weights)  # Use v1 as a base
  n1 = len(graph1)
  n2 = len(graph2)
  for i in range(n1):
    row_sum = sum(weights[i][:n2])  # Only consider valid indices for graph2
    if row_sum > 0:
      for j in range(n2):
        weights[i][j] /= row_sum
  return weights


def priority_v14(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [([0.0] * max_node) for _ in range(max_node)]
  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
      neighbor_similarity = 0
      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] == 1 and graph2[j][l] == 1:
            neighbor_similarity += weights[k][l]

      degree_diff = abs(degrees1[i] - degrees2[j])
      similarity = (weights[i][j] + neighbor_similarity) - (0.1 * degree_diff)
      refined_weights[i][j] = 1 / (1 + math.exp(-similarity))

  # Normalize row-wise and handle cases where row sum is zero
  for i in range(n1):
    row_sum = sum(refined_weights[i][:n2])
    if row_sum > 0:
      for j in range(n2):
        refined_weights[i][j] /= row_sum
    else:
      for j in range(n2):
        refined_weights[i][j] = 1 / n2

  # Normalize column-wise (this is the improvement over v1)
  for j in range(n2):
    col_sum = sum(refined_weights[i][j] for i in range(n1))
    if col_sum > 0:
      for i in range(n1):
        refined_weights[i][j] /= col_sum
    else:  # Should ideally not happen after row normalization, but handle just in case
      for i in range(n1):
        refined_weights[i][j] = 1 / n1

  return refined_weights

def priority_v15(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Computes probabilities using degree difference and neighborhood structure similarity.
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
          degree_similarity = 1.0 / (1.0 + degree_diff)


          neighbors1 = [k for k in range(n1) if graph1[i][k]]
          neighbors2 = [l for l in range(n2) if graph2[j][l]]

          neighborhood_similarity = 0
          for n1_neighbor in neighbors1:
              for n2_neighbor in neighbors2:
                  neighborhood_similarity = max(neighborhood_similarity, weights[n1_neighbor][n2_neighbor] if 0<=n1_neighbor<max_node and 0<= n2_neighbor<max_node else 0) # Consider the most similar neighbor


          weights[i][j] = (degree_similarity + neighborhood_similarity) / 2


  return weights
# Input Scores: [40.0, 22.0, 23.0, 32.0, 55.0, 24.0, 58.0, 27.0, 46.0, 56.0, 30.0, 31.0, 46.0, 23.0, 23.0, 31.0, 47.0, 32.0, 24.0, 20.0, 79.0, 20.0, 32.0, 65.0, 22.0, 20.0, 36.0, 44.0, 36.0, 56.0, 31.0, 37.0, 43.0, 63.0, 20.0, 35.0, 72.0, 50.0, 21.0, 30.0, 43.0, 49.0, 37.0, 34.0, 40.0, 27.0, 32.0, 26.0, 21.0, 63.0, 27.0, 24.0, 25.0, 25.0, 47.0, 28.0, 76.0, 23.0, 42.0, 21.0, 34.0, 26.0, 31.0, 51.0, 40.0, 10.0, 43.0, 22.0, 31.0, 28.0, 43.0, 26.0, 35.0, 48.0, 23.0, 58.0, 30.0, 43.0, 37.0, 31.0, 29.0, 45.0, 38.0, 48.0, 43.0, 81.0, 35.0, 33.0, 38.0, 37.0, 43.0, 29.0, 37.0, 23.0, 24.0, 26.0, 21.0, 36.0, 34.0, 23.0, 25.0, 24.0, 31.0, 40.0, 33.0, 49.0, 51.0, 33.0, 15.0, 18.0, 20.0, 28.0, 36.0, 62.0, 23.0, 30.0, 15.0, 42.0, 28.0, 20.0, 30.0, 25.0, 43.0, 35.0, 27.0, 56.0, 69.0, 36.0, 17.0, 29.0, 21.0, 28.0, 29.0, 47.0, 27.0, 24.0, 23.0, 30.0, 20.0, 34.0, 34.0, 31.0, 42.0, 34.0, 26.0, 40.0, 27.0, 21.0, 39.0, 35.0, 17.0, 25.0, 23.0, 32.0, 45.0, 64.0, 33.0, 16.0, 45.0, 39.0, 38.0, 59.0, 24.0, 23.0, 34.0, 24.0, 83.0, 31.0, 11.0, 20.0, 40.0, 45.0, 23.0, 45.0, 21.0, 29.0, 38.0, 67.0, 37.0, 31.0, 63.0, 63.0, 25.0, 37.0, 42.0, 20.0, 65.0, 71.0, 60.0, 12.0, 19.0, 34.0, 20.0, 62.0, 63.0, 19.0, 45.0, 20.0, 30.0, 34.0, 24.0, 37.0, 30.0, 18.0, 27.0, 30.0, 24.0, 44.0, 20.0, 36.0, 10.0, 24.0, 38.0, 34.0, 20.0, 32.0, 45.0, 44.0, 49.0, 30.0, 25.0, 29.0, 36.0, 38.0, 43.0, 83.0, 42.0, 25.0, 29.0, 23.0, 17.0, 39.0, 23.0, 43.0, 30.0, 41.0, 28.0, 68.0, 49.0, 40.0, 51.0, 21.0, 66.0, 20.0, 41.0, 30.0, 66.0, 23.0, 19.0, 30.0, 56.0, 66.0, 44.0, 26.0, 53.0, 36.0, 28.0, 32.0, 25.0, 29.0, 17.0, 20.0, 21.0, 19.0, 36.0, 63.0, 18.0, 29.0, 26.0, 34.0, 37.0, 45.0, 32.0, 22.0, 55.0, 34.0, 30.0, 30.0, 12.0, 42.0, 71.0, 25.0, 45.0, 41.0, 27.0, 34.0, 29.0, 49.0, 62.0, 34.0, 31.0, 33.0, 25.0, 50.0, 28.0, 48.0, 48.0, 51.0, 64.0, 32.0, 20.0, 53.0, 30.0, 46.0, 77.0, 54.0, 25.0, 63.0, 37.0, 35.0, 24.0, 54.0, 31.0, 60.0, 60.0, 74.0, 19.0, 44.0, 20.0, 68.0, 20.0, 37.0, 31.0, 26.0, 39.0, 50.0, 30.0, 36.0, 62.0, 39.0, 28.0, 63.0, 35.0, 69.0, 22.0, 25.0, 51.0, 31.0, 34.0, 30.0, 42.0, 16.0, 30.0, 35.0, 38.0, 39.0, 27.0, 36.0, 18.0, 29.0, 72.0, 30.0, 25.0, 38.0, 27.0, 34.0, 38.0, 28.0, 31.0, 32.0, 35.0, 73.0, 36.0, 34.0, 53.0, 24.0, 32.0, 41.0, 25.0, 22.0, 37.0, 26.0, 34.0, 47.0, 48.0, 49.0, 24.0, 29.0, 38.0, 53.0, 30.0, 23.0, 48.0, 15.0, 30.0, 38.0, 51.0, 29.0, 32.0, 19.0, 51.0, 30.0, 32.0, 26.0, 18.0, 32.0, 55.0, 44.0, 62.0, 35.0, 41.0, 29.0, 15.0, 15.0, 53.0, 73.0, 29.0, 43.0, 23.0, 79.0, 35.0, 25.0, 28.0, 38.0, 22.0, 31.0, 27.0, 30.0, 27.0, 35.0, 24.0, 20.0, 21.0, 26.0, 68.0, 24.0, 26.0, 35.0, 17.0, 19.0, 65.0, 29.0, 25.0, 53.0, 31.0, 18.0, 24.0, 20.0, 38.0, 54.0, 16.0, 20.0, 28.0, 38.0, 24.0, 16.0, 57.0, 34.0, 21.0, 55.0, 20.0, 34.0, 29.0, 66.0, 27.0, 23.0, 82.0, 34.0, 25.0, 31.0, 11.0, 20.0, 15.0, 26.0, 31.0, 39.0, 26.0, 20.0, 42.0, 37.0, 39.0, 31.0, 23.0, 24.0, 22.0, 25.0, 54.0, 37.0, 32.0, 52.0, 21.0, 42.0, 32.0, 39.0, 29.0, 22.0, 22.0, 76.0, 20.0, 24.0, 43.0, 39.0, 75.0, 59.0, 31.0, 62.0, 28.0, 31.0, 56.0, 25.0, 19.0, 22.0, 30.0, 24.0, 38.0, 53.0, 61.0, 24.0, 36.0, 23.0, 52.0, 32.0, 40.0, 36.0, 32.0, 41.0, 26.0, 27.0, 31.0, 66.0, 27.0, 33.0, 61.0, 24.0, 35.0, 41.0, 43.0, 22.0, 28.0, 28.0, 45.0, 34.0, 25.0, 36.0, 68.0, 32.0, 37.0, 38.0, 33.0, 47.0, 45.0, 20.0, 37.0, 57.0, 20.0, 75.0, 27.0, 22.0, 35.0, 19.0, 27.0, 62.0, 51.0, 35.0, 53.0, 47.0, 19.0, 89.0, 28.0, 24.0, 56.0, 17.0, 56.0, 35.0, 41.0, 27.0, 22.0, 54.0, 44.0, 24.0, 21.0, 60.0, 31.0, 26.0, 63.0, 22.0, 18.0, 89.0, 29.0, 51.0, 36.0, 57.0, 25.0, 24.0, 53.0, 52.0, 41.0, 22.0, 30.0, 63.0, 16.0, 62.0, 14.0, 25.0, 27.0, 49.0, 28.0, 16.0, 53.0, 33.0, 56.0, 29.0, 21.0, 25.0, 26.0, 31.0, 32.0, 29.0, 55.0, 28.0, 61.0, 44.0, 34.0, 70.0, 35.0, 26.0, 40.0, 63.0, 31.0, 27.0, 38.0, 43.0, 26.0, 31.0, 44.0, 67.0, 30.0, 34.0, 20.0, 24.0, 38.0, 27.0, 21.0, 44.0, 35.0, 33.0, 24.0, 27.0, 28.0, 23.0, 11.0, 61.0, 20.0, 60.0, 19.0, 38.0, 32.0, 19.0, 31.0, 25.0, 37.0, 25.0, 42.0, 53.0, 29.0, 52.0, 30.0, 30.0, 24.0, 20.0, 54.0, 24.0, 25.0, 20.0, 46.0, 12.0, 42.0, 42.0, 34.0, 32.0, 19.0, 37.0, 39.0, 24.0, 29.0, 45.0, 24.0, 22.0, 22.0, 23.0, 26.0, 37.0, 37.0, 25.0, 77.0, 32.0, 34.0, 35.0, 51.0, 32.0, 24.0, 19.0, 75.0, 37.0, 58.0, 30.0, 12.0, 79.0, 67.0, 25.0, 24.0, 23.0, 61.0, 24.0, 49.0, 68.0, 57.0, 36.0, 53.0, 18.0, 35.0, 38.0, 45.0, 37.0, 24.0, 43.0, 46.0, 45.0, 28.0, 29.0, 21.0, 43.0, 28.0, 68.0, 26.0, 76.0, 24.0, 40.0, 27.0, 57.0, 34.0, 45.0, 42.0, 67.0, 25.0, 32.0, 31.0, 18.0, 24.0, 13.0, 22.0, 30.0, 14.0, 28.0, 28.0, 34.0, 35.0, 23.0, 58.0, 70.0, 28.0, 37.0, 30.0, 15.0, 75.0, 23.0, 34.0, 31.0, 39.0, 11.0, 31.0, 41.0, 43.0, 30.0, 28.0, 26.0, 27.0, 24.0, 24.0, 34.0, 39.0, 69.0, 22.0, 43.0, 29.0, 41.0, 29.0, 19.0, 33.0, 46.0, 41.0, 27.0, 44.0, 33.0, 27.0, 31.0, 14.0, 41.0, 20.0, 45.0, 35.0, 57.0, 45.0, 38.0, 19.0, 22.0, 19.0, 77.0, 40.0, 53.0, 23.0, 39.0, 29.0, 44.0, 55.0, 43.0, 33.0, 30.0, 19.0, 26.0, 12.0, 43.0, 35.0, 38.0, 11.0, 28.0, 36.0, 31.0, 25.0, 34.0, 51.0, 33.0, 30.0, 56.0, 34.0, 69.0, 26.0, 16.0, 17.0, 23.0, 33.0, 21.0, 65.0, 13.0, 19.0, 28.0, 65.0, 17.0, 25.0, 31.0, 35.0, 40.0, 28.0, 25.0, 20.0, 29.0, 39.0, 70.0, 25.0, 30.0, 22.0, 62.0, 49.0, 13.0, 49.0, 25.0, 30.0, 21.0, 36.0, 30.0, 22.0, 38.0, 36.0, 26.0, 50.0, 25.0, 46.0, 39.0, 26.0, 24.0, 42.0, 25.0, 72.0, 41.0, 45.0, 20.0, 35.0, 54.0, 27.0, 32.0, 42.0, 43.0, 23.0, 28.0, 36.0, 19.0, 29.0, 32.0, 29.0, 33.0, 24.0, 56.0, 73.0, 46.0, 54.0, 49.0, 53.0, 27.0, 25.0, 30.0, 40.0, 45.0, 26.0, 27.0, 18.0, 26.0, 56.0, 56.0, 19.0, 22.0, 28.0, 34.0, 29.0, 67.0, 33.0, 30.0, 21.0, 44.0, 35.0, 28.0, 28.0, 14.0, 71.0, 22.0, 43.0, 34.0, 57.0, 28.0, 37.0, 58.0, 26.0, 36.0, 32.0, 20.0, 19.0, 15.0, 62.0, 29.0, 30.0, 41.0, 27.0, 32.0, 42.0, 68.0, 24.0, 21.0, 8.0, 12.0, 10.0, 36.0, 33.0, 24.0, 41.0, 21.0, 20.0, 49.0, 22.0, 42.0, 81.0, 49.0, 51.0, 34.0, 30.0, 58.0, 21.0, 25.0, 31.0, 50.0, 21.0, 43.0, 26.0, 27.0, 51.0, 34.0, 27.0, 50.0, 28.0, 33.0, 19.0, 29.0, 34.0, 61.0, 27.0, 25.0, 22.0, 44.0, 70.0, 20.0, 26.0, 17.0, 20.0, 44.0, 45.0, 40.0, 26.0, 17.0, 36.0, 60.0, 37.0, 30.0, 24.0, 45.0, 25.0, 28.0, 28.0, 56.0, 27.0, 55.0, 28.0, 32.0, 17.0, 21.0, 23.0, 36.0, 44.0, 36.0, 31.0, 30.0, 17.0, 28.0, 60.0, 43.0, 21.0, 31.0, 58.0, 63.0, 23.0, 32.0, 29.0, 51.0, 48.0, 26.0, 33.0, 39.0, 19.0, 50.0, 53.0, 32.0, 31.0, 21.0, 25.0, 77.0, 29.0, 27.0, 58.0, 49.0, 49.0, 33.0, 35.0, 27.0, 28.0, 50.0, 31.0, 24.0, 24.0, 33.0, 45.0, 62.0, 71.0, 39.0, 32.0, 56.0, 40.0, 34.0, 47.0, 29.0, 35.0, 60.0, 35.0, 37.0, 17.0, 43.0, 26.0, 15.0, 14.0, 22.0, 18.0, 35.0, 26.0, 21.0, 31.0, 28.0]

##### Best Scores: [36.0, 13.0, 14.0, 28.0, 52.0, 19.0, 55.0, 21.0, 40.0, 53.0, 30.0, 28.0, 36.0, 20.0, 19.0, 31.0, 45.0, 31.0, 22.0, 15.0, 77.0, 20.0, 26.0, 64.0, 17.0, 16.0, 26.0, 43.0, 34.0, 48.0, 29.0, 28.0, 35.0, 58.0, 13.0, 27.0, 68.0, 48.0, 15.0, 28.0, 33.0, 43.0, 33.0, 30.0, 37.0, 25.0, 26.0, 20.0, 16.0, 61.0, 24.0, 15.0, 25.0, 20.0, 34.0, 26.0, 69.0, 18.0, 33.0, 19.0, 32.0, 24.0, 25.0, 40.0, 33.0, 10.0, 32.0, 12.0, 27.0, 27.0, 36.0, 19.0, 31.0, 45.0, 16.0, 52.0, 24.0, 34.0, 29.0, 30.0, 24.0, 43.0, 36.0, 41.0, 41.0, 77.0, 31.0, 24.0, 33.0, 31.0, 34.0, 21.0, 31.0, 20.0, 20.0, 23.0, 16.0, 25.0, 24.0, 20.0, 16.0, 23.0, 27.0, 27.0, 28.0, 44.0, 47.0, 29.0, 9.0, 18.0, 16.0, 26.0, 32.0, 59.0, 19.0, 23.0, 15.0, 40.0, 25.0, 17.0, 25.0, 20.0, 40.0, 25.0, 24.0, 50.0, 67.0, 31.0, 14.0, 23.0, 19.0, 24.0, 29.0, 47.0, 19.0, 20.0, 18.0, 29.0, 17.0, 28.0, 31.0, 31.0, 38.0, 25.0, 20.0, 32.0, 19.0, 16.0, 35.0, 30.0, 17.0, 18.0, 13.0, 25.0, 44.0, 56.0, 30.0, 14.0, 37.0, 32.0, 29.0, 53.0, 17.0, 19.0, 26.0, 17.0, 73.0, 24.0, 11.0, 14.0, 33.0, 42.0, 19.0, 30.0, 21.0, 26.0, 35.0, 63.0, 30.0, 27.0, 55.0, 54.0, 23.0, 25.0, 37.0, 20.0, 55.0, 66.0, 52.0, 9.0, 18.0, 31.0, 20.0, 62.0, 60.0, 19.0, 45.0, 17.0, 25.0, 33.0, 20.0, 37.0, 26.0, 9.0, 25.0, 28.0, 20.0, 40.0, 14.0, 31.0, 7.0, 20.0, 37.0, 34.0, 18.0, 28.0, 42.0, 40.0, 41.0, 23.0, 20.0, 16.0, 32.0, 28.0, 40.0, 77.0, 41.0, 19.0, 28.0, 21.0, 5.0, 35.0, 19.0, 28.0, 28.0, 35.0, 27.0, 66.0, 43.0, 33.0, 43.0, 19.0, 64.0, 16.0, 35.0, 29.0, 58.0, 19.0, 15.0, 22.0, 54.0, 56.0, 40.0, 17.0, 49.0, 35.0, 26.0, 27.0, 19.0, 26.0, 15.0, 14.0, 17.0, 18.0, 31.0, 60.0, 17.0, 21.0, 20.0, 28.0, 30.0, 39.0, 31.0, 20.0, 46.0, 25.0, 25.0, 28.0, 10.0, 27.0, 67.0, 22.0, 38.0, 34.0, 25.0, 22.0, 24.0, 45.0, 61.0, 30.0, 27.0, 21.0, 23.0, 47.0, 21.0, 40.0, 34.0, 47.0, 61.0, 27.0, 15.0, 47.0, 30.0, 42.0, 72.0, 52.0, 19.0, 60.0, 30.0, 28.0, 15.0, 52.0, 31.0, 47.0, 53.0, 74.0, 13.0, 42.0, 17.0, 68.0, 15.0, 28.0, 27.0, 20.0, 34.0, 49.0, 30.0, 35.0, 48.0, 35.0, 25.0, 59.0, 33.0, 60.0, 16.0, 25.0, 50.0, 29.0, 22.0, 22.0, 37.0, 13.0, 28.0, 24.0, 30.0, 36.0, 27.0, 28.0, 18.0, 26.0, 65.0, 25.0, 25.0, 31.0, 24.0, 31.0, 33.0, 24.0, 29.0, 31.0, 29.0, 72.0, 29.0, 22.0, 48.0, 23.0, 25.0, 34.0, 19.0, 19.0, 33.0, 23.0, 32.0, 40.0, 39.0, 47.0, 18.0, 23.0, 37.0, 46.0, 24.0, 18.0, 43.0, 15.0, 25.0, 28.0, 47.0, 14.0, 31.0, 16.0, 41.0, 30.0, 30.0, 25.0, 13.0, 20.0, 53.0, 38.0, 60.0, 25.0, 41.0, 29.0, 10.0, 15.0, 52.0, 71.0, 23.0, 34.0, 20.0, 77.0, 26.0, 19.0, 22.0, 38.0, 22.0, 27.0, 23.0, 30.0, 22.0, 30.0, 20.0, 13.0, 16.0, 19.0, 65.0, 21.0, 20.0, 30.0, 10.0, 16.0, 63.0, 27.0, 24.0, 43.0, 27.0, 18.0, 24.0, 13.0, 32.0, 52.0, 15.0, 16.0, 24.0, 33.0, 21.0, 14.0, 49.0, 25.0, 20.0, 49.0, 14.0, 25.0, 25.0, 66.0, 24.0, 22.0, 80.0, 32.0, 18.0, 24.0, 9.0, 17.0, 13.0, 24.0, 29.0, 38.0, 16.0, 13.0, 42.0, 31.0, 27.0, 28.0, 21.0, 20.0, 18.0, 24.0, 53.0, 32.0, 27.0, 47.0, 18.0, 36.0, 25.0, 39.0, 29.0, 16.0, 22.0, 74.0, 13.0, 18.0, 38.0, 31.0, 71.0, 57.0, 25.0, 61.0, 24.0, 25.0, 56.0, 23.0, 9.0, 20.0, 26.0, 20.0, 34.0, 47.0, 57.0, 19.0, 32.0, 22.0, 48.0, 30.0, 32.0, 30.0, 23.0, 40.0, 16.0, 19.0, 21.0, 64.0, 26.0, 30.0, 54.0, 21.0, 23.0, 35.0, 37.0, 18.0, 16.0, 21.0, 36.0, 27.0, 19.0, 34.0, 63.0, 29.0, 29.0, 34.0, 30.0, 43.0, 40.0, 17.0, 32.0, 48.0, 20.0, 74.0, 25.0, 15.0, 30.0, 19.0, 25.0, 56.0, 47.0, 24.0, 50.0, 45.0, 17.0, 81.0, 20.0, 20.0, 49.0, 16.0, 54.0, 17.0, 35.0, 16.0, 16.0, 43.0, 37.0, 24.0, 21.0, 57.0, 26.0, 18.0, 63.0, 20.0, 15.0, 84.0, 23.0, 48.0, 34.0, 49.0, 21.0, 17.0, 53.0, 49.0, 27.0, 19.0, 25.0, 62.0, 12.0, 61.0, 11.0, 21.0, 18.0, 47.0, 27.0, 14.0, 45.0, 24.0, 52.0, 21.0, 18.0, 23.0, 24.0, 24.0, 27.0, 27.0, 48.0, 20.0, 53.0, 34.0, 27.0, 70.0, 30.0, 19.0, 40.0, 56.0, 29.0, 23.0, 35.0, 42.0, 23.0, 30.0, 42.0, 56.0, 22.0, 31.0, 15.0, 15.0, 34.0, 22.0, 19.0, 33.0, 31.0, 33.0, 21.0, 23.0, 21.0, 13.0, 6.0, 59.0, 15.0, 50.0, 13.0, 36.0, 32.0, 15.0, 27.0, 25.0, 22.0, 21.0, 32.0, 50.0, 22.0, 45.0, 27.0, 24.0, 21.0, 18.0, 52.0, 20.0, 20.0, 17.0, 39.0, 12.0, 39.0, 36.0, 30.0, 29.0, 15.0, 35.0, 37.0, 24.0, 22.0, 43.0, 17.0, 15.0, 16.0, 20.0, 24.0, 37.0, 31.0, 11.0, 74.0, 29.0, 34.0, 24.0, 45.0, 28.0, 19.0, 13.0, 75.0, 32.0, 51.0, 22.0, 10.0, 69.0, 63.0, 16.0, 18.0, 17.0, 55.0, 20.0, 49.0, 62.0, 46.0, 29.0, 46.0, 14.0, 32.0, 36.0, 38.0, 32.0, 24.0, 41.0, 39.0, 42.0, 24.0, 24.0, 15.0, 42.0, 23.0, 66.0, 26.0, 68.0, 20.0, 36.0, 20.0, 48.0, 28.0, 41.0, 37.0, 57.0, 21.0, 27.0, 31.0, 13.0, 17.0, 8.0, 17.0, 26.0, 8.0, 20.0, 28.0, 29.0, 29.0, 17.0, 51.0, 69.0, 25.0, 30.0, 30.0, 11.0, 73.0, 19.0, 27.0, 26.0, 32.0, 10.0, 29.0, 33.0, 37.0, 28.0, 28.0, 18.0, 18.0, 23.0, 17.0, 32.0, 30.0, 61.0, 19.0, 33.0, 29.0, 32.0, 25.0, 14.0, 25.0, 37.0, 35.0, 27.0, 39.0, 24.0, 22.0, 30.0, 11.0, 39.0, 16.0, 43.0, 35.0, 50.0, 31.0, 38.0, 18.0, 21.0, 19.0, 72.0, 33.0, 50.0, 18.0, 35.0, 25.0, 34.0, 55.0, 38.0, 29.0, 23.0, 14.0, 18.0, 12.0, 41.0, 24.0, 31.0, 11.0, 20.0, 31.0, 25.0, 16.0, 29.0, 45.0, 26.0, 25.0, 46.0, 26.0, 66.0, 23.0, 12.0, 14.0, 22.0, 29.0, 14.0, 62.0, 12.0, 14.0, 21.0, 57.0, 14.0, 21.0, 29.0, 33.0, 36.0, 22.0, 21.0, 19.0, 27.0, 33.0, 69.0, 21.0, 26.0, 19.0, 60.0, 48.0, 9.0, 45.0, 24.0, 21.0, 19.0, 25.0, 26.0, 21.0, 36.0, 33.0, 22.0, 42.0, 18.0, 42.0, 32.0, 22.0, 21.0, 37.0, 18.0, 66.0, 38.0, 42.0, 20.0, 28.0, 43.0, 17.0, 28.0, 29.0, 39.0, 19.0, 25.0, 28.0, 19.0, 24.0, 30.0, 20.0, 30.0, 18.0, 44.0, 68.0, 44.0, 51.0, 47.0, 49.0, 22.0, 24.0, 21.0, 32.0, 41.0, 23.0, 20.0, 17.0, 19.0, 51.0, 49.0, 11.0, 16.0, 28.0, 34.0, 21.0, 66.0, 31.0, 25.0, 14.0, 43.0, 24.0, 18.0, 23.0, 14.0, 70.0, 19.0, 35.0, 34.0, 57.0, 24.0, 29.0, 53.0, 21.0, 27.0, 30.0, 18.0, 18.0, 15.0, 50.0, 17.0, 30.0, 35.0, 22.0, 32.0, 31.0, 61.0, 20.0, 21.0, 8.0, 12.0, 8.0, 32.0, 32.0, 19.0, 40.0, 20.0, 13.0, 48.0, 16.0, 35.0, 78.0, 44.0, 48.0, 29.0, 20.0, 54.0, 9.0, 22.0, 27.0, 37.0, 17.0, 39.0, 21.0, 19.0, 48.0, 26.0, 24.0, 37.0, 28.0, 31.0, 16.0, 24.0, 26.0, 53.0, 19.0, 20.0, 19.0, 43.0, 63.0, 17.0, 22.0, 14.0, 19.0, 40.0, 37.0, 40.0, 26.0, 12.0, 34.0, 54.0, 29.0, 25.0, 21.0, 43.0, 18.0, 23.0, 22.0, 50.0, 25.0, 51.0, 23.0, 28.0, 15.0, 17.0, 20.0, 30.0, 39.0, 30.0, 27.0, 29.0, 15.0, 20.0, 51.0, 32.0, 16.0, 27.0, 49.0, 62.0, 19.0, 25.0, 27.0, 48.0, 38.0, 26.0, 26.0, 28.0, 18.0, 36.0, 46.0, 32.0, 27.0, 17.0, 22.0, 68.0, 27.0, 22.0, 53.0, 47.0, 42.0, 25.0, 26.0, 25.0, 25.0, 47.0, 17.0, 22.0, 20.0, 29.0, 34.0, 56.0, 71.0, 33.0, 21.0, 53.0, 33.0, 31.0, 43.0, 26.0, 32.0, 55.0, 32.0, 37.0, 13.0, 43.0, 18.0, 10.0, 11.0, 19.0, 16.0, 23.0, 20.0, 20.0, 31.0, 22.0]

##### Ground Truths: [36, 10, 13, 22, 46, 18, 51, 21, 38, 51, 24, 27, 32, 20, 18, 29, 41, 29, 22, 14, 75, 14, 25, 63, 14, 15, 22, 40, 31, 45, 25, 26, 32, 55, 11, 27, 66, 44, 15, 21, 28, 43, 32, 28, 36, 25, 22, 19, 14, 61, 19, 13, 24, 18, 29, 26, 65, 17, 29, 19, 31, 22, 22, 36, 32, 10, 26, 12, 26, 22, 34, 17, 26, 44, 13, 50, 24, 31, 25, 27, 20, 36, 32, 37, 41, 74, 27, 24, 26, 27, 33, 17, 30, 16, 20, 21, 16, 24, 20, 16, 15, 20, 25, 26, 26, 43, 45, 21, 9, 17, 13, 25, 31, 56, 15, 20, 13, 40, 24, 14, 20, 20, 37, 21, 19, 47, 62, 26, 12, 23, 17, 20, 25, 47, 19, 18, 16, 27, 16, 26, 31, 31, 33, 22, 16, 30, 17, 16, 32, 30, 14, 17, 13, 21, 40, 55, 28, 14, 35, 32, 26, 53, 16, 19, 23, 15, 72, 23, 11, 10, 31, 37, 19, 28, 21, 25, 34, 61, 30, 26, 54, 52, 21, 23, 36, 19, 52, 66, 47, 9, 17, 29, 16, 58, 59, 15, 41, 15, 23, 31, 19, 34, 24, 8, 24, 27, 19, 39, 14, 31, 7, 16, 35, 31, 14, 23, 41, 40, 41, 22, 18, 16, 31, 28, 37, 75, 36, 16, 24, 18, 5, 34, 19, 25, 25, 32, 25, 66, 43, 31, 41, 16, 64, 12, 34, 24, 58, 16, 15, 22, 50, 52, 37, 17, 44, 30, 26, 20, 16, 26, 13, 12, 17, 16, 30, 60, 15, 21, 19, 25, 27, 38, 28, 19, 43, 22, 25, 26, 10, 23, 66, 18, 36, 30, 22, 19, 19, 42, 57, 28, 25, 20, 19, 42, 17, 37, 33, 46, 60, 27, 13, 45, 26, 39, 71, 49, 18, 58, 27, 26, 12, 50, 29, 39, 53, 74, 13, 42, 17, 62, 14, 26, 26, 20, 31, 47, 29, 32, 47, 34, 23, 55, 33, 58, 15, 22, 48, 26, 17, 21, 36, 11, 23, 24, 27, 32, 23, 27, 16, 22, 63, 20, 18, 26, 19, 29, 32, 22, 29, 30, 28, 68, 27, 19, 44, 20, 22, 33, 15, 17, 33, 19, 32, 33, 34, 40, 15, 22, 36, 45, 23, 17, 43, 14, 20, 25, 43, 13, 27, 14, 38, 28, 29, 22, 12, 16, 52, 33, 57, 23, 41, 29, 8, 15, 47, 64, 19, 34, 18, 76, 26, 17, 15, 37, 20, 23, 23, 29, 20, 25, 19, 12, 14, 17, 63, 17, 16, 27, 10, 14, 54, 24, 20, 41, 22, 17, 22, 13, 27, 51, 13, 13, 21, 29, 19, 12, 48, 24, 18, 48, 12, 25, 21, 63, 24, 16, 78, 24, 15, 20, 9, 15, 11, 22, 28, 38, 14, 13, 42, 31, 24, 26, 17, 20, 16, 22, 52, 31, 25, 42, 18, 31, 20, 33, 23, 14, 16, 74, 11, 17, 38, 31, 69, 57, 21, 60, 22, 22, 51, 20, 9, 17, 21, 17, 31, 47, 56, 19, 32, 18, 46, 28, 29, 28, 21, 38, 14, 19, 18, 63, 25, 26, 52, 16, 19, 33, 36, 16, 15, 21, 36, 26, 19, 32, 61, 26, 25, 28, 27, 43, 39, 15, 30, 46, 17, 73, 23, 15, 30, 13, 23, 53, 44, 22, 49, 43, 17, 73, 20, 18, 49, 16, 50, 16, 34, 16, 16, 41, 35, 21, 18, 56, 22, 17, 61, 18, 15, 82, 23, 45, 31, 48, 19, 16, 50, 47, 23, 17, 24, 62, 9, 58, 11, 16, 18, 46, 26, 13, 43, 21, 52, 21, 16, 19, 22, 24, 25, 23, 44, 18, 50, 30, 23, 64, 28, 19, 37, 53, 26, 18, 35, 39, 20, 27, 42, 55, 19, 31, 15, 15, 31, 21, 17, 30, 29, 32, 21, 21, 20, 13, 6, 59, 14, 46, 13, 34, 30, 15, 27, 22, 21, 21, 29, 48, 22, 42, 26, 22, 16, 12, 50, 15, 19, 14, 37, 7, 37, 31, 26, 25, 13, 32, 37, 17, 20, 40, 14, 14, 16, 20, 20, 36, 28, 11, 73, 26, 32, 21, 40, 28, 19, 11, 74, 28, 49, 20, 10, 65, 63, 15, 18, 15, 49, 17, 47, 59, 46, 28, 40, 13, 30, 35, 37, 31, 21, 35, 38, 38, 16, 24, 13, 36, 19, 64, 22, 66, 19, 34, 20, 43, 25, 40, 33, 53, 20, 24, 28, 13, 13, 8, 16, 22, 7, 19, 24, 29, 27, 14, 48, 64, 22, 29, 30, 11, 71, 17, 27, 23, 30, 8, 28, 29, 37, 24, 28, 17, 18, 21, 14, 30, 28, 61, 19, 31, 25, 31, 23, 11, 23, 33, 30, 26, 39, 24, 20, 25, 11, 38, 16, 40, 35, 45, 31, 34, 17, 15, 17, 70, 31, 49, 15, 33, 18, 31, 53, 37, 27, 18, 12, 17, 12, 40, 22, 30, 10, 19, 30, 19, 16, 26, 43, 25, 23, 46, 19, 65, 16, 10, 10, 19, 25, 12, 59, 12, 12, 17, 56, 13, 19, 29, 30, 35, 20, 18, 15, 27, 31, 63, 20, 23, 16, 57, 45, 7, 44, 19, 19, 19, 21, 24, 19, 31, 33, 18, 40, 18, 40, 30, 21, 16, 35, 13, 64, 38, 42, 20, 22, 40, 16, 25, 25, 36, 18, 21, 27, 18, 24, 25, 15, 24, 18, 34, 66, 44, 45, 45, 45, 17, 23, 18, 30, 32, 20, 20, 17, 19, 46, 45, 11, 16, 25, 34, 21, 61, 27, 24, 12, 37, 21, 15, 22, 13, 69, 19, 34, 31, 54, 22, 24, 52, 18, 24, 27, 14, 18, 10, 45, 17, 27, 33, 20, 26, 31, 58, 17, 19, 8, 10, 7, 30, 29, 18, 35, 18, 10, 46, 16, 29, 76, 36, 47, 28, 20, 49, 9, 21, 26, 36, 16, 39, 20, 15, 47, 25, 20, 36, 25, 28, 14, 21, 22, 50, 19, 19, 15, 43, 61, 17, 20, 13, 15, 34, 33, 40, 21, 9, 31, 50, 25, 23, 18, 41, 16, 18, 18, 50, 22, 51, 20, 26, 13, 16, 19, 25, 39, 28, 19, 26, 13, 20, 49, 24, 15, 27, 47, 59, 17, 20, 27, 45, 35, 25, 24, 26, 14, 34, 46, 29, 24, 12, 17, 66, 27, 18, 51, 44, 39, 21, 26, 22, 23, 46, 15, 22, 20, 29, 34, 54, 67, 29, 19, 48, 30, 27, 40, 23, 32, 54, 29, 34, 12, 41, 16, 7, 9, 16, 16, 22, 19, 17, 27, 18]
##### Test Results: RMSE - 2.878629269201042, MAE: 2.2384473197781887, Num Gt: 215/1082

