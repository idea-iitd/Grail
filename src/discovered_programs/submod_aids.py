import math
import numpy as np


def priority_v1(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`.  Avoids division by zero and uses matrix operations for efficiency."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = np.zeros((max_node, max_node))

    # Convert to numpy for faster operations
    graph1_np = np.array(graph1)
    graph2_np = np.array(graph2)
    weights_np = np.array(weights)

    degrees1 = graph1_np.sum(axis=1)
    degrees2 = graph2_np.sum(axis=1)


    for i in range(n1):
        for j in range(n2):
            node_similarity = weights_np[i, j]
            degree_difference = abs(degrees1[i] - degrees2[j])
            degree_similarity = 1 / (1 + degree_difference) if degree_difference > 0 else 1
            
            neighbor_similarity = (graph1_np[i, :].reshape(1, -1) @ weights_np[:n1, :n2] @ graph2_np[:, j].reshape(-1, 1))[0,0] # Matrix multiplication for neighbor similarity

            refined_weights[i, j] = 0.5 * node_similarity + 0.3 * degree_similarity + 0.2 * neighbor_similarity

    # Normalize rows and columns.  Add a small value to avoid division by zero.
    row_sums = refined_weights.sum(axis=1, keepdims=True)
    refined_weights = refined_weights / (row_sums + 1e-9)  # Add a small value to prevent division by zero

    col_sums = refined_weights.sum(axis=0, keepdims=True)
    refined_weights = refined_weights / (col_sums + 1e-9) # Add a small value to prevent division by zero

    return refined_weights.tolist()

def priority_v2(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1` -  uses numpy for efficiency."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = np.zeros((max_node, max_node))
  degrees1 = np.array([sum(row) for row in graph1])
  degrees2 = np.array([sum(row) for row in graph2])
  graph1 = np.array(graph1)
  graph2 = np.array(graph2)

  for i in range(n1):
    for j in range(n2):
        neighbor_similarity = np.sum((graph1[i,:][:,None] * graph2[j,:]) * (1.0 / (np.abs(degrees1[:,None] - degrees2) + 1)))
        weights[i, j] = (1.0 / (abs(degrees1[i] - degrees2[j]) + 1)) + neighbor_similarity
  return weights.tolist() # Convert back to list of lists as specified in the function signature

def priority_v3(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [([0.0] * max_node) for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      node_similarity = weights[i][j]
      neighbors1 = set((k for k in range(n1) if graph1[i][k]))
      neighbors2 = set((l for l in range(n2) if graph2[j][l]))
      
      intersection_size = len(neighbors1.intersection(neighbors2))
      union_size = len(neighbors1.union(neighbors2))

      if union_size > 0:  # Avoid division by zero
        neighbor_similarity = intersection_size / union_size
      else:
        neighbor_similarity = 1.0 if (not neighbors1 and not neighbors2) else 0.0 # If both are empty, similarity is 1

      degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
      degree_similarity = 1 / (1 + degree_diff) # Inversely proportional to degree difference


      refined_weights[i][j] = (node_similarity + neighbor_similarity + degree_similarity) / 3 # Combine all similarity measures



  # Normalize rows and columns iteratively for better convergence
  for _ in range(10):  # Increased iterations for potentially better results
      for i in range(n1):
          row_sum = sum(refined_weights[i][:n2])
          if row_sum > 0:
              for j in range(n2):
                  refined_weights[i][j] /= row_sum
      for j in range(n2):
          col_sum = sum(refined_weights[i][j] for i in range(n1))
          if col_sum > 0:
              for i in range(n1):
                  refined_weights[i][j] /= col_sum


  return refined_weights


def priority_v4(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]
    total_similarity = 0

    # Precompute degrees for efficiency
    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            node_similarity = weights[i][j]

            # Avoid division by zero
            degree1 = degrees1[i] + 1e-9
            degree2 = degrees2[j] + 1e-9
            degree_similarity = 2 / ((degree1 / degree2) + (degree2 / degree1))

            neighbor_similarity = 0
            for k in range(n1):
                for l in range(n2):
                    if graph1[i][k] == 1 and graph2[j][l] == 1:
                        edge_similarity = weights[k][l]
                    elif graph1[i][k] == 0 and graph2[j][l] == 0:
                        edge_similarity = 1 - weights[k][l]
                    else:
                        edge_similarity = 0
                    neighbor_similarity += edge_similarity
            
            # Normalize neighbor similarity
            neighbor_similarity /= (n1 * n2)

            # Combine similarities with balanced weights
            refined_weights[i][j] = (node_similarity + degree_similarity + neighbor_similarity) / 3
            total_similarity += refined_weights[i][j]

    if total_similarity > 0:
        for i in range(n1):
            for j in range(n2):
                refined_weights[i][j] /= total_similarity
    elif n1 > 0 and n2 > 0:
        uniform_weight = 1.0 / (n1 * n2)
        for i in range(n1):
            for j in range(n2):
                refined_weights[i][j] = uniform_weight
    return refined_weights



def priority_v5(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Improves `priority_v1` by incorporating edge existence more directly.
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
            for l in range(n2):
                # Penalize if edges don't match
                if graph1[i][k] != graph2[j][l]:
                    neighbor_similarity -= 1
                else:  # Reward if edges match
                    neighbor_similarity += (1.0 / (abs((degrees1[k] - degrees2[l])) + 1))

        weights[i][j] = ((1.0 / (abs((degrees1[i] - degrees2[j])) + 1)) + neighbor_similarity)


  # Normalize and handle potential negative weights
  min_weight = min(min(row) for row in weights[:n1][:n2])  # Consider only relevant part
  if min_weight < 0:
      for i in range(n1):
          for j in range(n2):
              weights[i][j] += abs(min_weight)

  total_weight = sum(sum(row) for row in weights[:n1][:n2])
  if total_weight > 0:
      for i in range(n1):
          for j in range(n2):
              weights[i][j] /= total_weight
  return weights

def priority_v6(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Combines degree difference and neighbor sum difference for probability calculation.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]
  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]
  neighbor_sums1 = [sum(graph1[i][k] * sum(graph1[k]) for k in range(n1)) for i in range(n1)]
  neighbor_sums2 = [sum(graph2[j][k] * sum(graph2[k]) for k in range(n2)) for j in range(n2)]

  for i in range(n1):
    for j in range(n2):
      degree_diff = abs(degrees1[i] - degrees2[j])
      neighbor_diff = abs(neighbor_sums1[i] - neighbor_sums2[j])
      weights[i][j] = 1.0 / (degree_diff + neighbor_diff + 1)  # Combine both differences

  return weights


def priority_v7(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
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
            degree_diff = abs(degrees1[i] - degrees2[j])
            common_neighbors = 0
            for neighbor1 in neighbors1[i]:
                for neighbor2 in neighbors2[j]:
                    # Consider existing weight in common neighbor calculation
                    common_neighbors += weights[neighbor1][neighbor2]  # Key Improvement
            neighbor_similarity = common_neighbors / (len(neighbors1[i]) + len(neighbors2[j]) + 1e-6) if (len(neighbors1[i]) + len(neighbors2[j])) > 0 else 0 # avoid division by zero


            weights[i][j] = (1.0 / (degree_diff + 1)) + neighbor_similarity

    return weights


def priority_v8(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`.  Normalizes across both rows and columns
    and incorporates neighbor similarity more effectively."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            node_similarity = weights[i][j]
            degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
            neighbor_similarity = 0

            neighbors1 = [k for k in range(n1) if graph1[i][k]]
            neighbors2 = [l for l in range(n2) if graph2[j][l]]

            for k in neighbors1:
                for l in neighbors2:
                    neighbor_similarity += weights[k][l]
            
            neighbor_similarity /= (len(neighbors1) * len(neighbors2) + 1e-6) # avoid division by zero

            refined_weights[i][j] = node_similarity + 0.1 * neighbor_similarity - 0.1 * degree_diff



    # Row normalization
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum

    # Column normalization (added for improved performance)
    for j in range(n2):
        col_sum = sum(refined_weights[i][j] for i in range(n1))
        if col_sum > 0:
            for i in range(n1):
                refined_weights[i][j] /= col_sum

    return refined_weights


def priority_v9(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v1`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    degrees1 = [sum(row) for row in graph1]
    degrees2 = [sum(row) for row in graph2]

    for i in range(n1):
        for j in range(n2):
            node_similarity = weights[i][j]  # Use initial weights
            degree_similarity = 1.0 / (abs(degrees1[i] - degrees2[j]) + 1)

            neighbor_similarity = 0
            neighbors1 = [k for k in range(n1) if graph1[i][k]]
            neighbors2 = [l for l in range(n2) if graph2[j][l]]

            for k in neighbors1:
                for l in neighbors2:
                    neighbor_similarity += weights[k][l]  # Use initial weights here as well

            if neighbors1 and neighbors2:
                neighbor_similarity /= (len(neighbors1) * len(neighbors2))

            refined_weights[i][j] = node_similarity * degree_similarity + neighbor_similarity


    # Normalize the refined weights (optional but recommended)
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum

    for j in range(n2):
        col_sum = sum(refined_weights[i][j] for i in range(n1))
        if col_sum > 0:
            for i in range(n1):
                refined_weights[i][j] /= col_sum


    return refined_weights


def priority_v10(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`."""
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [([0.0] * max_node) for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            node_similarity = weights[i][j]

            # Use sets for efficient neighbor comparison
            neighbors1 = set(k for k in range(n1) if graph1[i][k])
            neighbors2 = set(l for l in range(n2) if graph2[j][l])

            # Calculate common neighbors
            common_neighbors = neighbors1.intersection(neighbors2)
            
            # Calculate Jaccard similarity (Intersection over Union) for neighbors
            neighbor_similarity = len(common_neighbors) / (len(neighbors1.union(neighbors2)) or 1) # Avoid division by zero

            # Penalize degree difference more smoothly, using a squared difference and scaling factor
            degree_diff = (sum(graph1[i]) - sum(graph2[j]))**2 * 0.1


            refined_weights[i][j] = node_similarity + neighbor_similarity - degree_diff


    # Normalize row-wise then column-wise to ensure probabilities sum to 1
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum

    for j in range(n2):
        col_sum = sum(refined_weights[i][j] for i in range(n1))
        if col_sum > 0:
            for i in range(n1):
                refined_weights[i][j] /= col_sum

    return refined_weights


def priority_v11(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0`.
  Uses a normalized similarity score based on both degree difference and neighbor similarity.
  Handles cases where either graph is empty more gracefully.
  """
  n1 = len(graph1)
  n2 = len(graph2)

  if n1 == 0 or n2 == 0:  # Handle empty graphs
      max_node = max(n1, n2)
      return [[0.0] * max_node for _ in range(max_node)]

  max_node = max(n1, n2)
  refined_weights = [[0.0] * max_node for _ in range(max_node)]
  deg1 = [sum(row) for row in graph1]
  deg2 = [sum(row) for row in graph2]

  max_similarity = 0  # Keep track of the maximum similarity value

  for i in range(n1):
      for j in range(n2):
          neighbor_similarity = 0
          for k in range(n1):
              for l in range(n2):
                  if graph1[i][k] and graph2[j][l]:
                      neighbor_similarity += weights[k][l]

          degree_difference = abs(deg1[i] - deg2[j])
          if degree_difference == 0: # avoid division by zero
              degree_similarity = 1.0
          else:
              degree_similarity = 1.0 / (1.0 + degree_difference)  # Inversely proportional to degree difference
          
          similarity = degree_similarity + neighbor_similarity
          refined_weights[i][j] = similarity
          max_similarity = max(max_similarity, similarity)  # Update max_similarity


  if max_similarity > 0:  # Normalize only if max_similarity is not zero
      for i in range(n1):
          for j in range(n2):
              refined_weights[i][j] /= max_similarity
  else: # If all similarities are 0, give uniform probability (better than all zeros)
      for i in range(n1):
          for j in range(n2):
              refined_weights[i][j] = 1.0 / (n1 * n2)


  return refined_weights


def priority_v12(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v0`.

  This version addresses potential issues in `priority_v0` related to normalization and handling of empty graphs.
  It also incorporates a more robust similarity calculation considering both present and absent edges.

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
      node_similarity = weights[i][j]
      degree_diff = abs(sum(graph1[i]) - sum(graph2[j]))
      neighbor_similarity = 0
      for k in range(n1):
        for l in range(n2):
          if graph1[i][k] == graph2[j][l]:  # Both edges present or both absent
            neighbor_similarity += weights[k][l]
          else: # penalize if the edges do not match.
              neighbor_similarity -= weights[k][l]

      refined_weights[i][j] = node_similarity - 0.1 * degree_diff + 0.2 * neighbor_similarity

  # Normalize refined weights
  total_similarity = sum(sum(row) for row in refined_weights)
  if total_similarity > 0:
      for i in range(n1):
          for j in range(n2):
              refined_weights[i][j] /= total_similarity
  elif n1 > 0 and n2 > 0: # handle cases where total similarity is 0 but graphs are not empty.
      for i in range(n1):
          for j in range(n2):
              refined_weights[i][j] = 1.0 / (n1 * n2)

  return refined_weights


def priority_v13(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  refined_weights = [([0.0] * max_node) for _ in range(max_node)]

  for i in range(n1):
    for j in range(n2):
      node_similarity = weights[i][j]
      neighbors1 = set((k for k in range(n1) if graph1[i][k]))
      neighbors2 = set((l for l in range(n2) if graph2[j][l]))

      # Enhanced neighbor similarity calculation using Jaccard index
      common_neighbors = neighbors1.intersection(neighbors2)
      neighbor_similarity = len(common_neighbors) / (len(neighbors1.union(neighbors2)) or 1)  # Avoid division by zero

      # Reduced emphasis on degree difference, added normalization
      degree1 = sum(graph1[i])
      degree2 = sum(graph2[j])
      max_degree = max(degree1, degree2)
      degree_diff = abs(degree1 - degree2) / (max_degree or 1) # Normalized degree difference
      
      refined_weights[i][j] = node_similarity + neighbor_similarity - degree_diff

  # Iterative refinement using Sinkhorn normalization
  for _ in range(10):  # Adjust number of iterations as needed
      for i in range(n1):
          row_sum = sum(refined_weights[i][:n2])
          if row_sum > 0:
              for j in range(n2):
                  refined_weights[i][j] /= row_sum

      for j in range(n2):
          col_sum = sum((refined_weights[i][j] for i in range(n1)))
          if col_sum > 0:
              for i in range(n1):
                  refined_weights[i][j] /= col_sum
  return refined_weights



def priority_v14(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    Improves v1 by considering common neighbors more effectively.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)
  weights = [[0.0] * max_node for _ in range(max_node)]
  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  for i in range(n1):
    for j in range(n2):
        common_neighbors = 0
        for k in range(n1):
            for l in range(n2):
                if graph1[i][k] and graph2[j][l]: # Check if both edges exist
                    common_neighbors += 1.0 / (degrees1[k] + degrees2[l] + 2)  # Add a small constant to avoid division by zero
        weights[i][j] = (1.0 / (abs(degrees1[i] - degrees2[j]) + 1)) + common_neighbors

  # Normalize to probabilities (similar to v1)
  for i in range(n1):
      row_sum = sum(weights[i])
      if row_sum > 0:
          for j in range(n2):
              weights[i][j] /= row_sum
  return weights


def priority_v15(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
    """Improved version of `priority_v0`.
    Uses a more robust normalization and neighbor similarity calculation.
    """
    n1 = len(graph1)
    n2 = len(graph2)
    max_node = max(n1, n2)
    refined_weights = [[0.0] * max_node for _ in range(max_node)]

    for i in range(n1):
        for j in range(n2):
            node_similarity = weights[i][j]
            
            degree1 = sum(graph1[i])
            degree2 = sum(graph2[j])
            degree_diff = abs(degree1 - degree2) / (max(degree1, degree2, 1)) # Normalize degree difference

            neighbors1 = [k for k in range(n1) if graph1[i][k]]
            neighbors2 = [l for l in range(n2) if graph2[j][l]]

            neighbor_similarity = 0
            for k in neighbors1:
                for l in neighbors2:
                    neighbor_similarity += weights[k][l]
            
            neighbor_similarity /= (math.sqrt(len(neighbors1) * len(neighbors2)) + 1e-06) # Better normalization for sparse graphs


            refined_weights[i][j] = node_similarity + 0.5 * neighbor_similarity - 0.5 * degree_diff # Adjusted weights


    # Row-wise normalization
    for i in range(n1):
        row_sum = sum(refined_weights[i][:n2])
        if row_sum > 0:
            for j in range(n2):
                refined_weights[i][j] /= row_sum
        else: # handle empty rows
            for j in range(n2):
                refined_weights[i][j] = 1.0/n2 if n2>0 else 0.0

    # Column-wise normalization
    for j in range(n2):
        col_sum = sum(refined_weights[i][j] for i in range(n1))
        if col_sum > 0:
            for i in range(n1):
                refined_weights[i][j] /= col_sum
        else: # handle empty columns
            for i in range(n1):
                refined_weights[i][j] = 1.0/n1 if n1>0 else 0.0


    return refined_weights
# Input Scores: [12.0, 12.0, 15.0, 12.0, 9.0, 12.0, 11.0, 10.0, 14.0, 0.0, 12.0, 13.0, 12.0, 15.0, 10.0, 15.0, 11.0, 17.0, 10.0, 11.0, 10.0, 14.0, 9.0, 12.0, 10.0, 2.0, 12.0, 16.0, 11.0, 7.0, 3.0, 13.0, 16.0, 10.0, 8.0, 8.0, 13.0, 15.0, 11.0, 10.0, 12.0, 11.0, 12.0, 10.0, 9.0, 13.0, 11.0, 12.0, 13.0, 11.0, 12.0, 14.0, 13.0, 11.0, 14.0, 13.0, 19.0, 12.0, 8.0, 14.0, 11.0, 12.0, 9.0, 16.0, 12.0, 6.0, 12.0, 8.0, 12.0, 13.0, 12.0, 15.0, 2.0, 18.0, 10.0, 3.0, 11.0, 10.0, 10.0, 12.0, 6.0, 12.0, 16.0, 14.0, 9.0, 9.0, 14.0, 13.0, 11.0, 9.0, 16.0, 11.0, 14.0, 11.0, 15.0, 10.0, 12.0, 11.0, 13.0, 12.0, 12.0, 10.0, 8.0, 15.0, 12.0, 13.0, 9.0, 14.0, 10.0, 7.0, 12.0, 14.0, 10.0, 12.0, 15.0, 10.0, 10.0, 16.0, 16.0, 17.0, 10.0, 8.0, 18.0, 10.0, 13.0, 14.0, 14.0, 13.0, 8.0, 11.0, 11.0, 10.0, 14.0, 8.0, 10.0, 6.0, 10.0, 9.0, 12.0, 7.0, 12.0, 6.0, 16.0, 5.0, 14.0, 10.0, 5.0, 15.0, 5.0, 11.0, 10.0, 8.0, 12.0, 11.0, 16.0, 14.0, 12.0, 18.0, 13.0, 7.0, 12.0, 13.0, 12.0, 10.0, 12.0, 10.0, 22.0, 9.0, 11.0, 12.0, 13.0, 12.0, 14.0, 13.0, 15.0, 12.0, 9.0, 12.0, 13.0, 15.0, 8.0, 11.0, 17.0, 9.0, 14.0, 13.0, 9.0, 9.0, 14.0, 9.0, 11.0, 12.0, 9.0, 10.0, 10.0, 12.0, 9.0, 8.0, 9.0, 17.0, 15.0, 10.0, 8.0, 8.0, 12.0, 15.0, 13.0, 11.0, 8.0, 12.0, 10.0, 10.0, 16.0, 12.0, 12.0, 12.0, 14.0, 11.0, 15.0, 12.0, 12.0, 4.0, 8.0, 12.0, 8.0, 11.0, 12.0, 11.0, 12.0, 13.0, 11.0, 15.0, 13.0, 9.0, 13.0, 15.0, 14.0, 11.0, 13.0, 6.0, 6.0, 10.0, 11.0, 12.0, 16.0, 9.0, 6.0, 10.0, 9.0, 11.0, 13.0, 12.0, 11.0, 8.0, 6.0, 9.0, 9.0, 9.0, 12.0, 9.0, 15.0, 10.0, 8.0, 13.0, 13.0, 15.0, 8.0, 9.0, 12.0, 9.0, 14.0, 11.0, 5.0, 9.0, 11.0, 8.0, 8.0, 10.0, 8.0, 11.0, 12.0, 14.0, 13.0, 10.0, 7.0, 8.0, 19.0, 6.0, 12.0, 16.0, 7.0, 15.0, 13.0, 12.0, 18.0, 11.0, 11.0, 16.0, 14.0, 9.0, 11.0, 5.0, 10.0, 17.0, 9.0, 10.0, 11.0, 15.0, 10.0, 9.0, 14.0, 11.0, 11.0, 10.0, 11.0, 12.0, 15.0, 18.0, 14.0, 16.0, 12.0, 9.0, 8.0, 6.0, 13.0, 10.0, 8.0, 8.0, 11.0, 12.0, 10.0, 10.0, 8.0, 8.0, 9.0, 17.0, 9.0, 12.0, 11.0, 11.0, 17.0, 12.0, 12.0, 13.0, 12.0, 9.0, 20.0, 12.0, 10.0, 15.0, 12.0, 16.0, 16.0, 7.0, 14.0, 13.0, 11.0, 11.0, 11.0, 19.0, 17.0, 12.0, 10.0, 6.0, 10.0, 8.0, 9.0, 16.0, 6.0, 11.0, 12.0, 17.0, 12.0, 12.0, 15.0, 8.0, 8.0, 8.0, 14.0, 11.0, 10.0, 10.0, 15.0, 11.0, 10.0, 7.0, 7.0, 15.0, 9.0, 9.0, 12.0, 21.0, 16.0, 12.0, 12.0, 11.0, 11.0, 8.0, 11.0, 7.0, 9.0, 15.0, 9.0, 12.0, 9.0, 13.0, 13.0, 8.0, 11.0, 19.0, 12.0, 9.0, 12.0, 10.0, 14.0, 15.0, 13.0, 9.0, 10.0, 11.0, 12.0, 13.0, 9.0, 12.0, 11.0, 17.0, 6.0, 12.0, 10.0, 13.0, 17.0, 12.0, 14.0, 14.0, 12.0, 13.0, 17.0, 11.0, 13.0, 11.0, 12.0, 12.0, 12.0, 11.0, 11.0, 7.0, 9.0, 9.0, 14.0, 9.0, 5.0, 2.0, 17.0, 10.0, 18.0, 12.0, 11.0, 9.0, 8.0, 17.0, 11.0, 17.0, 16.0, 15.0, 12.0, 13.0, 13.0, 8.0, 9.0, 17.0, 10.0, 9.0, 16.0, 8.0, 11.0, 11.0, 16.0, 13.0, 12.0, 7.0, 9.0, 15.0, 14.0, 9.0, 15.0, 7.0, 10.0, 13.0, 9.0, 13.0, 9.0, 9.0, 14.0, 12.0, 17.0, 13.0, 9.0, 13.0, 9.0, 13.0, 13.0, 13.0, 17.0, 7.0, 10.0, 18.0, 17.0, 10.0, 9.0, 10.0, 9.0, 12.0, 7.0, 10.0, 11.0, 10.0, 14.0, 19.0, 20.0, 14.0, 14.0, 9.0, 9.0, 14.0, 14.0, 11.0, 12.0, 4.0, 8.0, 15.0, 20.0, 11.0, 9.0, 9.0, 9.0, 17.0, 6.0, 5.0, 9.0, 11.0, 10.0, 19.0, 13.0, 11.0, 11.0, 16.0, 10.0, 15.0, 15.0, 8.0, 12.0, 8.0, 15.0, 10.0, 9.0, 10.0, 12.0, 15.0, 7.0, 7.0, 13.0, 12.0, 15.0, 12.0, 14.0, 11.0, 10.0, 15.0, 14.0, 13.0, 13.0, 14.0, 15.0, 12.0, 11.0, 13.0, 9.0, 13.0, 17.0, 12.0, 5.0, 10.0, 10.0, 11.0, 16.0, 16.0, 10.0, 10.0, 17.0, 16.0, 18.0, 12.0, 12.0, 13.0, 10.0, 14.0, 8.0, 7.0, 5.0, 9.0, 11.0, 11.0, 12.0, 13.0, 8.0, 11.0, 12.0, 19.0, 10.0, 10.0, 12.0, 10.0, 10.0, 17.0, 15.0, 9.0, 13.0, 9.0, 7.0, 14.0, 9.0, 10.0, 18.0, 14.0, 17.0, 9.0, 13.0, 12.0, 7.0, 9.0, 10.0, 13.0, 10.0, 12.0, 14.0, 16.0, 12.0, 10.0, 14.0, 10.0, 11.0, 18.0, 10.0, 19.0, 7.0, 16.0, 13.0, 11.0, 11.0, 10.0, 15.0, 9.0, 13.0, 8.0, 16.0, 15.0, 13.0, 9.0, 13.0, 8.0, 11.0, 14.0, 14.0, 14.0, 11.0, 7.0, 12.0, 15.0, 16.0, 12.0, 9.0, 14.0, 12.0, 13.0, 11.0, 11.0, 12.0, 9.0, 14.0, 18.0, 13.0, 8.0, 10.0, 16.0, 11.0, 9.0, 8.0, 7.0, 12.0, 8.0, 11.0, 10.0, 14.0, 13.0, 14.0, 12.0, 15.0, 13.0, 11.0, 7.0, 11.0, 15.0, 11.0, 15.0, 10.0, 12.0, 10.0, 5.0, 11.0, 8.0, 13.0, 14.0, 17.0, 9.0, 8.0, 15.0, 17.0, 11.0, 7.0, 8.0, 16.0, 14.0, 24.0, 11.0, 16.0, 12.0, 10.0, 10.0, 9.0, 10.0, 11.0, 8.0, 13.0, 11.0, 10.0, 14.0, 13.0, 13.0, 9.0, 13.0, 0.0, 16.0, 5.0, 11.0, 12.0, 18.0, 12.0, 15.0, 13.0, 15.0, 9.0, 10.0, 11.0, 17.0, 11.0, 12.0, 11.0, 11.0, 9.0, 14.0, 8.0, 12.0, 15.0, 10.0, 13.0, 14.0, 9.0, 10.0, 14.0, 8.0, 13.0, 18.0, 11.0, 11.0, 13.0, 8.0, 14.0, 12.0, 10.0, 8.0, 13.0, 10.0, 17.0, 12.0, 16.0, 17.0, 7.0, 15.0, 15.0, 14.0, 21.0, 14.0, 9.0, 12.0, 12.0, 11.0, 11.0, 20.0, 14.0, 8.0, 11.0, 9.0, 10.0, 20.0, 7.0, 8.0, 11.0, 10.0, 13.0, 14.0, 10.0, 9.0, 15.0, 13.0, 13.0, 11.0, 10.0, 11.0, 16.0, 11.0, 8.0, 15.0, 11.0, 11.0, 11.0, 10.0, 10.0, 11.0, 12.0, 13.0, 5.0, 9.0, 18.0, 11.0, 12.0, 11.0, 9.0, 8.0, 11.0, 17.0, 11.0, 15.0, 9.0, 15.0, 11.0, 14.0, 16.0, 10.0, 8.0, 18.0, 13.0, 8.0, 14.0, 9.0, 9.0, 12.0, 10.0, 10.0, 14.0, 9.0, 13.0, 13.0, 14.0, 16.0, 9.0, 10.0, 13.0, 8.0, 18.0, 11.0, 14.0, 9.0, 9.0, 11.0, 9.0, 15.0, 14.0, 8.0, 13.0, 14.0, 12.0, 7.0, 13.0, 15.0, 11.0, 18.0, 16.0, 15.0, 18.0, 9.0, 11.0, 6.0, 14.0, 8.0, 14.0, 10.0, 8.0, 10.0, 14.0, 14.0, 16.0, 12.0, 13.0, 14.0, 12.0, 9.0, 9.0, 11.0, 15.0, 11.0, 12.0, 11.0, 11.0, 13.0, 9.0, 12.0, 12.0, 9.0, 12.0, 8.0, 14.0, 7.0, 12.0, 12.0, 13.0, 10.0, 12.0, 12.0, 9.0, 7.0, 12.0, 10.0, 12.0, 12.0, 18.0, 12.0, 10.0, 10.0, 16.0, 11.0, 11.0, 10.0, 9.0, 15.0, 10.0, 13.0, 13.0, 16.0, 8.0, 10.0, 10.0, 10.0, 13.0, 11.0, 10.0, 14.0, 14.0, 11.0, 12.0, 9.0, 11.0, 18.0, 8.0, 11.0, 9.0, 15.0, 12.0, 7.0, 19.0, 10.0, 15.0, 9.0, 12.0, 9.0, 17.0, 14.0, 12.0, 14.0, 0.0, 11.0, 11.0, 10.0, 15.0, 7.0, 19.0, 9.0, 9.0, 8.0, 14.0, 10.0, 8.0, 11.0, 12.0, 9.0, 9.0, 9.0, 11.0, 12.0, 11.0, 9.0]

##### Best Scores: [11.0, 12.0, 9.0, 12.0, 7.0, 9.0, 6.0, 9.0, 12.0, 0.0, 10.0, 11.0, 5.0, 11.0, 9.0, 9.0, 9.0, 10.0, 10.0, 9.0, 6.0, 14.0, 7.0, 10.0, 9.0, 2.0, 10.0, 16.0, 9.0, 7.0, 3.0, 10.0, 14.0, 9.0, 8.0, 6.0, 10.0, 11.0, 9.0, 9.0, 10.0, 11.0, 5.0, 6.0, 8.0, 13.0, 8.0, 7.0, 8.0, 7.0, 9.0, 10.0, 13.0, 8.0, 9.0, 7.0, 16.0, 9.0, 6.0, 10.0, 8.0, 12.0, 7.0, 11.0, 8.0, 6.0, 10.0, 8.0, 10.0, 9.0, 11.0, 10.0, 2.0, 18.0, 9.0, 3.0, 9.0, 10.0, 10.0, 6.0, 6.0, 12.0, 15.0, 10.0, 6.0, 7.0, 11.0, 8.0, 8.0, 5.0, 16.0, 9.0, 14.0, 7.0, 12.0, 9.0, 9.0, 8.0, 8.0, 7.0, 10.0, 8.0, 7.0, 10.0, 5.0, 9.0, 8.0, 13.0, 7.0, 7.0, 12.0, 13.0, 8.0, 12.0, 11.0, 10.0, 7.0, 11.0, 9.0, 12.0, 8.0, 8.0, 7.0, 6.0, 9.0, 10.0, 12.0, 11.0, 6.0, 11.0, 10.0, 6.0, 9.0, 8.0, 7.0, 6.0, 4.0, 8.0, 9.0, 6.0, 10.0, 4.0, 16.0, 4.0, 11.0, 10.0, 5.0, 10.0, 5.0, 9.0, 10.0, 8.0, 10.0, 9.0, 16.0, 11.0, 9.0, 14.0, 8.0, 6.0, 11.0, 10.0, 10.0, 8.0, 7.0, 9.0, 18.0, 9.0, 8.0, 9.0, 10.0, 10.0, 10.0, 11.0, 14.0, 5.0, 8.0, 9.0, 8.0, 13.0, 6.0, 10.0, 14.0, 6.0, 12.0, 10.0, 8.0, 8.0, 14.0, 9.0, 11.0, 12.0, 9.0, 8.0, 7.0, 9.0, 9.0, 6.0, 9.0, 14.0, 10.0, 10.0, 8.0, 8.0, 10.0, 13.0, 9.0, 9.0, 8.0, 11.0, 10.0, 8.0, 16.0, 10.0, 11.0, 11.0, 8.0, 6.0, 8.0, 12.0, 6.0, 4.0, 8.0, 12.0, 8.0, 10.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 12.0, 8.0, 11.0, 10.0, 14.0, 8.0, 7.0, 6.0, 5.0, 6.0, 9.0, 4.0, 13.0, 6.0, 5.0, 8.0, 6.0, 10.0, 12.0, 11.0, 11.0, 7.0, 6.0, 5.0, 5.0, 9.0, 10.0, 9.0, 11.0, 10.0, 7.0, 8.0, 12.0, 10.0, 6.0, 8.0, 7.0, 9.0, 9.0, 6.0, 5.0, 5.0, 8.0, 5.0, 8.0, 7.0, 8.0, 7.0, 9.0, 11.0, 11.0, 10.0, 7.0, 8.0, 15.0, 5.0, 9.0, 14.0, 6.0, 9.0, 12.0, 9.0, 18.0, 7.0, 8.0, 13.0, 5.0, 7.0, 10.0, 5.0, 6.0, 13.0, 9.0, 9.0, 10.0, 14.0, 8.0, 9.0, 7.0, 11.0, 11.0, 8.0, 9.0, 8.0, 11.0, 17.0, 10.0, 11.0, 10.0, 8.0, 6.0, 5.0, 9.0, 7.0, 7.0, 8.0, 11.0, 6.0, 9.0, 8.0, 8.0, 8.0, 7.0, 11.0, 7.0, 12.0, 11.0, 8.0, 17.0, 12.0, 10.0, 13.0, 9.0, 6.0, 18.0, 11.0, 10.0, 11.0, 7.0, 14.0, 13.0, 5.0, 11.0, 11.0, 7.0, 7.0, 11.0, 17.0, 12.0, 10.0, 10.0, 6.0, 9.0, 8.0, 8.0, 10.0, 4.0, 7.0, 10.0, 11.0, 12.0, 6.0, 14.0, 8.0, 6.0, 5.0, 11.0, 9.0, 10.0, 10.0, 12.0, 11.0, 8.0, 6.0, 7.0, 15.0, 9.0, 9.0, 10.0, 14.0, 13.0, 8.0, 8.0, 8.0, 10.0, 8.0, 7.0, 7.0, 8.0, 9.0, 9.0, 10.0, 9.0, 11.0, 10.0, 8.0, 7.0, 12.0, 11.0, 6.0, 8.0, 8.0, 14.0, 10.0, 7.0, 9.0, 10.0, 11.0, 8.0, 12.0, 8.0, 10.0, 10.0, 17.0, 5.0, 11.0, 7.0, 12.0, 17.0, 6.0, 11.0, 11.0, 9.0, 11.0, 9.0, 8.0, 11.0, 5.0, 10.0, 11.0, 11.0, 10.0, 10.0, 5.0, 5.0, 9.0, 11.0, 8.0, 5.0, 0.0, 11.0, 8.0, 9.0, 8.0, 7.0, 8.0, 6.0, 13.0, 9.0, 16.0, 10.0, 13.0, 11.0, 7.0, 9.0, 7.0, 9.0, 11.0, 10.0, 7.0, 11.0, 6.0, 9.0, 9.0, 14.0, 7.0, 9.0, 3.0, 5.0, 11.0, 8.0, 9.0, 15.0, 7.0, 10.0, 7.0, 9.0, 13.0, 7.0, 9.0, 12.0, 9.0, 10.0, 8.0, 6.0, 13.0, 9.0, 8.0, 8.0, 9.0, 13.0, 7.0, 10.0, 16.0, 15.0, 8.0, 7.0, 7.0, 9.0, 11.0, 7.0, 10.0, 9.0, 10.0, 12.0, 17.0, 20.0, 12.0, 12.0, 8.0, 5.0, 9.0, 9.0, 9.0, 9.0, 4.0, 8.0, 12.0, 14.0, 9.0, 9.0, 8.0, 7.0, 10.0, 4.0, 5.0, 9.0, 11.0, 8.0, 13.0, 7.0, 10.0, 9.0, 14.0, 8.0, 11.0, 9.0, 6.0, 8.0, 8.0, 12.0, 9.0, 8.0, 6.0, 9.0, 11.0, 6.0, 7.0, 9.0, 10.0, 12.0, 10.0, 12.0, 7.0, 9.0, 8.0, 9.0, 13.0, 11.0, 11.0, 12.0, 11.0, 10.0, 10.0, 5.0, 13.0, 8.0, 12.0, 5.0, 8.0, 8.0, 9.0, 10.0, 11.0, 7.0, 8.0, 12.0, 13.0, 12.0, 10.0, 9.0, 13.0, 9.0, 9.0, 7.0, 5.0, 5.0, 8.0, 9.0, 10.0, 10.0, 10.0, 7.0, 8.0, 10.0, 19.0, 10.0, 8.0, 8.0, 9.0, 8.0, 9.0, 13.0, 5.0, 7.0, 5.0, 7.0, 7.0, 8.0, 8.0, 15.0, 10.0, 12.0, 7.0, 8.0, 8.0, 5.0, 6.0, 8.0, 10.0, 10.0, 11.0, 14.0, 10.0, 11.0, 7.0, 9.0, 10.0, 10.0, 11.0, 10.0, 13.0, 7.0, 16.0, 11.0, 10.0, 9.0, 10.0, 12.0, 7.0, 9.0, 8.0, 12.0, 8.0, 9.0, 9.0, 10.0, 8.0, 7.0, 9.0, 8.0, 11.0, 7.0, 5.0, 10.0, 12.0, 10.0, 8.0, 9.0, 6.0, 10.0, 5.0, 10.0, 8.0, 9.0, 9.0, 14.0, 16.0, 8.0, 8.0, 10.0, 13.0, 11.0, 7.0, 6.0, 7.0, 7.0, 6.0, 11.0, 8.0, 8.0, 12.0, 11.0, 8.0, 13.0, 12.0, 11.0, 7.0, 6.0, 7.0, 5.0, 14.0, 6.0, 5.0, 7.0, 5.0, 10.0, 7.0, 10.0, 14.0, 14.0, 7.0, 5.0, 7.0, 11.0, 7.0, 7.0, 8.0, 15.0, 13.0, 20.0, 10.0, 12.0, 12.0, 9.0, 8.0, 9.0, 8.0, 8.0, 7.0, 10.0, 9.0, 8.0, 12.0, 6.0, 8.0, 9.0, 11.0, 0.0, 12.0, 5.0, 7.0, 9.0, 9.0, 11.0, 11.0, 12.0, 9.0, 8.0, 10.0, 10.0, 14.0, 8.0, 9.0, 11.0, 6.0, 9.0, 7.0, 8.0, 9.0, 12.0, 10.0, 13.0, 11.0, 4.0, 10.0, 12.0, 8.0, 7.0, 14.0, 11.0, 10.0, 9.0, 8.0, 12.0, 2.0, 7.0, 8.0, 9.0, 10.0, 13.0, 11.0, 14.0, 14.0, 7.0, 11.0, 15.0, 8.0, 17.0, 9.0, 7.0, 10.0, 7.0, 9.0, 7.0, 15.0, 11.0, 8.0, 11.0, 7.0, 8.0, 18.0, 5.0, 7.0, 10.0, 10.0, 8.0, 10.0, 10.0, 9.0, 11.0, 10.0, 10.0, 8.0, 9.0, 11.0, 9.0, 11.0, 8.0, 10.0, 9.0, 10.0, 10.0, 10.0, 8.0, 9.0, 9.0, 10.0, 5.0, 8.0, 15.0, 11.0, 12.0, 10.0, 6.0, 6.0, 10.0, 16.0, 6.0, 10.0, 7.0, 13.0, 10.0, 11.0, 8.0, 8.0, 8.0, 11.0, 8.0, 8.0, 13.0, 9.0, 9.0, 6.0, 9.0, 7.0, 10.0, 6.0, 9.0, 11.0, 10.0, 12.0, 7.0, 9.0, 10.0, 8.0, 18.0, 11.0, 10.0, 7.0, 7.0, 8.0, 9.0, 8.0, 9.0, 7.0, 13.0, 10.0, 11.0, 5.0, 13.0, 8.0, 6.0, 16.0, 12.0, 10.0, 11.0, 8.0, 9.0, 6.0, 14.0, 7.0, 9.0, 8.0, 8.0, 10.0, 14.0, 8.0, 11.0, 10.0, 13.0, 10.0, 9.0, 9.0, 7.0, 11.0, 12.0, 8.0, 9.0, 9.0, 11.0, 11.0, 7.0, 12.0, 6.0, 8.0, 12.0, 6.0, 12.0, 7.0, 10.0, 11.0, 8.0, 10.0, 8.0, 7.0, 8.0, 7.0, 10.0, 7.0, 10.0, 7.0, 9.0, 12.0, 9.0, 10.0, 12.0, 8.0, 10.0, 10.0, 8.0, 15.0, 10.0, 13.0, 10.0, 11.0, 8.0, 9.0, 8.0, 8.0, 12.0, 11.0, 10.0, 11.0, 13.0, 11.0, 8.0, 9.0, 8.0, 17.0, 7.0, 10.0, 9.0, 9.0, 11.0, 7.0, 8.0, 10.0, 13.0, 9.0, 6.0, 8.0, 15.0, 13.0, 10.0, 11.0, 0.0, 8.0, 8.0, 9.0, 12.0, 7.0, 16.0, 6.0, 9.0, 8.0, 9.0, 9.0, 8.0, 10.0, 10.0, 7.0, 7.0, 7.0, 9.0, 12.0, 9.0, 8.0]

##### Ground Truths: [11, 12, 9, 12, 7, 9, 6, 9, 12, 0, 10, 10, 5, 11, 8, 8, 9, 9, 9, 7, 6, 14, 7, 10, 7, 2, 10, 16, 9, 7, 3, 9, 12, 7, 8, 6, 10, 11, 8, 9, 10, 11, 5, 5, 8, 13, 8, 7, 8, 7, 9, 10, 13, 8, 9, 7, 16, 9, 6, 9, 8, 12, 7, 11, 8, 6, 10, 8, 8, 9, 11, 9, 2, 18, 7, 3, 9, 8, 10, 6, 6, 12, 15, 9, 6, 7, 11, 8, 8, 5, 16, 9, 14, 7, 12, 9, 8, 8, 7, 7, 10, 8, 7, 10, 5, 9, 6, 12, 7, 7, 12, 13, 8, 12, 11, 10, 7, 11, 9, 12, 8, 8, 7, 6, 7, 10, 10, 11, 6, 11, 10, 6, 9, 8, 7, 6, 4, 8, 9, 6, 10, 4, 16, 4, 11, 10, 5, 10, 5, 9, 9, 8, 10, 9, 16, 11, 9, 14, 8, 6, 11, 9, 10, 8, 6, 9, 18, 9, 8, 9, 9, 10, 8, 11, 14, 5, 8, 9, 8, 13, 6, 10, 14, 6, 11, 10, 8, 8, 14, 9, 11, 12, 9, 8, 7, 9, 9, 6, 9, 12, 10, 9, 8, 8, 9, 13, 8, 8, 8, 11, 10, 7, 14, 9, 11, 11, 8, 6, 8, 11, 6, 4, 8, 12, 8, 9, 9, 9, 9, 9, 8, 9, 11, 8, 11, 10, 12, 7, 7, 6, 5, 6, 9, 4, 13, 6, 5, 8, 6, 10, 12, 11, 11, 7, 6, 5, 5, 9, 10, 9, 11, 10, 7, 8, 12, 10, 6, 8, 7, 9, 9, 6, 5, 5, 8, 5, 8, 7, 8, 7, 9, 10, 10, 9, 7, 8, 13, 5, 9, 14, 6, 9, 12, 9, 18, 7, 8, 13, 5, 7, 10, 5, 6, 13, 8, 8, 10, 14, 8, 9, 7, 11, 9, 8, 9, 8, 11, 17, 10, 11, 10, 7, 6, 5, 9, 7, 7, 7, 11, 6, 9, 8, 8, 8, 7, 10, 7, 10, 9, 8, 17, 11, 10, 10, 9, 6, 18, 11, 10, 11, 5, 14, 13, 5, 11, 10, 7, 7, 11, 17, 12, 10, 10, 6, 9, 7, 7, 10, 4, 7, 10, 11, 11, 6, 14, 8, 6, 5, 11, 9, 8, 10, 12, 11, 8, 6, 7, 15, 9, 9, 10, 14, 13, 8, 7, 8, 10, 8, 7, 6, 8, 9, 9, 10, 9, 11, 10, 8, 7, 12, 9, 6, 8, 8, 14, 10, 7, 9, 10, 11, 8, 12, 8, 10, 10, 17, 5, 11, 7, 12, 17, 6, 9, 11, 9, 11, 9, 8, 10, 5, 10, 11, 9, 10, 9, 5, 5, 9, 10, 7, 5, 0, 11, 8, 8, 8, 7, 8, 6, 13, 8, 16, 10, 11, 11, 7, 8, 6, 9, 11, 9, 7, 11, 6, 9, 9, 14, 7, 8, 3, 5, 11, 8, 9, 14, 7, 10, 7, 9, 13, 7, 9, 12, 8, 10, 7, 6, 13, 8, 8, 8, 9, 13, 7, 10, 16, 15, 8, 7, 7, 8, 8, 7, 10, 9, 9, 11, 17, 20, 12, 12, 8, 5, 9, 9, 9, 8, 4, 8, 12, 14, 9, 9, 8, 7, 10, 4, 5, 9, 11, 8, 13, 7, 10, 9, 14, 7, 11, 9, 6, 8, 8, 12, 9, 8, 6, 9, 11, 6, 7, 9, 10, 12, 10, 12, 7, 9, 8, 9, 12, 11, 11, 12, 11, 10, 10, 5, 13, 8, 12, 5, 6, 8, 9, 10, 11, 7, 8, 12, 12, 12, 10, 8, 13, 9, 9, 7, 5, 5, 8, 9, 8, 10, 10, 7, 8, 8, 19, 10, 6, 8, 9, 8, 9, 13, 5, 7, 5, 7, 7, 7, 8, 13, 10, 12, 7, 8, 8, 5, 6, 8, 10, 10, 11, 14, 10, 11, 7, 9, 9, 10, 11, 10, 13, 7, 16, 11, 8, 9, 10, 12, 7, 9, 8, 11, 8, 9, 8, 10, 8, 7, 9, 8, 9, 7, 5, 9, 12, 10, 8, 7, 6, 10, 5, 9, 8, 9, 9, 14, 16, 8, 7, 10, 12, 10, 6, 6, 7, 7, 6, 10, 6, 8, 12, 10, 8, 13, 10, 9, 6, 6, 7, 5, 14, 6, 5, 7, 5, 10, 6, 10, 13, 14, 7, 5, 7, 11, 7, 7, 8, 13, 12, 20, 8, 12, 11, 8, 8, 7, 6, 8, 7, 10, 9, 8, 10, 6, 7, 9, 11, 0, 12, 5, 7, 9, 9, 9, 11, 12, 9, 7, 10, 10, 13, 8, 9, 10, 6, 9, 7, 8, 9, 12, 10, 13, 10, 4, 10, 12, 8, 7, 14, 11, 10, 9, 8, 11, 2, 7, 8, 9, 10, 13, 11, 14, 14, 7, 11, 13, 8, 17, 9, 7, 9, 7, 8, 7, 15, 11, 6, 11, 7, 8, 18, 5, 7, 10, 10, 8, 9, 10, 8, 11, 10, 7, 8, 9, 9, 8, 10, 8, 9, 9, 10, 10, 10, 8, 9, 9, 10, 5, 8, 15, 9, 12, 10, 6, 6, 10, 16, 6, 10, 7, 13, 10, 10, 8, 8, 8, 11, 8, 8, 12, 8, 7, 6, 9, 7, 10, 6, 9, 10, 10, 12, 7, 7, 10, 8, 18, 10, 10, 7, 6, 7, 9, 8, 9, 7, 13, 10, 10, 5, 13, 8, 6, 16, 12, 10, 11, 8, 9, 6, 14, 7, 8, 8, 8, 9, 14, 8, 11, 10, 12, 10, 9, 9, 7, 9, 11, 8, 9, 9, 11, 11, 7, 12, 6, 8, 11, 6, 12, 7, 10, 9, 8, 9, 8, 7, 8, 7, 10, 7, 10, 7, 9, 12, 8, 10, 12, 8, 8, 10, 7, 15, 10, 13, 10, 11, 8, 9, 8, 8, 12, 11, 10, 11, 13, 11, 7, 9, 8, 15, 7, 9, 9, 9, 10, 7, 8, 10, 13, 9, 6, 8, 15, 13, 9, 10, 0, 8, 8, 7, 12, 7, 15, 6, 9, 8, 9, 9, 8, 10, 10, 6, 7, 7, 9, 12, 9, 8]
##### Test Results: RMSE - 0.5966573556070519, MAE: 0.234, Num Gt: 824/1000

