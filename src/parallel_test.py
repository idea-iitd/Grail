import importlib
import ast
import copy
import numpy as np
from math import ceil
import math
import time
import math
import numpy as np
import heapq
import multiprocessing

def process_chunk(chunk, function_names, module_name, n):
    """Process a chunk of graph pairs across all functions."""
    results = []
    module = importlib.import_module(module_name)

    for idx, (g_pair, nf, gt) in chunk:
        g10, g20 = g_pair
        nf1, nf2 = nf
        max_node = max(len(nf1), len(nf2))
        if(idx%10==0):
            print(idx)

        g1, g2, ged = convert_to_adjmatrix(g10, g20, max_node)
        x = len(g1[0])
        y1, y2 = len(nf1), len(nf2)
        labels10, labels20 = assign_labels_based_on_features(nf1, nf2)
        nf1 = nf1 + [[-1]] * (x - y1)
        nf2 = nf2 + [[-1]] * (x - y2)

        labels1, labels2 = assign_labels_based_on_features(nf1, nf2)

        weights = [[1.0] * max_node for _ in range(max_node)]
        for i in range(max_node):
            for j in range(max_node):
                if i >= len(labels10) or j >= len(labels20) or labels10[i] == "eps" or labels20[j] == "eps":
                    weights[i][j] = 0.0
                elif labels10[i] == labels20[j]:
                    weights[i][j] = 1.0
                else:
                    weights[i][j] = 0.0

        weights = np.array(weights)
        row_sums = weights.sum(axis=1, keepdims=True)
        weights = np.divide(weights, row_sums, out=weights, where=row_sums != 0).tolist()

        funcs = []
        fun_cnt = 0
        for i, func_name in enumerate(function_names):
            priority = getattr(module, func_name)
            try:
                pr = priority(copy.deepcopy(g1), copy.deepcopy(g2), weights)
            except:
                func_flag[i] = False
                continue

            if pr is not None:
                for _ in range(n):
                    try:
                        x = priority(copy.deepcopy(g1), copy.deepcopy(g2), pr)
                    except:
                        func_flag[i] = False
                        break
                    if x is None:
                        break
                    pr = x
            else:
                pr = weights

            if func_flag[i]:
                funcs.append(pr)
                fun_cnt+=1
            else:
                funcs.append(100000)
        
        # print(f'Num functions without error on test set: {fun_cnt}')#Expected no: 15

        def compute_costs_for_n(g1, g2, labels1, labels2, weights):
            costs = []
            for func in funcs:
                if(func == 100000):
                    cost = 100000
                else:
                    nodemap = solve(g1, g2, labels1,labels2,func)

                    cost = 0
                    y = 0
                    for v1 in range(len(g1)):
                        mapped_v1 = nodemap[v1]
                        if labels1[v1] != "eps" and labels2[mapped_v1] != "eps" and labels1[v1] != labels2[mapped_v1]:
                            y += 1

                    for v1 in range(len(g1)):
                        for v2 in range(len(g1[0])):
                            mapped_v1 = nodemap.get(v1, -1)
                            mapped_v2 = nodemap.get(v2, -1)
                            if g1[v1][v2] == 1:
                                if mapped_v1 != -1 and mapped_v2 != -1 and g2[mapped_v1][mapped_v2] != 1:
                                    cost += 1
                            elif g1[v1][v2] == 0:
                                if mapped_v1 != -1 and mapped_v2 != -1 and g2[mapped_v1][mapped_v2] != 0:
                                    cost += 1

                    cost = cost / 2 + ged + y
                costs.append(cost)
            return costs

        row = [idx, gt] + compute_costs_for_n(g1, g2, labels1, labels2, weights)
        results.append(row)

    return results

def evaluate(testset, func_set, layer_val, num_processes):
    # import os
    # print(os.cpu_count())
    """Parallelized RMSE computation for graph edit distances with multiprocessing."""
    module_name = f'top_fns_transfer.{func_set}' 

    with open(f'../data/test/{testset}_test.txt', 'r') as file:
        content = file.read()

        g_pairs_str = content.split("g_pairs: ")[1].split("node_features: ")[0].strip()
        nf_str = content.split("node_features: ")[1].split("ground_truth: ")[0].strip()
        ground_truth_str = content.split("ground_truth: ")[1].strip()

        g_pairs = ast.literal_eval(g_pairs_str)
        nfs = ast.literal_eval(nf_str)
        ground_truth = ast.literal_eval(ground_truth_str)

    n = layer_val
    
    with open(f'./top_15/f_val_set_{testset}_funcs_{func_set}.txt', 'r') as file:
        function_names = [line.strip() for line in file.readlines()]

    print("testing")
    start_time=time.time()
    
    global func_flag
    func_flag = [True] * len(function_names)

    # Split graph pairs into chunks for parallel processing
    data = list(enumerate(zip(g_pairs, nfs, ground_truth), start=1))
    chunk_size = ceil(len(data) / num_processes)
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    # Use multiprocessing to process chunks
    all_results = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_chunk, [(chunk, function_names, module_name, n) for chunk in chunks])
        for chunk_result in results:
            all_results.extend(chunk_result)

    # Sort and write results to Excel
    geds = []
    gts = []
    exact_matches = 0
    total_pairs = len(all_results)
    for row in sorted(all_results, key=lambda x: x[0]):  # Sort rows by index
        gts.append(row[1])
        values = row[2:]  # Values to consider for min
        
        # Filter values based on flags
        filtered_values = [value for value, flag in zip(values, func_flag) if flag]
        
        # Calculate min for filtered values (or handle empty case)
        min_value = min(filtered_values) if filtered_values else float('inf')
        geds.append(min_value)
        
        # Check if the minimum filtered value equals row[1]
        if min_value == row[1]:
            exact_matches += 1
            
    end_time=time.time()
    rmse = np.sqrt(np.mean((np.array(gts) - np.array(geds)) ** 2))
    mae = np.mean(np.abs((np.array(gts) - np.array(geds))))
    emr = exact_matches / total_pairs
    print(f"n: {n}, RMSE: {rmse}, MAE: {mae}, EMR: {emr}")
    with open('transfer_results.txt', 'a') as res_file:
        res_file.write(f"##### Test Set: {testset}, Func Set: {func_set} #####\n")
        res_file.write(f"n: {n}, RMSE: {rmse}, MAE: {mae}, EMR: {emr}\n")
        res_file.write(f"Time taken: {end_time-start_time} secs\n\n")
    
    print("Time taken: ",end_time-start_time," secs")


def convert_to_adjmatrix(edge_list1, edge_list2, max_node):
    adjacency_matrix1 = np.zeros((max_node, max_node), dtype=int)
    adjacency_matrix2 = np.zeros((max_node, max_node), dtype=int)

    adjacency_matrix1[edge_list1[0], edge_list1[1]] = 1
    adjacency_matrix2[edge_list2[0], edge_list2[1]] = 1

    ged = 2 * max_node - 2 - max(max(edge_list1[0], default=0), max(edge_list1[1], default=0)) \
          - max(max(edge_list2[0], default=0), max(edge_list2[1], default=0))
    return adjacency_matrix1.tolist(), adjacency_matrix2.tolist(), ged

def assign_labels_based_on_features(nftrs1,nftrs2):
    labels = {}  # Dictionary to store unique feature-to-label mapping
    node_labels1 = []  # Dictionary to store node index to label mapping
    node_labels2 = []
    current_label_index = 0  # Tracks the next label to assign

    # Alphabet list for labeling
    alphabet = [chr(ord('a') + i) for i in range(26)]  # 'a', 'b', ..., 'z'
    
    # If more than 26 labels are needed, extend the alphabet (aa, ab, etc.)
    z=len(nftrs1)+len(nftrs2)
    while len(alphabet) < z :
        alphabet += [a + b for a in alphabet for b in alphabet][:z - len(alphabet)]
    
    node_labels1, node_labels2 = [], []


    for i, features in enumerate(nftrs1):
        features_tuple = tuple(features)  # Convert features to a tuple to make it hashable
        if features_tuple==(-1,):
           node_labels1.append("eps")
        else:
        # Check if features already have an assigned label
          if features_tuple in labels:
            node_labels1.append(labels[features_tuple])  # Assign existing label
          else:
              # Assign a new label
              labels[features_tuple] = alphabet[current_label_index]
              node_labels1.append(alphabet[current_label_index])
              current_label_index += 1

    for i, features in enumerate(nftrs2):
        features_tuple = tuple(features)  # Convert features to a tuple to make it hashable
        
        if features_tuple==(-1,):
           node_labels2.append("eps")
        # Check if features already have an assigned label
        else:
           if features_tuple in labels:
            node_labels2.append(labels[features_tuple])  # Assign existing label
           else:
            labels[features_tuple] = alphabet[current_label_index]
            node_labels2.append(alphabet[current_label_index])
            current_label_index += 1   

    return node_labels1, node_labels2

def solve(
    graph1: list[list[int]], 
    graph2: list[list[int]], 
    labels1, labels2, weights
) -> dict:
    """Builds a mapping between 2 labelled graphs using Neighbor Bias Mapper and label similarity.
    
    Args:
        graph1: adjacency matrix of graph 1
        graph2: adjacency matrix of graph 2
        labels1: labels of graph 1 nodes
        labels2: labels of graph 2 nodes
        priority: function to calculate priority between nodes
    
    Returns:
        nodemappings: A dictionary mapping nodes of graph1 to graph2.
    """
    n1, n2 = len(graph1), len(graph2)
    
    # Function to calculate label similarity between nodes
    def label_similarity(u, v):
        return 1 if labels1[u] == labels2[v] else 0
    

    W=weights

    PQ = []
    mate = [-1] * n1  # stores best match in graph2 for each node in graph1
    wt = [-math.inf] * n1  # stores the best weight for each node in graph1

    for u in range(n1):
        # print(u)
        v_m = max(range(n2), key=lambda v: W[u][v])  # best match for u in graph2
        heapq.heappush(PQ, (-W[u][v_m], u, v_m))  # store negative weight for max-heap behavior
        mate[u] = v_m
        wt[u] = W[u][v_m]

    matched = set()
    nodemappings = {}

    # Function to get neighbors up to 1 hop away
    def get_1hop_neighbors(u, graph):
        return [i for i in range(len(graph[u])) if graph[u][i] > 0]

    # Function to get neighbors up to 2 hops away
    def get_2hop_neighbors(u, graph):
        neighbors = set()
        for neighbor in range(len(graph[u])):
            if graph[u][neighbor] > 0:
                neighbors.add(neighbor)
                for second_neighbor in range(len(graph[neighbor])):
                    if graph[neighbor][second_neighbor] > 0:
                        neighbors.add(second_neighbor)
        return neighbors

    while PQ:
        _, u, v = heapq.heappop(PQ)
        if u in nodemappings:
            continue
        if v in matched:
            v_m = max((w for w in range(n2) if w not in matched), key=lambda x: W[u][x], default=None)
            if v_m is not None:
                heapq.heappush(PQ, (-W[u][v_m], u, v_m))
                mate[u] = v_m
                wt[u] = W[u][v_m]
            continue

        nodemappings[u] = v
        matched.add(v)

        neighbors_u = get_1hop_neighbors(u, graph1)
        neighbors_v = get_1hop_neighbors(v, graph2)
        
        for u_prime in neighbors_u:
            if u_prime in nodemappings:
                continue
            for v_prime in neighbors_v:
                if v_prime in matched:
                    continue
                # Update weight for u_prime and v_prime based on label similarity and 1-hop neighbors
                W[u_prime][v_prime] += W[u][v] + label_similarity(u_prime, v_prime)
                if W[u_prime][v_prime] > wt[u_prime]:
                    mate[u_prime] = v_prime
                    wt[u_prime] = W[u_prime][v_prime]
                    heapq.heappush(PQ, (-W[u_prime][v_prime], u_prime, v_prime))

    # Second pass: further refine weights using 2-hop neighbors
    PQ = []
    for u in range(n1):
        v_m = max(range(n2), key=lambda v: W[u][v])  # best match for u in graph2 after 1-hop update
        heapq.heappush(PQ, (-W[u][v_m], u, v_m))
        mate[u] = v_m
        wt[u] = W[u][v_m]

    # Process the priority queue again for 2-hop neighbors
    while PQ:
        _, u, v = heapq.heappop(PQ)
        if u in nodemappings:
            continue
        if v in matched:
            v_m = max((w for w in range(n2) if w not in matched), key=lambda x: W[u][x], default=None)
            if v_m is not None:
                heapq.heappush(PQ, (-W[u][v_m], u, v_m))
                mate[u] = v_m
                wt[u] = W[u][v_m]
            continue

        # Mark (u, v) as matched
        nodemappings[u] = v
        matched.add(v)

        # Update weights based on 2-hop neighbors
        neighbors_u = get_2hop_neighbors(u, graph1)
        neighbors_v = get_2hop_neighbors(v, graph2)
        
        for u_prime in neighbors_u:
            if u_prime in nodemappings:
                continue
            for v_prime in neighbors_v:
                if v_prime in matched:
                    continue
                # Update weight for u_prime and v_prime based on 2-hop neighbors
                W[u_prime][v_prime] += W[u][v] + label_similarity(u_prime, v_prime)
                if W[u_prime][v_prime] > wt[u_prime]:
                    mate[u_prime] = v_prime
                    wt[u_prime] = W[u_prime][v_prime]
                    heapq.heappush(PQ, (-W[u_prime][v_prime], u_prime, v_prime))

    return nodemappings

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description="Script for pos-processing logs"
        )

    parser.add_argument(
        '--testset', 
        type=str, 
        default='aids', 
        help="[linux|imdb|aids|ogbg-molhiv|ogbg-molpcba|ogbg-code2]"
    )

    parser.add_argument(
        '--func_set', 
        type=str, 
        default='linux', 
        help="[linux|imdb|aids|ogbg-molhiv|ogbg-molpcba|ogbg-code2|mixture]"
    )

    args = parser.parse_args()
    layer_map = {'aids':0, 'imdb':2, 'linux':0, 'ogbg-molhiv':1, 'ogbg-molpcba':0, 'ogbg-code2':1}
    layer_val = layer_map[args.testset]

    evaluate(args.testset, args.func_set, layer_val, num_processes=96)