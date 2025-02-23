# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for evaluating programs proposed by the Sampler."""
import ast
import signal
from typing import Any
from collections.abc import Sequence
from timeout_decorator import timeout
import copy
from typing import Any
import time
import numpy as np
import itertools
import _testmultiphase
import code_manipulation
import programs_database
import importlib
import astunparse
import networkx as nx
import math
import random
from operator import itemgetter
import os
import heapq
import math

class _FunctionLineVisitor(ast.NodeVisitor):
  """Visitor that finds the last line number of a function with a given name."""

  def __init__(self, target_function_name: str) -> None:
    self._target_function_name: str = target_function_name
    self._function_end_line: int | None = None

  def visit_FunctionDef(self, node: Any) -> None:  # pylint: disable=invalid-name
    """Collects the end line number of the target function."""
    if node.name == self._target_function_name:
      self._function_end_line = node.end_lineno
    self.generic_visit(node)

  @property
  def function_end_line(self) -> int:
    """Line number of the final line of function `target_function_name`."""
    assert self._function_end_line is not None  # Check internal correctness.
    return self._function_end_line

def _trim_function_body(generated_code: str) -> str:
  """Extracts the body of the generated function, trimming anything after it."""
  # print("in fn : ",generated_code)
  if not generated_code:
    return ''
  code = f'def fake_function_header():\n{generated_code}'
  tree = None
  # We keep trying and deleting code from the end until the parser succeeds.
  # print("code: ", code)
  while tree is None:
    try:
      
      tree = ast.parse(code)
    except SyntaxError as e:
      # print("syntax err")
      # code = '\n'.join(code.splitlines()[:e.lineno - 1])
      tree=''
      code=''
      pass
  if not code:
    # Nothing could be saved from `generated_code`
    return ''

  visitor = _FunctionLineVisitor('fake_function_header')
  visitor.visit(tree)
  body_lines = code.splitlines()[1:visitor.function_end_line]
  return '\n'.join(body_lines) + '\n\n'

def _trim_function_body1(generated_code: str) -> str:
    #* code to remove extra information
  
    index = generated_code.find('```')
  
    # If the substring is found, remove everything from that point onwards
    if index != -1:
      generated_code = generated_code[:index]
    
    if generated_code.lstrip()=="return 0.0":
        return _trim_function_body(generated_code)
        
    print("From evaluator: generated code: ", generated_code)
   
    try:
      tree = ast.parse(generated_code)
    except:
      s=_trim_function_body(generated_code)
      return s
    
    if not isinstance(tree, ast.Module) or not tree.body:
        raise ValueError("Invalid function string")

    for i, node in enumerate(tree.body):
      if isinstance(node, ast.FunctionDef) and node.name.startswith('priority'):
        function_def = tree.body[i]
        # print(f"Function {i}:")
        # print(f"  Name: {node.name}")
        # print(f"  Arguments: {[arg.arg for arg in node.args.args]}")
        # print(f"  Body: {ast.dump(node.body[0], indent=4)}")
        # print()
        break

    name = function_def.name
    args = ', '.join(arg.arg for arg in function_def.args.args)

    # Remove docstring from the body
    if function_def.body and isinstance(function_def.body[0], ast.Expr) and isinstance(function_def.body[0].value, ast.Str):
        docstring_len = len(astunparse.unparse(function_def.body[0]))
        body = astunparse.unparse(function_def.body[1:])
    else:
        docstring_len = 0
        body = astunparse.unparse(function_def.body)

    return_type = None

    if function_def.returns is not None:
        return_type = astunparse.unparse(function_def.returns)

    docstring = ast.get_docstring(function_def)
    return str(body)


def _sample_to_program(
    generated_code: str,
    version_generated: int | None,
    template: code_manipulation.Program,
    function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
  body = _trim_function_body1(generated_code)

  if body=='':
     return None,''
  
  if version_generated is not None:
    body = code_manipulation.rename_function_calls(
        body,
        f'{function_to_evolve}_v{version_generated}',
        function_to_evolve)

  program = copy.deepcopy(template)
  evolved_function = program.get_function(function_to_evolve)
  evolved_function.body = body
  
  return evolved_function, str(program)



def block_children(scores, admissible_set, new_element):
  """Modifies `scores` to -inf for elements blocked by `new_element`."""
  n = admissible_set.shape[-1]
  powers = np.array([3 ** i for i in range(n - 1, -1, -1)], dtype=np.int32)

  invalid_vals_raw = {
      (0, 0): (0,),
      (0, 1): (1,),
      (0, 2): (2,),
      (1, 0): (1,),
      (1, 1): (0, 1, 2),
      (1, 2): (1, 2),
      (2, 0): (2,),
      (2, 1): (1, 2),
      (2, 2): (0, 1, 2),
  }
  invalid_vals = [[np.array(invalid_vals_raw[(i, j)], dtype=np.int32)
                  for j in range(3)] for i in range(3)]

  # Block 2^w elements with the same support as `new_element`.
  w = np.count_nonzero(new_element)
  all_12s = np.array(list(itertools.product((1, 2), repeat=w)), dtype=np.int32)
  blocking = np.einsum('aw,w->a', all_12s, powers[new_element != 0])
  scores[blocking] = -np.inf

  # Block elements disallowed by a pair of an extant point and `new_element`.
  for extant_element in admissible_set:
    blocking = np.zeros(shape=(1,), dtype=np.int32)
    for e1, e2, power in zip(extant_element, new_element, powers):
      blocking = (blocking[:, None] + (invalid_vals[e1][e2] * power)[None, :]
                  ).ravel()
    scores[blocking] = -np.inf  

def convert_to_adjmatrix(edge_list1, edge_list2, max_node) -> list[list[int]]: 
  adjacency_matrix1 = [[0] * max_node for _ in range(max_node)]
  adjacency_matrix2 = [[0] * max_node for _ in range(max_node)]

  for edge in zip(edge_list1[0], edge_list1[1]):
    source, destination = edge
    adjacency_matrix1[source][destination] = 1
  
  for edge in zip(edge_list2[0], edge_list2[1]):
    source, destination = edge
    adjacency_matrix2[source][destination] = 1

  ged=2*max_node -2 - max(max(edge_list1[0]), max(edge_list1[1])) - max(max(edge_list2[0]), max(edge_list2[1]))
  return adjacency_matrix1, adjacency_matrix2, ged 

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

def evaluate(desc : tuple[list[list[int]], list[list[int]], list[list[float]], list[list[float]], int]):
  """Returns the graph edit distance based on the node mappings"""
  g1,g2,nf1,nf2,layer_val = desc
  max_node = max(len(nf1),len(nf2))

  g1, g2,ged = convert_to_adjmatrix(g1,g2,max_node)
  # print('Max Node: ', max_node)
 
  x = len(g1[0])
  y1 = len(nf1)  # y is the number of nodes with features
  y2 = len(nf2)

  nf1 = nf1 + [[-1]] * (x - y1)
  nf2 = nf2 + [[-1]] * (x - y2)

  # for labelled
  labels1, labels2 = assign_labels_based_on_features(nf1,nf2) 

  weights = [[0.0] * max_node for _ in range(max_node)]
  for i in range(max_node):
        for j in range(max_node):
            # Handle dummy nodes (explicitly excluded from labels)
            if i >= len(labels1) or j >= len(labels2) or labels1[i] == "eps" or labels2[j] == "eps":
                weights[i][j] = 0.0
            elif labels1[i] == labels2[j]:
                weights[i][j] = 1.0
            elif labels1[i] != labels2[j]:
                weights[i][j] = 0.0
  #normalise
  for i in range(max_node):
        row_sum = sum(weights[i][:max_node]) # Only normalize within valid range of n2
        if row_sum > 0:
            for j in range(max_node):
                weights[i][j] /= row_sum

  wts = priority(copy.deepcopy(g1), copy.deepcopy(g2), weights)

  if wts is not None:
    for _ in range(layer_val): 
      x = priority(copy.deepcopy(g1), copy.deepcopy(g2), wts)
      if x is None:
        break
      wts=x
  
  nodemap = solve(copy.deepcopy(g1),copy.deepcopy(g2), copy.deepcopy(labels1), copy.deepcopy(labels2), wts)  
  # print("Nodemap:", nodemap)
  cost = 0
  y=0
  
  for v1 in range(len(g1)):
     mapped_v1 = nodemap[v1]
     if labels1[v1]!="eps" and labels2[mapped_v1]!="eps" and labels1[v1]!=labels2[mapped_v1]:
        y+=1

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

  cost = cost/2+ ged+ y

  print("Nodemap:", nodemap)
  print("Cost:", cost)
  return cost 


def solve(
    graph1: list[list[int]], 
    graph2: list[list[int]],
    labels1, labels2,
    weights: list[list[float]]
) -> dict:
    """Builds a mapping between 2 labelled graphs using Neighbor Bias Mapper and label similarity.
    
    Args:
        graph1: adjacency matrix of graph 1
        graph2: adjacency matrix of graph 2
        weights: A weight matrix representing the likelihood of mapping nodes between `graph1` and `graph2`.
    
    Returns:
        nodemappings: A dictionary mapping nodes of graph1 to graph2.
    """
    def label_similarity(u, v):
        return 1 if labels1[u] == labels2[v] else 0
    n1, n2 = len(graph1), len(graph2)

    
    # Initial priority weights adjusted with label similarity
    W=weights
    
    # Priority queue to store potential matches with highest priority first
    PQ = []
    mate = [-1] * n1  # stores best match in graph2 for each node in graph1
    wt = [-math.inf] * n1  # stores the best weight for each node in graph1

    # Initialize PQ with best initial matches
    for u in range(n1):
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

    # Process the priority queue
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

        # Update weights based on 1-hop neighbors
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

# @funsearch.evolve
def priority(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:

  with open('temp.py', 'r') as file:
    lines = file.read()

  functions = {}

  # Execute the code in temp.txt
  exec(lines, functions)

  max_node = (max(max(graph1[0]), max(graph1[1]),max(graph2[0]),max(graph2[1])) + 1)
  result = [[0.0] * max_node for _ in range(max_node)]

  for func_name, func in functions.items():
    if func_name.startswith('priority_v2') and callable(func):
      # Call the function
      result = func(graph1, graph2, weights)
      return result

  for func_name, func in functions.items():
    if func_name.startswith('priority_v1') and callable(func):
      # Call the function
      result = func(graph1, graph2, weights)
      return result

  return result

def timeout_handler():
  raise TimeoutError("Execution timed out")

class Sandbox:
  """Sandbox for executing generated code."""
  @timeout(60, timeout_exception=TimeoutError)
  def run(
        self,
        program: str,
        function_to_run: str,
        test_input: str,
        timeout_seconds: int,
        testing: bool,
        layer_val:int
    ) -> tuple[Any, bool]:
      """Returns `function_to_run(test_input)` and whether execution succeeded."""
      # print(test_input)
      #! replace G1, G2 with test_input
      G1 = test_input[0]
      G2 = test_input[1]
      nf1=test_input[2]
      nf2=test_input[3]

      if testing:
         code_to_execute = """result = evaluate([{},{},{},{},{}])""".format(G1,G2,nf1,nf2,layer_val)
      else:
          code_to_execute = """result = evaluate([{},{},{},{},{}])""".format(G1,G2,nf1,nf2,0) #use only single layer(layer_val=0) for training for time saving

      # exit(0)
      with open('temp.py', 'r') as file:
        lines = file.read()
      # print("Code executing :", lines)

      # Set up initial values
      execution_succeeded = False
      result = None

      exec_globals = {"np": np, "itertools": itertools, "nx" : nx, "random": random}
      exec_locals = {"block_children" : block_children, 
                     "assign_labels_based_on_features": assign_labels_based_on_features,
                     "convert_to_adjmatrix" :  convert_to_adjmatrix,
                      "priority" : priority, "solve" : solve, "evaluate" : evaluate, "result" : result}

      try:
          # Execute the code within a time limit
          start_time = time.time()
          exec(code_to_execute, exec_globals, exec_locals)
          # print("ok")
          result = exec_locals.get("result")
          print("Code executed successfully, result= ", result)
          execution_time = time.time() - start_time

          # Check if execution succeeded and did not exceed the timeout
          if execution_time < timeout_seconds:
              result = exec_locals.get("result")
              execution_succeeded = True
          else:
              result = f"Execution timed out ({timeout_seconds} seconds)"
      
      except TimeoutError as e:
        # Handle timeout
        result = 0
        print("Timeout")
        execution_succeeded = False


      except:
          # Handle exceptions during execution
          result = f"Error !"
          execution_succeeded = False


      finally:
        # Cancel the alarm (even if execution is completed or failed)
        signal.alarm(0)

      # print("result:",result)

      return result, execution_succeeded
      raise NotImplementedError(
          'Must provide a sandbox for executing untrusted code.')


def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
  """Returns whether the generated function is calling an earlier version."""
  for name in code_manipulation.get_functions_called(program):
    # In `program` passed into this function the most recently generated
    # function has already been renamed to `function_to_evolve` (wihout the
    # suffix). Therefore any function call starting with `function_to_evolve_v`
    # is a call to an ancestor function.
    if name.startswith(f'{function_to_evolve}_v'):
      return True
  return False

def calculate_average_upper_bound(geds):
    return np.mean(geds)

def compute_marginal_gain(programs_list, best_upper_bound, func_budget):
  #instead of taking min here among all existing functions, take marginal gain based on best b.
  if(len(programs_list) <= func_budget):
    best_b_functions_idx = None
    min_values = programs_list.min(axis=0)
    current_average_upper_bound = calculate_average_upper_bound(min_values)
    print("curr_upper_bound: ", current_average_upper_bound)
    if(math.isinf(best_upper_bound)):
      marginal_gain = current_average_upper_bound
      # print("marginal_gain: ", marginal_gain)
    else: 
      marginal_gain = best_upper_bound - current_average_upper_bound #* curr should always be less than or equal to best
  else:
    upper_bound_results = np.mean(programs_list, axis=1)
    min_idx = np.argmin(upper_bound_results)
    # print('Upper Bound Results: ', upper_bound_results)
    # print('Min Idx: ', min_idx)
    best_b_functions_idx = []
    current_min_scores = programs_list[min_idx]
    best_b_functions_idx.append(min_idx)
    current_average_upper_bound = upper_bound_results[min_idx]
    
    # Step 2: Iterate to complete the selection up to the budget
    for j in range(1, func_budget):
      best_function = None
      best_upper_bound_reduction = 0.0
      best_function_scores = None

      for i, scores in enumerate(programs_list):
        if i in best_b_functions_idx:
            continue
        # Calculate marginal RMSE reduction if adding this function
        min_scores = np.minimum(current_min_scores, scores)
        upper_bound_reduction = current_average_upper_bound - calculate_average_upper_bound(min_scores)
        # print('Best upper bound reduction: ', best_upper_bound_reduction)
        # print("Upper bound reduction for", i, ":", upper_bound_reduction)
        if upper_bound_reduction > best_upper_bound_reduction:
          best_function = i
          best_upper_bound_reduction = upper_bound_reduction
          best_function_scores = scores
          # print('here: ', i, best_function)  
      
      # Update current RMSE and metrics with the newly added function
      if best_function is not None:
        best_b_functions_idx.append(best_function)
        current_min_scores = np.minimum(current_min_scores, best_function_scores)
        current_average_upper_bound = calculate_average_upper_bound(current_min_scores)
        # print('Best Upper Bound: ', best_upper_bound)
        # print('current_min_scores: ', current_min_scores)
        # print('current_average_upper_bound: ', current_average_upper_bound)
        # print('best_b_functions_idx: ', best_b_functions_idx)

    # print('Best functions: ', best_b_functions_idx)
    marginal_gain = best_upper_bound - current_average_upper_bound
  return marginal_gain, current_average_upper_bound, best_b_functions_idx
   
class Evaluator:
  """Class that analyses functions generated by LLMs."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      template: code_manipulation.Program,
      function_to_evolve: str,
      function_to_run: str,
      inputs: Sequence[Any],
      layer_val:int,
      store_suffix:str,
      timeout_seconds: int = 60,
  ):
    self._database = database
    self._template = template
    self._function_to_evolve = function_to_evolve
    self._function_to_run = function_to_run
    self._inputs = inputs
    self.best_upper_bound = math.inf
    self._timeout_seconds = timeout_seconds
    self._sandbox = Sandbox()
    self.layer_val = layer_val
    self.store_suffix = store_suffix
    self.best_b_func_idx = None
    self.program_scores = []

  def analyse(
      self,
      sample: str,
      island_id: int | None,
      version_generated: int | None,
      upper_bound_init: float,
      func_budget:int
  ) -> None:
    self.best_upper_bound = min(upper_bound_init, self.best_upper_bound)

    """Compiles the sample into a program and executes it on test inputs."""
    new_function, program = _sample_to_program(
        sample, version_generated, self._template, self._function_to_evolve)
    print("No. of inputs: ", len(self._inputs))
    
    scores_per_test = {}
    scores_per_test_list = []
    test_case_fail = False

    for i, current_input in enumerate(self._inputs):
      #* gets actual ged based on priority
      test_output, runs_ok = self._sandbox.run(
          program, self._function_to_run, current_input, self._timeout_seconds, 0, self.layer_val)
      
      # print(f"test_ouput {i} outside:",test_output, runs_ok)
      if (runs_ok and not _calls_ancestor(program, self._function_to_evolve)
          and test_output is not None):
        if not isinstance(test_output, (int, float)):
          raise ValueError('@function.run did not return an int/float score.')
        current_input=tuple([tuple([tuple(current_input[0][0]), tuple(current_input[0][1])]),\
          tuple([tuple(current_input[1][0]), tuple(current_input[1][1])])])
        
        scores_per_test[current_input] = test_output
        scores_per_test_list.append(test_output)
      else:
        test_case_fail = True
    
    if runs_ok and not test_case_fail:
      self.program_scores.append(scores_per_test_list) #will not append anything if runs_ok is false, then marginal gain should be 0

    # marginal gain upon adding the recent function to the set of generated functions
      marginal_gain_curr_func, current_average_upper_bound, best_b_functions_idx = compute_marginal_gain(np.array(self.program_scores), self.best_upper_bound, func_budget)
      if(best_b_functions_idx is not None):
        self.best_b_func_idx = best_b_functions_idx
     
      self.best_upper_bound = min(self.best_upper_bound, current_average_upper_bound)
    else:
      runs_ok = False
      marginal_gain_curr_func = 0.0
 
    #* hack to handle passing single value 
    score_array = {}
    for i, current_input in enumerate(self._inputs):
      current_input=tuple([tuple([tuple(current_input[0][0]), tuple(current_input[0][1])]),\
        tuple([tuple(current_input[1][0]), tuple(current_input[1][1])])])
      score_array[current_input] = marginal_gain_curr_func

    if runs_ok and (not(math.isnan(marginal_gain_curr_func))): 
      self._database.register_program(new_function, island_id, score_array)

    #* changes made for logging   
    return marginal_gain_curr_func, scores_per_test_list, not runs_ok, self.best_upper_bound, new_function #return scores_per_test_list instead of scores_per_test
  
  def test(
      self,
      functions,
      function_prog,
      test_inputs,
      test_ground_truths,
      epoch, 
      llm_calls,
      dataset,
      submod_time
  ) -> list[int]:
    #rmse, mae, number of graphs for which gt is achieved
    print("##### Testing ##########")
    test_results = []
    scores_per_function = []
    save_path = f'./best_functions/{dataset}/{submod_time}'
    # Create the directory if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    func_file = open(f"{save_path}/best_b_functions_{epoch}_{llm_calls}{self.store_suffix}.txt", "w")

    for i, func in enumerate(functions):
      func_file.write(f"Function {i+1}:\n")
      func_file.write(func)
      scores_per_input = []
      with open('temp.py', 'w') as file:
        file.write("import numpy as np"+"\n"+ "import itertools\n" + "import networkx as nx\n" + func)
        file.flush()
      for j, current_input in enumerate(test_inputs):
        test_output, runs_ok = self._sandbox.run(
            function_prog[i], self._function_to_run, current_input, self._timeout_seconds, 1, self.layer_val)
        if (runs_ok and test_output is not None):
          if not isinstance(test_output, (int, float)):
            raise ValueError('@function.run did not return an int/float score.')
          scores_per_input.append(test_output)
        elif(not runs_ok and len(scores_per_input)>0):#sometimes a function works but fails on few test cases
          scores_per_input.append(math.inf) 

      func_file.write(f'\nInput Scores: {scores_per_input}')
      func_file.write("\n\n")
      scores_per_function.append(scores_per_input)

    #overall score
    scores_per_function = np.array(scores_per_function)
    best_scores = scores_per_function.min(axis=0)
    # print("Best Scores: ", best_scores)
    # print("Test Ground Truths: ", test_ground_truths)
    func_file.write(f'##### Best Scores: {str(best_scores.tolist())}\n\n')
    func_file.write(f'##### Ground Truths: {str(test_ground_truths)}\n')

    assert len(best_scores) == len(test_ground_truths), "Error: Length of best scores and test ground truths do not match."
    for i in range(len(best_scores)):
      if(best_scores[i] < test_ground_truths[i]):
        print(f'Best score less than test_ground_truth for idx {i}, Best-Score: {best_scores[i]}, Test-Ground-Truth: {test_ground_truths[i]}')
    assert all(best_scores[i] >= test_ground_truths[i] for i in range(len(best_scores))), "Error: Some best scores are less than the test ground truths."
    
    rmse = np.sqrt(((best_scores - np.array(test_ground_truths)) ** 2).mean())
    test_results.append(rmse)

    mae = np.abs(best_scores - np.array(test_ground_truths)).mean()
    test_results.append(mae)

    num_matches = np.sum(best_scores == np.array(test_ground_truths))
    test_results.append(num_matches)
    
    func_file.write(f'##### Test Results: RMSE - {test_results[0]}, MAE: {test_results[1]}, Num Gt: {test_results[2]}/{len(test_inputs)}\n\n')

    func_file.close()
    return test_results
      


