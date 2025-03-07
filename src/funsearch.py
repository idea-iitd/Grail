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

"""A single-threaded implementation of the FunSearch pipeline."""
from collections.abc import Sequence
from typing import Any

import code_manipulation
import config as config_lib
import evaluator
import programs_database
import sampler
from logger import Logger
import sys
import traceback
#input reading
import ast
import random
import numpy as np
import argparse
from ast import literal_eval
import math

def _extract_function_names(specification: str) -> tuple[str, str]:
  """Returns the name of the function to evolve and of the function to run."""
  # Read code from specification file.

  run_functions = list(
      code_manipulation.yield_decorated(specification, 'funsearch', 'run'))
  if len(run_functions) != 1:
    raise ValueError('Expected 1 function decorated with `@funsearch.run`.')
  evolve_functions = list(
      code_manipulation.yield_decorated(specification, 'funsearch', 'evolve'))
  if len(evolve_functions) != 1:
    raise ValueError('Expected 1 function decorated with `@funsearch.evolve`.')
  return evolve_functions[0], run_functions[0]


def main(specification: str, dataset: str, inputs: Sequence[Any], input_ids: Sequence[Any], test_inputs: Sequence[Any], test_ground_truth: Sequence[Any], config: config_lib.Config, layer_val:int, store_suffix:str):
  """Launches a FunSearch experiment."""
  function_to_evolve, function_to_run = _extract_function_names(specification)

  template = code_manipulation.text_to_program(specification)
  database = programs_database.ProgramsDatabase(
      config.programs_database, template, function_to_evolve)
  
  
  evaluators = []
  for _ in range(config.num_evaluators):
    evaluators.append(evaluator.Evaluator(
        database,
        template,
        function_to_evolve,
        function_to_run,
        inputs,
        layer_val,
        store_suffix
    ))

  # We send the initial implementation to be analysed by one of the evaluators.
  initial = template.get_function(function_to_evolve).body

  _, ged_preds, _, _, _ = evaluators[0].analyse(initial, island_id=None, version_generated=None, upper_bound_init=math.inf, func_budget=15)
  upper_bound_init = np.mean(np.array(ged_preds))
  print('Init Ged: {}, Upper bound: {}'.format(ged_preds, upper_bound_init))
  logger = Logger(dataset, store_suffix)
  samplers = [sampler.Sampler(database, evaluators, config.samples_per_prompt, logger, input_ids, inputs, test_inputs, test_ground_truth, upper_bound_init, dataset, store_suffix)
              for _ in range(config.num_samplers)]
  total_token_count = 0

  try:
    for s in samplers:
      s.sample()
      total_token_count+= s.total_token_count
    print(f"Total tokens consumed: {total_token_count}")
  except Exception as e:
    # Catch any exception and ensure logs are saved
    print(f"An error occurred: {e}")
    traceback.print_exc()
    logger.finalize_logging()
    print(f"Total tokens consumed: {total_token_count}")
    sys.exit(1)  # Exit with an error code
  except KeyboardInterrupt:
    # This will be caught by the signal handler automatically
    print(f"Total tokens consumed: {total_token_count}")
    pass
  finally:
    # Ensure logs are saved even if there's no error but the loop is exited
    print(f"Total tokens consumed: {total_token_count}")
    logger.finalize_logging()
  print('############## After sampling ##############')
  logger.save_to_json()

with open("graph.txt", 'r') as f:
    specification = f.read()
config = config_lib.Config()

  
def process_file(file_path):
  with open(file_path, 'r') as file:
    content = file.read()
    g_pairs_str = content.split("g_pairs: ")[1].split("node_features: ")[0].strip()
    nf_str = content.split("node_features: ")[1].split("ground_truth: ")[0].strip()
    ground_truth_str = content.split("ground_truth: ")[1].strip()
    g_pairs = ast.literal_eval(g_pairs_str)
    nfs = ast.literal_eval(nf_str)
    ground_truth = ast.literal_eval(ground_truth_str)
    data = []
    for i, (g_pair, nfs, ged) in enumerate(zip(g_pairs, nfs, ground_truth)):
      # Assume g_pair is of the form [G1, G2] and node features are extracted as nf1 and nf2
      G1, G2 = g_pair
      nf1, nf2 = nfs  # Replace with actual logic for node features if available
      data.append([G1, G2, nf1, nf2, ged])  
  return data, ground_truth

parser = argparse.ArgumentParser(
        description="Script for pos-processing logs"
    )

parser.add_argument(
    '--dataset', 
    type=str, 
    default='aids', 
    help="[linux|imdb|aids|ogbg-molhiv|ogbg-molpcba|ogbg-code2|mixture]"
)

parser.add_argument(
    '--train_path', 
    type=str, 
    default='../data/train', 
    help="path of folder where all train data is kept"
)

parser.add_argument(
    '--test_path', 
    type=str, 
    default='../data/test', 
    help="path of folder where all test data is kept"
)

parser.add_argument(
    '--suffix', 
    type=str, 
    default='', 
    help="suffix for storing logs etc. (if needed)"
)

args = parser.parse_args()

#only to be used in testing, selected based on upper bound reduction on validation set using top 50 unique functions from logs of multiple runs
layer_map = {'aids':0, 'imdb':2, 'linux':0, 'ogbg-molhiv':1, 'ogbg-molpcba':0, 'ogbg-code2':1}
layer_val = layer_map[args.dataset]


train_data, _=process_file(args.train_path + '/' + args.dataset + '.txt')
test_data, test_ground_truth=process_file(args.test_path + '/' + args.dataset + '_test.txt')

input_ids = list(range(len(train_data)))

#Note: ground truth for train data is not used. It is only used for test data evaluation
main(specification, args.dataset, train_data, input_ids, test_data, test_ground_truth, config, layer_val, args.suffix)
