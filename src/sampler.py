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

"""Class for sampling new programs."""
from collections.abc import Collection, Sequence
# import vertexai
# from vertexai.preview.language_models import CodeGenerationModel
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import numpy as np
from typing import Any
import programs_database
import evaluator
import time
from logger import Logger
import re
import math
import pickle
import copy
import os
import json

GOOGLE_API_KEY= 'YOUR_API_KEY' 
genai.configure(api_key=GOOGLE_API_KEY)
prompts=0
now=time.time()

parameters = {
    "temperature": 0.99,
    "max_output_tokens": 2048 
}

class LLM:
  """Language model that predicts continuation of provided source code."""

  def __init__(self, samples_per_prompt: int) -> None:
    self._samples_per_prompt = samples_per_prompt
    self.total_token_count = 0

  def _draw_sample(self, prompt: str) -> str:
    """Returns a predicted continuation of `prompt`."""
    global prompts
    code_model = genai.GenerativeModel(model_name="models/gemini-1.5-pro") 
    response = code_model.generate_content(prompt,
    generation_config=genai.types.GenerationConfig(
        # Only one candidate for now.
        candidate_count=1,# self._samples_per_prompt -> not implemented in api, gives 400 error code
        max_output_tokens=parameters['max_output_tokens'],
        temperature=parameters['temperature'],
    ),safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT:HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:HarmBlockThreshold.BLOCK_ONLY_HIGH
    })
      
    prompts+=1
    print("Token counts: ", response.usage_metadata)
    self.total_token_count += response.usage_metadata.total_token_count
    return response.text

  def draw_samples(self, prompt: str) -> Collection[str]:
    """Returns multiple predicted continuations of `prompt`."""
    return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)], self.total_token_count


class Sampler:
  """Node that samples program continuations and sends them for analysis."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      evaluators: Sequence[evaluator.Evaluator],
      samples_per_prompt: int,
      logger: Logger,
      input_ids: Sequence[Any],
      inputs: Sequence[Any],
      test_inputs: Sequence[Any], 
      test_ground_truth: Sequence[Any],
      upper_bound_init: float,
      dataset:str,
      store_suffix:str
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self.samples_per_prompt = samples_per_prompt
    self._llm = LLM(samples_per_prompt)
    self._logger = logger
    self._counter = 0
    self.total_token_count = 0
    self.input_ids = input_ids
    self._inputs = inputs
    self.upper_bound_init = upper_bound_init
    self.best_upper_bound = upper_bound_init
    self.test_inputs = test_inputs
    self.test_ground_truth = test_ground_truth
    self.func_budget = 15
    self.function_scores_list = []
    self.function_list = []
    self.function_sample = [] #* needed because priority reads from temp.py, fails to run correctly on directly using program
    self.test_results = []
    self.dataset = dataset
    self.test_after_interval = False
    self.store_suffix = store_suffix

  def sample(self):
    """Continuously gets prompts, samples programs, sends them for analysis."""
    count = 0
    num_llm_calls = 0
    start_time=time.time()
    global now, prompts
    overall_best_upper_bound = math.inf
    while count < 200:#* equivalent to epochs
      start_submod = time.time()
      test_time = 0
      while((time.time() - start_submod) <= self._database._config.submod_reset_time):
        # to handle synstax errors from text_to_program(text)
        try:
          prompt = self._database.get_prompt()
        except: 
          continue
        if(time.time()-now<60 and prompts>=350):
          print("SLEEPING NOW !")
          time.sleep(60+(now-time.time())+10)
          now = time.time()
          prompts=0
        if(time.time()-now>60):
          now = time.time()
          prompts = 0
        samples, tokens_consumed = self._llm.draw_samples(prompt.code)
        num_llm_calls += self.samples_per_prompt
        print('token consumed: ', tokens_consumed)
        self.total_token_count += tokens_consumed
        print(f"Total tokens consumed: {self.total_token_count}")
  
        for sample in samples:
          # print("sample code is:",sample)
          sample=sample.lstrip("```python")
          sample=sample.rstrip("```")
          
          # Remove any remaining ``` tokens
          sample = sample.replace("```", "")
          # Remove unwanted tokens or patterns (example: remove everything after "Explanation:")
          sample = re.sub(r'\n*\*\*Explanation:.*', '', sample, flags=re.DOTALL)
          # Remove unwanted tokens or patterns (example: remove everything after "Key improvements:")
          if('Key improvements' in sample):
            sample = re.sub(r'\n*Key improvements.*', '', sample, flags=re.DOTALL)
          
          if('Key Improvements' in sample):
            sample = re.sub(r'\n*Key Improvements.*', '', sample, flags=re.DOTALL)

          functions = re.findall(r"(def\s+[\w_]+\s*\(.*?\):.*?return\s+[^\n]+)", sample, re.DOTALL)
      
          if functions:
              # Get the last function with its return statement
              last_function = functions[-1]
              # Find where this function ends in the original response
              end_index = sample.rfind(last_function) + len(last_function)
              # Return text up to the end of the last function
              sample = sample[:end_index]
          
          # print("sample code after post-processing:",sample)

          with open('temp.py', 'w') as file:
            
            file.write("import numpy as np"+"\n"+ "import itertools\n" + "import networkx as nx\n" + sample)
            file.flush()
    
          #log results here
          selected_input_ids = None
          chosen_evaluator = np.random.choice(self._evaluators)
          marginal_gain, scores, error_flag, current_upper_bound, program = chosen_evaluator.analyse(
              sample, prompt.island_id, prompt.version_generated, self.upper_bound_init, self.func_budget)
          self.best_upper_bound = current_upper_bound

          if(error_flag == 0):
            #put here to keep correspondance between scores and functions: program is sample_to_program(function)
            self.function_list.append(program)
            self.function_scores_list.append(scores)
            self.function_sample.append(sample)

          if(len(scores) == 0):
            print("########## Check Sample! ##########")
            print("Sample: \n", sample)
            print('Error Flag: ', error_flag)
            print("##########")
                
          if(selected_input_ids is None):
            selected_input_ids = self.input_ids
          #* log results: only log for the graphs that are not discarded
          log_time = time.time() - start_time - test_time
          
          self._logger.log(selected_input_ids, prompt, sample, error_flag, marginal_gain, scores, log_time, count, num_llm_calls)
          with open(f'train_info_{self.dataset}{self.store_suffix}.txt', 'a') as tfile:
            tfile.write('Epoch: {}, LLM Calls: {}, Training Time Since Start: {}, Current Best Avg Upper Bound: {}\n'.format(count, num_llm_calls, log_time, current_upper_bound))
        
        
        if num_llm_calls%8 == 0:
          if(self.best_upper_bound < overall_best_upper_bound):
            overall_best_upper_bound = self.best_upper_bound
            print(f"Overall Best Upper Bound: {overall_best_upper_bound}")
            with open(f'best_epoch_llm_calls_{self.dataset}{self.store_suffix}.txt', 'w') as efile:
              efile.write(f"Overall Best Upper Bound: {overall_best_upper_bound}")
              efile.write(f"Epoch: {count}")
              efile.write(f"LLM Calls: {num_llm_calls}")
              efile.write(f"Time Since Start: {time.time() - start_time - test_time}")

          if(self.test_after_interval):
            test_start = time.time()
            step_time = time.time() - start_time
            test_scores = chosen_evaluator.test(self.function_sample, self.function_list, self.test_inputs, self.test_ground_truth, count, num_llm_calls, self.dataset, 'before_submod')
            test_time += time.time()-test_start
            # Save test scores vs count, llm_calls, total_tokens in a dictionary
            self.test_results.append({
                'epoch': count,
                'llm_calls': num_llm_calls,
                'time_step': step_time,
                'total_tokens': self.total_token_count,
                'rmse': test_scores[0],
                'mae': test_scores[1],
                'num_gt': test_scores[2],
                'before_submod': 'Yes'
            })
            print(f"######### Before Submod => Epoch: {count} ; LLM Calls: {num_llm_calls} - Test Scores: RMSE - {test_scores[0]}, MAE: {test_scores[1]}, Num Gt: {test_scores[2]}/{len(self.test_inputs)} ########")
            if not os.path.exists('./plots'):
              os.makedirs('./plots')

            with open(f'./plots/results_epoch_{self.dataset}{self.store_suffix}.pkl', 'wb') as f:
              pickle.dump(self.test_results, f)
          else: 
            save_path = f'best_functions/{self.dataset}/before_submod'
            # Create the directory if it does not exist
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        
            # Write the functions to the file
            with open(f"{save_path}/best_b_functions_{count}_{num_llm_calls}{self.store_suffix}.txt", "w") as file:
              for i, func in enumerate(self.function_sample):
                file.write(f"Function {i+1}:\n")
                file.write(func)
                file.write("\n\n")

      # few checks        
      are_rows_same = all(row == self.function_scores_list[0] for row in self.function_scores_list)
      print("All rows are the same:" if are_rows_same else "Rows are not the same.")
      assert len(self.function_scores_list) == len(self.function_list), f"Shape mismatch between function_scores_list and function_list- Len: {len(self.function_scores_list)}, {self.function_scores_list}; Len: {len(self.function_list)}, {self.function_list}"
      # checks end
            
      # reset all islands  
      self._database.reset_submod()

      best_b_functions = []     
      best_b_functions_idx = [] 
      best_b_functions_str = []
      best_b_function_scores = []

      upper_bound_results = np.mean(np.array(self.function_scores_list), axis=1)
      min_idx = np.argmin(upper_bound_results)

      best_b_functions.append(self.function_list[min_idx])
      best_b_functions_str.append(self.function_sample[min_idx])
      best_b_functions_idx.append(min_idx)
      best_b_function_scores.append(self.function_scores_list[min_idx])
  
      # run greedy

      # Step 2: Greedily select the remaining functions up to the budget b
      current_average_upper_bound = upper_bound_results[min_idx]
      current_min_scores = np.array(self.function_scores_list[min_idx])

      print('Num of error free functions: ', len(self.function_list))

      # register initial program
      island_id = np.random.randint(len(self._database._islands))
      score_array = {}
      for j, current_input in enumerate(self._inputs):
        current_input=tuple([tuple([tuple(current_input[0][0]), tuple(current_input[0][1])]),\
        tuple([tuple(current_input[1][0]), tuple(current_input[1][1])])])
        score_array[current_input] = current_average_upper_bound
      self._database.register_program(self.function_list[min_idx], island_id, score_array)

      for _ in range(1, min(self.func_budget, len(self.function_list))):
        best_function = None
        best_upper_bound_reduction = 0
        best_function_scores = None
        best_function_idx = None

        for i, scores in enumerate(self.function_scores_list):
          if i in best_b_functions_idx:
              continue
          # Calculate marginal RMSE reduction if adding this function
          min_scores = np.minimum(current_min_scores, np.array(scores))
          upper_bound_reduction = current_average_upper_bound - np.mean(min_scores)
          # print("Upper bound reduction for", col, ":", upper_bound_reduction)
          if upper_bound_reduction > best_upper_bound_reduction:
            best_function_idx = i
            best_function = self.function_list[i]
            best_upper_bound_reduction = upper_bound_reduction
            best_function_scores = scores
        
        # Update current RMSE and metrics with the newly added function
        if best_function_idx is not None:
          best_b_functions_idx.append(best_function_idx)
          best_b_functions.append(best_function)
          best_b_functions_str.append(self.function_sample[best_function_idx])
          best_b_function_scores.append(best_function_scores)
          current_min_scores = np.minimum(current_min_scores, best_function_scores)
          current_average_upper_bound = np.mean(current_min_scores)

          # Step 6: Assign best function to an island
          # Randomly sample an island from the list of islands
          island_id = np.random.randint(len(self._database._islands))
          # Store the current function in the selected island
          score_array = {}
          for j, current_input in enumerate(self._inputs):
            current_input=tuple([tuple([tuple(current_input[0][0]), tuple(current_input[0][1])]),\
            tuple([tuple(current_input[1][0]), tuple(current_input[1][1])])])
            score_array[current_input] = best_upper_bound_reduction
          
          self._database.register_program(best_function, island_id, score_array)

      # Update the final list of selected b functions
      self.function_list = best_b_functions
      self.function_scores_list = best_b_function_scores
      self.function_sample = best_b_functions_str

      #Test the selected functions: timesteps, llm_calls, token_consumed vs rmse, mae, number of ground truth achieved

      if(len(best_b_functions) > 0 and self.test_after_interval):
        epoch_time = time.time()-start_time-test_time
        print('Num functions: ', len(best_b_functions))
        test_scores = chosen_evaluator.test(best_b_functions_str, best_b_functions, self.test_inputs, self.test_ground_truth, count, num_llm_calls, self.dataset, 'after_submod')
        # Save test scores vs count, llm_calls, total_tokens in a dictionary
        self.test_results.append({
            'epoch': count,
            'llm_calls': num_llm_calls,
            'time_step': epoch_time,
            'total_tokens': self.total_token_count,
            'rmse': test_scores[0],
            'mae':test_scores[1],
            'num_gt':test_scores[2],
            'before_submod': 'No'
        })
        print(f"######### After Submod => Epoch: {count} ; LLM Calls: {num_llm_calls} - Test Scores: RMSE - {test_scores[0]}, MAE: {test_scores[1]}, Num Gt: {test_scores[2]}/{len(self.test_inputs)} ########")
        # Save the dictionary as a pickle file
        with open(f'plots/results_epoch_{self.dataset}{self.store_suffix}.pkl', 'wb') as f:
            pickle.dump(self.test_results, f)

      elif(len(best_b_functions) > 0):
        save_path = f'best_functions/{self.dataset}/after_submod'
        # Create the directory if it does not exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        # Write the functions to the file
        with open(f"{save_path}/best_b_functions_{count}_{num_llm_calls}{self.store_suffix}.txt", "w") as file:
          for i, func in enumerate(best_b_functions_str):
            file.write(f"Function {i+1}:\n")
            file.write(func)
            file.write("\n\n")

      count += 1
