import json
import ast
import copy
from typing import Any
import importlib
from openpyxl import Workbook
import ast
import numpy as np
# from feature import *
import math
import re
from typing import Dict, List, Tuple
import os
import pickle
import numpy as np
import networkx as nx
import itertools
import heapq
import argparse
# wb = Workbook()
# ws = wb.active

########## HELPER FUNCTIONS #########

def reduce_to_unique_functions(func_dict: Dict[int, Dict]) -> Dict[int, Dict]:
    unique_functions = {}
    seen_functions = {}  # To map unique function codes to their function number in unique_functions
    total_count = len(func_dict)
    print("Total functions extracted: ", func_dict)
    unique_count = 0

    for func_num, data in func_dict.items():
        func_code = data['function']
        # Check if the function code is unique
        if func_code not in seen_functions:
            # If unique, add it to unique_functions and record its function number
            seen_functions[func_code] = func_num
            unique_functions[func_num] = {
                'function': func_code,
            }         
            unique_count += 1
        else:
            # If duplicate, skip
            continue

    # Calculate redundancy percentage: -1 indicates empty log
    redundancy_percentage = (1 - unique_count / total_count) * 100 if total_count != 0 else -1
    if(redundancy_percentage != -1):
        print(f"Redundancy Percentage: {redundancy_percentage:.2f}%")
    else:
        print("Empty log or all erroneous programs. Check logs!!")
    return unique_functions


def extract_function_code(response: str, function_name: str) -> str:
    # Find the start of the desired function
    start_index = response.find(f"def {function_name}(")
    if start_index == -1:
        return "Function not found"
    
    # From the start of the function, look for the next function definition or the end of the response
    end_index = response.find("\ndef ", start_index + 1)
    
    # If no other function is found after this one, capture until the end of the response
    if end_index == -1:
        end_index = len(response)
    
    # Extract and return the function definition
    return response[start_index:end_index].strip()

if __name__ == "__main__": 
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(
        description="Script for pos-processing logs"
    )

    parser.add_argument(
        '--log_path', 
        type=str, 
        # default='logs/submodularity/', 
        help="absolute path where logs are being stored"
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='linux', 
        help="[linux|imdb|aids|ogbg-molhiv|ogbg-molpcba|ogbg-code2]"
    )
    args = parser.parse_args()

    
    with open(args.log_path, "r") as file:
        data = json.load(file)

    func_dict = {}
    counter = 0
    # Iterate over each dictionary block in the JSON file
    for entry_index, entry in enumerate(data):
        # Check if 'error' is False and then extract code
        if not entry.get("error", True):
            response_code = entry["response"]
            # print(response_code)
            print(f"\nExecuting code block {entry_index}...")
            function_pattern = re.compile(r'def (priority_v\d+)\(')
            for match in function_pattern.finditer(response_code):
                func_name = match.group(1)
                # print("Func name: ", func_name)
                func_code = extract_function_code(response_code, func_name)
                if(func_code != None): # remove functions that call ancestors for correctness after re-naming unique functions
                    if('weights = priority_v1' not in func_code and 'weights = priority_v0' not in func_code):
                        func_dict[counter] = {'function':func_code}
                        counter+=1

    unique_functions = reduce_to_unique_functions(func_dict)

    outpath = './top_fns_transfer/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        print(f"Directory created: {outpath}")
    else:
        print(f"Directory already exists: {outpath}")
    
    outpath1 = './val_transfer/'
    if not os.path.exists(outpath1):
        os.makedirs(outpath1)
        print(f"Directory created: {outpath1}")
    else:
        print(f"Directory already exists: {outpath1}")

    # Write results to a text file
    fn_counter = 0
    nf = open(f'{outpath1}functions_{args.dataset}.txt', 'w')
    with open(f'{outpath}{args.dataset}.py', 'w') as file:
        file.write(f'#Path: {args.log_path}\n')
        file.write(f'#Num Unique Functions in logs: {len(unique_functions)}\n\n')
        for key, value in unique_functions.items():
            # Replace the first line of the function definition
            function_code = value['function']
            function_code = re.sub(r'def priority_v\d+', f'def priority_v{key}', function_code, count=1)
            file.write(function_code + "\n\n")
            nf.write(f'priority_v{key}\n')
            fn_counter +=1
            if(fn_counter==65): #extra functions so that if codes through runtime errors on test or validation examples, they can be discarded. Otherwise we aim to filter from top-50
                break
    nf.close()