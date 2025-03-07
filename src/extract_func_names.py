import re
import argparse

parser = argparse.ArgumentParser(
        description="Script for pos-processing logs"
    )

parser.add_argument(
    '--dataset', 
    type=str, 
    default='aids', 
    help="[linux|imdb|aids|ogbg-molhiv|ogbg-molpcba|ogbg-code2|mixture]"
)

args = parser.parse_args()

dataset = args.dataset

# for dataset in datasets:
# Input and output file paths
input_file = f"./discovered_programs/submod_{dataset}.py"  # Replace with the path to your Python file
if(dataset in ['code2', 'molhiv', 'molpcba']):
    dataset = 'ogbg-'+dataset
output_file = f"./top_15/funcs_{dataset}.txt'"     # File where function names will be saved

# Regular expression to match function definitions of the form 'def priority_v{i}'
function_pattern = r"def\s+(priority_v\d+)\s*\("
all_functions = []

# Read the file and extract all function names and their full bodies
with open(input_file, "r") as file:
    content = file.read()

# Extract all function definitions and their names
matches = re.finditer(function_pattern, content)
for match in matches:
    function_name = match.group(1)
    all_functions.append(function_name)

# Filter out function names that occur in the body of other functions
valid_functions = []
for function_name in all_functions:
    # Check if the function name appears elsewhere in the file (excluding its own definition)
    body_pattern = rf"\b{function_name}\s*\("
    if len(re.findall(body_pattern, content)) == 1:  # It appears only in its own definition
        valid_functions.append(function_name)


# Save valid function names to the output file
with open(output_file, "w") as file:
    for func_name in valid_functions:
        file.write(func_name + "\n")

print(f"Extracted {len(valid_functions)} valid function names and saved them to {output_file}.")