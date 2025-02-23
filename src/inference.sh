#!/bin/bash

# Usage: chmod +x inference.sh ; ./inference.sh <test_dataset_name> <function_dataset_name>
testset="$1"
func_set="$2"

# Check if both arguments are provided
if [ -z "$testset" ] || [ -z "$func_set" ]; then
  echo "Usage: $0 <test_dataset_name> <function_dataset_name>"
  exit 1
fi

# Check if --testset is equal to --func_set or if func_set is "mixture"
if [ "$testset" == "$func_set" ] || [ "$func_set" == "mixture" ]; then
  echo "Running inference.py..."
  python inference.py --testset "$testset" --func_set "$func_set"
else
  echo "Running parallel_test.py..."
  python parallel_test.py --testset "$testset" --func_set "$func_set"
fi
