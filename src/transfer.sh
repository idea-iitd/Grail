#VAL-TEST PAIR

#!/bin/bash

# Define arrays of datasets
datasets=("aids" "ogbg-molpcba" "imdb" "linux" "ogbg-code2" "ogbg-molhiv")

# Loop over valset and testset combinations
for valset in "${datasets[@]}"; do
  for funcset in "${datasets[@]}"; do
    echo "Running commands for valset/testset=$valset and funcset=$funcset"
    
    # Run the commands
    python parallel_val.py --valset "$valset" --func_set "$funcset"
    python calc_wo_gt.py --valset "$valset" --func_set "$funcset"
    python parallel_test.py --testset "$valset" --func_set "$funcset"
  done
done
