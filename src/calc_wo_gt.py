import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import combinations
import argparse
import os

#this is new version of calc_greedy.py which doesn't use ground trurhs and calculates average
#reduction in upper bound for val set

def calculate_average_upper_bound(predicted_values):
    """
    Calculate the average upper bound of the predicted values.
    """
    return np.mean(predicted_values)


def read_excel_data(file_path):
    """Read the Excel file and return a DataFrame."""
    df = pd.read_excel(file_path)
    return df

def greedy_selection_with_metrics(df, budget):
    """
    Perform greedy selection of functions based on RMSE reduction, and calculate additional metrics.
    
    Parameters:
    df: pd.DataFrame, the data with graph pair index, ground truth, and function scores.
    budget: int, number of functions to select.
    
    Returns:
    A tuple containing:
    - List of selected functions
    - RMSE achieved with the selection
    - Median of squared errors
    - Count of exact matches (graph pairs with GED)
    """
    # ground_truth = df.iloc[:, 1].values  # Assuming the second column is the ground truth
    score_columns = list(df.columns[2:])  # Columns with function scores
    print("budget:", budget)
    # Step 1: Choose the function with the minimum RMSE
    upper_bound_results = {}
    for col in score_columns:
        scores = df[col].values
        upper_bound_results[col] = calculate_average_upper_bound(scores)
    
    selected_functions = [min(upper_bound_results, key=upper_bound_results.get)]
    current_min_scores = df[selected_functions[0]].values
    current_average_upper_bound = upper_bound_results[selected_functions[0]]
    
    
    # Step 2: Iterate to complete the selection up to the budget
    for i in range(1, budget):
        best_function = None
        best_upper_bound_reduction = 0
        for col in score_columns:
            if (col in selected_functions) or  (100000 in df[col].values):
                continue
            # Calculate marginal RMSE reduction if adding this function
            min_scores = np.minimum(current_min_scores, df[col].values)
            upper_bound_reduction = current_average_upper_bound - calculate_average_upper_bound(min_scores)
            # print("Upper bound reduction for", col, ":", upper_bound_reduction)
            if upper_bound_reduction > best_upper_bound_reduction:
                best_function = col
                best_upper_bound_reduction = upper_bound_reduction
        
        # Update current RMSE and metrics with the newly added function
        if best_function is not None:
            selected_functions.append(best_function)
            current_min_scores = np.minimum(current_min_scores, df[best_function].values)
            current_average_upper_bound = calculate_average_upper_bound(current_min_scores)
    print("Selected functions:", selected_functions)
    print("Current average upper bound:", current_average_upper_bound)
    return selected_functions, current_average_upper_bound

def plot_metrics(df, val_set, func):
    """
    Plot RMSE, median of squared errors, and count of exact matches as functions of budget.
    
    Parameters:
    df: pd.DataFrame, the data with graph pair index, ground truth, and function scores.
    """
    budgets = [15]#extract 15 programs based on submodular marginal gains
    funcs = []
    average_upper_bound = []
    outpath = './top_15/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        print(f"Directory created: {outpath}")
    else:
        print(f"Directory already exists: {outpath}")

    func_file = open(f'{outpath}f_val_set_{val_set}_funcs_{func}.txt', 'w')
    for b in budgets:
        selected_funcs, ub = greedy_selection_with_metrics(df, b)
        for func in selected_funcs:
            func_file.write(func + '\n')
        if(len(budgets) == 1):
            func_file.close()
        else:
            func_file.write(f'######## Budget {b} functions above ###### ')
        funcs.append(selected_funcs)
        average_upper_bound.append(ub)

    if(len(budgets) > 1):
        func_file.close()
   
    print("Average upper bound:", average_upper_bound)
    print("Selected functions:", funcs)
    # Plotting metrics
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot RMSE and median squared error on the primary y-axis
    ax1.plot(budgets, average_upper_bound, marker='o', color='b', label='Average upper bound')
    ax1.set_xlabel('Budget (b)')
    ax1.set_ylabel('Average upper bound')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    
    plt.title('Greedy Selection Metrics vs Budget for Linux val set')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Script for pos-processing logs"
        )

    parser.add_argument(
        '--valset', 
        type=str, 
        default='aids', 
        help="[linux|imdb|aids|ogbg-molhiv|ogbg-molpcba|ogbg-code2]"
    )

    parser.add_argument(
        '--func_set', 
        type=str, 
        default='linux', 
        help="[linux|imdb|aids|ogbg-molhiv|ogbg-molpcba|ogbg-code2]"
    )

    args = parser.parse_args()

    layer_map = {'aids':0, 'imdb':2, 'linux':0, 'ogbg-molhiv':1, 'ogbg-molpcba':0, 'ogbg-code2':1}
    layer_val = layer_map[args.valset]

    file_path = f'./val_transfer/val_set_{args.valset}_funcs_{args.func_set}_{layer_val}.xlsx'
    df = read_excel_data(file_path)
    plot_metrics(df, args.valset, args.func_set)
