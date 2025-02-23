import pandas as pd
from datetime import datetime
import ast
import signal
import sys
import time
import os

class Logger:
    def __init__(self, dataset, store_suffix):
        # Initialize an empty DataFrame with appropriate columns
        self.df = pd.DataFrame(columns=['time_step', 'prompt', 'response', 'error', 'marginal_gain', 'absolute_scores', 'epoch', 'num_llm_calls'])
        self.dataset = dataset
        self.suffix = store_suffix
        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def log(self, graph_ids, prompt, response, error, marginal_gain, abs_scores, time, count, num_llm_calls):
        # Create a new record as a dictionary
        record = pd.DataFrame([{
            'time_step': time,
            'graph_ids': graph_ids,
            'prompt': prompt,
            'response': response,
            'error':error,
            'marginal_gain': marginal_gain,
            'absolute_scores': abs_scores,
            'epoch': count,
            'num_llm_calls': num_llm_calls
        }])
        # Append the record to the DataFrame
        self.df = pd.concat([self.df, record], ignore_index=True)

    def get_logs(self):
        # Return the DataFrame containing all logs
        return self.df

    def save_to_json(self):
        # Generate a filename with the current date and time
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        #!change path here
        save_path = f'logs/{self.dataset}'
        # Create the directory if it does not exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = f'{save_path}/{timestamp_str}_{self.suffix}.json'
        # Save the DataFrame to the CSV file
        self.df.to_json(filename, orient='records', date_format='iso', indent=4)
        print(f"Logs saved to {filename}")
    
    def signal_handler(self, sig, frame):
        # Handle the signal and save the logs before exiting
        print("\nProgram interrupted! Saving logs...")
        self.save_to_json()  # Or self.save_to_csv() depending on your preferred format
        sys.exit(0)  # Exit the program gracefully
        
    def finalize_logging(self):
        # Save logs when the program is about to exit due to an error or other reasons
        print("Finalizing and saving logs...")
        self.save_to_json()  # Or self.save_to_csv() depending on your preferred format
