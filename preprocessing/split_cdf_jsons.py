import os
import json
import random
from glob import glob
from pathlib import Path

def process_cdf_json(input_file, output_dir):
    # Load the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Get the main key (e.g., 'danet_cdf')
    main_key = list(data.keys())[0]
    
    # Process each subkey (e.g., 'danet_Real')
    for subkey in data[main_key]:
        # Get test data
        test_data = data[main_key][subkey]['test']
        test_keys = list(test_data.keys())
        
        # Calculate number of items to move (70% of test data)
        num_to_move = int(len(test_keys) * 0.7)
        
        # Randomly select keys to move
        keys_to_move = random.sample(test_keys, num_to_move)
        
        # Move selected items from test to train
        for key in keys_to_move:
            data[main_key][subkey]['train'][key] = test_data[key]
            del data[main_key][subkey]['test'][key]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save modified data to new JSON file
    output_file = os.path.join(output_dir, os.path.basename(input_file))
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Processed {input_file} -> {output_file}")

def main():
    # Input directory containing CDF json files
    input_dir = "./datasets/DF40/data_json"
    # Output directory for modified json files
    output_dir = "./preprocessing/new_json"
    
    # Find all *_cdf.json files
    cdf_files = glob(os.path.join(input_dir, "*_cdf.json"))
    
    # Process each file
    for file in cdf_files:
        process_cdf_json(file, output_dir)

if __name__ == "__main__":
    main()