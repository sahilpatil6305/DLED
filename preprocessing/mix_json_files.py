import os
import json
import random
from glob import glob
from pathlib import Path

def get_base_name(filename):
    """Get the base name without _cdf.json or _ff.json"""
    return filename.replace('_cdf.json', '').replace('_ff.json', '')

def load_json_file(filepath):
    """Load a JSON file and return its contents"""
    with open(filepath, 'r') as f:
        return json.load(f)

def mix_entries(cdf_entries, ff_entries):
    """Mix entries from CDF and FF files, using equal numbers from each"""
    # Get the minimum number of entries we can use from each source
    min_entries = min(len(cdf_entries), len(ff_entries))
    
    # Randomly sample the same number of entries from each source
    cdf_keys = random.sample(list(cdf_entries.keys()), min_entries)
    ff_keys = random.sample(list(ff_entries.keys()), min_entries)
    
    # Create mixed entries
    mixed_entries = {}
    for cdf_key, ff_key in zip(cdf_keys, ff_keys):
        # Add CDF entry with cdf_ prefix
        mixed_entries[f"cdf_{cdf_key}"] = cdf_entries[cdf_key]
        # Add FF entry with ff_ prefix
        mixed_entries[f"ff_{ff_key}"] = ff_entries[ff_key]
    
    return mixed_entries

def process_json_pair(cdf_file, ff_file, output_dir):
    """Process a pair of CDF and FF files to create a mixed version"""
    # Load both files
    cdf_data = load_json_file(cdf_file)
    ff_data = load_json_file(ff_file)
    
    # Get the main keys (e.g., 'danet_cdf', 'danet_ff')
    cdf_main_key = list(cdf_data.keys())[0]
    ff_main_key = list(ff_data.keys())[0]
    
    # Create the mixed data structure
    mixed_data = {}
    mixed_main_key = cdf_main_key.replace('_cdf', '_mixed')
    mixed_data[mixed_main_key] = {}
    
    # Process each subkey (e.g., 'danet_Real')
    for subkey in cdf_data[cdf_main_key]:
        if subkey in ff_data[ff_main_key]:
            mixed_data[mixed_main_key][subkey] = {
                'train': mix_entries(
                    cdf_data[cdf_main_key][subkey]['train'],
                    ff_data[ff_main_key][subkey]['train']
                ),
                'test': mix_entries(
                    cdf_data[cdf_main_key][subkey]['test'],
                    ff_data[ff_main_key][subkey]['test']
                )
            }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save mixed data to new JSON file
    output_file = os.path.join(output_dir, f"{get_base_name(os.path.basename(cdf_file))}_mixed.json")
    with open(output_file, 'w') as f:
        json.dump(mixed_data, f, indent=4)
    
    print(f"Created mixed file: {output_file}")

def main():
    # Directories
    cdf_dir = "./preprocessing/new_json"
    ff_dir = "./datasets/DF40/data_json"
    output_dir = "./preprocessing/mixed_json"
    
    # Get all CDF files
    cdf_files = glob(os.path.join(cdf_dir, "*_cdf.json"))
    
    # Process each CDF file
    for cdf_file in cdf_files:
        # Get corresponding FF file
        base_name = get_base_name(os.path.basename(cdf_file))
        ff_file = os.path.join(ff_dir, f"{base_name}_ff.json")
        
        # Check if FF file exists
        if os.path.exists(ff_file):
            process_json_pair(cdf_file, ff_file, output_dir)
        else:
            print(f"Warning: No matching FF file found for {cdf_file}")

if __name__ == "__main__":
    main()
