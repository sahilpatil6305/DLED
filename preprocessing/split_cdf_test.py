import os
import json
import random
import argparse
from typing import Dict, List, Tuple

def load_json(file_path: str) -> Dict:
    """Load a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data: Dict, file_path: str):
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def split_data(data: Dict, train_ratio: float = 0.7) -> Tuple[Dict, Dict]:
    """
    Split the data into train and test sets while maintaining the same structure.
    
    Args:
        data: Dictionary containing the dataset
        train_ratio: Ratio of data to use for training (default: 0.7)
        
    Returns:
        Tuple of (train_data, test_data)
    """
    dataset_name = list(data.keys())[0]  # Get the main dataset key
    train_dataset_name = dataset_name.replace('test', 'train')
    
    # Initialize output dictionaries
    train_data = {train_dataset_name: {}}
    test_data = {dataset_name: {}}
    
    # For each category (e.g., ddim_Real, ddim_Fake, etc.)
    for category in data[dataset_name]:
        train_data[train_dataset_name][category] = {'train': {}}
        test_data[dataset_name][category] = {'test': {}}
        
        # Get all video IDs
        all_videos = list(data[dataset_name][category]['test'].keys())
        
        # Shuffle the videos
        random.shuffle(all_videos)
        
        # Calculate split point
        split_idx = int(len(all_videos) * train_ratio)
        
        # Split videos into train and test sets
        train_videos = all_videos[:split_idx]
        test_videos = all_videos[split_idx:]
        
        # Fill train data
        for video_id in train_videos:
            train_data[train_dataset_name][category]['train'][video_id] = \
                data[dataset_name][category]['test'][video_id]
        
        # Fill test data
        for video_id in test_videos:
            test_data[dataset_name][category]['test'][video_id] = \
                data[dataset_name][category]['test'][video_id]
    
    return train_data, test_data

def main():
    parser = argparse.ArgumentParser(description='Split CDF test datasets into train and test sets')
    parser.add_argument('--input_dir', type=str, 
                      default='./datasets/DF40/json_used',
                      help='Directory containing the input JSON files')
    parser.add_argument('--output_dir', type=str, 
                      default='./preprocessing/json_split',
                      help='Directory to save the output JSON files')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                      help='Ratio of data to use for training (default: 0.7)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # List of CDF test files to process
    cdf_files = [
        'DF40_CDF_test_all.json',
        'DF40_CDF_test_closed_all.json',
        'DF40_CDF_test_seen.json',
        'DF40_CDF_test_unseen_category.json',
        'DF40_CDF_test_unseen_forgery.json'
    ]
    
    # Process each file
    for filename in cdf_files:
        print(f"Processing {filename}...")
        
        # Load data
        input_path = os.path.join(args.input_dir, filename)
        data = load_json(input_path)
        
        # Split data
        train_data, test_data = split_data(data, args.train_ratio)
        
        # Save train data
        train_filename = filename.replace('test', 'train')
        train_path = os.path.join(args.output_dir, train_filename)
        save_json(train_data, train_path)
        
        # Save test data
        test_path = os.path.join(args.output_dir, filename)
        save_json(test_data, test_path)
        
        # Get statistics
        dataset_name = list(data.keys())[0]
        train_dataset_name = dataset_name.replace('test', 'train')
        total_videos = sum(len(data[dataset_name][cat]['test']) 
                         for cat in data[dataset_name])
        train_videos = sum(len(train_data[train_dataset_name][cat]['train']) 
                         for cat in train_data[train_dataset_name])
        test_videos = sum(len(test_data[dataset_name][cat]['test']) 
                         for cat in test_data[dataset_name])
        
        # Print statistics
        print(f"Total videos: {total_videos}")
        print(f"Train videos: {train_videos}")
        print(f"Test videos: {test_videos}")
        print(f"Train ratio: {train_videos/total_videos:.2%}")
        print("----------------------------------------")

if __name__ == '__main__':
    main()
