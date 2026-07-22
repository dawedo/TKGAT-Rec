import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def add_timestamps_to_ratings(input_file, output_file):
    """
    Add realistic timestamps to ratings file
    """
    # Read existing ratings
    ratings = pd.read_csv(input_file, sep='\t', header=None, 
                         names=['user', 'item', 'rating'])
    
    # Sort by user to maintain sequence
    ratings = ratings.sort_values(['user', 'item'])
    
    # Generate timestamps
    base_date = datetime(2020, 1, 1)  # Starting date
    timestamps = []
    
    for user in ratings['user'].unique():
        user_ratings = ratings[ratings['user'] == user]
        n_ratings = len(user_ratings)
        
        # Generate increasing timestamps for this user
        # Random intervals between 1-30 days
        intervals = np.random.randint(1, 31, size=n_ratings)
        cumulative_days = np.cumsum(intervals)
        
        user_timestamps = [
            int((base_date + timedelta(days=int(days))).timestamp())
            for days in cumulative_days
        ]
        timestamps.extend(user_timestamps)
    
    ratings['timestamp'] = timestamps
    
    # Save with timestamps
    ratings.to_csv(output_file, sep='\t', header=False, index=False)
    print(f"Added timestamps to {len(ratings)} ratings")
    print(f"Date range: {datetime.fromtimestamp(min(timestamps))} to {datetime.fromtimestamp(max(timestamps))}")

# Usage
add_timestamps_to_ratings(
    '../data/music/ratings_finalold.txt',
    '../data/music/ratings_finaltimestamps.txt'
)