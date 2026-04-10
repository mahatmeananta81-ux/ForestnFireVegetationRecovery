# cleanup_inventory.py  — run after downloading CSVs from Drive
import os
import pandas as pd
import hashlib

import config

MIN_FIRES = 1          # delete if zero qualifying fires
MIN_AREA_KM2 = 24.0    # your strict threshold

# Assuming config is defined elsewhere in your project
inventory_dir = config.INVENTORY_DIR

def get_file_hash(filepath):
    """Generates an MD5 hash of the file to check for exact content duplicates."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        # Read in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

deleted = 0
seen_hashes = set()

for fname in os.listdir(inventory_dir):
    if not fname.endswith('.csv'):
        continue
    
    fpath = os.path.join(inventory_dir, fname)
    
    try:
        # 1. Check for exact duplicate content
        file_hash = get_file_hash(fpath)
        if file_hash in seen_hashes:
            os.remove(fpath)
            print(f"  Deleted (duplicate content): {fname}")
            deleted += 1
            continue  # Skip to the next file
            
        # Add the hash to our tracker so we can spot future duplicates
        seen_hashes.add(file_hash)

        # 2. Check for empty files and strict area thresholds
        df = pd.read_csv(fpath)
        if len(df) == 0:
            os.remove(fpath)
            print(f"  Deleted (empty): {fname}")
            deleted += 1
        elif df['burn_area_km2'].max() < MIN_AREA_KM2:
            os.remove(fpath)
            print(f"  Deleted (no qualifying fires, max={df['burn_area_km2'].max():.1f} km²): {fname}")
            deleted += 1
            
    except Exception as e:
        print(f"  Could not process {fname}: {e}")

# Note: Filtered the final count to only show remaining CSVs
remaining_csvs = len([f for f in os.listdir(inventory_dir) if f.endswith('.csv')])
print(f"\nDeleted {deleted} files. Remaining: {remaining_csvs} CSVs.")