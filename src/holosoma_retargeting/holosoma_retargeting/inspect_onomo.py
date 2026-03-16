import torch
import numpy as np
import pandas as pd
import os

# Configuration
DATA_PATH = "demo_data/OMOMO_new/sub10_largebox_049.pt"
OUTPUT_REPORT = "omomo_dimension_report.csv"

def excavate():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    # 1. Load Data
    data = torch.load(DATA_PATH, map_location='cpu')
    if hasattr(data, 'detach'): data = data.detach()
    data_np = data.numpy()
    num_frames, num_dims = data_np.shape

    print(f"Excavating {num_dims} dimensions over {num_frames} frames...")

    analysis_results = []

    # 2. Analyze each dimension
    for i in range(num_dims):
        col = data_np[:, i]
        unique_vals = np.unique(col)
        num_unique = len(unique_vals)
        std_val = np.std(col)
        
        # Calculate switching frequency for discrete signals
        switches = np.sum(col[1:] != col[:-1])
        
        # Determine Category
        if std_val < 1e-6:
            category = "Static/Padding"
        elif num_unique <= 5:
            if switches > 10:
                category = "Contact (High Freq - Feet?)"
            else:
                category = "Contact (Low Freq - Hands/Object?)"
        else:
            category = "Continuous (Pose/Vel/Root)"

        analysis_results.append({
            "Index": i,
            "Category": category,
            "Mean": np.mean(col),
            "Std": std_val,
            "Min": np.min(col),
            "Max": np.max(col),
            "Unique_Count": num_unique,
            "Switch_Count": switches,
            "Values_Sample": str(unique_vals[:5]) if num_unique <= 10 else "Continuous"
        })

    # 3. Save to CSV for your spreadsheet review
    df = pd.DataFrame(analysis_results)
    df.to_csv(OUTPUT_REPORT, index=False)
    
    # 4. Summary Findings
    print("\n" + "="*40)
    print("DATA EXCAVATION SUMMARY")
    print("="*40)
    print(f"Total Dimensions: {num_dims}")
    print(f"Static/Padding:  {len(df[df['Category'] == 'Static/Padding'])} dims")
    print(f"Contact/Discrete: {len(df[df['Category'].str.contains('Contact')])} dims")
    print(f"Continuous Data:  {len(df[df['Category'] == 'Continuous (Pose/Vel/Root)'])} dims")
    print(f"\nReport saved to: {OUTPUT_REPORT}")
    
    # 5. Identify Blocks (Expert Logic)
    # Finding sequences of similar categories
    df['Block'] = (df['Category'] != df['Category'].shift()).cumsum()
    blocks = df.groupby(['Block', 'Category']).agg({'Index': ['min', 'max']})
    print("\n[Structural Layout Guess]:")
    for (block_id, cat), row in blocks.iterrows():
        start, end = row['Index']['min'], row['Index']['max']
        if (end - start) > 2: # Only show significant segments
            print(f"Indices {start:3d} - {end:3d} --> {cat}")

if __name__ == "__main__":
    excavate()