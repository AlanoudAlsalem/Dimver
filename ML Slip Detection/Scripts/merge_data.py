import pandas as pd

print("Loading Wall Data (Slip)...")
wall_df = pd.read_csv('wall_data.csv')

print("Loading Grip Data (Normal)...")
grip_df = pd.read_csv('grip_data.csv')

final_df = pd.concat([wall_df, grip_df])

final_df = final_df.sample(frac=1).reset_index(drop=True)

final_df.to_csv('master_dataset.csv', index=False)
print(f"Merged! Total Training Data: {len(final_df)} rows.")
