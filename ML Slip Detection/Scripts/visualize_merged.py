import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("Loading merged dataset...")
df = pd.read_csv('master_dataset.csv')

plt.figure(figsize=(10, 8))

plt.scatter(df['v_cmd'], df['slip'], 
            c=df['slip'], cmap='viridis', 
            alpha=0.1, edgecolors='none')

cbar = plt.colorbar()
cbar.set_label('Slip Magnitude (m/s)')

plt.title('The "Physics Map": Grip (Bottom) vs. Slip (Top)', fontsize=16)
plt.xlabel('Commanded Speed (m/s)', fontsize=12)
plt.ylabel('Measured Slip (m/s)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

plt.text(0.5, 0.8, 'Wall Impact Zone\n(High Slip)', 
         fontsize=12, color='red', ha='center', fontweight='bold')
plt.text(0.6, 0.05, 'Normal Driving Zone\n(Grip)', 
         fontsize=12, color='purple', ha='center', fontweight='bold')

output_img = 'final_thesis_plot.png'
plt.tight_layout()
plt.savefig(output_img)
print(f"Graph saved to {output_img}")
