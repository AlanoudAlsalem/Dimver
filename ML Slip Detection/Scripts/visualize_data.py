import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("Loading dataset...")
df = pd.read_csv('master_dataset.csv')

plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
run_data = df[df['run_id'] == 4] # Run ID 4 is the 5th run (1.0 m/s)

time_vals = run_data['time'].to_numpy()
cmd_vals = run_data['v_cmd'].to_numpy()
wheel_vals = run_data['v_wheel'].to_numpy()
ground_vals = run_data['v_ground'].to_numpy()

plt.plot(time_vals, cmd_vals, 'b--', label='Command (Input)', alpha=0.7)
plt.plot(time_vals, wheel_vals, 'g-', label='Wheel Speed (Sensor)', linewidth=2)
plt.plot(time_vals, ground_vals, 'r-', label='Ground Speed (Truth)', linewidth=2)

plt.title('Evidence of Slip: Wheel vs. Ground Speed (Run #4)', fontsize=14)
plt.ylabel('Speed (m/s)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.scatter(df['v_cmd'].to_numpy(), df['slip'].to_numpy(), alpha=0.05, color='purple')

plt.title('Slip Ratio vs. Command Speed (All 24k Points)', fontsize=14)
plt.xlabel('Command Speed (m/s)')
plt.ylabel('Measured Slip (m/s)')
plt.grid(True)

output_img = 'slip_analysis.png'
plt.tight_layout()
plt.savefig(output_img)
print(f"Graph saved to {output_img}")
print("Open this image to see your physics in action!")
