import rosbag
import csv
import glob

# --- CONFIGURATION ---
# Find all bag files that match the pattern "data_run_*.bag"
bag_files = sorted(glob.glob("data_run_*.bag"))
output_file = 'master_dataset.csv'
wheel_radius = 0.03
# ---------------------

print(f"Found {len(bag_files)} bag files. Processing...")

with open(output_file, mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    # Header: Run_ID, Time, Command, Wheel_Speed, Ground_Speed, Slip
    writer.writerow(['run_id', 'time', 'v_cmd', 'v_wheel', 'v_ground', 'slip'])

    for run_id, bag_file in enumerate(bag_files):
        print(f"Processing {bag_file}...")
        
        bag = rosbag.Bag(bag_file)
        start_time = None
        
        current_cmd = 0.0
        current_wheel_v = 0.0

        for topic, msg, t in bag.read_messages(topics=['/cmd_vel', '/joint_states', '/gazebo/model_states']):
            timestamp = t.to_sec()
            if start_time is None: start_time = timestamp
            relative_time = timestamp - start_time

            if topic == '/cmd_vel':
                current_cmd = msg.linear.x

            elif topic == '/joint_states':
                if len(msg.velocity) >= 4:
                    avg_rad_s = sum(msg.velocity) / len(msg.velocity)
                    current_wheel_v = avg_rad_s * wheel_radius

            elif topic == '/gazebo/model_states':
                try:
                    robot_index = msg.name.index('mobile_manipulator') 
                    current_ground_v = msg.twist[robot_index].linear.x
                    
                    slip_val = current_wheel_v - current_ground_v

                    if current_cmd > 0.01:
                         writer.writerow([run_id, relative_time, current_cmd, current_wheel_v, current_ground_v, slip_val])
                
                except ValueError:
                    continue
        
        bag.close()

print(f"SUCCESS! All data merged into {output_file}")
