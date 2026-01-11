import rosbag
import csv
import math

# --- CONFIGURATION ---
bag_file = 'slip_training.bag'
output_file = 'final_slip_data.csv'
wheel_radius = 0.03 
# ---------------------

print(f"Processing {bag_file}...")

with open(output_file, mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['time', 'v_cmd', 'v_wheel', 'v_ground', 'slip_difference'])

    current_cmd = 0.0
    current_wheel_v = 0.0
    current_ground_v = 0.0
    
    start_time = None

    bag = rosbag.Bag(bag_file)

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

                slip_diff = current_wheel_v - current_ground_v

                writer.writerow([relative_time, current_cmd, current_wheel_v, current_ground_v, slip_diff])
                
            except ValueError:
                continue

    bag.close()

print(f"Done! Data saved to {output_file}")
