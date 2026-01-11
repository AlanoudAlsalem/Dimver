import rospy
import time
import os
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

def reset_robot():
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        state = ModelState()
        state.model_name = 'mobile_manipulator'
        
        # Reset position to (0, 0, 0)
        state.pose.position.x = 0.0
        state.pose.position.y = 0.0
        state.pose.position.z = 0.1
        state.pose.orientation.x = 0.0
        state.pose.orientation.y = 0.0
        state.pose.orientation.z = 0.0
        state.pose.orientation.w = 1.0
        
        set_state(state)
        print("Robot Reset to Start.")
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")

def collect_run(speed, run_id):
    print(f"--- STARTING RUN {run_id} (Speed: {speed} m/s) ---")
    
    bag_name = f"data_run_{run_id}.bag"
    os.system(f"timeout 6s rosbag record -O {bag_name} /cmd_vel /joint_states /gazebo/model_states &")
    
    time.sleep(1)
    # 2. Move the robot
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    move_cmd = Twist()
    move_cmd.linear.x = speed
    
    # Publish for 5 seconds
    start_time = time.time()
    rate = rospy.Rate(10) # 10hz
    while time.time() - start_time < 5:
        pub.publish(move_cmd)
        rate.sleep()
        
    # 3. Stop
    move_cmd.linear.x = 0.0
    pub.publish(move_cmd)
    print("Run Complete.")
    time.sleep(1)

if __name__ == '__main__':
    rospy.init_node('data_collector')
    time.sleep(2)
    
    speeds = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    for i, speed in enumerate(speeds):
        reset_robot()
        time.sleep(1) 
        collect_run(speed, i)
        time.sleep(1) # Cool down

    print("ALL DATA COLLECTED.")
