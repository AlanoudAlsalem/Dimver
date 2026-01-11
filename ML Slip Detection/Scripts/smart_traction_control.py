import rospy
import joblib
import pandas as pd
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState

# --- CONFIGURATION ---
model_path = 'slip_detector_model.pkl'
wheel_radius = 0.03
target_speed = 0.8  # We WANT to go fast
slip_limit = 0.2    # If slip > 0.2 m/s, activate TCS
# ---------------------

class TractionController:
    def __init__(self):
        rospy.init_node('traction_control_system')
        
        print(f"Loading AI Model...")
        self.model = joblib.load(model_path)
        print("Traction Control System (TCS) Online.")
        
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/joint_states', JointState, self.joint_callback)
        
        self.current_wheel_v = 0.0
        self.current_throttle = 0.0
        
        # Run at 10Hz (10 times a second)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        self.slip_counter = 0 #filtering noise

    def joint_callback(self, msg):
        if len(msg.velocity) >= 4:
            avg_rad_s = sum(msg.velocity) / len(msg.velocity)
            self.current_wheel_v = avg_rad_s * wheel_radius

    def control_loop(self, event):
        input_data = pd.DataFrame([[self.current_throttle, self.current_wheel_v]], 
                                columns=['v_cmd', 'v_wheel'])
        predicted_slip = self.model.predict(input_data)[0]
        
        if predicted_slip > slip_limit:
            self.slip_counter += 1  # Add a strike
        else:
            self.slip_counter = 0   # Reset if grip returns
            
        if self.slip_counter >= 3:
            rospy.logwarn(f"SUSTAINED SLIP ({predicted_slip:.2f} m/s). TCS ENGAGED!")
            self.current_throttle -= 0.1
            if self.current_throttle < 0.0: self.current_throttle = 0.0
            
        else:
            if self.current_throttle < target_speed:
                self.current_throttle += 0.05
                rospy.loginfo(f"Grip Good. Accelerating to {self.current_throttle:.2f} m/s")

        move_cmd = Twist()
        move_cmd.linear.x = self.current_throttle
        self.pub.publish(move_cmd)

if __name__ == '__main__':
    try:
        controller = TractionController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
