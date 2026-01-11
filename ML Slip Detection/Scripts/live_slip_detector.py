import rospy
import joblib
import pandas as pd
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState

# --- CONFIGURATION ---
model_path = 'slip_detector_model.pkl'
wheel_radius = 0.03
# ---------------------

class SlipDetector:
    def __init__(self):
        rospy.init_node('live_slip_detector')
        
        print(f"Loading AI Model from {model_path}...")
        self.model = joblib.load(model_path)
        print("Model Loaded! AI is watching...")
        
        self.current_cmd = 0.0
        self.current_wheel_v = 0.0

        rospy.Subscriber('/cmd_vel', Twist, self.cmd_callback)
        rospy.Subscriber('/joint_states', JointState, self.joint_callback)

        self.timer = rospy.Timer(rospy.Duration(0.1), self.predict_slip)

    def cmd_callback(self, msg):
        self.current_cmd = msg.linear.x

    def joint_callback(self, msg):
        if len(msg.velocity) >= 4:
            avg_rad_s = sum(msg.velocity) / len(msg.velocity)
            self.current_wheel_v = avg_rad_s * wheel_radius

    def predict_slip(self, event):
        if self.current_cmd < 0.01:
            return

        # Prepare input for the AI (Must match training columns: v_cmd, v_wheel)
        # We use a DataFrame because the model expects feature names
        input_data = pd.DataFrame([[self.current_cmd, self.current_wheel_v]], 
                                columns=['v_cmd', 'v_wheel'])
        
        # ASK THE AI
        predicted_slip = self.model.predict(input_data)[0]
        
        # Check Safety
        if predicted_slip > 0.1: # Threshold for "Bad Slip"
            rospy.logwarn(f"!!! SLIP DETECTED !!! Prediction: {predicted_slip:.3f} m/s")
        else:
            rospy.loginfo(f"Grip is Good. Prediction: {predicted_slip:.3f} m/s")

if __name__ == '__main__':
    try:
        detector = SlipDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
