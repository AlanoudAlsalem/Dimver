# Dimver
Dimver is a rover with a mounted 2-DOF robotic arm capable of obstacle avoidance, navigation, and object manipulation. The hardware setup is based on the MBot Mega, which comes with an Arduino Mega, and a Raspberry Pi responsible for centralized system control.

A camera mounted near a room ceiling is calibrated and performs ApriTag detection to localize the rover as it navigates through the room. A simple UI allows for sending commands to the rover for object picking and placing, as well as sending new goals it should navigate to. 
