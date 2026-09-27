# Dimver
This is an implementation of a **mobile manipulator** consisting of a mecanum-wheel base and a **2-DOF planar robotic arm**, controlled by a MegaPi motor controller and a Raspberry Pi. The system addresses wheel slip where the no-slip assumption underlying standard kinematics breaks down on low-traction surfaces, causing odometry drift and control instability.

- A **supervised learning model** is trained in ROS-Gazebo simulation to estimate slip magnitude from proprioceptive measurements alone. The estimator is deployed in a **finite-state traction controller** that reduces commanded velocity when slip is detected and restores it once traction recovers.
- **Global localization** is provided by an overhead camera detecting an **AprilTag** mounted on the robot, using planar homography to map pixel coordinates to world coordinates.

 <p align="center">
   <img src="https://github.com/user-attachments/assets/63a60c85-a89a-42ba-914c-c33e06185130" alt="HW" width=40%>
 </p>

 ## System Architecture 

  <p align="center">
   <img src="https://github.com/user-attachments/assets/305d08cc-e282-4715-8853-ab4cc14ec33f" alt="ARCH" width=60%>
 </p>

The vision and operator interface run on a laptop. Safety-critical logic (traction control, arm sequencing, collision/obstacle handling) runs locally on the Raspberry Pi, which issues motor commands through the MegaPi. This separation means the robot can still stop safely if the vision link is lost.

 ## Mathematical Foundations

#### Manipulator Kinematics
The 2-DOF arm is modeled as a planar serial chain with link lengths $(L_1, L_2)$ and joint angles $(\theta_1, \theta_2)$.

The end-effector position is given by:

$$
x = L_1 \cos(\theta_1) + L_2 \cos(\theta_1 + \theta_2)
$$

$$
y = L_1 \sin(\theta_1) + L_2 \sin(\theta_1 + \theta_2)
$$

Inverse kinematics is solved geometrically using the law of cosines. With $r = \sqrt{x^2 + y^2}$:

$$
cos\( \theta_2 \) = \frac{x^2 + y^2 - L_1^2 - L_2^2}
{2L_1L_2}
$$
$$
\theta_2 = atan2 \left(
\pm\sqrt{1-\cos^2(\theta_2)},
\cos(\theta_2)
\right)
$$

$$ \theta_1 = atan2(y,x) - atan2\left(
L_2\sin(\theta_2),
L_1 + L_2\cos(\theta_2)
\right)
$$
#### Manipulator Dynamics
#### Slip Formulation

 ## Traction Control 

 ## Vision and Localization 

 ## Results 

 



