# Dimver
This is an implementation of a **mobile manipulator** consisting of a mecanum-wheel base and a **2-DOF planar robotic arm**, controlled by a MegaPi motor controller and a Raspberry Pi. The system addresses wheel slip where the no-slip assumption underlying standard kinematics breaks down on low-traction surfaces, causing odometry drift and control instability.

- A **supervised learning model** is trained in ROS-Gazebo simulation to estimate slip magnitude from proprioceptive measurements alone. The estimator is deployed in a **finite-state traction controller** that reduces commanded velocity when slip is detected and restores it once traction recovers.
- **Global localization** is provided by an overhead camera detecting an **AprilTag** mounted on the robot, using planar homography to map pixel coordinates to world coordinates.

 <p align="center">
   <img src="https://github.com/user-attachments/assets/63a60c85-a89a-42ba-914c-c33e06185130" alt="HW" width=40%>
 </p>

 ## System Architecture 

  <p align="center">
   <img src="https://github.com/user-attachments/assets/305d08cc-e282-4715-8853-ab4cc14ec33f" alt="ARCH" width=50%>
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
Joint torques required for a desired trajectory follow the Euler-Lagrange formulation:

$$
\tau = 
M(\theta)\ddot{\theta}
+
C(\theta,\dot{\theta})
+
G(\theta)
$$

where $(M(\theta))$ is the inertia matrix, $(C(\theta,\dot{\theta}))$ captures Coriolis/centrifugal effects, and $(G(\theta))$ represents the gravitational torque, determined by the mass distribution and the horizontal projection of each link's center of mass.

#### Slip Formulation

<p align="center">
   <img src="https://github.com/user-attachments/assets/53e79a2a-dc55-410f-9c6b-65c073eb644f" alt="PMAP" width=50%>
 </p>

The classical slip ratio is defined in terms of the wheel radius ($r$), wheel angular velocity ($\omega_{\text{wheel}}$), and ground velocity ($v_{\text{ground}}$):

$$
\lambda_{\text{ratio}} = 1 - \frac{v_{\text{ground}}}
{r\omega_{\text{wheel}}},
\quad
\lambda_{\text{ratio}} \in [0,1]
$$

Rather than using the normalized slip ratio, this project uses a signed velocity divergence that maps directly to the control correction:

$$
\lambda = v_{\text{wheel}} - v_{\text{ground}}
$$

The wheel-induced velocity is estimated from the mean angular velocity of the four wheels:

$$
v_{\text{wheel}} =
\frac{r}{4}
\sum_{i=1}^{4}
\dot{\theta}_i
$$

The regression target used during training is therefore $v_{\text{wheel}} - v_{\text{ground}}$ where $v_{\text{ground}}$ is available only in simulation and is used solely as an offline supervisory label.

 ## Traction Control 

The controller operates as a three-state FSM built on the slip signal $\lambda$ defined above:

<p align="center">
   <img src="https://github.com/user-attachments/assets/ddc6bee6-e166-4746-abc7-87f4e6b87471" alt="FSM" width=60%>
 </p>

- **S0 (Normal)**: tracks commanded velocity; slip below threshold.
- **S1 (Suspected)**: transient filter state in which a single violation doesn't trigger intervention.
- **S2 (Mitigation)**: active after Γ consecutive violations; velocity is reduced by a decay factor and ramped back up via a recovery gain once slip resolves.

<div align="center">

| Parameter | Description | Value |
|:---------:|:------------|------:|
| **`λ_th`** | Safety threshold | **0.20 m/s** |
| **`Γ`** | Consecutive violations before mitigation | **3 frames** |
| **`α`** | Mitigation (velocity decay) factor | **0.10** |
| **`β`** | Recovery (ramp-up) gain | **0.05** |

</div>

 ## Vision and Localization 

 ```mathematica
┌─────────────────┐    ┌───────────────┐    ┌─────────────────────┐    ┌─────────────────┐    ┌─────────────┐
│ Overhead Camera │ →  │ Undistortion  │ →  │ 4-Click Homography  │ →  │ AprilTag Detect │ →  │ (x, y, yaw) │
└─────────────────┘    └───────────────┘    │     (one-time)      │    └─────────────────┘    └─────────────┘
                                             └─────────────────────┘
```

<p align="center">
   <img src="https://github.com/user-attachments/assets/15e434cd-05c4-4bbf-9fd0-cf60d31c9508" alt="Exp Trajectory" width=60%>
 </p>

- Camera calibrated with a 7×10 checkerboard; median reprojection error 1.87 px.
- World frame established via a one-time 4-click calibration of the arena corners.
- Pose streamed to the Raspberry Pi over UDP at ~30 Hz, with timestamps and validity flags; pose messages older than 0.35 s are rejected.
- Measured tag visibility during the logged run was 43.12%, primarily due to the arm occluding the tag during manipulation. - The FSM-based safety arbitration handled these gaps without divergence or collision.

 ## Results 

 <div align="center">

| Metric | Result |
|:------:|:------:|
| **Slip model R² (held-out, simulation)** | **≈ 0.95** |
| **Multi-waypoint mission** | **22.57 m in 231.05 s** |
| **Goal convergence** | **Within 5 cm, in 10–15 s** |
| **Linear velocity cap** | **0.10 m/s** *(accounts for ~33 ms network latency)* |
| **Camera calibration RMSE** | **Median 1.87 px** |
| **Vision tag visibility** | **43.12%** |

</div>

Comparison Against Baseline Controllers:

<div align="center">

<img src="https://github.com/user-attachments/assets/0d192386-6f0c-424b-828d-7f89ca62b483" alt="RES1" width=75%>

| Controller | Final Position Error (cm) | Slip Duration (s) | Recovery Time (s) |
|:----------|--------------------------:|------------------:|------------------:|
| **No Mitigation** | 9.9 | 31.7 | 0.19 |
| **Threshold Clamp** | N/A *(failed to regain traction)* | 140.2 | N/A |
| **ML + FSM (proposed)** | **9.8** | **17.7** | **0.16** |

</div>

 



