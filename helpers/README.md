<!--
Open-H Embodiment Dataset README Template (v1.0)
Please fill out this template and include it in the ./metadata directory of your LeRobot dataset.
This file helps others understand the context and details of your contribution.
-->

# README

---

## 📋 At a Glance

Teleoperated demonstrations of a da Vinci robot performing peg transfer on a 3D printed model with silicone pegs.

---

## 📖 Dataset Overview

This dataset contains trajectories of novices using the dVRK to perform peg transfer. It includes successful trials, failures, and recovery attempts as well. There are recordings where an episode covers a full peg transfer task (all pegs transferred), and recordings where 1 episode contains the manipulation of 1 peg.

| | |
| :--- | :--- |
| **Total Trajectories** | `61` |
| **Total Hours** | `?` |
| **Data Type** | `[ ] Clinical` `[ ] Ex-Vivo` `[X] Table-Top Phantom` `[ ] Digital Simulation` `[ ] Physical Simulation` `[ ] Other (If checked, update "Other")` |
| **License** | CC BY 4.0 |
| **Version** | `[1.0]` |

---

## 🎯 Tasks & Domain

### Domain


- [X] **Surgical Robotics**
- [ ] **Ultrasound Robotics**
- [ ] **Other Healthcare Robotics** 

### Demonstrated Skills

- Peg transfer

---

## 🔬 Data Collection Details

### Collection Method


- [X] **Human Teleoperation**
- [ ] **Programmatic/State-Machine**
- [ ] **AI Policy / Autonomous**
- [ ] **Other** (Please specify)

### Operator Details

| | Description |
| :--- | :--- |
| **Operator Count** | `2` |
| **Operator Skill Level** | `[ ] Expert (e.g., Surgeon, Sonographer)` <br> `[ ] Intermediate (e.g., Trained Researcher)` <br> `[X] Novice (e.g., ML Researcher with minimal experience)` <br> `[ ] N/A` |
| **Collection Period** | From `[2026-01-15]` to `[2026-01-16]` |

### Recovery Demonstrations


- [X] **Yes**
- [ ] **No**

**If yes, please briefly describe the recovery process:**


For x demonstrations, the pegs are not grasped accurately and they fall back right after lifting, the operator has to re-grasp the peg

---

## 💡 Diversity Dimensions


- [X] **Camera Position / Angle**
- [X] **Lighting Conditions**
- [ ] **Target Object** (e.g., different phantom models, suture types)
- [ ] **Spatial Layout** (e.g., placing the target suture needle in various locations)
- [ ] **Robot Embodiment** (if multiple robots were used)
- [ ] **Task Execution** (e.g., different techniques for the same task)
- [ ] **Background / Scene**
- [X] **Other** (Please specify: `[Setup joints]`)

*If you checked any of the above please briefly elaborate below.*

- Endoscope lighting was gradually reduced throughout the trials
- Camera posiiton was slightly varied
- Set up joint configuration can be changed and recorded



---

## 🛠️ Equipment & Setup

### Robotic Platform(s)


- **Robot 1:** da Vinci Classic (with da Vinci Research Kit)

### Sensors & Cameras


| Type | Model/Details |
| :--- | :--- |
| **Primary Camera** | `Stereo Endoscopic Camera, 720x576 @ 30fps` |
| **Wrist Camera** | `Endoscopic Camera (x2), 640x480 @ 30fps` |

---

## 🎯 Action & State Space Representation


### Action Space Representation

**Primary Action Representation:**
- [x] **Absolute Cartesian** (position/orientation relative to robot base)
- [ ] **Relative Cartesian** (delta position/orientation from current pose)
- [ ] **Joint Space** (direct joint angle commands)
- [ ] **Other** (Please specify: `[Your Representation]`)

**Orientation Representation:**
- [x] **Quaternions** (x, y, z, w)
- [ ] **Euler Angles** (roll, pitch, yaw)
- [ ] **Axis-Angle** (rotation vector)
- [ ] **Rotation Matrix** (3x3 matrix)
- [ ] **Other** (Please specify: `[Your Representation]`)

**Reference Frame:**
- [x] **Robot Base Frame**
- [ ] **Tool/End-Effector Frame**
- [ ] **World/Global Frame**
- [ ] **Camera Frame**
- [ ] **Other** (Please specify: `[Your Frame]`)

**Action Dimensions:**

```
action: [x, y, z, qx, qy, qz, qw, gripper]
- x, y, z: Absolute position in robot base frame (meters)
- qx, qy, qz, qw: Absolute orientation as quaternion
- gripper: Gripper opening angle (radians)
```

### State Space Representation

**State Information Included:**
- [ ] **Joint Positions** (all articulated joints)
- [ ] **Joint Velocities**
- [x] **End-Effector Pose** (Cartesian position/orientation)
- [ ] **Force/Torque Readings**
- [x] **Gripper State** (position, force, etc.)
- [ ] **Other** (Please specify: `[Your State Info]`)

**State Dimensions:**

```
observation.state: [x, y, z, qx, qy, qz, qw, gripper]
- x, y, z: Absolute position in robot base frame (meters)
- qx, qy, qz, qw: Absolute orientation as quaternion
- gripper: Gripper opening angle (radians)
```

### 📋 Additional Representations


**Set up joint configuration:**
- **`topic: /SUJ/***/measured_cp`**: 
  ```
  PSM1=(0.072154,0.837602,1.106661,0.164813,0.089878,-0.239018,0.952696) 
  PSM2=(0.179024,0.849484,0.889040,-0.111648,0.242206,0.101221,0.958449) 
  ECM=(0.237244,0.955400,0.406429,-0.382510,0.009886,-0.023867,0.923590)
  ```


---

## ⏱️ Data Synchronization Approach


*Each modality (DeckLink cameras, USB cameras, and robotic kinematics) was recorded time-stamped on the same PC. A post-processing synchronization script segmented trials using explicit start/end markers and used the kinematic time series as the reference timeline. For each kinematic timestamp, the temporally nearest image frame from each camera stream was selected. The original stereoendoscope of the system is capable of only ~20FPS, thus the <50ms delay can not always be ensured.* 

---

## 👥 Attribution & Contact


| | |
| :--- | :--- |
| **Dataset Lead** | `[Kristóf Takács, Eszter Lukács, Tamás Haidegger]` |
| **Institution** | `[Obuda University]` |
| **Contact Email** | `[krsitof.takacs@irob.uni-obuda.hu, eszter.lukacs@irob.uni-obuda.hu, haidegger@irob.uni-obuda.hu]` |
| **Citation (BibTeX)** | |
