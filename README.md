# Surgical Data Recording and Dataset Preparation Pipeline

This repository contains a lightweight data collection and post-processing pipeline for robot-assisted surgical training experiments using the da Vinci Research Kit (dVRK).

Its main purpose is to help record synchronized multimodal data during task execution, organize the resulting episodes, and prepare the dataset for downstream analysis or machine learning workflows.

## What this project does

The scripts in this repository support the full workflow from recording to dataset formatting:

- recording robot motion and camera data
- saving image sequences from the available video streams
- synchronizing kinematics with recorded frames
- reorganizing raw recordings into episode/task folders
- validating formatting before conversion
- converting the processed data into LeRobot format

In short, this repo is meant to bridge the gap between **raw experimental recordings** and a **clean, structured dataset** ready for analysis, annotation, or learning-based applications.


## Repository scripts

### `RecordingLauncher.py`
Convenience script for launching a recording session and coordinating the main recording components.

### `daVinciFrameSequenceRecorder.py`
Records and saves image sequences from different available camera streams.

### `daVinciKinematicsRecorder.py`
Captures robot kinematics during the task by subscibing to dvrk-related ROS topics.

### `Synchronizer.py`
Aligns the recorded kinematics and image streams in time so that the modalities can be used together.

### `reallocate_tasks.py`
Reorganizes the recorded episodes into different subtasks.

### `reallocate_episodes.py`
Splits or rearranges recorded episodes into subfodlers, e.g., for *perfect* - *recovery* - *failure* breakdown.

### `validate_formatting.py`
Checks dataset's compliance with the expected LeRobot format.

### `dvrk_zarr_to_lerobot.py`
Converts the processed recordings into a LeRobot-style dataset format.

### `FolderChecker.py`, `video_hour_counter.py`, *`helpers`*

Utility scripts, task-specific smaller automations.


## Project context

The pipeline was designed for multimodal surgical robotics data collection with a da Vinci system, including teleoperated demonstrations on a tabletop phantom. The broader goal is to support reproducible dataset creation for surgical skill assessment, workflow analysis, and machine learning research with a relatively flexible data recording software.

In its current state the data sources include:

- stereo endoscopic video (Blackmagic DeckLink)
- tool-wrist camera streams (USB cameras)
- RealSense camera strems
- robot end-effector pose / kinematics (ROS topics)
- gripper or tool-state information (ROS topics)
- episode metadata (ROS topics)


## Notes

This repository focuses on **data engineering for experiments**, not on model training itself. It is best used as a preprocessing and dataset-preparation layer before annotation, benchmarking, or machine learning.


## Contact

If you use or extend this repository, please document your recording setup, task definition, and folder conventions clearly so that the resulting dataset remains reproducible and easy to interpret.