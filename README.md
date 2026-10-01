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


## Citation

If you use this repository, the Open-H acquisition pipeline, or the associated dataset in your research, please cite:

> E. Lukács, K. Takács, and T. Haidegger,  
> **“Open-H Acquisition Pipeline and Surgical Robotics Dataset Evaluation,”**  
> *2026 IEEE 30th International Conference on Intelligent Engineering Systems (INES)*,  
> Budapest, Hungary, 2026, pp. 219–226.  
> DOI: [10.1109/INES69513.2026.11661245](https://doi.org/10.1109/INES69513.2026.11661245)

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{lukacs2026openh,
  author    = {Luk{\'a}cs, E. and Tak{\'a}cs, K. and Haidegger, T.},
  title     = {Open-H Acquisition Pipeline and Surgical Robotics Dataset Evaluation},
  booktitle = {2026 IEEE 30th International Conference on Intelligent Engineering Systems (INES)},
  year      = {2026},
  pages     = {219--226},
  address   = {Budapest, Hungary},
  doi       = {10.1109/INES69513.2026.11661245}
}
```

</details>

GitHub also provides citation metadata through the repository's [`CITATION.cff`](CITATION.cff) file.

## Acknowledgment

This work is related to the MedLaBotX project (2024-1.2.3-HU-RIZONT-00069).
Project 2024-1.2.3-HU-RIZONT-00069 has been implemented with support provided by the Ministry of Culture and Innovation of Hungary from the National Research, Development, and Innovation Fund, financed under the 2024-1.2.3-HU-RIZONT funding scheme.
