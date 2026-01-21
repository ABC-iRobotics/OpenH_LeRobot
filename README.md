# OpenH_LeRobot

**Open-H dataset recorder** — a small Python toolchain for recording multi-modal surgical robotics datasets (video + kinematics), synchronizing them, and converting the result into the **LeRobot** dataset format.

This repository currently contains a set of standalone scripts (no package structure) intended to be run from the repo root. :contentReference[oaicite:0]{index=0}

---

## What this repo does

- **Record image sequences** from one or more cameras (e.g., endoscope streams, external cameras). :contentReference[oaicite:1]{index=1}  
- **Record robot kinematics** (e.g., da Vinci / dVRK telemetry exported to CSV-like logs). :contentReference[oaicite:2]{index=2}  
- **Synchronize** camera frames to kinematic samples (nearest-timestamp matching / alignment logic handled in the synchronizer script). :contentReference[oaicite:3]{index=3}  
- **Convert** a recorded dataset into **LeRobot (v2.x-style)** structure for training/benchmarking. :contentReference[oaicite:4]{index=4}  
- Provide small helpers to **validate formatting**, **reallocate episodes**, and compute **total video hours**. :contentReference[oaicite:5]{index=5}  

---

## Repository layout

Top-level scripts (as of the latest repo view):

- `RecordingLauncher.py` — orchestration entrypoint to run recording components. :contentReference[oaicite:6]{index=6}  
- `daVinciFrameSequenceRecorder.py` — camera/image-sequence recorder. :contentReference[oaicite:7]{index=7}  
- `daVinciKinematicsRecorder.py` — kinematics recorder/logger. :contentReference[oaicite:8]{index=8}  
- `Synchronizer.py` — aligns image timestamps to kinematics (and produces synchronized outputs). :contentReference[oaicite:9]{index=9}  
- `dvrk_zarr_to_lerobot.py` — conversion utility to LeRobot dataset format. :contentReference[oaicite:10]{index=10}  
- `validate_formatting.py` — checks that output files/folders match expected conventions. :contentReference[oaicite:11]{index=11}  
- `reallocate_episodes.py` — reorganize/renumber episodes between splits or categories. :contentReference[oaicite:12]{index=12}  
- `video_hour_counter.py` — sums duration over videos in a folder. :contentReference[oaicite:13]{index=13}  
- `helpers/` — utility functions used by the scripts. :contentReference[oaicite:14]{index=14}  

---

## Quickstart

### 1) Clone
```bash
git clone https://github.com/ABC-iRobotics/OpenH_LeRobot.git
cd OpenH_LeRobot
