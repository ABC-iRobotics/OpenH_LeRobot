#!/usr/bin/env python
"""
A script to convert DVRK (da Vinci Research Kit) robotics data into the LeRobot format (v2.1).

This script processes DVRK surgical robot datasets organized in directory structures
with CSV kinematics data and multiple camera views. It handles both perfect and
recovery demonstrations, extracting dual-arm PSM states, actions, and multi-camera
observations into a LeRobotDataset for the Hugging Face Hub.

Expected DVRK Dataset Structure:
--------------------------------
The script expects a directory structure organized by tissue and subtasks:

/path/to/dataset/
├── tissue_10/                          # Tissue phantom number
│   ├── 1_suture_throw/                 # Subtask directory
│   │   ├── episode_001/                # Individual episode
│   │   │   ├── left_img_dir/           # Left endoscope images
│   │   │   │   └── frame000000_left.jpg
│   │   │   ├── right_img_dir/          # Right endoscope images  
│   │   │   │   └── frame000000_right.jpg
│   │   │   ├── endo_psm1/              # PSM1 wrist camera
│   │   │   │   └── frame000000_psm1.jpg
│   │   │   ├── endo_psm2/              # PSM2 wrist camera
│   │   │   │   └── frame000000_psm2.jpg
│   │   │   └── ee_csv.csv              # Kinematics data (16D state + actions)
│   │   └── episode_002/
│   └── 2_needle_pass_recovery/         # Recovery demonstrations
└── tissue_11/

Data Format:
------------
- **Actions**: 16D dual-PSM Cartesian poses + jaw positions (absolute coordinates + quaternions)
- **States**: 16D dual-PSM current poses + jaw positions
- **Images**: 4 camera views (endoscope left/right, PSM1/2 wrist cameras)
- **Metadata**: Tool types, instruction text, recovery/perfect labels

Usage:
------
    python dvrk_zarr_to_lerobot.py --data-path /path/to/dataset --repo-id username/dataset-name

To also push to the Hugging Face Hub:
    python dvrk_zarr_to_lerobot.py --data-path /path/to/dataset --repo-id username/dataset-name --push-to-hub

Dependencies:
-------------
- lerobot v0.3.3
- tyro
- pandas
- PIL
- numpy
"""

import shutil # offers a number of high-level operations on files
from pathlib import Path #  provides a more intuitive and readable way to handle file paths
import tyro # tool for generating CLI (command-line interface) interfaces
import numpy as np # library used for working with arrays
import os # provides functions for interacting with the operating system
import pandas as pd # Python library used for working with data sets
from PIL import Image # to manipulate image files
import time # provides various time-related functions
from lerobot.datasets.lerobot_dataset import LeRobotDataset # the writer/encoder that creates LeRobot formatted dataset
from lerobot.constants import HF_LEROBOT_HOME # where LeRobot stores datasets locally
from lerobot.datasets.utils import write_info # writes metadata info (splits etc.) to disk

#IMPORTANT: this script assumes the CSV contains exactly these column names
# This is the robot state at each time step
states_name = [
    "psm1_pose.position.x",
    "psm1_pose.position.y",
    "psm1_pose.position.z",
    "psm1_pose.orientation.x",
    "psm1_pose.orientation.y",
    "psm1_pose.orientation.z",
    "psm1_pose.orientation.w",
    "psm1_jaw",
    "psm2_pose.position.x",
    "psm2_pose.position.y",
    "psm2_pose.position.z",
    "psm2_pose.orientation.x",
    "psm2_pose.orientation.y",
    "psm2_pose.orientation.z",
    "psm2_pose.orientation.w",
    "psm2_jaw",
    "ecm_pose.position.x",
    "ecm_pose.position.y",
    "ecm_pose.position.z",
    "ecm_pose.orientation.x",
    "ecm_pose.orientation.y",
    "ecm_pose.orientation.z",
    "ecm_pose.orientation.w",
    
]
metas_name = [
    "suj_psm1_pose.position.x",
    "suj_psm1_pose.position.y",
    "suj_psm1_pose.position.z",
    "suj_psm1_pose.orientation.x",
    "suj_psm1_pose.orientation.y",
    "suj_psm1_pose.orientation.z",
    "suj_psm1_pose.orientation.w",
    "suj_psm2_pose.position.x",
    "suj_psm2_pose.position.y",
    "suj_psm2_pose.position.z",
    "suj_psm2_pose.orientation.x",
    "suj_psm2_pose.orientation.y",
    "suj_psm2_pose.orientation.z",
    "suj_psm2_pose.orientation.w",
    "suj_ecm_pose.position.x",
    "suj_ecm_pose.position.y",
    "suj_ecm_pose.position.z",
    "suj_ecm_pose.orientation.x",
    "suj_ecm_pose.orientation.y",
    "suj_ecm_pose.orientation.z",
    "suj_ecm_pose.orientation.w",
]
metas_tool_name= [
    "psm1_tool_type",
    "psm2_tool_type",
]

# This is the action (command / setpoint) at each time step
actions_name = [
    "psm1_sp.position.x",
    "psm1_sp.position.y",
    "psm1_sp.position.z",
    "psm1_sp.orientation.x",
    "psm1_sp.orientation.y",
    "psm1_sp.orientation.z",
    "psm1_sp.orientation.w",
    "psm1_jaw_sp",
    "psm2_sp.position.x",
    "psm2_sp.position.y",
    "psm2_sp.position.z",
    "psm2_sp.orientation.x",
    "psm2_sp.orientation.y",
    "psm2_sp.orientation.z",
    "psm2_sp.orientation.w",
    "psm2_jaw_sp",
    "ecm_sp.position.x",
    "ecm_sp.position.y",
    "ecm_sp.position.z",
    "ecm_sp.orientation.x",
    "ecm_sp.orientation.y",
    "ecm_sp.orientation.z",
    "ecm_sp.orientation.w",
]


#IMPORTANT: assumes your frames are named sequentially like and there is no other files in the dir
def read_images(image_dir: str, file_pattern: str) -> np.ndarray:
    """Reads images from a directory into a NumPy array."""
    images = []
    # count images in the dir
    num_images = len(
        [
            name
            for name in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, name))
        ]
    )
    for idx in range(num_images):
        # If pattern is "frame{:06d}_left.jpg" then: idx=1 → frame000001_left.jpg
        filename = os.path.join(image_dir, file_pattern.format(idx))

        # If a frame number is missing, it prints a warning and skips it.
        #IMPORTANT: skipping creates shorter arrays, later indexing can become inconsistent
        if not os.path.exists(filename):
            print(f"Warning: {filename} does not exist.")
            continue
        
        # Convert to numpy and ensure 3 channels
        img = Image.open(filename)
        channels = len(img.getbands())
        if channels == 1:
            img_array = np.array(img)
            img_array = img_array.reshape([img_array.shape[0], img_array.shape[1],1])
            img_array = np.repeat(img_array, 3, axis=2).astype(np.uint8)
        else: 
            img_array = np.array(img)[..., :3]

        images.append(img_array)

    if images:
        return np.stack(images)
    else:
        return np.empty((0, 0, 0, channels), dtype=np.uint8)


def process_episode(dataset: LeRobotDataset, episode_path, states_name, metas_name, actions_name, subtask_prompt):
    """Processes a single episode, save the data to lerobot format"""

    # Paths to image directories (endo is the wrist camera)
    #IMPORTANT: If folder names differ even slightly the script fails.
    left_dir = os.path.join(episode_path, "left_img_dir")
    right_dir = os.path.join(episode_path, "right_img_dir")
    psm1_dir = os.path.join(episode_path, "endo_psm1")
    psm2_dir = os.path.join(episode_path, "endo_psm2")
    realsense_c_dir = os.path.join(episode_path, "realsense_color")
    realsense_d_dir = os.path.join(episode_path, "realsense_depth")
    csv_file = os.path.join(episode_path, "ee_csv.csv")

    # Read CSV to determine the number of frames (excluding header)
    df = pd.read_csv(csv_file)  #Loads CSV into a DataFrame

    # Read images from each camera
    left_images = read_images(left_dir, "frame_{:06d}.jpg")
    right_images = read_images(right_dir, "frame_{:06d}.jpg")
    psm1_images = read_images(psm1_dir, "frame_{:06d}.jpg")
    psm2_images = read_images(psm2_dir, "frame_{:06d}.jpg")
    realsense_c_images = read_images(realsense_c_dir, "frame_{:06d}.jpg")
    realsense_d_images = read_images(realsense_d_dir, "frame_{:06d}.png")

    #print(left_images.shape, right_images.shape, psm1_images.shape, psm2_images.shape, realsense_c_images.shape, realsense_d_images.shape)
    #IMPORTANT: it uses left camera as a base shape, if the right is shorter the indexing can crash
    num_frames = min(len(df), left_images.shape[0])

    # Read kinematics data and convert to structured array with headers
    kinematics_data = np.array(
        [tuple(row) for row in df.to_numpy()], #  DataFrame into NumPy 2D array like [(100, 1), (200, 3)]
        dtype=[(col, df[col].dtype.str) for col in df.columns], # creates the schema for the structured array
    )
    # print(kinematics_data[0])

    # Frame creation loop
    for i in range(num_frames):
        numeric_metas = []
        string_metas = []

        for n in metas_name:
            val = kinematics_data[n][i]
            if isinstance(val, (float, int, np.floating, np.integer)):
                numeric_metas.append(n)
            else:
                string_metas.append(n)

        frame = {
            "observation.state": np.hstack(
                [kinematics_data[n][i] for n in states_name]
            ).astype(np.float32),
            "action": np.hstack([kinematics_data[n][i] for n in actions_name]).astype(
                np.float32
            ),
            "instruction.text": subtask_prompt, #IMPORTANT: instruction is stored per-frame
            "observation.meta.tool.psm1": kinematics_data["psm1_tool_type"][i], #IMPORTANT: tool names are hardcoded
            "observation.meta.tool.psm2": kinematics_data["psm2_tool_type"][i],
            "observation.meta.suj": np.hstack([kinematics_data[n][i] for n in metas_name]).astype(
                np.float32),
        }

        #print(frame)
        # Attach images
        for cam_name, images in [
            ("endoscope.left", left_images),
            ("endoscope.right", right_images),
            ("wrist.left", psm2_images),
            ("wrist.right", psm1_images),
            ("realsense.color", realsense_c_images),
            ("realsense.depth", realsense_d_images),
        ]:
            if images.size > 0:
                frame[f"observation.images.{cam_name}"] = images[i]

        #IMPORTANT: timestamp stored in nanoseconds.
        timestamp_sec = (kinematics_data["timestamp"][i] - kinematics_data["timestamp"][0]) * 1e-9  ## turn nano sec to sec
        #print(timestamp_sec)
        dataset.add_frame(frame, task=subtask_prompt, timestamp=timestamp_sec)

    return dataset


def convert_data_to_lerobot(
    data_path: Path, repo_id: str, *, push_to_hub: bool = False, phantom_name: str
):
    """
    Converts a single Zarr store with episode boundaries to a LeRobotDataset.

    Args:
        data_path: The path to the Zarr store file/directory.
        repo_id: The repository ID for the dataset on the Hugging Face Hub.
        push_to_hub: Whether to push the dataset to the Hub after conversion.
    """
    final_output_path = os.path.join(HF_LEROBOT_HOME, repo_id)
    print(final_output_path)

    #IMPORTANT: deletes everything under that folder
    if os.path.exists(final_output_path):
        print(f"Removing existing dataset at {final_output_path}")
        shutil.rmtree(final_output_path)

    # Initialize a LeRobotDataset with the desired features.
    dataset = LeRobotDataset.create(
        repo_id=repo_id,    
        root=os.path.expanduser('~/'+repo_id), # Added to not use .cache folder --> videos generated automatically
        use_videos=True,
        robot_type="dvrk",
        fps=30, #IMPORTANT: set this correctly
        features={
            "observation.images.endoscope.left": {
                "dtype": "video",
                "shape": (576, 720, 3), #IMPORTANT: set this correctly
                "names": ["height", "width", "channel"],
            },
            "observation.images.endoscope.right": {
                "dtype": "video",
                "shape": (576, 720, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist.left": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist.right": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.realsense.color": {
                "dtype": "video",
                "shape": (720, 1280, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.realsense.depth": {
                "dtype": "video",
                "shape": (720, 1280, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(states_name),),
                "names": [states_name],
            },
            "action": {
                "dtype": "float32",
                "shape": (len(actions_name),),
                "names": [actions_name],
            },
            
            "observation.meta.tool.psm1": {
                "dtype": "string",
                "shape": (1,),
                "names": ["psm1_tool_type"],
            },
            "observation.meta.tool.psm2": {
                "dtype": "string",
                "shape": (1,),
                "names": ["psm2_tool_type"],
            },
            "observation.meta.suj": {
                "dtype": "float32",
                "shape": (len(metas_name),),
                "names": [metas_name],
            },
            "instruction.text": {
                "dtype": "string",
                "shape": (1,),
                "description": "Natural language command for the robot", #IMPORTANT: set this
            },
        },
        image_writer_processes=16,
        image_writer_threads=20,
        tolerance_s=0.1,
        # batch_encoding_size=12,
    )

    # measure time taken to complete the process
    start_time = time.time()
    perfect_demo_count = 0 # counter of episodes
    recovery_demo_count = 0 # counter of recovery episodes
    failure_demo_count = 0 # counter of failure episodes

    tissue_dir = os.path.join(data_path, phantom_name)

    if not os.path.exists(tissue_dir):
        print(f"Warning: {tissue_dir} does not exist.")
        exit()

    #IMPORTANT: os nem sorrendbe olvassa be a fájlokat, ezzel forse-olunk egy sorrendet
    folders = os.listdir(tissue_dir)
    recoveries = [folder for folder in folders if folder.endswith("recovery")]
    failures = [folder for folder in folders if folder.endswith("failure")]
    perfects = [folder for folder in folders if folder not in recoveries and folder not in failures]
    recoveries.sort()
    failures.sort()
    perfects.sort()
    perfects.extend(recoveries)
    perfects.extend(failures)
    folders = perfects
    print(f"{folders}")
    #exit # for testing
    
    ## process all demos (perfect and recovery)
    for subtask_name in folders:
        try:
            subtask_dir = os.path.join(tissue_dir, subtask_name)
            if not os.path.isdir(subtask_dir):
                continue

            # If folder is 1_suture_throw  ["1","suture","throw"]  prompt "suture throw"
            subtask_prompt = " ".join(subtask_name.split("_")[1:])
            is_recovery = subtask_prompt.endswith("recovery")
            is_failure = subtask_prompt.endswith("failure")
            
            if is_recovery:
                subtask_prompt = subtask_prompt[:-9]  # Remove " recovery" suffix
            if is_failure:
                subtask_prompt = subtask_prompt[:-8]  # Remove " failure" suffix
            
            #IMPORTANT: eposide mappák is sorrendbe legyenek feldolgozva
            episode_folders_sorted = os.listdir(subtask_dir)
            episode_folders_sorted.sort()

            for episode_name in episode_folders_sorted:
                episode_dir = os.path.join(subtask_dir, episode_name)
                if not os.path.isdir(episode_dir):
                    continue
                
                dataset = process_episode(
                    dataset, episode_dir, states_name, metas_name, actions_name, subtask_prompt
                )

                dataset.save_episode()
                if is_recovery:
                    recovery_demo_count += 1
                elif is_failure:
                    failure_demo_count += 1
                else:
                    perfect_demo_count += 1

        except Exception as e:
            print(f"Error processing episode {episode_dir}: {e}")
            dataset.clear_episode_buffer()

        print(
            f"subtask {subtask_name} processed successful, time taken: {time.time() - start_time}"
        )

    total_episode_count = perfect_demo_count + recovery_demo_count + failure_demo_count
    print(f"perfect_demo_count: {perfect_demo_count}")
    print(f"recovery_demo_count: {recovery_demo_count}")
    print(f"failure_demo_count: {failure_demo_count}")   
    print(f"Total episodes processed: {total_episode_count}")

    # -------------------------
    # Stratified random splits
    # -------------------------
    train_split = 0.8
    val_split = 0.1
    # test_split is implied by remainder

    def _indices_to_slice_ranges(indices: list[int]) -> str:
        """
        Convert sorted indices -> compact end-exclusive slice ranges.
        Example: [1,2,3,4, 7,8,9,10,11] -> "1:5, 7:12"
        """
        if not indices:
            return ""
        indices = sorted(indices)
        ranges = []
        start = prev = indices[0]
        for x in indices[1:]:
            if x == prev + 1:
                prev = x
                continue
            # close current range [start..prev] -> "start:prev+1"
            ranges.append(f"{start}:{prev+1}")
            start = prev = x
        ranges.append(f"{start}:{prev+1}")
        return ", ".join(ranges)

    def _stratified_split_indices(
        perfect_count: int,
        recovery_count: int,
        failure_count: int,
        train_split: float,
        val_split: float,
        seed: int = 42,
        ) -> tuple[list[int], list[int], list[int]]:
        """
        Episodes are assumed ordered as:
        perfect:  [0 .. perfect_count-1]
        recovery: [perfect_count .. perfect_count+recovery_count-1]
        failure:  [perfect_count+recovery_count .. total-1]

        For each group: shuffle, then take train%, then val% of remaining, rest test.
        """
        rng = np.random.default_rng(seed)

        def split_block(start: int, count: int) -> tuple[list[int], list[int], list[int]]:
            idx = np.arange(start, start + count)
            rng.shuffle(idx)

            n_train = int(np.floor(train_split * count))
            remaining = count - n_train
            n_val = int(np.round(val_split * count))
            # Ensure we don't exceed remaining (can happen for tiny counts)
            n_val = min(n_val, remaining)

            train_idx = idx[:n_train].tolist()
            val_idx = idx[n_train:n_train + n_val].tolist()
            test_idx = idx[n_train + n_val:].tolist()
            return train_idx, val_idx, test_idx

        p_train, p_val, p_test = split_block(0, perfect_count)
        r_train, r_val, r_test = split_block(perfect_count, recovery_count)
        f_train, f_val, f_test = split_block(perfect_count + recovery_count, failure_count)

        train = p_train + r_train + f_train
        val   = p_val   + r_val   + f_val
        test  = p_test  + r_test  + f_test

        # Sort for nicer range compression + readability
        train.sort()
        val.sort()
        test.sort()
        return train, val, test


    # Build the split lists
    train_ids, val_ids, test_ids = _stratified_split_indices(
        perfect_demo_count,
        recovery_demo_count,
        failure_demo_count,
        train_split=train_split,
        val_split=val_split,
        seed=0,  # change if you want a different random split, or make it CLI-configurable
    )

    # write split in meta
    # IMPORTANT: order of the episodes (perfect episodes first, then all recovery episodes) is still true,
    # we're just selecting *indices* from within those blocks.
    dataset.meta.info["splits"] = {
        "train": _indices_to_slice_ranges(train_ids),
        "val": _indices_to_slice_ranges(val_ids),
        "test": _indices_to_slice_ranges(test_ids),
        "perfect": f"0:{perfect_demo_count}",
        "recovery": f"{perfect_demo_count}:{perfect_demo_count + recovery_demo_count}",
        "failure": f"{perfect_demo_count + recovery_demo_count}:{total_episode_count}",
    }




    '''
    train_split = 0.8
    val_split = 0.1
    test_split = 1 - val_count - train_count

    train_count = int(train_split * total_episode_count)
    val_count = int(val_split * total_episode_count)
    test_count = int(test_split * total_episode_count)

    # write split in meta
    #IMPORTANT: order of the episodes (perfect episodes first, then all recovery episodes)
    dataset.meta.info["splits"] = {
        "train": "0:{}".format(train_count), # exapmle 0:80
        "val": "{}:{}".format(train_count, train_count + val_count), # example 80:80+10
        "test": "{}:{}".format(train_count + val_count, total_episode_count), # example 90:100
        "perfect": f"0:{perfect_demo_count}",  # perfect episodes
        "recovery": f"{perfect_demo_count}:{perfect_demo_count + recovery_demo_count}",  # recovery episodes
        "failure": f"{perfect_demo_count + recovery_demo_count}:{total_episode_count}",   # failure episodes
    }
    '''

    dataset.meta.info["episode_count"] = {
        "perfect": f"{perfect_demo_count}",  # perfect episodes
        "recovery": f"{recovery_demo_count}",  # recovery episodes
        "failure": f"{failure_demo_count}",   # failure episodes
        "total": f"{total_episode_count}", # total number of episodes
    }

    write_info(dataset.meta.info, dataset.root)

    print("Custom split configuration saved!")
    #IMPORTANT: the task is not always suturing
    print(f"Processed successful, time taken: {time.time() - start_time}")


def main(
    phantom_name: str,    
    repo_id: str, 
    data_path: Path = Path("/home/lorant/OpenH_LeRobot/dataset/"),
    *,
    push_to_hub: bool = True,
):
    """
    Main entry point for the conversion script.

    Args:
        data_path: The path to the dataset.
        repo_id: The desired Hugging Face Hub repository ID (e.g., 'username/dataset-name').
        push_to_hub: If True, uploads the dataset to the Hub after conversion.
    """
    if not data_path.exists():
        print(f"Error: The provided path does not exist: {data_path}")
        print("Please provide a valid path to your data.")
        return

    if repo_id == "your-username/your-dataset-name":
        print(
            "Warning: Using the default repo_id. Please specify your own with --repo-id."
        )

    convert_data_to_lerobot(data_path, repo_id, push_to_hub=push_to_hub, phantom_name=phantom_name)


if __name__ == "__main__":
    tyro.cli(main)
