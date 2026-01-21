#!/usr/bin/env python3
"""
daVinciRecorder.py

ROS2 multi-topic -> single CSV recorder @ fixed sample rate (default 30 SPS).

FINAL COLUMN POLICY:
- EXACTLY ONE timestamp column: record_time_ns (ROS2 clock, ns)
- NO seq column
- NO *_has_msg columns
- Missing data is encoded as NaN

Topics (latest-sample, zero-order-hold at 30 Hz):
1) /PSM1/local/measured_cp         (PoseStamped)
2) /PSM1/jaw/measured_js           (JointState, position[0])
3) /PSM2/local/measured_cp         (PoseStamped)
4) /PSM2/jaw/measured_js           (JointState, position[0])
5) /PSM1/local/setpoint_cp         (PoseStamped)
6) /PSM1/jaw/setpoint_js           (JointState, position[0])
7) /PSM2/local/setpoint_cp         (PoseStamped)
8) /PSM2/jaw/setpoint_js           (JointState, position[0])

Graceful shutdown WITHOUT Ctrl-C:
- Press ENTER or type 'q' + ENTER
"""

import argparse
import csv
import os
import threading
import sys
import time
from datetime import datetime
from typing import Any, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String


def ros_now_ns(node: Node) -> int:
    #return int(node.get_clock().now().nanoseconds)
    return time.time_ns()

def extract_pose_fields(msg: Any) -> List[float]:
    try:
        p = msg.pose.position
        q = msg.pose.orientation
        return [
            float(p.x), float(p.y), float(p.z),
            float(q.x), float(q.y), float(q.z), float(q.w),
        ]
    except Exception:
        return [float("nan")] * 7


def extract_joint_pos0(msg: Any) -> float:
    try:
        return float(msg.position[0])
    except Exception:
        return float("nan")
    
def extract_tool_type(msg: Any) -> str:
    if msg is None:
        return "UNKNOWN"
    try:
        return str(msg.data)
    except Exception:
        return ""

class MultiTopicCsvRecorder(Node):
    def __init__(self, sps: float, out_dir: str, session_name: str = ""):
        super().__init__("davinci_csv_recorder")

        tool_qos = QoSProfile(depth=1)
        tool_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        tool_qos.reliability = QoSReliabilityPolicy.RELIABLE

        self.sps = float(sps)
        self.period = 1.0 / self.sps
        self._running = True

        # Topics
        self.topics = {
            "psm1_measured_cp": ("/PSM1/local/measured_cp", PoseStamped),
            "psm1_measured_jaw": ("/PSM1/jaw/measured_js", JointState),
            "psm2_measured_cp": ("/PSM2/local/measured_cp", PoseStamped),
            "psm2_measured_jaw": ("/PSM2/jaw/measured_js", JointState),
            "ecm_measured_cp": ("/ECM/local/measured_cp", PoseStamped),

            "suj_psm1_measured_cp": ("/SUJ/PSM1/measured_cp", PoseStamped),
            "suj_psm2_measured_cp": ("/SUJ/PSM2/measured_cp", PoseStamped),
            "suj_ecm_measured_cp": ("/SUJ/ECM/measured_cp", PoseStamped),
            "psm1_tool_type": ("/PSM1/tool_type", String),
            "psm2_tool_type": ("/PSM2/tool_type", String),

            "psm1_setpoint_cp": ("/PSM1/local/setpoint_cp", PoseStamped),
            "psm1_setpoint_jaw": ("/PSM1/jaw/setpoint_js", JointState),
            "psm2_setpoint_cp": ("/PSM2/local/setpoint_cp", PoseStamped),
            "psm2_setpoint_jaw": ("/PSM2/jaw/setpoint_js", JointState),
            "ecm_setpoint_cp": ("/ECM/local/setpoint_cp", PoseStamped),

        }

        self.latest = {k: None for k in self.topics}

        for key, (topic, msg_type) in self.topics.items():
            qos = tool_qos if key in ("psm1_tool_type", "psm2_tool_type") else 10
            self.create_subscription(
                msg_type,
                topic,
                lambda msg, k=key: self._cb(k, msg),
                qos,
            )

        # Output CSV
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = session_name.strip() or ts
        self.csv_path = os.path.join(
            out_dir, f"dvrk_psm_meas_setpoint_{tag}_{int(self.sps)}sps.csv"
        )

        self.csv_file = open(self.csv_path, "w", newline="")
        self.writer = csv.writer(self.csv_file)

        # Header
        self.writer.writerow([
            "timestamp",

            # measured
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

            # meta
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
            "psm1_tool_type",
            "psm2_tool_type",

            # setpoint
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
        ])
        self.csv_file.flush()

        # Timer
        self.timer = self.create_timer(self.period, self._tick)

        # Keyboard exit
        # self._input_thread = threading.Thread(target=self._wait_for_exit, daemon=True)
        # self._input_thread.start()

        self.get_logger().info(f"Recording -> {self.csv_path}")
        # self.get_logger().info("Press ENTER or type 'q' + ENTER to stop recording")

    def _cb(self, key: str, msg: Any):
        self.latest[key] = msg

    def _tick(self):
        if not self._running:
            return

        t_ns = ros_now_ns(self)
        row: List[Any] = [t_ns]

        # measured
        row += extract_pose_fields(self.latest["psm1_measured_cp"])
        row.append(extract_joint_pos0(self.latest["psm1_measured_jaw"]))
        row += extract_pose_fields(self.latest["psm2_measured_cp"])
        row.append(extract_joint_pos0(self.latest["psm2_measured_jaw"]))
        row += extract_pose_fields(self.latest["ecm_measured_cp"])

        # meta
        row += extract_pose_fields(self.latest["suj_psm1_measured_cp"])
        row += extract_pose_fields(self.latest["suj_psm2_measured_cp"])
        row += extract_pose_fields(self.latest["suj_ecm_measured_cp"])
        row.append(extract_tool_type(self.latest["psm1_tool_type"]))
        row.append(extract_tool_type(self.latest["psm2_tool_type"]))

        # setpoint
        row += extract_pose_fields(self.latest["psm1_setpoint_cp"])
        row.append(extract_joint_pos0(self.latest["psm1_setpoint_jaw"]))
        row += extract_pose_fields(self.latest["psm2_setpoint_cp"])
        row.append(extract_joint_pos0(self.latest["psm2_setpoint_jaw"]))
        row += extract_pose_fields(self.latest["ecm_setpoint_cp"])

        self.writer.writerow(row)

        if int(t_ns / 1e9) % 1 == 0:
            self.csv_file.flush()

    def _wait_for_exit(self):
        try:
            user_input = sys.stdin.readline().strip().lower()
            if user_input == "" or user_input == "q":
                self.get_logger().info("Exit requested by user input")
                self.shutdown()
        except Exception:
            pass

    def shutdown(self):
        if not self._running:
            return
        self._running = False
        try:
            self.timer.cancel()
        except Exception:
            pass
        self.close()
        if rclpy.ok():
            rclpy.shutdown()

    def close(self):
        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sps", type=float, default=30.0)
    parser.add_argument("--session-name", default="", help="Session name (e.g., YYYYMMDD_HHMMSS) to embed in CSV filename for Synchronizer")
    parser.add_argument("--out-dir", default="./recordings_csv")
    args = parser.parse_args()

    rclpy.init()
    node = MultiTopicCsvRecorder(args.sps, args.out_dir, session_name=args.session_name)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.close()
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
