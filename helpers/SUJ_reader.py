#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class SUJReaderOnce(Node):
    def __init__(self):
        super().__init__("suj_reader_once")

        self.output_txt = "suj_measured_cp.txt"

        self.data = {
            "PSM1": None,
            "PSM2": None,
            "ECM": None,
        }

        self.create_subscription(
            PoseStamped,
            "/SUJ/PSM1/measured_cp",
            lambda msg: self.cb(msg, "PSM1"),
            10,
        )
        self.create_subscription(
            PoseStamped,
            "/SUJ/PSM2/measured_cp",
            lambda msg: self.cb(msg, "PSM2"),
            10,
        )
        self.create_subscription(
            PoseStamped,
            "/SUJ/ECM/measured_cp",
            lambda msg: self.cb(msg, "ECM"),
            10,
        )

        self.get_logger().info("Waiting for ONE message from PSM1, PSM2, ECM...")

    def cb(self, msg: PoseStamped, arm: str):
        if self.data[arm] is None:
            self.data[arm] = msg
            self.get_logger().info(f"{arm} received")

        # If all three arrived → write once and exit
        if all(v is not None for v in self.data.values()):
            self.write_and_exit()

    def write_and_exit(self):
        p1 = self.data["PSM1"].pose
        p2 = self.data["PSM2"].pose
        p3 = self.data["ECM"].pose

        # Use ROS timestamp from PSM1 (all are practically same time)
        t = self.data["PSM1"].header.stamp
        t_str = f"{t.sec}.{t.nanosec:09d}"

        line = (
            f"t={t_str} | "
            f"PSM1=({p1.position.x:.6f},{p1.position.y:.6f},{p1.position.z:.6f},"
            f"{p1.orientation.x:.6f},{p1.orientation.y:.6f},{p1.orientation.z:.6f},{p1.orientation.w:.6f}) | "
            f"PSM2=({p2.position.x:.6f},{p2.position.y:.6f},{p2.position.z:.6f},"
            f"{p2.orientation.x:.6f},{p2.orientation.y:.6f},{p2.orientation.z:.6f},{p2.orientation.w:.6f}) | "
            f"ECM=({p3.position.x:.6f},{p3.position.y:.6f},{p3.position.z:.6f},"
            f"{p3.orientation.x:.6f},{p3.orientation.y:.6f},{p3.orientation.z:.6f},{p3.orientation.w:.6f})"
        )

        with open(self.output_txt, "w") as f:
            f.write(line + "\n")

        self.get_logger().info("Written ONE line. Exiting.")
        rclpy.shutdown()


def main():
    rclpy.init()
    node = SUJReaderOnce()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
