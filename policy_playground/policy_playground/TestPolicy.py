import math

import numpy as np
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.duration import Duration


class TestPolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"TestPolicy starting. Task: {task.id}")

        start_time = self.time_now()
        timeout = Duration(seconds=30.0)
        last_stamp = None

        while (self.time_now() - start_time) < timeout:
            obs = get_observation()
            if obs is None:
                self.sleep_for(0.01)
                continue

            stamp = obs.center_image.header.stamp
            if stamp == last_stamp:
                self.sleep_for(0.005)
                continue
            last_stamp = stamp

            tcp = obs.controller_state.tcp_pose
            wrench = obs.wrist_wrench.wrench
            joints = obs.joint_states.position[:6]

            self.get_logger().info(
                f"TCP: ({tcp.position.x:.3f}, {tcp.position.y:.3f}, {tcp.position.z:.3f}) | "
                f"F: ({wrench.force.x:.2f}, {wrench.force.y:.2f}, {wrench.force.z:.2f}) | "
                f"Joints: [{', '.join(f'{j:.2f}' for j in joints)}]"
            )

            t = stamp.sec + stamp.nanosec / 1e9
            y = 0.3 * math.sin(2.0 * math.pi * t / 5.0)

            self.set_pose_target(
                move_robot=move_robot,
                pose=Pose(
                    position=Point(x=-0.4, y=y, z=0.3),
                    orientation=Quaternion(x=1.0, y=0.0, z=0.0, w=0.0),
                ),
            )

        self.get_logger().info("TestPolicy done.")
        return True
