from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import DetectedCondition

import math


class ReachTarget(Task):

    def init_task(self) -> None:
        self.target = Shape('target')
        # self.distractor0 = Shape('distractor0')
        # self.distractor1 = Shape('distractor1')
        self.boundaries = Shape('boundary')
        success_sensor = ProximitySensor('success')
        self.register_success_conditions(
            [DetectedCondition(self.robot.arm.get_tip(), success_sensor)])

    def init_episode(self, index: int) -> List[str]:
        color_name, color_rgb = colors[index]
        self.target.set_color(color_rgb)
        # color_choices = np.random.choice(
        #     list(range(index)) + list(range(index + 1, len(colors))),
        #     size=2, replace=False)
        # for ob, i in zip([self.distractor0, self.distractor1], color_choices):
        #     name, rgb = colors[i]
        #     ob.set_color(rgb)
        b = SpawnBoundary([self.boundaries])
        # for ob in [self.target, self.distractor0, self.distractor1]:
        #     b.sample(ob, min_distance=0.2,
        #              min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))
        #   org
        b.sample(self.target, min_distance=0.2, min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))
        
        #randomize robot initial pose
        action = np.random.normal(-1, 1, size=(3,))
        joint_positions = self.robot.arm.solve_ik(
            action, quaternion=[0, 0, 0, 1], relative_to=None)
        self.robot.arm.set_joint_target_positions(joint_positions)

        return ['reach the %s target' % color_name,
                'touch the %s ball with the panda gripper' % color_name,
                'reach the %s sphere' %color_name]

    def variation_count(self) -> int:
        return len(colors)

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def get_low_dim_state(self) -> np.ndarray:
        # One of the few tasks that have a custom low_dim_state function.
        target_pos = np.array(self.target.get_position())
        tip_pos = np.array(self.robot.arm.get_tip().get_position())
        return np.concatenate((target_pos, tip_pos), axis=None)

    def is_static_workspace(self) -> bool:
        return True

    # custom part : override reward
    def reward(self) -> float:
        #if self.robot.gripper.check_collision(self.target):
        suc, _ = self.success()
        if suc:
            print('======================success!======================')
            r = 10
        else :
            g_pos = self.target.get_position()
            t_pos = self.robot.arm.get_tip().get_position()
            dis_sqr = (g_pos[0]-t_pos[0]) * (g_pos[0]-t_pos[0]) + (g_pos[1]-t_pos[1]) * (g_pos[1]-t_pos[1]) + (g_pos[2]-t_pos[2]) * (g_pos[2]-t_pos[2])
            dis = math.sqrt(dis_sqr)
            r = 0 - dis
        return r

