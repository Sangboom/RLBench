from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import DetectedCondition


class ReachMovingTarget(Task):

    def init_task(self) -> None:
        self.target = Shape('target')
        self.boundaries = Shape('boundary')
        success_sensor = ProximitySensor('success')
        self.register_success_conditions(
            [DetectedCondition(self.robot.arm.get_tip(), success_sensor)])

    def init_episode(self, index: int) -> List[str]:
        color_name, color_rgb = colors[index]
        self.target.set_color((1, 0, 0))
        b = SpawnBoundary([self.boundaries])
        b.sample(self.target, min_distance=0.2, min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))

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

    def step(self) -> None:
        A = self.target.get_position()
        if A[1] < -1.0:
            v = 0.01
        elif A[1] > 1.0:
            v = -0.01

        self.target.set_position(A + v)
        
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
