from typing import List
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition
import numpy as np
import math


class SlideBlockToTarget(Task):

    def init_task(self) -> None:
        self.block = Shape('block')
        self.target = ProximitySensor('success')
        self.register_success_conditions([
            DetectedCondition(Shape('block'), ProximitySensor('success'))])

    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index
        return ['slide the block to target',
                'slide the block onto the target',
                'push the block until it is sitting on top of the target',
                'slide the block towards the green target',
                'cover the target with the block by pushing the block in its'
                ' direction']

    def variation_count(self) -> int:
        return 1

    #custom part

    def get_low_dim_state(self) -> np.ndarray:
        # One of the few tasks that have a custom low_dim_state function.
        block_pos = np.array(self.block.get_position())
        target_pos = np.array(self.target.get_position())
        tip_pos = np.array(self.robot.arm.get_tip().get_position())
        return np.concatenate((block_pos, target_pos, tip_pos), axis=None)

    def reward(self) -> float:
        #if self.robot.gripper.check_collision(self.target):
        suc, _ = self.success()
        if suc:
            print('======================success!======================')
            r = 10
        else :
            b_pos = self.block.get_position()
            g_pos = self.target.get_position()
            t_pos = self.robot.arm.get_tip().get_position()
            dis_sqr1 = (g_pos[0]-t_pos[0]) * (g_pos[0]-t_pos[0]) + (g_pos[1]-t_pos[1]) * (g_pos[1]-t_pos[1]) + (g_pos[2]-t_pos[2]) * (g_pos[2]-t_pos[2])
            dis_sqr2 = (b_pos[0]-t_pos[0]) * (b_pos[0]-t_pos[0]) + (b_pos[1]-t_pos[1]) * (b_pos[1]-t_pos[1]) + (b_pos[2]-t_pos[2]) * (b_pos[2]-t_pos[2])
            dis_sqr3 = (g_pos[0]-b_pos[0]) * (g_pos[0]-b_pos[0]) + (g_pos[1]-b_pos[1]) * (g_pos[1]-b_pos[1]) + (g_pos[2]-b_pos[2]) * (g_pos[2]-b_pos[2])
            dis = math.sqrt(dis_sqr1) + math.sqrt(dis_sqr2) + math.sqrt(dis_sqr3)
            r = 0 - dis
        return r