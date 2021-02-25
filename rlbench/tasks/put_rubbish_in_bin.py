from typing import List
import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition
import math

class PutRubbishInBin(Task):

    def init_task(self):
        success_sensor = ProximitySensor('success')
        self.target = ProximitySensor('success')
        self.rubbish = Shape('rubbish')
        self.register_graspable_objects([self.rubbish])
        self.register_success_conditions(
            [DetectedCondition(self.rubbish, success_sensor)])

    def init_episode(self, index: int) -> List[str]:
        tomato1 = Shape('tomato1')
        tomato2 = Shape('tomato2')
        x1, y1, z1 = tomato2.get_position()
        x2, y2, z2 = self.rubbish.get_position()
        x3, y3, z3 = tomato1.get_position()
        pos = np.random.randint(3)
        if pos == 0:
            self.rubbish.set_position([x1, y1, z2])
            tomato2.set_position([x2, y2, z1])
        elif pos == 2:
            self.rubbish.set_position([x3, y3, z2])
            tomato1.set_position([x2, y2, z3])

        return ['put rubbish in bin',
                'drop the rubbish into the bin',
                'pick up the rubbish and leave it in the trash can',
                'throw away the trash, leaving any other objects alone',
                'chuck way any rubbish on the table rubbish']

    def variation_count(self) -> int:
        return 1

    def get_low_dim_state(self) -> np.ndarray:
        # One of the few tasks that have a custom low_dim_state function.
        rubbish_pos = np.array(self.rubbish.get_position())
        target_pos = np.array(self.target.get_position())
        tip_pos = np.array(self.robot.arm.get_tip().get_position())
        return np.concatenate((rubbish_pos, target_pos, tip_pos), axis=None)

    def reward(self) -> float:
        #if self.robot.gripper.check_collision(self.target):
        suc, _ = self.success()
        if suc:
            print('======================success!======================')
            r = 10
        else :
            b_pos = self.rubbish.get_position()
            g_pos = self.target.get_position()
            t_pos = self.robot.arm.get_tip().get_position()
            dis_sqr1 = (g_pos[0]-t_pos[0]) * (g_pos[0]-t_pos[0]) + (g_pos[1]-t_pos[1]) * (g_pos[1]-t_pos[1]) + (g_pos[2]-t_pos[2]) * (g_pos[2]-t_pos[2])
            dis_sqr2 = (b_pos[0]-t_pos[0]) * (b_pos[0]-t_pos[0]) + (b_pos[1]-t_pos[1]) * (b_pos[1]-t_pos[1]) + (b_pos[2]-t_pos[2]) * (b_pos[2]-t_pos[2])
            dis_sqr3 = (g_pos[0]-b_pos[0]) * (g_pos[0]-b_pos[0]) + (g_pos[1]-b_pos[1]) * (g_pos[1]-b_pos[1]) + (g_pos[2]-b_pos[2]) * (g_pos[2]-b_pos[2])
            dis = math.sqrt(dis_sqr1) + math.sqrt(dis_sqr2) + math.sqrt(dis_sqr3)
            r = 0 - dis
        return r