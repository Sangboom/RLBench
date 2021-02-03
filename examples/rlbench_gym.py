import gym
import rlbench.gym
import numpy as np

env = gym.make('reach_target-state-v0', render_mode='human')

training_steps = 120
episode_length = 40
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        obs = env.reset()
    a = env.action_space.sample()
    obs, reward, terminate, _ = env.step(a)
    env.render()  # Note: rendering increases step time.

print('Done')
env.close()