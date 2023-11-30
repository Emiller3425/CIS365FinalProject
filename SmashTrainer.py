# Referenced for this code: https://www.youtube.com/watch?v=XbWhJdQgi7E
# It's the whole series of videos, actually, through to part 3.
#
# The part that should actually take our custom environment from SmashGym.py and use it to build
# out an agent, using the PPO algorithm. There are others available, but this is the one the guy in
# the video used so why the hell not?

import gymnasium
from gymnasium.envs.registration import register
from SmashGym import CustomGame
from stable_baselines3 import PPO
import os

model_directory = "models"
log_directory = "logs"

if not os.path.exists(model_directory):
    os.makedirs(model_directory)

if not os.path.exists(log_directory):
    os.makedirs(log_directory)


env = CustomGame()
env.reset()

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_directory)
for i in range(1,30):
    model.learn(total_timesteps=100000, reset_num_timesteps=False, tb_log_name="TestModel")
    model.save(f"{model_directory}/TestModel/{100000*i}")

episodes = 10

for ep in range(episodes):
    obs = env.reset
    done = False
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())

env.close()