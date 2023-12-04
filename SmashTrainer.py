# Referenced for this code: https://www.youtube.com/watch?v=XbWhJdQgi7E
# We watched the whole series of these videos, actually, and there are four of them.
# Also got some help from this: https://towardsdatascience.com/how-to-train-an-ai-to-play-any-game-f1489f3bc5c
#
# This is the part that should actually take our custom environment from SmashGym.py and use it to build
# out an agent, using the PPO algorithm. There are others available, but this is the one the guy in
# the video used so we decided to start with it and go from there.

import gymnasium
from gymnasium.envs.registration import register
from SmashGym import CustomGame
from stable_baselines3 import PPO
import os

# For saving.
model_directory = "models"

if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Initialize our environment.
env = CustomGame()
env.reset()

# Create our model.
model = PPO("MultiInputPolicy", env, verbose=1)
# Iterate many times through our space, learn the game. We never got around to adjusting these values because it wasn't
# until we got this booted up for the first time that we realized how infeasible the concept was.
for i in range(1,120):
    model.learn(total_timesteps=100000)
    model.save(f"{model_directory}/TestModel/{100000*i}")

episodes = 10

# Test out our agent and see how things went. We never actually got this far.
for ep in range(episodes):
    obs = env.reset
    done = False
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())

env.close()