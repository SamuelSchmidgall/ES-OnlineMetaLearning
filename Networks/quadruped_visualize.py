import gym
import pybullet_envs
from Networks.networks_numpy2 import *


env_id = "CrippledAntBulletEnv-v0"
envrn = gym.make(env_id, render=True)

import pickle
with open("save_ESnetWALK.pkl", "rb") as f:
    agent = pickle.load(f)

state = envrn.reset()

while True:
    envrn.render()
    action = agent.forward(state)[0]
    state, _, done, _ = envrn.step(action)
    if done:
        state = envrn.reset()






















