import gym
import pybullet_envs
from Networks.networks_numpy import *


env_id = "CrippledAnt-v0"
envrn = gym.make(env_id)

import pickle
with open("save_ESnetWALKgated.pkl", "rb") as f:
    agent = pickle.load(f)

state = envrn.reset()

while True:
    envrn.render()
    state = state.reshape((1, state.size))
    action = agent.forward(state)[0]
    state, _, done, _ = envrn.step(action)
    if done:
        state = envrn.reset()






















