import gym
from Networks.networks_numpy import *

import pybullet_envs
import rex_gym
env_id = "RexWalk-v0"
envrn = gym.make(env_id)

import pickle
with open("save_ESnet.pkl", "rb") as f:
    agent = pickle.load(f)

agent = agent.network
state = envrn.reset()

while True:
    action = agent.forward(state)[0]
    state, _, done, _ = envrn.step(action)
    if done:
        state = envrn.reset()






















