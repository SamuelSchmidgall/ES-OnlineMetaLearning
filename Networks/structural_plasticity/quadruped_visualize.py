import gym
import pybullet_envs
from Networks.structural_plasticity.networks_numpy1 import *


env_id = "InvertedDoublePendulum-v2"
envrn = gym.make(env_id)

import pickle
with open("model_sp2.pkl", "rb") as f:
    agent = pickle.load(f)

agent.reset()
state = envrn.reset()

while True:
    envrn.render()
    state = state.reshape((1, state.size))
    action = agent.forward(state)[0]
    state, _, done, _ = envrn.step(action)
    if done:
        agent.reset()
        state = envrn.reset()






















